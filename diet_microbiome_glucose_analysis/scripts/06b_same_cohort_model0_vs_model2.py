#!/usr/bin/env python3
"""
Same-cohort QC for Microbiome -> CGM:

Purpose
-------
Separate two effects that are mixed when comparing the original Model 0
with Model 2:

    1) sample-selection effect
       Original Model 0 cohort -> Model-2 complete-case cohort

    2) covariate-adjustment effect
       Same-cohort Model 0 -> Model 2

For each CGM phenotype, this script reconstructs exactly the Model-2
complete-case cohort, then reruns unadjusted microbiome-wide regressions:

    CGM phenotype Z ~ species CLR-Z

on that same participant set.

Required prior outputs
----------------------
05_microbiome_cgm_all_model0.csv
06_microbiome_cgm_all_model2.csv

Run
---
python 06b_same_cohort_model0_vs_model2.py --self-test
python 06b_same_cohort_model0_vs_model2.py
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist, spearmanr


# ============================================================
# PATHS
# ============================================================

ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
ANALYSIS = ROOT / "diet_microbiome_glucose_analysis"

MODEL_DIR = ANALYSIS / "outputs" / "models"
REPORT_DIR = ANALYSIS / "outputs" / "reports"

MICRO_CANDIDATES = [
    ROOT / "gut_microbiome_deal" / "data" / "08_species_clr_zscore.csv",
]

CGM_CANDIDATES = [
    ROOT / "cgm_deal" / "outputs" / "data" / "05_cgm_core_phenotypes.csv",
    ROOT / "cgm_deal" / "data" / "05_cgm_core_phenotypes.csv",
]

COVAR_CANDIDATES = [
    ROOT / "co-variant" / "outputs" / "data" / "02_covariate_master.csv",
]

ORIGINAL_MODEL0_FILE = MODEL_DIR / "05_microbiome_cgm_all_model0.csv"
MODEL2_FILE = MODEL_DIR / "06_microbiome_cgm_all_model2.csv"


# ============================================================
# DEFINITIONS
# ============================================================

OUTCOMES = {
    "mean_glucose": "cgm_mean_z",
    "glucose_cv": "cgm_cv_z",
    "time_above_140": "cgm_above_140_z",
}

MODEL2_COVARS = [
    "age_years",
    "sleep_duration_hours_day",
    "physical_activity_met_h_week",
    "vitamin_use",
    "hormone_use",
    "sex",
    "education_level",
    "smoking_status",
    "cgm_device_type",
]

META_COLS = {
    "participant_id",
    "cohort",
    "research_stage",
    "array_index",
    "shannon_index",
    "simpson_index",
}

EXPECTED_SPECIES = 379
EXCLUSION_COL = "exclude_diabetes_or_a10"


# ============================================================
# HELPERS
# ============================================================

def normalize_id(series):
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def resolve_one(candidates, label):
    existing = [p for p in candidates if p.exists()]

    if len(existing) == 1:
        return existing[0]

    if len(existing) > 1:
        print(
            f"NOTE: multiple {label} candidates exist; "
            f"using first priority:\n  {existing[0]}"
        )
        return existing[0]

    attempted = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        f"Could not find {label}. Tried:\n{attempted}"
    )


def require_columns(frame, columns, label):
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise RuntimeError(
            f"{label} missing required columns: {missing}"
        )


def bh_fdr(pvalues):
    p = np.asarray(pvalues, dtype=float)

    if p.ndim != 1:
        raise ValueError("BH-FDR expects a one-dimensional vector.")

    if len(p) == 0:
        return np.array([], dtype=float)

    if not np.isfinite(p).all():
        raise RuntimeError("BH-FDR received NaN/Inf p-values.")

    if ((p < 0) | (p > 1)).any():
        raise RuntimeError("BH-FDR received p-values outside [0,1].")

    n = len(p)
    order = np.argsort(p)
    ranked = p[order]

    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    out = np.empty_like(q)
    out[order] = q
    return out


def infer_species_columns(micro):
    taxonomic = [
        c for c in micro.columns
        if c not in META_COLS
        and ("s__" in c or "|s_" in c or "|s__" in c)
    ]

    if len(taxonomic) == EXPECTED_SPECIES:
        return taxonomic

    fallback = [
        c for c in micro.columns
        if c not in META_COLS
    ]

    if len(fallback) != EXPECTED_SPECIES:
        raise RuntimeError(
            "Could not identify exactly 379 species columns.\n"
            f"Taxonomic candidates: {len(taxonomic)}\n"
            f"Non-metadata candidates: {len(fallback)}"
        )

    return fallback


def fit_all_species_univariate(X, y):
    """
    Fit:
        y = intercept_j + beta_j * X_j + error
    for all species j simultaneously.
    """

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D.")

    if X.shape[0] != len(y):
        raise ValueError("X and y participant counts differ.")

    if not np.isfinite(X).all():
        raise RuntimeError("Species matrix contains NaN/Inf.")

    if not np.isfinite(y).all():
        raise RuntimeError("Outcome contains NaN/Inf.")

    n = len(y)
    if n < 5:
        raise RuntimeError("Too few participants.")

    X_mean = X.mean(axis=0)
    y_mean = y.mean()

    Xc = X - X_mean
    yc = y - y_mean

    sxx = np.sum(Xc ** 2, axis=0)

    if (sxx <= 0).any():
        bad = np.where(sxx <= 0)[0][:20].tolist()
        raise RuntimeError(
            f"Species with zero variance at indexes: {bad}"
        )

    beta = np.sum(Xc * yc[:, None], axis=0) / sxx

    intercept = y_mean - beta * X_mean
    fitted = intercept[None, :] + X * beta[None, :]
    resid = y[:, None] - fitted

    df = n - 2
    sse = np.sum(resid ** 2, axis=0)
    mse = sse / df
    se = np.sqrt(mse / sxx)

    t_stat = beta / se
    p_value = 2 * t_dist.sf(np.abs(t_stat), df=df)

    tcrit = float(t_dist.ppf(0.975, df=df))
    ci_lower = beta - tcrit * se
    ci_upper = beta + tcrit * se

    return {
        "beta": beta,
        "SE": se,
        "t": t_stat,
        "p_value": p_value,
        "CI95_lower": ci_lower,
        "CI95_upper": ci_upper,
    }


def comparison_metrics(left, right, left_label, right_label):
    """
    left/right must contain:
      species, beta, FDR
    """

    merged = left.merge(
        right,
        on="species",
        how="inner",
        validate="one_to_one",
        suffixes=(f"_{left_label}", f"_{right_label}"),
    )

    left_sig = merged[f"FDR_{left_label}"] < 0.05
    right_sig = merged[f"FDR_{right_label}"] < 0.05

    retained = int((left_sig & right_sig).sum())
    lost = int((left_sig & ~right_sig).sum())
    gained = int((~left_sig & right_sig).sum())

    both = merged.loc[left_sig & right_sig]

    same_direction_retained = int(
        (
            np.sign(both[f"beta_{left_label}"])
            == np.sign(both[f"beta_{right_label}"])
        ).sum()
    )

    rho, rho_p = spearmanr(
        merged[f"beta_{left_label}"],
        merged[f"beta_{right_label}"],
    )

    merged[f"delta_beta_{right_label}_minus_{left_label}"] = (
        merged[f"beta_{right_label}"]
        - merged[f"beta_{left_label}"]
    )

    return merged, {
        f"{left_label}_significant_FDR05": int(left_sig.sum()),
        f"{right_label}_significant_FDR05": int(right_sig.sum()),
        "retained_significant": retained,
        "lost": lost,
        "gained": gained,
        "same_direction_among_retained": same_direction_retained,
        "beta_spearman_all_species": float(rho),
        "beta_spearman_p": float(rho_p),
    }


def self_test():
    print("=" * 100)
    print("SELF TEST — SAME-COHORT MODEL 0 QC")
    print("=" * 100)

    rng = np.random.default_rng(20260904)

    n = 1000
    p = 40

    X = rng.normal(size=(n, p))
    y = 0.5 * X[:, 0] - 0.4 * X[:, 1] + rng.normal(size=n)

    fit = fit_all_species_univariate(X, y)

    assert fit["beta"][0] > 0.4
    assert fit["beta"][1] < -0.3

    q = bh_fdr(fit["p_value"])
    assert q[0] < 0.05
    assert q[1] < 0.05

    print(f"species_0 beta={fit['beta'][0]:.4f}, FDR={q[0]:.3e}")
    print(f"species_1 beta={fit['beta'][1]:.4f}, FDR={q[1]:.3e}")
    print("Regression: PASS")
    print("BH-FDR: PASS")
    print("SELF TEST: PASS")


# ============================================================
# MAIN
# ============================================================

def run_analysis():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not ORIGINAL_MODEL0_FILE.exists():
        raise FileNotFoundError(
            f"Missing original Model 0 output:\n{ORIGINAL_MODEL0_FILE}"
        )

    if not MODEL2_FILE.exists():
        raise FileNotFoundError(
            f"Missing Model 2 output:\n{MODEL2_FILE}"
        )

    micro_file = resolve_one(
        MICRO_CANDIDATES,
        "microbiome CLR-Z file",
    )

    cgm_file = resolve_one(
        CGM_CANDIDATES,
        "CGM phenotype file",
    )

    covar_file = resolve_one(
        COVAR_CANDIDATES,
        "covariate master file",
    )

    print("=" * 100)
    print("MICROBIOME -> CGM — SAME-COHORT MODEL 0 vs MODEL 2 QC")
    print("=" * 100)

    micro = pd.read_csv(micro_file, low_memory=False)
    cgm = pd.read_csv(cgm_file, low_memory=False)
    covar = pd.read_csv(covar_file, low_memory=False)

    original_model0 = pd.read_csv(
        ORIGINAL_MODEL0_FILE,
        low_memory=False,
    )

    model2 = pd.read_csv(
        MODEL2_FILE,
        low_memory=False,
    )

    for label, frame in [
        ("microbiome", micro),
        ("CGM", cgm),
        ("covariate master", covar),
    ]:
        require_columns(frame, ["participant_id"], label)

        frame["participant_id"] = normalize_id(
            frame["participant_id"]
        )

        if frame["participant_id"].duplicated().any():
            raise RuntimeError(
                f"{label} contains duplicate participant_id."
            )

    require_columns(
        cgm,
        ["participant_id", *OUTCOMES.values()],
        "CGM",
    )

    require_columns(
        covar,
        ["participant_id", EXCLUSION_COL, *MODEL2_COVARS],
        "covariate master",
    )

    require_columns(
        original_model0,
        [
            "outcome_name",
            "species",
            "beta_species",
            "FDR_within_outcome379",
            "N",
        ],
        "original Model 0",
    )

    require_columns(
        model2,
        [
            "outcome_name",
            "species",
            "beta_species",
            "FDR_within_outcome379",
            "N",
        ],
        "Model 2",
    )

    species = infer_species_columns(micro)

    cohort = (
        micro[["participant_id", *species]]
        .merge(
            cgm[["participant_id", *OUTCOMES.values()]],
            on="participant_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            covar[
                [
                    "participant_id",
                    EXCLUSION_COL,
                    *MODEL2_COVARS,
                ]
            ],
            on="participant_id",
            how="left",
            validate="one_to_one",
        )
    )

    exclusion_numeric = pd.to_numeric(
        cohort[EXCLUSION_COL],
        errors="coerce",
    )

    primary = cohort.loc[
        ~exclusion_numeric.eq(1)
    ].copy()

    print("Primary eligible before outcome/model2 filtering:", len(primary))
    print("Species:", len(species))

    all_same_cohort = []
    summary_rows = []
    all_comparisons = []

    for outcome_name, outcome_col in OUTCOMES.items():
        print()
        print("=" * 100)
        print(outcome_name)
        print("=" * 100)

        complete = primary[
            [outcome_col, *MODEL2_COVARS]
        ].notna().all(axis=1)

        selected = primary.loc[complete].copy()

        X = (
            selected[species]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float64)
        )

        y = pd.to_numeric(
            selected[outcome_col],
            errors="coerce",
        ).to_numpy(dtype=np.float64)

        if not np.isfinite(X).all():
            raise RuntimeError(
                f"{outcome_name}: species matrix contains NaN/Inf."
            )

        if not np.isfinite(y).all():
            raise RuntimeError(
                f"{outcome_name}: outcome contains NaN/Inf."
            )

        fit = fit_all_species_univariate(X, y)
        fdr = bh_fdr(fit["p_value"])

        same = pd.DataFrame({
            "outcome_name": outcome_name,
            "outcome_col": outcome_col,
            "species": species,
            "N": len(selected),
            "beta_species": fit["beta"],
            "SE": fit["SE"],
            "t": fit["t"],
            "p_value": fit["p_value"],
            "CI95_lower": fit["CI95_lower"],
            "CI95_upper": fit["CI95_upper"],
            "FDR_within_outcome379": fdr,
        })

        same["significant_FDR05"] = (
            same["FDR_within_outcome379"] < 0.05
        )

        same.to_csv(
            MODEL_DIR
            / f"06b_microbiome_cgm_{outcome_name}_same_cohort_model0.csv",
            index=False,
        )

        all_same_cohort.append(same)

        # Check cohort N against Model 2 output.
        m2_piece = model2.loc[
            model2["outcome_name"].eq(outcome_name)
        ].copy()

        m2_N_values = sorted(
            pd.to_numeric(
                m2_piece["N"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        if len(m2_N_values) != 1:
            raise RuntimeError(
                f"{outcome_name}: Model2 file has unexpected N values "
                f"{m2_N_values}"
            )

        model2_N = m2_N_values[0]

        n_match = (len(selected) == model2_N)

        print("Same-cohort Model0 N:", len(selected))
        print("Recorded Model2 N:", model2_N)
        print("N matches Model2:", n_match)

        if not n_match:
            raise RuntimeError(
                f"{outcome_name}: reconstructed same-cohort N="
                f"{len(selected)} but Model2 N={model2_N}. "
                "Stop: cohort reconstruction is not identical."
            )

        print(
            "Same-cohort Model0 FDR<0.05:",
            int(same["significant_FDR05"].sum()),
        )

        # -------- original M0 -> same-cohort M0
        orig_piece = original_model0.loc[
            original_model0["outcome_name"].eq(outcome_name)
        ][
            [
                "species",
                "beta_species",
                "FDR_within_outcome379",
                "N",
            ]
        ].copy()

        orig_N_values = sorted(
            pd.to_numeric(
                orig_piece["N"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        orig_cmp = orig_piece[
            [
                "species",
                "beta_species",
                "FDR_within_outcome379",
            ]
        ].rename(
            columns={
                "beta_species": "beta",
                "FDR_within_outcome379": "FDR",
            }
        )

        same_cmp = same[
            [
                "species",
                "beta_species",
                "FDR_within_outcome379",
            ]
        ].rename(
            columns={
                "beta_species": "beta",
                "FDR_within_outcome379": "FDR",
            }
        )

        cmp_sample, metrics_sample = comparison_metrics(
            orig_cmp,
            same_cmp,
            "original_model0",
            "same_cohort_model0",
        )

        cmp_sample["outcome_name"] = outcome_name
        cmp_sample["comparison_type"] = "sample_selection"

        # -------- same-cohort M0 -> Model2
        m2_cmp = m2_piece[
            [
                "species",
                "beta_species",
                "FDR_within_outcome379",
            ]
        ].rename(
            columns={
                "beta_species": "beta",
                "FDR_within_outcome379": "FDR",
            }
        )

        cmp_adjust, metrics_adjust = comparison_metrics(
            same_cmp,
            m2_cmp,
            "same_cohort_model0",
            "model2",
        )

        cmp_adjust["outcome_name"] = outcome_name
        cmp_adjust["comparison_type"] = "covariate_adjustment"

        all_comparisons.extend(
            [cmp_sample, cmp_adjust]
        )

        summary_rows.append({
            "outcome_name": outcome_name,
            "original_model0_N": (
                orig_N_values[0] if len(orig_N_values) == 1 else np.nan
            ),
            "same_cohort_model0_N": len(selected),
            "model2_N": model2_N,

            "original_model0_significant_FDR05":
                metrics_sample["original_model0_significant_FDR05"],
            "same_cohort_model0_significant_FDR05":
                metrics_sample["same_cohort_model0_significant_FDR05"],
            "model2_significant_FDR05":
                metrics_adjust["model2_significant_FDR05"],

            "sample_selection_retained":
                metrics_sample["retained_significant"],
            "sample_selection_lost":
                metrics_sample["lost"],
            "sample_selection_gained":
                metrics_sample["gained"],
            "sample_selection_beta_spearman":
                metrics_sample["beta_spearman_all_species"],

            "adjustment_retained":
                metrics_adjust["retained_significant"],
            "adjustment_lost":
                metrics_adjust["lost"],
            "adjustment_gained":
                metrics_adjust["gained"],
            "adjustment_same_direction_among_retained":
                metrics_adjust["same_direction_among_retained"],
            "adjustment_beta_spearman":
                metrics_adjust["beta_spearman_all_species"],
        })

        print()
        print("SAMPLE-SELECTION EFFECT")
        print(
            f"Original Model0 significant: "
            f"{metrics_sample['original_model0_significant_FDR05']}"
        )
        print(
            f"Same-cohort Model0 significant: "
            f"{metrics_sample['same_cohort_model0_significant_FDR05']}"
        )
        print(
            f"retained={metrics_sample['retained_significant']} "
            f"lost={metrics_sample['lost']} "
            f"gained={metrics_sample['gained']} "
            f"beta_rho={metrics_sample['beta_spearman_all_species']:.4f}"
        )

        print()
        print("COVARIATE-ADJUSTMENT EFFECT")
        print(
            f"Same-cohort Model0 significant: "
            f"{metrics_adjust['same_cohort_model0_significant_FDR05']}"
        )
        print(
            f"Model2 significant: "
            f"{metrics_adjust['model2_significant_FDR05']}"
        )
        print(
            f"retained={metrics_adjust['retained_significant']} "
            f"lost={metrics_adjust['lost']} "
            f"gained={metrics_adjust['gained']} "
            f"same_direction_retained="
            f"{metrics_adjust['same_direction_among_retained']} "
            f"beta_rho={metrics_adjust['beta_spearman_all_species']:.4f}"
        )

    same_all = pd.concat(
        all_same_cohort,
        ignore_index=True,
    )

    same_all.to_csv(
        MODEL_DIR / "06b_microbiome_cgm_all_same_cohort_model0.csv",
        index=False,
    )

    comparison_all = pd.concat(
        all_comparisons,
        ignore_index=True,
        sort=False,
    )

    comparison_all.to_csv(
        REPORT_DIR / "06b_same_cohort_model0_vs_model2_all_comparisons.csv",
        index=False,
    )

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        REPORT_DIR / "06b_same_cohort_model0_vs_model2_summary.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print(summary.to_string(index=False))

    print()
    print("=" * 100)
    print("HOW TO READ THIS")
    print("=" * 100)
    print(
        "Original Model0 -> Same-cohort Model0 = sample-selection effect."
    )
    print(
        "Same-cohort Model0 -> Model2 = covariate-adjustment effect."
    )
    print(
        "Only the second comparison isolates adjustment, because N is identical."
    )

    print()
    print("=" * 100)
    print("SAVED")
    print("=" * 100)
    print(
        MODEL_DIR
        / "06b_microbiome_cgm_all_same_cohort_model0.csv"
    )
    print(
        REPORT_DIR
        / "06b_same_cohort_model0_vs_model2_summary.csv"
    )
    print(
        REPORT_DIR
        / "06b_same_cohort_model0_vs_model2_all_comparisons.csv"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run synthetic unadjusted-regression/FDR test.",
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    run_analysis()


if __name__ == "__main__":
    main()
