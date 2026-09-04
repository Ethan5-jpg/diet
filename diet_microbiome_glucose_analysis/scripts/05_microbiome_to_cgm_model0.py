#!/usr/bin/env python3
"""
Microbiome -> CGM, Model 0 (unadjusted)

Primary analysis:
    CGM phenotype Z ~ species CLR-Z

Three primary CGM phenotypes:
    cgm_mean_z
    cgm_cv_z
    cgm_above_140_z

Microbiome:
    all 379 post-QC species from 08_species_clr_zscore.csv

Primary cohort:
    microbiome ∩ CGM
    exclude only participants with explicit exclude_diabetes_or_a10 == 1
    retain unknown exclusion status for the primary analysis

Multiple testing:
    1) BH-FDR within each CGM phenotype (379 tests)  <-- primary species-wide FDR
    2) BH-FDR across all 3 x 379 = 1137 tests       <-- stricter global sensitivity

Run:
    python 05_microbiome_to_cgm_model0.py --self-test
    python 05_microbiome_to_cgm_model0.py
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist


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


# ============================================================
# DEFINITIONS
# ============================================================

OUTCOMES = {
    "mean_glucose": "cgm_mean_z",
    "glucose_cv": "cgm_cv_z",
    "time_above_140": "cgm_above_140_z",
}

META_COLS = {
    "participant_id",
    "cohort",
    "research_stage",
    "array_index",
    "shannon_index",
    "simpson_index",
}

EXPECTED_SPECIES = 379


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


def bh_fdr(pvalues):
    p = np.asarray(pvalues, dtype=float)

    if p.ndim != 1:
        raise ValueError("BH-FDR expects a one-dimensional p-value vector.")

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
    """
    Prefer taxonomic-name columns. If that does not yield exactly 379,
    fall back to all non-metadata columns, but require exactly 379.
    """

    taxonomic = [
        c
        for c in micro.columns
        if c not in META_COLS
        and (
            "s__" in c
            or "|s_" in c
            or "|s__" in c
        )
    ]

    if len(taxonomic) == EXPECTED_SPECIES:
        return taxonomic

    fallback = [
        c
        for c in micro.columns
        if c not in META_COLS
    ]

    if len(fallback) != EXPECTED_SPECIES:
        preview = "\n".join(
            f"  {c}" for c in micro.columns[:30]
        )
        raise RuntimeError(
            "Could not identify exactly 379 species columns.\n"
            f"Taxonomic-name candidates: {len(taxonomic)}\n"
            f"Non-metadata candidates: {len(fallback)}\n"
            "First columns in microbiome file:\n"
            f"{preview}"
        )

    return fallback


def fit_all_species_univariate(X, y):
    """
    Fit P independent simple OLS models simultaneously:

        y = intercept_j + beta_j * X_j + error

    X: shape (N, P)
       species CLR-Z values

    y: shape (N,)
       one CGM phenotype Z

    Returns:
        beta, se, t, p, r2
        each shape (P,)
    """

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError("X must be a 2D matrix.")

    if y.ndim != 1:
        raise ValueError("y must be a 1D vector.")

    if X.shape[0] != len(y):
        raise ValueError("X and y have different participant counts.")

    if len(y) < 5:
        raise RuntimeError("Too few participants for regression.")

    if not np.isfinite(X).all():
        raise RuntimeError("Species matrix contains NaN/Inf.")

    if not np.isfinite(y).all():
        raise RuntimeError("CGM outcome contains NaN/Inf.")

    X_mean = X.mean(axis=0)
    y_mean = y.mean()

    Xc = X - X_mean
    yc = y - y_mean

    sxx = np.sum(Xc ** 2, axis=0)

    if (sxx <= 0).any():
        bad = np.where(sxx <= 0)[0][:20].tolist()
        raise RuntimeError(
            "At least one species has zero variance in this cohort. "
            f"Column indexes: {bad}"
        )

    beta = np.sum(Xc * yc[:, None], axis=0) / sxx

    intercept = y_mean - beta * X_mean

    fitted = (
        intercept[None, :]
        + X * beta[None, :]
    )

    resid = y[:, None] - fitted

    n = len(y)
    df = n - 2

    sse = np.sum(resid ** 2, axis=0)
    mse = sse / df

    se = np.sqrt(mse / sxx)

    t_stat = beta / se

    p = 2 * t_dist.sf(
        np.abs(t_stat),
        df=df,
    )

    sst = np.sum(yc ** 2)

    if sst <= 0:
        raise RuntimeError("CGM outcome has zero variance.")

    r2 = 1 - sse / sst

    return beta, se, t_stat, p, r2


def numeric_matrix(frame, columns, label):
    out = (
        frame[columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64)
    )

    if not np.isfinite(out).all():
        bad_count = int((~np.isfinite(out)).sum())
        raise RuntimeError(
            f"{label} contains {bad_count} NaN/Inf values."
        )

    return out


def self_test():
    print("=" * 95)
    print("SELF TEST — MICROBIOME -> CGM MODEL 0")
    print("=" * 95)

    rng = np.random.default_rng(20260904)

    n = 800
    p = 30

    X = rng.normal(size=(n, p))
    noise = rng.normal(scale=0.8, size=n)

    # Known signal:
    # species 0 positive, species 1 negative.
    y = (
        0.65 * X[:, 0]
        - 0.45 * X[:, 1]
        + noise
    )

    beta, se, t_stat, pvals, r2 = fit_all_species_univariate(X, y)
    q = bh_fdr(pvals)

    assert beta[0] > 0.5, beta[0]
    assert beta[1] < -0.3, beta[1]
    assert q[0] < 0.05, q[0]
    assert q[1] < 0.05, q[1]
    assert np.isfinite(beta).all()
    assert np.isfinite(se).all()
    assert np.isfinite(t_stat).all()
    assert np.isfinite(pvals).all()
    assert np.isfinite(r2).all()

    # BH monotonicity sanity via known ordered p-values.
    test_p = np.array([0.001, 0.01, 0.02, 0.5])
    test_q = bh_fdr(test_p)

    assert np.all((test_q >= 0) & (test_q <= 1))
    assert test_q[0] <= test_q[1] <= test_q[2] <= test_q[3]

    print("Synthetic regression test: PASS")
    print(f"species_0 beta={beta[0]:.4f}, FDR={q[0]:.3e}")
    print(f"species_1 beta={beta[1]:.4f}, FDR={q[1]:.3e}")
    print("BH-FDR sanity test: PASS")
    print("SELF TEST: PASS")


# ============================================================
# MAIN ANALYSIS
# ============================================================

def run_analysis():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

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

    print("=" * 95)
    print("MICROBIOME -> CGM — MODEL 0")
    print("=" * 95)

    print("Microbiome:", micro_file)
    print("CGM:", cgm_file)
    print("Covariates/exclusion:", covar_file)

    micro = pd.read_csv(
        micro_file,
        low_memory=False,
    )

    cgm = pd.read_csv(
        cgm_file,
        low_memory=False,
    )

    covar = pd.read_csv(
        covar_file,
        low_memory=False,
    )

    for name, frame in [
        ("microbiome", micro),
        ("CGM", cgm),
        ("covariate master", covar),
    ]:
        if "participant_id" not in frame.columns:
            raise RuntimeError(
                f"{name} file has no participant_id column."
            )

        frame["participant_id"] = normalize_id(
            frame["participant_id"]
        )

        if frame["participant_id"].duplicated().any():
            duplicated = (
                frame.loc[
                    frame["participant_id"].duplicated(keep=False),
                    "participant_id",
                ]
                .value_counts()
                .head(20)
                .to_dict()
            )
            raise RuntimeError(
                f"Duplicate participant_id in {name}: {duplicated}"
            )

    for outcome_col in OUTCOMES.values():
        if outcome_col not in cgm.columns:
            raise RuntimeError(
                f"CGM file is missing required outcome: {outcome_col}"
            )

    exclusion_col = "exclude_diabetes_or_a10"

    if exclusion_col not in covar.columns:
        raise RuntimeError(
            f"Covariate master is missing {exclusion_col}."
        )

    species = infer_species_columns(micro)

    print()
    print("Microbiome participants:", len(micro))
    print("CGM participants:", len(cgm))
    print("Species:", len(species))

    # Keep only what this analysis needs from CGM/covariates.
    cgm_keep = [
        "participant_id",
        *OUTCOMES.values(),
    ]

    covar_keep = [
        "participant_id",
        exclusion_col,
    ]

    cohort = (
        micro[
            ["participant_id", *species]
        ]
        .merge(
            cgm[cgm_keep],
            on="participant_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            covar[covar_keep],
            on="participant_id",
            how="left",
            validate="one_to_one",
        )
    )

    print("Microbiome ∩ CGM:", len(cohort))

    exclusion_numeric = pd.to_numeric(
        cohort[exclusion_col],
        errors="coerce",
    )

    explicit_excluded = exclusion_numeric.eq(1)
    exclusion_unknown = exclusion_numeric.isna()

    primary = cohort.loc[
        ~explicit_excluded
    ].copy()

    print(
        "Explicit known diabetes/A10 excluded:",
        int(explicit_excluded.sum()),
    )

    print(
        "Exclusion status unknown but retained:",
        int(
            exclusion_unknown.loc[
                ~explicit_excluded
            ].sum()
        ),
    )

    print(
        "Primary eligible before outcome missingness:",
        len(primary),
    )

    # Species should be complete after half-minimum + CLR + Z.
    X_all = numeric_matrix(
        primary,
        species,
        "Species CLR-Z matrix",
    )

    all_results = []
    summary_rows = []

    for outcome_name, outcome_col in OUTCOMES.items():
        print()
        print("=" * 95)
        print(outcome_name)
        print("=" * 95)

        y_all = pd.to_numeric(
            primary[outcome_col],
            errors="coerce",
        ).to_numpy(dtype=float)

        valid = np.isfinite(y_all)

        X = X_all[valid, :]
        y = y_all[valid]

        print("N:", len(y))
        print("Species tested:", len(species))

        beta, se, t_stat, p, r2 = fit_all_species_univariate(
            X,
            y,
        )

        fdr_within = bh_fdr(p)

        result = pd.DataFrame(
            {
                "outcome_name": outcome_name,
                "outcome_col": outcome_col,
                "species": species,
                "N": len(y),
                "beta_species": beta,
                "SE": se,
                "t": t_stat,
                "p_value": p,
                "FDR_within_outcome379": fdr_within,
                "R2": r2,
            }
        )

        result["CI95_lower"] = (
            result["beta_species"]
            - 1.96 * result["SE"]
        )

        result["CI95_upper"] = (
            result["beta_species"]
            + 1.96 * result["SE"]
        )

        result["significant_within_FDR05"] = (
            result["FDR_within_outcome379"] < 0.05
        )

        result["direction"] = np.where(
            result["beta_species"] > 0,
            "positive",
            np.where(
                result["beta_species"] < 0,
                "negative",
                "zero",
            ),
        )

        sig = result.loc[
            result["significant_within_FDR05"]
        ].copy()

        n_sig = len(sig)
        n_pos = int((sig["beta_species"] > 0).sum())
        n_neg = int((sig["beta_species"] < 0).sum())

        print("FDR<0.05 within 379:", n_sig)
        print("  positive:", n_pos)
        print("  negative:", n_neg)

        print()
        print("TOP POSITIVE")
        print("-" * 95)
        print(
            result.sort_values(
                "beta_species",
                ascending=False,
            )
            .head(10)[
                [
                    "species",
                    "beta_species",
                    "p_value",
                    "FDR_within_outcome379",
                ]
            ]
            .to_string(index=False)
        )

        print()
        print("TOP NEGATIVE")
        print("-" * 95)
        print(
            result.sort_values(
                "beta_species",
                ascending=True,
            )
            .head(10)[
                [
                    "species",
                    "beta_species",
                    "p_value",
                    "FDR_within_outcome379",
                ]
            ]
            .to_string(index=False)
        )

        summary_rows.append(
            {
                "outcome_name": outcome_name,
                "outcome_col": outcome_col,
                "N": len(y),
                "species_tested": len(species),
                "significant_within_FDR05": n_sig,
                "positive_within_FDR05": n_pos,
                "negative_within_FDR05": n_neg,
                "min_p": float(result["p_value"].min()),
                "min_FDR_within": float(
                    result["FDR_within_outcome379"].min()
                ),
                "max_positive_beta": float(
                    result["beta_species"].max()
                ),
                "min_negative_beta": float(
                    result["beta_species"].min()
                ),
            }
        )

        result.to_csv(
            MODEL_DIR
            / f"05_microbiome_cgm_{outcome_name}_model0.csv",
            index=False,
        )

        all_results.append(result)

    combined = pd.concat(
        all_results,
        ignore_index=True,
    )

    # Stricter sensitivity correction across all 1137 tests.
    combined["FDR_global1137"] = bh_fdr(
        combined["p_value"].to_numpy()
    )

    combined["significant_global_FDR05"] = (
        combined["FDR_global1137"] < 0.05
    )

    # Re-save outcome-specific tables including global FDR.
    for outcome_name in OUTCOMES:
        piece = combined.loc[
            combined["outcome_name"].eq(outcome_name)
        ].copy()

        piece.to_csv(
            MODEL_DIR
            / f"05_microbiome_cgm_{outcome_name}_model0.csv",
            index=False,
        )

    combined.to_csv(
        MODEL_DIR
        / "05_microbiome_cgm_all_model0.csv",
        index=False,
    )

    summary = pd.DataFrame(summary_rows)

    global_summary = (
        combined.groupby(
            "outcome_name",
            as_index=False,
        )
        .agg(
            significant_global_FDR05=(
                "significant_global_FDR05",
                "sum",
            ),
            min_FDR_global1137=(
                "FDR_global1137",
                "min",
            ),
        )
    )

    summary = summary.merge(
        global_summary,
        on="outcome_name",
        how="left",
        validate="one_to_one",
    )

    summary.to_csv(
        REPORT_DIR
        / "05_microbiome_cgm_model0_summary.csv",
        index=False,
    )

    # Cross-outcome audit: not a primary inferential target,
    # but useful for seeing whether the same species recur.
    wide_sig = combined.pivot(
        index="species",
        columns="outcome_name",
        values="significant_within_FDR05",
    )

    wide_beta = combined.pivot(
        index="species",
        columns="outcome_name",
        values="beta_species",
    )

    outcome_order = list(OUTCOMES.keys())

    recurrence = pd.DataFrame(
        index=wide_sig.index
    )

    recurrence["n_outcomes_FDR05"] = (
        wide_sig[outcome_order]
        .sum(axis=1)
        .astype(int)
    )

    recurrence["all3_FDR05"] = (
        recurrence["n_outcomes_FDR05"] == 3
    )

    recurrence["all3_positive"] = (
        wide_beta[outcome_order]
        .gt(0)
        .all(axis=1)
    )

    recurrence["all3_negative"] = (
        wide_beta[outcome_order]
        .lt(0)
        .all(axis=1)
    )

    for outcome_name in outcome_order:
        recurrence[f"{outcome_name}_beta"] = (
            wide_beta[outcome_name]
        )

        recurrence[
            f"{outcome_name}_FDR_within"
        ] = (
            combined.loc[
                combined["outcome_name"].eq(outcome_name),
                ["species", "FDR_within_outcome379"],
            ]
            .set_index("species")
            ["FDR_within_outcome379"]
        )

    recurrence = (
        recurrence
        .reset_index()
        .sort_values(
            [
                "n_outcomes_FDR05",
                "species",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    recurrence.to_csv(
        REPORT_DIR
        / "05_microbiome_cgm_cross_outcome_recurrence_model0.csv",
        index=False,
    )

    cohort_summary = pd.DataFrame(
        [
            {
                "microbiome_participants": len(micro),
                "cgm_participants": len(cgm),
                "microbiome_cgm_inner_join": len(cohort),
                "explicit_known_diabetes_or_a10_excluded": int(
                    explicit_excluded.sum()
                ),
                "exclusion_status_unknown_but_retained": int(
                    exclusion_unknown.loc[
                        ~explicit_excluded
                    ].sum()
                ),
                "primary_eligible_before_outcome_missingness": len(primary),
                "species_tested": len(species),
                "mean_glucose_N": int(
                    np.isfinite(
                        pd.to_numeric(
                            primary["cgm_mean_z"],
                            errors="coerce",
                        )
                    ).sum()
                ),
                "glucose_cv_N": int(
                    np.isfinite(
                        pd.to_numeric(
                            primary["cgm_cv_z"],
                            errors="coerce",
                        )
                    ).sum()
                ),
                "time_above_140_N": int(
                    np.isfinite(
                        pd.to_numeric(
                            primary["cgm_above_140_z"],
                            errors="coerce",
                        )
                    ).sum()
                ),
            }
        ]
    )

    cohort_summary.to_csv(
        REPORT_DIR
        / "05_microbiome_cgm_model0_cohort_summary.csv",
        index=False,
    )

    print()
    print("=" * 95)
    print("SUMMARY")
    print("=" * 95)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 95)
    print("CROSS-OUTCOME RECURRENCE")
    print("=" * 95)

    counts = (
        recurrence["n_outcomes_FDR05"]
        .value_counts()
        .sort_index()
    )

    for n_outcomes, n_species in counts.items():
        print(
            f"Significant in {int(n_outcomes)} / 3 outcomes: "
            f"{int(n_species)} species"
        )

    print()
    print("=" * 95)
    print("INTERPRETATION")
    print("=" * 95)
    print("This is MODEL 0 (unadjusted).")
    print(
        "Primary microbiome-wide multiplicity correction is "
        "BH-FDR within each CGM phenotype (379 tests)."
    )
    print(
        "FDR_global1137 is saved as a stricter sensitivity correction "
        "across all three phenotype families."
    )
    print(
        "Do not yet interpret significant species as independent "
        "microbiome-CGM associations; Model 2 adjustment is required next."
    )

    print()
    print("=" * 95)
    print("SAVED")
    print("=" * 95)
    print(
        MODEL_DIR
        / "05_microbiome_cgm_all_model0.csv"
    )
    print(
        REPORT_DIR
        / "05_microbiome_cgm_model0_summary.csv"
    )
    print(
        REPORT_DIR
        / "05_microbiome_cgm_model0_cohort_summary.csv"
    )
    print(
        REPORT_DIR
        / "05_microbiome_cgm_cross_outcome_recurrence_model0.csv"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run synthetic regression/FDR tests without reading HPP data.",
    )
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    run_analysis()


if __name__ == "__main__":
    main()
