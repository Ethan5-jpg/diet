#!/usr/bin/env python3
"""Step 15f — New diet -> microbiome -> CGM cross-sectional mediation-style analysis.

This is the new-exposure analogue of canonical Step10.

Inputs
------
Formal new Diet-CGM cohort:
    outputs/new_diet_extension/data/15d0_new_diet_cgm_cohort.csv

Robust bridge paths:
    outputs/new_diet_extension/bridge/15e1_robust_candidate_paths.csv

Microbiome:
    gut_microbiome_deal/data/08_species_clr_zscore.csv

Frozen candidate set from Step15e1
----------------------------------
NOVA4:
    mean_glucose      5
    glucose_cv       14
    time_above_140    1

Carbohydrate_pct (exploratory):
    mean_glucose     14
    glucose_cv       10
    time_above_140    1

Total = 45 robust paths.

Model
-----
X = diet exposure Z
M = species CLR-Z
Y = CGM phenotype Z

Mediator:
    M ~ X + covariates

Outcome:
    Y ~ X + M + covariates

Total:
    Y ~ X + covariates

indirect = a * b

Covariates
----------
Mirrors the canonical Model3 mediation adjustment and adds the prespecified
mean_daily_energy_kcal term used throughout the new-exposure adjusted models:

- age
- sex
- education
- smoking
- sleep duration
- physical activity
- vitamin use
- hormone use
- BMI
- CGM device
- mean_daily_energy_kcal

No hPDI-specific alcohol term is used.
Antibiotic/PPI are NOT included in this primary run; if the primary/strict
results are meaningful, protocol-window antibiotic/PPI sensitivity should be
performed afterward, as was done for the original score analysis.

Multiplicity
------------
Primary mediation FDR:
    BH-FDR within each diet-exposure x CGM-outcome candidate family.

Sensitivity:
    BH-FDR across all 45 paths.

Interpretation
--------------
Cross-sectional mediation-style / indirect-association analysis only.
Do NOT interpret as causal or temporal mediation.

Cohort modes
------------
primary:
    primary_cgm_analysis_eligible == True

strict:
    strict_known_nondiabetes_a10_eligible == True

Outputs are isolated under:
    outputs/new_diet_extension/mediation/
"""

from __future__ import annotations

from pathlib import Path
import argparse
import time
import numpy as np
import pandas as pd


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
BASE = (
    ROOT / "diet_microbiome_glucose_analysis" / "outputs"
    / "new_diet_extension"
)

COHORT_FILE = BASE / "data" / "15d0_new_diet_cgm_cohort.csv"
BRIDGE_FILE = BASE / "bridge" / "15e1_robust_candidate_paths.csv"
MICRO_FILE = ROOT / "gut_microbiome_deal" / "data" / "08_species_clr_zscore.csv"

OUT_DIR = BASE / "mediation"

EXPOSURES = {
    "NOVA4": {
        "column": "NOVA4_z",
        "analysis_role": "primary_extension",
    },
    "Carbohydrate_pct": {
        "column": "Carbohydrate_pct_z",
        "analysis_role": "exploratory_extension",
    },
}

OUTCOMES = {
    "mean_glucose": "cgm_mean_z",
    "glucose_cv": "cgm_cv_z",
    "time_above_140": "cgm_above_140_z",
}

EXPECTED_CONTEXT_COUNTS = {
    ("NOVA4", "mean_glucose"): 5,
    ("NOVA4", "glucose_cv"): 14,
    ("NOVA4", "time_above_140"): 1,
    ("Carbohydrate_pct", "mean_glucose"): 14,
    ("Carbohydrate_pct", "glucose_cv"): 10,
    ("Carbohydrate_pct", "time_above_140"): 1,
}

COVARIATES = [
    "age_years",
    "sex",
    "education_level",
    "smoking_status",
    "sleep_duration_hours_day",
    "physical_activity_met_h_week",
    "vitamin_use",
    "hormone_use",
    "bmi",
    "cgm_device_type",
    "mean_daily_energy_kcal",
]

CATEGORICAL = {
    "sex",
    "education_level",
    "smoking_status",
    "cgm_device_type",
}


def require(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


def norm_id(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def truthy(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).eq(1)
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "1.0", "yes", "y", "t"})
    )


def bh_fdr(p_values) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan, dtype=float)

    ok = np.isfinite(p)
    if not ok.any():
        return out

    p_ok = p[ok]
    n = len(p_ok)
    order = np.argsort(p_ok)
    ranked = p_ok[order]

    q = ranked * n / np.arange(1, n + 1, dtype=float)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    restored = np.empty(n, dtype=float)
    restored[order] = q
    out[ok] = restored
    return out


def terminal_species(full_taxonomy: str) -> str:
    text = str(full_taxonomy)
    if "|s__" in text:
        return text.rsplit("|s__", 1)[-1]
    return text


def encode_covariates(df: pd.DataFrame) -> pd.DataFrame:
    parts = []

    for col in COVARIATES:
        if col in CATEGORICAL:
            dummies = pd.get_dummies(
                df[col].astype(str),
                prefix=col,
                drop_first=True,
                dtype=float,
            )
            parts.append(dummies)
        else:
            x = pd.to_numeric(df[col], errors="coerce").astype(float)
            parts.append(x.to_frame(col))

    if not parts:
        return pd.DataFrame(index=df.index)

    return pd.concat(parts, axis=1)


def add_intercept(*arrays) -> np.ndarray:
    n = len(arrays[0])
    cols = [np.ones(n, dtype=float)]
    for arr in arrays:
        cols.append(np.asarray(arr, dtype=float).reshape(n, -1))
    return np.column_stack(cols)


def mediation_point_estimate(x, m, y, cov):
    x = np.asarray(x, dtype=float)
    m = np.asarray(m, dtype=float)
    y = np.asarray(y, dtype=float)
    cov = np.asarray(cov, dtype=float)

    X_med = add_intercept(x, cov)
    X_out = add_intercept(x, m, cov)
    X_total = add_intercept(x, cov)

    beta_med, _, rank_med, _ = np.linalg.lstsq(X_med, m, rcond=None)
    beta_out, _, rank_out, _ = np.linalg.lstsq(X_out, y, rcond=None)
    beta_total, _, rank_total, _ = np.linalg.lstsq(X_total, y, rcond=None)

    a = float(beta_med[1])
    direct = float(beta_out[1])
    b = float(beta_out[2])
    total = float(beta_total[1])
    indirect = a * b

    prop = indirect / total if abs(total) > 1e-12 else np.nan

    condition = max(
        np.linalg.cond(X_med),
        np.linalg.cond(X_out),
        np.linalg.cond(X_total),
    )

    return {
        "a_diet_to_microbiome": a,
        "b_microbiome_to_cgm": b,
        "indirect_effect": indirect,
        "direct_effect": direct,
        "total_effect": total,
        "mediated_proportion": prop,
        "rank_mediator_model": int(rank_med),
        "rank_outcome_model": int(rank_out),
        "rank_total_model": int(rank_total),
        "condition_number_max": float(condition),
        "decomposition_gap_total_minus_direct_indirect": (
            total - direct - indirect
        ),
    }


def bootstrap_mediation(x, m, y, cov, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(x)

    indirect = np.empty(n_boot, dtype=float)
    direct = np.empty(n_boot, dtype=float)
    total = np.empty(n_boot, dtype=float)
    prop = np.full(n_boot, np.nan, dtype=float)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)

        est = mediation_point_estimate(
            x[idx],
            m[idx],
            y[idx],
            cov[idx, :],
        )

        indirect[i] = est["indirect_effect"]
        direct[i] = est["direct_effect"]
        total[i] = est["total_effect"]

        if abs(est["total_effect"]) > 1e-8:
            prop[i] = est["indirect_effect"] / est["total_effect"]

    # Canonical two-sided bootstrap sign P with +1 correction.
    le0 = (np.sum(indirect <= 0) + 1) / (n_boot + 1)
    ge0 = (np.sum(indirect >= 0) + 1) / (n_boot + 1)
    p_indirect = min(1.0, 2.0 * min(le0, ge0))

    valid_prop = prop[np.isfinite(prop)]

    return {
        "indirect_CI95_lower": float(np.percentile(indirect, 2.5)),
        "indirect_CI95_upper": float(np.percentile(indirect, 97.5)),
        "direct_CI95_lower": float(np.percentile(direct, 2.5)),
        "direct_CI95_upper": float(np.percentile(direct, 97.5)),
        "total_CI95_lower": float(np.percentile(total, 2.5)),
        "total_CI95_upper": float(np.percentile(total, 97.5)),
        "mediated_proportion_CI95_lower": (
            float(np.percentile(valid_prop, 2.5))
            if len(valid_prop) else np.nan
        ),
        "mediated_proportion_CI95_upper": (
            float(np.percentile(valid_prop, 97.5))
            if len(valid_prop) else np.nan
        ),
        "bootstrap_p_indirect": float(p_indirect),
        "bootstrap_n": int(n_boot),
        "bootstrap_valid_proportion_n": int(len(valid_prop)),
        "bootstrap_total_crosses_zero_fraction": float(
            np.mean(np.sign(total) != np.sign(np.nanmedian(total)))
        ),
    }


def run_one_path(
    row,
    cohort,
    microbiome,
    bootstrap,
    base_seed,
):
    exposure = str(row["diet_score"])
    outcome = str(row["cgm_outcome"])
    species = str(row["species"])

    exposure_col = EXPOSURES[exposure]["column"]
    outcome_col = OUTCOMES[outcome]

    if species not in microbiome.columns:
        raise RuntimeError(
            f"Exact species column absent from microbiome CLR-Z table:\n{species}"
        )

    needed = [
        "participant_id",
        exposure_col,
        outcome_col,
        *COVARIATES,
    ]

    left = cohort[needed].copy().rename(
        columns={
            exposure_col: "_X",
            outcome_col: "_Y",
        }
    )
    med = microbiome[["participant_id", species]].copy().rename(
        columns={species: "_M"}
    )

    d = left.merge(
        med,
        on="participant_id",
        how="inner",
        validate="one_to_one",
    )

    for c in ["_X", "_Y", "_M"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # Numeric covariates are converted before complete-case filtering.
    for c in COVARIATES:
        if c not in CATEGORICAL:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna(
        subset=["_X", "_Y", "_M", *COVARIATES]
    ).copy()

    if len(d) < 100:
        raise RuntimeError(
            f"Too few complete cases for {exposure}/{outcome}/{species}: N={len(d)}"
        )

    cov_df = encode_covariates(d)

    x = d["_X"].to_numpy(float)
    m = d["_M"].to_numpy(float)
    y = d["_Y"].to_numpy(float)
    cov = cov_df.to_numpy(float)

    if not (
        np.isfinite(x).all()
        and np.isfinite(m).all()
        and np.isfinite(y).all()
        and np.isfinite(cov).all()
    ):
        raise RuntimeError(
            f"Non-finite model matrix for {exposure}/{outcome}/{species}"
        )

    point = mediation_point_estimate(x, m, y, cov)

    seed_offset = sum(ord(ch) for ch in f"{exposure}|{outcome}|{species}")
    boot = bootstrap_mediation(
        x,
        m,
        y,
        cov,
        n_boot=bootstrap,
        seed=(base_seed + seed_offset) % (2**32 - 1),
    )

    out = {
        "diet_score": exposure,
        "analysis_role": row["analysis_role"],
        "cgm_outcome": outcome,
        "species": species,
        "species_label": (
            row["species_label"]
            if "species_label" in row.index
            else terminal_species(species)
        ),
        "N": len(d),
        "n_covariate_columns_after_dummy_encoding": cov.shape[1],
        "covariates": ";".join(COVARIATES),
        **point,
        **boot,
    }

    # Carry forward bridge metadata when present.
    carry = [
        "direction_rule_type",
        "direction_rule_text",
        "expected_beta_product_sign",
        "model2_beta_diet",
        "model2_beta_cgm",
        "model2_beta_product",
        "model3_beta_diet",
        "model3_beta_cgm",
        "model3_beta_product",
    ]
    for c in carry:
        if c in row.index:
            out[c] = row[c]

    out["indirect_same_sign_as_total"] = bool(
        np.isfinite(out["total_effect"])
        and np.sign(out["indirect_effect"]) == np.sign(out["total_effect"])
    )
    out["indirect_CI_excludes_zero"] = bool(
        out["indirect_CI95_lower"] > 0
        or out["indirect_CI95_upper"] < 0
    )
    out["mediated_proportion_outside_0_1"] = bool(
        np.isfinite(out["mediated_proportion"])
        and (
            out["mediated_proportion"] < 0
            or out["mediated_proportion"] > 1
        )
    )
    out["proportion_potentially_unstable"] = bool(
        out["bootstrap_total_crosses_zero_fraction"] > 0.05
        or not np.isfinite(out["mediated_proportion"])
    )

    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort-mode",
        choices=["primary", "strict"],
        default="primary",
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--max-paths",
        type=int,
        default=None,
        help="Smoke-test only; restrict to first N frozen paths.",
    )
    args = parser.parse_args()

    if args.bootstrap < 20:
        raise RuntimeError("Use >=20 bootstrap iterations.")

    for p, label in [
        (COHORT_FILE, "Step15d0 cohort"),
        (BRIDGE_FILE, "Step15e1 robust candidate paths"),
        (MICRO_FILE, "microbiome CLR-Z table"),
    ]:
        require(p, label)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(COHORT_FILE, low_memory=False)
    bridges = pd.read_csv(BRIDGE_FILE, low_memory=False)
    micro = pd.read_csv(MICRO_FILE, low_memory=False)

    for label, df in [
        ("cohort", cohort),
        ("microbiome", micro),
    ]:
        if "participant_id" not in df.columns:
            raise RuntimeError(f"{label} missing participant_id")
        df["participant_id"] = norm_id(df["participant_id"])
        if df["participant_id"].duplicated().any():
            raise RuntimeError(f"{label} has duplicate participant_id")

    required_bridge = {
        "diet_score",
        "analysis_role",
        "cgm_outcome",
        "species",
    }
    missing_bridge = sorted(required_bridge - set(bridges.columns))
    if missing_bridge:
        raise RuntimeError(
            f"Bridge table missing columns: {missing_bridge}"
        )

    if bridges["analysis_role"].isna().any():
        raise RuntimeError(
            "Bridge candidate table still contains missing analysis_role."
        )

    if len(bridges) != 45:
        raise RuntimeError(
            f"Frozen Step15e1 candidate set should contain 45 paths; found {len(bridges)}"
        )

    observed_counts = (
        bridges.groupby(["diet_score", "cgm_outcome"])
        .size()
        .to_dict()
    )
    if observed_counts != EXPECTED_CONTEXT_COUNTS:
        raise RuntimeError(
            "Frozen bridge context counts differ from Step15e1.\n"
            f"Observed: {observed_counts}\n"
            f"Expected: {EXPECTED_CONTEXT_COUNTS}"
        )

    unknown_exposure = sorted(set(bridges["diet_score"]) - set(EXPOSURES))
    unknown_outcome = sorted(set(bridges["cgm_outcome"]) - set(OUTCOMES))
    if unknown_exposure:
        raise RuntimeError(f"Unexpected exposure(s): {unknown_exposure}")
    if unknown_outcome:
        raise RuntimeError(f"Unexpected outcome(s): {unknown_outcome}")

    for c in [
        *[cfg["column"] for cfg in EXPOSURES.values()],
        *OUTCOMES.values(),
        *COVARIATES,
    ]:
        if c not in cohort.columns:
            raise RuntimeError(f"Cohort missing required model column: {c}")

    if args.cohort_mode == "primary":
        flag = "primary_cgm_analysis_eligible"
    else:
        flag = "strict_known_nondiabetes_a10_eligible"

    if flag not in cohort.columns:
        raise RuntimeError(f"Cohort missing eligibility flag: {flag}")

    eligible = truthy(cohort[flag])
    formal_n = len(cohort)
    cohort = cohort.loc[eligible].copy()

    if args.max_paths is not None:
        bridges = bridges.head(args.max_paths).copy()

    # Descriptive recurrence annotation across the frozen 45-path extension set.
    recurrence = (
        bridges.groupby("species", as_index=False)
        .agg(
            n_robust_paths_new_extension=("species", "size"),
            n_diet_exposures_new_extension=("diet_score", "nunique"),
            n_cgm_outcomes_new_extension=("cgm_outcome", "nunique"),
        )
    )
    bridges = bridges.merge(
        recurrence,
        on="species",
        how="left",
        validate="many_to_one",
    )

    bridges["recurrence_class_new_extension"] = np.where(
        bridges["n_robust_paths_new_extension"].ge(2),
        "recurrent_multiple_robust_paths",
        "single_robust_path",
    )

    # Deterministic analysis order: primary extension first, recurrent first.
    role_rank = {
        "primary_extension": 1,
        "exploratory_extension": 2,
    }
    bridges["_role_rank"] = bridges["analysis_role"].map(role_rank).fillna(99)
    bridges = (
        bridges.sort_values(
            [
                "_role_rank",
                "n_robust_paths_new_extension",
                "diet_score",
                "cgm_outcome",
                "species",
            ],
            ascending=[True, False, True, True, True],
        )
        .drop(columns="_role_rank")
        .reset_index(drop=True)
    )

    print("=== STEP 15f NEW DIET -> MICROBIOME -> CGM MEDIATION-STYLE ===")
    print(f"COHORT_MODE={args.cohort_mode}")
    print(f"FORMAL_COHORT_N={formal_n}")
    print(f"ELIGIBLE_COHORT_N={len(cohort)}")
    print(f"BOOTSTRAP={args.bootstrap}")
    print(f"PATHS_SELECTED={len(bridges)}")
    print(f"UNIQUE_EXACT_SPECIES={bridges['species'].nunique()}")
    print("ANTIBIOTIC_PPI_INCLUDED=False")
    print("MEAN_DAILY_ENERGY_INCLUDED=True")
    print("CAUSAL_MEDIATION_CLAIM=False")

    print("\n--- SELECTED PATHS BY CONTEXT ---")
    print(
        bridges.groupby(
            ["diet_score", "analysis_role", "cgm_outcome"],
            as_index=False,
        )
        .agg(
            paths=("species", "size"),
            unique_exact_species=("species", "nunique"),
        )
        .to_string(index=False)
    )

    results = []
    start = time.time()

    for i, (_, row) in enumerate(bridges.iterrows(), start=1):
        label = (
            row["species_label"]
            if "species_label" in row.index
            else terminal_species(row["species"])
        )

        print(
            f"\n[{i}/{len(bridges)}] "
            f"{row['diet_score']} -> {label} -> {row['cgm_outcome']}"
        )

        t0 = time.time()
        res = run_one_path(
            row=row,
            cohort=cohort,
            microbiome=micro,
            bootstrap=args.bootstrap,
            base_seed=args.seed,
        )
        res["cohort_mode"] = args.cohort_mode
        res["elapsed_seconds"] = time.time() - t0

        for c in [
            "n_robust_paths_new_extension",
            "n_diet_exposures_new_extension",
            "n_cgm_outcomes_new_extension",
            "recurrence_class_new_extension",
        ]:
            res[c] = row[c]

        results.append(res)

        print(
            f"N={res['N']} | "
            f"a={res['a_diet_to_microbiome']:+.5f} | "
            f"b={res['b_microbiome_to_cgm']:+.5f} | "
            f"indirect={res['indirect_effect']:+.6f} "
            f"[{res['indirect_CI95_lower']:+.6f}, "
            f"{res['indirect_CI95_upper']:+.6f}] | "
            f"Pboot={res['bootstrap_p_indirect']:.4g}"
        )

    results = pd.DataFrame(results)

    # Primary multiplicity: within each exposure x outcome candidate family.
    results["FDR_BH_within_exposure_outcome"] = np.nan
    for _, idx in results.groupby(
        ["diet_score", "cgm_outcome"]
    ).groups.items():
        idx = list(idx)
        results.loc[
            idx, "FDR_BH_within_exposure_outcome"
        ] = bh_fdr(
            results.loc[idx, "bootstrap_p_indirect"].to_numpy()
        )

    # Global 45-path sensitivity.
    results["FDR_BH_global_all_paths"] = bh_fdr(
        results["bootstrap_p_indirect"].to_numpy()
    )

    results["mediation_FDR05_primary"] = (
        results["FDR_BH_within_exposure_outcome"] < .05
    )

    results["candidate_mediator_consistent"] = (
        results["mediation_FDR05_primary"]
        & results["indirect_same_sign_as_total"]
        & results["indirect_CI_excludes_zero"]
    )

    results = results.sort_values(
        [
            "analysis_role",
            "mediation_FDR05_primary",
            "candidate_mediator_consistent",
            "FDR_BH_within_exposure_outcome",
            "bootstrap_p_indirect",
        ],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)

    prefix = (
        "15f_primary"
        if args.cohort_mode == "primary"
        else "15f_strict"
    )

    PATH_OUT = OUT_DIR / f"{prefix}_mediation_paths.csv"
    SUMMARY_OUT = OUT_DIR / f"{prefix}_mediation_summary.csv"
    SIG_OUT = OUT_DIR / f"{prefix}_mediation_significant.csv"
    CONSISTENT_OUT = OUT_DIR / f"{prefix}_mediation_direction_consistent.csv"
    RECURRENCE_OUT = OUT_DIR / f"{prefix}_mediation_species_recurrence.csv"
    TXT_OUT = OUT_DIR / f"{prefix}_mediation_summary.txt"

    results.to_csv(PATH_OUT, index=False)

    sig = results.loc[results["mediation_FDR05_primary"]].copy()
    sig.to_csv(SIG_OUT, index=False)

    consistent = results.loc[
        results["candidate_mediator_consistent"]
    ].copy()
    consistent.to_csv(CONSISTENT_OUT, index=False)

    summary = (
        results.groupby(
            ["diet_score", "analysis_role", "cgm_outcome"],
            as_index=False,
        )
        .agg(
            paths_tested=("species", "size"),
            unique_species=("species", "nunique"),
            median_N=("N", "median"),
            significant_mediation_FDR05=(
                "mediation_FDR05_primary", "sum"
            ),
            direction_consistent_candidates=(
                "candidate_mediator_consistent", "sum"
            ),
            min_bootstrap_p=("bootstrap_p_indirect", "min"),
            min_FDR_primary=(
                "FDR_BH_within_exposure_outcome", "min"
            ),
        )
    )
    summary.to_csv(SUMMARY_OUT, index=False)

    recurrence_out = (
        results.groupby("species", as_index=False)
        .agg(
            species_label=("species_label", "first"),
            tested_paths=("species", "size"),
            significant_paths=(
                "mediation_FDR05_primary", "sum"
            ),
            consistent_paths=(
                "candidate_mediator_consistent", "sum"
            ),
            n_diet_exposures=("diet_score", "nunique"),
            n_cgm_outcomes=("cgm_outcome", "nunique"),
            diet_exposures=(
                "diet_score",
                lambda x: ";".join(sorted(set(map(str, x)))),
            ),
            cgm_outcomes=(
                "cgm_outcome",
                lambda x: ";".join(sorted(set(map(str, x)))),
            ),
        )
        .sort_values(
            ["consistent_paths", "significant_paths", "tested_paths"],
            ascending=False,
        )
    )
    recurrence_out.to_csv(RECURRENCE_OUT, index=False)

    elapsed = time.time() - start

    print("\n--- FINAL MEDIATION SUMMARY ---")
    print(summary.to_string(index=False))

    print("\n--- OVERALL ---")
    print(f"PATHS_TESTED={len(results)}")
    print(f"UNIQUE_SPECIES_TESTED={results['species'].nunique()}")
    print(
        "FDR_SIGNIFICANT="
        f"{int(results['mediation_FDR05_primary'].sum())}"
    )
    print(
        "DIRECTION_CONSISTENT="
        f"{int(results['candidate_mediator_consistent'].sum())}"
    )
    print(
        "MEDIATED_PROPORTION_OUTSIDE_0_1="
        f"{int(results['mediated_proportion_outside_0_1'].sum())}"
    )
    print(
        "POTENTIALLY_UNSTABLE_PROPORTION="
        f"{int(results['proportion_potentially_unstable'].sum())}"
    )
    print(f"ELAPSED_SECONDS={elapsed:.1f}")

    lines = [
        "=== STEP 15f NEW-DIET MEDIATION-STYLE SUMMARY ===",
        f"COHORT_MODE={args.cohort_mode}",
        f"BOOTSTRAP={args.bootstrap}",
        f"PATHS_TESTED={len(results)}",
        f"UNIQUE_SPECIES_TESTED={results['species'].nunique()}",
        f"FDR_SIGNIFICANT={int(results['mediation_FDR05_primary'].sum())}",
        f"DIRECTION_CONSISTENT={int(results['candidate_mediator_consistent'].sum())}",
        f"MEDIATED_PROPORTION_OUTSIDE_0_1={int(results['mediated_proportion_outside_0_1'].sum())}",
        f"POTENTIALLY_UNSTABLE_PROPORTION={int(results['proportion_potentially_unstable'].sum())}",
        "ANTIBIOTIC_PPI_INCLUDED=False",
        "CAUSAL_MEDIATION_CLAIM=False",
        "",
        "--- SUMMARY BY CONTEXT ---",
        summary.to_string(index=False),
        "",
        "INTERPRETATION:",
        "- Cross-sectional mediation-style indirect associations only.",
        "- Primary BH-FDR is within each exposure x CGM-outcome mediator family.",
        "- Carbohydrate_pct remains exploratory even if significant.",
        "- Mediated proportion is secondary and should be interpreted cautiously.",
        "- Protocol-window antibiotic/PPI sensitivity can be run after primary/strict QC.",
        "",
        f"PATH_RESULTS={PATH_OUT}",
        f"SUMMARY={SUMMARY_OUT}",
        f"SIGNIFICANT={SIG_OUT}",
        f"CONSISTENT={CONSISTENT_OUT}",
        f"RECURRENCE={RECURRENCE_OUT}",
    ]
    TXT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
