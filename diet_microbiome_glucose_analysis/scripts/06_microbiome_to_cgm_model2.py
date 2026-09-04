#!/usr/bin/env python3
"""
Microbiome -> CGM, Model 2 (primary adjusted model)

Model:
    CGM phenotype Z ~ species CLR-Z
                     + age
                     + sex
                     + education
                     + smoking
                     + sleep duration
                     + physical activity
                     + vitamin use
                     + hormone use
                     + CGM device type

Primary cohort:
    microbiome ∩ CGM
    exclude only participants with explicit exclude_diabetes_or_a10 == 1
    retain unknown exclusion status in the primary analysis
    require complete Model-2 covariates

Outcomes:
    cgm_mean_z
    cgm_cv_z
    cgm_above_140_z

Multiple testing:
    Primary: BH-FDR separately within each CGM phenotype (379 species tests)
    Complementary: BH-FDR across all 1137 tests jointly

Notes:
    - This is the microbiome -> CGM analysis, so no diet-score-specific
      alcohol covariate is added here.
    - Continuous covariates are z-standardized within each outcome-specific
      complete-case cohort.
    - Species are the already processed CLR-Z features from
      08_species_clr_zscore.csv.

Run:
    python 06_microbiome_to_cgm_model2.py --self-test
    python 06_microbiome_to_cgm_model2.py
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

MODEL0_FILE = MODEL_DIR / "05_microbiome_cgm_all_model0.csv"


# ============================================================
# ANALYSIS DEFINITIONS
# ============================================================

OUTCOMES = {
    "mean_glucose": "cgm_mean_z",
    "glucose_cv": "cgm_cv_z",
    "time_above_140": "cgm_above_140_z",
}

CONTINUOUS_COVARS = [
    "age_years",
    "sleep_duration_hours_day",
    "physical_activity_met_h_week",
]

BINARY_COVARS = [
    "vitamin_use",
    "hormone_use",
]

CATEGORICAL_COVARS = [
    "sex",
    "education_level",
    "smoking_status",
    "cgm_device_type",
]

MODEL2_COVARS = (
    CONTINUOUS_COVARS
    + BINARY_COVARS
    + CATEGORICAL_COVARS
)

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
            f"Taxonomic-name candidates: {len(taxonomic)}\n"
            f"Non-metadata candidates: {len(fallback)}"
        )

    return fallback


def normalize_text(series):
    return series.astype(str).str.strip().str.lower()


def numeric_binary(series, name):
    x = pd.to_numeric(series, errors="coerce")

    if x.isna().any():
        raise RuntimeError(f"{name} contains missing/non-numeric values.")

    unique = set(x.unique().tolist())

    if not unique.issubset({0, 1, 0.0, 1.0}):
        raise RuntimeError(
            f"{name} expected 0/1; got values: {sorted(unique)[:20]}"
        )

    return x.to_numpy(dtype=float)


def zscore(series, name):
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)

    if not np.isfinite(x).all():
        raise RuntimeError(f"{name} contains missing/non-finite values.")

    mean = float(x.mean())
    sd = float(x.std(ddof=0))

    if not np.isfinite(sd) or sd <= 0:
        raise RuntimeError(f"{name} has zero/non-finite SD.")

    return (x - mean) / sd


def build_covariate_design(frame):
    """
    Covariate-only design C, including intercept.
    References:
      sex: female
      education: high
      smoking: never
      CGM device: alphabetically first observed category
    """

    cols = [np.ones(len(frame), dtype=float)]
    names = ["intercept"]

    # Continuous covariates
    for variable in CONTINUOUS_COVARS:
        cols.append(zscore(frame[variable], variable))
        names.append(f"{variable}_z")

    # Binary covariates
    for variable in BINARY_COVARS:
        cols.append(numeric_binary(frame[variable], variable))
        names.append(variable)

    # Sex: female reference
    sex = normalize_text(frame["sex"]).replace({
        "f": "female",
        "m": "male",
        "0": "female",
        "0.0": "female",
        "1": "male",
        "1.0": "male",
    })

    valid_sex = sex.isin(["female", "male"])
    if not valid_sex.all():
        bad = frame.loc[~valid_sex, "sex"].value_counts().head(20).to_dict()
        raise RuntimeError(f"Unexpected sex values: {bad}")

    cols.append(sex.eq("male").astype(float).to_numpy())
    names.append("sex_male")

    # Education: high reference
    education = normalize_text(frame["education_level"])
    education = education.str.replace("education_", "", regex=False)

    valid_education = education.isin(["high", "low"])
    if not valid_education.all():
        bad = (
            frame.loc[~valid_education, "education_level"]
            .value_counts()
            .head(20)
            .to_dict()
        )
        raise RuntimeError(f"Unexpected education values: {bad}")

    cols.append(education.eq("low").astype(float).to_numpy())
    names.append("education_low")

    # Smoking: never reference
    smoking = normalize_text(frame["smoking_status"])

    valid_smoking = smoking.isin(["never", "former", "current"])
    if not valid_smoking.all():
        bad = (
            frame.loc[~valid_smoking, "smoking_status"]
            .value_counts()
            .head(20)
            .to_dict()
        )
        raise RuntimeError(f"Unexpected smoking values: {bad}")

    cols.append(smoking.eq("former").astype(float).to_numpy())
    names.append("smoking_former")

    cols.append(smoking.eq("current").astype(float).to_numpy())
    names.append("smoking_current")

    # Device: deterministic reference category
    device = frame["cgm_device_type"].astype(str).str.strip()
    categories = sorted(device.unique().tolist())

    if len(categories) < 2:
        raise RuntimeError(
            "cgm_device_type has fewer than 2 categories "
            "in this complete-case cohort."
        )

    device_reference = categories[0]

    for category in categories[1:]:
        cols.append(device.eq(category).astype(float).to_numpy())
        safe = (
            str(category)
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
        )
        names.append(f"cgm_device_{safe}")

    C = np.column_stack(cols).astype(np.float64)

    rank = int(np.linalg.matrix_rank(C))
    condition = float(np.linalg.cond(C))

    if rank != C.shape[1]:
        raise RuntimeError(
            "Covariate design is rank deficient: "
            f"rank={rank}, columns={C.shape[1]}, names={names}"
        )

    return C, names, rank, condition, device_reference


def residualize_against_covariates(C, M):
    """
    Residualize vector/matrix M against columns of C using QR.
    """

    Q, _ = np.linalg.qr(C, mode="reduced")
    return M - Q @ (Q.T @ M)


def fit_all_species_adjusted(X, y, C):
    """
    Fit all adjusted models using Frisch-Waugh-Lovell:

        y ~ species_j + covariates

    X : N x P species matrix
    y : N vector
    C : N x K covariate design including intercept
    """

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)

    if not np.isfinite(X).all():
        raise RuntimeError("Species matrix contains NaN/Inf.")

    if not np.isfinite(y).all():
        raise RuntimeError("Outcome contains NaN/Inf.")

    if not np.isfinite(C).all():
        raise RuntimeError("Covariate matrix contains NaN/Inf.")

    n, p = X.shape
    rank_c = int(np.linalg.matrix_rank(C))

    if n <= rank_c + 2:
        raise RuntimeError("Insufficient residual degrees of freedom.")

    Xr = residualize_against_covariates(C, X)
    yr = residualize_against_covariates(C, y)

    sxx = np.sum(Xr ** 2, axis=0)

    if (sxx <= 1e-12).any():
        bad = np.where(sxx <= 1e-12)[0][:20].tolist()
        raise RuntimeError(
            "At least one species has no residual variance after "
            f"covariate adjustment. Column indexes: {bad}"
        )

    xy = Xr.T @ yr
    beta = xy / sxx

    # Full-model residual SSE for each species.
    yryr = float(yr @ yr)
    sse = yryr - (xy ** 2) / sxx
    sse = np.maximum(sse, 0.0)

    df_resid = n - rank_c - 1
    mse = sse / df_resid

    se = np.sqrt(mse / sxx)
    t_stat = beta / se
    p_value = 2 * t_dist.sf(np.abs(t_stat), df=df_resid)

    tcrit = float(t_dist.ppf(0.975, df=df_resid))
    ci_lower = beta - tcrit * se
    ci_upper = beta + tcrit * se

    # Partial R^2 for the species term.
    partial_r2 = (t_stat ** 2) / (t_stat ** 2 + df_resid)

    return {
        "beta": beta,
        "SE": se,
        "t": t_stat,
        "p_value": p_value,
        "CI95_lower": ci_lower,
        "CI95_upper": ci_upper,
        "partial_R2": partial_r2,
        "df_resid": df_resid,
        "rank_covariates": rank_c,
    }


def require_columns(frame, columns, label):
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise RuntimeError(
            f"{label} missing required columns: {missing}"
        )


def self_test():
    """
    Synthetic confounding test:
    species X0 is correlated with confounder z;
    y depends on both X0 and z.
    Adjusted beta should recover the true species effect.
    """

    print("=" * 100)
    print("SELF TEST — MICROBIOME -> CGM MODEL 2")
    print("=" * 100)

    rng = np.random.default_rng(20260904)

    n = 1200
    p = 25

    z = rng.normal(size=n)

    X = rng.normal(size=(n, p))
    X[:, 0] = 0.7 * z + rng.normal(scale=0.8, size=n)

    true_beta = 0.35

    y = (
        true_beta * X[:, 0]
        + 0.9 * z
        + rng.normal(scale=0.8, size=n)
    )

    C = np.column_stack([
        np.ones(n),
        z,
    ])

    adjusted = fit_all_species_adjusted(X, y, C)
    beta0 = float(adjusted["beta"][0])
    p0 = float(adjusted["p_value"][0])

    # Unadjusted simple-regression beta for comparison.
    x0 = X[:, 0]
    unadjusted_beta = float(
        np.sum((x0 - x0.mean()) * (y - y.mean()))
        / np.sum((x0 - x0.mean()) ** 2)
    )

    assert abs(beta0 - true_beta) < 0.08, (beta0, true_beta)
    assert p0 < 0.05
    assert abs(unadjusted_beta - true_beta) > abs(beta0 - true_beta)

    q = bh_fdr(adjusted["p_value"])
    assert np.isfinite(q).all()

    print(f"True species beta:       {true_beta:.4f}")
    print(f"Unadjusted species beta: {unadjusted_beta:.4f}")
    print(f"Adjusted species beta:   {beta0:.4f}")
    print(f"Adjusted p:              {p0:.3e}")
    print("FWL adjusted-regression test: PASS")
    print("BH-FDR test: PASS")
    print("SELF TEST: PASS")


# ============================================================
# MAIN
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

    print("=" * 100)
    print("MICROBIOME -> CGM — MODEL 2")
    print("=" * 100)
    print("Model: CGM phenotype Z ~ species CLR-Z + CGM Model-2 covariates")
    print("Covariates:")
    for x in MODEL2_COVARS:
        print("  -", x)

    print()
    print("Microbiome:", micro_file)
    print("CGM:", cgm_file)
    print("Covariates/exclusion:", covar_file)

    micro = pd.read_csv(micro_file, low_memory=False)
    cgm = pd.read_csv(cgm_file, low_memory=False)
    covar = pd.read_csv(covar_file, low_memory=False)

    for label, frame in [
        ("microbiome", micro),
        ("CGM", cgm),
        ("covariate master", covar),
    ]:
        require_columns(frame, ["participant_id"], label)
        frame["participant_id"] = normalize_id(frame["participant_id"])

        if frame["participant_id"].duplicated().any():
            bad = (
                frame.loc[
                    frame["participant_id"].duplicated(keep=False),
                    "participant_id",
                ]
                .value_counts()
                .head(20)
                .to_dict()
            )
            raise RuntimeError(
                f"Duplicate participant_id in {label}: {bad}"
            )

    require_columns(
        cgm,
        ["participant_id", *OUTCOMES.values()],
        "CGM file",
    )

    require_columns(
        covar,
        ["participant_id", EXCLUSION_COL, *MODEL2_COVARS],
        "covariate master",
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
            covar[["participant_id", EXCLUSION_COL, *MODEL2_COVARS]],
            on="participant_id",
            how="left",
            validate="one_to_one",
        )
    )

    exclusion_numeric = pd.to_numeric(
        cohort[EXCLUSION_COL],
        errors="coerce",
    )

    explicit_excluded = exclusion_numeric.eq(1)
    exclusion_unknown = exclusion_numeric.isna()

    primary = cohort.loc[~explicit_excluded].copy()

    print()
    print("=" * 100)
    print("COHORT")
    print("=" * 100)
    print("Microbiome participants:", len(micro))
    print("CGM participants:", len(cgm))
    print("Species:", len(species))
    print("Microbiome ∩ CGM:", len(cohort))
    print(
        "Explicit known diabetes/A10 excluded:",
        int(explicit_excluded.sum()),
    )
    print(
        "Exclusion status unknown but retained:",
        int(exclusion_unknown.loc[~explicit_excluded].sum()),
    )
    print(
        "Primary eligible before Model-2 complete-case filtering:",
        len(primary),
    )

    # Covariate coverage audit before outcome-specific filtering.
    print()
    print("=" * 100)
    print("MODEL-2 COVARIATE COVERAGE IN PRIMARY COHORT")
    print("=" * 100)

    coverage_rows = []

    for variable in MODEL2_COVARS:
        nonmissing = int(primary[variable].notna().sum())
        missing = int(primary[variable].isna().sum())

        coverage_rows.append({
            "variable": variable,
            "N_nonmissing": nonmissing,
            "N_missing": missing,
            "missing_pct": 100 * missing / len(primary),
        })

    coverage = pd.DataFrame(coverage_rows)
    print(coverage.to_string(index=False))

    coverage.to_csv(
        REPORT_DIR / "06_microbiome_cgm_model2_covariate_coverage.csv",
        index=False,
    )

    all_results = []
    summary_rows = []

    for outcome_name, outcome_col in OUTCOMES.items():
        print()
        print("=" * 100)
        print(outcome_name)
        print("=" * 100)

        needed = [
            outcome_col,
            *MODEL2_COVARS,
        ]

        complete = primary[needed].notna().all(axis=1)
        selected = primary.loc[complete].copy()

        if len(selected) < 100:
            raise RuntimeError(
                f"{outcome_name}: too few Model-2 complete cases: "
                f"{len(selected)}"
            )

        X = (
            selected[species]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float64)
        )

        if not np.isfinite(X).all():
            raise RuntimeError(
                f"{outcome_name}: species matrix has NaN/Inf."
            )

        y = pd.to_numeric(
            selected[outcome_col],
            errors="coerce",
        ).to_numpy(dtype=np.float64)

        if not np.isfinite(y).all():
            raise RuntimeError(
                f"{outcome_name}: outcome has NaN/Inf after filtering."
            )

        C, covar_names, covar_rank, covar_condition, device_reference = (
            build_covariate_design(selected)
        )

        fit = fit_all_species_adjusted(X, y, C)

        result = pd.DataFrame({
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
            "partial_R2": fit["partial_R2"],
        })

        result["FDR_within_outcome379"] = bh_fdr(
            result["p_value"].to_numpy()
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
        ]

        n_sig = len(sig)
        n_pos = int((sig["beta_species"] > 0).sum())
        n_neg = int((sig["beta_species"] < 0).sum())

        print("N:", len(selected))
        print("Species tested:", len(species))
        print("Covariate parameters incl. intercept:", C.shape[1])
        print("Covariate design rank:", covar_rank)
        print(f"Covariate condition number: {covar_condition:.3f}")
        print("CGM device reference:", device_reference)
        print("FDR<0.05 within 379:", n_sig)
        print("  positive:", n_pos)
        print("  negative:", n_neg)

        print()
        print("TOP POSITIVE")
        print("-" * 100)
        print(
            result.sort_values(
                ["beta_species", "p_value"],
                ascending=[False, True],
            )
            .head(10)[
                [
                    "species",
                    "beta_species",
                    "p_value",
                    "FDR_within_outcome379",
                    "partial_R2",
                ]
            ]
            .to_string(index=False)
        )

        print()
        print("TOP NEGATIVE")
        print("-" * 100)
        print(
            result.sort_values(
                ["beta_species", "p_value"],
                ascending=[True, True],
            )
            .head(10)[
                [
                    "species",
                    "beta_species",
                    "p_value",
                    "FDR_within_outcome379",
                    "partial_R2",
                ]
            ]
            .to_string(index=False)
        )

        result.to_csv(
            MODEL_DIR / f"06_microbiome_cgm_{outcome_name}_model2.csv",
            index=False,
        )

        all_results.append(result)

        summary_rows.append({
            "outcome_name": outcome_name,
            "outcome_col": outcome_col,
            "N": len(selected),
            "species_tested": len(species),
            "n_covariate_parameters_including_intercept": C.shape[1],
            "covariate_design_rank": covar_rank,
            "covariate_condition_number": covar_condition,
            "cgm_device_reference": device_reference,
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
            "max_partial_R2": float(
                result["partial_R2"].max()
            ),
        })

    combined = pd.concat(
        all_results,
        ignore_index=True,
    )

    # Joint BH correction across all 1137 tests.
    # This controls FDR for the combined family; it is complementary,
    # not guaranteed to be numerically more conservative within every
    # individual phenotype.
    combined["FDR_global1137"] = bh_fdr(
        combined["p_value"].to_numpy()
    )
    combined["significant_global_FDR05"] = (
        combined["FDR_global1137"] < 0.05
    )

    # Re-save phenotype files with global FDR included.
    for outcome_name in OUTCOMES:
        piece = combined.loc[
            combined["outcome_name"].eq(outcome_name)
        ].copy()

        piece.to_csv(
            MODEL_DIR / f"06_microbiome_cgm_{outcome_name}_model2.csv",
            index=False,
        )

    combined.to_csv(
        MODEL_DIR / "06_microbiome_cgm_all_model2.csv",
        index=False,
    )

    summary = pd.DataFrame(summary_rows)

    global_counts = (
        combined.groupby("outcome_name", as_index=False)
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
        global_counts,
        on="outcome_name",
        how="left",
        validate="one_to_one",
    )

    summary.to_csv(
        REPORT_DIR / "06_microbiome_cgm_model2_summary.csv",
        index=False,
    )

    # ========================================================
    # MODEL 0 -> MODEL 2 COMPARISON
    # ========================================================

    if MODEL0_FILE.exists():
        model0 = pd.read_csv(MODEL0_FILE, low_memory=False)

        needed0 = {
            "outcome_name",
            "species",
            "beta_species",
            "FDR_within_outcome379",
        }

        if needed0.issubset(model0.columns):
            prev = model0[
                [
                    "outcome_name",
                    "species",
                    "beta_species",
                    "p_value",
                    "FDR_within_outcome379",
                ]
            ].rename(
                columns={
                    "beta_species": "beta_model0",
                    "p_value": "p_model0",
                    "FDR_within_outcome379": "FDR_model0_within379",
                }
            )

            current = combined[
                [
                    "outcome_name",
                    "species",
                    "beta_species",
                    "p_value",
                    "FDR_within_outcome379",
                ]
            ].rename(
                columns={
                    "beta_species": "beta_model2",
                    "p_value": "p_model2",
                    "FDR_within_outcome379": "FDR_model2_within379",
                }
            )

            comp = prev.merge(
                current,
                on=["outcome_name", "species"],
                validate="one_to_one",
            )

            comp["significant_model0"] = (
                comp["FDR_model0_within379"] < 0.05
            )
            comp["significant_model2"] = (
                comp["FDR_model2_within379"] < 0.05
            )

            comp["same_direction"] = (
                np.sign(comp["beta_model0"])
                == np.sign(comp["beta_model2"])
            )

            comp.to_csv(
                REPORT_DIR / "06_microbiome_cgm_model0_vs_model2_all.csv",
                index=False,
            )

            comparison_summary = []

            for outcome_name in OUTCOMES:
                piece = comp.loc[
                    comp["outcome_name"].eq(outcome_name)
                ].copy()

                retained = int(
                    (
                        piece["significant_model0"]
                        & piece["significant_model2"]
                    ).sum()
                )
                lost = int(
                    (
                        piece["significant_model0"]
                        & ~piece["significant_model2"]
                    ).sum()
                )
                gained = int(
                    (
                        ~piece["significant_model0"]
                        & piece["significant_model2"]
                    ).sum()
                )

                both = piece.loc[
                    piece["significant_model0"]
                    & piece["significant_model2"]
                ]

                same_direction_both = int(
                    both["same_direction"].sum()
                )

                rho, rho_p = spearmanr(
                    piece["beta_model0"],
                    piece["beta_model2"],
                )

                comparison_summary.append({
                    "outcome_name": outcome_name,
                    "model0_significant_FDR05": int(
                        piece["significant_model0"].sum()
                    ),
                    "model2_significant_FDR05": int(
                        piece["significant_model2"].sum()
                    ),
                    "retained_significant": retained,
                    "lost_after_adjustment": lost,
                    "gained_after_adjustment": gained,
                    "same_direction_among_retained": same_direction_both,
                    "beta_spearman_all_species": float(rho),
                    "beta_spearman_p": float(rho_p),
                })

            comparison_summary = pd.DataFrame(
                comparison_summary
            )

            comparison_summary.to_csv(
                REPORT_DIR
                / "06_microbiome_cgm_model0_vs_model2_summary.csv",
                index=False,
            )

            print()
            print("=" * 100)
            print("MODEL 0 -> MODEL 2 COMPARISON")
            print("=" * 100)
            print(comparison_summary.to_string(index=False))
        else:
            print()
            print(
                "NOTE: Model-0 file found but required comparison "
                "columns were missing; comparison skipped."
            )
    else:
        print()
        print(
            "NOTE: Model-0 combined file not found; "
            "Model0->Model2 comparison skipped."
        )

    # ========================================================
    # CROSS-OUTCOME RECURRENCE
    # ========================================================

    wide_sig = combined.pivot(
        index="species",
        columns="outcome_name",
        values="significant_within_FDR05",
    )

    recurrence = pd.DataFrame(index=wide_sig.index)

    outcome_order = list(OUTCOMES.keys())

    recurrence["n_outcomes_FDR05"] = (
        wide_sig[outcome_order]
        .sum(axis=1)
        .astype(int)
    )

    for outcome_name in outcome_order:
        piece = (
            combined.loc[
                combined["outcome_name"].eq(outcome_name),
                [
                    "species",
                    "beta_species",
                    "FDR_within_outcome379",
                ],
            ]
            .set_index("species")
        )

        recurrence[f"{outcome_name}_beta"] = (
            piece["beta_species"]
        )
        recurrence[f"{outcome_name}_FDR_within"] = (
            piece["FDR_within_outcome379"]
        )

    recurrence = (
        recurrence.reset_index()
        .sort_values(
            ["n_outcomes_FDR05", "species"],
            ascending=[False, True],
        )
    )

    recurrence.to_csv(
        REPORT_DIR
        / "06_microbiome_cgm_cross_outcome_recurrence_model2.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(summary.to_string(index=False))

    print()
    print("=" * 100)
    print("CROSS-OUTCOME RECURRENCE")
    print("=" * 100)

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
    print("=" * 100)
    print("INTERPRETATION")
    print("=" * 100)
    print("This is MODEL 2, the primary adjusted microbiome -> CGM analysis.")
    print(
        "Primary multiplicity correction: BH-FDR within each phenotype "
        "(379 species tests)."
    )
    print(
        "FDR_global1137 controls FDR over the combined 1137-test family; "
        "it is complementary and is NOT guaranteed to be numerically "
        "more conservative within each phenotype."
    )
    print(
        "The next sensitivity step is Model 3 = Model 2 + BMI."
    )

    print()
    print("=" * 100)
    print("SAVED")
    print("=" * 100)
    print(MODEL_DIR / "06_microbiome_cgm_all_model2.csv")
    print(REPORT_DIR / "06_microbiome_cgm_model2_summary.csv")
    print(
        REPORT_DIR
        / "06_microbiome_cgm_model0_vs_model2_summary.csv"
    )
    print(
        REPORT_DIR
        / "06_microbiome_cgm_cross_outcome_recurrence_model2.csv"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run synthetic adjusted-regression test without HPP data.",
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    run_analysis()


if __name__ == "__main__":
    main()
