#!/usr/bin/env python3
"""
QC for the current four-score Diet -> CGM analysis.

Checks:
1) BMI same-cohort comparison: Model 2 vs Model 3 on the exact same Model-3 participants.
2) Strict known non-diabetes/A10 sensitivity for Model 2 and Model 3.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.stats import t as t_dist

ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
ANALYSIS = ROOT / "diet_microbiome_glucose_analysis"

COHORT_FILE = ANALYSIS / "outputs" / "data" / "00_current4_diet_cgm_cohort.csv"
PRIMARY_MODEL2_FILE = ANALYSIS / "outputs" / "models" / "02_current4_diet_cgm_model2.csv"
PRIMARY_MODEL3_FILE = ANALYSIS / "outputs" / "models" / "03_current4_diet_cgm_model3_bmi.csv"

REPORT_DIR = ANALYSIS / "outputs" / "reports"
MODEL_DIR = ANALYSIS / "outputs" / "models"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SCORES = {
    "AHEI": "AHEI_z",
    "AMED": "AMED_z",
    "hPDI": "hPDI_z",
    "rEDIH": "rEDIH_z",
}

OUTCOMES = {
    "mean_glucose": "cgm_mean_z",
    "glucose_cv": "cgm_cv_z",
    "time_above_140": "cgm_above_140_z",
}

BASE_CONTINUOUS = [
    "age_years",
    "sleep_duration_hours_day",
    "physical_activity_met_h_week",
]

BINARY = [
    "vitamin_use",
    "hormone_use",
]

BASE_CATEGORICAL = [
    "sex",
    "education_level",
    "smoking_status",
    "cgm_device_type",
]


def bh_fdr(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if not np.isfinite(p).all():
        raise RuntimeError("BH-FDR received NaN/Inf p-values.")
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


def as_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    text = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "1.0": True,
        "false": False,
        "0": False,
        "0.0": False,
        "nan": False,
        "none": False,
        "": False,
    }

    unknown = sorted(set(text.unique()) - set(mapping))
    if unknown:
        raise RuntimeError(
            "Unexpected boolean values: "
            + ", ".join(map(str, unknown[:20]))
        )

    return text.map(mapping).astype(bool)


def standardize(series, name):
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise RuntimeError(f"{name} contains missing/non-finite values.")
    mean = float(x.mean())
    sd = float(x.std(ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        raise RuntimeError(f"{name} has zero/non-finite SD.")
    return (x - mean) / sd


def binary_numeric(series, name):
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise RuntimeError(f"{name} contains missing/non-finite values.")
    unique = set(np.unique(x).tolist())
    if not unique.issubset({0.0, 1.0}):
        raise RuntimeError(
            f"{name} expected 0/1; got {sorted(unique)[:20]}"
        )
    return x


def normalize_text(series):
    return series.astype(str).str.strip().str.lower()


def build_design(frame, score_name, score_col, add_bmi):
    names = ["intercept", score_col]

    score = pd.to_numeric(frame[score_col], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(score).all():
        raise RuntimeError(f"{score_name}: invalid diet-score values.")

    cols = [
        np.ones(len(frame), dtype=float),
        score,
    ]

    continuous = BASE_CONTINUOUS.copy()

    if score_name == "hPDI":
        continuous.append("alcohol_intake_g_day")

    if add_bmi:
        continuous.append("bmi")

    for variable in continuous:
        cols.append(standardize(frame[variable], variable))
        names.append(f"{variable}_z")

    for variable in BINARY:
        cols.append(binary_numeric(frame[variable], variable))
        names.append(variable)

    # sex: female reference
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

    # education: high reference
    education = (
        normalize_text(frame["education_level"])
        .str.replace("education_", "", regex=False)
    )
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

    # smoking: never reference
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

    cols.extend([
        smoking.eq("former").astype(float).to_numpy(),
        smoking.eq("current").astype(float).to_numpy(),
    ])
    names.extend([
        "smoking_former",
        "smoking_current",
    ])

    # CGM device: deterministic reference
    device = frame["cgm_device_type"].astype(str).str.strip()
    categories = sorted(device.unique().tolist())

    if len(categories) < 2:
        raise RuntimeError(
            "cgm_device_type has fewer than 2 categories in this cohort."
        )

    reference = categories[0]

    for category in categories[1:]:
        cols.append(device.eq(category).astype(float).to_numpy())
        safe = str(category).replace(" ", "_").replace("/", "_")
        names.append(f"cgm_device_{safe}")

    X = np.column_stack(cols).astype(np.float64)

    rank = int(np.linalg.matrix_rank(X))
    if rank != X.shape[1]:
        raise RuntimeError(
            f"Rank-deficient design: rank={rank}, columns={X.shape[1]}, "
            f"names={names}"
        )

    return X, names, float(np.linalg.cond(X)), reference


def fit_ols(frame, score_name, score_col, outcome_col, add_bmi):
    X, names, condition, device_reference = build_design(
        frame,
        score_name,
        score_col,
        add_bmi=add_bmi,
    )

    y = pd.to_numeric(frame[outcome_col], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise RuntimeError(f"{outcome_col} contains invalid values.")

    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    coef = xtx_inv @ X.T @ y

    fitted = X @ coef
    resid = y - fitted
    df_resid = n - k

    sse = float(resid @ resid)
    mse = sse / df_resid

    beta = float(coef[1])
    se = float(np.sqrt(mse * xtx_inv[1, 1]))
    t_value = beta / se
    p_value = float(2 * t_dist.sf(abs(t_value), df=df_resid))

    tcrit = float(t_dist.ppf(0.975, df=df_resid))
    ci_lower = beta - tcrit * se
    ci_upper = beta + tcrit * se

    return {
        "N": n,
        "beta_diet": beta,
        "SE_diet": se,
        "CI95_lower": ci_lower,
        "CI95_upper": ci_upper,
        "p_value": p_value,
        "condition_number": condition,
        "n_parameters": k,
        "df_resid": df_resid,
        "cgm_device_reference": device_reference,
    }


def required_covariates(score_name, model):
    cols = BASE_CONTINUOUS + BINARY + BASE_CATEGORICAL

    if score_name == "hPDI":
        cols = cols + ["alcohol_intake_g_day"]

    if model == 3:
        cols = cols + ["bmi"]

    return cols


def complete_mask(
    frame,
    score_name,
    score_col,
    outcome_col,
    model,
    eligibility_col,
):
    needed = [score_col, outcome_col] + required_covariates(score_name, model)

    return (
        as_bool(frame[eligibility_col])
        & frame[needed].notna().all(axis=1)
    )


def safe_corr(x, y):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    mask = (
        x.notna()
        & y.notna()
        & np.isfinite(x)
        & np.isfinite(y)
    )

    x = x.loc[mask]
    y = y.loc[mask]

    if len(x) < 3:
        return {
            "N": len(x),
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
        }

    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)

    return {
        "N": len(x),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_rho": float(sr),
        "spearman_p": float(sp),
    }


def require_columns(frame, columns):
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise RuntimeError(
            "Required columns missing from formal cohort: "
            + ", ".join(missing)
        )


def run_same_cohort_bmi_qc(df):
    rows = []
    corr_rows = []

    for score_name, score_col in SCORES.items():

        score_level_mask = (
            as_bool(df["primary_cgm_analysis_eligible"])
            & df[
                [score_col]
                + required_covariates(score_name, 3)
            ].notna().all(axis=1)
        )

        score_level = df.loc[score_level_mask].copy()

        for variable_label, variable_col in [
            ("diet_score", score_col),
            ("mean_glucose", "cgm_mean_z"),
            ("glucose_cv", "cgm_cv_z"),
            ("time_above_140", "cgm_above_140_z"),
        ]:
            stats = safe_corr(
                score_level["bmi"],
                score_level[variable_col],
            )

            corr_rows.append({
                "score": score_name,
                "BMI_correlated_with": variable_label,
                **stats,
            })

        for outcome_name, outcome_col in OUTCOMES.items():

            mask = complete_mask(
                df,
                score_name,
                score_col,
                outcome_col,
                model=3,
                eligibility_col="primary_cgm_analysis_eligible",
            )

            same = df.loc[mask].copy()

            no_bmi = fit_ols(
                same,
                score_name,
                score_col,
                outcome_col,
                add_bmi=False,
            )

            plus_bmi = fit_ols(
                same,
                score_name,
                score_col,
                outcome_col,
                add_bmi=True,
            )

            rows.append({
                "diet_score": score_name,
                "outcome_name": outcome_name,
                "same_cohort_N": len(same),
                "beta_model2_same_cohort": no_bmi["beta_diet"],
                "p_model2_same_cohort": no_bmi["p_value"],
                "beta_model3_plus_bmi": plus_bmi["beta_diet"],
                "p_model3_plus_bmi": plus_bmi["p_value"],
                "beta_change_model3_minus_model2": (
                    plus_bmi["beta_diet"]
                    - no_bmi["beta_diet"]
                ),
                "beta_ratio_model3_over_model2": (
                    plus_bmi["beta_diet"] / no_bmi["beta_diet"]
                    if no_bmi["beta_diet"] != 0
                    else np.nan
                ),
                "sign_flip": (
                    np.sign(no_bmi["beta_diet"])
                    != np.sign(plus_bmi["beta_diet"])
                ),
                "condition_model2_same_cohort": no_bmi["condition_number"],
                "condition_model3_plus_bmi": plus_bmi["condition_number"],
            })

    result = pd.DataFrame(rows)

    result["FDR_model2_same_cohort_primary12"] = bh_fdr(
        result["p_model2_same_cohort"]
    )

    result["FDR_model3_plus_bmi_primary12"] = bh_fdr(
        result["p_model3_plus_bmi"]
    )

    return result, pd.DataFrame(corr_rows)


def run_strict_sensitivity(df, model):
    rows = []

    for score_name, score_col in SCORES.items():
        for outcome_name, outcome_col in OUTCOMES.items():

            mask = complete_mask(
                df,
                score_name,
                score_col,
                outcome_col,
                model=model,
                eligibility_col="strict_known_nondiabetes_a10_eligible",
            )

            selected = df.loc[mask].copy()

            fit = fit_ols(
                selected,
                score_name,
                score_col,
                outcome_col,
                add_bmi=(model == 3),
            )

            rows.append({
                "diet_score": score_name,
                "outcome_name": outcome_name,
                "model": model,
                **fit,
            })

    result = pd.DataFrame(rows)
    result["FDR_BH_primary12"] = bh_fdr(result["p_value"])
    return result


def load_primary_results(path, model):
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)

    keep = [
        "diet_score",
        "outcome_name",
        "N",
        "beta_diet",
        "CI95_lower",
        "CI95_upper",
        "p_value",
        "FDR_BH_primary12",
    ]

    missing = [c for c in keep if c not in df.columns]
    if missing:
        return pd.DataFrame()

    out = df[keep].copy()

    out.columns = [
        "diet_score",
        "outcome_name",
        f"primary_model{model}_N",
        f"primary_model{model}_beta",
        f"primary_model{model}_CI95_lower",
        f"primary_model{model}_CI95_upper",
        f"primary_model{model}_p",
        f"primary_model{model}_FDR12",
    ]

    return out


def main():
    if not COHORT_FILE.exists():
        raise FileNotFoundError(COHORT_FILE)

    df = pd.read_csv(COHORT_FILE, low_memory=False)

    required = [
        "primary_cgm_analysis_eligible",
        "strict_known_nondiabetes_a10_eligible",
        "age_years",
        "sex",
        "education_level",
        "smoking_status",
        "sleep_duration_hours_day",
        "physical_activity_met_h_week",
        "vitamin_use",
        "hormone_use",
        "cgm_device_type",
        "alcohol_intake_g_day",
        "bmi",
        *SCORES.values(),
        *OUTCOMES.values(),
    ]

    require_columns(df, required)

    print("=" * 105)
    print("CGM BMI / STRICT-SENSITIVITY QC")
    print("=" * 105)

    print("formal cohort rows:", len(df))
    print(
        "primary eligible:",
        int(as_bool(df["primary_cgm_analysis_eligible"]).sum()),
    )
    print(
        "strict known non-diabetes/A10 eligible:",
        int(as_bool(df["strict_known_nondiabetes_a10_eligible"]).sum()),
    )

    same_cohort, correlations = run_same_cohort_bmi_qc(df)

    same_path = REPORT_DIR / "04_bmi_same_cohort_qc.csv"
    corr_path = REPORT_DIR / "04_bmi_correlation_qc.csv"

    same_cohort.to_csv(same_path, index=False)
    correlations.to_csv(corr_path, index=False)

    print()
    print("=" * 105)
    print("1. BMI SAME-COHORT CHECK")
    print("=" * 105)

    show = same_cohort[
        [
            "diet_score",
            "outcome_name",
            "same_cohort_N",
            "beta_model2_same_cohort",
            "beta_model3_plus_bmi",
            "beta_change_model3_minus_model2",
            "sign_flip",
            "p_model2_same_cohort",
            "p_model3_plus_bmi",
            "FDR_model2_same_cohort_primary12",
            "FDR_model3_plus_bmi_primary12",
        ]
    ]

    print(show.to_string(index=False))

    print()
    print("SIGN FLIPS ONLY")
    print("-" * 105)

    flips = show.loc[show["sign_flip"]]

    if len(flips):
        print(flips.to_string(index=False))
    else:
        print("NONE")

    print()
    print("BMI CORRELATIONS — SPEARMAN")
    print("-" * 105)

    print(
        correlations[
            [
                "score",
                "BMI_correlated_with",
                "N",
                "spearman_rho",
                "spearman_p",
            ]
        ].to_string(index=False)
    )

    strict2 = run_strict_sensitivity(df, model=2)
    strict3 = run_strict_sensitivity(df, model=3)

    strict2_path = MODEL_DIR / "04_strict_known_nondiabetes_a10_model2.csv"
    strict3_path = MODEL_DIR / "04_strict_known_nondiabetes_a10_model3_bmi.csv"

    strict2.to_csv(strict2_path, index=False)
    strict3.to_csv(strict3_path, index=False)

    print()
    print("=" * 105)
    print("2. STRICT KNOWN NON-DIABETES/A10 — MODEL 2")
    print("=" * 105)

    print(
        strict2[
            [
                "diet_score",
                "outcome_name",
                "N",
                "beta_diet",
                "CI95_lower",
                "CI95_upper",
                "p_value",
                "FDR_BH_primary12",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 105)
    print("3. STRICT KNOWN NON-DIABETES/A10 — MODEL 3")
    print("=" * 105)

    print(
        strict3[
            [
                "diet_score",
                "outcome_name",
                "N",
                "beta_diet",
                "CI95_lower",
                "CI95_upper",
                "p_value",
                "FDR_BH_primary12",
            ]
        ].to_string(index=False)
    )

    primary2 = load_primary_results(PRIMARY_MODEL2_FILE, model=2)
    primary3 = load_primary_results(PRIMARY_MODEL3_FILE, model=3)

    strict2_cmp = strict2[
        [
            "diet_score",
            "outcome_name",
            "N",
            "beta_diet",
            "p_value",
            "FDR_BH_primary12",
        ]
    ].rename(
        columns={
            "N": "strict_model2_N",
            "beta_diet": "strict_model2_beta",
            "p_value": "strict_model2_p",
            "FDR_BH_primary12": "strict_model2_FDR12",
        }
    )

    strict3_cmp = strict3[
        [
            "diet_score",
            "outcome_name",
            "N",
            "beta_diet",
            "p_value",
            "FDR_BH_primary12",
        ]
    ].rename(
        columns={
            "N": "strict_model3_N",
            "beta_diet": "strict_model3_beta",
            "p_value": "strict_model3_p",
            "FDR_BH_primary12": "strict_model3_FDR12",
        }
    )

    comparison = strict2_cmp.merge(
        strict3_cmp,
        on=["diet_score", "outcome_name"],
        validate="one_to_one",
    )

    if len(primary2):
        comparison = primary2.merge(
            comparison,
            on=["diet_score", "outcome_name"],
            how="right",
            validate="one_to_one",
        )

    if len(primary3):
        comparison = comparison.merge(
            primary3,
            on=["diet_score", "outcome_name"],
            how="left",
            validate="one_to_one",
        )

    comparison_path = REPORT_DIR / "04_primary_vs_strict_sensitivity.csv"
    comparison.to_csv(comparison_path, index=False)

    print()
    print("=" * 105)
    print("4. KEY ASSOCIATIONS — PRIMARY VS STRICT")
    print("=" * 105)

    key = comparison.loc[
        (
            (comparison["diet_score"] == "AMED")
            & (comparison["outcome_name"] == "mean_glucose")
        )
        |
        (
            (comparison["diet_score"] == "rEDIH")
            & (comparison["outcome_name"] == "glucose_cv")
        )
    ]

    print(key.to_string(index=False))

    print()
    print("=" * 105)
    print("SAVED")
    print("=" * 105)

    for path in [
        same_path,
        corr_path,
        strict2_path,
        strict3_path,
        comparison_path,
    ]:
        print(path)


if __name__ == "__main__":
    main()
