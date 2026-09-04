#!/usr/bin/env python3
"""Shared implementation for current-four-score diet → CGM Model 0/2/3."""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
ANALYSIS_ROOT = ROOT / "diet_microbiome_glucose_analysis"
DATA_DIR = ANALYSIS_ROOT / "outputs" / "data"
MODEL_DIR = ANALYSIS_ROOT / "outputs" / "models"
REPORT_DIR = ANALYSIS_ROOT / "outputs" / "reports"

COHORT_FILE = DATA_DIR / "00_current4_diet_cgm_cohort.csv"

SCORES = {
    "AHEI": "AHEI_z",
    "AMED": "AMED_z",
    "hPDI": "hPDI_z",
    "rEDIH": "rEDIH_z",
}
PRIMARY_OUTCOMES = {
    "mean_glucose": "cgm_mean_z",
    "glucose_cv": "cgm_cv_z",
    "time_above_140": "cgm_above_140_z",
}

BASE_CONTINUOUS = [
    "age_years",
    "sleep_duration_hours_day",
    "physical_activity_met_h_week",
]
BINARY = ["vitamin_use", "hormone_use"]


def bh_fdr(p_values):
    p = np.asarray(p_values, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise RuntimeError("BH-FDR received invalid p-values.")
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


def to_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    text = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True, "1": True, "1.0": True,
        "false": False, "0": False, "0.0": False,
        "nan": False, "none": False, "": False,
    }
    bad = sorted(set(text.unique()) - set(mapping))
    if bad:
        raise RuntimeError(f"Unexpected boolean values: {bad[:20]}")
    return text.map(mapping).astype(bool)


def require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError("Formal diet-CGM cohort missing columns: " + ", ".join(missing))


def z_numeric(series, name):
    x = pd.to_numeric(series, errors="coerce").to_numpy(float)
    if not np.isfinite(x).all():
        raise RuntimeError(f"{name} has missing/non-finite values in selected cohort.")
    mu = float(x.mean())
    sd = float(x.std(ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        raise RuntimeError(f"{name} has invalid SD={sd}.")
    return (x - mu) / sd, mu, sd


def binary_numeric(series, name):
    x = pd.to_numeric(series, errors="coerce").to_numpy(float)
    if not np.isfinite(x).all():
        raise RuntimeError(f"{name} has missing/non-finite values in selected cohort.")
    values = set(np.unique(x).tolist())
    if not values.issubset({0.0, 1.0}):
        raise RuntimeError(f"{name} expected 0/1, got {sorted(values)[:20]}")
    return x


def normalize_text(series):
    return series.astype(str).str.strip().str.lower()


def add_dummy(columns, names, params, values, name, reference, score, model):
    arr = values.astype(float).to_numpy()
    columns.append(arr)
    names.append(name)
    params.append({
        "score": score,
        "model": model,
        "variable": name,
        "encoding": "dummy",
        "reference": reference,
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=0)),
    })


def build_design(df, score_name, score_col, model):
    score_x = pd.to_numeric(df[score_col], errors="coerce").to_numpy(float)
    if not np.isfinite(score_x).all():
        raise RuntimeError(f"{score_name}: invalid score values after cohort selection.")

    columns = [np.ones(len(df), dtype=float), score_x]
    names = ["intercept", score_col]
    params = []

    if model >= 2:
        continuous = BASE_CONTINUOUS.copy()
        if model >= 3:
            continuous.append("bmi")
        if score_name == "hPDI":
            continuous.append("alcohol_intake_g_day")

        for cov in continuous:
            z, mu, sd = z_numeric(df[cov], cov)
            columns.append(z)
            names.append(cov + "_z")
            params.append({
                "score": score_name,
                "model": model,
                "variable": cov,
                "encoding": "z_continuous",
                "reference": "",
                "mean": mu,
                "sd": sd,
            })

        for cov in BINARY:
            x = binary_numeric(df[cov], cov)
            columns.append(x)
            names.append(cov)
            params.append({
                "score": score_name,
                "model": model,
                "variable": cov,
                "encoding": "binary_0_1",
                "reference": "0",
                "mean": float(x.mean()),
                "sd": float(x.std(ddof=0)),
            })

        sex = normalize_text(df["sex"]).replace({
            "f": "female", "m": "male", "0": "female", "1": "male",
            "0.0": "female", "1.0": "male",
        })
        bad = ~sex.isin(["female", "male"])
        if bad.any():
            raise RuntimeError(f"{score_name}: bad sex values {df.loc[bad,'sex'].value_counts().head().to_dict()}")
        add_dummy(columns, names, params, sex.eq("male"), "sex_male", "female", score_name, model)

        edu = normalize_text(df["education_level"]).str.replace("education_", "", regex=False)
        bad = ~edu.isin(["high", "low"])
        if bad.any():
            raise RuntimeError(f"{score_name}: bad education values {df.loc[bad,'education_level'].value_counts().head().to_dict()}")
        add_dummy(columns, names, params, edu.eq("low"), "education_low", "high", score_name, model)

        smoking = normalize_text(df["smoking_status"])
        bad = ~smoking.isin(["never", "former", "current"])
        if bad.any():
            raise RuntimeError(f"{score_name}: bad smoking values {df.loc[bad,'smoking_status'].value_counts().head().to_dict()}")
        add_dummy(columns, names, params, smoking.eq("former"), "smoking_former", "never", score_name, model)
        add_dummy(columns, names, params, smoking.eq("current"), "smoking_current", "never", score_name, model)

        device = normalize_text(df["cgm_device_type"])
        if device.isin(["nan", "none", ""]).any():
            raise RuntimeError(f"{score_name}: missing cgm_device_type inside Model {model} complete cases.")
        levels = sorted(device.unique().tolist())
        preferred_reference = "abbott_freestyle_libre_pro"
        reference = preferred_reference if preferred_reference in levels else levels[0]
        for level in levels:
            if level == reference:
                continue
            safe = "".join(ch if ch.isalnum() else "_" for ch in level).strip("_")
            add_dummy(
                columns, names, params,
                device.eq(level), f"cgm_device_{safe}", reference, score_name, model
            )

    X = np.column_stack(columns).astype(float)
    if not np.isfinite(X).all():
        raise RuntimeError(f"{score_name}: design matrix has NaN/Inf.")
    rank = int(np.linalg.matrix_rank(X))
    if rank != X.shape[1]:
        raise RuntimeError(
            f"{score_name}: rank-deficient design {rank}/{X.shape[1]} columns; names={names}"
        )
    return X, names, pd.DataFrame(params), float(np.linalg.cond(X))


def fit_ols(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    if n <= k:
        raise RuntimeError(f"OLS has N={n} <= parameters={k}.")
    if not np.isfinite(y).all():
        raise RuntimeError("Outcome contains NaN/Inf after selection.")

    xtx_inv = np.linalg.inv(X.T @ X)
    coef = xtx_inv @ X.T @ y
    fitted = X @ coef
    residual = y - fitted
    df_resid = n - k
    sse = float(residual @ residual)
    mse = sse / df_resid
    se_all = np.sqrt(np.diag(xtx_inv) * mse)
    t_all = coef / se_all
    p_all = 2 * t_dist.sf(np.abs(t_all), df=df_resid)
    tcrit = float(t_dist.ppf(0.975, df=df_resid))

    y_centered = y - y.mean()
    sst = float(y_centered @ y_centered)
    r2 = 1 - sse / sst

    return {
        "coef": coef,
        "se": se_all,
        "t": t_all,
        "p": p_all,
        "ci_lower": coef - tcrit * se_all,
        "ci_upper": coef + tcrit * se_all,
        "r2": r2,
        "df_resid": df_resid,
    }


def load_cohort():
    if not COHORT_FILE.exists():
        raise FileNotFoundError(
            f"Formal cohort not found: {COHORT_FILE}\n"
            "Run 00_build_formal_diet_cgm_cohort.py first."
        )
    df = pd.read_csv(COHORT_FILE, low_memory=False)
    require_columns(df, ["participant_id", "primary_cgm_analysis_eligible"])
    if df["participant_id"].duplicated().any():
        raise RuntimeError("Formal cohort has duplicate participant IDs.")
    return df


def flag_for(score_name, model):
    prefix = "hpdi" if score_name == "hPDI" else "amed"
    return f"{prefix}_cgm_model{model}_covariates_complete"


def run(model):
    if model not in (0, 2, 3):
        raise ValueError("model must be 0, 2, or 3")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_cohort()

    required = list(SCORES.values()) + list(PRIMARY_OUTCOMES.values())
    if model >= 2:
        required += [
            "age_years", "sex", "education_level", "smoking_status",
            "sleep_duration_hours_day", "physical_activity_met_h_week",
            "vitamin_use", "hormone_use", "cgm_device_type",
            "alcohol_intake_g_day",
            "amed_cgm_model2_covariates_complete",
            "hpdi_cgm_model2_covariates_complete",
        ]
    if model >= 3:
        required += [
            "bmi",
            "amed_cgm_model3_covariates_complete",
            "hpdi_cgm_model3_covariates_complete",
        ]
    require_columns(df, sorted(set(required)))

    eligible = to_bool(df["primary_cgm_analysis_eligible"])

    result_rows = []
    coefficient_frames = []
    design_frames = []

    print("=" * 100)
    print(f"CURRENT FOUR-SCORE DIET → CGM — MODEL {model}")
    print("=" * 100)
    print("Primary cohort excludes explicit known diabetes or A10 medication use.")
    if model == 2:
        print("Model 2 adjusts age, sex, education, smoking, sleep, PA, vitamin, hormone, CGM device;")
        print("hPDI additionally adjusts alcohol intake.")
    elif model == 3:
        print("Model 3 = Model 2 + BMI.")

    for score_name, score_col in SCORES.items():
        score_num = pd.to_numeric(df[score_col], errors="coerce")

        if model == 0:
            model_complete = pd.Series(True, index=df.index)
            flag_col = "none"
        else:
            flag_col = flag_for(score_name, model)
            model_complete = to_bool(df[flag_col])

        for outcome_name, outcome_col in PRIMARY_OUTCOMES.items():
            outcome_num = pd.to_numeric(df[outcome_col], errors="coerce")
            valid = (
                eligible
                & model_complete
                & score_num.notna()
                & outcome_num.notna()
                & np.isfinite(score_num.to_numpy(float))
                & np.isfinite(outcome_num.to_numpy(float))
            )
            selected = df.loc[valid].copy()
            y = outcome_num.loc[valid].to_numpy(float)

            X, design_names, params, condition = build_design(
                selected, score_name, score_col, model
            )
            fit = fit_ols(X, y)

            beta = float(fit["coef"][1])
            se = float(fit["se"][1])
            p = float(fit["p"][1])
            lo = float(fit["ci_lower"][1])
            hi = float(fit["ci_upper"][1])

            result_rows.append({
                "diet_score": score_name,
                "exposure": score_col,
                "outcome_name": outcome_name,
                "outcome": outcome_col,
                "model": model,
                "N": len(selected),
                "complete_case_flag": flag_col,
                "n_parameters": X.shape[1],
                "df_resid": fit["df_resid"],
                "design_condition_number": condition,
                "beta_diet": beta,
                "SE_diet": se,
                "CI95_lower": lo,
                "CI95_upper": hi,
                "t_diet": float(fit["t"][1]),
                "p_value": p,
                "R2_full_model": float(fit["r2"]),
            })

            coef = pd.DataFrame({
                "term": design_names,
                "beta": fit["coef"],
                "SE": fit["se"],
                "t": fit["t"],
                "p": fit["p"],
                "CI95_lower": fit["ci_lower"],
                "CI95_upper": fit["ci_upper"],
            })
            coef.insert(0, "outcome", outcome_col)
            coef.insert(0, "diet_score", score_name)
            coef.insert(0, "model", model)
            coefficient_frames.append(coef)

            if not params.empty:
                params = params.copy()
                params["outcome"] = outcome_col
                params["N"] = len(selected)
                params["design_condition_number"] = condition
                design_frames.append(params)

            print(
                f"{score_name:5s} | {outcome_name:14s} | N={len(selected):4d} | "
                f"beta={beta:+.5f} | P={p:.4g} | cond={condition:.2f}"
            )

    results = pd.DataFrame(result_rows)
    if len(results) != 12:
        raise RuntimeError(f"Expected 12 primary models, got {len(results)}")

    # Primary multiplicity family: all 4 dietary scores × 3 primary CGM phenotypes.
    results["FDR_BH_primary12"] = bh_fdr(results["p_value"].to_numpy())
    results["nominal_P05"] = results["p_value"] < 0.05
    results["FDR_primary12_05"] = results["FDR_BH_primary12"] < 0.05

    # Diagnostic only: BH within each score's three CGM outcomes.
    results["FDR_BH_within_score3"] = np.nan
    for score_name in SCORES:
        mask = results["diet_score"].eq(score_name)
        results.loc[mask, "FDR_BH_within_score3"] = bh_fdr(
            results.loc[mask, "p_value"].to_numpy()
        )

    result_path = MODEL_DIR / f"0{1 if model == 0 else model}_current4_diet_cgm_model{model}.csv"
    # Stable explicit names are easier to understand than the numeric fallback above.
    explicit = {
        0: MODEL_DIR / "01_current4_diet_cgm_model0.csv",
        2: MODEL_DIR / "02_current4_diet_cgm_model2.csv",
        3: MODEL_DIR / "03_current4_diet_cgm_model3_bmi.csv",
    }
    result_path = explicit[model]
    results.to_csv(result_path, index=False)

    coef_path = REPORT_DIR / f"0{1 if model == 0 else model}_current4_all_coefficients_model{model}.csv"
    pd.concat(coefficient_frames, ignore_index=True).to_csv(coef_path, index=False)

    if design_frames:
        design_path = REPORT_DIR / f"0{model}_current4_design_parameters_model{model}.csv"
        pd.concat(design_frames, ignore_index=True).to_csv(design_path, index=False)
    else:
        design_path = None

    summary = {
        "model": model,
        "models_fitted": 12,
        "minimum_N": int(results["N"].min()),
        "maximum_N": int(results["N"].max()),
        "nominal_P05_models": int(results["nominal_P05"].sum()),
        "FDR_primary12_05_models": int(results["FDR_primary12_05"].sum()),
        "minimum_p": float(results["p_value"].min()),
        "minimum_FDR_primary12": float(results["FDR_BH_primary12"].min()),
    }
    summary_path = REPORT_DIR / f"0{1 if model == 0 else model}_current4_diet_cgm_model{model}_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(results[[
        "diet_score", "outcome_name", "N", "beta_diet", "CI95_lower", "CI95_upper",
        "p_value", "FDR_BH_primary12", "FDR_BH_within_score3"
    ]].to_string(index=False))
    print()
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("SAVED:")
    print(result_path)
    print(summary_path)
    print(coef_path)
    if design_path is not None:
        print(design_path)


if __name__ == "__main__":
    raise SystemExit("Import this module from a model wrapper.")
