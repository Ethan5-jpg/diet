#!/usr/bin/env python3
"""Unchanged numerical helpers extracted from 07_microbiome_to_cgm_model3_bmi.py.
See docs/legacy_provenance.json. No legacy input/output entry points included.
"""
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

BASE_CONTINUOUS_COVARS = [
    "age_years",
    "sleep_duration_hours_day",
    "physical_activity_met_h_week",
]


BINARY_COVARS = [
    "vitamin_use",
    "hormone_use",
]


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


def build_covariate_design(frame, include_bmi):
    """
    Covariate-only design including intercept.

    References:
      sex: female
      education: high
      smoking: never
      CGM device: alphabetically first observed category
    """

    cols = [np.ones(len(frame), dtype=float)]
    names = ["intercept"]

    continuous = list(BASE_CONTINUOUS_COVARS)
    if include_bmi:
        continuous.append("bmi")

    for variable in continuous:
        cols.append(zscore(frame[variable], variable))
        names.append(f"{variable}_z")

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
    Q, _ = np.linalg.qr(C, mode="reduced")
    return M - Q @ (Q.T @ M)


def fit_all_species_adjusted(X, y, C):
    """
    Fit all species using Frisch-Waugh-Lovell:

        y ~ species_j + covariates
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
