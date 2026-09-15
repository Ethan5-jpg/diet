#!/usr/bin/env python3
"""Unchanged numerical helpers extracted from cgm_models_current4_common.py.
See docs/legacy_provenance.json. No legacy input/output entry points included.
"""
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

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
