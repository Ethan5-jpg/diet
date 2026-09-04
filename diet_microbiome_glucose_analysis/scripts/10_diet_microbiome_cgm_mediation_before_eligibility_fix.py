#!/usr/bin/env python3
"""
10 — Cross-sectional Diet -> Microbiome -> CGM mediation-style analysis

Purpose
-------
Run formal path-specific mediation-style models after the completed association
pipeline:

    Diet score (X) -> microbial species CLR-Z (M) -> CGM phenotype Z (Y)

This script is designed for the CURRENT FOUR-SCORE HPP analysis and follows the
logic of the source diet-gut-liver paper as closely as the currently assembled
HPP covariates allow:

1. Use BMI-adjusted Diet -> CGM Model 3 to define diet-CGM pairs with a
   statistically supported total association.
2. Within those pairs, use robust Diet -> Microbiome -> CGM paths retained by
   step 09.
3. Fit all mediation models on the SAME complete-case participants.
4. Estimate:
       a = Diet -> Microbiome
       b = Microbiome -> CGM conditional on Diet
       indirect effect = a * b
       direct effect
       total effect
       mediated proportion = indirect / total
5. Obtain 95% percentile bootstrap confidence intervals with 1,000 iterations
   by default.
6. Apply Benjamini-Hochberg FDR:
       - primary: within each diet-score x CGM-outcome mediation family
       - sensitivity: across all mediation paths in this run

Important interpretation
------------------------
This is CROSS-SECTIONAL mediation-style analysis. It cannot establish temporal
or causal mediation.

The original diet-gut-liver paper also adjusted mediation models for antibiotic
and proton pump inhibitor (PPI) use. The current co-variant master assembled in
this project may not yet contain those fields. This script will automatically
include them IF recognized columns are present, and will explicitly report if
they are absent.

Current HPP-specific addition:
    cgm_device_type
is included because the outcome is CGM-derived and the previous adjusted CGM
models already treated device type as an adjustment variable.

Primary candidate gate
----------------------
By default, step 10 only formally mediates robust step-09 paths whose
diet-score -> CGM association remains significant in BMI-adjusted Model 3
(FDR_BH_primary12 < 0.05).

This avoids unstable "mediated proportions" for exposure-outcome pairs with
essentially no total association.

Use --all-robust-paths only as an exploratory sensitivity analysis.

Default inputs
--------------
/home/ec2-user/Desktop/HPP/Data/
    diet_microbiome_glucose_analysis/
        outputs/data/00_current4_diet_cgm_cohort.csv
        outputs/models/03_current4_diet_cgm_model3_bmi.csv
        outputs/reports/09_bridge_candidate_paths_long.csv
    gut_microbiome_deal/data/08_species_clr_zscore.csv
    co-variant/outputs/data/02_covariate_master.csv

Default outputs
---------------
outputs/models/
    10_mediation_paths_model3.csv

outputs/reports/
    10_mediation_summary.csv
    10_mediation_significant.csv
    10_mediation_tier1_significant.csv
    10_mediation_diet_outcome_gate.csv
    10_mediation_covariate_audit.csv
    10_mediation_summary.txt

Run
---
Smoke test:
    python 10_diet_microbiome_cgm_mediation.py --self-test

Quick real-data check:
    python 10_diet_microbiome_cgm_mediation.py --bootstrap 100 --max-paths 3

Primary analysis:
    python 10_diet_microbiome_cgm_mediation.py --bootstrap 1000
"""

from __future__ import annotations

from pathlib import Path
import argparse
import math
import time
import warnings

import numpy as np
import pandas as pd

ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
ANALYSIS_DIR = ROOT / "diet_microbiome_glucose_analysis"

DEFAULT_COHORT = (
    ANALYSIS_DIR / "outputs" / "data" / "00_current4_diet_cgm_cohort.csv"
)
DEFAULT_DIET_CGM_MODEL3 = (
    ANALYSIS_DIR / "outputs" / "models" / "03_current4_diet_cgm_model3_bmi.csv"
)
DEFAULT_BRIDGES = (
    ANALYSIS_DIR / "outputs" / "reports" / "09_bridge_candidate_paths_long.csv"
)
DEFAULT_MICROBIOME = (
    ROOT / "gut_microbiome_deal" / "data" / "08_species_clr_zscore.csv"
)
DEFAULT_COVARIATES = (
    ROOT / "co-variant" / "outputs" / "data" / "02_covariate_master.csv"
)

OUT_MODELS = ANALYSIS_DIR / "outputs" / "models"
OUT_REPORTS = ANALYSIS_DIR / "outputs" / "reports"

SCORES = ["AHEI", "AMED", "hPDI", "rEDIH"]
OUTCOMES = ["mean_glucose", "glucose_cv", "time_above_140"]

# Candidate names are intentionally broad so the script can work across the
# naming variants already used during this project.
SCORE_COLUMN_CANDIDATES = {
    "AHEI": [
        "AHEI_z",
        "ahei_z",
        "ahei_score_z",
        "ahei_energy_adjusted_z",
        "ahei_score_energy_adjusted_z",
        "mahei7_score_energy_adjusted_z",
        "mahei7_z",
    ],
    "AMED": [
        "AMED_z",
        "amed_z",
        "amed_score_z",
        "amed_energy_adjusted_z",
        "amed_score_energy_adjusted_z",
    ],
    "hPDI": [
        "hPDI_z",
        "hpdi_z",
        "hpdi_score_z",
        "hpdi_energy_adjusted_z",
        "hpdi_score_energy_adjusted_z",
    ],
    "rEDIH": [
        "rEDIH_z",
        "redih_z",
        "redih_score_z",
        "redih_energy_adjusted_z",
        "redih_score_energy_adjusted_z",
        "redih_energy_adjusted_score_z",
    ],
}

OUTCOME_COLUMN_CANDIDATES = {
    "mean_glucose": [
        "cgm_mean_z",
        "mean_glucose_z",
        "cgm_mean_glucose_z",
    ],
    "glucose_cv": [
        "cgm_cv_z",
        "glucose_cv_z",
    ],
    "time_above_140": [
        "cgm_above_140_z",
        "time_above_140_z",
        "cgm_time_above_140_z",
    ],
}

# Required current Model-3 covariates.
BASE_COVARIATE_CANDIDATES = {
    "age_years": ["age_years", "age"],
    "sex": ["sex", "gender"],
    "education_level": ["education_level", "education"],
    "smoking_status": ["smoking_status", "smoking"],
    "sleep_duration_hours_day": [
        "sleep_duration_hours_day",
        "sleep_duration",
    ],
    "physical_activity_met_h_week": [
        "physical_activity_met_h_week",
        "physical_activity",
    ],
    "vitamin_use": ["vitamin_use"],
    "hormone_use": ["hormone_use"],
    "bmi": ["bmi", "BMI"],
    "cgm_device_type": ["cgm_device_type", "device_type"],
}

OPTIONAL_MICROBIOME_COVARIATE_CANDIDATES = {
    "antibiotic_use": [
        "antibiotic_use",
        "antibiotics_use",
        "antibiotic_medication_use",
        "recent_antibiotic_use",
    ],
    "ppi_use": [
        "ppi_use",
        "proton_pump_inhibitor_use",
        "proton_pump_inhibitors_use",
        "ppi_medication_use",
    ],
}

ALCOHOL_CANDIDATES = [
    "alcohol_intake_g_day",
    "alcohol_g_day",
    "alcohol_intake",
]

CATEGORICAL_LOGICAL_NAMES = {
    "sex",
    "education_level",
    "smoking_status",
    "cgm_device_type",
}

NUMERIC_LOGICAL_NAMES = {
    "age_years",
    "sleep_duration_hours_day",
    "physical_activity_met_h_week",
    "vitamin_use",
    "hormone_use",
    "bmi",
    "antibiotic_use",
    "ppi_use",
    "alcohol_intake_g_day",
}


def bh_fdr(p_values):
    """
    Benjamini-Hochberg FDR correction implemented with NumPy only.
    No statsmodels dependency is required.
    """
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan, dtype=float)

    ok = np.isfinite(p)
    if not ok.any():
        return out

    p_ok = p[ok]
    m = len(p_ok)

    order = np.argsort(p_ok)
    ranked = p_ok[order]

    adjusted = ranked * m / np.arange(1, m + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    restored = np.empty(m, dtype=float)
    restored[order] = adjusted
    out[ok] = restored
    return out


def normalize_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "1.0": True,
        "0.0": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
        "t": True,
        "f": False,
    }
    x = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )
    return x.fillna(False).astype(bool)


def find_column(columns, candidates, label, required=True):
    cols = list(columns)
    exact = {str(c): c for c in cols}
    lower = {str(c).lower(): c for c in cols}

    for cand in candidates:
        if cand in exact:
            return exact[cand]
        if cand.lower() in lower:
            return lower[cand.lower()]

    if required:
        raise RuntimeError(
            f"Could not identify column for {label}.\n"
            f"Tried: {candidates}\n"
            f"Available columns ({len(cols)}):\n"
            + "\n".join(f"  - {c}" for c in cols)
        )
    return None


def find_participant_id(df, source_name):
    candidates = [
        "participant_id",
        "Participant_ID",
        "participant",
        "subject_id",
        "id",
    ]
    return find_column(
        df.columns,
        candidates,
        f"participant ID in {source_name}",
        required=True,
    )


def clean_participant_id(series):
    # Preserve string IDs; only remove accidental whitespace.
    return series.astype(str).str.strip()


def deduplicate_one_row_per_participant(df, pid_col, source_name):
    d = df.copy()
    d[pid_col] = clean_participant_id(d[pid_col])

    if not d[pid_col].duplicated().any():
        return d

    # Exact duplicate rows are harmless.
    before = len(d)
    d = d.drop_duplicates()
    if not d[pid_col].duplicated().any():
        print(
            f"{source_name}: removed {before - len(d)} exact duplicate row(s)."
        )
        return d

    # If multiple non-identical rows remain, silently picking one would be unsafe.
    counts = d.loc[d[pid_col].duplicated(keep=False), pid_col].value_counts()
    raise RuntimeError(
        f"{source_name} has multiple non-identical rows per participant. "
        "Step 10 will not guess which visit to use.\n"
        f"Top duplicated participant IDs:\n{counts.head(20).to_string()}"
    )


def identify_score_columns(cohort):
    out = {}
    for score in SCORES:
        out[score] = find_column(
            cohort.columns,
            SCORE_COLUMN_CANDIDATES[score],
            f"{score} Z-score in formal cohort",
        )
    return out


def identify_outcome_columns(cohort):
    out = {}
    for outcome in OUTCOMES:
        out[outcome] = find_column(
            cohort.columns,
            OUTCOME_COLUMN_CANDIDATES[outcome],
            f"{outcome} Z-score in formal cohort",
        )
    return out


def identify_covariates(covariates):
    mapping = {}
    audit_rows = []

    for logical, candidates in BASE_COVARIATE_CANDIDATES.items():
        col = find_column(
            covariates.columns,
            candidates,
            logical,
            required=True,
        )
        mapping[logical] = col
        audit_rows.append({
            "logical_name": logical,
            "source_column": col,
            "status": "required_found",
        })

    for logical, candidates in OPTIONAL_MICROBIOME_COVARIATE_CANDIDATES.items():
        col = find_column(
            covariates.columns,
            candidates,
            logical,
            required=False,
        )
        if col is not None:
            mapping[logical] = col
            status = "optional_found_and_included"
        else:
            status = "optional_missing_not_included"
        audit_rows.append({
            "logical_name": logical,
            "source_column": col,
            "status": status,
        })

    alcohol = find_column(
        covariates.columns,
        ALCOHOL_CANDIDATES,
        "alcohol intake",
        required=False,
    )
    if alcohol is not None:
        mapping["alcohol_intake_g_day"] = alcohol
        status = "hPDI_specific_found"
    else:
        status = "hPDI_specific_missing"

    audit_rows.append({
        "logical_name": "alcohol_intake_g_day",
        "source_column": alcohol,
        "status": status,
    })

    return mapping, pd.DataFrame(audit_rows)


def prepare_covariate_frame(covariates, pid_col, mapping):
    keep = [pid_col] + list(dict.fromkeys(mapping.values()))
    d = covariates[keep].copy()
    d = deduplicate_one_row_per_participant(
        d,
        pid_col,
        "covariate master",
    )

    rename = {pid_col: "participant_id"}
    for logical, source in mapping.items():
        rename[source] = logical
    d = d.rename(columns=rename)

    # Convert known numeric variables.
    for logical in NUMERIC_LOGICAL_NAMES:
        if logical in d.columns:
            d[logical] = pd.to_numeric(d[logical], errors="coerce")

    return d


def prepare_cohort_frame(cohort, pid_col, score_map, outcome_map):
    keep = [pid_col] + list(score_map.values()) + list(outcome_map.values())
    keep = list(dict.fromkeys(keep))

    d = cohort[keep].copy()
    d = deduplicate_one_row_per_participant(
        d,
        pid_col,
        "formal Diet-CGM cohort",
    )

    rename = {pid_col: "participant_id"}
    rename.update({v: k for k, v in score_map.items()})
    rename.update({v: k for k, v in outcome_map.items()})

    d = d.rename(columns=rename)

    for col in SCORES + OUTCOMES:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    return d


def prepare_microbiome_frame(microbiome, pid_col):
    d = microbiome.copy()
    d = deduplicate_one_row_per_participant(
        d,
        pid_col,
        "microbiome CLR-Z table",
    )
    if pid_col != "participant_id":
        d = d.rename(columns={pid_col: "participant_id"})
    return d


def choose_model3_fdr_column(model3):
    preferred = [
        "FDR_BH_primary12",
        "fdr_bh_primary12",
        "FDR_primary12",
        "fdr_primary12",
    ]
    col = find_column(
        model3.columns,
        preferred,
        "Model-3 primary FDR",
        required=False,
    )
    if col is not None:
        return col

    # Strongly prefer a column containing both "FDR" and "primary".
    candidates = [
        c for c in model3.columns
        if "fdr" in str(c).lower()
        and "primary" in str(c).lower()
    ]
    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(
        "Could not uniquely identify the Model-3 primary FDR column.\n"
        f"Available columns:\n"
        + "\n".join(f"  - {c}" for c in model3.columns)
    )


def build_diet_outcome_gate(model3, threshold):
    score_col = find_column(
        model3.columns,
        ["diet_score", "score"],
        "diet score label in Model 3",
    )
    outcome_col = find_column(
        model3.columns,
        ["outcome_name", "cgm_outcome", "outcome"],
        "CGM outcome label in Model 3",
    )
    fdr_col = choose_model3_fdr_column(model3)

    d = model3[[score_col, outcome_col, fdr_col]].copy()
    d.columns = ["diet_score", "cgm_outcome", "diet_cgm_model3_FDR"]
    d["diet_cgm_model3_FDR"] = pd.to_numeric(
        d["diet_cgm_model3_FDR"],
        errors="coerce",
    )
    d["diet_cgm_model3_supported"] = (
        d["diet_cgm_model3_FDR"] < threshold
    )

    # one row per score-outcome is expected.
    if d.duplicated(["diet_score", "cgm_outcome"]).any():
        raise RuntimeError(
            "Model-3 diet-CGM result has duplicate diet_score/outcome rows."
        )

    return d


def select_candidate_paths(bridges, gate, use_all_robust_paths):
    required = [
        "diet_score",
        "cgm_outcome",
        "species",
    ]
    missing = [c for c in required if c not in bridges.columns]
    if missing:
        raise RuntimeError(
            "Step-09 bridge table missing columns: "
            + ", ".join(missing)
        )

    d = bridges.copy()

    if "robust_direction_consistent_candidate" in d.columns:
        d = d.loc[
            normalize_bool(d["robust_direction_consistent_candidate"])
        ].copy()

    d = d.merge(
        gate,
        on=["diet_score", "cgm_outcome"],
        how="left",
        validate="many_to_one",
    )

    if d["diet_cgm_model3_supported"].isna().any():
        bad = (
            d.loc[
                d["diet_cgm_model3_supported"].isna(),
                ["diet_score", "cgm_outcome"],
            ]
            .drop_duplicates()
        )
        raise RuntimeError(
            "Some step-09 paths cannot be matched to Model-3 diet-CGM results:\n"
            + bad.to_string(index=False)
        )

    if not use_all_robust_paths:
        d = d.loc[d["diet_cgm_model3_supported"]].copy()

    if d.empty:
        raise RuntimeError(
            "No candidate mediation paths remain after the Model-3 diet-CGM gate."
        )

    return d


def encode_covariates(frame, logical_covariates):
    """
    Return a numeric covariate matrix with fixed one-hot encoding.

    The returned DataFrame contains no intercept; fit_ols adds it.
    """
    parts = []

    for col in logical_covariates:
        if col not in frame.columns:
            raise RuntimeError(f"Covariate column missing after merge: {col}")

        s = frame[col]

        if col in CATEGORICAL_LOGICAL_NAMES:
            # Cast through string only after complete-case filtering.
            cat = s.astype(str)
            dummies = pd.get_dummies(
                cat,
                prefix=col,
                drop_first=True,
                dtype=float,
            )
            parts.append(dummies)
        else:
            x = pd.to_numeric(s, errors="coerce").astype(float)
            parts.append(x.to_frame(col))

    if not parts:
        return pd.DataFrame(index=frame.index)

    return pd.concat(parts, axis=1)


def add_intercept(*arrays):
    n = len(arrays[0])
    cols = [np.ones(n, dtype=float)]
    cols.extend(np.asarray(a, dtype=float).reshape(n, -1) for a in arrays)
    return np.column_stack(cols)


def lstsq_beta(X, y):
    beta, _, rank, singular = np.linalg.lstsq(X, y, rcond=None)
    return beta, int(rank), singular


def mediation_point_estimate(x, m, y, cov):
    """
    Linear mediation with identical covariates in all three regressions.

    Mediator model:
        M ~ X + C

    Outcome model:
        Y ~ X + M + C

    Total model:
        Y ~ X + C
    """
    x = np.asarray(x, dtype=float)
    m = np.asarray(m, dtype=float)
    y = np.asarray(y, dtype=float)
    cov = np.asarray(cov, dtype=float)

    X_med = add_intercept(x, cov)
    X_out = add_intercept(x, m, cov)
    X_total = add_intercept(x, cov)

    beta_med, rank_med, s_med = lstsq_beta(X_med, m)
    beta_out, rank_out, s_out = lstsq_beta(X_out, y)
    beta_total, rank_total, s_total = lstsq_beta(X_total, y)

    a = float(beta_med[1])
    direct = float(beta_out[1])
    b = float(beta_out[2])
    total = float(beta_total[1])
    indirect = a * b

    if abs(total) > 1e-12:
        prop = indirect / total
    else:
        prop = np.nan

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
        "rank_mediator_model": rank_med,
        "rank_outcome_model": rank_out,
        "rank_total_model": rank_total,
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

    for b_idx in range(n_boot):
        idx = rng.integers(0, n, size=n)

        est = mediation_point_estimate(
            x[idx],
            m[idx],
            y[idx],
            cov[idx, :],
        )

        indirect[b_idx] = est["indirect_effect"]
        direct[b_idx] = est["direct_effect"]
        total[b_idx] = est["total_effect"]

        if abs(est["total_effect"]) > 1e-8:
            prop[b_idx] = (
                est["indirect_effect"] / est["total_effect"]
            )

    # Two-sided bootstrap sign p-value with +1 correction.
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
            np.mean(
                np.sign(total)
                != np.sign(np.nanmedian(total))
            )
        ),
    }


def get_covariate_list(score, available_columns):
    covars = [
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
    ]

    # Include the source-paper microbiome confounders if they are available.
    for optional in ["antibiotic_use", "ppi_use"]:
        if optional in available_columns:
            covars.append(optional)

    # Match the source-paper score-specific adjustment.
    if score == "hPDI":
        if "alcohol_intake_g_day" not in available_columns:
            raise RuntimeError(
                "hPDI mediation requires alcohol_intake_g_day, but the current "
                "covariate master does not contain a recognized alcohol column."
            )
        covars.append("alcohol_intake_g_day")

    return covars


def run_one_path(
    path_row,
    cohort_cov,
    microbiome,
    outcome_col,
    score_col,
    bootstrap,
    base_seed,
):
    score = path_row["diet_score"]
    outcome = path_row["cgm_outcome"]
    species = path_row["species"]

    if species not in microbiome.columns:
        raise RuntimeError(
            "Candidate species column is absent from microbiome CLR-Z file:\n"
            f"{species}"
        )

    covars = get_covariate_list(score, cohort_cov.columns)

    left = cohort_cov[
        ["participant_id", score_col, outcome_col] + covars
    ].copy()
    left = left.rename(
        columns={
            score_col: "_X",
            outcome_col: "_Y",
        }
    )

    med = microbiome[["participant_id", species]].copy()
    med = med.rename(columns={species: "_M"})

    d = left.merge(
        med,
        on="participant_id",
        how="inner",
        validate="one_to_one",
    )

    required = ["_X", "_Y", "_M"] + covars

    # Ensure numeric X/Y/M.
    for c in ["_X", "_Y", "_M"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # Missingness is handled exactly once so all three regressions use same N.
    d = d.dropna(subset=required).copy()

    if len(d) < 100:
        raise RuntimeError(
            f"Too few complete cases for {score} / {outcome} / {species}: "
            f"N={len(d)}"
        )

    cov_df = encode_covariates(d, covars)

    x = d["_X"].to_numpy(dtype=float)
    m = d["_M"].to_numpy(dtype=float)
    y = d["_Y"].to_numpy(dtype=float)
    cov = cov_df.to_numpy(dtype=float)

    point = mediation_point_estimate(x, m, y, cov)

    # Deterministic path-specific seed.
    seed_offset = sum(ord(ch) for ch in f"{score}|{outcome}|{species}")
    boot = bootstrap_mediation(
        x,
        m,
        y,
        cov,
        n_boot=bootstrap,
        seed=(base_seed + seed_offset) % (2**32 - 1),
    )

    out = {
        "diet_score": score,
        "cgm_outcome": outcome,
        "species": species,
        "N": len(d),
        "n_covariate_columns_after_dummy_encoding": cov.shape[1],
        "covariates": ";".join(covars),
        **point,
        **boot,
    }

    # Bring forward useful step-09 metadata if available.
    for c in [
        "species_label",
        "candidate_rank",
        "priority_tier",
        "n_robust_paths",
        "n_diet_scores",
        "n_cgm_outcomes",
        "diet_cgm_model3_FDR",
        "primary_beta_diet",
        "primary_beta_cgm",
        "primary_beta_product",
        "bmi_beta_diet",
        "bmi_beta_cgm",
        "bmi_beta_product",
    ]:
        if c in path_row.index:
            out[c] = path_row[c]

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


def self_test():
    print("=" * 100)
    print("SELF TEST — LINEAR MEDIATION BOOTSTRAP")
    print("=" * 100)

    rng = np.random.default_rng(1234)
    n = 1800

    c1 = rng.normal(size=n)
    c2 = rng.normal(size=n)
    x = rng.normal(size=n)

    # True a ≈ 0.5.
    m = 0.5 * x + 0.3 * c1 - 0.2 * c2 + rng.normal(scale=0.8, size=n)

    # True b ≈ 0.4, direct ≈ -0.2, indirect ≈ +0.2.
    y = (
        -0.2 * x
        + 0.4 * m
        + 0.2 * c1
        + 0.1 * c2
        + rng.normal(scale=0.8, size=n)
    )

    cov = np.column_stack([c1, c2])

    point = mediation_point_estimate(x, m, y, cov)
    boot = bootstrap_mediation(
        x,
        m,
        y,
        cov,
        n_boot=200,
        seed=20260904,
    )

    assert abs(point["a_diet_to_microbiome"] - 0.5) < 0.08
    assert abs(point["b_microbiome_to_cgm"] - 0.4) < 0.08
    assert abs(point["indirect_effect"] - 0.2) < 0.07
    assert (
        boot["indirect_CI95_lower"] > 0
    ), "Synthetic positive indirect effect should exclude zero."

    print(
        f"a={point['a_diet_to_microbiome']:.4f}, "
        f"b={point['b_microbiome_to_cgm']:.4f}, "
        f"indirect={point['indirect_effect']:.4f}"
    )
    print(
        "Indirect 95% bootstrap CI: "
        f"[{boot['indirect_CI95_lower']:.4f}, "
        f"{boot['indirect_CI95_upper']:.4f}]"
    )
    print("Linear path recovery: PASS")
    print("Bootstrap CI: PASS")
    print("SELF TEST: PASS")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cohort", default=str(DEFAULT_COHORT))
    parser.add_argument(
        "--diet-cgm-model3",
        default=str(DEFAULT_DIET_CGM_MODEL3),
    )
    parser.add_argument("--bridges", default=str(DEFAULT_BRIDGES))
    parser.add_argument("--microbiome", default=str(DEFAULT_MICROBIOME))
    parser.add_argument("--covariates", default=str(DEFAULT_COVARIATES))

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Bootstrap iterations per mediation path. Default: 1000.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260904,
    )
    parser.add_argument(
        "--diet-cgm-fdr-threshold",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--all-robust-paths",
        action="store_true",
        help=(
            "Exploratory sensitivity: ignore the significant Model-3 "
            "diet-CGM total-effect gate."
        ),
    )
    parser.add_argument(
        "--max-paths",
        type=int,
        default=None,
        help="Debug/smoke-test only: run first N selected paths.",
    )
    parser.add_argument("--self-test", action="store_true")

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if args.bootstrap < 20:
        raise RuntimeError(
            "Use at least 20 bootstrap iterations. "
            "Use 100 for a quick real-data check and 1000 for the primary run."
        )

    input_paths = {
        "cohort": Path(args.cohort),
        "diet_cgm_model3": Path(args.diet_cgm_model3),
        "bridges": Path(args.bridges),
        "microbiome": Path(args.microbiome),
        "covariates": Path(args.covariates),
    }

    for label, path in input_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} not found:\n{path}")

    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("10 — DIET -> MICROBIOME -> CGM CROSS-SECTIONAL MEDIATION-STYLE ANALYSIS")
    print("=" * 100)
    print(f"Bootstrap iterations per path: {args.bootstrap}")
    print(f"Random seed base: {args.seed}")
    print(
        "Candidate gate: "
        + (
            "ALL robust step-09 paths (exploratory sensitivity)"
            if args.all_robust_paths
            else (
                "robust step-09 paths AND significant BMI-adjusted "
                f"Diet->CGM Model-3 FDR < {args.diet_cgm_fdr_threshold}"
            )
        )
    )

    cohort_raw = pd.read_csv(input_paths["cohort"], low_memory=False)
    model3 = pd.read_csv(input_paths["diet_cgm_model3"], low_memory=False)
    bridges = pd.read_csv(input_paths["bridges"], low_memory=False)
    microbiome_raw = pd.read_csv(input_paths["microbiome"], low_memory=False)
    covariates_raw = pd.read_csv(input_paths["covariates"], low_memory=False)

    cohort_pid = find_participant_id(cohort_raw, "formal Diet-CGM cohort")
    micro_pid = find_participant_id(microbiome_raw, "microbiome CLR-Z table")
    cov_pid = find_participant_id(covariates_raw, "covariate master")

    score_map = identify_score_columns(cohort_raw)
    outcome_map = identify_outcome_columns(cohort_raw)
    cov_map, cov_audit = identify_covariates(covariates_raw)

    cohort = prepare_cohort_frame(
        cohort_raw,
        cohort_pid,
        score_map,
        outcome_map,
    )
    microbiome = prepare_microbiome_frame(
        microbiome_raw,
        micro_pid,
    )
    covariates = prepare_covariate_frame(
        covariates_raw,
        cov_pid,
        cov_map,
    )

    # One participant-level table for diet/outcome/covariates.
    cohort_cov = cohort.merge(
        covariates,
        on="participant_id",
        how="left",
        validate="one_to_one",
    )

    gate = build_diet_outcome_gate(
        model3,
        threshold=args.diet_cgm_fdr_threshold,
    )

    selected = select_candidate_paths(
        bridges,
        gate,
        use_all_robust_paths=args.all_robust_paths,
    )

    # Deterministic order: Tier1 first, then rank, then score/outcome/species.
    if "priority_tier" in selected.columns:
        tier_rank = {
            "Tier1_cross_context_recurrent": 1,
            "Tier2_repeated": 2,
            "Tier3_single_robust_path": 3,
        }
        selected["_tier_rank"] = (
            selected["priority_tier"]
            .map(tier_rank)
            .fillna(99)
        )
    else:
        selected["_tier_rank"] = 99

    sort_cols = ["_tier_rank"]
    if "candidate_rank" in selected.columns:
        sort_cols.append("candidate_rank")
    sort_cols += ["diet_score", "cgm_outcome", "species"]

    selected = (
        selected.sort_values(sort_cols)
        .drop(columns=["_tier_rank"])
        .reset_index(drop=True)
    )

    if args.max_paths is not None:
        selected = selected.head(args.max_paths).copy()

    print()
    print("=" * 100)
    print("COLUMN / COVARIATE AUDIT")
    print("=" * 100)
    print("Diet score columns:")
    for k, v in score_map.items():
        print(f"  {k}: {v}")
    print("CGM outcome columns:")
    for k, v in outcome_map.items():
        print(f"  {k}: {v}")

    print()
    print(cov_audit.to_string(index=False))

    missing_optional = cov_audit.loc[
        cov_audit["status"].eq("optional_missing_not_included"),
        "logical_name",
    ].tolist()

    if missing_optional:
        print()
        print(
            "SOURCE-PAPER OPTIONAL MICROBIOME COVARIATES CURRENTLY ABSENT: "
            + ", ".join(missing_optional)
        )
        print(
            "The analysis will continue, but this is not yet a fully covariate-identical "
            "replication of the source paper's mediation adjustment."
        )

    print()
    print("=" * 100)
    print("BMI-ADJUSTED DIET -> CGM GATE")
    print("=" * 100)
    print(
        gate.sort_values(
            ["diet_score", "cgm_outcome"]
        ).to_string(index=False)
    )

    print()
    print("=" * 100)
    print("SELECTED MEDIATION PATHS")
    print("=" * 100)
    print(f"Paths selected: {len(selected)}")
    print(f"Unique species: {selected['species'].nunique()}")
    print(
        selected.groupby(
            ["diet_score", "cgm_outcome"],
            dropna=False,
        )
        .size()
        .rename("n_paths")
        .reset_index()
        .to_string(index=False)
    )

    results = []
    total_paths = len(selected)
    start_all = time.time()

    for i, (_, row) in enumerate(selected.iterrows(), start=1):
        score = row["diet_score"]
        outcome = row["cgm_outcome"]
        species_label = (
            row["species_label"]
            if "species_label" in row.index
            else row["species"]
        )

        print()
        print("-" * 100)
        print(
            f"[{i}/{total_paths}] "
            f"{score} -> {species_label} -> {outcome}"
        )

        t0 = time.time()

        res = run_one_path(
            path_row=row,
            cohort_cov=cohort_cov,
            microbiome=microbiome,
            outcome_col=outcome,
            score_col=score,
            bootstrap=args.bootstrap,
            base_seed=args.seed,
        )

        res["elapsed_seconds"] = time.time() - t0
        results.append(res)

        print(
            f"N={res['N']} | "
            f"a={res['a_diet_to_microbiome']:+.5f} | "
            f"b={res['b_microbiome_to_cgm']:+.5f} | "
            f"indirect={res['indirect_effect']:+.6f} "
            f"[{res['indirect_CI95_lower']:+.6f}, "
            f"{res['indirect_CI95_upper']:+.6f}] | "
            f"p_boot={res['bootstrap_p_indirect']:.4g} | "
            f"prop={res['mediated_proportion']:+.3f}"
        )

    results = pd.DataFrame(results)

    # Primary multiplicity block: candidate mediators within each score-outcome.
    results["FDR_BH_within_score_outcome"] = np.nan
    for _, idx in results.groupby(
        ["diet_score", "cgm_outcome"]
    ).groups.items():
        idx = list(idx)
        results.loc[idx, "FDR_BH_within_score_outcome"] = bh_fdr(
            results.loc[idx, "bootstrap_p_indirect"]
        )

    # Global sensitivity family.
    results["FDR_BH_global_all_mediation"] = bh_fdr(
        results["bootstrap_p_indirect"]
    )

    results["mediation_FDR05_primary"] = (
        results["FDR_BH_within_score_outcome"] < 0.05
    )

    # A conservative interpretation flag:
    # significant indirect + same sign as total + CI excludes zero.
    results["candidate_mediator_consistent"] = (
        results["mediation_FDR05_primary"]
        & results["indirect_same_sign_as_total"]
        & results["indirect_CI_excludes_zero"]
    )

    # Sort primary results.
    results = results.sort_values(
        [
            "mediation_FDR05_primary",
            "candidate_mediator_consistent",
            "FDR_BH_within_score_outcome",
            "bootstrap_p_indirect",
        ],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    model_out = OUT_MODELS / "10_mediation_paths_model3.csv"
    summary_out = OUT_REPORTS / "10_mediation_summary.csv"
    sig_out = OUT_REPORTS / "10_mediation_significant.csv"
    tier1_sig_out = OUT_REPORTS / "10_mediation_tier1_significant.csv"
    gate_out = OUT_REPORTS / "10_mediation_diet_outcome_gate.csv"
    cov_out = OUT_REPORTS / "10_mediation_covariate_audit.csv"
    txt_out = OUT_REPORTS / "10_mediation_summary.txt"

    results.to_csv(model_out, index=False)
    gate.to_csv(gate_out, index=False)
    cov_audit.to_csv(cov_out, index=False)

    sig = results.loc[
        results["mediation_FDR05_primary"]
    ].copy()
    sig.to_csv(sig_out, index=False)

    if "priority_tier" in results.columns:
        tier1_sig = results.loc[
            results["mediation_FDR05_primary"]
            & results["priority_tier"].eq(
                "Tier1_cross_context_recurrent"
            )
        ].copy()
    else:
        tier1_sig = results.iloc[0:0].copy()

    tier1_sig.to_csv(tier1_sig_out, index=False)

    summary = (
        results.groupby(
            ["diet_score", "cgm_outcome"],
            dropna=False,
        )
        .agg(
            paths_tested=("species", "size"),
            unique_species=("species", "nunique"),
            median_N=("N", "median"),
            significant_mediation_FDR05=(
                "mediation_FDR05_primary",
                "sum",
            ),
            direction_consistent_candidates=(
                "candidate_mediator_consistent",
                "sum",
            ),
            min_bootstrap_p=(
                "bootstrap_p_indirect",
                "min",
            ),
            min_FDR_primary=(
                "FDR_BH_within_score_outcome",
                "min",
            ),
        )
        .reset_index()
    )
    summary.to_csv(summary_out, index=False)

    elapsed_all = time.time() - start_all

    lines = []
    lines.append("=" * 100)
    lines.append("10 — DIET -> MICROBIOME -> CGM MEDIATION-STYLE SUMMARY")
    lines.append("=" * 100)
    lines.append(f"Bootstrap iterations per path: {args.bootstrap}")
    lines.append(f"Paths tested: {len(results)}")
    lines.append(f"Unique species tested: {results['species'].nunique()}")
    lines.append(
        f"Primary FDR<0.05 mediation paths: "
        f"{int(results['mediation_FDR05_primary'].sum())}"
    )
    lines.append(
        f"Direction-consistent candidate mediators: "
        f"{int(results['candidate_mediator_consistent'].sum())}"
    )
    lines.append(f"Elapsed seconds: {elapsed_all:.1f}")
    lines.append("")
    lines.append("MODEL")
    lines.append("-" * 100)
    lines.append("Mediator: M ~ Diet + covariates")
    lines.append("Outcome:  CGM ~ Diet + M + covariates")
    lines.append("Total:    CGM ~ Diet + covariates")
    lines.append("Indirect = a*b")
    lines.append("Mediated proportion = indirect/total")
    lines.append("95% CI = percentile nonparametric bootstrap")
    lines.append("")
    lines.append("MULTIPLE TESTING")
    lines.append("-" * 100)
    lines.append(
        "Primary BH-FDR is calculated within each diet-score x CGM-outcome "
        "candidate-mediator family."
    )
    lines.append(
        "A global BH-FDR across all mediation paths is also saved as a sensitivity metric."
    )
    lines.append("")
    lines.append("CURRENT COVARIATE LIMITATION")
    lines.append("-" * 100)
    if missing_optional:
        lines.append(
            "The following source-paper microbiome covariates were not available "
            "in the current co-variant master and were NOT adjusted here: "
            + ", ".join(missing_optional)
        )
    else:
        lines.append(
            "Recognized antibiotic and PPI covariates were available and included."
        )
    lines.append(
        "CGM device type is included as an HPP CGM-specific adjustment variable."
    )
    lines.append("")
    lines.append("SUMMARY BY DIET SCORE / CGM OUTCOME")
    lines.append("-" * 100)
    lines.append(summary.to_string(index=False))
    lines.append("")
    lines.append("SIGNIFICANT MEDIATION PATHS")
    lines.append("-" * 100)

    display_cols = [
        "diet_score",
        "cgm_outcome",
        "species_label",
        "priority_tier",
        "N",
        "a_diet_to_microbiome",
        "b_microbiome_to_cgm",
        "indirect_effect",
        "indirect_CI95_lower",
        "indirect_CI95_upper",
        "mediated_proportion",
        "bootstrap_p_indirect",
        "FDR_BH_within_score_outcome",
        "candidate_mediator_consistent",
    ]
    display_cols = [c for c in display_cols if c in sig.columns]

    if len(sig):
        lines.append(sig[display_cols].to_string(index=False))
    else:
        lines.append("None at primary FDR < 0.05.")

    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 100)
    lines.append(
        "These are cross-sectional statistical mediation patterns, not proof of "
        "causal or temporal mediation."
    )
    lines.append(
        "Mediated proportion should be interpreted cautiously when the indirect "
        "effect opposes the total effect or when bootstrap total effects approach zero."
    )

    txt_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print("=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print(summary.to_string(index=False))
    print()
    print(
        f"Primary FDR<0.05 mediation paths: "
        f"{int(results['mediation_FDR05_primary'].sum())}"
    )
    print(
        f"Direction-consistent candidate mediators: "
        f"{int(results['candidate_mediator_consistent'].sum())}"
    )

    if "priority_tier" in results.columns:
        print(
            "Tier-1 significant mediation paths: "
            f"{len(tier1_sig)}"
        )

    print()
    print("=" * 100)
    print("TOP SIGNIFICANT MEDIATION PATHS")
    print("=" * 100)
    if len(sig):
        print(
            sig[display_cols]
            .head(30)
            .to_string(index=False)
        )
    else:
        print("None at primary FDR < 0.05.")

    print()
    print("=" * 100)
    print("SAVED")
    print("=" * 100)
    for p in [
        model_out,
        summary_out,
        sig_out,
        tier1_sig_out,
        gate_out,
        cov_out,
        txt_out,
    ]:
        print(p)


if __name__ == "__main__":
    main()
