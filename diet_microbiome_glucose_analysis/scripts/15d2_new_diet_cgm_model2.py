#!/usr/bin/env python3
"""Step 15d2 — New diet -> CGM Model2.

Primary adjusted CGM models for the three frozen new dietary exposures.

Model2-new
----------
CGM outcome Z ~ exposure Z
              + age
              + sleep duration
              + physical activity
              + vitamin use
              + hormone use
              + sex
              + education
              + smoking
              + CGM device
              + mean_daily_energy_kcal

This reuses the canonical cgm_models_current4_common.py encoding and adds only
the prespecified total-energy covariate.

No BMI in Model2.
No hPDI-specific alcohol term.

Multiplicity
------------
Primary extension family:
    EAT13 + NOVA4 = 2 exposures x 3 outcomes = 6 tests
    -> BH-FDR across 6 tests.

Exploratory family:
    Carbohydrate_pct = 1 exposure x 3 outcomes = 3 tests
    -> BH-FDR across 3 tests.

Additional diagnostics:
- BH-FDR across all 9 tests;
- BH-FDR within each exposure across 3 outcomes.

Outputs
-------
diet_microbiome_glucose_analysis/outputs/new_diet_extension/
    models/
        15d2_new_diet_cgm_model2.csv
    reports/
        15d2_new_diet_cgm_all_coefficients_model2.csv
        15d2_new_diet_cgm_model2_covariate_parameters.csv
        15d2_model0_vs_model2_comparison.csv
        15d2_new_diet_cgm_model2_summary.csv
        15d2_new_diet_cgm_model2_summary.txt
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
BASE = (
    ROOT / "diet_microbiome_glucose_analysis" / "outputs"
    / "new_diet_extension"
)
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "models"
REPORT_DIR = BASE / "reports"

COHORT = DATA_DIR / "15d0_new_diet_cgm_cohort.csv"
COMMON_PATH = (
    ROOT / "diet_microbiome_glucose_analysis" / "scripts"
    / "cgm_models_current4_common.py"
)
MODEL0_FILE = MODEL_DIR / "15d1_new_diet_cgm_model0.csv"

RESULT_OUT = MODEL_DIR / "15d2_new_diet_cgm_model2.csv"
COEF_OUT = REPORT_DIR / "15d2_new_diet_cgm_all_coefficients_model2.csv"
PARAM_OUT = REPORT_DIR / "15d2_new_diet_cgm_model2_covariate_parameters.csv"
COMPARE_OUT = REPORT_DIR / "15d2_model0_vs_model2_comparison.csv"
SUMMARY_CSV = REPORT_DIR / "15d2_new_diet_cgm_model2_summary.csv"
SUMMARY_TXT = REPORT_DIR / "15d2_new_diet_cgm_model2_summary.txt"

EXPOSURES = {
    "EAT13": {
        "col": "EAT13_z",
        "role": "primary_extension",
    },
    "NOVA4": {
        "col": "NOVA4_z",
        "role": "primary_extension",
    },
    "Carbohydrate_pct": {
        "col": "Carbohydrate_pct_z",
        "role": "exploratory_extension",
    },
}

OUTCOMES = {
    "mean_glucose": "cgm_mean_z",
    "glucose_cv": "cgm_cv_z",
    "time_above_140": "cgm_above_140_z",
}

COMPLETE_FLAG = "new_cgm_model2_covariates_complete"
ENERGY_COL = "mean_daily_energy_kcal"


def require(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all():
        raise RuntimeError("BH-FDR received non-finite p-values")
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


def add_fdr_columns(results: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    out["nominal_P05"] = out["p_value"] < .05

    out["FDR_BH_all9"] = bh_fdr(out["p_value"].to_numpy())
    out["FDR_all9_05"] = out["FDR_BH_all9"] < .05

    out["FDR_BH_primary_extension6"] = np.nan
    primary = out["analysis_role"].eq("primary_extension")
    if int(primary.sum()) != 6:
        raise RuntimeError(f"Expected 6 primary-extension tests, got {int(primary.sum())}")
    out.loc[primary, "FDR_BH_primary_extension6"] = bh_fdr(
        out.loc[primary, "p_value"].to_numpy()
    )
    out["FDR_primary_extension6_05"] = (
        out["FDR_BH_primary_extension6"] < .05
    ).fillna(False)

    out["FDR_BH_exploratory3"] = np.nan
    exploratory = out["analysis_role"].eq("exploratory_extension")
    if int(exploratory.sum()) != 3:
        raise RuntimeError(f"Expected 3 exploratory tests, got {int(exploratory.sum())}")
    out.loc[exploratory, "FDR_BH_exploratory3"] = bh_fdr(
        out.loc[exploratory, "p_value"].to_numpy()
    )
    out["FDR_exploratory3_05"] = (
        out["FDR_BH_exploratory3"] < .05
    ).fillna(False)

    out["FDR_BH_within_exposure3"] = np.nan
    for exposure in EXPOSURES:
        mask = out["diet_score"].eq(exposure)
        if int(mask.sum()) != 3:
            raise RuntimeError(
                f"Expected 3 outcomes for {exposure}, got {int(mask.sum())}"
            )
        out.loc[mask, "FDR_BH_within_exposure3"] = bh_fdr(
            out.loc[mask, "p_value"].to_numpy()
        )

    out["role_specific_FDR05"] = np.where(
        out["analysis_role"].eq("primary_extension"),
        out["FDR_primary_extension6_05"],
        out["FDR_exploratory3_05"],
    ).astype(bool)
    return out


def compare_model0_model2(model0: pd.DataFrame, model2: pd.DataFrame) -> pd.DataFrame:
    key = ["diet_score", "outcome_name"]
    m = model0[
        key + ["N", "beta_diet", "p_value", "role_specific_FDR05"]
    ].merge(
        model2[
            key + ["N", "beta_diet", "p_value", "role_specific_FDR05"]
        ],
        on=key,
        suffixes=("_model0", "_model2"),
        validate="one_to_one",
    )
    if len(m) != 9:
        raise RuntimeError(f"Expected 9 Model0-vs-Model2 rows, got {len(m)}")

    m["beta_same_sign"] = (
        np.sign(pd.to_numeric(m["beta_diet_model0"], errors="coerce"))
        == np.sign(pd.to_numeric(m["beta_diet_model2"], errors="coerce"))
    )
    m["abs_beta_change"] = (
        pd.to_numeric(m["beta_diet_model2"], errors="coerce")
        - pd.to_numeric(m["beta_diet_model0"], errors="coerce")
    ).abs()
    m["relative_abs_beta_change_vs_model0"] = (
        m["abs_beta_change"]
        / pd.to_numeric(m["beta_diet_model0"], errors="coerce").abs().replace(0, np.nan)
    )
    m["gate_retained"] = (
        m["role_specific_FDR05_model0"].astype(bool)
        & m["role_specific_FDR05_model2"].astype(bool)
    )
    m["gate_lost"] = (
        m["role_specific_FDR05_model0"].astype(bool)
        & ~m["role_specific_FDR05_model2"].astype(bool)
    )
    m["gate_gained"] = (
        ~m["role_specific_FDR05_model0"].astype(bool)
        & m["role_specific_FDR05_model2"].astype(bool)
    )
    return m


def main() -> int:
    for p, label in [
        (COHORT, "Step15d0 formal new-diet CGM cohort"),
        (COMMON_PATH, "canonical CGM helper"),
        (MODEL0_FILE, "Step15d1 Model0 results"),
    ]:
        require(p, label)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    common = load_module(COMMON_PATH, "step15d2_cgm_common")

    # Extend canonical Model2 continuous covariates by total energy only.
    original_base = list(common.BASE_CONTINUOUS)
    if ENERGY_COL in original_base:
        raise RuntimeError(
            f"{ENERGY_COL} unexpectedly already present in canonical BASE_CONTINUOUS"
        )
    common.BASE_CONTINUOUS = original_base + [ENERGY_COL]

    df = pd.read_csv(COHORT, low_memory=False)

    required = {
        "participant_id",
        "primary_cgm_analysis_eligible",
        COMPLETE_FLAG,
        ENERGY_COL,
        *[cfg["col"] for cfg in EXPOSURES.values()],
        *OUTCOMES.values(),
        "age_years",
        "sleep_duration_hours_day",
        "physical_activity_met_h_week",
        "vitamin_use",
        "hormone_use",
        "sex",
        "education_level",
        "smoking_status",
        "cgm_device_type",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            "Formal new-diet CGM cohort missing columns: "
            + ", ".join(missing)
        )

    eligible = common.to_bool(df["primary_cgm_analysis_eligible"])
    complete = common.to_bool(df[COMPLETE_FLAG])

    rows = []
    coef_frames = []
    param_frames = []

    print("=== STEP 15d2 NEW DIET -> CGM MODEL2 ===")
    print("MODEL=canonical CGM Model2 + mean_daily_energy_kcal")
    print("BMI_INCLUDED=False")
    print("ALCOHOL_INCLUDED=False")
    print("PRIMARY_FDR_FAMILY=EAT13 + NOVA4 = 6 tests")
    print("EXPLORATORY_FDR_FAMILY=Carbohydrate_pct = 3 tests")
    print("ALL_THREE_COMPLETE_REQUIRED=False")

    try:
        for exposure, cfg in EXPOSURES.items():
            x_num = pd.to_numeric(df[cfg["col"]], errors="coerce")

            for outcome_name, outcome_col in OUTCOMES.items():
                y_num = pd.to_numeric(df[outcome_col], errors="coerce")

                valid = (
                    eligible
                    & complete
                    & x_num.notna()
                    & y_num.notna()
                    & np.isfinite(x_num.to_numpy(float))
                    & np.isfinite(y_num.to_numpy(float))
                )
                selected = df.loc[valid].copy()
                y = y_num.loc[valid].to_numpy(float)

                X, names, params, cond = common.build_design(
                    selected,
                    exposure,
                    cfg["col"],
                    2,
                )

                # Hard model guards.
                if names.count(f"{ENERGY_COL}_z") != 1:
                    raise RuntimeError(
                        f"{exposure}: expected exactly one {ENERGY_COL}_z term"
                    )
                if "bmi_z" in names:
                    raise RuntimeError(f"{exposure}: BMI entered Model2 unexpectedly")
                if "alcohol_intake_g_day_z" in names:
                    raise RuntimeError(
                        f"{exposure}: hPDI-specific alcohol entered Model2"
                    )
                device_terms = [n for n in names if n.startswith("cgm_device_")]
                if not device_terms:
                    raise RuntimeError(
                        f"{exposure}: no CGM-device dummy found in Model2 design"
                    )

                fit = common.fit_ols(X, y)

                rows.append({
                    "diet_score": exposure,
                    "analysis_role": cfg["role"],
                    "exposure": cfg["col"],
                    "outcome_name": outcome_name,
                    "outcome": outcome_col,
                    "model": 2,
                    "N": len(selected),
                    "complete_case_flag": COMPLETE_FLAG,
                    "n_parameters": X.shape[1],
                    "df_resid": fit["df_resid"],
                    "design_condition_number": cond,
                    "beta_diet": float(fit["coef"][1]),
                    "SE_diet": float(fit["se"][1]),
                    "CI95_lower": float(fit["ci_lower"][1]),
                    "CI95_upper": float(fit["ci_upper"][1]),
                    "t_diet": float(fit["t"][1]),
                    "p_value": float(fit["p"][1]),
                    "R2_full_model": float(fit["r2"]),
                })

                coef = pd.DataFrame({
                    "term": names,
                    "beta": fit["coef"],
                    "SE": fit["se"],
                    "t": fit["t"],
                    "p": fit["p"],
                    "CI95_lower": fit["ci_lower"],
                    "CI95_upper": fit["ci_upper"],
                })
                coef.insert(0, "outcome", outcome_col)
                coef.insert(0, "diet_score", exposure)
                coef.insert(0, "model", 2)
                coef_frames.append(coef)

                if not params.empty:
                    ptab = params.copy()
                    ptab.insert(0, "outcome", outcome_col)
                    ptab["N"] = len(selected)
                    ptab["design_condition_number"] = cond
                    param_frames.append(ptab)

                print(
                    f"{exposure:18s} | {outcome_name:14s} | "
                    f"N={len(selected):4d} | "
                    f"beta={fit['coef'][1]:+.5f} | "
                    f"P={fit['p'][1]:.6g}"
                )
    finally:
        common.BASE_CONTINUOUS = original_base

    results = pd.DataFrame(rows)
    if len(results) != 9:
        raise RuntimeError(f"Expected 9 Model2 models, got {len(results)}")

    results = add_fdr_columns(results)
    results.to_csv(RESULT_OUT, index=False)

    pd.concat(coef_frames, ignore_index=True).to_csv(COEF_OUT, index=False)
    if param_frames:
        pd.concat(param_frames, ignore_index=True).to_csv(PARAM_OUT, index=False)

    model0 = pd.read_csv(MODEL0_FILE, low_memory=False)
    comparison = compare_model0_model2(model0, results)
    comparison.to_csv(COMPARE_OUT, index=False)

    summary = {
        "model": 2,
        "models_fitted": len(results),
        "minimum_N": int(results["N"].min()),
        "maximum_N": int(results["N"].max()),
        "nominal_P05_models": int(results["nominal_P05"].sum()),
        "FDR_all9_05_models": int(results["FDR_all9_05"].sum()),
        "primary_extension_FDR6_05_models": int(
            results.loc[
                results["analysis_role"].eq("primary_extension"),
                "FDR_primary_extension6_05",
            ].sum()
        ),
        "exploratory_FDR3_05_models": int(
            results.loc[
                results["analysis_role"].eq("exploratory_extension"),
                "FDR_exploratory3_05",
            ].sum()
        ),
        "model0_role_specific_gates_retained": int(comparison["gate_retained"].sum()),
        "model0_role_specific_gates_lost": int(comparison["gate_lost"].sum()),
        "model0_role_specific_gates_gained": int(comparison["gate_gained"].sum()),
        "beta_signs_same_model0_vs_model2": int(comparison["beta_same_sign"].sum()),
    }
    pd.DataFrame([summary]).to_csv(SUMMARY_CSV, index=False)

    display_cols = [
        "diet_score",
        "analysis_role",
        "outcome_name",
        "N",
        "beta_diet",
        "CI95_lower",
        "CI95_upper",
        "p_value",
        "FDR_BH_primary_extension6",
        "FDR_BH_exploratory3",
        "FDR_BH_all9",
        "FDR_BH_within_exposure3",
        "role_specific_FDR05",
    ]

    print("\n--- MODEL2 RESULTS ---")
    print(results[display_cols].to_string(index=False))

    print("\n--- MODEL0 vs MODEL2 ---")
    print(
        comparison[
            [
                "diet_score",
                "outcome_name",
                "N_model0",
                "N_model2",
                "beta_diet_model0",
                "beta_diet_model2",
                "beta_same_sign",
                "role_specific_FDR05_model0",
                "role_specific_FDR05_model2",
                "gate_retained",
                "gate_lost",
                "gate_gained",
            ]
        ].to_string(index=False)
    )

    print("\n--- SUMMARY ---")
    for k, v in summary.items():
        print(f"{k}={v}")

    lines = [
        "=== STEP 15d2 NEW DIET -> CGM MODEL2 SUMMARY ===",
        "MODEL=canonical CGM Model2 + mean_daily_energy_kcal",
        "BMI_INCLUDED=False",
        "ALCOHOL_INCLUDED=False",
        "PRIMARY_FDR_FAMILY=EAT13 + NOVA4 = 6 tests",
        "EXPLORATORY_FDR_FAMILY=Carbohydrate_pct = 3 tests",
        "",
        "--- MODEL2 RESULTS ---",
        results[display_cols].to_string(index=False),
        "",
        "--- MODEL0 vs MODEL2 ---",
        comparison.to_string(index=False),
        "",
        "--- SUMMARY ---",
        *[f"{k}={v}" for k, v in summary.items()],
        "",
        "INTERPRETATION:",
        "- Model2 is the primary adjusted Diet->CGM model for the new exposures.",
        "- If Model0->Model2 changes materially while N drops substantially, use",
        "  a same-cohort Model0-vs-Model2 QC before attributing changes to adjustment.",
        "- Final bridge eligibility requires Model2 role-specific FDR significance",
        "  plus Model3 BMI robustness.",
        "",
        f"RESULTS={RESULT_OUT}",
        f"COMPARISON={COMPARE_OUT}",
    ]
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
