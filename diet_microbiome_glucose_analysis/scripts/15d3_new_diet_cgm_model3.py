#!/usr/bin/env python3
"""Step 15d3 — New diet -> CGM Model3 (+BMI).

BMI sensitivity model for the three frozen new dietary exposures.

Model3-new
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
              + BMI

This reuses canonical cgm_models_current4_common.py encoding.
The only extension to the canonical model is the prespecified
mean_daily_energy_kcal covariate; BMI is added by canonical Model3 logic.

Multiplicity
------------
Primary extension family:
    EAT13 + NOVA4 = 6 tests -> BH-FDR across 6.

Exploratory family:
    Carbohydrate_pct = 3 tests -> BH-FDR across 3.

Additional diagnostics:
- BH-FDR across all 9 tests;
- BH-FDR within each exposure across 3 outcomes.

Bridge gate
-----------
A path is eligible for later bridge analysis only if:
1) Model2 role-specific FDR < 0.05, AND
2) Model3 role-specific FDR < 0.05, AND
3) Model2 and Model3 diet coefficients have the same sign.

This script does NOT run bridge or mediation.

Outputs
-------
diet_microbiome_glucose_analysis/outputs/new_diet_extension/
    models/
        15d3_new_diet_cgm_model3.csv
    reports/
        15d3_new_diet_cgm_all_coefficients_model3.csv
        15d3_new_diet_cgm_model3_covariate_parameters.csv
        15d3_model2_vs_model3_comparison.csv
        15d3_bridge_eligible_diet_cgm_contexts.csv
        15d3_new_diet_cgm_model3_summary.csv
        15d3_new_diet_cgm_model3_summary.txt
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import numpy as np
import pandas as pd


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
MODEL2_FILE = MODEL_DIR / "15d2_new_diet_cgm_model2.csv"

RESULT_OUT = MODEL_DIR / "15d3_new_diet_cgm_model3.csv"
COEF_OUT = REPORT_DIR / "15d3_new_diet_cgm_all_coefficients_model3.csv"
PARAM_OUT = REPORT_DIR / "15d3_new_diet_cgm_model3_covariate_parameters.csv"
COMPARE_OUT = REPORT_DIR / "15d3_model2_vs_model3_comparison.csv"
GATE_OUT = REPORT_DIR / "15d3_bridge_eligible_diet_cgm_contexts.csv"
SUMMARY_CSV = REPORT_DIR / "15d3_new_diet_cgm_model3_summary.csv"
SUMMARY_TXT = REPORT_DIR / "15d3_new_diet_cgm_model3_summary.txt"

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

COMPLETE_FLAG = "new_cgm_model3_covariates_complete"
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


def main() -> int:
    for p, label in [
        (COHORT, "Step15d0 formal new-diet CGM cohort"),
        (COMMON_PATH, "canonical CGM helper"),
        (MODEL2_FILE, "Step15d2 Model2 results"),
    ]:
        require(p, label)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    common = load_module(COMMON_PATH, "step15d3_cgm_common")

    # Canonical Model3 adds BMI. We append only total energy.
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
        "bmi",
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

    print("=== STEP 15d3 NEW DIET -> CGM MODEL3 (+BMI) ===")
    print("MODEL=canonical CGM Model3 + mean_daily_energy_kcal")
    print("BMI_INCLUDED=True")
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
                    3,
                )

                if names.count(f"{ENERGY_COL}_z") != 1:
                    raise RuntimeError(
                        f"{exposure}: expected exactly one {ENERGY_COL}_z term"
                    )
                if names.count("bmi_z") != 1:
                    raise RuntimeError(
                        f"{exposure}: expected exactly one bmi_z term"
                    )
                if "alcohol_intake_g_day_z" in names:
                    raise RuntimeError(
                        f"{exposure}: hPDI-specific alcohol entered Model3"
                    )
                device_terms = [n for n in names if n.startswith("cgm_device_")]
                if not device_terms:
                    raise RuntimeError(
                        f"{exposure}: no CGM-device dummy found in Model3 design"
                    )

                fit = common.fit_ols(X, y)

                rows.append({
                    "diet_score": exposure,
                    "analysis_role": cfg["role"],
                    "exposure": cfg["col"],
                    "outcome_name": outcome_name,
                    "outcome": outcome_col,
                    "model": 3,
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
                coef.insert(0, "model", 3)
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
        raise RuntimeError(f"Expected 9 Model3 models, got {len(results)}")
    results = add_fdr_columns(results)
    results.to_csv(RESULT_OUT, index=False)

    pd.concat(coef_frames, ignore_index=True).to_csv(COEF_OUT, index=False)
    if param_frames:
        pd.concat(param_frames, ignore_index=True).to_csv(PARAM_OUT, index=False)

    model2 = pd.read_csv(MODEL2_FILE, low_memory=False)

    key = ["diet_score", "outcome_name"]
    comp = model2[
        key + [
            "analysis_role",
            "N",
            "beta_diet",
            "p_value",
            "role_specific_FDR05",
        ]
    ].merge(
        results[
            key + [
                "N",
                "beta_diet",
                "p_value",
                "role_specific_FDR05",
            ]
        ],
        on=key,
        suffixes=("_model2", "_model3"),
        validate="one_to_one",
    )
    if len(comp) != 9:
        raise RuntimeError(f"Expected 9 Model2-vs-Model3 comparisons, got {len(comp)}")

    comp["beta_same_sign"] = (
        np.sign(pd.to_numeric(comp["beta_diet_model2"], errors="coerce"))
        == np.sign(pd.to_numeric(comp["beta_diet_model3"], errors="coerce"))
    )
    comp["model2_gate"] = comp["role_specific_FDR05_model2"].astype(bool)
    comp["model3_gate"] = comp["role_specific_FDR05_model3"].astype(bool)
    comp["bridge_gate_eligible"] = (
        comp["model2_gate"]
        & comp["model3_gate"]
        & comp["beta_same_sign"]
    )
    comp["gate_lost_in_model3"] = (
        comp["model2_gate"] & ~comp["model3_gate"]
    )
    comp["gate_gained_in_model3"] = (
        ~comp["model2_gate"] & comp["model3_gate"]
    )
    comp.to_csv(COMPARE_OUT, index=False)

    gate = comp.loc[comp["bridge_gate_eligible"]].copy()
    gate.to_csv(GATE_OUT, index=False)

    summary = {
        "model": 3,
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
        "model2_gates": int(comp["model2_gate"].sum()),
        "model3_gates": int(comp["model3_gate"].sum()),
        "bridge_gate_eligible_contexts": int(comp["bridge_gate_eligible"].sum()),
        "gate_lost_in_model3": int(comp["gate_lost_in_model3"].sum()),
        "gate_gained_in_model3": int(comp["gate_gained_in_model3"].sum()),
        "model2_model3_sign_flips": int((~comp["beta_same_sign"]).sum()),
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
        "role_specific_FDR05",
    ]

    print("\n--- MODEL3 RESULTS ---")
    print(results[display_cols].to_string(index=False))

    print("\n--- MODEL2 vs MODEL3 ---")
    print(
        comp[
            [
                "diet_score",
                "outcome_name",
                "N_model2",
                "N_model3",
                "beta_diet_model2",
                "beta_diet_model3",
                "beta_same_sign",
                "model2_gate",
                "model3_gate",
                "bridge_gate_eligible",
                "gate_lost_in_model3",
                "gate_gained_in_model3",
            ]
        ].to_string(index=False)
    )

    print("\n--- BRIDGE-ELIGIBLE DIET-CGM CONTEXTS ---")
    if len(gate):
        print(
            gate[
                [
                    "diet_score",
                    "analysis_role",
                    "outcome_name",
                    "beta_diet_model2",
                    "beta_diet_model3",
                    "model2_gate",
                    "model3_gate",
                    "bridge_gate_eligible",
                ]
            ].to_string(index=False)
        )
    else:
        print("NONE")

    print("\n--- SUMMARY ---")
    for k, v in summary.items():
        print(f"{k}={v}")

    lines = [
        "=== STEP 15d3 NEW DIET -> CGM MODEL3 SUMMARY ===",
        "MODEL=canonical CGM Model3 + mean_daily_energy_kcal",
        "BMI_INCLUDED=True",
        "ALCOHOL_INCLUDED=False",
        "PRIMARY_FDR_FAMILY=EAT13 + NOVA4 = 6 tests",
        "EXPLORATORY_FDR_FAMILY=Carbohydrate_pct = 3 tests",
        "",
        "--- MODEL3 RESULTS ---",
        results[display_cols].to_string(index=False),
        "",
        "--- MODEL2 vs MODEL3 ---",
        comp.to_string(index=False),
        "",
        "--- BRIDGE-ELIGIBLE CONTEXTS ---",
        gate.to_string(index=False),
        "",
        "--- SUMMARY ---",
        *[f"{k}={v}" for k, v in summary.items()],
        "",
        "BRIDGE RULE:",
        "- Model2 role-specific FDR <0.05",
        "- Model3 role-specific FDR <0.05",
        "- Model2/Model3 diet beta same sign",
        "- Carbohydrate_pct remains exploratory even if bridge-eligible.",
        "",
        f"RESULTS={RESULT_OUT}",
        f"COMPARISON={COMPARE_OUT}",
        f"BRIDGE_GATES={GATE_OUT}",
    ]
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
