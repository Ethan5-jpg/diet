#!/usr/bin/env python3
"""Step 15d1 — New diet -> CGM Model0.

Unadjusted models for the three frozen new dietary exposures:

    CGM outcome Z ~ exposure Z

Outcomes
--------
- mean_glucose
- glucose_cv
- time_above_140

Multiplicity
------------
Pre-frozen analysis roles are respected:

Primary extension family:
    EAT13 + NOVA4
    2 exposures x 3 outcomes = 6 tests
    -> BH-FDR across these 6 tests.

Exploratory family:
    Carbohydrate_pct
    1 exposure x 3 outcomes = 3 tests
    -> BH-FDR across these 3 tests.

Additional diagnostics:
- BH-FDR across all 9 new-exposure tests;
- BH-FDR within each exposure's 3 outcomes.

For downstream bridge gating:
- EAT13/NOVA4 use FDR_primary_extension6_05.
- Carbohydrate_pct uses FDR_exploratory3_05 and remains exploratory.

No covariates are included in Model0.
No all-three-complete requirement is imposed.
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

RESULT_OUT = MODEL_DIR / "15d1_new_diet_cgm_model0.csv"
COEF_OUT = REPORT_DIR / "15d1_new_diet_cgm_all_coefficients_model0.csv"
SUMMARY_CSV = REPORT_DIR / "15d1_new_diet_cgm_model0_summary.csv"
SUMMARY_TXT = REPORT_DIR / "15d1_new_diet_cgm_model0_summary.txt"

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


def main() -> int:
    for p, label in [
        (COHORT, "Step15d0 formal new-diet CGM cohort"),
        (COMMON_PATH, "canonical CGM helper"),
    ]:
        require(p, label)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    common = load_module(COMMON_PATH, "step15d1_cgm_common")
    df = pd.read_csv(COHORT, low_memory=False)

    required = {
        "participant_id",
        "primary_cgm_analysis_eligible",
        *[cfg["col"] for cfg in EXPOSURES.values()],
        *OUTCOMES.values(),
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            "Formal new-diet CGM cohort missing columns: "
            + ", ".join(missing)
        )

    eligible = common.to_bool(df["primary_cgm_analysis_eligible"])

    rows = []
    coef_frames = []

    print("=== STEP 15d1 NEW DIET -> CGM MODEL0 ===")
    print("MODEL=CGM outcome Z ~ exposure Z")
    print("COVARIATES=NONE")
    print("ALL_THREE_COMPLETE_REQUIRED=False")
    print("PRIMARY_FDR_FAMILY=EAT13 + NOVA4 = 6 tests")
    print("EXPLORATORY_FDR_FAMILY=Carbohydrate_pct = 3 tests")

    for exposure, cfg in EXPOSURES.items():
        x_num = pd.to_numeric(df[cfg["col"]], errors="coerce")

        for outcome_name, outcome_col in OUTCOMES.items():
            y_num = pd.to_numeric(df[outcome_col], errors="coerce")

            valid = (
                eligible
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
                0,
            )
            fit = common.fit_ols(X, y)

            rows.append({
                "diet_score": exposure,
                "analysis_role": cfg["role"],
                "exposure": cfg["col"],
                "outcome_name": outcome_name,
                "outcome": outcome_col,
                "model": 0,
                "N": len(selected),
                "complete_case_flag": "none",
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
            coef.insert(0, "model", 0)
            coef_frames.append(coef)

            print(
                f"{exposure:18s} | {outcome_name:14s} | "
                f"N={len(selected):4d} | "
                f"beta={fit['coef'][1]:+.5f} | "
                f"P={fit['p'][1]:.6g}"
            )

    results = pd.DataFrame(rows)
    if len(results) != 9:
        raise RuntimeError(f"Expected 9 models, got {len(results)}")

    results["nominal_P05"] = results["p_value"] < .05

    # Diagnostic global family across all 9 new-exposure tests.
    results["FDR_BH_all9"] = bh_fdr(results["p_value"].to_numpy())
    results["FDR_all9_05"] = results["FDR_BH_all9"] < .05

    # Primary extension family: EAT13 + NOVA4 = 6 tests.
    results["FDR_BH_primary_extension6"] = np.nan
    primary_mask = results["analysis_role"].eq("primary_extension")
    if int(primary_mask.sum()) != 6:
        raise RuntimeError(
            f"Expected 6 primary-extension tests, got {int(primary_mask.sum())}"
        )
    results.loc[
        primary_mask, "FDR_BH_primary_extension6"
    ] = bh_fdr(
        results.loc[primary_mask, "p_value"].to_numpy()
    )
    results["FDR_primary_extension6_05"] = (
        results["FDR_BH_primary_extension6"] < .05
    ).fillna(False)

    # Exploratory family: carbohydrate = 3 tests.
    results["FDR_BH_exploratory3"] = np.nan
    exploratory_mask = results["analysis_role"].eq("exploratory_extension")
    if int(exploratory_mask.sum()) != 3:
        raise RuntimeError(
            f"Expected 3 exploratory tests, got {int(exploratory_mask.sum())}"
        )
    results.loc[
        exploratory_mask, "FDR_BH_exploratory3"
    ] = bh_fdr(
        results.loc[exploratory_mask, "p_value"].to_numpy()
    )
    results["FDR_exploratory3_05"] = (
        results["FDR_BH_exploratory3"] < .05
    ).fillna(False)

    # Diagnostic within-exposure 3-outcome BH.
    results["FDR_BH_within_exposure3"] = np.nan
    for exposure in EXPOSURES:
        mask = results["diet_score"].eq(exposure)
        if int(mask.sum()) != 3:
            raise RuntimeError(
                f"Expected 3 outcomes for {exposure}, got {int(mask.sum())}"
            )
        results.loc[
            mask, "FDR_BH_within_exposure3"
        ] = bh_fdr(results.loc[mask, "p_value"].to_numpy())

    # Unified gate flag, preserving role distinction.
    results["role_specific_FDR05"] = np.where(
        results["analysis_role"].eq("primary_extension"),
        results["FDR_primary_extension6_05"],
        results["FDR_exploratory3_05"],
    ).astype(bool)

    results.to_csv(RESULT_OUT, index=False)
    pd.concat(coef_frames, ignore_index=True).to_csv(COEF_OUT, index=False)

    summary = {
        "model": 0,
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

    print("\n--- MODEL0 RESULTS ---")
    print(results[display_cols].to_string(index=False))

    print("\n--- SUMMARY ---")
    for k, v in summary.items():
        print(f"{k}={v}")

    lines = [
        "=== STEP 15d1 NEW DIET -> CGM MODEL0 SUMMARY ===",
        "MODEL=CGM outcome Z ~ exposure Z",
        "COVARIATES=NONE",
        "PRIMARY_FDR_FAMILY=EAT13 + NOVA4 = 6 tests",
        "EXPLORATORY_FDR_FAMILY=Carbohydrate_pct = 3 tests",
        "GLOBAL_9_TEST_FDR=diagnostic",
        "",
        "--- RESULTS ---",
        results[display_cols].to_string(index=False),
        "",
        "--- SUMMARY ---",
        *[f"{k}={v}" for k, v in summary.items()],
        "",
        "GATING RULE FOR LATER BRIDGE:",
        "- EAT13/NOVA4: use role_specific_FDR05 based on primary-extension 6-test BH.",
        "- Carbohydrate_pct: use role_specific_FDR05 based on exploratory 3-test BH.",
        "- Model0 is descriptive/unadjusted; final bridge gate will use Model2 primary",
        "  plus Model3 BMI sensitivity.",
        "",
        f"RESULTS={RESULT_OUT}",
        f"COEFFICIENTS={COEF_OUT}",
    ]
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
