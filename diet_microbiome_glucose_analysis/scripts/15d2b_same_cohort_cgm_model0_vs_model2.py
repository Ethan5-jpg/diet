#!/usr/bin/env python3
"""Step 15d2b — exact same-cohort Model0 vs Model2 QC for new Diet->CGM.

Purpose
-------
Step15d2 Model2 uses substantially smaller complete-case samples than Model0.
This QC decomposes the apparent Model0->Model2 changes into:

A) sample-selection effect:
   full-sample Model0 -> same-cohort Model0

B) covariate-adjustment effect:
   same-cohort Model0 -> Model2

For each exposure × outcome, same-cohort Model0 is refit on EXACTLY the rows
used by Step15d2 Model2.

Multiplicity is recomputed for same-cohort Model0 using the same frozen families:
- primary extension: EAT13 + NOVA4 = 6 tests;
- exploratory: Carbohydrate_pct = 3 tests;
- all 9 tests additionally reported as diagnostic.

No canonical file is modified.
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
FULL_MODEL0 = MODEL_DIR / "15d1_new_diet_cgm_model0.csv"
MODEL2 = MODEL_DIR / "15d2_new_diet_cgm_model2.csv"

SAME0_OUT = MODEL_DIR / "15d2b_samecohort_new_diet_cgm_model0.csv"
COMPARE_OUT = REPORT_DIR / "15d2b_samecohort_model0_vs_model2.csv"
SUMMARY_TXT = REPORT_DIR / "15d2b_samecohort_model0_vs_model2_summary.txt"

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


def add_fdr(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["FDR_BH_all9"] = bh_fdr(out["p_value"].to_numpy())
    out["FDR_all9_05"] = out["FDR_BH_all9"] < .05

    out["FDR_BH_primary_extension6"] = np.nan
    p_mask = out["analysis_role"].eq("primary_extension")
    if int(p_mask.sum()) != 6:
        raise RuntimeError("Expected 6 primary-extension rows")
    out.loc[p_mask, "FDR_BH_primary_extension6"] = bh_fdr(
        out.loc[p_mask, "p_value"].to_numpy()
    )
    out["FDR_primary_extension6_05"] = (
        out["FDR_BH_primary_extension6"] < .05
    ).fillna(False)

    out["FDR_BH_exploratory3"] = np.nan
    e_mask = out["analysis_role"].eq("exploratory_extension")
    if int(e_mask.sum()) != 3:
        raise RuntimeError("Expected 3 exploratory rows")
    out.loc[e_mask, "FDR_BH_exploratory3"] = bh_fdr(
        out.loc[e_mask, "p_value"].to_numpy()
    )
    out["FDR_exploratory3_05"] = (
        out["FDR_BH_exploratory3"] < .05
    ).fillna(False)

    out["role_specific_FDR05"] = np.where(
        out["analysis_role"].eq("primary_extension"),
        out["FDR_primary_extension6_05"],
        out["FDR_exploratory3_05"],
    ).astype(bool)

    return out


def main() -> int:
    for p, label in [
        (COHORT, "formal new-diet CGM cohort"),
        (COMMON_PATH, "canonical CGM helper"),
        (FULL_MODEL0, "full Model0 results"),
        (MODEL2, "Model2 results"),
    ]:
        require(p, label)

    common = load_module(COMMON_PATH, "step15d2b_cgm_common")
    df = pd.read_csv(COHORT, low_memory=False)
    full0 = pd.read_csv(FULL_MODEL0, low_memory=False)
    model2 = pd.read_csv(MODEL2, low_memory=False)

    needed = {
        "primary_cgm_analysis_eligible",
        COMPLETE_FLAG,
        *[cfg["col"] for cfg in EXPOSURES.values()],
        *OUTCOMES.values(),
    }
    missing = sorted(needed - set(df.columns))
    if missing:
        raise RuntimeError("Cohort missing: " + ", ".join(missing))

    eligible = common.to_bool(df["primary_cgm_analysis_eligible"])
    complete = common.to_bool(df[COMPLETE_FLAG])

    rows = []

    print("=== STEP 15d2b SAME-COHORT MODEL0 vs MODEL2 QC ===")
    print("PURPOSE=separate sample-selection from covariate-adjustment")
    print("PRIMARY_FDR_FAMILY=6 tests")
    print("EXPLORATORY_FDR_FAMILY=3 tests")

    for exposure, cfg in EXPOSURES.items():
        x = pd.to_numeric(df[cfg["col"]], errors="coerce")

        for outcome_name, outcome_col in OUTCOMES.items():
            y = pd.to_numeric(df[outcome_col], errors="coerce")

            valid = (
                eligible
                & complete
                & x.notna()
                & y.notna()
                & np.isfinite(x.to_numpy(float))
                & np.isfinite(y.to_numpy(float))
            )
            selected = df.loc[valid].copy()
            y_use = y.loc[valid].to_numpy(float)

            X, names, params, cond = common.build_design(
                selected,
                exposure,
                cfg["col"],
                0,
            )
            fit = common.fit_ols(X, y_use)

            rows.append({
                "diet_score": exposure,
                "analysis_role": cfg["role"],
                "outcome_name": outcome_name,
                "outcome": outcome_col,
                "N": len(selected),
                "beta_diet": float(fit["coef"][1]),
                "SE_diet": float(fit["se"][1]),
                "CI95_lower": float(fit["ci_lower"][1]),
                "CI95_upper": float(fit["ci_upper"][1]),
                "p_value": float(fit["p"][1]),
                "R2_full_model": float(fit["r2"]),
            })

    same0 = add_fdr(pd.DataFrame(rows))
    same0.to_csv(SAME0_OUT, index=False)

    key = ["diet_score", "outcome_name"]

    a = full0[
        key + ["N", "beta_diet", "p_value", "role_specific_FDR05"]
    ].merge(
        same0[
            key + ["N", "beta_diet", "p_value", "role_specific_FDR05"]
        ],
        on=key,
        suffixes=("_full0", "_same0"),
        validate="one_to_one",
    )

    b = same0[
        key + ["N", "beta_diet", "p_value", "role_specific_FDR05"]
    ].merge(
        model2[
            key + ["N", "beta_diet", "p_value", "role_specific_FDR05"]
        ],
        on=key,
        suffixes=("_same0", "_model2"),
        validate="one_to_one",
    )

    comp = a.merge(
        b,
        on=key,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_b"),
    )

    # Remove duplicated same0 fields introduced by second merge.
    for col in [
        "N_same0_b",
        "beta_diet_same0_b",
        "p_value_same0_b",
        "role_specific_FDR05_same0_b",
    ]:
        if col in comp.columns:
            base = col.replace("_b", "")
            if base in comp.columns:
                # Hard equality check.
                left = comp[base]
                right = comp[col]
                if pd.api.types.is_numeric_dtype(left) or pd.api.types.is_numeric_dtype(right):
                    if not np.allclose(
                        pd.to_numeric(left, errors="coerce"),
                        pd.to_numeric(right, errors="coerce"),
                        equal_nan=True,
                    ):
                        raise RuntimeError(f"Same0 merge mismatch for {base}")
                else:
                    if not left.equals(right):
                        raise RuntimeError(f"Same0 merge mismatch for {base}")
            comp = comp.drop(columns=[col])

    comp["full0_to_same0_same_sign"] = (
        np.sign(pd.to_numeric(comp["beta_diet_full0"]))
        == np.sign(pd.to_numeric(comp["beta_diet_same0"]))
    )
    comp["same0_to_model2_same_sign"] = (
        np.sign(pd.to_numeric(comp["beta_diet_same0"]))
        == np.sign(pd.to_numeric(comp["beta_diet_model2"]))
    )

    comp["sample_selection_gate_lost"] = (
        comp["role_specific_FDR05_full0"].astype(bool)
        & ~comp["role_specific_FDR05_same0"].astype(bool)
    )
    comp["sample_selection_gate_gained"] = (
        ~comp["role_specific_FDR05_full0"].astype(bool)
        & comp["role_specific_FDR05_same0"].astype(bool)
    )
    comp["adjustment_gate_lost"] = (
        comp["role_specific_FDR05_same0"].astype(bool)
        & ~comp["role_specific_FDR05_model2"].astype(bool)
    )
    comp["adjustment_gate_gained"] = (
        ~comp["role_specific_FDR05_same0"].astype(bool)
        & comp["role_specific_FDR05_model2"].astype(bool)
    )

    comp.to_csv(COMPARE_OUT, index=False)

    print("\n--- SAME-COHORT DECOMPOSITION ---")
    show = [
        "diet_score",
        "outcome_name",
        "N_full0",
        "N_same0",
        "beta_diet_full0",
        "beta_diet_same0",
        "beta_diet_model2",
        "role_specific_FDR05_full0",
        "role_specific_FDR05_same0",
        "role_specific_FDR05_model2",
        "sample_selection_gate_lost",
        "sample_selection_gate_gained",
        "adjustment_gate_lost",
        "adjustment_gate_gained",
        "full0_to_same0_same_sign",
        "same0_to_model2_same_sign",
    ]
    print(comp[show].to_string(index=False))

    summary = {
        "sample_selection_gate_lost": int(comp["sample_selection_gate_lost"].sum()),
        "sample_selection_gate_gained": int(comp["sample_selection_gate_gained"].sum()),
        "adjustment_gate_lost": int(comp["adjustment_gate_lost"].sum()),
        "adjustment_gate_gained": int(comp["adjustment_gate_gained"].sum()),
        "full0_to_same0_sign_flips": int((~comp["full0_to_same0_same_sign"]).sum()),
        "same0_to_model2_sign_flips": int((~comp["same0_to_model2_same_sign"]).sum()),
    }

    print("\n--- SUMMARY ---")
    for k, v in summary.items():
        print(f"{k}={v}")

    lines = [
        "=== STEP 15d2b SAME-COHORT MODEL0 vs MODEL2 QC ===",
        "",
        comp[show].to_string(index=False),
        "",
        "--- SUMMARY ---",
        *[f"{k}={v}" for k, v in summary.items()],
        "",
        "INTERPRETATION:",
        "- full0 -> same0 isolates sample-selection effect.",
        "- same0 -> Model2 isolates covariate-adjustment effect on the exact same rows.",
        "- Role-specific FDR families are identical to Steps15d1/15d2.",
        "",
        f"SAME0={SAME0_OUT}",
        f"COMPARISON={COMPARE_OUT}",
    ]
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
