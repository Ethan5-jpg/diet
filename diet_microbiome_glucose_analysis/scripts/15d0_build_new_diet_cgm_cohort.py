#!/usr/bin/env python3
"""Step 15d0 — build formal new-diet × CGM analysis cohort.

This starts the Diet→CGM branch for the three frozen new exposures after the
Diet→Microbiome branch has completed Model0/Model2/Model3.

Frozen exposures
----------------
1) EAT13_z
   modified EAT-Lancet-13, primary extension.

2) NOVA4_z
   NOVA4 observed energy share using total dietary energy denominator,
   restricted to participants with >=80% NOVA energy-mapping coverage,
   primary extension.

3) Carbohydrate_pct_z
   carbohydrate energy percentage, exploratory macronutrient exposure.

Cohort logic
------------
This mirrors the canonical 00_build_formal_diet_cgm_cohort.py:
- one row per participant;
- merge score/exposure table with final selected baseline CGM phenotype table;
- merge canonical covariates;
- primary cohort excludes ONLY explicit known diabetes/A10 positives;
- unknown diabetes/A10 status is retained and flagged;
- each exposure uses its own maximum valid sample;
- no all-three-exposures-complete requirement.

New-exposure Model2/3 rule
--------------------------
Model2-new = canonical AMED-like CGM Model2 covariates
             + mean_daily_energy_kcal

Model3-new = Model2-new + BMI

No hPDI-specific alcohol adjustment is used for these new exposures.

This script FITS NO MODELS. It only creates the formal cohort and reports
exposure-specific overlap / complete-case counts.

Outputs
-------
diet_microbiome_glucose_analysis/outputs/new_diet_extension/
    data/
        15d0_new_diet_cgm_cohort.csv
    reports/
        15d0_new_diet_cgm_cohort_summary.csv
        15d0_new_diet_cgm_exposure_coverage.csv
        15d0_new_diet_cgm_summary.txt
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")

CANDIDATE = (
    ROOT / "diet_microbiome_glucose_analysis" / "outputs" / "reports"
    / "new_diet_extension" / "15b_new_diet_extension_candidate_master.csv"
)
CGM_FILE = (
    ROOT / "cgm_deal" / "outputs" / "data" / "05_cgm_core_phenotypes.csv"
)
COV_FILE = (
    ROOT / "co-variant" / "outputs" / "data" / "02_covariate_master.csv"
)

BASE = (
    ROOT / "diet_microbiome_glucose_analysis" / "outputs"
    / "new_diet_extension"
)
DATA_DIR = BASE / "data"
REPORT_DIR = BASE / "reports"

OUT_FILE = DATA_DIR / "15d0_new_diet_cgm_cohort.csv"
SUMMARY_CSV = REPORT_DIR / "15d0_new_diet_cgm_cohort_summary.csv"
COVERAGE_CSV = REPORT_DIR / "15d0_new_diet_cgm_exposure_coverage.csv"
SUMMARY_TXT = REPORT_DIR / "15d0_new_diet_cgm_summary.txt"

EXPOSURES = {
    "EAT13": {
        "source": "modified_eat_lancet13_z",
        "alias": "EAT13_z",
        "role": "primary_extension",
    },
    "NOVA4": {
        "source": "nova4_total_energy_pct_cov80_z",
        "alias": "NOVA4_z",
        "role": "primary_extension",
    },
    "Carbohydrate_pct": {
        "source": "carbohydrate_energy_pct_z",
        "alias": "Carbohydrate_pct_z",
        "role": "exploratory_extension",
    },
}

PRIMARY_OUTCOMES = {
    "mean_glucose": "cgm_mean_z",
    "glucose_cv": "cgm_cv_z",
    "time_above_140": "cgm_above_140_z",
}

COVARIATE_COLUMNS = [
    "participant_id",
    "cohort",
    "age_years",
    "sex",
    "education_level",
    "smoking_status",
    "sleep_duration_hours_day",
    "physical_activity_met_h_week",
    "vitamin_use",
    "hormone_use",
    "cgm_device_type",
    "bmi",
    "known_diabetes",
    "a10_medication_use",
    "exclude_diabetes_or_a10",
    "amed_cgm_model2_covariates_complete",
    "amed_cgm_model3_covariates_complete",
]


def require(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


def norm_id(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def require_unique(df: pd.DataFrame, label: str) -> None:
    if df["participant_id"].duplicated().any():
        ex = (
            df.loc[df["participant_id"].duplicated(keep=False), "participant_id"]
            .head(20)
            .tolist()
        )
        raise RuntimeError(
            f"{label} has duplicate participant_id values: {ex}"
        )


def truthy(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).eq(1)
    x = s.astype(str).str.strip().str.lower()
    mapping = {
        "true": True, "1": True, "1.0": True,
        "false": False, "0": False, "0.0": False,
        "nan": False, "none": False, "": False,
    }
    bad = sorted(set(x.unique()) - set(mapping))
    if bad:
        raise RuntimeError(
            f"Unexpected boolean values: {bad[:20]}"
        )
    return x.map(mapping).astype(bool)


def main() -> int:
    for p, label in [
        (CANDIDATE, "Step15b frozen candidate exposure master"),
        (CGM_FILE, "final CGM phenotype table"),
        (COV_FILE, "covariate master"),
    ]:
        require(p, label)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(CANDIDATE, low_memory=False)
    cgm = pd.read_csv(CGM_FILE, low_memory=False)
    cov = pd.read_csv(
        COV_FILE,
        usecols=lambda c: c in set(COVARIATE_COLUMNS),
        low_memory=False,
    )

    for label, df in [
        ("score master", scores),
        ("CGM", cgm),
        ("covariates", cov),
    ]:
        if "participant_id" not in df.columns:
            raise RuntimeError(f"{label} missing participant_id")
        df["participant_id"] = norm_id(df["participant_id"])
        require_unique(df, label)

    required_score = {
        "participant_id",
        "mean_daily_energy_kcal",
        *[cfg["source"] for cfg in EXPOSURES.values()],
    }
    missing_score = sorted(required_score - set(scores.columns))
    if missing_score:
        raise RuntimeError(
            "Candidate master missing required fields: "
            + ", ".join(missing_score)
        )

    missing_outcomes = sorted(
        set(PRIMARY_OUTCOMES.values()) - set(cgm.columns)
    )
    if missing_outcomes:
        raise RuntimeError(
            "CGM phenotype table missing: " + ", ".join(missing_outcomes)
        )

    missing_cov = sorted(set(COVARIATE_COLUMNS) - set(cov.columns))
    if missing_cov:
        raise RuntimeError(
            "Covariate master missing: " + ", ".join(missing_cov)
        )

    score_piece = scores[
        [
            "participant_id",
            "mean_daily_energy_kcal",
            *[cfg["source"] for cfg in EXPOSURES.values()],
        ]
    ].copy()

    for exposure, cfg in EXPOSURES.items():
        score_piece[cfg["alias"]] = pd.to_numeric(
            score_piece[cfg["source"]],
            errors="coerce",
        )

    score_piece["mean_daily_energy_kcal"] = pd.to_numeric(
        score_piece["mean_daily_energy_kcal"], errors="coerce"
    )

    # Keep useful CGM metadata + all phenotype fields. CGM file is already one
    # selected baseline connection per participant, as in the canonical cohort.
    aligned = score_piece.merge(
        cgm,
        on="participant_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_cgm"),
    )

    aligned = aligned.merge(
        cov[COVARIATE_COLUMNS],
        on="participant_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_cov"),
    )

    # Canonical primary diabetes/A10 exclusion logic.
    exclusion = pd.to_numeric(
        aligned["exclude_diabetes_or_a10"], errors="coerce"
    )
    aligned["known_diabetes_or_a10_excluded"] = exclusion.eq(1)
    aligned["diabetes_a10_status_known"] = exclusion.notna()
    aligned["primary_cgm_analysis_eligible"] = ~aligned[
        "known_diabetes_or_a10_excluded"
    ]
    aligned["strict_known_nondiabetes_a10_eligible"] = exclusion.eq(0)

    # New common complete-case flags: canonical AMED-like flags + energy.
    energy = pd.to_numeric(
        aligned["mean_daily_energy_kcal"], errors="coerce"
    )
    energy_ok = energy.notna() & np.isfinite(energy.to_numpy(float))

    aligned["new_cgm_model2_covariates_complete"] = (
        truthy(aligned["amed_cgm_model2_covariates_complete"])
        & energy_ok
    )
    aligned["new_cgm_model3_covariates_complete"] = (
        truthy(aligned["amed_cgm_model3_covariates_complete"])
        & energy_ok
    )

    # Numeric outcome guards.
    for col in PRIMARY_OUTCOMES.values():
        aligned[col] = pd.to_numeric(aligned[col], errors="coerce")

    # Keep rows with at least one new exposure + at least one primary outcome.
    exposure_aliases = [cfg["alias"] for cfg in EXPOSURES.values()]
    aligned["has_any_new_exposure"] = aligned[
        exposure_aliases
    ].notna().any(axis=1)
    aligned["has_any_primary_cgm_outcome"] = aligned[
        list(PRIMARY_OUTCOMES.values())
    ].notna().any(axis=1)

    aligned = aligned.loc[
        aligned["has_any_new_exposure"]
        & aligned["has_any_primary_cgm_outcome"]
    ].copy()

    aligned = aligned.sort_values(
        "participant_id", kind="mergesort"
    ).reset_index(drop=True)
    require_unique(aligned, "formal new-diet CGM cohort")

    eligible = truthy(aligned["primary_cgm_analysis_eligible"])
    strict = truthy(
        aligned["strict_known_nondiabetes_a10_eligible"]
    )
    m2_complete = truthy(
        aligned["new_cgm_model2_covariates_complete"]
    )
    m3_complete = truthy(
        aligned["new_cgm_model3_covariates_complete"]
    )

    coverage_rows = []
    for exposure, cfg in EXPOSURES.items():
        score = pd.to_numeric(
            aligned[cfg["alias"]], errors="coerce"
        )
        score_ok = score.notna() & np.isfinite(score.to_numpy(float))

        row = {
            "exposure": exposure,
            "analysis_role": cfg["role"],
            "score_nonmissing_in_diet_cgm": int(score_ok.sum()),
            "score_nonmissing_primary_eligible": int(
                (score_ok & eligible).sum()
            ),
            "score_nonmissing_strict_known_nondiabetes": int(
                (score_ok & strict).sum()
            ),
            "model2_complete_primary_eligible": int(
                (score_ok & eligible & m2_complete).sum()
            ),
            "model3_complete_primary_eligible": int(
                (score_ok & eligible & m3_complete).sum()
            ),
        }

        for outcome_name, outcome_col in PRIMARY_OUTCOMES.items():
            y = pd.to_numeric(aligned[outcome_col], errors="coerce")
            y_ok = y.notna() & np.isfinite(y.to_numpy(float))

            row[f"{outcome_name}_model0_N"] = int(
                (score_ok & eligible & y_ok).sum()
            )
            row[f"{outcome_name}_model2_N"] = int(
                (score_ok & eligible & m2_complete & y_ok).sum()
            )
            row[f"{outcome_name}_model3_N"] = int(
                (score_ok & eligible & m3_complete & y_ok).sum()
            )

        coverage_rows.append(row)

    coverage = pd.DataFrame(coverage_rows)

    summary = {
        "candidate_score_participants": len(scores),
        "cgm_participants": len(cgm),
        "score_cgm_inner_join": int(
            len(
                scores[["participant_id"]].merge(
                    cgm[["participant_id"]],
                    on="participant_id",
                    how="inner",
                )
            )
        ),
        "formal_rows_written": len(aligned),
        "known_diabetes_or_a10_excluded": int(
            truthy(aligned["known_diabetes_or_a10_excluded"]).sum()
        ),
        "primary_analysis_eligible": int(eligible.sum()),
        "exclusion_status_unknown_but_retained": int(
            (~truthy(aligned["diabetes_a10_status_known"])).sum()
        ),
        "strict_known_nondiabetes_a10_eligible": int(strict.sum()),
        "model2_common_complete_primary_eligible": int(
            (eligible & m2_complete).sum()
        ),
        "model3_common_complete_primary_eligible": int(
            (eligible & m3_complete).sum()
        ),
    }

    aligned.to_csv(OUT_FILE, index=False)
    coverage.to_csv(COVERAGE_CSV, index=False)
    pd.DataFrame([summary]).to_csv(SUMMARY_CSV, index=False)

    print("=== STEP 15d0 FORMAL NEW-DIET × CGM COHORT ===")
    print("NO_ASSOCIATION_MODELS_FIT=True")
    print("ALL_THREE_COMPLETE_REQUIRED=False")
    print("PRIMARY_EXCLUSION=explicit known diabetes/A10 positives only")
    print(
        "MODEL2_NEW=canonical AMED-like Model2 + mean_daily_energy_kcal"
    )
    print("MODEL3_NEW=Model2-new + BMI")
    print("ALCOHOL_INCLUDED=False")

    print("\n--- COHORT SUMMARY ---")
    for k, v in summary.items():
        print(f"{k}={v}")

    print("\n--- EXPOSURE-SPECIFIC COVERAGE ---")
    print(coverage.to_string(index=False))

    lines = [
        "=== STEP 15d0 FORMAL NEW-DIET × CGM COHORT ===",
        "NO_ASSOCIATION_MODELS_FIT=True",
        "ALL_THREE_COMPLETE_REQUIRED=False",
        "PRIMARY_EXCLUSION=explicit known diabetes/A10 positives only",
        "MODEL2_NEW=canonical AMED-like Model2 + mean_daily_energy_kcal",
        "MODEL3_NEW=Model2-new + BMI",
        "ALCOHOL_INCLUDED=False",
        "",
        "--- COHORT SUMMARY ---",
        *[f"{k}={v}" for k, v in summary.items()],
        "",
        "--- EXPOSURE-SPECIFIC COVERAGE ---",
        coverage.to_string(index=False),
        "",
        f"COHORT={OUT_FILE}",
        f"COVERAGE={COVERAGE_CSV}",
    ]
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
