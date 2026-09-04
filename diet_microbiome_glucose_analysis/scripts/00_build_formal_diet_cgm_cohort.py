#!/usr/bin/env python3
"""Build the formal current-four-score diet × CGM analysis cohort.

This supersedes the older AMED/hPDI-only alignment for the current proof-of-
concept pipeline.  It uses the already consolidated four-score table, the
final one-row-per-participant CGM phenotype table, and the current covariate
master.  Participants with known diabetes or known A10 medication use are
excluded from the primary analysis cohort; unknown exclusion status is kept
but explicitly reported for later sensitivity analysis.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/ec2-user/Desktop/HPP/Data")

SCORE_FILE = (
    ROOT
    / "diet_microbiome_analysis"
    / "data"
    / "00_current_four_scores_all_participants.csv"
)
CGM_FILE = ROOT / "cgm_deal" / "outputs" / "data" / "05_cgm_core_phenotypes.csv"
COV_FILE = ROOT / "co-variant" / "outputs" / "data" / "02_covariate_master.csv"

ANALYSIS_ROOT = ROOT / "diet_microbiome_glucose_analysis"
DATA_DIR = ANALYSIS_ROOT / "outputs" / "data"
REPORT_DIR = ANALYSIS_ROOT / "outputs" / "reports"

OUT_FILE = DATA_DIR / "00_current4_diet_cgm_cohort.csv"
SUMMARY_FILE = REPORT_DIR / "00_current4_diet_cgm_cohort_summary.csv"
SCORE_COVERAGE_FILE = REPORT_DIR / "00_current4_diet_cgm_score_coverage.csv"

SCORES = ["AHEI_z", "AMED_z", "hPDI_z", "rEDIH_z"]
PRIMARY_OUTCOMES = ["cgm_mean_z", "cgm_cv_z", "cgm_above_140_z"]

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
    "alcohol_intake_g_day",
    "bmi",
    "known_diabetes",
    "a10_medication_use",
    "exclude_diabetes_or_a10",
    "amed_cgm_model2_covariates_complete",
    "hpdi_cgm_model2_covariates_complete",
    "amed_cgm_model3_covariates_complete",
    "hpdi_cgm_model3_covariates_complete",
]


def norm_id(series):
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def require_columns(df, columns, label):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise RuntimeError(f"{label} missing columns: {', '.join(missing)}")


def require_unique_id(df, label):
    if df["participant_id"].duplicated().any():
        example = (
            df.loc[df["participant_id"].duplicated(keep=False), "participant_id"]
            .head(20)
            .tolist()
        )
        raise RuntimeError(f"{label} has duplicate participant_id values: {example}")


def read_table(path, label):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    require_columns(df, ["participant_id"], label)
    df["participant_id"] = norm_id(df["participant_id"])
    require_unique_id(df, label)
    return df


def to_numeric_nullable(series):
    return pd.to_numeric(series, errors="coerce")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    scores = read_table(SCORE_FILE, "four-score table")
    cgm = read_table(CGM_FILE, "final CGM phenotype table")
    cov = read_table(COV_FILE, "covariate master")

    require_columns(scores, SCORES, "four-score table")
    require_columns(cgm, PRIMARY_OUTCOMES, "final CGM phenotype table")
    require_columns(cov, COVARIATE_COLUMNS, "covariate master")

    # Keep only the current four scores and their useful companion columns.
    score_keep = [
        c for c in scores.columns
        if c == "participant_id"
        or c.startswith("AHEI_")
        or c.startswith("AMED_")
        or c.startswith("hPDI_")
        or c.startswith("rEDIH_")
        or c == "all_4_scores_complete"
    ]
    scores = scores[score_keep].copy()

    # CGM table is already one selected baseline connection per participant.
    # Keep its metadata plus all standardized/clean/raw phenotypes.
    aligned = scores.merge(
        cgm,
        on="participant_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_cgm"),
    )

    cov_piece = cov[COVARIATE_COLUMNS].copy()
    aligned = aligned.merge(
        cov_piece,
        on="participant_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_cov"),
    )

    # Use only explicit positive evidence to exclude. Unknown status is retained
    # in the primary cohort and separately flagged for sensitivity analysis.
    exclusion = to_numeric_nullable(aligned["exclude_diabetes_or_a10"])
    aligned["known_diabetes_or_a10_excluded"] = exclusion.eq(1)
    aligned["diabetes_a10_status_known"] = exclusion.notna()
    aligned["primary_cgm_analysis_eligible"] = ~aligned[
        "known_diabetes_or_a10_excluded"
    ]
    aligned["strict_known_nondiabetes_a10_eligible"] = exclusion.eq(0)

    for c in SCORES + PRIMARY_OUTCOMES:
        aligned[c] = pd.to_numeric(aligned[c], errors="coerce")

    # At least one current score + at least one primary CGM outcome.
    aligned["has_any_current_score"] = aligned[SCORES].notna().any(axis=1)
    aligned["has_any_primary_cgm_outcome"] = aligned[PRIMARY_OUTCOMES].notna().any(axis=1)

    aligned = aligned.loc[
        aligned["has_any_current_score"]
        & aligned["has_any_primary_cgm_outcome"]
    ].copy()

    aligned = aligned.sort_values("participant_id", kind="mergesort").reset_index(drop=True)
    require_unique_id(aligned, "formal diet-CGM cohort")

    score_rows = []
    for score in SCORES:
        score_rows.append(
            {
                "score": score.replace("_z", ""),
                "score_nonmissing_in_diet_cgm": int(aligned[score].notna().sum()),
                "score_nonmissing_primary_eligible": int(
                    (aligned[score].notna() & aligned["primary_cgm_analysis_eligible"]).sum()
                ),
                "score_nonmissing_strict_known_nondiabetes": int(
                    (
                        aligned[score].notna()
                        & aligned["strict_known_nondiabetes_a10_eligible"]
                    ).sum()
                ),
            }
        )

    summary = {
        "score_table_participants": len(scores),
        "cgm_participants": len(cgm),
        "score_cgm_inner_join": int(len(scores.merge(cgm[["participant_id"]], on="participant_id", how="inner"))),
        "formal_rows_written": len(aligned),
        "known_diabetes_or_a10_excluded": int(aligned["known_diabetes_or_a10_excluded"].sum()),
        "primary_analysis_eligible": int(aligned["primary_cgm_analysis_eligible"].sum()),
        "exclusion_status_unknown_but_retained": int((~aligned["diabetes_a10_status_known"]).sum()),
        "strict_known_nondiabetes_a10_eligible": int(aligned["strict_known_nondiabetes_a10_eligible"].sum()),
        "all_4_scores_complete": int(aligned[SCORES].notna().all(axis=1).sum()),
    }
    for outcome in PRIMARY_OUTCOMES:
        summary[f"{outcome}_nonmissing_primary_eligible"] = int(
            (aligned[outcome].notna() & aligned["primary_cgm_analysis_eligible"]).sum()
        )

    aligned.to_csv(OUT_FILE, index=False)
    pd.DataFrame([summary]).to_csv(SUMMARY_FILE, index=False)
    pd.DataFrame(score_rows).to_csv(SCORE_COVERAGE_FILE, index=False)

    print("=" * 92)
    print("FORMAL CURRENT-4 DIET × CGM COHORT")
    print("=" * 92)
    for k, v in summary.items():
        print(f"{k}: {v}")
    print()
    print(pd.DataFrame(score_rows).to_string(index=False))
    print()
    print("Primary exclusion rule: exclude only explicit exclude_diabetes_or_a10 == 1")
    print("Unknown exclusion status is retained and flagged for later strict sensitivity analysis.")
    print("SAVED:")
    print(OUT_FILE)
    print(SUMMARY_FILE)
    print(SCORE_COVERAGE_FILE)


if __name__ == "__main__":
    main()
