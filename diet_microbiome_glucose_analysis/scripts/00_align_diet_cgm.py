#!/usr/bin/env python3
"""Align baseline AMED/hPDI scores with final CGM phenotypes."""

from __future__ import print_function

import argparse
import os

import pandas as pd

from analysis_utils import (
    AMED_EXPOSURE,
    BASELINE_KEY,
    DEFAULT_AMED_CSV,
    DEFAULT_CGM_CSV,
    DEFAULT_DATA_DIR,
    DEFAULT_HPDI_CSV,
    DEFAULT_REPORT_DIR,
    EXPOSURES,
    HPDI_EXPOSURE,
    PRIMARY_CGM_OUTCOMES,
    filter_baseline,
    numeric_series,
    print_summary,
    read_csv_preserve_ids,
    require_columns,
    require_unique,
    write_csv,
    write_summary,
)


def _prepare_amed(amed, cohort, research_stage):
    prepared = filter_baseline(
        amed, "AMED", cohort=cohort, research_stage=research_stage
    )
    if AMED_EXPOSURE not in prepared.columns:
        raise ValueError(
            "AMED CSV is missing %s; rerun 05_calculate_amed_scores.py"
            % AMED_EXPOSURE
        )
    require_unique(prepared, BASELINE_KEY, "AMED")
    prepared[AMED_EXPOSURE] = numeric_series(
        prepared, AMED_EXPOSURE, "AMED", allow_missing=True
    )
    return prepared[BASELINE_KEY + [AMED_EXPOSURE]].copy()


def _prepare_hpdi(hpdi, cohort, research_stage):
    prepared = filter_baseline(
        hpdi, "hPDI", cohort=cohort, research_stage=research_stage
    )
    if HPDI_EXPOSURE not in prepared.columns:
        raise ValueError(
            "hPDI CSV is missing %s; rerun scripts/hpdi/05_calculate_hpdi_scores.py"
            % HPDI_EXPOSURE
        )
    require_unique(prepared, BASELINE_KEY, "hPDI")
    prepared[HPDI_EXPOSURE] = numeric_series(
        prepared, HPDI_EXPOSURE, "hPDI", allow_missing=True
    )
    return prepared[BASELINE_KEY + [HPDI_EXPOSURE]].copy()


def _prepare_cgm(cgm, cohort, research_stage):
    prepared = filter_baseline(
        cgm, "CGM", cohort=cohort, research_stage=research_stage
    )
    require_columns(prepared, BASELINE_KEY + PRIMARY_CGM_OUTCOMES, "CGM")
    require_unique(prepared, BASELINE_KEY, "CGM")
    for outcome in PRIMARY_CGM_OUTCOMES:
        prepared[outcome] = numeric_series(
            prepared, outcome, "CGM", allow_missing=True
        )
    return prepared


def _prepare_gut_membership(gut, cohort, research_stage):
    prepared = filter_baseline(
        gut, "gut microbiome", cohort=cohort, research_stage=research_stage
    )
    sample_counts = (
        prepared.groupby(BASELINE_KEY, sort=False)
        .size()
        .rename("gut_sample_count")
        .reset_index()
    )
    sample_counts["has_gut_sample"] = True
    require_unique(sample_counts, BASELINE_KEY, "gut participant membership")
    return sample_counts


def _build_membership(amed, hpdi, cgm, gut_membership=None):
    amed_members = amed[BASELINE_KEY].copy()
    amed_members["has_amed_score"] = amed[AMED_EXPOSURE].notna().to_numpy()
    hpdi_members = hpdi[BASELINE_KEY].copy()
    hpdi_members["has_hpdi_score"] = hpdi[HPDI_EXPOSURE].notna().to_numpy()
    cgm_members = cgm[BASELINE_KEY].copy()
    cgm_members["has_cgm_row"] = True

    membership = amed_members.merge(
        hpdi_members, on=BASELINE_KEY, how="outer", validate="one_to_one"
    )
    membership = membership.merge(
        cgm_members, on=BASELINE_KEY, how="outer", validate="one_to_one"
    )
    if gut_membership is not None:
        membership = membership.merge(
            gut_membership,
            on=BASELINE_KEY,
            how="outer",
            validate="one_to_one",
        )
    else:
        membership["gut_sample_count"] = 0
        membership["has_gut_sample"] = False

    for column in [
        "has_amed_score",
        "has_hpdi_score",
        "has_cgm_row",
        "has_gut_sample",
    ]:
        membership[column] = membership[column].eq(True).astype(bool)
    membership["gut_sample_count"] = (
        pd.to_numeric(membership["gut_sample_count"], errors="coerce")
        .fillna(0)
        .astype("int64")
    )
    membership["has_any_diet_score"] = membership[
        ["has_amed_score", "has_hpdi_score"]
    ].any(axis=1)
    membership["has_diet_and_cgm"] = (
        membership["has_any_diet_score"] & membership["has_cgm_row"]
    )
    membership["has_amed_hpdi_and_cgm"] = (
        membership["has_amed_score"]
        & membership["has_hpdi_score"]
        & membership["has_cgm_row"]
    )
    membership["has_diet_cgm_and_gut"] = (
        membership["has_diet_and_cgm"] & membership["has_gut_sample"]
    )
    return membership.sort_values(BASELINE_KEY, kind="mergesort").reset_index(drop=True)


def align_diet_cgm(
    amed,
    hpdi,
    cgm,
    gut=None,
    cohort="10k",
    research_stage="00_00_visit",
):
    """Return aligned diet–CGM data, source membership, and overlap counts."""

    amed_prepared = _prepare_amed(amed, cohort, research_stage)
    hpdi_prepared = _prepare_hpdi(hpdi, cohort, research_stage)
    cgm_prepared = _prepare_cgm(cgm, cohort, research_stage)
    gut_membership = None
    if gut is not None:
        gut_membership = _prepare_gut_membership(gut, cohort, research_stage)

    diet = amed_prepared.merge(
        hpdi_prepared,
        on=BASELINE_KEY,
        how="outer",
        validate="one_to_one",
    )
    diet = diet.loc[diet[EXPOSURES].notna().any(axis=1)].copy()
    aligned = diet.merge(
        cgm_prepared,
        on=BASELINE_KEY,
        how="inner",
        validate="one_to_one",
    )
    require_unique(aligned, BASELINE_KEY, "aligned diet-CGM table")

    membership = _build_membership(
        amed_prepared, hpdi_prepared, cgm_prepared, gut_membership
    )
    summary = {
        "amed_participants": int(membership["has_amed_score"].sum()),
        "hpdi_participants": int(membership["has_hpdi_score"].sum()),
        "diet_union_participants": int(membership["has_any_diet_score"].sum()),
        "cgm_participants": int(membership["has_cgm_row"].sum()),
        "amed_and_cgm_participants": int(
            (membership["has_amed_score"] & membership["has_cgm_row"]).sum()
        ),
        "hpdi_and_cgm_participants": int(
            (membership["has_hpdi_score"] & membership["has_cgm_row"]).sum()
        ),
        "diet_union_and_cgm_participants": int(
            membership["has_diet_and_cgm"].sum()
        ),
        "amed_hpdi_and_cgm_participants": int(
            membership["has_amed_hpdi_and_cgm"].sum()
        ),
        "gut_membership_evaluated": bool(gut is not None),
        "gut_participants": int(membership["has_gut_sample"].sum()),
        "diet_cgm_and_gut_participants": int(
            membership["has_diet_cgm_and_gut"].sum()
        ),
        "aligned_rows_written": int(len(aligned)),
    }
    return aligned, membership, summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Align baseline AMED/hPDI Z-scores with final CGM phenotypes."
    )
    parser.add_argument("--amed-csv", default=DEFAULT_AMED_CSV)
    parser.add_argument("--hpdi-csv", default=DEFAULT_HPDI_CSV)
    parser.add_argument("--cgm-csv", default=DEFAULT_CGM_CSV)
    parser.add_argument("--gut-csv", default=None)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--cohort", default="10k")
    parser.add_argument("--research-stage", default="00_00_visit")
    return parser.parse_args()


def main():
    args = parse_args()
    amed = read_csv_preserve_ids(args.amed_csv)
    hpdi = read_csv_preserve_ids(args.hpdi_csv)
    cgm = read_csv_preserve_ids(args.cgm_csv)
    gut = read_csv_preserve_ids(args.gut_csv) if args.gut_csv else None
    aligned, membership, summary = align_diet_cgm(
        amed,
        hpdi,
        cgm,
        gut=gut,
        cohort=args.cohort,
        research_stage=args.research_stage,
    )

    aligned_path = os.path.join(args.data_dir, "00_diet_cgm_aligned.csv")
    membership_path = os.path.join(args.report_dir, "00_id_membership.csv")
    summary_path = os.path.join(args.report_dir, "00_overlap_summary.csv")
    write_csv(aligned, aligned_path)
    write_csv(membership, membership_path)
    write_summary(summary, summary_path)
    print_summary("00 DIET-CGM ALIGNMENT", summary)
    print("ALIGNED_FILE=%s" % os.path.abspath(aligned_path))
    print("MEMBERSHIP_REPORT=%s" % os.path.abspath(membership_path))


if __name__ == "__main__":
    main()
