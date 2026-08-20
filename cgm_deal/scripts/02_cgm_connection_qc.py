#!/usr/bin/env python3
"""Apply fixed CGM connection-level quality-control criteria."""

from __future__ import print_function

import argparse
import os

import numpy as np
import pandas as pd

from cgm_utils import (
    CORE_OUTCOMES,
    DEFAULT_DATA_DIR,
    DEFAULT_IGLU_CSV,
    DEFAULT_REPORT_DIR,
    FULL_KEY,
    PRIMARY_OUTCOMES,
    normalize_identity_columns,
    print_summary,
    read_csv_preserve_ids,
    require_columns,
    require_unique,
    write_csv,
    write_summary,
)


def _exclusion_reason(row):
    reasons = []
    if not bool(row["qc_days_ge_10"]):
        reasons.append("days_lt_10_or_missing")
    if not bool(row["qc_loss_le_030"]):
        reasons.append("loss_gt_0.30_or_missing")
    if not bool(row["qc_primary_complete"]):
        reasons.append("primary_phenotype_missing")
    return "|".join(reasons)


def apply_connection_qc(selected, iglu):
    """Join selected connections to iglu and apply the three fixed QC rules."""

    selected = selected.copy()
    iglu = iglu.copy()
    normalize_identity_columns(selected)
    normalize_identity_columns(iglu)
    require_columns(
        selected,
        FULL_KEY + ["cgm_days", "cgm_qc_loss_fraction"],
        "selected baseline connections",
    )
    require_columns(iglu, FULL_KEY + CORE_OUTCOMES, "iglu")
    require_unique(selected, FULL_KEY, "selected baseline connections")
    require_unique(selected, ["participant_id"], "selected baseline connections")
    require_unique(iglu, FULL_KEY, "iglu")

    phenotype_table = iglu[FULL_KEY + CORE_OUTCOMES].copy()
    joined = selected.merge(
        phenotype_table,
        on=FULL_KEY,
        how="left",
        validate="one_to_one",
        indicator="_iglu_merge",
    )
    unmatched = joined["_iglu_merge"] != "both"
    if bool(unmatched.any()):
        examples = joined.loc[unmatched, FULL_KEY].head(10).to_dict("records")
        raise ValueError(
            "%d selected baseline connections are missing from iglu; examples=%s"
            % (int(unmatched.sum()), examples)
        )
    joined = joined.drop(columns=["_iglu_merge"])

    joined["cgm_days"] = pd.to_numeric(joined["cgm_days"], errors="coerce")
    joined["cgm_qc_loss_fraction"] = pd.to_numeric(
        joined["cgm_qc_loss_fraction"], errors="coerce"
    )
    for outcome in CORE_OUTCOMES:
        numeric = pd.to_numeric(joined[outcome], errors="coerce")
        joined[outcome] = numeric.mask(~np.isfinite(numeric))

    joined["qc_days_ge_10"] = joined["cgm_days"].ge(10)
    joined["qc_loss_le_030"] = joined["cgm_qc_loss_fraction"].le(0.30)
    joined["qc_primary_complete"] = joined[PRIMARY_OUTCOMES].notna().all(axis=1)
    joined["cgm_qc_pass"] = joined[
        ["qc_days_ge_10", "qc_loss_le_030", "qc_primary_complete"]
    ].all(axis=1)
    joined["cgm_qc_exclusion_reason"] = joined.apply(_exclusion_reason, axis=1)

    passed = joined.loc[joined["cgm_qc_pass"]].copy()
    excluded = joined.loc[~joined["cgm_qc_pass"]].copy()
    require_unique(passed, ["participant_id"], "CGM QC pass table")

    summary = {
        "selected_participants_input": int(len(joined)),
        "participants_days_ge_10": int(joined["qc_days_ge_10"].sum()),
        "participants_loss_le_030": int(joined["qc_loss_le_030"].sum()),
        "participants_primary_complete": int(joined["qc_primary_complete"].sum()),
        "participants_passing_all_connection_qc": int(len(passed)),
        "participants_excluded": int(len(excluded)),
    }
    return joined, passed, excluded, summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply fixed connection-level CGM quality control."
    )
    parser.add_argument(
        "--selected-csv",
        default=os.path.join(DEFAULT_DATA_DIR, "01_baseline_cgm_connections.csv"),
    )
    parser.add_argument("--iglu-csv", default=DEFAULT_IGLU_CSV)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    selected = read_csv_preserve_ids(args.selected_csv)
    iglu = read_csv_preserve_ids(args.iglu_csv)
    all_qc, passed, excluded, summary = apply_connection_qc(selected, iglu)

    all_path = os.path.join(args.data_dir, "02_cgm_qc_all.csv")
    pass_path = os.path.join(args.data_dir, "02_cgm_qc_pass_all.csv")
    exclusion_path = os.path.join(args.report_dir, "02_cgm_qc_exclusions.csv")
    summary_path = os.path.join(args.report_dir, "02_cgm_qc_summary.csv")
    write_csv(all_qc, all_path)
    write_csv(passed, pass_path)
    write_csv(excluded, exclusion_path)
    write_summary(summary, summary_path)
    print_summary("02 CONNECTION QC", summary)
    print("PASS_FILE=%s" % os.path.abspath(pass_path))
    print("EXCLUSION_REPORT=%s" % os.path.abspath(exclusion_path))


if __name__ == "__main__":
    main()
