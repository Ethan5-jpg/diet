#!/usr/bin/env python3
"""Extract the seven fixed CGM phenotypes and apply legality checks."""

from __future__ import print_function

import argparse
import os

import numpy as np
import pandas as pd

from cgm_utils import (
    CORE_OUTCOMES,
    DEFAULT_DATA_DIR,
    DEFAULT_REPORT_DIR,
    FULL_KEY,
    NONNEGATIVE_OUTCOMES,
    PERCENT_OUTCOMES,
    empty_report,
    normalize_identity_columns,
    print_summary,
    read_csv_preserve_ids,
    require_columns,
    require_unique,
    write_csv,
    write_summary,
)


OPTIONAL_METADATA = [
    "collection_timestamp",
    "connection_timestamp",
    "cgm_device_type",
    "cgm_days",
    "cgm_qc_loss_fraction",
    "qc_days_ge_10",
    "qc_loss_le_030",
    "qc_primary_complete",
    "cgm_qc_pass",
]

DATAPOINT_COLUMN = "number_of_cgm_datapoints_available_for_the_connection"


def _append_legality_records(
    records,
    frame,
    outcome,
    original,
    invalid_mask,
    reason,
):
    if not bool(invalid_mask.any()):
        return
    for row_index in frame.index[invalid_mask]:
        records.append(
            {
                "participant_id": frame.at[row_index, "participant_id"],
                "connection_id": frame.at[row_index, "connection_id"],
                "outcome": outcome,
                "source_value": original.at[row_index],
                "invalid_reason": reason,
                "action": "set_missing",
            }
        )


def extract_core_phenotypes(qc_pass):
    """Return fixed raw phenotype columns and an invalid-value report."""

    qc_pass = qc_pass.copy()
    normalize_identity_columns(qc_pass)
    require_columns(qc_pass, FULL_KEY + CORE_OUTCOMES, "CGM QC pass table")
    require_unique(qc_pass, FULL_KEY, "CGM QC pass table")
    require_unique(qc_pass, ["participant_id"], "CGM QC pass table")

    metadata = list(FULL_KEY)
    for column in OPTIONAL_METADATA:
        if column in qc_pass.columns and column not in metadata:
            metadata.append(column)
    output = qc_pass[metadata].copy()

    if DATAPOINT_COLUMN in qc_pass.columns:
        output["cgm_datapoints"] = pd.to_numeric(
            qc_pass[DATAPOINT_COLUMN], errors="coerce"
        )
        output["cgm_datapoints_status"] = "available"
    else:
        output["cgm_datapoints"] = np.nan
        output["cgm_datapoints_status"] = "not_available_in_export"

    records = []
    for outcome in CORE_OUTCOMES:
        original = qc_pass[outcome]
        numeric = pd.to_numeric(original, errors="coerce")
        non_numeric = original.notna() & numeric.isna()
        _append_legality_records(
            records,
            qc_pass,
            outcome,
            original,
            non_numeric,
            "non_numeric",
        )

        non_finite = numeric.notna() & ~np.isfinite(numeric)
        _append_legality_records(
            records,
            qc_pass,
            outcome,
            original,
            non_finite,
            "non_finite",
        )

        invalid = non_numeric | non_finite
        if outcome in PERCENT_OUTCOMES:
            out_of_range = (
                numeric.notna()
                & np.isfinite(numeric)
                & ~numeric.between(0.0, 100.0)
            )
            _append_legality_records(
                records,
                qc_pass,
                outcome,
                original,
                out_of_range,
                "outside_0_100",
            )
            invalid = invalid | out_of_range
        if outcome in NONNEGATIVE_OUTCOMES:
            negative = numeric.notna() & np.isfinite(numeric) & numeric.lt(0.0)
            _append_legality_records(
                records,
                qc_pass,
                outcome,
                original,
                negative,
                "negative_value",
            )
            invalid = invalid | negative

        output[outcome + "_raw"] = numeric.mask(invalid)

    report_columns = [
        "participant_id",
        "connection_id",
        "outcome",
        "source_value",
        "invalid_reason",
        "action",
    ]
    if records:
        legality_report = pd.DataFrame(records, columns=report_columns)
    else:
        legality_report = empty_report(report_columns)

    require_unique(output, ["participant_id"], "raw CGM phenotype table")
    summary = {
        "participants_input": int(len(qc_pass)),
        "participants_retained": int(len(output)),
        "illegal_values_set_missing": int(len(legality_report)),
        "core_outcomes_extracted": int(len(CORE_OUTCOMES)),
        "datapoint_count_available": bool(DATAPOINT_COLUMN in qc_pass.columns),
    }
    for outcome in CORE_OUTCOMES:
        summary[outcome + "_raw_nonmissing"] = int(
            output[outcome + "_raw"].notna().sum()
        )
    return output, legality_report, summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract and validate the seven fixed CGM phenotypes."
    )
    parser.add_argument(
        "--qc-pass-csv",
        default=os.path.join(DEFAULT_DATA_DIR, "02_cgm_qc_pass_all.csv"),
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    qc_pass = read_csv_preserve_ids(args.qc_pass_csv)
    raw, legality_report, summary = extract_core_phenotypes(qc_pass)

    raw_path = os.path.join(args.data_dir, "03_iglu_core_raw.csv")
    legality_path = os.path.join(args.report_dir, "03_phenotype_legality.csv")
    summary_path = os.path.join(args.report_dir, "03_extraction_summary.csv")
    write_csv(raw, raw_path)
    write_csv(legality_report, legality_path)
    write_summary(summary, summary_path)
    print_summary("03 CORE PHENOTYPE EXTRACTION", summary)
    print("DATA_FILE=%s" % os.path.abspath(raw_path))
    print("LEGALITY_REPORT=%s" % os.path.abspath(legality_path))


if __name__ == "__main__":
    main()
