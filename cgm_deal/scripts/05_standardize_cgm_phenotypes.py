#!/usr/bin/env python3
"""Standardize cleaned CGM phenotypes and write the final one-row-per-person table."""

from __future__ import print_function

import argparse
import os

import numpy as np
import pandas as pd

from cgm_utils import (
    CORE_OUTCOMES,
    DEFAULT_DATA_DIR,
    DEFAULT_REPORT_DIR,
    normalize_identity_columns,
    print_summary,
    read_csv_preserve_ids,
    require_columns,
    require_unique,
    write_csv,
    write_summary,
)


def standardize_clean_phenotypes(cleaned):
    """Add Z-scores computed from each phenotype's clean nonmissing sample."""

    cleaned = cleaned.copy()
    normalize_identity_columns(cleaned)
    required = []
    for outcome in CORE_OUTCOMES:
        required.extend([outcome + "_raw", outcome + "_clean"])
    require_columns(cleaned, ["participant_id"] + required, "clean CGM table")
    require_unique(cleaned, ["participant_id"], "clean CGM table")

    final = cleaned.copy()
    parameter_rows = []
    for outcome in CORE_OUTCOMES:
        clean_column = outcome + "_clean"
        z_column = outcome + "_z"
        numeric = pd.to_numeric(final[clean_column], errors="coerce").astype(float)
        finite_mask = numeric.notna() & np.isfinite(numeric)
        if int(finite_mask.sum()) < 2:
            raise ValueError(
                "%s has fewer than two finite clean values; Z-score is undefined"
                % outcome
            )
        mean_value = float(numeric.loc[finite_mask].mean())
        sample_sd = float(numeric.loc[finite_mask].std(ddof=1))
        if not np.isfinite(sample_sd) or sample_sd <= 0.0:
            raise ValueError(
                "%s has a zero/non-finite clean sample SD; Z-score is undefined"
                % outcome
            )
        final[z_column] = (numeric - mean_value) / sample_sd
        final.loc[~finite_mask, z_column] = np.nan
        parameter_rows.append(
            {
                "outcome": outcome,
                "clean_nonmissing_n": int(finite_mask.sum()),
                "clean_mean": mean_value,
                "clean_sample_sd": sample_sd,
            }
        )

    final["A10_excluded_flag"] = np.nan
    final["A10_exclusion_status"] = "not_evaluated"

    phenotype_columns = []
    for outcome in CORE_OUTCOMES:
        phenotype_columns.extend(
            [outcome + "_raw", outcome + "_clean", outcome + "_z"]
        )
    a10_columns = ["A10_excluded_flag", "A10_exclusion_status"]
    metadata_columns = [
        column
        for column in final.columns
        if column not in phenotype_columns and column not in a10_columns
    ]
    final = final[metadata_columns + phenotype_columns + a10_columns]
    require_unique(final, ["participant_id"], "final CGM phenotype table")

    parameters = pd.DataFrame(parameter_rows)
    return final, parameters


def summarize_final(final):
    summary = {
        "participants_final": int(len(final)),
        "outcomes_standardized": int(len(CORE_OUTCOMES)),
        "A10_exclusion_status": "not_evaluated",
    }
    for outcome in CORE_OUTCOMES:
        summary[outcome + "_z_nonmissing"] = int(final[outcome + "_z"].notna().sum())
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create Z-scores and the final fixed CGM phenotype file."
    )
    parser.add_argument(
        "--clean-csv",
        default=os.path.join(DEFAULT_DATA_DIR, "04_cgm_core_clean.csv"),
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    cleaned = read_csv_preserve_ids(args.clean_csv)
    final, parameters = standardize_clean_phenotypes(cleaned)
    summary = summarize_final(final)

    final_path = os.path.join(args.data_dir, "05_cgm_core_phenotypes.csv")
    parameter_path = os.path.join(args.report_dir, "05_standardization_parameters.csv")
    summary_path = os.path.join(args.report_dir, "05_final_summary.csv")
    write_csv(final, final_path)
    write_csv(parameters, parameter_path)
    write_summary(summary, summary_path)
    print_summary("05 STANDARDIZATION AND FINAL OUTPUT", summary)
    print("FINAL_FILE=%s" % os.path.abspath(final_path))
    print("PARAMETER_REPORT=%s" % os.path.abspath(parameter_path))


if __name__ == "__main__":
    main()
