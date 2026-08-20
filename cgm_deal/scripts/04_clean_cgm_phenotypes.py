#!/usr/bin/env python3
"""Apply the article's 95%-interval, 8-SD, and 5-SD phenotype cleaning."""

from __future__ import print_function

import argparse
import math
import os

import numpy as np
import pandas as pd

from cgm_utils import (
    CORE_OUTCOMES,
    DEFAULT_DATA_DIR,
    DEFAULT_REPORT_DIR,
    empty_report,
    normalize_identity_columns,
    print_summary,
    read_csv_preserve_ids,
    require_columns,
    require_unique,
    write_csv,
    write_summary,
)


COVERAGE_TARGET = 0.95
EXTREME_SD_THRESHOLD = 8.0
WINSOR_SD_THRESHOLD = 5.0


def clean_outcome_series(values, outcome, coverage=COVERAGE_TARGET):
    """Clean one phenotype and return values, parameters, and action masks.

    The shortest sorted window contains ``ceil(coverage * N)`` finite values.
    If several windows have the same width, the first (lowest-boundary) window is
    used, making the result deterministic.
    """

    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    finite_mask = numeric.notna() & np.isfinite(numeric)
    finite_values = np.sort(numeric.loc[finite_mask].to_numpy(dtype=float))
    count = int(len(finite_values))
    if count < 2:
        raise ValueError(
            "%s has fewer than two finite observations; cleaning is undefined"
            % outcome
        )
    if not (0.0 < float(coverage) <= 1.0):
        raise ValueError("coverage must lie in (0, 1]")

    window_count = max(2, int(math.ceil(float(coverage) * count)))
    window_count = min(window_count, count)
    widths = finite_values[window_count - 1 :] - finite_values[: count - window_count + 1]
    start_index = int(np.argmin(widths))
    interval_low = float(finite_values[start_index])
    interval_high = float(finite_values[start_index + window_count - 1])

    in_interval = finite_values[
        (finite_values >= interval_low) & (finite_values <= interval_high)
    ]
    center_mean = float(np.mean(in_interval))
    center_sd = float(np.std(in_interval, ddof=1))
    if not np.isfinite(center_sd) or center_sd <= 0.0:
        raise ValueError(
            "%s has a zero/non-finite SD inside the shortest 95%% interval" % outcome
        )

    extreme_lower = center_mean - EXTREME_SD_THRESHOLD * center_sd
    extreme_upper = center_mean + EXTREME_SD_THRESHOLD * center_sd
    winsor_lower = center_mean - WINSOR_SD_THRESHOLD * center_sd
    winsor_upper = center_mean + WINSOR_SD_THRESHOLD * center_sd

    nonfinite_input = numeric.notna() & ~np.isfinite(numeric)
    extreme_mask = nonfinite_input | (
        finite_mask
        & ((numeric < extreme_lower) | (numeric > extreme_upper))
    )
    winsor_mask = (
        finite_mask
        & ~extreme_mask
        & ((numeric < winsor_lower) | (numeric > winsor_upper))
    )

    cleaned = numeric.mask(extreme_mask)
    cleaned = cleaned.clip(lower=winsor_lower, upper=winsor_upper)
    parameters = {
        "outcome": outcome,
        "coverage_target": float(coverage),
        "finite_n": count,
        "window_target_n": int(window_count),
        "window_actual_n_including_boundary_ties": int(len(in_interval)),
        "interval_low": interval_low,
        "interval_high": interval_high,
        "center_mean": center_mean,
        "center_sample_sd": center_sd,
        "extreme_lower_8sd": extreme_lower,
        "extreme_upper_8sd": extreme_upper,
        "winsor_lower_5sd": winsor_lower,
        "winsor_upper_5sd": winsor_upper,
        "extreme_values_set_missing": int(extreme_mask.sum()),
        "values_winsorized": int(winsor_mask.sum()),
    }
    return cleaned, parameters, extreme_mask, winsor_mask


def clean_core_phenotypes(raw):
    """Apply the fixed cleaning independently to all seven raw outcomes."""

    raw = raw.copy()
    normalize_identity_columns(raw)
    raw_columns = [outcome + "_raw" for outcome in CORE_OUTCOMES]
    require_columns(raw, ["participant_id", "connection_id"] + raw_columns, "raw CGM table")
    require_unique(raw, ["participant_id"], "raw CGM table")

    cleaned_frame = raw.copy()
    parameter_rows = []
    action_rows = []
    for outcome in CORE_OUTCOMES:
        raw_column = outcome + "_raw"
        clean_column = outcome + "_clean"
        cleaned, parameters, extreme_mask, winsor_mask = clean_outcome_series(
            raw[raw_column], outcome
        )
        cleaned_frame[clean_column] = cleaned
        parameter_rows.append(parameters)

        for row_index in raw.index[extreme_mask]:
            action_rows.append(
                {
                    "participant_id": raw.at[row_index, "participant_id"],
                    "connection_id": raw.at[row_index, "connection_id"],
                    "outcome": outcome,
                    "raw_value": raw.at[row_index, raw_column],
                    "clean_value": np.nan,
                    "action": "set_missing_beyond_8sd",
                }
            )
        for row_index in raw.index[winsor_mask]:
            action_rows.append(
                {
                    "participant_id": raw.at[row_index, "participant_id"],
                    "connection_id": raw.at[row_index, "connection_id"],
                    "outcome": outcome,
                    "raw_value": raw.at[row_index, raw_column],
                    "clean_value": cleaned.at[row_index],
                    "action": "winsorized_at_5sd",
                }
            )

    parameter_report = pd.DataFrame(parameter_rows)
    action_columns = [
        "participant_id",
        "connection_id",
        "outcome",
        "raw_value",
        "clean_value",
        "action",
    ]
    if action_rows:
        action_report = pd.DataFrame(action_rows, columns=action_columns)
    else:
        action_report = empty_report(action_columns)

    require_unique(cleaned_frame, ["participant_id"], "clean CGM table")
    summary = {
        "participants_input": int(len(raw)),
        "participants_retained": int(len(cleaned_frame)),
        "outcomes_cleaned": int(len(CORE_OUTCOMES)),
        "extreme_values_set_missing": int(
            parameter_report["extreme_values_set_missing"].sum()
        ),
        "values_winsorized": int(parameter_report["values_winsorized"].sum()),
    }
    for outcome in CORE_OUTCOMES:
        summary[outcome + "_clean_nonmissing"] = int(
            cleaned_frame[outcome + "_clean"].notna().sum()
        )
    return cleaned_frame, parameter_report, action_report, summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean seven CGM phenotypes using the fixed article method."
    )
    parser.add_argument(
        "--raw-csv",
        default=os.path.join(DEFAULT_DATA_DIR, "03_iglu_core_raw.csv"),
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    raw = read_csv_preserve_ids(args.raw_csv)
    cleaned, parameters, actions, summary = clean_core_phenotypes(raw)

    clean_path = os.path.join(args.data_dir, "04_cgm_core_clean.csv")
    parameter_path = os.path.join(args.report_dir, "04_cleaning_parameters.csv")
    action_path = os.path.join(args.report_dir, "04_outlier_actions.csv")
    summary_path = os.path.join(args.report_dir, "04_cleaning_summary.csv")
    write_csv(cleaned, clean_path)
    write_csv(parameters, parameter_path)
    write_csv(actions, action_path)
    write_summary(summary, summary_path)
    print_summary("04 ROBUST PHENOTYPE CLEANING", summary)
    print("DATA_FILE=%s" % os.path.abspath(clean_path))
    print("PARAMETER_REPORT=%s" % os.path.abspath(parameter_path))
    print("OUTLIER_REPORT=%s" % os.path.abspath(action_path))


if __name__ == "__main__":
    main()
