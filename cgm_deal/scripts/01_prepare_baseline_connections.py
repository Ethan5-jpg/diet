#!/usr/bin/env python3
"""Select one baseline CGM connection per participant."""

from __future__ import print_function

import argparse
import os

import numpy as np
import pandas as pd

from cgm_utils import (
    CONNECTION_KEY,
    DEFAULT_CGM_CSV,
    DEFAULT_DAILY_CSV,
    DEFAULT_DATA_DIR,
    DEFAULT_REPORT_DIR,
    FULL_KEY,
    empty_report,
    normalize_identity_columns,
    print_summary,
    read_csv_preserve_ids,
    require_columns,
    require_unique,
    write_csv,
    write_summary,
)


LOSS_COLUMN = "percentage_of_cgm_datapoints_lost_in_qc"
DATAPOINT_COLUMN = "number_of_cgm_datapoints_available_for_the_connection"
TIMESTAMP_CANDIDATES = ["collection_timestamp", "connection_timestamp"]


def _standardize_timestamp_column(frame):
    """Accept either known timestamp label and expose collection_timestamp."""

    if "collection_timestamp" in frame.columns:
        return frame
    if "connection_timestamp" in frame.columns:
        return frame.rename(columns={"connection_timestamp": "collection_timestamp"})
    raise ValueError(
        "cgm is missing a connection timestamp; expected one of: %s"
        % ", ".join(TIMESTAMP_CANDIDATES)
    )


def derive_connection_days(daily):
    """Derive and validate the number of sequential days for every connection."""

    daily = daily.copy()
    normalize_identity_columns(daily)
    require_columns(
        daily,
        FULL_KEY + ["connection_day", "collection_date"],
        "iglu_daily",
    )

    day_number = pd.to_numeric(daily["connection_day"], errors="coerce")
    if bool(day_number.isna().any()):
        raise ValueError("iglu_daily.connection_day contains missing/non-numeric values")
    if bool(((day_number % 1) != 0).any()):
        raise ValueError("iglu_daily.connection_day contains non-integer values")
    daily["_connection_day_numeric"] = day_number.astype("int64")

    if bool(daily["collection_date"].isna().any()):
        raise ValueError("iglu_daily.collection_date contains missing values")

    duplicate_day = daily.duplicated(
        CONNECTION_KEY + ["_connection_day_numeric"], keep=False
    )
    if bool(duplicate_day.any()):
        raise ValueError(
            "iglu_daily has %d duplicate connection-day rows"
            % int(duplicate_day.sum())
        )

    grouped = daily.groupby(CONNECTION_KEY, sort=False, dropna=False)
    derived = grouped.agg(
        daily_rows=("_connection_day_numeric", "size"),
        cgm_days=("_connection_day_numeric", "nunique"),
        first_connection_day=("_connection_day_numeric", "min"),
        last_connection_day=("_connection_day_numeric", "max"),
        unique_collection_dates=("collection_date", "nunique"),
    ).reset_index()

    bad_sequence = (
        (derived["first_connection_day"] != 1)
        | (derived["last_connection_day"] != derived["cgm_days"])
        | (derived["daily_rows"] != derived["cgm_days"])
        | (derived["unique_collection_dates"] != derived["cgm_days"])
    )
    if bool(bad_sequence.any()):
        examples = derived.loc[bad_sequence].head(5).to_dict("records")
        raise ValueError(
            "iglu_daily has %d connections with a non-sequential or inconsistent "
            "day series; examples=%s" % (int(bad_sequence.sum()), examples)
        )

    require_unique(derived, CONNECTION_KEY, "derived connection days")
    return derived


def _require_exact_daily_coverage(cgm, connection_days):
    """Require daily-derived connection keys to exactly match cgm."""

    left = cgm[CONNECTION_KEY].drop_duplicates().copy()
    right = connection_days[CONNECTION_KEY].drop_duplicates().copy()
    comparison = left.merge(right, on=CONNECTION_KEY, how="outer", indicator=True)
    missing_daily = comparison[comparison["_merge"] == "left_only"]
    extra_daily = comparison[comparison["_merge"] == "right_only"]
    if len(missing_daily) or len(extra_daily):
        raise ValueError(
            "Daily coverage does not match cgm: cgm_without_daily=%d, "
            "daily_without_cgm=%d"
            % (len(missing_daily), len(extra_daily))
        )


def _find_unresolved_ties(baseline, priority_columns):
    """Return duplicate candidates not separable by available numeric priorities."""

    duplicate_people = baseline.duplicated("participant_id", keep=False)
    candidates = baseline.loc[duplicate_people]
    if candidates.empty:
        return candidates
    tie_columns = ["participant_id"] + priority_columns
    return candidates.loc[candidates.duplicated(tie_columns, keep=False)]


def prepare_baseline_connections(
    cgm,
    daily,
    cohort="10k",
    research_stage="00_00_visit",
):
    """Return selected baseline rows, duplicate candidates, and a flow summary."""

    cgm = _standardize_timestamp_column(cgm.copy())
    daily = daily.copy()
    normalize_identity_columns(cgm)
    normalize_identity_columns(daily)
    require_columns(
        cgm,
        FULL_KEY
        + [
            "collection_timestamp",
            "cgm_device_type",
            LOSS_COLUMN,
        ],
        "cgm",
    )
    require_unique(cgm, FULL_KEY, "cgm")
    require_unique(cgm, CONNECTION_KEY, "cgm connection table")

    connection_days = derive_connection_days(daily)
    _require_exact_daily_coverage(cgm, connection_days)
    merged = cgm.merge(
        connection_days,
        on=CONNECTION_KEY,
        how="left",
        validate="one_to_one",
    )

    baseline = merged.loc[
        (merged["cohort"] == str(cohort))
        & (merged["research_stage"] == str(research_stage))
    ].copy()
    if baseline.empty:
        raise ValueError(
            "No rows matched cohort=%s and research_stage=%s"
            % (cohort, research_stage)
        )

    baseline["cgm_qc_loss_fraction"] = pd.to_numeric(
        baseline[LOSS_COLUMN], errors="coerce"
    )
    if bool(baseline["cgm_qc_loss_fraction"].isna().any()):
        raise ValueError("Baseline cgm loss fraction contains missing/non-numeric values")
    loss_out_of_range = ~baseline["cgm_qc_loss_fraction"].between(0.0, 1.0)
    if bool(loss_out_of_range.any()):
        raise ValueError("Baseline cgm loss fraction contains values outside [0, 1]")

    baseline["collection_timestamp_parsed"] = pd.to_datetime(
        baseline["collection_timestamp"], errors="coerce", utc=True
    )
    if bool(baseline["collection_timestamp_parsed"].isna().any()):
        raise ValueError("Baseline collection_timestamp contains invalid values")

    sort_columns = ["participant_id", "cgm_days", "cgm_qc_loss_fraction"]
    ascending = [True, False, True]
    tie_priority = ["cgm_days", "cgm_qc_loss_fraction"]

    if DATAPOINT_COLUMN in baseline.columns:
        baseline[DATAPOINT_COLUMN] = pd.to_numeric(
            baseline[DATAPOINT_COLUMN], errors="coerce"
        )
        if bool(baseline[DATAPOINT_COLUMN].isna().any()):
            raise ValueError("Available datapoint count contains missing/non-numeric values")
        sort_columns.append(DATAPOINT_COLUMN)
        ascending.append(False)
        tie_priority.append(DATAPOINT_COLUMN)
    else:
        unresolved = _find_unresolved_ties(baseline, tie_priority)
        if not unresolved.empty:
            examples = unresolved[
                ["participant_id", "connection_id", "cgm_days", "cgm_qc_loss_fraction"]
            ].to_dict("records")
            raise ValueError(
                "Missing datapoint count leaves duplicate baseline connections tied "
                "after cgm_days and loss; cannot apply the fixed selection priority. "
                "Examples=%s" % examples[:10]
            )

    sort_columns.extend(["collection_timestamp_parsed", "connection_id"])
    ascending.extend([True, True])
    ranked = baseline.sort_values(
        sort_columns,
        ascending=ascending,
        kind="mergesort",
    ).copy()
    ranked["selection_rank"] = ranked.groupby("participant_id", sort=False).cumcount() + 1
    ranked["selected_flag"] = ranked["selection_rank"] == 1

    duplicate_mask = ranked.duplicated("participant_id", keep=False)
    duplicate_report = ranked.loc[duplicate_mask].copy()
    if duplicate_report.empty:
        duplicate_report = empty_report(
            list(ranked.columns) + ["selection_reason"]
        )
    else:
        duplicate_report["selection_reason"] = np.where(
            duplicate_report["selected_flag"],
            "selected_by_fixed_priority",
            "not_selected_by_fixed_priority",
        )

    selected = ranked.loc[ranked["selected_flag"]].copy()
    selected = selected.drop(
        columns=["collection_timestamp_parsed", "selection_rank", "selected_flag"]
    )
    require_unique(selected, ["participant_id"], "selected baseline connections")

    summary = {
        "cgm_connections_total": int(len(cgm)),
        "daily_connections_total": int(len(connection_days)),
        "baseline_connections": int(len(baseline)),
        "baseline_participants": int(baseline["participant_id"].nunique()),
        "participants_with_multiple_connections": int(
            baseline.loc[
                baseline.duplicated("participant_id", keep=False), "participant_id"
            ].nunique()
        ),
        "selected_participants": int(len(selected)),
        "datapoint_count_available": bool(DATAPOINT_COLUMN in baseline.columns),
    }
    return selected, duplicate_report, summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select one baseline CGM connection per participant."
    )
    parser.add_argument("--cgm-csv", default=DEFAULT_CGM_CSV)
    parser.add_argument("--daily-csv", default=DEFAULT_DAILY_CSV)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--cohort", default="10k")
    parser.add_argument("--research-stage", default="00_00_visit")
    return parser.parse_args()


def main():
    args = parse_args()
    cgm = read_csv_preserve_ids(args.cgm_csv)
    daily = read_csv_preserve_ids(args.daily_csv)
    selected, duplicates, summary = prepare_baseline_connections(
        cgm,
        daily,
        cohort=args.cohort,
        research_stage=args.research_stage,
    )

    data_path = os.path.join(args.data_dir, "01_baseline_cgm_connections.csv")
    duplicate_path = os.path.join(
        args.report_dir, "01_duplicate_baseline_connections.csv"
    )
    summary_path = os.path.join(args.report_dir, "01_selection_summary.csv")
    write_csv(selected, data_path)
    write_csv(duplicates, duplicate_path)
    write_summary(summary, summary_path)
    print_summary("01 BASELINE CONNECTION SELECTION", summary)
    print("DATA_FILE=%s" % os.path.abspath(data_path))
    print("DUPLICATE_REPORT=%s" % os.path.abspath(duplicate_path))


if __name__ == "__main__":
    main()
