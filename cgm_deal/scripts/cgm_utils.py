#!/usr/bin/env python3
"""Shared utilities and fixed field definitions for the CGM workflow."""

from __future__ import print_function

import os

import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
HPP_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

DEFAULT_SOURCE_DIR = os.path.join(HPP_ROOT, "Data", "Transfer", "cgm")
DEFAULT_CGM_CSV = os.path.join(DEFAULT_SOURCE_DIR, "cgm.csv")
DEFAULT_IGLU_CSV = os.path.join(DEFAULT_SOURCE_DIR, "iglu.csv")
DEFAULT_DAILY_CSV = os.path.join(DEFAULT_SOURCE_DIR, "iglu_daily.csv")

DEFAULT_OUTPUT_DIR = os.path.join(WORKFLOW_DIR, "outputs")
DEFAULT_DATA_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "data")
DEFAULT_REPORT_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "reports")

FULL_KEY = [
    "participant_id",
    "cohort",
    "research_stage",
    "array_index",
    "connection_id",
]

# array_index identifies a row inside repeated/array tables.  In iglu_daily it
# changes from day to day, so it must not be used to define a CGM connection.
CONNECTION_KEY = [
    "participant_id",
    "cohort",
    "research_stage",
    "connection_id",
]

TEXT_ID_COLUMNS = [
    "participant_id",
    "cohort",
    "research_stage",
    "connection_id",
]

PRIMARY_OUTCOMES = ["cgm_mean", "cgm_cv", "cgm_above_140"]

CORE_OUTCOMES = [
    "cgm_mean",
    "cgm_cv",
    "cgm_above_140",
    "cgm_gmi",
    "cgm_in_range_63_140",
    "cgm_mage",
    "cgm_modd",
]

PERCENT_OUTCOMES = ["cgm_above_140", "cgm_in_range_63_140"]
NONNEGATIVE_OUTCOMES = ["cgm_cv", "cgm_gmi", "cgm_mage", "cgm_modd"]


def ensure_directory(path):
    """Create an output directory if needed and return its absolute path."""

    absolute = os.path.abspath(path)
    if not os.path.isdir(absolute):
        os.makedirs(absolute)
    return absolute


def read_csv_preserve_ids(path):
    """Read a CSV while preserving identity columns as literal text."""

    if not os.path.isfile(path):
        raise FileNotFoundError("Input CSV does not exist: %s" % path)

    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    dtype_map = {}
    for column in TEXT_ID_COLUMNS:
        if column in header.columns:
            dtype_map[column] = str

    frame = pd.read_csv(
        path,
        dtype=dtype_map,
        low_memory=False,
        encoding="utf-8-sig",
    )
    normalize_identity_columns(frame)
    return frame


def normalize_identity_columns(frame):
    """Trim text IDs and reject missing/empty identities in place."""

    present_identity = [column for column in FULL_KEY if column in frame.columns]
    for column in present_identity:
        missing = frame[column].isna()
        if bool(missing.any()):
            raise ValueError(
                "Identity column %s contains %d missing values"
                % (column, int(missing.sum()))
            )

        if column == "array_index":
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if bool(numeric.isna().any()):
                raise ValueError("array_index contains non-numeric values")
            fractional = numeric % 1 != 0
            if bool(fractional.any()):
                raise ValueError("array_index contains non-integer values")
            frame[column] = numeric.astype("int64")
        else:
            normalized = frame[column].astype(str).str.strip()
            if bool(normalized.eq("").any()):
                raise ValueError("Identity column %s contains empty values" % column)
            frame[column] = normalized


def require_columns(frame, columns, table_name):
    """Raise a readable error when required fields are absent."""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            "%s is missing required columns: %s"
            % (table_name, ", ".join(missing))
        )


def require_unique(frame, columns, table_name):
    """Require a table to have at most one row for each supplied key."""

    require_columns(frame, columns, table_name)
    duplicate_mask = frame.duplicated(columns, keep=False)
    if bool(duplicate_mask.any()):
        example = frame.loc[duplicate_mask, columns].head(5).to_dict("records")
        raise ValueError(
            "%s has %d rows with duplicate key %s; examples=%s"
            % (
                table_name,
                int(duplicate_mask.sum()),
                "+".join(columns),
                example,
            )
        )


def write_csv(frame, path):
    """Write a stable UTF-8 CSV after creating its parent directory."""

    ensure_directory(os.path.dirname(os.path.abspath(path)))
    frame.to_csv(path, index=False, encoding="utf-8")


def write_summary(summary, path):
    """Write a metric/value mapping as a two-column CSV."""

    rows = []
    for metric in sorted(summary.keys()):
        rows.append({"metric": metric, "value": summary[metric]})
    write_csv(pd.DataFrame(rows, columns=["metric", "value"]), path)


def empty_report(columns):
    """Return an empty report with a predictable column order."""

    return pd.DataFrame(columns=columns)


def print_summary(title, summary):
    """Print concise, grep-friendly terminal output for a pipeline step."""

    print("=== %s ===" % title)
    for key in sorted(summary.keys()):
        print("%s=%s" % (key.upper(), summary[key]))
