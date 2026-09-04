#!/usr/bin/env python3
"""Shared paths, field definitions, and validation for diet–CGM analysis."""

from __future__ import print_function

import os

import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_ROOT = os.path.abspath(os.path.join(ANALYSIS_DIR, ".."))

DEFAULT_AMED_CSV = os.path.join(
    DATA_ROOT,
    "diet_deal",
    "outputs",
    "05_diet_scores",
    "amed",
    "amed_participant_scores.csv",
)
DEFAULT_HPDI_CSV = os.path.join(
    DATA_ROOT,
    "diet_deal",
    "outputs",
    "05_diet_scores",
    "hpdi",
    "hpdi_participant_scores.csv",
)
DEFAULT_CGM_CSV = os.path.join(
    DATA_ROOT,
    "cgm_deal",
    "outputs",
    "data",
    "05_cgm_core_phenotypes.csv",
)
DEFAULT_GUT_CSV = os.path.join(
    DATA_ROOT,
    "Transfer",
    "gut_microbiome",
    "gut_microbiome.csv",
)

DEFAULT_OUTPUT_DIR = os.path.join(ANALYSIS_DIR, "outputs")
DEFAULT_DATA_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "data")
DEFAULT_REPORT_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "reports")
DEFAULT_MODEL_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "models")

BASELINE_KEY = ["participant_id", "cohort", "research_stage"]
TEXT_ID_COLUMNS = BASELINE_KEY + ["connection_id"]

AMED_EXPOSURE = "amed_energy_adjusted_score_z"
HPDI_EXPOSURE = "hpdi_score_energy_adjusted_z"
EXPOSURES = [AMED_EXPOSURE, HPDI_EXPOSURE]
PRIMARY_CGM_OUTCOMES = ["cgm_mean_z", "cgm_cv_z", "cgm_above_140_z"]


def ensure_directory(path):
    absolute = os.path.abspath(path)
    if not os.path.isdir(absolute):
        os.makedirs(absolute)
    return absolute


def read_csv_preserve_ids(path):
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
    normalize_identity(frame)
    return frame


def normalize_identity(frame):
    for column in BASELINE_KEY:
        if column not in frame.columns:
            continue
        if bool(frame[column].isna().any()):
            raise ValueError("Identity column %s contains missing values" % column)
        values = frame[column].astype(str).str.strip()
        if bool(values.eq("").any()):
            raise ValueError("Identity column %s contains empty values" % column)
        frame[column] = values


def require_columns(frame, columns, source_name):
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            "%s is missing required columns: %s"
            % (source_name, ", ".join(missing))
        )


def require_unique(frame, columns, source_name):
    require_columns(frame, columns, source_name)
    duplicate = frame.duplicated(columns, keep=False)
    if bool(duplicate.any()):
        examples = frame.loc[duplicate, columns].head(5).to_dict("records")
        raise ValueError(
            "%s has %d duplicate-key rows; examples=%s"
            % (source_name, int(duplicate.sum()), examples)
        )


def filter_baseline(frame, source_name, cohort="10k", research_stage="00_00_visit"):
    frame = frame.copy()
    normalize_identity(frame)
    require_columns(frame, BASELINE_KEY, source_name)
    filtered = frame.loc[
        frame["cohort"].eq(str(cohort))
        & frame["research_stage"].eq(str(research_stage))
    ].copy()
    if filtered.empty:
        raise ValueError(
            "%s has no rows for cohort=%s, research_stage=%s"
            % (source_name, cohort, research_stage)
        )
    return filtered


def numeric_series(frame, column, source_name, allow_missing=True):
    require_columns(frame, [column], source_name)
    original = frame[column]
    numeric = pd.to_numeric(original, errors="coerce")
    invalid = original.notna() & numeric.isna()
    if bool(invalid.any()):
        raise ValueError(
            "%s.%s has %d non-numeric values"
            % (source_name, column, int(invalid.sum()))
        )
    nonfinite = numeric.notna() & ~np.isfinite(numeric)
    if bool(nonfinite.any()):
        raise ValueError(
            "%s.%s has %d non-finite values"
            % (source_name, column, int(nonfinite.sum()))
        )
    if not allow_missing and bool(numeric.isna().any()):
        raise ValueError("%s.%s contains missing values" % (source_name, column))
    return numeric.astype(float)


def write_csv(frame, path):
    ensure_directory(os.path.dirname(os.path.abspath(path)))
    frame.to_csv(path, index=False, encoding="utf-8")


def write_summary(summary, path):
    rows = []
    for metric in sorted(summary.keys()):
        rows.append({"metric": metric, "value": summary[metric]})
    write_csv(pd.DataFrame(rows, columns=["metric", "value"]), path)


def print_summary(title, summary):
    print("=== %s ===" % title)
    for metric in sorted(summary.keys()):
        print("%s=%s" % (metric.upper(), summary[metric]))
