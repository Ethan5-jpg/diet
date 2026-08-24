#!/usr/bin/env python3
"""Copy approved HPP covariate parquets and convert them to CSV.

The script is intentionally limited to a fixed manifest. It refreshes only
those managed raw and CSV files, restores named parquet index levels as normal
columns, and verifies the copied bytes and converted row counts.
"""

from __future__ import print_function

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple, Sequence

import pandas as pd


DEFAULT_SOURCE_ROOT = Path("/home/ec2-user/studies/hpp_datasets")
DEFAULT_PROJECT_ROOT = Path("/home/ec2-user/Desktop/HPP/Data/co-variant")


class DatasetSpec(NamedTuple):
    label: str
    source_relative_path: str
    csv_filename: str
    required_columns: Sequence[str]


VISIT_INDEX_COLUMNS = (
    "participant_id",
    "cohort",
    "research_stage",
    "array_index",
)


DATASET_SPECS = (
    DatasetSpec(
        "POPULATION",
        "population/population.parquet",
        "population.csv",
        ("participant_id", "cohort"),
    ),
    DatasetSpec(
        "SOCIODEMOGRAPHICS_INITIAL_MEDICAL",
        "sociodemographics/initial_medical.parquet",
        "sociodemographics_initial_medical.csv",
        VISIT_INDEX_COLUMNS,
    ),
    DatasetSpec(
        "LIFESTYLE_AND_ENVIRONMENT",
        "lifestyle_and_environment/lifestyle_and_environment.parquet",
        "lifestyle_and_environment.csv",
        VISIT_INDEX_COLUMNS,
    ),
    DatasetSpec(
        "MEDICATIONS",
        "medications/medications.parquet",
        "medications.csv",
        VISIT_INDEX_COLUMNS,
    ),
    DatasetSpec(
        "ANTHROPOMETRICS",
        "anthropometrics/anthropometrics.parquet",
        "anthropometrics.csv",
        VISIT_INDEX_COLUMNS,
    ),
    DatasetSpec(
        "FAMILY_HISTORY_INITIAL_MEDICAL",
        "family_history/initial_medical.parquet",
        "family_history_initial_medical.csv",
        VISIT_INDEX_COLUMNS,
    ),
    DatasetSpec(
        "MEDICAL_CONDITIONS",
        "medical_conditions/medical_conditions.parquet",
        "medical_conditions.csv",
        VISIT_INDEX_COLUMNS,
    ),
    DatasetSpec(
        "CGM",
        "cgm/cgm.parquet",
        "cgm.csv",
        VISIT_INDEX_COLUMNS + ("connection_id",),
    ),
)


def sha256_file(path):
    """Return the SHA-256 digest for a file without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_sources(source_root, specs=DATASET_SPECS):
    """Fail before writing outputs if any approved source is absent."""
    source_root = Path(source_root)
    missing = [
        source_root / spec.source_relative_path
        for spec in specs
        if not (source_root / spec.source_relative_path).is_file()
    ]
    if missing:
        formatted = "\n".join("- {}".format(path) for path in missing)
        raise FileNotFoundError(
            "Required HPP source file(s) are missing:\n{}".format(formatted)
        )


def atomic_copy(source_path, target_path):
    """Refresh one managed raw file using an atomic target replacement."""
    source_path = Path(source_path)
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}-".format(target_path.name),
        suffix=".tmp",
        dir=str(target_path.parent),
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)

    try:
        shutil.copy2(str(source_path), str(temporary_path))
        os.replace(str(temporary_path), str(target_path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def restore_named_index_columns(frame):
    """Move all named pandas index levels back into ordinary columns."""
    index_names = [name for name in frame.index.names if name is not None]
    if not index_names:
        return frame

    conflicts = sorted(set(index_names).intersection(frame.columns))
    if conflicts:
        raise ValueError(
            "Index names already exist as data columns: {}".format(conflicts)
        )
    return frame.reset_index()


def atomic_write_csv(frame, target_path):
    """Write one managed CSV and replace its prior version atomically."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}-".format(target_path.name),
        suffix=".tmp",
        dir=str(target_path.parent),
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)

    try:
        frame.to_csv(str(temporary_path), index=False)
        os.replace(str(temporary_path), str(target_path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def convert_one(spec, source_root, project_root, emit=print):
    """Copy, convert, and verify one manifest entry."""
    source_path = Path(source_root) / spec.source_relative_path
    raw_path = Path(project_root) / "raw" / spec.source_relative_path
    csv_path = Path(project_root) / "csv" / spec.csv_filename

    atomic_copy(source_path, raw_path)
    source_sha256 = sha256_file(source_path)
    raw_sha256 = sha256_file(raw_path)
    if source_sha256 != raw_sha256:
        raise ValueError(
            "Raw copy checksum mismatch for {}".format(spec.label)
        )

    parquet_frame = pd.read_parquet(str(raw_path))
    output_frame = restore_named_index_columns(parquet_frame)

    missing_columns = [
        column
        for column in spec.required_columns
        if column not in output_frame.columns
    ]
    if missing_columns:
        raise ValueError(
            "{} is missing required restored columns: {}".format(
                spec.label,
                missing_columns,
            )
        )

    atomic_write_csv(output_frame, csv_path)

    csv_header = list(pd.read_csv(str(csv_path), nrows=0).columns)
    csv_rows = len(
        pd.read_csv(
            str(csv_path),
            usecols=["participant_id"],
            dtype={"participant_id": str},
        )
    )

    if csv_rows != len(output_frame):
        raise ValueError(
            "CSV row count mismatch for {}: parquet={}, csv={}".format(
                spec.label,
                len(output_frame),
                csv_rows,
            )
        )

    missing_csv_columns = [
        column for column in spec.required_columns if column not in csv_header
    ]
    if missing_csv_columns:
        raise ValueError(
            "{} CSV is missing required columns: {}".format(
                spec.label,
                missing_csv_columns,
            )
        )

    summary = {
        "label": spec.label,
        "source": str(source_path),
        "raw": str(raw_path),
        "csv": str(csv_path),
        "rows": len(output_frame),
        "columns": len(output_frame.columns),
        "sha256": source_sha256,
    }

    emit("=== {} ===".format(spec.label))
    emit("SOURCE={}".format(source_path))
    emit("RAW={}".format(raw_path))
    emit("CSV={}".format(csv_path))
    emit("ROWS={}".format(summary["rows"]))
    emit("COLUMNS={}".format(summary["columns"]))
    emit("INDEX_FIELDS_RESTORED={}".format(list(spec.required_columns)))
    emit("SHA256_MATCH=True")

    return summary


def prepare_all(source_root, project_root, specs=DATASET_SPECS, emit=print):
    """Preflight all sources, then refresh all managed raw and CSV outputs."""
    source_root = Path(source_root)
    project_root = Path(project_root)

    validate_sources(source_root, specs)
    (project_root / "raw").mkdir(parents=True, exist_ok=True)
    (project_root / "csv").mkdir(parents=True, exist_ok=True)

    summaries = []
    for spec in specs:
        summaries.append(
            convert_one(
                spec,
                source_root,
                project_root,
                emit=emit,
            )
        )

    emit("=== FINAL ===")
    emit("FILES_PROCESSED={}".format(len(summaries)))
    emit("RAW_DIR={}".format(project_root / "raw"))
    emit("CSV_DIR={}".format(project_root / "csv"))
    emit("PROCESS_COMPLETED=True")
    return summaries


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Copy the eight approved HPP covariate parquets and convert "
            "them to CSV while restoring parquet index fields."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="HPP dataset root (default: %(default)s)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="co-variant output root (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print("HPP covariate source preparation started")
    print("SOURCE_ROOT={}".format(args.source_root))
    print("PROJECT_ROOT={}".format(args.project_root))

    try:
        prepare_all(args.source_root, args.project_root)
    except Exception as exc:
        print(
            "ERROR={}: {}".format(type(exc).__name__, exc),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
