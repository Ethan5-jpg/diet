#!/usr/bin/env python3
"""Targeted Step 11 audit for antibiotic/PPI sources in HPP.

This v2 deliberately avoids the expensive full-HPP parquet scan from v1.
Default behaviour:
  1) audit the already-prepared project medication CSV if present;
  2) otherwise audit ONLY HPP medications/medications.parquet;
  3) inspect small metadata/dictionary files ONLY under HPP medications/;
  4) measure J01/A02BC evidence, medication-table coverage, and timing vs diet;
  5) write read-only audit outputs; do NOT modify covariate master.

Optional --fallback-name-search performs a filename/path-only search across the
HPP tree. It never opens unrelated parquet files.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DATA_ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
HPP_ROOT = Path("/home/ec2-user/studies/hpp_datasets")

DEFAULT_MED_CSV = DATA_ROOT / "co-variant" / "csv" / "medications.csv"
DEFAULT_MED_PARQUET = HPP_ROOT / "medications" / "medications.parquet"
DEFAULT_MED_DIR = HPP_ROOT / "medications"
DEFAULT_DIET = (
    DATA_ROOT
    / "diet_deal"
    / "outputs"
    / "01_daily_summary"
    / "diet_participant_summary.csv"
)
DEFAULT_FORMAL = (
    DATA_ROOT
    / "diet_microbiome_glucose_analysis"
    / "outputs"
    / "data"
    / "00_current4_diet_cgm_cohort.csv"
)
DEFAULT_MASTER = DATA_ROOT / "co-variant" / "outputs" / "data" / "02_covariate_master.csv"
DEFAULT_OUT = (
    DATA_ROOT
    / "diet_microbiome_glucose_analysis"
    / "outputs"
    / "reports"
    / "11_antibiotic_ppi_source_audit_v2"
)

COHORT = "10k"
BASELINE_STAGE = "00_00_visit"
ATC_COLUMNS = ("atc3", "atc4", "atc5")
KEY_COLUMNS = ("participant_id", "cohort", "research_stage", "array_index")
TIME_CANDIDATES = (
    "collection_timestamp",
    "collection_date",
    "visit_timestamp",
    "visit_date",
    "medication_start_date",
    "start_date",
    "prescription_date",
    "dispense_date",
    "medication_end_date",
    "end_date",
)

TARGETS = {
    "antibiotic": {
        "prefix": "J01",
        "keywords": (
            "antibiotic",
            "antibiotics",
            "antibacterial",
            "antibacterials",
            "j01",
        ),
    },
    "ppi": {
        "prefix": "A02BC",
        "keywords": (
            "ppi",
            "proton pump",
            "proton_pump",
            "omeprazole",
            "esomeprazole",
            "lansoprazole",
            "pantoprazole",
            "rabeprazole",
            "dexlansoprazole",
            "a02bc",
        ),
    },
}

METADATA_NAME_TERMS = (
    "metadata",
    "dictionary",
    "codebook",
    "schema",
    "readme",
    "description",
    "fields",
    "columns",
)

TEXT_EXTS = {".csv", ".tsv", ".txt", ".md", ".json", ".yaml", ".yml"}
NAME_SEARCH_TERMS = (
    "medication",
    "medicine",
    "drug",
    "prescription",
    "pharmacy",
    "antibiotic",
    "antibacterial",
    "ppi",
    "proton",
    "omeprazole",
    "esomeprazole",
    "lansoprazole",
    "pantoprazole",
    "rabeprazole",
    "atc",
)


def log(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def clean_pid(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def restore_index(frame: pd.DataFrame) -> pd.DataFrame:
    names = [x for x in frame.index.names if x is not None]
    if not names:
        return frame
    conflicts = set(names).intersection(frame.columns)
    if conflicts:
        return frame
    return frame.reset_index()


def parse_code_cell(value) -> List[str]:
    """Normalize one ATC cell into alphanumeric uppercase codes."""
    if value is None:
        return []

    if isinstance(value, (list, tuple, set, np.ndarray)):
        parsed = list(value)
    else:
        try:
            if pd.isna(value):
                return []
        except Exception:
            pass

        parsed = value
        if isinstance(value, str):
            text = value.strip()
            if not text or text.lower() in {"nan", "none", "[]"}:
                return []
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                # Handles e.g. "[J01 A02BC]", "J01,A02BC", etc.
                parsed = re.split(r"[,;|\s]+", text.strip("[](){}"))

    if isinstance(parsed, (list, tuple, set, np.ndarray)):
        values = list(parsed)
    else:
        values = [parsed]

    output: List[str] = []
    for item in values:
        if item is None:
            continue
        try:
            if pd.isna(item):
                continue
        except Exception:
            pass
        code = re.sub(r"[^A-Z0-9]", "", str(item).upper())
        if code:
            output.append(code)
    return output


def read_header(path: Path) -> List[str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return list(pd.read_csv(path, nrows=0).columns)
    if suffix == ".tsv":
        return list(pd.read_csv(path, sep="\t", nrows=0).columns)
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(str(path))
            cols = list(pf.schema_arrow.names)
            # Pandas parquet index levels may not appear in schema_arrow.names.
            md = pf.schema_arrow.metadata or {}
            pandas_md = md.get(b"pandas")
            if pandas_md:
                try:
                    info = json.loads(pandas_md.decode("utf-8"))
                    for idx in info.get("index_columns", []):
                        if isinstance(idx, str) and idx not in cols:
                            cols.insert(0, idx)
                except Exception:
                    pass
            return cols
        except Exception:
            # Safe fallback: one targeted parquet only.
            return list(restore_index(pd.read_parquet(path)).columns)
    raise ValueError(f"Unsupported table type: {path}")


def choose_medication_source(csv_path: Path, parquet_path: Path) -> Path:
    if csv_path.is_file():
        return csv_path
    if parquet_path.is_file():
        return parquet_path
    raise FileNotFoundError(
        "No medication source found. Checked:\n"
        f"- {csv_path}\n"
        f"- {parquet_path}"
    )


def read_medication_table(path: Path, wanted_columns: Sequence[str]) -> pd.DataFrame:
    """Read only the targeted medication source; never scans unrelated datasets."""
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(
            path,
            sep=sep,
            usecols=[c for c in wanted_columns if c in read_header(path)],
            low_memory=False,
        )

    if suffix == ".parquet":
        # Try a column-selective read first. If the participant keys live in a
        # pandas index and selective read becomes awkward, fall back to a full
        # read of this ONE medication parquet only.
        try:
            frame = pd.read_parquet(path, columns=list(wanted_columns))
            frame = restore_index(frame)
            missing = [c for c in wanted_columns if c not in frame.columns]
            if missing:
                raise KeyError(f"missing after selected read: {missing}")
            return frame
        except Exception as exc:
            log(
                "Selective parquet read could not restore all required columns; "
                "falling back to ONE full medication parquet read only. "
                f"Reason: {type(exc).__name__}: {exc}"
            )
            frame = restore_index(pd.read_parquet(path))
            keep = [c for c in wanted_columns if c in frame.columns]
            return frame[keep].copy()

    raise ValueError(f"Unsupported medication source: {path}")


def baseline_filter(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "cohort" in out.columns:
        out = out.loc[out["cohort"].astype(str).eq(COHORT)].copy()
    if "research_stage" in out.columns:
        out = out.loc[out["research_stage"].astype(str).eq(BASELINE_STAGE)].copy()
    return out


def add_target_flags(frame: pd.DataFrame, atc_columns: Sequence[str]) -> pd.DataFrame:
    out = frame.copy()

    def collect_codes(row) -> List[str]:
        codes: List[str] = []
        for col in atc_columns:
            codes.extend(parse_code_cell(row[col]))
        return codes

    out["__all_atc_codes"] = out.apply(collect_codes, axis=1)
    for target, spec in TARGETS.items():
        prefix = spec["prefix"]
        out[f"__{target}_positive"] = out["__all_atc_codes"].map(
            lambda codes: any(code.startswith(prefix) for code in codes)
        )
    return out


def choose_time_columns(columns: Sequence[str]) -> List[str]:
    exact = [c for c in TIME_CANDIDATES if c in columns]
    extra = [
        c
        for c in columns
        if c not in exact
        and any(term in c.lower() for term in ("timestamp", "date", "time"))
    ]
    return exact + extra


def participant_flags(
    frame: pd.DataFrame,
    atc_columns: Sequence[str],
    time_columns: Sequence[str],
) -> pd.DataFrame:
    if "participant_id" not in frame.columns:
        raise ValueError("Medication source has no participant_id after index restoration.")

    work = add_target_flags(frame, atc_columns)
    work["participant_id"] = clean_pid(work["participant_id"])

    rows = []
    for pid, group in work.groupby("participant_id", sort=False):
        row = {
            "participant_id": pid,
            "has_baseline_medication_record": True,
            "antibiotic_atc_j01_positive": bool(group["__antibiotic_positive"].any()),
            "ppi_atc_a02bc_positive": bool(group["__ppi_positive"].any()),
            "baseline_medication_rows": int(len(group)),
        }
        for target in TARGETS:
            positive = group.loc[group[f"__{target}_positive"]].copy()
            row[f"{target}_positive_rows"] = int(len(positive))
            for col in time_columns:
                parsed = pd.to_datetime(positive[col], errors="coerce", utc=True)
                row[f"{target}_{col}_min"] = (
                    parsed.min().isoformat() if parsed.notna().any() else ""
                )
                row[f"{target}_{col}_max"] = (
                    parsed.max().isoformat() if parsed.notna().any() else ""
                )
        rows.append(row)
    return pd.DataFrame(rows)


def atc_frequency_table(frame: pd.DataFrame, atc_columns: Sequence[str]) -> pd.DataFrame:
    counts = {}
    for col in atc_columns:
        for value in frame[col].tolist():
            for code in parse_code_cell(value):
                counts[(col, code)] = counts.get((col, code), 0) + 1
    rows = [
        {"atc_column": col, "code": code, "row_occurrences": n}
        for (col, code), n in counts.items()
        if code.startswith("J01") or code.startswith("A02BC")
    ]
    if not rows:
        return pd.DataFrame(columns=["atc_column", "code", "row_occurrences"])
    return pd.DataFrame(rows).sort_values(
        ["atc_column", "row_occurrences", "code"], ascending=[True, False, True]
    )


def medication_time_qc(frame: pd.DataFrame, time_columns: Sequence[str]) -> pd.DataFrame:
    rows = []
    for col in time_columns:
        parsed = pd.to_datetime(frame[col], errors="coerce", utc=True)
        rows.append(
            {
                "time_column": col,
                "nonmissing_count": int(parsed.notna().sum()),
                "nonmissing_fraction": float(parsed.notna().mean()) if len(parsed) else np.nan,
                "min": parsed.min().isoformat() if parsed.notna().any() else "",
                "max": parsed.max().isoformat() if parsed.notna().any() else "",
            }
        )
    return pd.DataFrame(rows)


def load_diet(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    header = list(pd.read_csv(path, nrows=0).columns)
    needed = [
        c
        for c in (
            "participant_id",
            "cohort",
            "research_stage",
            "first_collection_date",
            "last_collection_date",
        )
        if c in header
    ]
    d = pd.read_csv(path, usecols=needed, low_memory=False)
    if "participant_id" not in d.columns:
        return pd.DataFrame()
    d["participant_id"] = clean_pid(d["participant_id"])
    if "cohort" in d.columns:
        d = d.loc[d["cohort"].astype(str).eq(COHORT)].copy()
    if "research_stage" in d.columns:
        stages = set(d["research_stage"].dropna().astype(str).unique())
        if BASELINE_STAGE in stages:
            d = d.loc[d["research_stage"].astype(str).eq(BASELINE_STAGE)].copy()
    d["diet_first_date"] = pd.to_datetime(
        d.get("first_collection_date"), errors="coerce", utc=True
    )
    d["diet_last_date"] = pd.to_datetime(
        d.get("last_collection_date"), errors="coerce", utc=True
    )
    return d[["participant_id", "diet_first_date", "diet_last_date"]].drop_duplicates(
        "participant_id"
    )


def load_population(path: Path, eligible_only: bool = False) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    header = list(pd.read_csv(path, nrows=0).columns)
    if "participant_id" not in header:
        return pd.DataFrame()
    usecols = ["participant_id"]
    if "primary_cgm_analysis_eligible" in header:
        usecols.append("primary_cgm_analysis_eligible")
    d = pd.read_csv(path, usecols=usecols, low_memory=False)
    d["participant_id"] = clean_pid(d["participant_id"])
    if eligible_only and "primary_cgm_analysis_eligible" in d.columns:
        v = d["primary_cgm_analysis_eligible"]
        if pd.api.types.is_bool_dtype(v):
            keep = v.fillna(False)
        else:
            keep = (
                v.astype(str)
                .str.strip()
                .str.lower()
                .isin({"true", "1", "1.0", "yes"})
            )
        d = d.loc[keep].copy()
    return d[["participant_id"]].drop_duplicates()


def coverage_summary(
    flags: pd.DataFrame,
    diet: pd.DataFrame,
    formal_path: Path,
    master_path: Path,
) -> pd.DataFrame:
    cohorts: List[Tuple[str, pd.DataFrame]] = []
    if master_path.is_file():
        cohorts.append(("covariate_master", load_population(master_path)))
    if not diet.empty:
        cohorts.append(("diet_logged", diet[["participant_id"]]))
    if formal_path.is_file():
        cohorts.append(("formal_diet_cgm", load_population(formal_path, eligible_only=False)))
        cohorts.append(("primary_diet_cgm_eligible", load_population(formal_path, eligible_only=True)))

    rows = []
    keep_cols = [
        "participant_id",
        "has_baseline_medication_record",
        "antibiotic_atc_j01_positive",
        "ppi_atc_a02bc_positive",
    ]
    for name, cohort in cohorts:
        if cohort.empty:
            continue
        merged = cohort.merge(flags[keep_cols], on="participant_id", how="left")
        for col in keep_cols[1:]:
            merged[col] = merged[col].map(lambda x: bool(x) if pd.notna(x) else False)
        n = len(merged)
        med_n = int(merged["has_baseline_medication_record"].sum())
        rows.append(
            {
                "cohort": name,
                "participants": n,
                "with_baseline_medication_record": med_n,
                "medication_record_coverage": med_n / n if n else np.nan,
                "antibiotic_J01_positive": int(merged["antibiotic_atc_j01_positive"].sum()),
                "ppi_A02BC_positive": int(merged["ppi_atc_a02bc_positive"].sum()),
                "not_in_medication_table": n - med_n,
                "zero_semantics_status": (
                    "participants absent from medication table remain UNKNOWN; "
                    "participants present but target code absent are only candidate zeros until metadata semantics review"
                ),
            }
        )
    return pd.DataFrame(rows)


def first_existing_target_time_col(flags: pd.DataFrame, target: str) -> Optional[str]:
    priority = [
        "collection_timestamp",
        "collection_date",
        "visit_timestamp",
        "visit_date",
        "medication_start_date",
        "start_date",
        "prescription_date",
        "dispense_date",
    ]
    for base in priority:
        col = f"{target}_{base}_min"
        if col in flags.columns and flags[col].astype(str).str.len().gt(0).any():
            return col
    candidates = [
        c
        for c in flags.columns
        if c.startswith(f"{target}_") and c.endswith("_min")
    ]
    for col in candidates:
        if flags[col].astype(str).str.len().gt(0).any():
            return col
    return None


def time_alignment(flags: pd.DataFrame, diet: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if flags.empty or diet.empty:
        return pd.DataFrame(), pd.DataFrame()

    detail_parts = []
    summary_rows = []

    for target, flag_col in (
        ("antibiotic", "antibiotic_atc_j01_positive"),
        ("ppi", "ppi_atc_a02bc_positive"),
    ):
        time_col = first_existing_target_time_col(flags, target)
        positives = flags.loc[flags[flag_col].fillna(False)].copy()
        if time_col is None:
            summary_rows.append(
                {
                    "target": target,
                    "positive_participants": len(positives),
                    "positive_with_diet_overlap": 0,
                    "time_source": "",
                    "parseable_time": 0,
                    "within_diet_window": 0,
                    "within_30d": 0,
                    "within_90d": 0,
                    "within_180d": 0,
                    "within_365d": 0,
                    "median_distance_days": np.nan,
                    "time_semantics_status": "no usable target-positive time field found",
                }
            )
            continue

        d = positives[["participant_id", flag_col, time_col]].merge(
            diet, on="participant_id", how="inner"
        )
        d["target"] = target
        d["time_source"] = time_col.replace(f"{target}_", "").removesuffix("_min")
        d["medication_record_time"] = pd.to_datetime(d[time_col], errors="coerce", utc=True)

        def classify(row):
            med = row["medication_record_time"]
            first = row["diet_first_date"]
            last = row["diet_last_date"]
            if pd.isna(med) or pd.isna(first) or pd.isna(last):
                return pd.Series([np.nan, "missing_time"])
            if med < first:
                return pd.Series([float((first - med).days), "before_diet_window"])
            if med > last:
                return pd.Series([float((med - last).days), "after_diet_window"])
            return pd.Series([0.0, "within_diet_window"])

        d[["distance_to_diet_window_days", "time_relation"]] = d.apply(classify, axis=1)
        detail_parts.append(d)
        valid = d["distance_to_diet_window_days"].dropna()
        summary_rows.append(
            {
                "target": target,
                "positive_participants": len(positives),
                "positive_with_diet_overlap": len(d),
                "time_source": d["time_source"].iloc[0] if len(d) else "",
                "parseable_time": int(valid.shape[0]),
                "within_diet_window": int(d["time_relation"].eq("within_diet_window").sum()),
                "within_30d": int(valid.le(30).sum()),
                "within_90d": int(valid.le(90).sum()),
                "within_180d": int(valid.le(180).sum()),
                "within_365d": int(valid.le(365).sum()),
                "median_distance_days": float(valid.median()) if len(valid) else np.nan,
                "time_semantics_status": (
                    "descriptive only: visit/collection timestamp is not assumed to equal drug start/end time"
                ),
            }
        )

    detail = pd.concat(detail_parts, ignore_index=True) if detail_parts else pd.DataFrame()
    return detail, pd.DataFrame(summary_rows)


def read_small_text(path: Path, max_bytes: int) -> str:
    if path.stat().st_size > max_bytes:
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def keyword_snippets(text: str, terms: Sequence[str], radius: int = 180) -> List[str]:
    lower = text.lower()
    out = []
    seen = set()
    for term in terms:
        pos = lower.find(term.lower())
        if pos < 0:
            continue
        lo = max(0, pos - radius)
        hi = min(len(text), pos + len(term) + radius)
        snip = re.sub(r"\s+", " ", text[lo:hi]).strip()
        if snip.lower() not in seen:
            seen.add(snip.lower())
            out.append(snip)
    return out[:20]


def scan_medication_metadata(med_dir: Path, max_mb: float = 20.0) -> pd.DataFrame:
    """Inspect only small metadata-like text files under medications/."""
    rows = []
    if not med_dir.is_dir():
        return pd.DataFrame()

    terms: List[str] = []
    for spec in TARGETS.values():
        terms.extend(spec["keywords"])
    terms.extend(["atc3", "atc4", "atc5", "collection_timestamp", "current medication"])
    terms = list(dict.fromkeys(terms))

    max_bytes = int(max_mb * 1024 * 1024)
    for path in med_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in TEXT_EXTS:
            continue
        low_path = str(path).lower()
        if not any(term in low_path for term in METADATA_NAME_TERMS):
            continue
        text = read_small_text(path, max_bytes)
        if not text:
            continue
        snippets = keyword_snippets(text, terms)
        if not snippets:
            continue
        lower = text.lower()
        rows.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "antibiotic_keyword_hit": any(
                    k.lower() in lower for k in TARGETS["antibiotic"]["keywords"]
                ),
                "ppi_keyword_hit": any(
                    k.lower() in lower for k in TARGETS["ppi"]["keywords"]
                ),
                "time_keyword_hit": any(k.lower() in lower for k in TIME_CANDIDATES),
                "snippets": " || ".join(snippets),
            }
        )
    return pd.DataFrame(rows)


def filename_only_fallback(root: Path) -> pd.DataFrame:
    """Traverse names only. Never opens parquet contents."""
    rows = []
    if not root.is_dir():
        return pd.DataFrame()
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden/system-like dirs where possible.
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if name.startswith("."):
                continue
            path = Path(dirpath) / name
            low = str(path).lower()
            hits = sorted({term for term in NAME_SEARCH_TERMS if term in low})
            if hits:
                rows.append(
                    {
                        "path": str(path),
                        "extension": path.suffix.lower(),
                        "name_hits": ";".join(hits),
                    }
                )
    return pd.DataFrame(rows)


def decision_table(
    flags: pd.DataFrame,
    metadata: pd.DataFrame,
    timing: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for target, spec in TARGETS.items():
        col = "antibiotic_atc_j01_positive" if target == "antibiotic" else "ppi_atc_a02bc_positive"
        n_positive = int(flags[col].sum()) if col in flags else 0
        has_metadata = False
        if not metadata.empty:
            hit_col = f"{target}_keyword_hit"
            if hit_col in metadata.columns:
                has_metadata = bool(metadata[hit_col].fillna(False).any())

        parseable = 0
        within90 = 0
        if not timing.empty:
            row = timing.loc[timing["target"].eq(target)]
            if not row.empty:
                parseable = int(row.iloc[0]["parseable_time"])
                within90 = int(row.iloc[0]["within_90d"])

        if n_positive == 0:
            status = "NO_TARGET_ATC_POSITIVES_IN_BASELINE_MEDICATION_TABLE"
        else:
            status = "TECHNICAL_CANDIDATE_FOUND_SEMANTICS_REVIEW_REQUIRED"

        rows.append(
            {
                "target": target,
                "atc_prefix": spec["prefix"],
                "positive_participants": n_positive,
                "metadata_keyword_evidence": has_metadata,
                "positive_with_parseable_time": parseable,
                "positive_within_90d_of_diet_window": within90,
                "step11_status": status,
                "can_update_covariate_master_now": False,
                "reason_not_yet_approved": (
                    "ATC positivity is valid technical evidence, but source semantics must confirm what the medication table represents; "
                    "participants absent from the table must remain unknown; timing interpretation must be defensible."
                ),
            }
        )
    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def write_summary(
    path: Path,
    source: Path,
    schema: Sequence[str],
    baseline_rows: int,
    baseline_participants: int,
    flags: pd.DataFrame,
    time_qc: pd.DataFrame,
    coverage: pd.DataFrame,
    timing: pd.DataFrame,
    metadata: pd.DataFrame,
    decisions: pd.DataFrame,
    runtime_seconds: float,
) -> None:
    lines = [
        "=== STEP 11 ANTIBIOTIC / PPI SOURCE AUDIT V2 ===",
        "MODE: READ-ONLY / TARGETED",
        "No covariate master was modified.",
        "",
        f"Medication source: {source}",
        f"Medication columns ({len(schema)}): {', '.join(schema)}",
        f"Baseline rows: {baseline_rows}",
        f"Baseline participants: {baseline_participants}",
        f"J01 positive participants: {int(flags['antibiotic_atc_j01_positive'].sum()) if not flags.empty else 0}",
        f"A02BC positive participants: {int(flags['ppi_atc_a02bc_positive'].sum()) if not flags.empty else 0}",
        "",
        "Important semantic rule:",
        "- participant absent from medication table = UNKNOWN, not 0",
        "- participant present but target code absent = candidate 0 only; final approval requires medication-table semantics review",
        "- collection/visit timestamp is descriptive timing evidence, not automatically drug start/end time",
        "",
        "-- Medication time fields --",
    ]
    if time_qc.empty:
        lines.append("No candidate time field found.")
    else:
        for r in time_qc.itertuples(index=False):
            lines.append(
                f"{r.time_column}: nonmissing={r.nonmissing_count} ({r.nonmissing_fraction:.3f}), range={r.min} .. {r.max}"
            )

    lines.extend(["", "-- Cohort coverage --"])
    if coverage.empty:
        lines.append("No cohort coverage file available.")
    else:
        for r in coverage.itertuples(index=False):
            lines.append(
                f"{r.cohort}: N={r.participants}, med-record={r.with_baseline_medication_record} "
                f"({r.medication_record_coverage:.3f}), J01+={r.antibiotic_J01_positive}, A02BC+={r.ppi_A02BC_positive}"
            )

    lines.extend(["", "-- Diet timing alignment (positive participants only) --"])
    if timing.empty:
        lines.append("No usable timing alignment available.")
    else:
        for r in timing.itertuples(index=False):
            lines.append(
                f"{r.target}: positives={r.positive_participants}, diet-overlap={r.positive_with_diet_overlap}, "
                f"time={r.time_source}, parseable={r.parseable_time}, within-window={r.within_diet_window}, "
                f"<=90d={r.within_90d}, median-distance={r.median_distance_days} days"
            )

    lines.extend(["", "-- Metadata evidence under HPP medications/ only --"])
    lines.append(f"Metadata hit files: {len(metadata)}")
    if not metadata.empty:
        for r in metadata.head(20).itertuples(index=False):
            lines.append(f"- {r.path}")

    lines.extend(["", "-- Step 11 decision --"])
    for r in decisions.itertuples(index=False):
        lines.append(
            f"{r.target.upper()} ({r.atc_prefix}): {r.step11_status}; positive N={r.positive_participants}; "
            f"can_update_master={r.can_update_covariate_master_now}"
        )

    lines.extend(
        [
            "",
            f"Runtime: {runtime_seconds:.1f} seconds",
            "This v2 did NOT open unrelated ECG/CGM/metabolomics/etc parquet files.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--medication-csv", default=str(DEFAULT_MED_CSV))
    p.add_argument("--medication-parquet", default=str(DEFAULT_MED_PARQUET))
    p.add_argument("--medication-dir", default=str(DEFAULT_MED_DIR))
    p.add_argument("--diet-summary", default=str(DEFAULT_DIET))
    p.add_argument("--formal-cohort", default=str(DEFAULT_FORMAL))
    p.add_argument("--covariate-master", default=str(DEFAULT_MASTER))
    p.add_argument("--output-dir", default=str(DEFAULT_OUT))
    p.add_argument(
        "--fallback-name-search",
        action="store_true",
        help=(
            "Optional filename/path-only search across HPP root if targeted medication audit is insufficient. "
            "It never opens unrelated parquet files."
        ),
    )
    p.add_argument("--hpp-root", default=str(HPP_ROOT))
    p.add_argument("--metadata-max-mb", type=float, default=20.0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    start = time.time()
    args = parse_args(argv)

    med_csv = Path(args.medication_csv)
    med_parquet = Path(args.medication_parquet)
    med_dir = Path(args.medication_dir)
    diet_path = Path(args.diet_summary)
    formal_path = Path(args.formal_cohort)
    master_path = Path(args.covariate_master)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    log("STEP 11 v2 starting: targeted medication audit only")
    log("No full-HPP parquet scan will be performed")

    source = choose_medication_source(med_csv, med_parquet)
    log(f"Medication source selected: {source}")

    log("Reading medication schema/header")
    schema = read_header(source)
    atc_cols = [c for c in ATC_COLUMNS if c in schema]
    time_cols = choose_time_columns(schema)

    if "participant_id" not in schema and source.suffix.lower() != ".parquet":
        raise ValueError(f"participant_id missing from medication source: {source}")
    if not atc_cols:
        raise ValueError(
            "No atc3/atc4/atc5 columns found in targeted medication source. "
            f"Columns: {schema}"
        )

    wanted = list(dict.fromkeys([*KEY_COLUMNS, *atc_cols, *time_cols]))
    log(f"Reading targeted medication columns only: {wanted}")
    med = read_medication_table(source, wanted)
    med = restore_index(med)
    missing_keys = [c for c in ("participant_id",) if c not in med.columns]
    if missing_keys:
        raise ValueError(f"Medication source missing required key(s): {missing_keys}")

    log(f"Medication rows loaded: {len(med):,}")
    baseline = baseline_filter(med)
    baseline["participant_id"] = clean_pid(baseline["participant_id"])
    log(
        f"Baseline filtered: rows={len(baseline):,}, participants={baseline['participant_id'].nunique():,}"
    )

    log("Parsing ATC codes and building TECHNICAL-ONLY participant flags")
    flags = participant_flags(baseline, atc_cols, time_cols)
    j01_n = int(flags["antibiotic_atc_j01_positive"].sum()) if not flags.empty else 0
    a02bc_n = int(flags["ppi_atc_a02bc_positive"].sum()) if not flags.empty else 0
    log(f"ATC positives: J01={j01_n:,}, A02BC={a02bc_n:,}")

    log("Building target ATC frequency table")
    atc_freq = atc_frequency_table(baseline, atc_cols)
    time_qc = medication_time_qc(baseline, time_cols)

    log("Loading diet window and cohort files for coverage/timing audit")
    diet = load_diet(diet_path)
    coverage = coverage_summary(flags, diet, formal_path, master_path)
    timing_detail, timing_summary = time_alignment(flags, diet)

    log("Scanning metadata/dictionary files under HPP medications/ only")
    metadata = scan_medication_metadata(med_dir, max_mb=args.metadata_max_mb)

    fallback = pd.DataFrame()
    if args.fallback_name_search:
        log("Optional fallback enabled: filename/path-only search across HPP root")
        log("Unrelated parquet contents will NOT be opened")
        fallback = filename_only_fallback(Path(args.hpp_root))
        log(f"Filename/path candidates found: {len(fallback):,}")

    decisions = decision_table(flags, metadata, timing_summary, coverage)

    log("Writing audit outputs")
    write_csv(
        pd.DataFrame(
            [
                {
                    "selected_source": str(source),
                    "source_type": source.suffix.lower(),
                    "all_columns": ";".join(schema),
                    "atc_columns": ";".join(atc_cols),
                    "time_columns": ";".join(time_cols),
                    "rows_loaded": len(med),
                    "baseline_rows": len(baseline),
                    "baseline_participants": baseline["participant_id"].nunique(),
                    "J01_positive_participants": j01_n,
                    "A02BC_positive_participants": a02bc_n,
                }
            ]
        ),
        outdir / "11v2_medication_source_summary.csv",
    )
    write_csv(atc_freq, outdir / "11v2_target_atc_code_frequencies.csv")
    write_csv(flags, outdir / "11v2_participant_flags_TECHNICAL_ONLY.csv")
    write_csv(time_qc, outdir / "11v2_medication_time_field_qc.csv")
    write_csv(coverage, outdir / "11v2_cohort_medication_coverage.csv")
    write_csv(timing_detail, outdir / "11v2_diet_time_alignment_detail_TECHNICAL_ONLY.csv")
    write_csv(timing_summary, outdir / "11v2_diet_time_alignment_summary.csv")
    write_csv(metadata, outdir / "11v2_medication_metadata_hits.csv")
    write_csv(decisions, outdir / "11v2_source_decision_summary.csv")
    if args.fallback_name_search:
        write_csv(fallback, outdir / "11v2_fallback_filename_candidates.csv")

    runtime = time.time() - start
    summary_path = outdir / "11v2_audit_summary.txt"
    write_summary(
        summary_path,
        source,
        schema,
        len(baseline),
        baseline["participant_id"].nunique(),
        flags,
        time_qc,
        coverage,
        timing_summary,
        metadata,
        decisions,
        runtime,
    )

    print("\n=== STEP 11 V2 KEY RESULTS ===", flush=True)
    print(f"SOURCE={source}", flush=True)
    print(f"BASELINE_ROWS={len(baseline)}", flush=True)
    print(f"BASELINE_PARTICIPANTS={baseline['participant_id'].nunique()}", flush=True)
    print(f"J01_POSITIVE_PARTICIPANTS={j01_n}", flush=True)
    print(f"A02BC_POSITIVE_PARTICIPANTS={a02bc_n}", flush=True)
    if not coverage.empty:
        print("\n--- COHORT COVERAGE ---", flush=True)
        print(coverage.to_string(index=False), flush=True)
    if not timing_summary.empty:
        print("\n--- TARGET TIMING VS DIET ---", flush=True)
        print(timing_summary.to_string(index=False), flush=True)
    print("\n--- STEP 11 DECISION ---", flush=True)
    print(decisions.to_string(index=False), flush=True)
    print(f"\nSUMMARY={summary_path}", flush=True)
    print(f"RUNTIME_SECONDS={runtime:.1f}", flush=True)
    print("MASTER_MODIFIED=False", flush=True)
    log("STEP 11 v2 finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
