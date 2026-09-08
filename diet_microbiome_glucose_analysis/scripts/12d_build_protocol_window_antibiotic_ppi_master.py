#!/usr/bin/env python3
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
HPP_ROOT = Path("/home/ec2-user/studies/hpp_datasets")

CANONICAL_MASTER = ROOT / "co-variant" / "outputs" / "data" / "02_covariate_master.csv"
MED_CSV = ROOT / "co-variant" / "csv" / "medications.csv"
EVENTS_PARQUET = HPP_ROOT / "events" / "events.parquet"
CGM_SELECTED = ROOT / "cgm_deal" / "outputs" / "data" / "01_baseline_cgm_connections.csv"
FORMAL_COHORT = (
    ROOT / "diet_microbiome_glucose_analysis" / "outputs" / "data"
    / "00_current4_diet_cgm_cohort.csv"
)

OUT_MASTER = (
    ROOT / "co-variant" / "outputs" / "data"
    / "02_covariate_master_antibiotic_ppi_during_logging.csv"
)
REPORT_DIR = ROOT / "diet_microbiome_glucose_analysis" / "outputs" / "reports"
OUT_WINDOW_QC = REPORT_DIR / "12d_protocol_window_source_qc.csv"
OUT_TIMING = REPORT_DIR / "12d_protocol_window_target_timing.csv"
OUT_SOURCE = REPORT_DIR / "12d_protocol_window_source_breakdown.csv"
OUT_MASTER_QC = REPORT_DIR / "12d_protocol_window_master_qc.csv"
OUT_FLAGS = REPORT_DIR / "12d_protocol_window_participant_flags_TECHNICAL.csv"
OUT_SUMMARY = REPORT_DIR / "12d_protocol_window_summary.txt"

COHORT = "10k"
STAGE = "00_00_visit"
ATC_COLS = ("atc3", "atc4", "atc5")
WINDOW_DAYS = 14


def clean_pid(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def restore_index(frame: pd.DataFrame) -> pd.DataFrame:
    names = [x for x in frame.index.names if x is not None]
    if not names:
        return frame
    if set(names).intersection(frame.columns):
        return frame
    return frame.reset_index()


def parse_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def parse_code_cell(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, np.ndarray)):
        values = list(value)
    else:
        try:
            if pd.isna(value):
                return []
        except Exception:
            pass
        if isinstance(value, str):
            text = value.strip()
            if not text or text.lower() in {"nan", "none", "[]"}:
                return []
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = re.split(r"[,;|\s]+", text.strip("[](){}"))
            values = list(parsed) if isinstance(parsed, (list, tuple, set, np.ndarray)) else [parsed]
        else:
            values = [value]

    out = []
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
            out.append(code)
    return out


def has_prefix(row: pd.Series, prefix: str) -> bool:
    codes = []
    for col in ATC_COLS:
        if col in row.index:
            codes.extend(parse_code_cell(row[col]))
    return any(code.startswith(prefix) for code in codes)


def filter_baseline(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "cohort" in out.columns:
        out = out.loc[out["cohort"].astype(str).eq(COHORT)].copy()
    if "research_stage" in out.columns:
        out = out.loc[out["research_stage"].astype(str).eq(STAGE)].copy()
    return out


def build_windows(master: pd.DataFrame):
    ids = master[["participant_id"]].copy()
    ids["participant_id"] = clean_pid(ids["participant_id"])

    if EVENTS_PARQUET.is_file():
        events = restore_index(pd.read_parquet(EVENTS_PARQUET))
        if "participant_id" not in events.columns:
            raise ValueError("events.parquet has no participant_id after index restoration")
        events["participant_id"] = clean_pid(events["participant_id"])
        events = filter_baseline(events)

        if "research_stage_timestamp" in events.columns:
            events["visit_timestamp"] = parse_dt(events["research_stage_timestamp"])
        elif "research_stage_date" in events.columns:
            events["visit_timestamp"] = parse_dt(events["research_stage_date"])
        else:
            raise ValueError("events.parquet has neither research_stage_timestamp nor research_stage_date")

        if "research_stage_type" in events.columns:
            events["_visit_rank"] = (
                events["research_stage_type"].astype(str).str.lower().str.contains("visit", na=False)
            ).map({True: 0, False: 1})
        else:
            events["_visit_rank"] = 0

        events["_missing_time"] = events["visit_timestamp"].isna().astype(int)
        events = events.sort_values(
            ["participant_id", "_visit_rank", "_missing_time", "visit_timestamp"],
            kind="mergesort",
        )
        event_piece = events.drop_duplicates("participant_id", keep="first")[
            ["participant_id", "visit_timestamp"]
        ]
        event_source = str(EVENTS_PARQUET)
    else:
        event_piece = pd.DataFrame(columns=["participant_id", "visit_timestamp"])
        event_source = "<events.parquet not found>"

    cgm_piece = pd.DataFrame(columns=["participant_id", "cgm_timestamp"])
    if CGM_SELECTED.is_file():
        header = list(pd.read_csv(CGM_SELECTED, nrows=0).columns)
        if {"participant_id", "collection_timestamp"}.issubset(header):
            cgm = pd.read_csv(
                CGM_SELECTED,
                usecols=["participant_id", "collection_timestamp"],
                low_memory=False,
            )
            cgm["participant_id"] = clean_pid(cgm["participant_id"])
            cgm["cgm_timestamp"] = parse_dt(cgm["collection_timestamp"])
            cgm_piece = cgm[["participant_id", "cgm_timestamp"]].drop_duplicates("participant_id")
    elif "cgm_collection_timestamp" in master.columns:
        cgm_piece = master[["participant_id", "cgm_collection_timestamp"]].copy()
        cgm_piece["cgm_timestamp"] = parse_dt(cgm_piece["cgm_collection_timestamp"])
        cgm_piece = cgm_piece[["participant_id", "cgm_timestamp"]]

    windows = (
        ids.merge(event_piece, on="participant_id", how="left", validate="one_to_one")
        .merge(cgm_piece, on="participant_id", how="left", validate="one_to_one")
    )

    windows["window_start"] = windows["visit_timestamp"].where(
        windows["visit_timestamp"].notna(), windows["cgm_timestamp"]
    )
    windows["window_source"] = np.select(
        [
            windows["visit_timestamp"].notna(),
            windows["visit_timestamp"].isna() & windows["cgm_timestamp"].notna(),
        ],
        ["events_baseline_visit", "selected_baseline_cgm_fallback"],
        default="missing",
    )
    windows["window_end"] = windows["window_start"] + pd.Timedelta(days=WINDOW_DAYS)
    windows["previsit_start"] = windows["window_start"] - pd.Timedelta(days=1)
    # FINAL author-like dietary logging window:
    # [baseline visit - 1 day, baseline visit + 14 days)
    windows["logging_window_start"] = windows["previsit_start"]
    windows["logging_window_end"] = windows["window_end"]

    both = windows["visit_timestamp"].notna() & windows["cgm_timestamp"].notna()
    windows["cgm_minus_visit_hours"] = np.nan
    windows.loc[both, "cgm_minus_visit_hours"] = (
        (windows.loc[both, "cgm_timestamp"] - windows.loc[both, "visit_timestamp"])
        .dt.total_seconds()
        .div(3600.0)
    )

    qc = [
        {"metric": "events_source", "value": event_source},
        {"metric": "participants", "value": len(windows)},
        {"metric": "visit_timestamp_available", "value": int(windows["visit_timestamp"].notna().sum())},
        {"metric": "cgm_timestamp_available", "value": int(windows["cgm_timestamp"].notna().sum())},
        {"metric": "protocol_window_available", "value": int(windows["logging_window_start"].notna().sum())},
        {"metric": "event_primary_windows", "value": int(windows["window_source"].eq("events_baseline_visit").sum())},
        {"metric": "cgm_fallback_windows", "value": int(windows["window_source"].eq("selected_baseline_cgm_fallback").sum())},
        {"metric": "missing_windows", "value": int(windows["window_source"].eq("missing").sum())},
    ]
    delta = windows.loc[both, "cgm_minus_visit_hours"].dropna()
    if len(delta):
        qc += [
            {"metric": "event_vs_cgm_median_abs_hours", "value": float(delta.abs().median())},
            {"metric": "event_vs_cgm_within_24h_fraction", "value": float(delta.abs().le(24).mean())},
            {"metric": "event_vs_cgm_within_72h_fraction", "value": float(delta.abs().le(72).mean())},
            {"metric": "event_vs_cgm_within_168h_fraction", "value": float(delta.abs().le(168).mean())},
        ]
    return windows, pd.DataFrame(qc)


def read_medications() -> pd.DataFrame:
    if not MED_CSV.is_file():
        raise FileNotFoundError(MED_CSV)

    header = list(pd.read_csv(MED_CSV, nrows=0).columns)
    required = ["participant_id", *ATC_COLS]
    missing = [c for c in required if c not in header]
    if missing:
        raise ValueError(f"Medication CSV missing required columns: {missing}")

    wanted = [
        c for c in (
            "participant_id", "cohort", "research_stage", "array_index",
            *ATC_COLS, "collection_timestamp", "collection_date", "start_date", "source"
        )
        if c in header
    ]
    med = pd.read_csv(MED_CSV, usecols=wanted, low_memory=False)
    med["participant_id"] = clean_pid(med["participant_id"])
    med = filter_baseline(med)

    med["_ts"] = parse_dt(med["collection_timestamp"]) if "collection_timestamp" in med.columns else pd.NaT
    med["_date"] = parse_dt(med["collection_date"]) if "collection_date" in med.columns else pd.NaT
    med["medication_record_time"] = med["_ts"].where(med["_ts"].notna(), med["_date"])
    med["antibiotic_target"] = med.apply(lambda r: has_prefix(r, "J01"), axis=1)
    med["ppi_target"] = med.apply(lambda r: has_prefix(r, "A02BC"), axis=1)
    return med


def classify_targets(med: pd.DataFrame, windows: pd.DataFrame):
    work = med.merge(
        windows[["participant_id", "window_start", "window_end", "previsit_start", "window_source"]],
        on="participant_id", how="left", validate="many_to_one",
    )
    t = work["medication_record_time"]
    start = work["window_start"]
    end = work["window_end"]
    pre = work["previsit_start"]

    work["time_relation"] = np.select(
        [
            t.isna() | start.isna(),
            t.lt(pre),
            t.ge(pre) & t.lt(start),
            t.ge(start) & t.lt(end),
            t.ge(end),
        ],
        [
            "missing_time_or_window",
            "before_protocol_window",
            "previsit_1d_component",
            "postvisit_14d_component",
            "after_protocol_window",
        ],
        default="unclassified",
    )

    target_rows = pd.concat(
        [
            work.loc[work["antibiotic_target"]].assign(target="antibiotic"),
            work.loc[work["ppi_target"]].assign(target="ppi"),
        ],
        ignore_index=True,
    )

    timing = (
        target_rows.groupby(["target", "time_relation"], dropna=False)
        .agg(rows=("participant_id", "size"), participants=("participant_id", "nunique"))
        .reset_index()
        .sort_values(["target", "time_relation"])
        if len(target_rows)
        else pd.DataFrame()
    )

    during = target_rows.loc[
        target_rows["time_relation"].isin(
            ["previsit_1d_component", "postvisit_14d_component"]
        )
    ].copy()
    if len(during):
        if "source" not in during.columns:
            during["source"] = "<column absent>"
        during["source"] = during["source"].astype(object).where(during["source"].notna(), "<missing>")
        source = (
            during.groupby(["target", "source"], dropna=False)
            .agg(rows=("participant_id", "size"), participants=("participant_id", "nunique"))
            .reset_index()
            .sort_values(["target", "participants"], ascending=[True, False])
        )
    else:
        source = pd.DataFrame()
    return target_rows, timing, source


def build_flags(master: pd.DataFrame, med: pd.DataFrame, windows: pd.DataFrame, target_rows: pd.DataFrame):
    ids = master[["participant_id"]].copy()
    ids["participant_id"] = clean_pid(ids["participant_id"])

    med_presence = med.groupby("participant_id").size().rename("baseline_medication_rows").reset_index()
    med_presence["has_baseline_medication_data"] = True

    flags = (
        ids.merge(
            windows[[
                "participant_id", "visit_timestamp", "cgm_timestamp", "window_start",
                "window_end", "previsit_start", "logging_window_start",
                "logging_window_end", "window_source", "cgm_minus_visit_hours"
            ]],
            on="participant_id", how="left", validate="one_to_one",
        )
        .merge(med_presence, on="participant_id", how="left", validate="one_to_one")
    )
    flags["has_baseline_medication_data"] = flags["has_baseline_medication_data"].fillna(False).astype(bool)
    flags["has_protocol_window"] = flags["logging_window_start"].notna()

    for target in ("antibiotic", "ppi"):
        tr = target_rows.loc[target_rows["target"].eq(target)].copy() if len(target_rows) else pd.DataFrame()
        broad_ids = set(tr["participant_id"].astype(str)) if len(tr) else set()
        during_ids = set(
            tr.loc[
                tr["time_relation"].isin(
                    ["previsit_1d_component", "postvisit_14d_component"]
                ),
                "participant_id",
            ].astype(str)
        ) if len(tr) else set()
        previsit_ids = set(
            tr.loc[
                tr["time_relation"].eq("previsit_1d_component"),
                "participant_id",
            ].astype(str)
        ) if len(tr) else set()
        after_ids = set(tr.loc[tr["time_relation"].eq("after_protocol_window"), "participant_id"].astype(str)) if len(tr) else set()
        before_ids = set(tr.loc[tr["time_relation"].eq("before_protocol_window"), "participant_id"].astype(str)) if len(tr) else set()
        if len(tr) and "source" in tr.columns:
            app_ids = set(
                tr.loc[
                    tr["time_relation"].isin(
                        ["previsit_1d_component", "postvisit_14d_component"]
                    )
                    & tr["source"].astype(str).str.lower().eq("app"),
                    "participant_id",
                ].astype(str)
            )
        else:
            app_ids = set()

        flags[f"{target}_broad_baseline_positive"] = flags["participant_id"].isin(broad_ids)
        flags[f"{target}_during_logging_positive"] = flags["participant_id"].isin(during_ids)
        flags[f"{target}_previsit_1d_positive_qc"] = flags["participant_id"].isin(previsit_ids)
        flags[f"{target}_after_window_positive_qc"] = flags["participant_id"].isin(after_ids)
        flags[f"{target}_before_previsit_positive_qc"] = flags["participant_id"].isin(before_ids)
        flags[f"{target}_app_during_logging_positive_qc"] = flags["participant_id"].isin(app_ids)

        out_col = "antibiotic_use" if target == "antibiotic" else "ppi_use"
        flags[out_col] = np.nan
        known = flags["has_baseline_medication_data"] & flags["has_protocol_window"]
        flags.loc[known, out_col] = 0.0
        flags.loc[known & flags[f"{target}_during_logging_positive"], out_col] = 1.0

    return flags


def parse_primary_flag(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes"})


def qc_cohort(name: str, cohort: pd.DataFrame, flags: pd.DataFrame) -> dict:
    x = cohort[["participant_id"]].drop_duplicates().copy()
    x["participant_id"] = clean_pid(x["participant_id"])
    x = x.merge(
        flags[[
            "participant_id", "has_baseline_medication_data", "has_protocol_window",
            "window_source", "antibiotic_use", "ppi_use",
            "antibiotic_broad_baseline_positive", "ppi_broad_baseline_positive",
            "antibiotic_previsit_1d_positive_qc", "ppi_previsit_1d_positive_qc",
            "antibiotic_app_during_logging_positive_qc", "ppi_app_during_logging_positive_qc",
        ]],
        on="participant_id", how="left", validate="one_to_one",
    )

    row = {
        "cohort": name,
        "participants": len(x),
        "medication_data_available": int(x["has_baseline_medication_data"].fillna(False).sum()),
        "protocol_window_available": int(x["has_protocol_window"].fillna(False).sum()),
        "event_primary_window": int(x["window_source"].eq("events_baseline_visit").sum()),
        "cgm_fallback_window": int(x["window_source"].eq("selected_baseline_cgm_fallback").sum()),
    }
    for target, col in (("antibiotic", "antibiotic_use"), ("ppi", "ppi_use")):
        s = pd.to_numeric(x[col], errors="coerce")
        row[f"{target}_known"] = int(s.notna().sum())
        row[f"{target}_positive_during_logging"] = int(s.eq(1).sum())
        row[f"{target}_negative_during_logging"] = int(s.eq(0).sum())
        row[f"{target}_unknown"] = int(s.isna().sum())
        row[f"{target}_broad_baseline_positive"] = int(x[f"{target}_broad_baseline_positive"].fillna(False).sum())
        row[f"{target}_previsit_1d_positive_qc"] = int(x[f"{target}_previsit_1d_positive_qc"].fillna(False).sum())
        row[f"{target}_app_during_logging_positive_qc"] = int(x[f"{target}_app_during_logging_positive_qc"].fillna(False).sum())
    return row


def main() -> int:
    if not CANONICAL_MASTER.is_file():
        raise FileNotFoundError(CANONICAL_MASTER)

    master = pd.read_csv(CANONICAL_MASTER, low_memory=False)
    if "participant_id" not in master.columns:
        raise ValueError("Canonical master has no participant_id")
    master["participant_id"] = clean_pid(master["participant_id"])
    if master["participant_id"].duplicated().any():
        raise ValueError("Canonical master participant_id is not unique")
    existing = [c for c in ("antibiotic_use", "ppi_use") if c in master.columns]
    if existing:
        raise RuntimeError(f"Refusing to replace existing columns in canonical master: {existing}")

    windows, window_qc = build_windows(master)
    med = read_medications()
    target_rows, timing, source = classify_targets(med, windows)
    flags = build_flags(master, med, windows, target_rows)

    out = master.merge(
        flags[[
            "participant_id", "antibiotic_use", "ppi_use",
            "has_baseline_medication_data", "has_protocol_window",
            "window_source", "logging_window_start", "logging_window_end",
        ]],
        on="participant_id", how="left", validate="one_to_one",
    )

    if len(out) != len(master):
        raise RuntimeError("Master row count changed")
    if out["participant_id"].duplicated().any():
        raise RuntimeError("Duplicate participant_id created")

    OUT_MASTER.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    out.to_csv(OUT_MASTER, index=False)
    window_qc.to_csv(OUT_WINDOW_QC, index=False)
    timing.to_csv(OUT_TIMING, index=False)
    source.to_csv(OUT_SOURCE, index=False)
    flags.to_csv(OUT_FLAGS, index=False)

    cohorts = [("covariate_master", master[["participant_id"]].copy())]
    if FORMAL_COHORT.is_file():
        formal = pd.read_csv(FORMAL_COHORT, low_memory=False)
        formal["participant_id"] = clean_pid(formal["participant_id"])
        cohorts.append(("formal_diet_cgm", formal[["participant_id"]].drop_duplicates()))
        if "primary_cgm_analysis_eligible" in formal.columns:
            keep = parse_primary_flag(formal["primary_cgm_analysis_eligible"])
            cohorts.append(("primary_diet_cgm_eligible", formal.loc[keep, ["participant_id"]].drop_duplicates()))

    qc = pd.DataFrame([qc_cohort(name, cohort, flags) for name, cohort in cohorts])
    qc.to_csv(OUT_MASTER_QC, index=False)

    lines = [
        "=== STEP 12d PROTOCOL-WINDOW ANTIBIOTIC/PPI MASTER ===",
        "CANONICAL_MASTER_MODIFIED=False",
        "Primary window = [baseline visit - 1 day, baseline visit + 14 days)",
        "Fallback = selected baseline CGM timestamp",
        "1 = target ATC in window; 0 = medication data + window available but target absent; NaN otherwise",
        "The pre-visit one-day component is INCLUDED in the primary during-logging exposure.",
        "",
        "-- WINDOW QC --",
        window_qc.to_string(index=False),
        "",
        "-- TARGET TIMING --",
        timing.to_string(index=False) if len(timing) else "<none>",
        "",
        "-- SOURCE BREAKDOWN --",
        source.to_string(index=False) if len(source) else "<none>",
        "",
        "-- COHORT QC --",
        qc.to_string(index=False),
        "",
        f"OUTPUT_MASTER={OUT_MASTER}",
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== STEP 12d KEY RESULTS ===")
    print("CANONICAL_MASTER_MODIFIED=False")
    print(f"OUTPUT_MASTER={OUT_MASTER}")
    print("\n--- WINDOW SOURCE QC ---")
    print(window_qc.to_string(index=False))
    print("\n--- TARGET TIMING ---")
    print(timing.to_string(index=False) if len(timing) else "<none>")
    print("\n--- DURING-WINDOW SOURCE BREAKDOWN ---")
    print(source.to_string(index=False) if len(source) else "<none>")
    print("\n--- COHORT QC ---")
    print(qc.to_string(index=False))
    print(f"\nSUMMARY={OUT_SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
