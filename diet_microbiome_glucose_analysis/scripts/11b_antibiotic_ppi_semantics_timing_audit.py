#!/usr/bin/env python3
"""Step 11b: focused antibiotic/PPI medication-source semantics + timing audit.

Purpose
-------
This script follows Step 11 v2 and does NOT modify the covariate master.
It answers the remaining questions needed before deriving antibiotic_use/ppi_use:

1. What HPP medication `source` values contribute J01 / A02BC records?
2. Are collection_timestamp / collection_date / start_date actually populated?
3. How many target-positive participants have a medication collection time inside
   their own diet logging window?
4. How much of the apparent missing PPI timing in v2 was caused by choosing a
   single time field globally rather than coalescing collection fields per row?

Outputs are read-only audit CSVs under outputs/reports/11_antibiotic_ppi_source_audit_v2/.
"""
from __future__ import annotations

import ast
import re
import warnings
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

DATA_ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
MED_CSV = DATA_ROOT / "co-variant" / "csv" / "medications.csv"
DIET_CSV = DATA_ROOT / "diet_deal" / "outputs" / "01_daily_summary" / "diet_participant_summary.csv"
FORMAL_CSV = (
    DATA_ROOT / "diet_microbiome_glucose_analysis" / "outputs" / "data" /
    "00_current4_diet_cgm_cohort.csv"
)
OUTDIR = (
    DATA_ROOT / "diet_microbiome_glucose_analysis" / "outputs" / "reports" /
    "11_antibiotic_ppi_source_audit_v2"
)

COHORT = "10k"
STAGE = "00_00_visit"
TARGETS = {"antibiotic": "J01", "ppi": "A02BC"}
ATC_COLS = ("atc3", "atc4", "atc5")
TIME_COLS = ("collection_timestamp", "collection_date", "start_date")
OPTIONAL_COLS = ("source", "medication", "api", "start_month", "start_year")


def clean_pid(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


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


def parse_datetime_mixed(series: pd.Series) -> pd.Series:
    """Parse mixed datetime strings once, without producing thousands of warnings."""
    try:
        # pandas >=2.0: explicitly allow heterogeneous ISO/date representations.
        return pd.to_datetime(series, format="mixed", errors="coerce", utc=True)
    except (TypeError, ValueError):
        # Backward-compatible fallback for older pandas.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return pd.to_datetime(series, errors="coerce", utc=True)


def collect_codes(row: pd.Series, atc_cols: Iterable[str]) -> List[str]:
    codes: List[str] = []
    for c in atc_cols:
        codes.extend(parse_code_cell(row[c]))
    return codes


def distance_to_window(ts: pd.Timestamp, first: pd.Timestamp, last: pd.Timestamp):
    if pd.isna(ts) or pd.isna(first) or pd.isna(last):
        return np.nan
    if first <= ts <= last:
        return 0.0
    if ts < first:
        return float((first - ts).total_seconds() / 86400.0)
    return float((ts - last).total_seconds() / 86400.0)


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if not MED_CSV.is_file():
        raise FileNotFoundError(MED_CSV)
    if not DIET_CSV.is_file():
        raise FileNotFoundError(DIET_CSV)

    header = list(pd.read_csv(MED_CSV, nrows=0).columns)
    atc_cols = [c for c in ATC_COLS if c in header]
    if "participant_id" not in header or not atc_cols:
        raise ValueError(f"Medication CSV missing participant_id/ATC columns. Header={header}")

    wanted = ["participant_id"]
    for c in ("cohort", "research_stage", *atc_cols, *TIME_COLS, *OPTIONAL_COLS):
        if c in header and c not in wanted:
            wanted.append(c)

    med = pd.read_csv(MED_CSV, usecols=wanted, low_memory=False)
    med["participant_id"] = clean_pid(med["participant_id"])
    if "cohort" in med:
        med = med.loc[med["cohort"].astype(str).eq(COHORT)].copy()
    if "research_stage" in med:
        med = med.loc[med["research_stage"].astype(str).eq(STAGE)].copy()

    med["__codes"] = med.apply(lambda r: collect_codes(r, atc_cols), axis=1)
    for target, prefix in TARGETS.items():
        med[f"__{target}"] = med["__codes"].map(lambda xs: any(x.startswith(prefix) for x in xs))

    # Parse each datetime field ONCE for the full baseline table.
    unparsed_rows = []
    for c in TIME_COLS:
        if c not in med.columns:
            continue
        parsed = parse_datetime_mixed(med[c])
        med[f"__{c}"] = parsed
        raw_nonmissing = med[c].notna() & med[c].astype(str).str.strip().ne("")
        bad = raw_nonmissing & parsed.isna()
        for raw in med.loc[bad, c].astype(str).drop_duplicates().head(30):
            unparsed_rows.append({"time_field": c, "raw_value": raw})

    # Per-row collection time: timestamp first, then date. This is intentionally
    # separate from medication start_date because they have different semantics.
    med["__collection_time"] = pd.NaT
    if "__collection_timestamp" in med:
        med["__collection_time"] = med["__collection_timestamp"]
    if "__collection_date" in med:
        med["__collection_time"] = med["__collection_time"].fillna(med["__collection_date"])

    diet_header = list(pd.read_csv(DIET_CSV, nrows=0).columns)
    diet_use = [c for c in ("participant_id", "cohort", "research_stage", "first_collection_date", "last_collection_date") if c in diet_header]
    diet = pd.read_csv(DIET_CSV, usecols=diet_use, low_memory=False)
    diet["participant_id"] = clean_pid(diet["participant_id"])
    if "cohort" in diet:
        diet = diet.loc[diet["cohort"].astype(str).eq(COHORT)].copy()
    if "research_stage" in diet and STAGE in set(diet["research_stage"].dropna().astype(str)):
        diet = diet.loc[diet["research_stage"].astype(str).eq(STAGE)].copy()
    diet["diet_first"] = parse_datetime_mixed(diet["first_collection_date"])
    diet["diet_last"] = parse_datetime_mixed(diet["last_collection_date"])
    diet = diet[["participant_id", "diet_first", "diet_last"]].drop_duplicates("participant_id")

    source_rows = []
    timing_rows = []
    participant_parts = []

    for target, prefix in TARGETS.items():
        flag = f"__{target}"
        pos = med.loc[med[flag]].copy()

        # Source breakdown: row count and unique-participant count.
        if "source" in pos.columns:
            temp = pos.copy()
            temp["source"] = temp["source"].astype(object).where(temp["source"].notna(), "<missing>").astype(str)
            for source_value, g in temp.groupby("source", dropna=False):
                source_rows.append({
                    "target": target,
                    "atc_prefix": prefix,
                    "source": source_value,
                    "rows": len(g),
                    "participants": g["participant_id"].nunique(),
                })
        else:
            source_rows.append({"target": target, "atc_prefix": prefix, "source": "<column absent>", "rows": len(pos), "participants": pos["participant_id"].nunique()})

        # Merge each positive row to its diet window.
        x = pos.merge(diet, on="participant_id", how="left")
        x["collection_distance_days"] = [
            distance_to_window(t, f, l)
            for t, f, l in zip(x["__collection_time"], x["diet_first"], x["diet_last"])
        ]
        x["collection_inside_diet_window"] = x["collection_distance_days"].eq(0)

        if "__start_date" in x.columns:
            x["start_before_or_during_diet_end"] = x["__start_date"].notna() & x["diet_last"].notna() & x["__start_date"].le(x["diet_last"])
        else:
            x["start_before_or_during_diet_end"] = False

        # Participant-level any evidence. IMPORTANT: start_date alone is not
        # treated as proof of current use because no stop/end date is available.
        p_rows = []
        for pid, g in x.groupby("participant_id", sort=False):
            valid_dist = pd.to_numeric(g["collection_distance_days"], errors="coerce").dropna()
            p_rows.append({
                "participant_id": pid,
                "target": target,
                "atc_prefix": prefix,
                "positive_rows": len(g),
                "has_diet_window": bool(g["diet_first"].notna().any() and g["diet_last"].notna().any()),
                "has_parseable_collection_time": bool(g["__collection_time"].notna().any()),
                "collection_inside_diet_window": bool(g["collection_inside_diet_window"].any()),
                "collection_within_30d": bool(valid_dist.le(30).any()) if len(valid_dist) else False,
                "collection_within_90d": bool(valid_dist.le(90).any()) if len(valid_dist) else False,
                "min_collection_distance_days": float(valid_dist.min()) if len(valid_dist) else np.nan,
                "has_start_date": bool(g.get("__start_date", pd.Series(index=g.index, dtype="datetime64[ns, UTC]")).notna().any()),
                "start_before_or_during_diet_end": bool(g["start_before_or_during_diet_end"].any()),
                "sources": ";".join(sorted(set(g["source"].dropna().astype(str)))) if "source" in g.columns else "",
            })
        p = pd.DataFrame(p_rows)
        participant_parts.append(p)

        def n_true(col: str) -> int:
            return int(p[col].fillna(False).sum()) if col in p else 0

        # Field-specific nonmissing counts among positive participants.
        field_stats = {}
        for c in TIME_COLS:
            pc = f"__{c}"
            if pc in pos.columns:
                field_stats[f"participants_with_{c}"] = pos.loc[pos[pc].notna(), "participant_id"].nunique()
            else:
                field_stats[f"participants_with_{c}"] = 0

        timing_rows.append({
            "target": target,
            "atc_prefix": prefix,
            "positive_participants": pos["participant_id"].nunique(),
            "positive_rows": len(pos),
            **field_stats,
            "participants_with_diet_window": n_true("has_diet_window"),
            "participants_with_any_collection_time": n_true("has_parseable_collection_time"),
            "participants_collection_inside_diet_window": n_true("collection_inside_diet_window"),
            "participants_collection_within_30d": n_true("collection_within_30d"),
            "participants_collection_within_90d": n_true("collection_within_90d"),
            "participants_with_start_date": n_true("has_start_date"),
            "participants_start_before_or_during_diet_end": n_true("start_before_or_during_diet_end"),
            "interpretation": "collection time inside diet window = direct timing evidence; start_date is supportive only because no stop/end date is available",
        })

    source_df = pd.DataFrame(source_rows)
    timing_df = pd.DataFrame(timing_rows)
    participant_df = pd.concat(participant_parts, ignore_index=True) if participant_parts else pd.DataFrame()
    unparsed_df = pd.DataFrame(unparsed_rows)

    # Optional formal/primary cohort counts for DIRECT inside-window evidence.
    cohort_rows = []
    if FORMAL_CSV.is_file() and not participant_df.empty:
        formal_header = list(pd.read_csv(FORMAL_CSV, nrows=0).columns)
        fcols = ["participant_id"]
        if "primary_cgm_analysis_eligible" in formal_header:
            fcols.append("primary_cgm_analysis_eligible")
        formal = pd.read_csv(FORMAL_CSV, usecols=fcols, low_memory=False)
        formal["participant_id"] = clean_pid(formal["participant_id"])
        cohorts = [("formal_diet_cgm", formal[["participant_id"]].drop_duplicates())]
        if "primary_cgm_analysis_eligible" in formal.columns:
            v = formal["primary_cgm_analysis_eligible"]
            keep = v.fillna(False) if pd.api.types.is_bool_dtype(v) else v.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes"})
            cohorts.append(("primary_diet_cgm_eligible", formal.loc[keep, ["participant_id"]].drop_duplicates()))
        for cname, cdf in cohorts:
            for target in TARGETS:
                p = participant_df.loc[participant_df["target"].eq(target)]
                m = cdf.merge(p, on="participant_id", how="left")
                cohort_rows.append({
                    "cohort": cname,
                    "target": target,
                    "participants": len(cdf),
                    "any_target_atc_positive": int(m["atc_prefix"].notna().sum()),
                    "direct_collection_inside_diet_window_positive": int(m["collection_inside_diet_window"].fillna(False).sum()),
                    "collection_within_90d_positive": int(m["collection_within_90d"].fillna(False).sum()),
                    "start_before_or_during_diet_end_positive_supportive_only": int(m["start_before_or_during_diet_end"].fillna(False).sum()),
                })
    cohort_df = pd.DataFrame(cohort_rows)

    source_df.to_csv(OUTDIR / "11b_target_source_breakdown.csv", index=False)
    timing_df.to_csv(OUTDIR / "11b_target_timing_summary.csv", index=False)
    participant_df.to_csv(OUTDIR / "11b_target_participant_timing_TECHNICAL_ONLY.csv", index=False)
    unparsed_df.to_csv(OUTDIR / "11b_unparsed_datetime_examples.csv", index=False)
    cohort_df.to_csv(OUTDIR / "11b_target_timing_by_analysis_cohort.csv", index=False)

    print("\n=== STEP 11b KEY RESULTS ===")
    print(f"BASELINE_MEDICATION_ROWS={len(med)}")
    print(f"BASELINE_MEDICATION_PARTICIPANTS={med['participant_id'].nunique()}")
    print("\n--- TARGET SOURCE BREAKDOWN ---")
    print(source_df.to_string(index=False) if not source_df.empty else "<none>")
    print("\n--- TARGET TIMING SUMMARY ---")
    print(timing_df.to_string(index=False) if not timing_df.empty else "<none>")
    if not cohort_df.empty:
        print("\n--- DIRECT TIMING EVIDENCE IN ANALYSIS COHORTS ---")
        print(cohort_df.to_string(index=False))
    print("\nIMPORTANT:")
    print("- No covariate master was modified.")
    print("- collection time inside the diet window is direct timing evidence.")
    print("- start_date alone is supportive, NOT proof of current use, because there is no stop/end date here.")
    print("- Review source values before defining 0/1/unknown.")
    print(f"OUTPUT_DIR={OUTDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
