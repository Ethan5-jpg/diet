#!/usr/bin/env python3
"""Step 12a: build a NON-DESTRUCTIVE antibiotic/PPI sensitivity covariate master.

This script does not overwrite the canonical 02_covariate_master.csv.

Operational definition used here
--------------------------------
For baseline HPP medication records (cohort=10k, research_stage=00_00_visit):

    antibiotic_use = 1 if any ATC code starts with J01
                     0 if participant has >=1 baseline medication record
                       but no J01 code
                   NaN if participant has no baseline medication record

    ppi_use        = 1 if any ATC code starts with A02BC
                     0 if participant has >=1 baseline medication record
                       but no A02BC code
                   NaN if participant has no baseline medication record

This deliberately mirrors the existing three-state medication semantics used
for vitamin_use / hormone_use / NSAID / A10 in the current covariate pipeline.

The stricter "collection timestamp inside observed diet first/last window"
definition from Step 11b is NOT used here as the main variable; it remains a
separate timing sensitivity because the observed food-event span is not the
same thing as the protocol's full two-week logging period.

Outputs
-------
co-variant/outputs/data/
    02_covariate_master_antibiotic_ppi_sensitivity.csv

diet_microbiome_glucose_analysis/outputs/reports/
    12a_antibiotic_ppi_master_qc.csv
    12a_antibiotic_ppi_master_summary.txt
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


DATA_ROOT = Path("/home/ec2-user/Desktop/HPP/Data")

MASTER_CSV = (
    DATA_ROOT / "co-variant" / "outputs" / "data" / "02_covariate_master.csv"
)
MED_CSV = DATA_ROOT / "co-variant" / "csv" / "medications.csv"
FORMAL_CSV = (
    DATA_ROOT
    / "diet_microbiome_glucose_analysis"
    / "outputs"
    / "data"
    / "00_current4_diet_cgm_cohort.csv"
)

OUT_MASTER = (
    DATA_ROOT
    / "co-variant"
    / "outputs"
    / "data"
    / "02_covariate_master_antibiotic_ppi_sensitivity.csv"
)
REPORT_DIR = (
    DATA_ROOT
    / "diet_microbiome_glucose_analysis"
    / "outputs"
    / "reports"
)
OUT_QC = REPORT_DIR / "12a_antibiotic_ppi_master_qc.csv"
OUT_SUMMARY = REPORT_DIR / "12a_antibiotic_ppi_master_summary.txt"

COHORT = "10k"
STAGE = "00_00_visit"
ATC_COLS = ("atc3", "atc4", "atc5")


def clean_pid(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


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
            if isinstance(parsed, (list, tuple, set, np.ndarray)):
                values = list(parsed)
            else:
                values = [parsed]
        else:
            values = [value]

    out: List[str] = []
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


def medication_flags(med: pd.DataFrame) -> pd.DataFrame:
    work = med.copy()
    work["participant_id"] = clean_pid(work["participant_id"])

    if "cohort" in work.columns:
        work = work.loc[work["cohort"].astype(str).eq(COHORT)].copy()
    if "research_stage" in work.columns:
        work = work.loc[
            work["research_stage"].astype(str).eq(STAGE)
        ].copy()

    rows = []
    for pid, group in work.groupby("participant_id", sort=False):
        codes: List[str] = []
        for col in ATC_COLS:
            for value in group[col].tolist():
                codes.extend(parse_code_cell(value))

        rows.append(
            {
                "participant_id": pid,
                "has_baseline_medication_record_step12": True,
                "antibiotic_use": float(
                    any(code.startswith("J01") for code in codes)
                ),
                "ppi_use": float(
                    any(code.startswith("A02BC") for code in codes)
                ),
            }
        )

    return pd.DataFrame(rows)


def parse_primary_flag(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "1.0", "yes"})
    )


def qc_one(name: str, cohort: pd.DataFrame, master: pd.DataFrame) -> dict:
    x = cohort[["participant_id"]].drop_duplicates().copy()
    x["participant_id"] = clean_pid(x["participant_id"])
    x = x.merge(
        master[
            [
                "participant_id",
                "has_baseline_medication_record_step12",
                "antibiotic_use",
                "ppi_use",
            ]
        ],
        on="participant_id",
        how="left",
        validate="one_to_one",
    )

    def stats(col: str):
        s = pd.to_numeric(x[col], errors="coerce")
        return {
            f"{col}_known": int(s.notna().sum()),
            f"{col}_positive": int(s.eq(1).sum()),
            f"{col}_negative": int(s.eq(0).sum()),
            f"{col}_missing": int(s.isna().sum()),
        }

    out = {
        "cohort": name,
        "participants": len(x),
        "with_baseline_medication_record": int(
            x["has_baseline_medication_record_step12"]
            .fillna(False)
            .astype(bool)
            .sum()
        ),
    }
    out.update(stats("antibiotic_use"))
    out.update(stats("ppi_use"))
    return out


def main() -> int:
    for path in (MASTER_CSV, MED_CSV):
        if not path.is_file():
            raise FileNotFoundError(path)

    master = pd.read_csv(MASTER_CSV, low_memory=False)
    med_header = list(pd.read_csv(MED_CSV, nrows=0).columns)

    required_med = ["participant_id", *ATC_COLS]
    missing = [c for c in required_med if c not in med_header]
    if missing:
        raise ValueError(
            f"Medication CSV missing required columns: {missing}"
        )

    med_cols = [
        c
        for c in (
            "participant_id",
            "cohort",
            "research_stage",
            *ATC_COLS,
        )
        if c in med_header
    ]
    med = pd.read_csv(MED_CSV, usecols=med_cols, low_memory=False)

    if "participant_id" not in master.columns:
        raise ValueError("Covariate master has no participant_id")

    master["participant_id"] = clean_pid(master["participant_id"])

    if master["participant_id"].duplicated().any():
        raise ValueError("Covariate master participant_id is not unique")

    existing = [
        c
        for c in (
            "antibiotic_use",
            "ppi_use",
            "has_baseline_medication_record_step12",
        )
        if c in master.columns
    ]
    if existing:
        raise ValueError(
            "Refusing to silently replace existing Step12 columns: "
            + ", ".join(existing)
        )

    flags = medication_flags(med)
    flags["participant_id"] = clean_pid(flags["participant_id"])

    out = master.merge(
        flags,
        on="participant_id",
        how="left",
        validate="one_to_one",
    )
    out["has_baseline_medication_record_step12"] = (
        out["has_baseline_medication_record_step12"]
        .fillna(False)
        .astype(bool)
    )

    # IMPORTANT: do NOT fill antibiotic_use / ppi_use NaN.
    # Missing means participant has no baseline medication record.
    if len(out) != len(master):
        raise RuntimeError("Master row count changed after medication merge")
    if out["participant_id"].duplicated().any():
        raise RuntimeError("Duplicate participant_id created after merge")

    OUT_MASTER.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_MASTER, index=False)

    cohorts = [
        ("covariate_master", out[["participant_id"]].copy())
    ]

    if FORMAL_CSV.is_file():
        formal = pd.read_csv(FORMAL_CSV, low_memory=False)
        formal["participant_id"] = clean_pid(formal["participant_id"])
        cohorts.append(
            (
                "formal_diet_cgm",
                formal[["participant_id"]].drop_duplicates(),
            )
        )
        if "primary_cgm_analysis_eligible" in formal.columns:
            keep = parse_primary_flag(
                formal["primary_cgm_analysis_eligible"]
            )
            cohorts.append(
                (
                    "primary_diet_cgm_eligible",
                    formal.loc[
                        keep, ["participant_id"]
                    ].drop_duplicates(),
                )
            )

    qc_rows = [qc_one(name, cohort, out) for name, cohort in cohorts]
    qc = pd.DataFrame(qc_rows)
    qc.to_csv(OUT_QC, index=False)

    lines = [
        "=== STEP 12a ANTIBIOTIC/PPI SENSITIVITY MASTER ===",
        "CANONICAL_MASTER_MODIFIED=False",
        f"input_master={MASTER_CSV}",
        f"medication_source={MED_CSV}",
        f"output_master={OUT_MASTER}",
        "",
        "Definition:",
        "antibiotic_use: J01* in baseline medication table",
        "ppi_use: A02BC* in baseline medication table",
        "1 = target ATC present",
        "0 = baseline medication record present but target ATC absent",
        "NaN = no baseline medication record",
        "",
        qc.to_string(index=False),
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== STEP 12a KEY RESULTS ===")
    print("CANONICAL_MASTER_MODIFIED=False")
    print(f"OUTPUT_MASTER={OUT_MASTER}")
    print()
    print(qc.to_string(index=False))
    print()
    print(f"QC={OUT_QC}")
    print(f"SUMMARY={OUT_SUMMARY}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
