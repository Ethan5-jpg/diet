#!/usr/bin/env python3
"""Audit analysis-specific complete-case attrition for microbiome and CGM models."""

from pathlib import Path
import pandas as pd

MASTER = Path("outputs/data/02_covariate_master.csv")

BASE_COMMON = [
    "age_years",
    "sex",
    "education_level",
    "smoking_status",
    "sleep_duration_hours_day",
    "physical_activity_met_h_week",
    "vitamin_use",
    "hormone_use",
]

ANALYSES = {
    "AMED MICROBIOME MODEL 2": BASE_COMMON,
    "hPDI MICROBIOME MODEL 2": BASE_COMMON + ["alcohol_intake_g_day"],
    "AMED CGM MODEL 2": BASE_COMMON + ["cgm_device_type"],
    "hPDI CGM MODEL 2": BASE_COMMON + ["cgm_device_type", "alcohol_intake_g_day"],
}


def audit(df, name, columns):
    print("\n" + "=" * 100)
    print(name)
    print("=" * 100)

    full = df[columns].notna().all(axis=1)
    full_n = int(full.sum())

    print("Participants in master:", len(df))
    print("Complete cases:", full_n)

    rows = []
    for removed in columns:
        remaining = [c for c in columns if c != removed]
        n_without = int(df[remaining].notna().all(axis=1).sum())
        rows.append(
            {
                "removed_variable": removed,
                "complete_without": n_without,
                "additional_people": n_without - full_n,
            }
        )
    rescue = pd.DataFrame(rows).sort_values("additional_people", ascending=False)

    print("\nLEAVE-ONE-OUT RESCUE")
    print(rescue.to_string(index=False))

    n_missing = df[columns].isna().sum(axis=1)
    single = n_missing.eq(1)

    blockers = []
    for c in columns:
        blockers.append(
            {
                "variable": c,
                "participants_blocked_only_by_this": int((single & df[c].isna()).sum()),
            }
        )
    blockers = pd.DataFrame(blockers).sort_values(
        "participants_blocked_only_by_this", ascending=False
    )

    print("\nEXACT SINGLE-VARIABLE BLOCKERS")
    print(blockers.to_string(index=False))


def main():
    if not MASTER.exists():
        raise FileNotFoundError(MASTER)

    df = pd.read_csv(MASTER, low_memory=False)

    for name, columns in ANALYSES.items():
        missing_cols = [c for c in columns if c not in df.columns]
        if missing_cols:
            raise ValueError(f"{name}: missing columns {missing_cols}")
        audit(df, name, columns)


if __name__ == "__main__":
    main()
