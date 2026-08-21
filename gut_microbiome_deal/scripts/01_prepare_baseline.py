from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

DATA_ROOT = Path("/home/ec2-user/Desktop/HPP/Data")

SOURCE = (
    DATA_ROOT
    / "Transfer"
    / "gut_microbiome"
    / "abundance"
    / "metaphlan4"
    / "aggregated"
    / "metaphlan_abundance_species.csv"
)

PROJECT = DATA_ROOT / "gut_microbiome_deal"

OUTPUT_DATA = (
    PROJECT
    / "data"
    / "01_metaphlan_species_baseline_9087.csv"
)

OUTPUT_REPORT = (
    PROJECT
    / "reports"
    / "01_baseline_qc.txt"
)


KEY_COLS = [
    "participant_id",
    "cohort",
    "research_stage",
    "array_index",
]


# ============================================================
# Load source
# ============================================================

print("=" * 80)
print("SOURCE")
print("=" * 80)

print("Input:", SOURCE)

df = pd.read_csv(SOURCE)

species_cols = [
    c for c in df.columns
    if c not in KEY_COLS
]

results = []

def report(text=""):
    print(text)
    results.append(str(text))


report(f"Raw rows: {len(df)}")
report(
    f"Raw unique participants: "
    f"{df['participant_id'].nunique()}"
)
report(f"Total columns: {len(df.columns)}")
report(f"Species columns: {len(species_cols)}")


# ============================================================
# Select baseline
# ============================================================

baseline = df[
    df["research_stage"].eq("00_00_visit")
].copy()

report("")
report("BASELINE BEFORE DEDUPLICATION")
report(f"Rows: {len(baseline)}")
report(
    f"Unique participants: "
    f"{baseline['participant_id'].nunique()}"
)

participant_counts = (
    baseline["participant_id"]
    .value_counts()
)

report(
    "Participants with >1 baseline record: "
    f"{int((participant_counts > 1).sum())}"
)


# ============================================================
# One baseline sample per participant
#
# Prefer array_index == 0.
# ============================================================

baseline["_priority"] = (
    baseline["array_index"]
    .ne(0)
    .astype(int)
)

baseline = (
    baseline
    .sort_values(
        [
            "participant_id",
            "_priority",
            "array_index",
        ]
    )
    .drop_duplicates(
        subset="participant_id",
        keep="first",
    )
    .drop(columns="_priority")
)

report("")
report("SELECTED BASELINE")
report(f"Rows: {len(baseline)}")
report(
    f"Unique participants: "
    f"{baseline['participant_id'].nunique()}"
)
report(
    "array_index == 0: "
    f"{int(baseline['array_index'].eq(0).sum())}"
)


# ============================================================
# Audit abundance encoding
#
# IMPORTANT:
# Do not transform anything yet.
# ============================================================

X = (
    baseline[species_cols]
    .apply(pd.to_numeric, errors="coerce")
)

n_nan = int(X.isna().sum().sum())
n_zero = int((X == 0).sum().sum())
n_positive = int((X > 0).sum().sum())

report("")
report("ABUNDANCE MATRIX")
report(f"Shape: {X.shape}")
report(f"NaN cells: {n_nan}")
report(f"Zero cells: {n_zero}")
report(f"Positive cells: {n_positive}")

report(
    "Species with >=1 non-missing value: "
    f"{int(X.notna().any(axis=0).sum())}"
)

report(
    "Species with >=1 positive value: "
    f"{int((X > 0).any(axis=0).sum())}"
)

report(
    "Species all-NaN: "
    f"{int(X.isna().all(axis=0).sum())}"
)

finite = X.to_numpy(dtype=float)
finite = finite[np.isfinite(finite)]

if finite.size:
    report(f"Minimum finite value: {finite.min()}")
    report(f"Maximum finite value: {finite.max()}")

positive = finite[finite > 0]

if positive.size:
    report(f"Minimum positive value: {positive.min()}")


# ============================================================
# Checkpoint
# ============================================================

report("")
report("=" * 80)
report("PAPER CHECKPOINT")
report("=" * 80)

report("Expected participants: 9087")
report("Expected species:      2088")

report(
    f"Observed participants: "
    f"{len(baseline)}"
)

report(
    f"Observed species:      "
    f"{len(species_cols)}"
)

if len(baseline) != 9087:
    raise RuntimeError(
        f"Participant checkpoint failed: "
        f"{len(baseline)} != 9087"
    )

if len(species_cols) != 2088:
    raise RuntimeError(
        f"Species checkpoint failed: "
        f"{len(species_cols)} != 2088"
    )

report("")
report("CHECKPOINT PASSED: 9087 × 2088")


# ============================================================
# Save
# ============================================================

OUTPUT_DATA.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_REPORT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

baseline.to_csv(
    OUTPUT_DATA,
    index=False,
)

OUTPUT_REPORT.write_text(
    "\n".join(results),
    encoding="utf-8",
)

report("")
print("Saved:")
print(OUTPUT_DATA)
print(OUTPUT_REPORT)