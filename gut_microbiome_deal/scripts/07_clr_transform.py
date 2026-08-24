from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
PROJECT = ROOT / "gut_microbiome_deal"

INPUT = (
    PROJECT
    / "data"
    / "05_species_half_min_imputed.csv"
)

OUTPUT = (
    PROJECT
    / "data"
    / "07_species_clr.csv"
)

REPORT = (
    PROJECT
    / "reports"
    / "07_clr_qc.txt"
)

KEY_COLS = [
    "participant_id",
    "cohort",
    "research_stage",
    "array_index",
]


# ============================================================
# Load
# ============================================================

df = pd.read_csv(INPUT)

species_cols = [
    c for c in df.columns
    if c not in KEY_COLS
]

X = df[species_cols].apply(
    pd.to_numeric,
    errors="coerce",
).to_numpy(dtype=np.float64)


# ============================================================
# QC before CLR
# ============================================================

if X.shape != (9074, 379):
    raise RuntimeError(
        f"Unexpected shape: {X.shape}"
    )

if not np.isfinite(X).all():
    raise RuntimeError(
        "NaN or infinity exists before CLR."
    )

if (X <= 0).any():
    raise RuntimeError(
        "CLR requires strictly positive values."
    )


# ============================================================
# CLR
# ============================================================

log_X = np.log(X)

row_log_mean = log_X.mean(
    axis=1,
    keepdims=True,
)

clr = log_X - row_log_mean


# ============================================================
# Fundamental CLR QC:
# each row should sum approximately to zero
# ============================================================

row_sum = clr.sum(axis=1)

max_abs_row_sum = np.max(
    np.abs(row_sum)
)

print("=" * 80)
print("CLR")
print("=" * 80)

print("Input shape:", X.shape)
print("Output shape:", clr.shape)

print(
    "Maximum absolute CLR row sum:",
    max_abs_row_sum,
)

if max_abs_row_sum > 1e-8:
    raise RuntimeError(
        "CLR row sums are not approximately zero."
    )


# ============================================================
# Save
# ============================================================

clr_df = pd.DataFrame(
    clr,
    columns=species_cols,
)

result = pd.concat(
    [
        df[KEY_COLS].reset_index(drop=True),
        clr_df,
    ],
    axis=1,
)

result.to_csv(
    OUTPUT,
    index=False,
)


# ============================================================
# Report
# ============================================================

REPORT.write_text(
    "\n".join([
        f"Samples: {clr.shape[0]}",
        f"Species: {clr.shape[1]}",
        f"Minimum CLR: {clr.min()}",
        f"Maximum CLR: {clr.max()}",
        f"Mean CLR: {clr.mean()}",
        f"Maximum absolute row sum: {max_abs_row_sum}",
        "",
        "CLR QC passed.",
    ]),
    encoding="utf-8",
)

print("\nSaved:")
print(OUTPUT)
print(REPORT)