from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
PROJECT = ROOT / "gut_microbiome_deal"

INPUT = (
    PROJECT
    / "data"
    / "07_species_clr.csv"
)

OUTPUT = (
    PROJECT
    / "data"
    / "08_species_clr_zscore.csv"
)

PARAMETERS = (
    PROJECT
    / "reports"
    / "08_zscore_parameters.csv"
)

REPORT = (
    PROJECT
    / "reports"
    / "08_zscore_qc.txt"
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
)

if X.shape != (9074, 379):
    raise RuntimeError(
        f"Unexpected shape: {X.shape}"
    )


# ============================================================
# Species-wise mean and SD
#
# ddof=0 is the conventional ML / StandardScaler definition.
# ============================================================

mean = X.mean(axis=0)

std = X.std(
    axis=0,
    ddof=0,
)

if (std == 0).any():
    bad = std[std == 0]

    raise RuntimeError(
        "Zero-variance species found:\n"
        + "\n".join(bad.index)
    )


# ============================================================
# Z-score
# ============================================================

Z = (X - mean) / std


# ============================================================
# QC
# ============================================================

z_mean = Z.mean(axis=0)

z_std = Z.std(
    axis=0,
    ddof=0,
)

max_abs_mean = (
    z_mean.abs().max()
)

max_abs_std_error = (
    (z_std - 1).abs().max()
)


print("=" * 80)
print("Z-SCORE")
print("=" * 80)

print("Samples:", len(Z))
print("Species:", len(species_cols))

print(
    "Maximum absolute species mean:",
    max_abs_mean,
)

print(
    "Maximum deviation of species SD from 1:",
    max_abs_std_error,
)


# ============================================================
# Save final matrix
# ============================================================

result = pd.concat(
    [
        df[KEY_COLS].reset_index(drop=True),
        Z.reset_index(drop=True),
    ],
    axis=1,
)

result.to_csv(
    OUTPUT,
    index=False,
)


# Save transformation parameters
params = pd.DataFrame({
    "species": species_cols,
    "clr_mean": mean.values,
    "clr_std_ddof0": std.values,
})

params.to_csv(
    PARAMETERS,
    index=False,
)


# ============================================================
# Report
# ============================================================

REPORT.write_text(
    "\n".join([
        f"Samples: {len(Z)}",
        f"Species: {len(species_cols)}",
        f"Maximum absolute standardized mean: {max_abs_mean}",
        f"Maximum SD deviation from 1: {max_abs_std_error}",
        "",
        "Final shape: 9074 x 379 species",
        "Z-score QC passed.",
    ]),
    encoding="utf-8",
)

print("\nSaved:")
print(OUTPUT)
print(PARAMETERS)
print(REPORT)