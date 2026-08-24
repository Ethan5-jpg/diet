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
    / "06_alpha_diversity.csv"
)

REPORT = (
    PROJECT
    / "reports"
    / "06_alpha_diversity_qc.txt"
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

if X.shape != (9074, 379):
    raise RuntimeError(
        f"Unexpected matrix shape: {X.shape}"
    )

if not np.isfinite(X).all():
    raise RuntimeError(
        "Non-finite values found before diversity calculation."
    )

if (X <= 0).any():
    raise RuntimeError(
        "Non-positive values found."
    )


# ============================================================
# Convert each sample to composition
#
# Shannon / Simpson are based on relative proportions.
# This also makes the calculation invariant to 0-1 vs 0-100 scale.
# ============================================================

row_sum = X.sum(axis=1, keepdims=True)

P = X / row_sum


# ============================================================
# Shannon diversity
#
# H = -sum(p_i * ln(p_i))
# ============================================================

shannon = -np.sum(
    P * np.log(P),
    axis=1,
)


# ============================================================
# Simpson diversity
#
# Here use 1 - sum(p_i^2)
# Higher value = higher diversity.
# ============================================================

simpson = 1.0 - np.sum(
    P ** 2,
    axis=1,
)


# ============================================================
# Save
# ============================================================

result = df[KEY_COLS].copy()

result["shannon_index"] = shannon
result["simpson_index"] = simpson

result.to_csv(
    OUTPUT,
    index=False,
)


# ============================================================
# QC report
# ============================================================

lines = []

def report(x=""):
    print(x)
    lines.append(str(x))


report("=" * 80)
report("ALPHA DIVERSITY")
report("=" * 80)

report(f"Samples: {len(result)}")
report(f"Species: {len(species_cols)}")

report("")
report("Shannon:")
report(
    pd.Series(shannon)
    .describe(
        percentiles=[
            0.01,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.99,
        ]
    )
    .to_string()
)

report("")
report("Simpson:")
report(
    pd.Series(simpson)
    .describe(
        percentiles=[
            0.01,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.99,
        ]
    )
    .to_string()
)

REPORT.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print("\nSaved:")
print(OUTPUT)
print(REPORT)