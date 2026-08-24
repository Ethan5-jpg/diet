from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
PROJECT = ROOT / "gut_microbiome_deal"

INPUT = (
    PROJECT
    / "data"
    / "04_sample_missingness_filtered.csv"
)

OUTPUT = (
    PROJECT
    / "data"
    / "05_species_half_min_imputed.csv"
)

DETAILS = (
    PROJECT
    / "reports"
    / "05_imputation_values_by_species.csv"
)

REPORT = (
    PROJECT
    / "reports"
    / "05_imputation_qc.txt"
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

lines = []


def report(x=""):
    print(x)
    lines.append(str(x))


report("=" * 80)
report("INPUT")
report("=" * 80)

report(f"Samples: {len(X)}")
report(f"Species: {len(species_cols)}")
report(f"Missing cells before imputation: {int(X.isna().sum().sum())}")


if len(X) != 9074:
    raise RuntimeError(
        f"Expected 9074 samples from strict sample QC, got {len(X)}"
    )

if len(species_cols) != 379:
    raise RuntimeError(
        f"Expected 379 species, got {len(species_cols)}"
    )


# ============================================================
# Minimum observed abundance for each species
# ============================================================

minimum_observed = X.min(
    axis=0,
    skipna=True,
)

if minimum_observed.isna().any():
    bad = minimum_observed[
        minimum_observed.isna()
    ]

    raise RuntimeError(
        "Some retained species have no observed abundance:\n"
        + "\n".join(bad.index)
    )


# ============================================================
# Half-minimum imputation value
# ============================================================

imputation_value = (
    minimum_observed / 2.0
)


report("")
report("=" * 80)
report("IMPUTATION VALUES")
report("=" * 80)

report(
    f"Minimum imputation value: "
    f"{imputation_value.min():.10g}%"
)

report(
    f"Maximum imputation value: "
    f"{imputation_value.max():.10g}%"
)

report(
    f"Median imputation value: "
    f"{imputation_value.median():.10g}%"
)

report("")
report("Paper reported approximately:")
report("range:  0.005% - 0.3%")
report("median: 0.02%")


# ============================================================
# Missing count per species before fill
# ============================================================

missing_before = X.isna().sum(axis=0)


# ============================================================
# Impute each column with its own half-minimum
# ============================================================

X_imputed = X.fillna(
    imputation_value
)


# ============================================================
# QC
# ============================================================

missing_after = int(
    X_imputed.isna().sum().sum()
)

report("")
report("=" * 80)
report("QC")
report("=" * 80)

report(
    f"Missing cells after imputation: "
    f"{missing_after}"
)

if missing_after != 0:
    raise RuntimeError(
        "Missing values remain after imputation."
    )

if (X_imputed <= 0).any().any():
    raise RuntimeError(
        "Non-positive values remain; CLR would be invalid."
    )

report(
    "All species abundances are strictly positive: YES"
)


# ============================================================
# Save species-level details
# ============================================================

details = pd.DataFrame({
    "species": species_cols,
    "minimum_observed_percent":
        minimum_observed.values,
    "half_minimum_imputation_percent":
        imputation_value.values,
    "missing_values_imputed":
        missing_before.values,
})

details = details.sort_values(
    "half_minimum_imputation_percent"
)

details.to_csv(
    DETAILS,
    index=False,
)


# ============================================================
# Save final imputed matrix
# ============================================================

result = pd.concat(
    [
        df[KEY_COLS].reset_index(drop=True),
        X_imputed.reset_index(drop=True),
    ],
    axis=1,
)

result.to_csv(
    OUTPUT,
    index=False,
)


REPORT.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print("\nSaved:")
print(OUTPUT)
print(DETAILS)
print(REPORT)