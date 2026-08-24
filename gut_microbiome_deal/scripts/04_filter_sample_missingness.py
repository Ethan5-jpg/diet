from pathlib import Path

import pandas as pd


DATA_ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
PROJECT = DATA_ROOT / "gut_microbiome_deal"

INPUT = (
    PROJECT
    / "data"
    / "03_species_mean_abundance_filtered.csv"
)

OUTPUT = (
    PROJECT
    / "data"
    / "04_sample_missingness_filtered.csv"
)

REPORT = (
    PROJECT
    / "reports"
    / "04_sample_missingness_qc.txt"
)

DETAILS = (
    PROJECT
    / "reports"
    / "04_sample_missingness_details.csv"
)

KEY_COLS = [
    "participant_id",
    "cohort",
    "research_stage",
    "array_index",
]

MISSING_THRESHOLD = 0.90


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

report(f"Samples before sample QC: {len(df)}")
report(f"Retained species: {len(species_cols)}")

if len(df) != 9087:
    raise RuntimeError(
        f"Expected 9087 samples, got {len(df)}"
    )

if len(species_cols) != 379:
    raise RuntimeError(
        f"Expected 379 species, got {len(species_cols)}"
    )


# ============================================================
# Missingness per sample
# ============================================================

missing_count = X.isna().sum(axis=1)

missing_fraction = (
    missing_count / len(species_cols)
)

nonmissing_count = (
    len(species_cols) - missing_count
)


# Paper:
# remove samples with MORE THAN 90% missing values
remove = missing_fraction > MISSING_THRESHOLD
keep = ~remove


report("")
report("=" * 80)
report("MISSINGNESS")
report("=" * 80)

report(
    f"Samples with >90% missing: "
    f"{int(remove.sum())}"
)

report(
    f"Samples retained: "
    f"{int(keep.sum())}"
)

report("")
report("Missing fraction distribution:")
report(
    missing_fraction.describe(
        percentiles=[
            0.01,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    ).to_string()
)

report("")
report("Non-missing species per sample:")
report(
    nonmissing_count.describe(
        percentiles=[
            0.01,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.99,
        ]
    ).to_string()
)


# ============================================================
# Boundary cases
# ============================================================

report("")
report("=" * 80)
report("BOUNDARY")
report("=" * 80)

report(
    f"379 × 90% = "
    f"{len(species_cols) * 0.90}"
)

report(
    "Therefore >90% missing means "
    "at least 342 missing species."
)

if remove.any():
    report(
        "Minimum missing count among removed samples: "
        f"{int(missing_count[remove].min())}"
    )

    report(
        "Minimum missing fraction among removed samples: "
        f"{missing_fraction[remove].min():.6f}"
    )

if keep.any():
    report(
        "Maximum missing count among retained samples: "
        f"{int(missing_count[keep].max())}"
    )

    report(
        "Maximum missing fraction among retained samples: "
        f"{missing_fraction[keep].max():.6f}"
    )


# ============================================================
# Save row-level QC details
# ============================================================

details = df[KEY_COLS].copy()

details["missing_species_count"] = missing_count
details["nonmissing_species_count"] = nonmissing_count
details["missing_fraction"] = missing_fraction
details["remove_missing_gt_90pct"] = remove

details.to_csv(
    DETAILS,
    index=False,
)


# ============================================================
# Filter
# ============================================================

filtered = df.loc[keep].copy()

filtered.to_csv(
    OUTPUT,
    index=False,
)


# ============================================================
# Paper checkpoint
# ============================================================

report("")
report("=" * 80)
report("PAPER CHECKPOINT")
report("=" * 80)

report("Before sample missingness QC: 9087")
report(
    f"Removed >90% missing:       {int(remove.sum())}"
)
report(
    f"After sample QC:            {len(filtered)}"
)

report("")
report("Paper expected: 9075")

if len(filtered) == 9075:
    report("")
    report("CHECKPOINT PASSED: 9087 -> 9075")
else:
    report("")
    report(
        "CHECKPOINT NOT MATCHED: "
        f"observed {len(filtered)}, expected 9075"
    )


REPORT.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print("\nSaved:")
print(OUTPUT)
print(DETAILS)
print(REPORT)