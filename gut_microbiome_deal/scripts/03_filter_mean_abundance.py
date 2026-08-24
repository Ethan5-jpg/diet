from pathlib import Path

import pandas as pd


DATA_ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
PROJECT = DATA_ROOT / "gut_microbiome_deal"

INPUT = (
    PROJECT
    / "data"
    / "02_species_prevalence_filtered.csv"
)

OUTPUT = (
    PROJECT
    / "data"
    / "03_species_mean_abundance_filtered.csv"
)

REPORT = (
    PROJECT
    / "reports"
    / "03_mean_abundance_qc.txt"
)

DETAILS = (
    PROJECT
    / "reports"
    / "03_species_mean_abundance_details.csv"
)

KEY_COLS = [
    "participant_id",
    "cohort",
    "research_stage",
    "array_index",
]

THRESHOLD = 0.01  # 当前表是百分比单位，所以0.01就是0.01%


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
report(f"Species after prevalence filter: {len(species_cols)}")
report(f"Mean abundance threshold: {THRESHOLD}%")

if len(species_cols) != 388:
    raise RuntimeError(
        f"Expected 388 species from step 02, got {len(species_cols)}"
    )


# ============================================================
# Method A:
# mean among observed/non-missing samples
# ============================================================

mean_observed = X.mean(
    axis=0,
    skipna=True,
)

keep_observed = (
    mean_observed >= THRESHOLD
)


# ============================================================
# Method B:
# missing abundance treated as zero
# mean across all 9087 participants
# ============================================================

mean_all_samples = (
    X.fillna(0)
    .mean(axis=0)
)

keep_all_samples = (
    mean_all_samples >= THRESHOLD
)


# ============================================================
# Compare
# ============================================================

n_observed = int(keep_observed.sum())
n_all = int(keep_all_samples.sum())

report("")
report("=" * 80)
report("TWO POSSIBLE MEAN DEFINITIONS")
report("=" * 80)

report(
    f"A. Observed-only mean >=0.01%: {n_observed}"
)

report(
    f"B. NaN-as-zero mean >=0.01%:   {n_all}"
)

report("")
report("Paper expected: 379")


# ============================================================
# Determine which one reproduces paper
# ============================================================

matches = []

if n_observed == 379:
    matches.append(
        ("observed_only", keep_observed)
    )

if n_all == 379:
    matches.append(
        ("nan_as_zero", keep_all_samples)
    )


if len(matches) == 0:
    report("")
    report("WARNING:")
    report(
        "Neither definition reproduces 379."
    )

    selected_method = None
    selected_keep = None

elif len(matches) == 1:

    selected_method, selected_keep = matches[0]

    report("")
    report(
        f"Selected method because it reproduces paper: "
        f"{selected_method}"
    )

else:

    report("")
    report(
        "WARNING: both definitions reproduce 379."
    )
    report(
        "Cannot distinguish denominator from this checkpoint alone."
    )

    selected_method = matches[0][0]
    selected_keep = matches[0][1]


# ============================================================
# Save audit details regardless
# ============================================================

details = pd.DataFrame({
    "species": species_cols,
    "mean_observed_only_percent":
        mean_observed.values,
    "mean_nan_as_zero_percent":
        mean_all_samples.values,
    "keep_observed_only":
        keep_observed.values,
    "keep_nan_as_zero":
        keep_all_samples.values,
})

details.to_csv(
    DETAILS,
    index=False,
)


# ============================================================
# Save filtered data only when we reproduce 379
# ============================================================

if selected_keep is not None:

    kept_species = (
        selected_keep[
            selected_keep
        ]
        .index
        .tolist()
    )

    filtered = pd.concat(
        [
            df[KEY_COLS].reset_index(drop=True),
            X[kept_species].reset_index(drop=True),
        ],
        axis=1,
    )

    filtered.to_csv(
        OUTPUT,
        index=False,
    )

    report("")
    report("=" * 80)
    report("PAPER CHECKPOINT")
    report("=" * 80)

    report("Before mean abundance filter: 388")
    report(
        f"After mean abundance filter:  {len(kept_species)}"
    )

    if len(kept_species) == 379:
        report("")
        report("CHECKPOINT PASSED: 388 -> 379")

else:

    report("")
    report(
        "Filtered dataset was NOT saved because "
        "the paper checkpoint was not reproduced."
    )


REPORT.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print("\nSaved audit:")
print(DETAILS)
print(REPORT)

if selected_keep is not None:
    print(OUTPUT)