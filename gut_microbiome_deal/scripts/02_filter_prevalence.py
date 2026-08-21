from pathlib import Path

import pandas as pd


DATA_ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
PROJECT = DATA_ROOT / "gut_microbiome_deal"

INPUT = (
    PROJECT
    / "data"
    / "01_metaphlan_species_baseline_9087.csv"
)

OUTPUT = (
    PROJECT
    / "data"
    / "02_species_prevalence_filtered.csv"
)

REPORT = (
    PROJECT
    / "reports"
    / "02_prevalence_qc.txt"
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

N = len(X)

report_lines = []


def report(x=""):
    print(x)
    report_lines.append(str(x))


report("=" * 80)
report("INPUT")
report("=" * 80)

report(f"Samples: {N}")
report(f"Starting species: {len(species_cols)}")


# ============================================================
# Confirm encoding
# ============================================================

zero_cells = int((X == 0).sum().sum())
positive_cells = int((X > 0).sum().sum())
nonmissing_cells = int(X.notna().sum().sum())

report("")
report("ENCODING")
report(f"Non-missing cells: {nonmissing_cells}")
report(f"Positive cells:    {positive_cells}")
report(f"Zero cells:        {zero_cells}")

if zero_cells != 0:
    raise RuntimeError(
        "Unexpected zero values found. "
        "Presence definition needs re-evaluation."
    )

if nonmissing_cells != positive_cells:
    raise RuntimeError(
        "Non-missing and positive cells differ."
    )


# ============================================================
# Prevalence
#
# In this MetaPhlAn table:
# detected species = non-missing positive value
# undetected species = NaN
# ============================================================

presence_count = X.notna().sum(axis=0)
prevalence = presence_count / N


report("")
report("=" * 80)
report("PREVALENCE DISTRIBUTION")
report("=" * 80)

for threshold in [
    0.01,
    0.05,
    0.10,
    0.20,
    0.50,
]:

    count = int(
        (prevalence >= threshold).sum()
    )

    report(
        f"Species prevalence >= {threshold:.0%}: "
        f"{count}"
    )


# ============================================================
# Paper threshold: >=10%
# ============================================================

keep = prevalence >= 0.10

kept_species = prevalence.index[keep].tolist()

filtered = pd.concat(
    [
        df[KEY_COLS].reset_index(drop=True),
        X[kept_species].reset_index(drop=True),
    ],
    axis=1,
)


report("")
report("=" * 80)
report("PAPER CHECKPOINT")
report("=" * 80)

report(f"Before prevalence filter: {len(species_cols)}")
report(f"After prevalence >=10%:   {len(kept_species)}")

report("")
report("Paper expected:")
report("2088 -> 388")


# ============================================================
# Extra boundary audit
# ============================================================

minimum_required = int(
    -(-N * 10 // 100)
)

report("")
report("BOUNDARY")
report(
    f"10% of {N} samples requires at least "
    f"{minimum_required} detections."
)

report(
    "Minimum presence count among retained species: "
    f"{int(presence_count[keep].min())}"
)

if (~keep).any():
    report(
        "Maximum presence count among removed species: "
        f"{int(presence_count[~keep].max())}"
    )


# ============================================================
# Save
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

filtered.to_csv(
    OUTPUT,
    index=False,
)

pd.DataFrame(
    {
        "species": prevalence.index,
        "presence_count": presence_count.values,
        "prevalence": prevalence.values,
        "keep_prevalence_10pct": keep.values,
    }
).sort_values(
    "prevalence",
    ascending=False,
).to_csv(
    PROJECT
    / "reports"
    / "02_species_prevalence_details.csv",
    index=False,
)

REPORT.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print("\nSaved:")
print(OUTPUT)
print(REPORT)