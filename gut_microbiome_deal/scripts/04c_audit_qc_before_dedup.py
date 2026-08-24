from pathlib import Path
import pandas as pd


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
RAW_ROOT = ROOT / "raw" / "gut_microbiome"
PROJECT = ROOT / "gut_microbiome_deal"

KEY = [
    "participant_id",
    "cohort",
    "research_stage",
    "array_index",
]


# ============================================================
# 1. Find raw MetaPhlAn species parquet
# ============================================================

matches = list(
    RAW_ROOT.rglob("metaphlan_abundance_species.parquet")
)

if len(matches) != 1:
    raise RuntimeError(
        f"Expected exactly 1 parquet, found {len(matches)}"
    )

PARQUET = matches[0]

print("Input:")
print(PARQUET)


# ============================================================
# 2. Load raw parquet
# ============================================================

df = pd.read_parquet(PARQUET)

if not (
    isinstance(df.index, pd.RangeIndex)
    and df.index.name is None
):
    df = df.reset_index()

species_cols = [
    c for c in df.columns
    if c not in KEY
]

print("\nSpecies columns:", len(species_cols))


# ============================================================
# 3. Keep ALL baseline records
#    IMPORTANT: NO participant deduplication here
# ============================================================

baseline = df[
    df["research_stage"].eq("00_00_visit")
].copy()

print("\n" + "=" * 80)
print("RAW BASELINE")
print("=" * 80)

print("Baseline sample records:", len(baseline))
print(
    "Unique participants:",
    baseline["participant_id"].nunique()
)

counts = baseline["participant_id"].value_counts()

print(
    "Participants with duplicate baseline records:",
    int((counts > 1).sum())
)

print(
    "Extra baseline sample records:",
    int((counts - 1).clip(lower=0).sum())
)


# ============================================================
# 4. Full microbiome pipeline using 9105 sample records
# ============================================================

X = baseline[species_cols]


# -------------------------
# Step 1: prevalence >=10%
# -------------------------

prevalence = X.notna().mean(axis=0)

keep_prev = prevalence >= 0.10

species388 = prevalence.index[keep_prev].tolist()

X388 = X[species388]


# -------------------------
# Step 2: observed-only mean >=0.01%
# -------------------------

mean_abundance = X388.mean(
    axis=0,
    skipna=True,
)

keep_mean = mean_abundance >= 0.01

species379 = mean_abundance.index[
    keep_mean
].tolist()

X379 = X388[species379]


print("\n" + "=" * 80)
print("SPECIES QC USING ALL 9105 BASELINE RECORDS")
print("=" * 80)

print("Start species:", len(species_cols))
print(
    "After prevalence >=10%:",
    len(species388)
)
print(
    "After mean abundance >=0.01%:",
    len(species379)
)


# ============================================================
# 5. Sample-level missingness
# ============================================================

missing_count = X379.isna().sum(axis=1)

missing_fraction = (
    missing_count / len(species379)
)

remove = missing_fraction > 0.90

baseline_qc = baseline[
    KEY
].copy()

baseline_qc["missing_count"] = missing_count
baseline_qc["missing_fraction"] = missing_fraction
baseline_qc["remove"] = remove


print("\n" + "=" * 80)
print("SAMPLE QC")
print("=" * 80)

print(
    "Sample records >90% missing:",
    int(remove.sum())
)

print(
    "Sample records retained:",
    int((~remove).sum())
)


# ============================================================
# 6. Count PARTICIPANTS after sample QC
# ============================================================

retained_rows = baseline_qc[
    ~baseline_qc["remove"]
].copy()

removed_rows = baseline_qc[
    baseline_qc["remove"]
].copy()

retained_participants = set(
    retained_rows["participant_id"]
)

removed_participants = set(
    removed_rows["participant_id"]
)

all_participants = set(
    baseline_qc["participant_id"]
)

# Participant is truly removed only if ALL of their
# baseline sample records failed QC.
fully_removed_participants = (
    all_participants
    - retained_participants
)


print("\n" + "=" * 80)
print("PARTICIPANT-LEVEL RESULT")
print("=" * 80)

print(
    "Starting unique participants:",
    len(all_participants)
)

print(
    "Participants having >=1 failed sample:",
    len(removed_participants)
)

print(
    "Participants with ALL baseline samples failed:",
    len(fully_removed_participants)
)

print(
    "Participants with >=1 retained baseline sample:",
    len(retained_participants)
)


# ============================================================
# 7. Inspect duplicate participants
# ============================================================

duplicate_ids = set(
    counts[counts > 1].index
)

dup_qc = baseline_qc[
    baseline_qc["participant_id"].isin(
        duplicate_ids
    )
].sort_values(
    ["participant_id", "array_index"]
)


print("\n" + "=" * 80)
print("DUPLICATE BASELINE QC")
print("=" * 80)

print(
    dup_qc[
        [
            "participant_id",
            "array_index",
            "missing_count",
            "missing_fraction",
            "remove",
        ]
    ].to_string(index=False)
)


# ============================================================
# 8. Compare species set with our current 379
# ============================================================

current379 = pd.read_csv(
    PROJECT
    / "data"
    / "03_species_mean_abundance_filtered.csv",
    nrows=0,
)

current_species = {
    c for c in current379.columns
    if c not in KEY
}

all_rows_species = set(species379)

print("\n" + "=" * 80)
print("379 SPECIES SET COMPARISON")
print("=" * 80)

print(
    "Current 9087-first pipeline:",
    len(current_species)
)

print(
    "9105-record pipeline:",
    len(all_rows_species)
)

print(
    "Symmetric difference:",
    len(
        current_species
        ^ all_rows_species
    )
)


# ============================================================
# 9. Paper checkpoint
# ============================================================

print("\n" + "=" * 80)
print("PAPER COMPARISON")
print("=" * 80)

print("Paper:")
print("2088 -> 388 -> 379")
print("9087 -> 9075 participants")

print("\nObserved using QC-before-dedup:")
print(
    f"{len(species_cols)} "
    f"-> {len(species388)} "
    f"-> {len(species379)}"
)

print(
    f"{len(all_participants)} "
    f"-> {len(retained_participants)} participants"
)


# ============================================================
# Save
# ============================================================

baseline_qc.to_csv(
    PROJECT
    / "reports"
    / "04c_qc_before_dedup_all_baseline_records.csv",
    index=False,
)