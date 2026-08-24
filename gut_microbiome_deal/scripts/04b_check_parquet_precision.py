from pathlib import Path
import pandas as pd
import numpy as np


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
# 1. 自动找到原始 species parquet
# ============================================================

matches = list(
    RAW_ROOT.rglob("metaphlan_abundance_species.parquet")
)

print("=" * 80)
print("RAW PARQUET SEARCH")
print("=" * 80)

for x in matches:
    print(x)

if len(matches) != 1:
    raise RuntimeError(
        f"Expected exactly 1 species parquet, found {len(matches)}"
    )

PARQUET = matches[0]


# ============================================================
# 2. 直接读取原始 parquet
# ============================================================

raw = pd.read_parquet(PARQUET)

print("\nOriginal parquet shape:", raw.shape)
print("Original index:", raw.index.names)

# HPP的participant信息可能在MultiIndex里
if not (
    isinstance(raw.index, pd.RangeIndex)
    and raw.index.name is None
):
    raw = raw.reset_index()

print("After reset_index shape:", raw.shape)

print("\nFirst columns:")
print(raw.columns[:10].tolist())

for col in KEY:
    if col not in raw.columns:
        raise RuntimeError(
            f"Missing required key column: {col}"
        )

species_cols = [
    c for c in raw.columns
    if c not in KEY
]

print("Species columns:", len(species_cols))


# ============================================================
# 3. baseline = 9087
# ============================================================

baseline = raw[
    raw["research_stage"].eq("00_00_visit")
].copy()

baseline["_priority"] = (
    baseline["array_index"].ne(0).astype(int)
)

baseline = (
    baseline
    .sort_values(
        ["participant_id", "_priority", "array_index"]
    )
    .drop_duplicates(
        "participant_id",
        keep="first"
    )
    .drop(columns="_priority")
)

print("\nBaseline participants:", len(baseline))

X = baseline[species_cols]


# ============================================================
# 4. prevalence >=10%
# ============================================================

prevalence = X.notna().mean(axis=0)

keep_prev = prevalence >= 0.10

X388 = X.loc[:, keep_prev]

print("\nAfter prevalence:", X388.shape[1])


# ============================================================
# 5. mean abundance >=0.01%
#    保留完整浮点精度
# ============================================================

means = X388.mean(axis=0, skipna=True)

keep_mean = means >= 0.01

X379 = X388.loc[:, keep_mean]

print("After mean abundance:", X379.shape[1])


# ============================================================
# 6. 打印0.01附近的species
# ============================================================

distance = (means - 0.01).abs()

near = (
    pd.DataFrame({
        "species": means.index,
        "mean_abundance": means.values,
        "distance_from_0.01": distance.values,
        "keep": keep_mean.values,
    })
    .sort_values("distance_from_0.01")
    .head(30)
)

print("\n" + "=" * 80)
print("SPECIES CLOSEST TO 0.01%")
print("=" * 80)

for _, row in near.iterrows():
    print(
        f"{row['mean_abundance']:.17g} "
        f"{'KEEP' if row['keep'] else 'DROP'} "
        f"{row['species']}"
    )


# ============================================================
# 7. sample missingness
# ============================================================

missing_count = X379.isna().sum(axis=1)
missing_fraction = missing_count / X379.shape[1]

remove = missing_fraction > 0.90

print("\n" + "=" * 80)
print("RAW PARQUET FULL PIPELINE")
print("=" * 80)

print("Start samples:", len(X))
print("Start species:", len(species_cols))
print("After prevalence:", X388.shape[1])
print("After mean:", X379.shape[1])
print("Samples >90% missing:", int(remove.sum()))
print("Samples retained:", int((~remove).sum()))

print("\nRemoved sample missing counts:")
print(
    missing_count[remove]
    .sort_values()
    .to_string()
)


# ============================================================
# 8. 和当前CSV得到的379 species比较
# ============================================================

CSV379 = (
    PROJECT
    / "data"
    / "03_species_mean_abundance_filtered.csv"
)

csv_header = pd.read_csv(
    CSV379,
    nrows=0,
)

csv_species = {
    c for c in csv_header.columns
    if c not in KEY
}

raw_species = set(X379.columns)

print("\n" + "=" * 80)
print("RAW PARQUET vs CURRENT CSV 379 SPECIES")
print("=" * 80)

print("Raw species:", len(raw_species))
print("CSV species:", len(csv_species))

print(
    "Symmetric difference:",
    len(raw_species ^ csv_species)
)

if raw_species != csv_species:

    print("\nOnly in raw:")
    for x in sorted(raw_species - csv_species):
        print(x)

    print("\nOnly in CSV:")
    for x in sorted(csv_species - raw_species):
        print(x)


# ============================================================
# 9. 保存边界审计
# ============================================================

near.to_csv(
    PROJECT
    / "reports"
    / "04b_species_near_mean_0.01.csv",
    index=False,
)