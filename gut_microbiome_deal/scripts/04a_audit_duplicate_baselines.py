from pathlib import Path
import pandas as pd

ROOT = Path("/home/ec2-user/Desktop/HPP/Data")

SOURCE = (
    ROOT / "Transfer" / "gut_microbiome"
    / "abundance" / "metaphlan4" / "aggregated"
    / "metaphlan_abundance_species.csv"
)

FILTERED_379 = (
    ROOT / "gut_microbiome_deal" / "data"
    / "03_species_mean_abundance_filtered.csv"
)

OUT = (
    ROOT / "gut_microbiome_deal" / "reports"
    / "04a_duplicate_baseline_missingness.csv"
)

KEY = [
    "participant_id",
    "cohort",
    "research_stage",
    "array_index",
]

# ============================================================
# 获取最终379个species列
# ============================================================

f379 = pd.read_csv(FILTERED_379, nrows=0)

species_cols = [
    c for c in f379.columns
    if c not in KEY
]

print("Final species:", len(species_cols))

# ============================================================
# 从原始species表重新读取baseline
# ============================================================

df = pd.read_csv(
    SOURCE,
    usecols=KEY + species_cols,
)

baseline = df[
    df["research_stage"].eq("00_00_visit")
].copy()

counts = baseline["participant_id"].value_counts()

dup_ids = counts[counts > 1].index

dup = baseline[
    baseline["participant_id"].isin(dup_ids)
].copy()

print("Duplicate participants:", len(dup_ids))
print("Duplicate rows:", len(dup))

# ============================================================
# 计算每条记录在379 species上的missingness
# ============================================================

X = dup[species_cols].apply(
    pd.to_numeric,
    errors="coerce",
)

dup["missing_count"] = X.isna().sum(axis=1)
dup["nonmissing_count"] = X.notna().sum(axis=1)
dup["missing_fraction"] = (
    dup["missing_count"] / len(species_cols)
)

dup["remove_gt90"] = (
    dup["missing_fraction"] > 0.90
)

# ============================================================
# 按participant展示array_index=0 vs 1
# ============================================================

show = dup[
    [
        "participant_id",
        "cohort",
        "research_stage",
        "array_index",
        "missing_count",
        "nonmissing_count",
        "missing_fraction",
        "remove_gt90",
    ]
].sort_values(
    ["participant_id", "array_index"]
)

print("\n" + "=" * 100)
print("DUPLICATE BASELINE COMPARISON")
print("=" * 100)

print(show.to_string(index=False))

# ============================================================
# 找是否存在两条记录跨越90%阈值
# ============================================================

cross = (
    show.groupby("participant_id")["remove_gt90"]
    .nunique()
)

cross_ids = cross[cross > 1].index

print("\n" + "=" * 100)
print("PARTICIPANTS WHO CROSS THE 90% THRESHOLD")
print("=" * 100)

print("Count:", len(cross_ids))

if len(cross_ids):
    print(
        show[
            show["participant_id"].isin(cross_ids)
        ].to_string(index=False)
    )

# ============================================================
# 比较不同选样策略最终会删多少人
# ============================================================

print("\n" + "=" * 100)
print("DUPLICATE SELECTION EFFECT")
print("=" * 100)

for index_value in [0, 1]:
    sub = show[
        show["array_index"].eq(index_value)
    ]

    print(
        f"array_index={index_value}: "
        f">90% missing = {int(sub['remove_gt90'].sum())}"
    )

show.to_csv(OUT, index=False)

print("\nSaved:")
print(OUT)