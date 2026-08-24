from pathlib import Path
import pandas as pd


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")

DIET_ROOT = ROOT / "diet_deal/outputs/05_diet_scores"
MICRO_ROOT = ROOT / "gut_microbiome_deal/data"

OUT_ROOT = ROOT / "diet_microbiome_analysis"
DATA_OUT = OUT_ROOT / "data"
REPORT_OUT = OUT_ROOT / "reports"

DATA_OUT.mkdir(parents=True, exist_ok=True)
REPORT_OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# CURRENT SCORE FILES
# 这是当前 v1 版本，不声明为论文最终严格复现版本
# ============================================================

FILES = {

    "AHEI": {
        "path":
            DIET_ROOT / "ahei/ahei_scores_final.csv",

        "z":
            "ahei_z",

        "quintile":
            "ahei_quintile",

        "adjusted":
            "ahei_energy_adjusted",
    },

    "AMED": {
        "path":
            DIET_ROOT / "amed/amed_participant_scores.csv",

        "z":
            "amed_energy_adjusted_score_z",

        "quintile":
            "amed_energy_adjusted_score_quintile",

        "adjusted":
            "amed_energy_adjusted_score",
    },

    "hPDI": {
        "path":
            DIET_ROOT / "hpdi/hpdi_participant_scores.csv",

        "z":
            "hpdi_score_energy_adjusted_z",

        "quintile":
            "hpdi_energy_adjusted_quintile",

        "adjusted":
            "hpdi_score_energy_adjusted",
    },

    "rEDIH": {
        "path":
            DIET_ROOT / "redih/redih_participant_scores_17.csv",

        "z":
            "redih_score_energy_adjusted_z_17",

        "quintile":
            "redih_energy_adjusted_quintile_17",

        "adjusted":
            "redih_score_energy_adjusted_17",
    },
}


MICRO_FILE = (
    MICRO_ROOT
    / "08_species_clr_zscore.csv"
)

ALPHA_FILE = (
    MICRO_ROOT
    / "06_alpha_diversity.csv"
)


# ============================================================
# ID NORMALIZATION
# ============================================================

def norm_id(s):

    return (
        s.astype(str)
        .str.strip()
        .str.replace(
            r"\.0$",
            "",
            regex=True
        )
    )


# ============================================================
# LOAD SCORES
# ============================================================

score_tables = []

summary_rows = []

print("=" * 90)
print("CURRENT FOUR-SCORE BUILD")
print("=" * 90)


for score, cfg in FILES.items():

    print(f"\n{score}")
    print("-" * 90)

    df = pd.read_csv(
        cfg["path"],
        low_memory=False,
    )

    if "participant_id" not in df.columns:

        raise ValueError(
            f"{score}: participant_id missing"
        )

    df["participant_id"] = norm_id(
        df["participant_id"]
    )

    if df["participant_id"].duplicated().any():

        raise RuntimeError(
            f"{score}: duplicate participant IDs"
        )

    required = [
        cfg["z"],
        cfg["adjusted"],
        cfg["quintile"],
    ]

    for c in required:

        if c not in df.columns:

            raise ValueError(
                f"{score}: missing column {c}"
            )

    out = df[
        [
            "participant_id",
            cfg["adjusted"],
            cfg["z"],
            cfg["quintile"],
        ]
    ].copy()

    out = out.rename(
        columns={
            cfg["adjusted"]:
                f"{score}_adjusted",

            cfg["z"]:
                f"{score}_z",

            cfg["quintile"]:
                f"{score}_quintile",
        }
    )

    print(
        "participants:",
        out["participant_id"].nunique(),
    )

    print(
        "valid Z score:",
        out[f"{score}_z"].notna().sum(),
    )

    summary_rows.append({
        "score": score,
        "participants":
            out["participant_id"].nunique(),

        "valid_z":
            out[f"{score}_z"].notna().sum(),
    })

    score_tables.append(out)


# ============================================================
# MERGE FOUR SCORES
# ============================================================

scores = score_tables[0]

for df in score_tables[1:]:

    scores = scores.merge(
        df,
        on="participant_id",
        how="outer",
        validate="one_to_one",
    )


zcols = [
    "AHEI_z",
    "AMED_z",
    "hPDI_z",
    "rEDIH_z",
]


scores["all_4_scores_complete"] = (
    scores[zcols]
    .notna()
    .all(axis=1)
)


scores_all4 = scores.loc[
    scores["all_4_scores_complete"]
].copy()


print()
print("=" * 90)
print("FOUR-SCORE INTERSECTION")
print("=" * 90)

print(
    "Union participants:",
    len(scores)
)

print(
    "Participants with all 4:",
    len(scores_all4)
)


# ============================================================
# MICROBIOME
# ============================================================

micro = pd.read_csv(
    MICRO_FILE,
    low_memory=False,
)

alpha = pd.read_csv(
    ALPHA_FILE,
    low_memory=False,
)


for name, df in [
    ("micro", micro),
    ("alpha", alpha),
]:

    if "participant_id" not in df.columns:

        raise ValueError(
            f"{name}: participant_id missing"
        )

    df["participant_id"] = norm_id(
        df["participant_id"]
    )

    if df["participant_id"].duplicated().any():

        raise RuntimeError(
            f"{name}: duplicate IDs"
        )


micro_ids = set(
    micro["participant_id"]
)

alpha_ids = set(
    alpha["participant_id"]
)

diet_ids = set(
    scores_all4["participant_id"]
)


print()
print("=" * 90)
print("MICROBIOME")
print("=" * 90)

print(
    "CLR-Z participants:",
    len(micro_ids)
)

print(
    "Alpha participants:",
    len(alpha_ids)
)

print(
    "CLR vs alpha difference:",
    len(micro_ids ^ alpha_ids)
)


# ============================================================
# FINAL COMMON COHORT
# ============================================================

common = (
    diet_ids
    & micro_ids
    & alpha_ids
)


print()
print("=" * 90)
print("FINAL CURRENT DIET × MICROBIOME COHORT")
print("=" * 90)

print(
    "4-score participants:",
    len(diet_ids)
)

print(
    "Microbiome participants:",
    len(micro_ids)
)

print(
    "COMMON participants:",
    len(common)
)

print(
    "Diet only:",
    len(diet_ids - micro_ids)
)

print(
    "Microbiome only:",
    len(micro_ids - diet_ids)
)


# ============================================================
# FIX SAME PARTICIPANT ORDER
# ============================================================

ids = pd.DataFrame({
    "participant_id":
        sorted(common)
})


diet_aligned = ids.merge(
    scores_all4,
    on="participant_id",
    how="left",
    validate="one_to_one",
)


alpha_aligned = ids.merge(
    alpha,
    on="participant_id",
    how="left",
    validate="one_to_one",
)


micro_aligned = ids.merge(
    micro,
    on="participant_id",
    how="left",
    validate="one_to_one",
)


assert (
    diet_aligned["participant_id"].tolist()
    ==
    alpha_aligned["participant_id"].tolist()
    ==
    micro_aligned["participant_id"].tolist()
)


# ============================================================
# SCORE CORRELATION — FIRST SANITY CHECK
# ============================================================

corr = (
    diet_aligned[zcols]
    .corr(method="spearman")
)


print()
print("=" * 90)
print("SPEARMAN CORRELATION")
print("=" * 90)

print(
    corr.round(3).to_string()
)


# ============================================================
# SAVE
# ============================================================

scores.to_csv(
    DATA_OUT
    / "00_current_four_scores_all_participants.csv",
    index=False,
)


diet_aligned.to_csv(
    DATA_OUT
    / "00_aligned_four_scores.csv",
    index=False,
)


alpha_aligned.to_csv(
    DATA_OUT
    / "00_aligned_alpha_diversity.csv",
    index=False,
)


micro_aligned.to_csv(
    DATA_OUT
    / "00_aligned_species_clr_zscore.csv",
    index=False,
)


ids.to_csv(
    DATA_OUT
    / "00_common_participant_ids.csv",
    index=False,
)


pd.DataFrame(
    summary_rows
).to_csv(
    REPORT_OUT
    / "00_score_counts.csv",
    index=False,
)


corr.to_csv(
    REPORT_OUT
    / "00_score_spearman_correlation.csv"
)


with open(
    REPORT_OUT
    / "00_alignment_summary.txt",
    "w",
) as f:

    f.write(
        f"""
CURRENT VERSION-1 ANALYSIS COHORT

AHEI:
current modified implementation

AMED:
current 8-component implementation

hPDI:
current 18-component implementation

rEDIH:
current 17-component implementation

This is a proof-of-concept analysis cohort.
These scores should not yet be described as exact
replications of the target-paper scores.

4-score participants:
{len(diet_ids)}

Microbiome participants:
{len(micro_ids)}

Final common cohort:
{len(common)}

Diet only:
{len(diet_ids - micro_ids)}

Microbiome only:
{len(micro_ids - diet_ids)}
"""
    )


print()
print("=" * 90)
print("SAVED")
print("=" * 90)

print(DATA_OUT)
print(REPORT_OUT)