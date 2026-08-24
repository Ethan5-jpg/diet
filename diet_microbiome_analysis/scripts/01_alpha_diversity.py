from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr


# ============================================================
# PATHS
# ============================================================

ROOT = Path("/home/ec2-user/Desktop/HPP/Data")

ANALYSIS = ROOT / "diet_microbiome_analysis"

DATA_DIR = ANALYSIS / "data"
REPORT_DIR = ANALYSIS / "reports"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


DIET_FILE = (
    DATA_DIR
    / "00_aligned_four_scores.csv"
)

ALPHA_FILE = (
    DATA_DIR
    / "00_aligned_alpha_diversity.csv"
)


# ============================================================
# SCORE DEFINITIONS
# ============================================================

SCORES = {

    "AHEI": {
        "z": "AHEI_z",
        "q": "AHEI_quintile",
    },

    "AMED": {
        "z": "AMED_z",
        "q": "AMED_quintile",
    },

    "hPDI": {
        "z": "hPDI_z",
        "q": "hPDI_quintile",
    },

    "rEDIH": {
        "z": "rEDIH_z",
        "q": "rEDIH_quintile",
    },
}


METRICS = [
    "shannon_index",
    "simpson_index",
]


# ============================================================
# HELPERS
# ============================================================

def normalize_id(series):

    return (
        series
        .astype(str)
        .str.strip()
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
    )


def normalize_quintile(series):
    """
    Convert:
      1 / 2 / ...
      Q1 / Q2 / ...
      1.0 / 2.0
    into integer 1..5.
    """

    s = (
        series
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace(
            "Q",
            "",
            regex=False,
        )
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
    )

    return pd.to_numeric(
        s,
        errors="coerce"
    )


def bh_fdr(pvalues):
    """
    Benjamini-Hochberg correction.
    Used here only as an additional exploratory QC.
    Raw Kruskal-Wallis P is the primary paper comparison.
    """

    p = np.asarray(
        pvalues,
        dtype=float
    )

    n = len(p)

    order = np.argsort(p)
    ranked = p[order]

    adjusted = (
        ranked
        * n
        / np.arange(
            1,
            n + 1
        )
    )

    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.clip(
        adjusted,
        0,
        1
    )

    out = np.empty(
        n,
        dtype=float
    )

    out[order] = adjusted

    return out


# ============================================================
# LOAD
# ============================================================

diet = pd.read_csv(
    DIET_FILE,
    low_memory=False,
)

alpha = pd.read_csv(
    ALPHA_FILE,
    low_memory=False,
)


diet["participant_id"] = normalize_id(
    diet["participant_id"]
)

alpha["participant_id"] = normalize_id(
    alpha["participant_id"]
)


# ============================================================
# BASIC INTEGRITY CHECK
# ============================================================

if diet["participant_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate participant IDs in diet file."
    )

if alpha["participant_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate participant IDs in alpha file."
    )


diet_ids = diet["participant_id"].tolist()
alpha_ids = alpha["participant_id"].tolist()


if diet_ids != alpha_ids:

    raise RuntimeError(
        "Diet and alpha participant order differs. "
        "Do not continue."
    )


for metric in METRICS:

    if metric not in alpha.columns:

        raise ValueError(
            f"Missing alpha metric: {metric}"
        )


# ============================================================
# MERGE
# ============================================================

keep_diet_cols = [
    "participant_id"
]

for cfg in SCORES.values():

    keep_diet_cols.extend(
        [
            cfg["z"],
            cfg["q"],
        ]
    )


df = diet[
    keep_diet_cols
].merge(
    alpha[
        [
            "participant_id",
            "shannon_index",
            "simpson_index",
        ]
    ],
    on="participant_id",
    how="inner",
    validate="one_to_one",
)


print(
    "=" * 90
)

print(
    "ALPHA DIVERSITY ANALYSIS"
)

print(
    "=" * 90
)

print(
    "Participants:",
    len(df)
)


# ============================================================
# NORMALIZE QUINTILES
# ============================================================

for score, cfg in SCORES.items():

    df[
        cfg["q"]
    ] = normalize_quintile(
        df[cfg["q"]]
    )

    valid_q = sorted(
        df[
            cfg["q"]
        ]
        .dropna()
        .unique()
    )

    print(
        f"{score:6s} quintiles:",
        valid_q
    )


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

description_rows = []

kw_rows = []

spearman_rows = []


for score, cfg in SCORES.items():

    print()
    print(
        "=" * 90
    )

    print(
        score
    )

    print(
        "=" * 90
    )


    for metric in METRICS:

        temp = df[
            [
                cfg["q"],
                cfg["z"],
                metric,
            ]
        ].dropna().copy()


        # ====================================================
        # GROUP DESCRIPTIVES
        # ====================================================

        print()
        print(
            metric
        )

        print(
            "-" * 90
        )


        groups = []

        for q in range(
            1,
            6
        ):

            values = temp.loc[
                temp[cfg["q"]].eq(q),
                metric,
            ].dropna()


            if len(values) == 0:

                print(
                    f"Q{q}: EMPTY"
                )

                continue


            groups.append(
                values.values
            )


            q1 = values.quantile(
                0.25
            )

            median = values.median()

            q3 = values.quantile(
                0.75
            )


            description_rows.append(
                {
                    "score": score,
                    "metric": metric,
                    "quintile": q,
                    "N": len(values),
                    "mean": values.mean(),
                    "sd": values.std(),
                    "median": median,
                    "q25": q1,
                    "q75": q3,
                    "min": values.min(),
                    "max": values.max(),
                }
            )


            print(
                f"Q{q}: "
                f"N={len(values):4d} | "
                f"median={median:.6f} | "
                f"IQR="
                f"{q1:.6f}–{q3:.6f}"
            )


        # ====================================================
        # KRUSKAL-WALLIS
        # ====================================================

        if len(groups) != 5:

            raise RuntimeError(
                f"{score} {metric}: "
                "not all five quintiles present."
            )


        kw = kruskal(
            *groups
        )


        # approximate epsilon-squared effect size
        n_total = sum(
            len(g)
            for g in groups
        )

        k = len(groups)

        epsilon_sq = (
            (
                kw.statistic
                - k
                + 1
            )
            /
            (
                n_total
                - k
            )
        )

        epsilon_sq = max(
            0,
            epsilon_sq
        )


        kw_rows.append(
            {
                "score": score,
                "metric": metric,
                "N": n_total,
                "H": kw.statistic,
                "df": k - 1,
                "p_raw": kw.pvalue,
                "epsilon_squared": epsilon_sq,
            }
        )


        print()

        print(
            "Kruskal-Wallis:"
        )

        print(
            f"  H = {kw.statistic:.6f}"
        )

        print(
            f"  df = {k - 1}"
        )

        print(
            f"  P = {kw.pvalue:.8g}"
        )

        print(
            f"  epsilon² = "
            f"{epsilon_sq:.6g}"
        )


        # ====================================================
        # SECONDARY SANITY CHECK:
        # CONTINUOUS SCORE VS DIVERSITY
        # ====================================================

        continuous = temp[
            [
                cfg["z"],
                metric,
            ]
        ].dropna()


        sp = spearmanr(
            continuous[cfg["z"]],
            continuous[metric],
        )


        spearman_rows.append(
            {
                "score": score,
                "metric": metric,
                "N": len(continuous),
                "rho": sp.statistic,
                "p": sp.pvalue,
            }
        )


        print(
            f"Continuous Spearman: "
            f"rho={sp.statistic:.4f}, "
            f"P={sp.pvalue:.8g}"
        )


# ============================================================
# RESULTS TABLE
# ============================================================

desc_df = pd.DataFrame(
    description_rows
)

kw_df = pd.DataFrame(
    kw_rows
)

sp_df = pd.DataFrame(
    spearman_rows
)


# Exploratory FDR across the 8 alpha-diversity tests
kw_df["p_fdr_8tests"] = bh_fdr(
    kw_df["p_raw"].values
)


# ============================================================
# PAPER COMPARISON
# ============================================================

def interpret_alpha_result(p):

    if p < 0.05:

        return "MATCH"

    if p < 0.10:

        return "CLOSE"

    return "DIFFERENT"


kw_df[
    "paper_expectation"
] = "significant"

kw_df[
    "comparison"
] = kw_df[
    "p_raw"
].apply(
    interpret_alpha_result
)


print()
print(
    "=" * 90
)

print(
    "SUMMARY"
)

print(
    "=" * 90
)


summary_view = kw_df[
    [
        "score",
        "metric",
        "N",
        "H",
        "p_raw",
        "p_fdr_8tests",
        "epsilon_squared",
        "comparison",
    ]
].copy()


print(
    summary_view.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6g}"
    )
)


# ============================================================
# HIGH-LEVEL PAPER CHECKPOINT
# ============================================================

print()
print(
    "=" * 90
)

print(
    "PAPER CHECKPOINT"
)

print(
    "=" * 90
)

print(
    "Paper expectation:"
)

print(
    "AHEI / AMED / hPDI / rEDIH "
    "show significant differences "
    "in alpha diversity across quintiles."
)

print()

for score in SCORES:

    x = kw_df.loc[
        kw_df["score"].eq(score)
    ]

    n_sig = int(
        (
            x["p_raw"]
            < 0.05
        ).sum()
    )

    if n_sig == 2:

        result = "MATCH"

    elif n_sig == 1:

        result = "PARTIAL"

    else:

        result = "DIFFERENT"

    print(
        f"{score:6s}: "
        f"{n_sig}/2 significant "
        f"-> {result}"
    )


# ============================================================
# SAVE
# ============================================================

desc_df.to_csv(
    REPORT_DIR
    / "01_alpha_diversity_by_quintile.csv",
    index=False,
)


kw_df.to_csv(
    REPORT_DIR
    / "01_alpha_diversity_kruskal_wallis.csv",
    index=False,
)


sp_df.to_csv(
    REPORT_DIR
    / "01_alpha_diversity_continuous_spearman.csv",
    index=False,
)


# Save combined working table as well
df.to_csv(
    DATA_DIR
    / "01_diet_alpha_analysis_table.csv",
    index=False,
)


with open(
    REPORT_DIR
    / "01_alpha_diversity_summary.txt",
    "w",
) as f:

    f.write(
        "ALPHA DIVERSITY ANALYSIS\n"
    )

    f.write(
        "=" * 80
        + "\n"
    )

    f.write(
        f"Participants: {len(df)}\n\n"
    )

    f.write(
        summary_view.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "Paper reference:\n"
    )

    f.write(
        "AHEI, AMED, hPDI and rEDIH "
        "were significantly associated "
        "with Shannon/Simpson diversity "
        "across quintiles.\n"
    )

    f.write(
        "\nImportant:\n"
    )

    f.write(
        "Current dietary scores are "
        "version-1/proxy implementations. "
        "Exact P values are not expected "
        "to reproduce the paper.\n"
    )


print()
print(
    "=" * 90
)

print(
    "SAVED"
)

print(
    "=" * 90
)

print(
    REPORT_DIR
    / "01_alpha_diversity_by_quintile.csv"
)

print(
    REPORT_DIR
    / "01_alpha_diversity_kruskal_wallis.csv"
)

print(
    REPORT_DIR
    / "01_alpha_diversity_continuous_spearman.csv"
)

print(
    REPORT_DIR
    / "01_alpha_diversity_summary.txt"
)