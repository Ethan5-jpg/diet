from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist


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

MICRO_FILE = (
    DATA_DIR
    / "00_aligned_species_clr_zscore.csv"
)


# ============================================================
# DEFINITIONS
# ============================================================

SCORES = {
    "AHEI": "AHEI_z",
    "AMED": "AMED_z",
    "hPDI": "hPDI_z",
    "rEDIH": "rEDIH_z",
}


META_COLS = [
    "participant_id",
    "cohort",
    "research_stage",
    "array_index",
]


# ============================================================
# HELPERS
# ============================================================

def normalize_id(s):

    return (
        s.astype(str)
        .str.strip()
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
    )


def bh_fdr(pvalues):

    p = np.asarray(
        pvalues,
        dtype=float,
    )

    n = len(p)

    order = np.argsort(p)

    ranked = p[order]

    q = (
        ranked
        * n
        / np.arange(
            1,
            n + 1,
        )
    )

    q = np.minimum.accumulate(
        q[::-1]
    )[::-1]

    q = np.clip(
        q,
        0,
        1,
    )

    out = np.empty_like(q)

    out[order] = q

    return out


def fit_univariate_ols(x, Y):
    """
    Fit all species simultaneously:

        Y_j = intercept + beta_j * x

    Parameters
    ----------
    x:
        shape (N,)

    Y:
        shape (N, P)

    Returns
    -------
    beta
    se
    t
    p
    r2
    """

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    Y = np.asarray(
        Y,
        dtype=np.float64,
    )


    # Center
    xc = x - x.mean()

    Y_mean = Y.mean(
        axis=0
    )

    Yc = (
        Y
        - Y_mean
    )


    # slope
    sxx = np.sum(
        xc ** 2
    )

    if sxx <= 0:

        raise RuntimeError(
            "Diet score has zero variance."
        )


    beta = (
        xc[:, None]
        * Yc
    ).sum(
        axis=0
    ) / sxx


    intercept = (
        Y_mean
        - beta
        * x.mean()
    )


    # residual
    fitted = (
        intercept[None, :]
        +
        x[:, None]
        * beta[None, :]
    )

    resid = (
        Y
        - fitted
    )


    n = len(x)

    df = n - 2


    sse = np.sum(
        resid ** 2,
        axis=0,
    )


    mse = (
        sse
        / df
    )


    se = np.sqrt(
        mse
        / sxx
    )


    t_stat = (
        beta
        / se
    )


    p = (
        2
        * t_dist.sf(
            np.abs(t_stat),
            df=df,
        )
    )


    # R²
    sst = np.sum(
        Yc ** 2,
        axis=0,
    )

    r2 = (
        1
        - sse / sst
    )


    return (
        beta,
        se,
        t_stat,
        p,
        r2,
    )


# ============================================================
# LOAD
# ============================================================

diet = pd.read_csv(
    DIET_FILE,
    low_memory=False,
)

micro = pd.read_csv(
    MICRO_FILE,
    low_memory=False,
)


diet["participant_id"] = normalize_id(
    diet["participant_id"]
)

micro["participant_id"] = normalize_id(
    micro["participant_id"]
)


if diet["participant_id"].duplicated().any():

    raise RuntimeError(
        "Duplicate diet IDs."
    )


if micro["participant_id"].duplicated().any():

    raise RuntimeError(
        "Duplicate microbiome IDs."
    )


if (
    diet["participant_id"].tolist()
    !=
    micro["participant_id"].tolist()
):

    raise RuntimeError(
        "Participant order differs."
    )


species = [
    c
    for c in micro.columns
    if c not in META_COLS
]


print("=" * 95)
print("MICROBIOME-WIDE ASSOCIATION — MODEL 0")
print("=" * 95)

print(
    "Participants:",
    len(diet)
)

print(
    "Species:",
    len(species)
)

print(
    "Model:"
)

print(
    "species CLR-Z ~ dietary score Z"
)

print(
    "Covariates: NONE"
)


if len(species) != 379:

    print(
        "WARNING: expected 379 species, "
        f"found {len(species)}"
    )


# ============================================================
# MATRIX
# ============================================================

Y = (
    micro[species]
    .apply(
        pd.to_numeric,
        errors="coerce",
    )
    .to_numpy(
        dtype=np.float64
    )
)


if not np.isfinite(Y).all():

    raise RuntimeError(
        "Species matrix contains NaN/Inf."
    )


# ============================================================
# FIT EACH DIET SCORE
# ============================================================

all_results = []

summary_rows = []


for score_name, score_col in SCORES.items():

    print()
    print("=" * 95)
    print(score_name)
    print("=" * 95)


    x = pd.to_numeric(
        diet[score_col],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )


    valid = np.isfinite(x)


    x_use = x[valid]

    Y_use = Y[valid, :]


    print(
        "N:",
        len(x_use)
    )


    (
        beta,
        se,
        t_stat,
        p,
        r2,
    ) = fit_univariate_ols(
        x_use,
        Y_use,
    )


    q = bh_fdr(
        p
    )


    result = pd.DataFrame(
        {
            "score": score_name,
            "species": species,
            "N": len(x_use),
            "beta": beta,
            "SE": se,
            "t": t_stat,
            "p": p,
            "FDR": q,
            "R2": r2,
        }
    )


    result[
        "CI_lower"
    ] = (
        result["beta"]
        - 1.96
        * result["SE"]
    )

    result[
        "CI_upper"
    ] = (
        result["beta"]
        + 1.96
        * result["SE"]
    )


    result[
        "significant_FDR05"
    ] = (
        result["FDR"]
        < 0.05
    )


    result[
        "direction"
    ] = np.where(
        result["beta"] > 0,
        "positive",
        "negative",
    )


    sig = result.loc[
        result[
            "significant_FDR05"
        ]
    ].copy()


    n_sig = len(sig)

    n_pos = int(
        (
            sig["beta"]
            > 0
        ).sum()
    )

    n_neg = int(
        (
            sig["beta"]
            < 0
        ).sum()
    )


    print(
        "FDR < 0.05:",
        n_sig
    )

    print(
        "  positive:",
        n_pos
    )

    print(
        "  negative:",
        n_neg
    )


    print()
    print(
        "TOP POSITIVE"
    )

    print(
        "-" * 95
    )

    top_pos = (
        result
        .sort_values(
            "beta",
            ascending=False,
        )
        .head(10)
    )

    print(
        top_pos[
            [
                "species",
                "beta",
                "p",
                "FDR",
            ]
        ].to_string(
            index=False
        )
    )


    print()
    print(
        "TOP NEGATIVE"
    )

    print(
        "-" * 95
    )

    top_neg = (
        result
        .sort_values(
            "beta",
            ascending=True,
        )
        .head(10)
    )

    print(
        top_neg[
            [
                "species",
                "beta",
                "p",
                "FDR",
            ]
        ].to_string(
            index=False
        )
    )


    result.to_csv(
        REPORT_DIR
        / f"03_MWAS_{score_name}_model0.csv",
        index=False,
    )


    summary_rows.append(
        {
            "score": score_name,
            "N": len(x_use),
            "species_tested":
                len(species),

            "significant_FDR05":
                n_sig,

            "positive_FDR05":
                n_pos,

            "negative_FDR05":
                n_neg,

            "min_p":
                result["p"].min(),

            "min_FDR":
                result["FDR"].min(),

            "max_positive_beta":
                result["beta"].max(),

            "min_negative_beta":
                result["beta"].min(),
        }
    )


    all_results.append(
        result
    )


# ============================================================
# COMBINED RESULTS
# ============================================================

combined = pd.concat(
    all_results,
    ignore_index=True,
)


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# SHARED SPECIES — 4/4
# ============================================================

wide_fdr = combined.pivot(
    index="species",
    columns="score",
    values="FDR",
)

wide_beta = combined.pivot(
    index="species",
    columns="score",
    values="beta",
)


required_scores = list(
    SCORES.keys()
)


shared_sig = (
    wide_fdr[
        required_scores
    ]
    .lt(0.05)
    .all(axis=1)
)


all_positive = (
    wide_beta[
        required_scores
    ]
    .gt(0)
    .all(axis=1)
)


all_negative = (
    wide_beta[
        required_scores
    ]
    .lt(0)
    .all(axis=1)
)


same_direction = (
    all_positive
    |
    all_negative
)


shared = pd.DataFrame(
    {
        "all4_FDR05":
            shared_sig,

        "all4_same_direction":
            same_direction,

        "all4_positive":
            all_positive,

        "all4_negative":
            all_negative,
    }
)


for score in required_scores:

    shared[
        f"{score}_beta"
    ] = wide_beta[
        score
    ]

    shared[
        f"{score}_FDR"
    ] = wide_fdr[
        score
    ]


shared[
    "shared_4of4_final"
] = (
    shared[
        "all4_FDR05"
    ]
    &
    shared[
        "all4_same_direction"
    ]
)


shared_final = (
    shared.loc[
        shared[
            "shared_4of4_final"
        ]
    ]
    .reset_index()
)


print()
print("=" * 95)
print("SHARED 4/4 SPECIES")
print("=" * 95)

print(
    "FDR <0.05 in all 4:",
    int(
        shared_sig.sum()
    )
)

print(
    "FDR <0.05 + same direction:",
    len(
        shared_final
    )
)

print(
    "All positive:",
    int(
        (
            shared[
                "shared_4of4_final"
            ]
            &
            shared[
                "all4_positive"
            ]
        ).sum()
    )
)

print(
    "All negative:",
    int(
        (
            shared[
                "shared_4of4_final"
            ]
            &
            shared[
                "all4_negative"
            ]
        ).sum()
    )
)


# ============================================================
# PAPER REFERENCE SPECIES AUDIT
# ============================================================

REFERENCE_SPECIES = {

    "Akkermansia muciniphila":
        "Akkermansia_muciniphila",

    "Clostridium phoceensis":
        "Clostridium_phoceensis",

    "Flavonifractor plautii":
        "Flavonifractor_plautii",

    "Dysosmobacter welbionis":
        "Dysosmobacter_welbionis",

    "Ruthenibacterium lactatiformans":
        "Ruthenibacterium_lactatiformans",

    "Bilophila wadsworthia":
        "Bilophila_wadsworthia",

    "Phocea massiliensis":
        "Phocea_massiliensis",
}


reference_rows = []


for display_name, pattern in REFERENCE_SPECIES.items():

    matches = [
        sp
        for sp in species
        if pattern.lower()
        in sp.lower()
    ]


    if not matches:

        reference_rows.append(
            {
                "reference_species":
                    display_name,

                "matched_species":
                    "NOT FOUND",
            }
        )

        continue


    for sp in matches:

        row = {
            "reference_species":
                display_name,

            "matched_species":
                sp,
        }


        for score in required_scores:

            row[
                f"{score}_beta"
            ] = (
                wide_beta.loc[
                    sp,
                    score,
                ]
            )

            row[
                f"{score}_FDR"
            ] = (
                wide_fdr.loc[
                    sp,
                    score,
                ]
            )


        reference_rows.append(
            row
        )


reference_df = pd.DataFrame(
    reference_rows
)


# ============================================================
# SAVE
# ============================================================

combined.to_csv(
    REPORT_DIR
    / "03_MWAS_all_scores_model0.csv",
    index=False,
)


summary.to_csv(
    REPORT_DIR
    / "03_MWAS_summary_model0.csv",
    index=False,
)


shared.reset_index().to_csv(
    REPORT_DIR
    / "03_shared_species_all379_model0.csv",
    index=False,
)


shared_final.to_csv(
    REPORT_DIR
    / "03_shared_4of4_species_model0.csv",
    index=False,
)


reference_df.to_csv(
    REPORT_DIR
    / "03_paper_reference_species_audit_model0.csv",
    index=False,
)


print()
print("=" * 95)
print("SUMMARY")
print("=" * 95)

print(
    summary.to_string(
        index=False
    )
)


print()
print("=" * 95)
print("IMPORTANT INTERPRETATION")
print("=" * 95)

print(
    "This is MODEL 0 (unadjusted)."
)

print(
    "Do NOT directly compare the number "
    "of significant species with the "
    "paper's adjusted Model 2 count "
    "(197-271 species per score)."
)

print(
    "Primary goals here:"
)

print(
    "1. confirm strong species-level "
    "diet-microbiome signal;"
)

print(
    "2. assess cross-score directional "
    "consistency;"
)

print(
    "3. identify a preliminary 4/4 "
    "shared species set."
)


print()
print("=" * 95)
print("SAVED")
print("=" * 95)

print(
    REPORT_DIR
    / "03_MWAS_summary_model0.csv"
)

print(
    REPORT_DIR
    / "03_shared_4of4_species_model0.csv"
)

print(
    REPORT_DIR
    / "03_paper_reference_species_audit_model0.csv"
)