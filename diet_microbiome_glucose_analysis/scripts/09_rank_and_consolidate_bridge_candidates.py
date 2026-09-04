#!/usr/bin/env python3
"""
09 — Consolidate and rank robust Diet -> Microbiome -> CGM bridge candidates

Input
-----
The main input is produced by 08_diet_microbiome_cgm_overlap.py:

    outputs/reports/08_robust_direction_consistent_candidates.csv

Each input row is one robust score–CGM–species path that:
    - is significant in Diet -> Microbiome Model 2
    - is significant in Microbiome -> CGM Model 2
    - remains present in the BMI-sensitive Model 3 layer
    - keeps the Diet->species direction
    - keeps the species->CGM direction
    - is direction-consistent with the biological polarity of the diet score

This script DOES NOT perform mediation analysis.
It consolidates repeated paths into exact species-level candidates and creates
transparent recurrence-based priority tiers for the next indirect-effect step.

Important taxonomic rule
------------------------
The exact full taxonomy string is the species identifier. We DO NOT collapse
species by the terminal species name alone, because identically named terminal
labels under different higher taxonomy can represent different features.

Outputs
-------
1. 09_bridge_species_master.csv
   One row per exact species feature.

2. 09_bridge_path_matrix.csv
   Boolean 379-style path matrix for the retained robust candidate species:
       species x 12 diet-score/CGM combinations.

3. 09_bridge_beta_product_matrix_model2.csv
   Primary Model-2 beta-product matrix.

4. 09_bridge_beta_product_matrix_model3.csv
   BMI-sensitive Model-3 beta-product matrix.

5. 09_bridge_candidate_paths_long.csv
   Long-form robust path table with useful labels/ranks.

6. 09_bridge_tier1_candidates.csv
   Highest-recurrence candidates.

7. 09_bridge_tier2_candidates.csv
   Moderate-recurrence candidates.

8. 09_bridge_tier3_candidates.csv
   Single robust-path candidates.

9. 09_bridge_network_edges.csv
   Edge-list-like table convenient for later Sankey/network plotting.

10. 09_bridge_summary.csv
    Overall counts.

11. 09_bridge_summary.txt
    Human-readable audit summary.

Priority tiers
--------------
TIER 1 — cross-context recurrent
    n_diet_scores >= 3
    OR
    n_cgm_outcomes >= 2

TIER 2 — repeated within one CGM context
    not Tier 1
    AND n_robust_paths >= 2

TIER 3 — single robust path
    n_robust_paths == 1

These tiers are intentionally recurrence-based.
Effect magnitude and FDR are reported as secondary descriptors and are NOT used
to manufacture a composite "biological importance score".

Run
---
    python 09_rank_and_consolidate_bridge_candidates.py --self-test
    python 09_rank_and_consolidate_bridge_candidates.py
"""

from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
ANALYSIS_DIR = ROOT / "diet_microbiome_glucose_analysis"

DEFAULT_INPUT = (
    ANALYSIS_DIR
    / "outputs"
    / "reports"
    / "08_robust_direction_consistent_candidates.csv"
)

OUT_REPORT_DIR = ANALYSIS_DIR / "outputs" / "reports"
OUT_DATA_DIR = ANALYSIS_DIR / "outputs" / "data"

SCORES = ["AHEI", "AMED", "hPDI", "rEDIH"]
OUTCOMES = ["mean_glucose", "glucose_cv", "time_above_140"]

PATH_ORDER = [
    f"{score}__{outcome}"
    for score in SCORES
    for outcome in OUTCOMES
]

REQUIRED_COLUMNS = [
    "diet_score",
    "cgm_outcome",
    "species",
    "primary_beta_diet",
    "primary_beta_cgm",
    "primary_beta_product",
    "primary_FDR_diet",
    "primary_FDR_cgm",
    "bmi_beta_diet",
    "bmi_beta_cgm",
    "bmi_beta_product",
    "bmi_FDR_diet",
    "bmi_FDR_cgm",
    "robust_overlap_both_models",
    "diet_beta_same_direction_M2_M3",
    "cgm_beta_same_direction_M2_M3",
    "path_product_same_direction_M2_M3",
    "robust_direction_consistent_candidate",
]


def truthy(series):
    """
    Convert common CSV boolean encodings to real booleans.

    Supported encodings include:
      True / False
      1 / 0
      1.0 / 0.0
      yes / no
      y / n
      t / f

    Missing or unexpected values raise an error instead of being silently
    converted, because these columns are QC flags.
    """
    if pd.api.types.is_bool_dtype(series):
        if series.isna().any():
            raise RuntimeError("Boolean QC flag contains missing values.")
        return series.astype(bool)

    # First handle numeric-like values. pandas often reads CSV boolean flags
    # written as 1/0 as float values 1.0/0.0.
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        bad_mask = ~numeric.isin([0.0, 1.0])
        if bad_mask.any():
            bad = (
                series.loc[bad_mask]
                .astype(str)
                .value_counts()
                .head(20)
                .to_dict()
            )
            raise RuntimeError(
                f"Could not parse numeric boolean values: {bad}"
            )
        return numeric.eq(1.0)

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "1.0": True,
        "0.0": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
        "t": True,
        "f": False,
    }

    converted = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    if converted.isna().any():
        bad = (
            series.loc[converted.isna()]
            .astype(str)
            .value_counts()
            .head(20)
            .to_dict()
        )
        raise RuntimeError(
            f"Could not parse boolean values: {bad}"
        )

    return converted.astype(bool)


def extract_species_label(full_taxonomy):
    """
    Produce a display label only.

    Full taxonomy remains the identifier used for joins and uniqueness.
    """
    s = str(full_taxonomy).strip()

    # Prefer the last explicit s__ segment.
    parts = s.split("|")
    for part in reversed(parts):
        part = part.strip()
        if part.startswith("s__"):
            label = part[3:].strip()
            return label if label else part

    # Fallback: terminal path segment.
    return parts[-1].strip() if parts else s


def safe_list_join(values, order=None):
    vals = [str(v) for v in pd.unique(pd.Series(values).dropna())]

    if order is not None:
        rank = {v: i for i, v in enumerate(order)}
        vals = sorted(vals, key=lambda x: rank.get(x, 10_000))
    else:
        vals = sorted(vals)

    return ";".join(vals)


def geom_mean_two_fdr(a, b):
    """
    Descriptive only: geometric mean of two FDR values.
    Not a combined P value and not used for significance testing.
    """
    a = max(float(a), np.finfo(float).tiny)
    b = max(float(b), np.finfo(float).tiny)
    return float(np.sqrt(a * b))


def validate_input(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing:
        raise RuntimeError(
            "08 robust candidate file is missing required columns:\n"
            + "\n".join(f"  - {x}" for x in missing)
        )

    if df.empty:
        raise RuntimeError("08 robust candidate file is empty.")

    if not set(df["diet_score"]).issubset(set(SCORES)):
        bad = sorted(set(df["diet_score"]) - set(SCORES))
        raise RuntimeError(f"Unexpected diet score(s): {bad}")

    if not set(df["cgm_outcome"]).issubset(set(OUTCOMES)):
        bad = sorted(set(df["cgm_outcome"]) - set(OUTCOMES))
        raise RuntimeError(f"Unexpected CGM outcome(s): {bad}")

    key = ["diet_score", "cgm_outcome", "species"]

    if df.duplicated(key).any():
        dup = (
            df.loc[df.duplicated(key, keep=False), key]
            .value_counts()
            .head(20)
        )
        raise RuntimeError(
            "Duplicate score–outcome–species paths found:\n"
            + dup.to_string()
        )

    for col in [
        "robust_overlap_both_models",
        "diet_beta_same_direction_M2_M3",
        "cgm_beta_same_direction_M2_M3",
        "path_product_same_direction_M2_M3",
        "robust_direction_consistent_candidate",
    ]:
        df[col] = truthy(df[col])

    # This input is supposed to contain only robust direction-consistent rows.
    for col in [
        "robust_overlap_both_models",
        "diet_beta_same_direction_M2_M3",
        "cgm_beta_same_direction_M2_M3",
        "path_product_same_direction_M2_M3",
        "robust_direction_consistent_candidate",
    ]:
        if not df[col].all():
            n_bad = int((~df[col]).sum())
            raise RuntimeError(
                f"Input contains {n_bad} row(s) with {col}=False. "
                "Use the dedicated 08 robust-candidate output, not a broader table."
            )

    numeric_cols = [
        "primary_beta_diet",
        "primary_beta_cgm",
        "primary_beta_product",
        "primary_FDR_diet",
        "primary_FDR_cgm",
        "bmi_beta_diet",
        "bmi_beta_cgm",
        "bmi_beta_product",
        "bmi_FDR_diet",
        "bmi_FDR_cgm",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

        if df[col].isna().any():
            raise RuntimeError(
                f"Non-numeric or missing values found in {col}."
            )

    # Recalculate product signs as a hard QC.
    calc_primary = df["primary_beta_diet"] * df["primary_beta_cgm"]
    calc_bmi = df["bmi_beta_diet"] * df["bmi_beta_cgm"]

    if not np.allclose(
        calc_primary,
        df["primary_beta_product"],
        rtol=1e-7,
        atol=1e-12,
    ):
        max_diff = float(
            np.max(np.abs(calc_primary - df["primary_beta_product"]))
        )
        raise RuntimeError(
            "primary_beta_product does not equal beta_diet * beta_cgm. "
            f"Maximum absolute difference = {max_diff}"
        )

    if not np.allclose(
        calc_bmi,
        df["bmi_beta_product"],
        rtol=1e-7,
        atol=1e-12,
    ):
        max_diff = float(
            np.max(np.abs(calc_bmi - df["bmi_beta_product"]))
        )
        raise RuntimeError(
            "bmi_beta_product does not equal beta_diet * beta_cgm. "
            f"Maximum absolute difference = {max_diff}"
        )

    return df


def classify_tier(n_paths, n_scores, n_outcomes):
    if n_scores >= 3 or n_outcomes >= 2:
        return "Tier1_cross_context_recurrent"

    if n_paths >= 2:
        return "Tier2_repeated"

    return "Tier3_single_robust_path"


def build_long_table(df):
    out = df.copy()

    out["species_label"] = out["species"].map(extract_species_label)
    out["path_id"] = (
        out["diet_score"].astype(str)
        + "__"
        + out["cgm_outcome"].astype(str)
    )

    out["primary_abs_beta_product"] = (
        out["primary_beta_product"].abs()
    )

    out["bmi_abs_beta_product"] = (
        out["bmi_beta_product"].abs()
    )

    out["primary_path_fdr_geomean_descriptive"] = [
        geom_mean_two_fdr(a, b)
        for a, b in zip(
            out["primary_FDR_diet"],
            out["primary_FDR_cgm"],
        )
    ]

    out["bmi_path_fdr_geomean_descriptive"] = [
        geom_mean_two_fdr(a, b)
        for a, b in zip(
            out["bmi_FDR_diet"],
            out["bmi_FDR_cgm"],
        )
    ]

    return out


def build_species_master(long_df):
    rows = []

    for species, g in long_df.groupby("species", sort=False):
        n_paths = int(len(g))
        n_scores = int(g["diet_score"].nunique())
        n_outcomes = int(g["cgm_outcome"].nunique())

        row = {
            "species": species,
            "species_label": extract_species_label(species),
            "n_robust_paths": n_paths,
            "n_diet_scores": n_scores,
            "n_cgm_outcomes": n_outcomes,
            "diet_scores": safe_list_join(
                g["diet_score"],
                order=SCORES,
            ),
            "cgm_outcomes": safe_list_join(
                g["cgm_outcome"],
                order=OUTCOMES,
            ),
            "path_ids": safe_list_join(
                g["path_id"],
                order=PATH_ORDER,
            ),
            "priority_tier": classify_tier(
                n_paths,
                n_scores,
                n_outcomes,
            ),
            "primary_min_FDR_diet": float(
                g["primary_FDR_diet"].min()
            ),
            "primary_min_FDR_cgm": float(
                g["primary_FDR_cgm"].min()
            ),
            "bmi_min_FDR_diet": float(
                g["bmi_FDR_diet"].min()
            ),
            "bmi_min_FDR_cgm": float(
                g["bmi_FDR_cgm"].min()
            ),
            "primary_mean_abs_beta_product": float(
                g["primary_abs_beta_product"].mean()
            ),
            "primary_max_abs_beta_product": float(
                g["primary_abs_beta_product"].max()
            ),
            "bmi_mean_abs_beta_product": float(
                g["bmi_abs_beta_product"].mean()
            ),
            "bmi_max_abs_beta_product": float(
                g["bmi_abs_beta_product"].max()
            ),
            "primary_all_path_products_same_sign_as_bmi": bool(
                np.all(
                    np.sign(g["primary_beta_product"])
                    == np.sign(g["bmi_beta_product"])
                )
            ),
            "all_diet_effects_same_sign_M2_M3": bool(
                np.all(
                    np.sign(g["primary_beta_diet"])
                    == np.sign(g["bmi_beta_diet"])
                )
            ),
            "all_cgm_effects_same_sign_M2_M3": bool(
                np.all(
                    np.sign(g["primary_beta_cgm"])
                    == np.sign(g["bmi_beta_cgm"])
                )
            ),
        }

        # Add per-score recurrence flags.
        for score in SCORES:
            row[f"has_{score}"] = bool(
                g["diet_score"].eq(score).any()
            )

        # Add per-outcome recurrence flags.
        for outcome in OUTCOMES:
            row[f"has_{outcome}"] = bool(
                g["cgm_outcome"].eq(outcome).any()
            )

        rows.append(row)

    master = pd.DataFrame(rows)

    tier_rank = {
        "Tier1_cross_context_recurrent": 1,
        "Tier2_repeated": 2,
        "Tier3_single_robust_path": 3,
    }

    master["tier_rank"] = master["priority_tier"].map(tier_rank)

    # Transparent ranking:
    # recurrence first; effect/FDR only break ties.
    master = master.sort_values(
        [
            "tier_rank",
            "n_cgm_outcomes",
            "n_diet_scores",
            "n_robust_paths",
            "primary_min_FDR_diet",
            "primary_min_FDR_cgm",
            "primary_mean_abs_beta_product",
        ],
        ascending=[
            True,
            False,
            False,
            False,
            True,
            True,
            False,
        ],
    ).reset_index(drop=True)

    master.insert(
        0,
        "candidate_rank",
        np.arange(1, len(master) + 1),
    )

    return master


def build_path_matrix(long_df, value_col=None):
    x = long_df.copy()

    x["path_id"] = pd.Categorical(
        x["path_id"],
        categories=PATH_ORDER,
        ordered=True,
    )

    if value_col is None:
        matrix = (
            x.assign(present=True)
            .pivot(
                index=["species", "species_label"],
                columns="path_id",
                values="present",
            )
            .fillna(False)
            .astype(bool)
        )
    else:
        matrix = x.pivot(
            index=["species", "species_label"],
            columns="path_id",
            values=value_col,
        )

    matrix = matrix.reindex(columns=PATH_ORDER)
    matrix.columns.name = None
    return matrix.reset_index()


def build_network_edges(long_df, species_master):
    rank_map = (
        species_master
        .set_index("species")["candidate_rank"]
        .to_dict()
    )

    tier_map = (
        species_master
        .set_index("species")["priority_tier"]
        .to_dict()
    )

    edges = long_df[
        [
            "diet_score",
            "species",
            "species_label",
            "cgm_outcome",
            "primary_beta_diet",
            "primary_beta_cgm",
            "primary_beta_product",
            "bmi_beta_diet",
            "bmi_beta_cgm",
            "bmi_beta_product",
            "primary_FDR_diet",
            "primary_FDR_cgm",
            "bmi_FDR_diet",
            "bmi_FDR_cgm",
        ]
    ].copy()

    edges["candidate_rank"] = edges["species"].map(rank_map)
    edges["priority_tier"] = edges["species"].map(tier_map)

    # Separate edge directions for future plotting.
    edges["diet_to_species_direction_M2"] = np.where(
        edges["primary_beta_diet"] > 0,
        "positive",
        "negative",
    )

    edges["species_to_cgm_direction_M2"] = np.where(
        edges["primary_beta_cgm"] > 0,
        "positive",
        "negative",
    )

    edges["diet_to_species_direction_M3"] = np.where(
        edges["bmi_beta_diet"] > 0,
        "positive",
        "negative",
    )

    edges["species_to_cgm_direction_M3"] = np.where(
        edges["bmi_beta_cgm"] > 0,
        "positive",
        "negative",
    )

    return edges.sort_values(
        ["candidate_rank", "diet_score", "cgm_outcome"]
    )


def build_summary(long_df, master):
    rows = []

    rows.append({
        "section": "overall",
        "item": "robust_path_rows",
        "count": len(long_df),
    })

    rows.append({
        "section": "overall",
        "item": "unique_exact_species",
        "count": master["species"].nunique(),
    })

    for tier in [
        "Tier1_cross_context_recurrent",
        "Tier2_repeated",
        "Tier3_single_robust_path",
    ]:
        rows.append({
            "section": "tier",
            "item": tier,
            "count": int(master["priority_tier"].eq(tier).sum()),
        })

    for score in SCORES:
        rows.append({
            "section": "diet_score",
            "item": score,
            "count": int(long_df["diet_score"].eq(score).sum()),
        })

    for outcome in OUTCOMES:
        rows.append({
            "section": "cgm_outcome",
            "item": outcome,
            "count": int(long_df["cgm_outcome"].eq(outcome).sum()),
        })

    # Exact number of unique species represented in each phenotype.
    for outcome in OUTCOMES:
        rows.append({
            "section": "unique_species_by_outcome",
            "item": outcome,
            "count": int(
                long_df.loc[
                    long_df["cgm_outcome"].eq(outcome),
                    "species",
                ].nunique()
            ),
        })

    return pd.DataFrame(rows)


def self_test():
    print("=" * 90)
    print("SELF TEST — BRIDGE CANDIDATE CONSOLIDATION")
    print("=" * 90)

    rows = [
        # Species A: 3 scores, 1 outcome -> Tier1
        ("AHEI", "glucose_cv", "k__Bacteria|s__Species_A"),
        ("AMED", "glucose_cv", "k__Bacteria|s__Species_A"),
        ("hPDI", "glucose_cv", "k__Bacteria|s__Species_A"),

        # Species B: 1 score, 2 outcomes -> Tier1
        ("rEDIH", "mean_glucose", "k__Bacteria|s__Species_B"),
        ("rEDIH", "glucose_cv", "k__Bacteria|s__Species_B"),

        # Species C: 2 scores, 1 outcome -> Tier2
        ("AHEI", "mean_glucose", "k__Bacteria|s__Species_C"),
        ("AMED", "mean_glucose", "k__Bacteria|s__Species_C"),

        # Species D: single robust path -> Tier3
        ("hPDI", "time_above_140", "k__Bacteria|s__Species_D"),
    ]

    data = []

    for score, outcome, species in rows:
        if score == "rEDIH":
            beta_diet = 0.2
            beta_cgm = 0.3
        else:
            beta_diet = 0.2
            beta_cgm = -0.3

        item = {
            "diet_score": score,
            "cgm_outcome": outcome,
            "species": species,
            "primary_beta_diet": beta_diet,
            "primary_beta_cgm": beta_cgm,
            "primary_beta_product": beta_diet * beta_cgm,
            "primary_FDR_diet": 0.01,
            "primary_FDR_cgm": 0.02,
            "bmi_beta_diet": beta_diet * 0.9,
            "bmi_beta_cgm": beta_cgm * 0.9,
            "bmi_beta_product": (beta_diet * 0.9) * (beta_cgm * 0.9),
            "bmi_FDR_diet": 0.015,
            "bmi_FDR_cgm": 0.025,
            "robust_overlap_both_models": 1.0,
            "diet_beta_same_direction_M2_M3": 1.0,
            "cgm_beta_same_direction_M2_M3": 1.0,
            "path_product_same_direction_M2_M3": 1.0,
            "robust_direction_consistent_candidate": 1.0,
        }
        data.append(item)

    df = pd.DataFrame(data)
    df = validate_input(df)
    long_df = build_long_table(df)
    master = build_species_master(long_df)

    got = (
        master.set_index("species_label")["priority_tier"].to_dict()
    )

    assert got["Species_A"] == "Tier1_cross_context_recurrent"
    assert got["Species_B"] == "Tier1_cross_context_recurrent"
    assert got["Species_C"] == "Tier2_repeated"
    assert got["Species_D"] == "Tier3_single_robust_path"

    matrix = build_path_matrix(long_df)

    assert len(matrix) == 4
    assert set(PATH_ORDER).issubset(matrix.columns)

    assert int(master["n_robust_paths"].sum()) == len(long_df)

    print("Tier classification: PASS")
    print("Exact taxonomy uniqueness: PASS")
    print("12-path matrix structure: PASS")
    print("Path-count conservation: PASS")
    print("SELF TEST: PASS")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=(
            "08 robust direction-consistent candidate CSV. "
            f"Default: {DEFAULT_INPUT}"
        ),
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{input_path}"
        )

    OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("09 — CONSOLIDATE ROBUST DIET -> MICROBIOME -> CGM CANDIDATES")
    print("=" * 100)
    print("Input:")
    print(input_path)
    print()
    print(
        "Exact full taxonomy strings are preserved as species identifiers."
    )
    print(
        "Priority is recurrence-based; no arbitrary composite importance score is used."
    )

    df = pd.read_csv(input_path, low_memory=False)
    df = validate_input(df)

    long_df = build_long_table(df)
    master = build_species_master(long_df)

    path_matrix = build_path_matrix(long_df)
    beta_m2_matrix = build_path_matrix(
        long_df,
        value_col="primary_beta_product",
    )
    beta_m3_matrix = build_path_matrix(
        long_df,
        value_col="bmi_beta_product",
    )

    network_edges = build_network_edges(
        long_df,
        master,
    )

    summary = build_summary(
        long_df,
        master,
    )

    # Attach species-level rank and tier back onto long form.
    metadata = master[
        [
            "species",
            "candidate_rank",
            "priority_tier",
            "n_robust_paths",
            "n_diet_scores",
            "n_cgm_outcomes",
        ]
    ]

    long_ranked = long_df.merge(
        metadata,
        on="species",
        how="left",
        validate="many_to_one",
    ).sort_values(
        [
            "candidate_rank",
            "diet_score",
            "cgm_outcome",
        ]
    )

    tier1 = master.loc[
        master["priority_tier"].eq(
            "Tier1_cross_context_recurrent"
        )
    ].copy()

    tier2 = master.loc[
        master["priority_tier"].eq(
            "Tier2_repeated"
        )
    ].copy()

    tier3 = master.loc[
        master["priority_tier"].eq(
            "Tier3_single_robust_path"
        )
    ].copy()

    paths = {
        "master":
            OUT_REPORT_DIR / "09_bridge_species_master.csv",
        "matrix":
            OUT_REPORT_DIR / "09_bridge_path_matrix.csv",
        "beta_m2":
            OUT_REPORT_DIR / "09_bridge_beta_product_matrix_model2.csv",
        "beta_m3":
            OUT_REPORT_DIR / "09_bridge_beta_product_matrix_model3.csv",
        "long":
            OUT_REPORT_DIR / "09_bridge_candidate_paths_long.csv",
        "tier1":
            OUT_REPORT_DIR / "09_bridge_tier1_candidates.csv",
        "tier2":
            OUT_REPORT_DIR / "09_bridge_tier2_candidates.csv",
        "tier3":
            OUT_REPORT_DIR / "09_bridge_tier3_candidates.csv",
        "edges":
            OUT_REPORT_DIR / "09_bridge_network_edges.csv",
        "summary":
            OUT_REPORT_DIR / "09_bridge_summary.csv",
        "txt":
            OUT_REPORT_DIR / "09_bridge_summary.txt",
    }

    master.to_csv(paths["master"], index=False)
    path_matrix.to_csv(paths["matrix"], index=False)
    beta_m2_matrix.to_csv(paths["beta_m2"], index=False)
    beta_m3_matrix.to_csv(paths["beta_m3"], index=False)
    long_ranked.to_csv(paths["long"], index=False)
    tier1.to_csv(paths["tier1"], index=False)
    tier2.to_csv(paths["tier2"], index=False)
    tier3.to_csv(paths["tier3"], index=False)
    network_edges.to_csv(paths["edges"], index=False)
    summary.to_csv(paths["summary"], index=False)

    lines = []

    lines.append("=" * 100)
    lines.append("09 — BRIDGE CANDIDATE CONSOLIDATION SUMMARY")
    lines.append("=" * 100)
    lines.append(f"Input robust path rows: {len(long_df)}")
    lines.append(
        f"Unique exact species features: {master['species'].nunique()}"
    )
    lines.append("")
    lines.append("PRIORITY TIERS")
    lines.append("-" * 100)
    lines.append(
        "Tier1_cross_context_recurrent: "
        + str(len(tier1))
    )
    lines.append(
        "Tier2_repeated: "
        + str(len(tier2))
    )
    lines.append(
        "Tier3_single_robust_path: "
        + str(len(tier3))
    )
    lines.append("")
    lines.append("ROBUST PATHS BY DIET SCORE")
    lines.append("-" * 100)

    for score in SCORES:
        n = int(long_df["diet_score"].eq(score).sum())
        u = int(
            long_df.loc[
                long_df["diet_score"].eq(score),
                "species",
            ].nunique()
        )
        lines.append(f"{score}: paths={n}, unique_species={u}")

    lines.append("")
    lines.append("ROBUST PATHS BY CGM OUTCOME")
    lines.append("-" * 100)

    for outcome in OUTCOMES:
        n = int(long_df["cgm_outcome"].eq(outcome).sum())
        u = int(
            long_df.loc[
                long_df["cgm_outcome"].eq(outcome),
                "species",
            ].nunique()
        )
        lines.append(
            f"{outcome}: paths={n}, unique_species={u}"
        )

    lines.append("")
    lines.append("TOP RECURRENT EXACT SPECIES FEATURES")
    lines.append("-" * 100)

    top_cols = [
        "candidate_rank",
        "species_label",
        "n_robust_paths",
        "n_diet_scores",
        "n_cgm_outcomes",
        "diet_scores",
        "cgm_outcomes",
        "priority_tier",
        "primary_min_FDR_diet",
        "primary_min_FDR_cgm",
    ]

    lines.append(
        master[top_cols]
        .head(30)
        .to_string(index=False)
    )

    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 100)
    lines.append(
        "Tier assignment is recurrence-based only. "
        "Tier 1 indicates repeated support across >=3 diet scores "
        "or >=2 CGM phenotypes."
    )
    lines.append(
        "A high tier is not evidence of causal mediation. "
        "Formal indirect-effect analysis is the next step."
    )
    lines.append(
        "The terminal species label is display-only. "
        "All computation and uniqueness use the full taxonomy string."
    )

    paths["txt"].write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("CORE COUNTS")
    print("=" * 100)
    print(f"Robust score-CGM-species paths: {len(long_df)}")
    print(
        f"Unique exact species features: {master['species'].nunique()}"
    )
    print(
        f"Tier 1 cross-context recurrent: {len(tier1)}"
    )
    print(
        f"Tier 2 repeated: {len(tier2)}"
    )
    print(
        f"Tier 3 single robust path: {len(tier3)}"
    )

    print()
    print("=" * 100)
    print("BY CGM OUTCOME")
    print("=" * 100)

    for outcome in OUTCOMES:
        subset = long_df.loc[
            long_df["cgm_outcome"].eq(outcome)
        ]

        print(
            f"{outcome}: "
            f"paths={len(subset)}, "
            f"unique_species={subset['species'].nunique()}"
        )

    print()
    print("=" * 100)
    print("TOP 30 RECURRENT CANDIDATES")
    print("=" * 100)
    print(
        master[
            [
                "candidate_rank",
                "species_label",
                "n_robust_paths",
                "n_diet_scores",
                "n_cgm_outcomes",
                "diet_scores",
                "cgm_outcomes",
                "priority_tier",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )

    print()
    print("=" * 100)
    print("NEXT-STEP READINESS")
    print("=" * 100)

    if len(tier1):
        print(
            "Tier 1 recurrent candidates exist. "
            "These should be the first candidates considered for "
            "formal indirect-effect / mediation-style analysis."
        )
    else:
        print(
            "No Tier 1 recurrent candidates. "
            "Proceed using Tier 2 robust repeated candidates."
        )

    print(
        "glucose_cv and mean_glucose should remain the main CGM outcomes; "
        "time_above_140 can remain exploratory unless recurrence is stronger than expected."
    )

    print()
    print("=" * 100)
    print("SAVED")
    print("=" * 100)

    for p in paths.values():
        print(p)


if __name__ == "__main__":
    main()
