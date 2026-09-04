#!/usr/bin/env python3
"""
08 — Diet -> Microbiome -> CGM overlap / direction-consistency analysis

Purpose
-------
Connect the two association layers already completed:

    Diet score -> species
    species -> CGM phenotype

This script DOES NOT perform mediation analysis.
It performs a transparent overlap and direction-consistency screen.

Primary analysis:
    Diet -> microbiome Model 2
    Microbiome -> CGM Model 2

BMI sensitivity:
    Diet -> microbiome Model 3 (+ BMI)
    Microbiome -> CGM Model 3 (+ BMI)

For every 4 x 3 = 12 diet-score / CGM-outcome pair:
    1. count significant diet-associated species
    2. count significant CGM-associated species
    3. intersect the two sets
    4. report beta_diet, beta_cgm, and beta-product direction
    5. classify whether the direction matches the expected score polarity
    6. optionally assess overlap enrichment using a hypergeometric test
    7. compare primary Model 2 candidates with BMI-sensitive Model 3 candidates

Expected score polarity
-----------------------
Higher AHEI / AMED / hPDI = healthier direction
    For adverse CGM outcomes (mean glucose, glucose CV, time >140),
    an expected indirect direction is beta_diet * beta_cgm < 0.

Higher rEDIH = more hyperinsulinemic / metabolically adverse direction
    For adverse CGM outcomes,
    an expected indirect direction is beta_diet * beta_cgm > 0.

Important:
    "direction-consistent" here means sign-consistent with the score polarity.
    It is NOT proof of mediation or causality.

Run
---
    python 08_diet_microbiome_cgm_overlap.py --self-test
    python 08_diet_microbiome_cgm_overlap.py

If diet-model files cannot be uniquely auto-detected:
    python 08_diet_microbiome_cgm_overlap.py \
        --diet-model2 /path/to/diet_model2_all_species.csv \
        --diet-model3 /path/to/diet_model3_all_species.csv
"""

from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")

DIET_ANALYSIS_DIR = ROOT / "diet_microbiome_analysis"
CGM_ANALYSIS_DIR = ROOT / "diet_microbiome_glucose_analysis"

OUT_DATA_DIR = CGM_ANALYSIS_DIR / "outputs" / "data"
OUT_REPORT_DIR = CGM_ANALYSIS_DIR / "outputs" / "reports"

CGM_MODEL2 = (
    CGM_ANALYSIS_DIR
    / "outputs"
    / "models"
    / "06_microbiome_cgm_all_model2.csv"
)

CGM_MODEL3 = (
    CGM_ANALYSIS_DIR
    / "outputs"
    / "models"
    / "07_microbiome_cgm_all_model3_bmi.csv"
)

EXPECTED_SPECIES = 379

SCORES = ["AHEI", "AMED", "hPDI", "rEDIH"]
OUTCOMES = ["mean_glucose", "glucose_cv", "time_above_140"]

# +1 means higher score is expected to be metabolically adverse;
# -1 means higher score is expected to be metabolically favorable.
EXPECTED_CGM_DIRECTION = {
    "AHEI": -1,
    "AMED": -1,
    "hPDI": -1,
    "rEDIH": +1,
}

SCORE_ALIASES = {
    "ahei": "AHEI",
    "mAHEI7".lower(): "AHEI",
    "amed": "AMED",
    "hpdi": "hPDI",
    "h_pdi": "hPDI",
    "redih": "rEDIH",
    "r_edih": "rEDIH",
}

# Candidate aliases for diet->microbiome detailed result tables.
DIET_SCORE_COL_ALIASES = [
    "score",
    "diet_score",
    "score_name",
    "diet_index",
    "exposure",
]

DIET_SPECIES_COL_ALIASES = [
    "species",
    "taxon",
    "feature",
]

DIET_BETA_COL_ALIASES = [
    "beta_diet",
    "beta_score",
    "beta",
    "coefficient",
    "coef",
    "estimate",
]

DIET_P_COL_ALIASES = [
    "p_value",
    "p",
    "pvalue",
]

DIET_FDR_COL_ALIASES = [
    "FDR",
    "fdr",
    "FDR05",
    "FDR_BH",
    "FDR_BH_within_score",
    "FDR_within_score",
    "FDR_within_score379",
    "q_value",
    "qvalue",
    "adj_p",
    "p_adj",
]


def bh_fdr(pvalues):
    p = np.asarray(pvalues, dtype=float)

    if len(p) == 0:
        return np.array([], dtype=float)

    if not np.isfinite(p).all():
        raise RuntimeError("BH-FDR received NaN/Inf.")

    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)

    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    out = np.empty_like(q)
    out[order] = q
    return out


def normalize_score(value):
    s = str(value).strip()
    compact = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()

    if compact in SCORE_ALIASES:
        return SCORE_ALIASES[compact]

    # Tolerate common longer labels.
    if "ahei" in compact:
        return "AHEI"
    if "amed" in compact:
        return "AMED"
    if "hpdi" in compact or "h_pdi" in compact:
        return "hPDI"
    if "redih" in compact or "r_edih" in compact:
        return "rEDIH"

    return s


def choose_column(columns, aliases):
    columns = list(columns)

    # exact match first
    for alias in aliases:
        if alias in columns:
            return alias

    lower_map = {str(c).lower(): c for c in columns}

    for alias in aliases:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]

    return None


def read_header(path):
    try:
        return pd.read_csv(path, nrows=0).columns.tolist()
    except Exception:
        return []


def candidate_diet_files(model_number):
    """
    Find plausible all-species diet->microbiome result files.

    We intentionally do not silently pick arbitrary CSVs.
    Candidates must:
      - be under diet_microbiome_analysis
      - contain a species-like column
      - contain beta and FDR-like information
      - have filename/path suggest Model 2 or Model 3
    """

    if not DIET_ANALYSIS_DIR.exists():
        raise FileNotFoundError(
            f"Diet analysis directory not found:\n{DIET_ANALYSIS_DIR}"
        )

    model_tokens = (
        ["model2", "model_2", "model-2"]
        if model_number == 2
        else ["model3", "model_3", "model-3", "bmi"]
    )

    found = []

    for path in DIET_ANALYSIS_DIR.rglob("*.csv"):
        name = str(path).lower()

        if not any(tok in name for tok in model_tokens):
            continue

        cols = read_header(path)
        if not cols:
            continue

        species_col = choose_column(cols, DIET_SPECIES_COL_ALIASES)
        beta_col = choose_column(cols, DIET_BETA_COL_ALIASES)
        fdr_col = choose_column(cols, DIET_FDR_COL_ALIASES)
        p_col = choose_column(cols, DIET_P_COL_ALIASES)
        score_col = choose_column(cols, DIET_SCORE_COL_ALIASES)

        if species_col and beta_col and (fdr_col or p_col):
            found.append(
                {
                    "path": path,
                    "score_col": score_col,
                    "species_col": species_col,
                    "beta_col": beta_col,
                    "fdr_col": fdr_col,
                    "p_col": p_col,
                }
            )

    return found


def load_single_diet_file(path):
    df = pd.read_csv(path, low_memory=False)

    score_col = choose_column(df.columns, DIET_SCORE_COL_ALIASES)
    species_col = choose_column(df.columns, DIET_SPECIES_COL_ALIASES)
    beta_col = choose_column(df.columns, DIET_BETA_COL_ALIASES)
    fdr_col = choose_column(df.columns, DIET_FDR_COL_ALIASES)
    p_col = choose_column(df.columns, DIET_P_COL_ALIASES)

    if not species_col:
        raise RuntimeError(f"{path}: no species column found.")
    if not beta_col:
        raise RuntimeError(f"{path}: no beta column found.")
    if not fdr_col and not p_col:
        raise RuntimeError(f"{path}: neither FDR nor p-value column found.")

    out = pd.DataFrame({
        "species": df[species_col].astype(str),
        "beta_diet": pd.to_numeric(df[beta_col], errors="coerce"),
    })

    if score_col:
        out["score"] = df[score_col].map(normalize_score)
    else:
        # Infer score from filename when files are split by score.
        normalized_name = normalize_score(path.stem)
        matched = None
        for score in SCORES:
            if score.lower() in path.stem.lower():
                matched = score
                break

        if matched is None:
            # More tolerant checks.
            stem = path.stem.lower()
            if "ahei" in stem:
                matched = "AHEI"
            elif "amed" in stem:
                matched = "AMED"
            elif "hpdi" in stem:
                matched = "hPDI"
            elif "redih" in stem:
                matched = "rEDIH"

        if matched is None:
            raise RuntimeError(
                f"{path}: no score column and score cannot be inferred "
                "from filename."
            )

        out["score"] = matched

    if fdr_col:
        out["FDR_diet"] = pd.to_numeric(df[fdr_col], errors="coerce")
    else:
        out["FDR_diet"] = np.nan

    if p_col:
        out["p_diet"] = pd.to_numeric(df[p_col], errors="coerce")
    else:
        out["p_diet"] = np.nan

    out["source_file"] = str(path)

    out = out.loc[out["score"].isin(SCORES)].copy()

    if out["beta_diet"].isna().any():
        raise RuntimeError(f"{path}: beta contains missing/non-numeric values.")

    return out


def validate_and_finalize_diet_table(df, model_label):
    """
    Require a complete 379-species table for every one of the four scores.
    Recompute within-score BH FDR from p-values only when FDR is absent.
    """

    needed = {"score", "species", "beta_diet", "FDR_diet", "p_diet"}
    missing = needed.difference(df.columns)

    if missing:
        raise RuntimeError(
            f"{model_label}: standardized diet table missing {missing}"
        )

    pieces = []

    for score in SCORES:
        piece = df.loc[df["score"].eq(score)].copy()

        # If the same file set introduced duplicates, fail rather than guess.
        if piece["species"].duplicated().any():
            dup = (
                piece.loc[
                    piece["species"].duplicated(keep=False),
                    "species",
                ]
                .value_counts()
                .head(20)
                .to_dict()
            )
            raise RuntimeError(
                f"{model_label} {score}: duplicate species rows: {dup}"
            )

        if len(piece) != EXPECTED_SPECIES:
            raise RuntimeError(
                f"{model_label} {score}: expected "
                f"{EXPECTED_SPECIES} species rows, got {len(piece)}."
            )

        if piece["FDR_diet"].isna().all():
            if piece["p_diet"].isna().any():
                raise RuntimeError(
                    f"{model_label} {score}: FDR absent and p-values incomplete."
                )

            piece["FDR_diet"] = bh_fdr(piece["p_diet"].to_numpy(float))

        elif piece["FDR_diet"].isna().any():
            raise RuntimeError(
                f"{model_label} {score}: FDR column is partially missing."
            )

        piece["diet_significant_FDR05"] = piece["FDR_diet"] < 0.05
        pieces.append(piece)

    result = pd.concat(pieces, ignore_index=True)

    universes = {
        score: set(result.loc[result["score"].eq(score), "species"])
        for score in SCORES
    }

    reference = universes[SCORES[0]]

    for score in SCORES[1:]:
        if universes[score] != reference:
            raise RuntimeError(
                f"{model_label}: species universe differs between "
                f"{SCORES[0]} and {score}."
            )

    return result


def load_diet_model(explicit_path, model_number):
    model_label = f"Diet Model {model_number}"

    if explicit_path is not None:
        path = Path(explicit_path)

        if not path.exists():
            raise FileNotFoundError(path)

        print(f"{model_label}: using explicit file:\n  {path}")

        df = load_single_diet_file(path)
        return validate_and_finalize_diet_table(df, model_label), [path]

    candidates = candidate_diet_files(model_number)

    if not candidates:
        raise RuntimeError(
            f"No plausible {model_label} detailed CSV found under:\n"
            f"{DIET_ANALYSIS_DIR}\n"
            f"Use --diet-model{model_number} to provide the exact file."
        )

    # First try each candidate as a single all-score table.
    valid_single = []

    for c in candidates:
        path = c["path"]

        try:
            df = load_single_diet_file(path)
            finalized = validate_and_finalize_diet_table(df, model_label)
            valid_single.append((path, finalized))
        except Exception:
            pass

    if len(valid_single) == 1:
        path, finalized = valid_single[0]
        print(f"{model_label}: auto-detected all-score file:\n  {path}")
        return finalized, [path]

    if len(valid_single) > 1:
        lines = "\n".join(f"  - {p}" for p, _ in valid_single)
        raise RuntimeError(
            f"Multiple valid all-score {model_label} files found:\n"
            f"{lines}\n"
            f"Use --diet-model{model_number} to choose explicitly."
        )

    # Otherwise try combining score-specific files.
    standardized = []

    for c in candidates:
        path = c["path"]

        try:
            piece = load_single_diet_file(path)
        except Exception:
            continue

        if len(piece):
            standardized.append(piece)

    if not standardized:
        raise RuntimeError(
            f"Found candidate files for {model_label}, but none could be "
            "standardized."
        )

    combined = pd.concat(standardized, ignore_index=True, sort=False)

    # Keep only one plausible source per score. We require exactly 379 species.
    chosen = []
    used_paths = []

    for score in SCORES:
        score_rows = combined.loc[combined["score"].eq(score)].copy()

        source_counts = (
            score_rows.groupby("source_file")["species"]
            .nunique()
            .sort_values(ascending=False)
        )

        exact_sources = source_counts.loc[
            source_counts.eq(EXPECTED_SPECIES)
        ].index.tolist()

        if len(exact_sources) != 1:
            details = source_counts.head(20).to_dict()

            raise RuntimeError(
                f"{model_label} {score}: expected exactly one score-specific "
                f"file with {EXPECTED_SPECIES} species. Found candidates: "
                f"{details}\n"
                f"Use --diet-model{model_number} with a single all-score file "
                "if available."
            )

        src = exact_sources[0]
        chosen.append(
            score_rows.loc[score_rows["source_file"].eq(src)].copy()
        )
        used_paths.append(Path(src))

    finalized = validate_and_finalize_diet_table(
        pd.concat(chosen, ignore_index=True),
        model_label,
    )

    print(f"{model_label}: auto-combined score-specific files:")
    for path in used_paths:
        print("  ", path)

    return finalized, used_paths


def load_cgm_model(path, model_label):
    if not path.exists():
        raise FileNotFoundError(
            f"{model_label} file not found:\n{path}"
        )

    df = pd.read_csv(path, low_memory=False)

    required = [
        "outcome_name",
        "species",
        "beta_species",
        "FDR_within_outcome379",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(
            f"{model_label} missing required columns: {missing}"
        )

    out = df[required].copy()
    out["outcome_name"] = out["outcome_name"].astype(str)
    out["species"] = out["species"].astype(str)
    out["beta_cgm"] = pd.to_numeric(
        out["beta_species"],
        errors="coerce",
    )
    out["FDR_cgm"] = pd.to_numeric(
        out["FDR_within_outcome379"],
        errors="coerce",
    )

    out = out.drop(columns=[
        "beta_species",
        "FDR_within_outcome379",
    ])

    if out["beta_cgm"].isna().any() or out["FDR_cgm"].isna().any():
        raise RuntimeError(f"{model_label}: beta/FDR contains NaN.")

    for outcome in OUTCOMES:
        piece = out.loc[out["outcome_name"].eq(outcome)]

        if len(piece) != EXPECTED_SPECIES:
            raise RuntimeError(
                f"{model_label} {outcome}: expected {EXPECTED_SPECIES} rows, "
                f"got {len(piece)}."
            )

        if piece["species"].duplicated().any():
            raise RuntimeError(
                f"{model_label} {outcome}: duplicate species rows."
            )

    universes = {
        o: set(out.loc[out["outcome_name"].eq(o), "species"])
        for o in OUTCOMES
    }

    reference = universes[OUTCOMES[0]]

    for o in OUTCOMES[1:]:
        if universes[o] != reference:
            raise RuntimeError(
                f"{model_label}: species universe differs across outcomes."
            )

    out["cgm_significant_FDR05"] = out["FDR_cgm"] < 0.05
    return out


def path_direction_text(beta_diet, beta_cgm):
    diet_species = "species↑" if beta_diet > 0 else "species↓"
    species_cgm = "CGM↑" if beta_cgm > 0 else "CGM↓"
    return f"score↑ → {diet_species} → {species_cgm}"


def analyze_layer(diet, cgm, layer_name):
    """
    Return species-level overlap rows and 12-row pair summary.
    """

    # Strict universe agreement.
    diet_universe = set(
        diet.loc[diet["score"].eq(SCORES[0]), "species"]
    )
    cgm_universe = set(
        cgm.loc[cgm["outcome_name"].eq(OUTCOMES[0]), "species"]
    )

    if diet_universe != cgm_universe:
        only_diet = sorted(diet_universe - cgm_universe)[:10]
        only_cgm = sorted(cgm_universe - diet_universe)[:10]

        raise RuntimeError(
            f"{layer_name}: diet and CGM species universes differ.\n"
            f"Only diet examples: {only_diet}\n"
            f"Only CGM examples: {only_cgm}"
        )

    overlap_rows = []
    summary_rows = []

    for score in SCORES:
        d = diet.loc[diet["score"].eq(score)].copy()

        for outcome in OUTCOMES:
            g = cgm.loc[cgm["outcome_name"].eq(outcome)].copy()

            merged = d[
                [
                    "score",
                    "species",
                    "beta_diet",
                    "FDR_diet",
                    "diet_significant_FDR05",
                ]
            ].merge(
                g[
                    [
                        "outcome_name",
                        "species",
                        "beta_cgm",
                        "FDR_cgm",
                        "cgm_significant_FDR05",
                    ]
                ],
                on="species",
                how="inner",
                validate="one_to_one",
            )

            if len(merged) != EXPECTED_SPECIES:
                raise RuntimeError(
                    f"{layer_name} {score} x {outcome}: merged universe "
                    f"is {len(merged)}, expected {EXPECTED_SPECIES}."
                )

            diet_sig = merged["diet_significant_FDR05"]
            cgm_sig = merged["cgm_significant_FDR05"]
            both = diet_sig & cgm_sig

            candidates = merged.loc[both].copy()

            candidates["analysis_layer"] = layer_name
            candidates["beta_product"] = (
                candidates["beta_diet"] * candidates["beta_cgm"]
            )

            candidates["product_direction"] = np.where(
                candidates["beta_product"] < 0,
                "negative",
                np.where(
                    candidates["beta_product"] > 0,
                    "positive",
                    "zero",
                ),
            )

            expected_sign = EXPECTED_CGM_DIRECTION[score]

            candidates["expected_beta_product_sign"] = (
                "negative" if expected_sign < 0 else "positive"
            )

            candidates["direction_consistent_with_score_polarity"] = (
                np.sign(candidates["beta_product"]) == expected_sign
            )

            candidates["path_direction"] = [
                path_direction_text(a, b)
                for a, b in zip(
                    candidates["beta_diet"],
                    candidates["beta_cgm"],
                )
            ]

            candidates["abs_beta_product"] = (
                candidates["beta_product"].abs()
            )

            candidates = candidates.sort_values(
                [
                    "direction_consistent_with_score_polarity",
                    "abs_beta_product",
                ],
                ascending=[False, False],
            )

            overlap_rows.append(candidates)

            M = EXPECTED_SPECIES
            K = int(diet_sig.sum())
            n = int(cgm_sig.sum())
            k = int(both.sum())

            expected_overlap = K * n / M

            enrichment_p = float(
                hypergeom.sf(k - 1, M, K, n)
            )

            consistent_n = int(
                candidates[
                    "direction_consistent_with_score_polarity"
                ].sum()
            )

            positive_product_n = int(
                (candidates["beta_product"] > 0).sum()
            )

            negative_product_n = int(
                (candidates["beta_product"] < 0).sum()
            )

            summary_rows.append({
                "analysis_layer": layer_name,
                "diet_score": score,
                "cgm_outcome": outcome,
                "species_universe": M,
                "diet_significant_FDR05": K,
                "cgm_significant_FDR05": n,
                "overlap_species": k,
                "expected_overlap_under_random": expected_overlap,
                "overlap_fold_enrichment": (
                    k / expected_overlap
                    if expected_overlap > 0 else np.nan
                ),
                "overlap_hypergeom_p": enrichment_p,
                "negative_beta_product": negative_product_n,
                "positive_beta_product": positive_product_n,
                "direction_consistent_with_score_polarity": consistent_n,
                "direction_consistent_fraction": (
                    consistent_n / k if k else np.nan
                ),
            })

            print()
            print(
                f"{layer_name} | {score} x {outcome}"
            )
            print("-" * 90)
            print(
                f"Diet FDR<0.05: {K} | "
                f"CGM FDR<0.05: {n} | "
                f"overlap: {k}"
            )
            print(
                f"Expected random overlap: {expected_overlap:.2f} | "
                f"fold: "
                f"{(k / expected_overlap if expected_overlap else np.nan):.2f} | "
                f"hypergeom P={enrichment_p:.3e}"
            )
            print(
                f"Direction-consistent: {consistent_n}/{k}"
                if k
                else "Direction-consistent: 0/0"
            )

            if k:
                print("TOP OVERLAP CANDIDATES")
                print(
                    candidates[
                        [
                            "species",
                            "beta_diet",
                            "beta_cgm",
                            "beta_product",
                            "FDR_diet",
                            "FDR_cgm",
                            "direction_consistent_with_score_polarity",
                            "path_direction",
                        ]
                    ]
                    .head(10)
                    .to_string(index=False)
                )

    overlap = pd.concat(
        overlap_rows,
        ignore_index=True,
        sort=False,
    )

    summary = pd.DataFrame(summary_rows)

    # Exploratory correction across the 12 pair-level enrichment tests.
    summary["overlap_hypergeom_FDR12"] = bh_fdr(
        summary["overlap_hypergeom_p"].to_numpy()
    )

    return overlap, summary


def compare_primary_vs_bmi(primary_overlap, bmi_overlap):
    keys = ["diet_score", "cgm_outcome", "species"]

    p = primary_overlap.copy()
    s = bmi_overlap.copy()

    p = p.rename(columns={
        "beta_diet": "primary_beta_diet",
        "beta_cgm": "primary_beta_cgm",
        "beta_product": "primary_beta_product",
        "FDR_diet": "primary_FDR_diet",
        "FDR_cgm": "primary_FDR_cgm",
        "direction_consistent_with_score_polarity":
            "primary_direction_consistent",
        "path_direction": "primary_path_direction",
    })

    s = s.rename(columns={
        "beta_diet": "bmi_beta_diet",
        "beta_cgm": "bmi_beta_cgm",
        "beta_product": "bmi_beta_product",
        "FDR_diet": "bmi_FDR_diet",
        "FDR_cgm": "bmi_FDR_cgm",
        "direction_consistent_with_score_polarity":
            "bmi_direction_consistent",
        "path_direction": "bmi_path_direction",
    })

    keep_p = keys + [
        "primary_beta_diet",
        "primary_beta_cgm",
        "primary_beta_product",
        "primary_FDR_diet",
        "primary_FDR_cgm",
        "primary_direction_consistent",
        "primary_path_direction",
    ]

    keep_s = keys + [
        "bmi_beta_diet",
        "bmi_beta_cgm",
        "bmi_beta_product",
        "bmi_FDR_diet",
        "bmi_FDR_cgm",
        "bmi_direction_consistent",
        "bmi_path_direction",
    ]

    merged = p[keep_p].merge(
        s[keep_s],
        on=keys,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    merged["present_primary_model2"] = merged["_merge"].isin(
        ["left_only", "both"]
    )

    merged["present_bmi_model3"] = merged["_merge"].isin(
        ["right_only", "both"]
    )

    merged["robust_overlap_both_models"] = merged["_merge"].eq("both")

    merged["diet_beta_same_direction_M2_M3"] = np.where(
        merged["robust_overlap_both_models"],
        np.sign(merged["primary_beta_diet"])
        == np.sign(merged["bmi_beta_diet"]),
        np.nan,
    )

    merged["cgm_beta_same_direction_M2_M3"] = np.where(
        merged["robust_overlap_both_models"],
        np.sign(merged["primary_beta_cgm"])
        == np.sign(merged["bmi_beta_cgm"]),
        np.nan,
    )

    merged["path_product_same_direction_M2_M3"] = np.where(
        merged["robust_overlap_both_models"],
        np.sign(merged["primary_beta_product"])
        == np.sign(merged["bmi_beta_product"]),
        np.nan,
    )

    merged["robust_direction_consistent_candidate"] = (
        merged["robust_overlap_both_models"]
        & merged["primary_direction_consistent"].fillna(False)
        & merged["bmi_direction_consistent"].fillna(False)
        & merged["diet_beta_same_direction_M2_M3"].fillna(False)
        & merged["cgm_beta_same_direction_M2_M3"].fillna(False)
    )

    merged = merged.drop(columns="_merge")

    pair_summary = (
        merged.groupby(
            ["diet_score", "cgm_outcome"],
            as_index=False,
        )
        .agg(
            primary_overlap_species=(
                "present_primary_model2",
                "sum",
            ),
            bmi_overlap_species=(
                "present_bmi_model3",
                "sum",
            ),
            overlap_present_in_both_models=(
                "robust_overlap_both_models",
                "sum",
            ),
            robust_direction_consistent_candidates=(
                "robust_direction_consistent_candidate",
                "sum",
            ),
        )
    )

    return merged, pair_summary


def self_test():
    print("=" * 90)
    print("SELF TEST — DIET -> MICROBIOME -> CGM OVERLAP")
    print("=" * 90)

    species = [f"species_{i}" for i in range(10)]

    diet = []
    for score in SCORES:
        for i, sp in enumerate(species):
            # First three significant.
            beta = [0.5, -0.4, 0.3][i] if i < 3 else 0.01
            fdr = 0.01 if i < 3 else 0.5
            diet.append({
                "score": score,
                "species": sp,
                "beta_diet": beta,
                "FDR_diet": fdr,
                "p_diet": fdr,
                "diet_significant_FDR05": fdr < 0.05,
            })

    diet = pd.DataFrame(diet)

    cgm = []
    for outcome in OUTCOMES:
        for i, sp in enumerate(species):
            # species 0 and 1 significant.
            beta = -0.6 if i == 0 else (0.4 if i == 1 else 0.01)
            fdr = 0.01 if i < 2 else 0.5
            cgm.append({
                "outcome_name": outcome,
                "species": sp,
                "beta_cgm": beta,
                "FDR_cgm": fdr,
                "cgm_significant_FDR05": fdr < 0.05,
            })

    cgm = pd.DataFrame(cgm)

    global EXPECTED_SPECIES
    old_expected = EXPECTED_SPECIES
    EXPECTED_SPECIES = 10

    try:
        overlap, summary = analyze_layer(
            diet,
            cgm,
            "SELFTEST",
        )
    finally:
        EXPECTED_SPECIES = old_expected

    assert len(summary) == 12

    # Every pair overlaps species 0 and 1.
    assert (summary["overlap_species"] == 2).all()

    # For healthy scores:
    # species0: + diet beta * - cgm beta = negative -> expected
    # species1: - diet beta * + cgm beta = negative -> expected
    for score in ["AHEI", "AMED", "hPDI"]:
        piece = overlap.loc[overlap["score"].eq(score)]
        assert piece["direction_consistent_with_score_polarity"].all()

    # rEDIH expects positive product, so these same products are not expected.
    redih = overlap.loc[overlap["score"].eq("rEDIH")]
    assert (~redih["direction_consistent_with_score_polarity"]).all()

    robust, robust_summary = compare_primary_vs_bmi(
        overlap.rename(
            columns={
                "score": "diet_score",
                "outcome_name": "cgm_outcome",
            }
        ),
        overlap.rename(
            columns={
                "score": "diet_score",
                "outcome_name": "cgm_outcome",
            }
        ),
    )

    assert robust["robust_overlap_both_models"].all()
    assert len(robust_summary) == 12

    print()
    print("Overlap counting: PASS")
    print("Direction polarity logic: PASS")
    print("12-pair structure: PASS")
    print("Primary-vs-BMI robustness merge: PASS")
    print("SELF TEST: PASS")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--diet-model2",
        default=None,
        help="Optional exact all-species Diet->Microbiome Model 2 CSV.",
    )

    parser.add_argument(
        "--diet-model3",
        default=None,
        help="Optional exact all-species Diet->Microbiome Model 3 CSV.",
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("DIET -> MICROBIOME -> CGM OVERLAP / DIRECTION ANALYSIS")
    print("=" * 100)
    print(
        "PRIMARY: Diet Model2 ∩ Microbiome->CGM Model2"
    )
    print(
        "BMI SENSITIVITY: Diet Model3 ∩ Microbiome->CGM Model3"
    )
    print(
        "This is an overlap/direction screen, NOT a mediation model."
    )

    diet_m2, diet_m2_sources = load_diet_model(
        args.diet_model2,
        model_number=2,
    )

    diet_m3, diet_m3_sources = load_diet_model(
        args.diet_model3,
        model_number=3,
    )

    cgm_m2 = load_cgm_model(
        CGM_MODEL2,
        "Microbiome->CGM Model 2",
    )

    cgm_m3 = load_cgm_model(
        CGM_MODEL3,
        "Microbiome->CGM Model 3",
    )

    print()
    print("=" * 100)
    print("PRIMARY MODEL 2 LAYER")
    print("=" * 100)

    primary_overlap, primary_summary = analyze_layer(
        diet_m2,
        cgm_m2,
        "PRIMARY_MODEL2",
    )

    print()
    print("=" * 100)
    print("BMI-SENSITIVITY MODEL 3 LAYER")
    print("=" * 100)

    bmi_overlap, bmi_summary = analyze_layer(
        diet_m3,
        cgm_m3,
        "BMI_SENSITIVITY_MODEL3",
    )

    robust, robust_summary = compare_primary_vs_bmi(
        primary_overlap.rename(
            columns={
                "score": "diet_score",
                "outcome_name": "cgm_outcome",
            }
        ),
        bmi_overlap.rename(
            columns={
                "score": "diet_score",
                "outcome_name": "cgm_outcome",
            }
        ),
    )

    # Save standardized inputs for reproducibility.
    diet_m2.to_csv(
        OUT_DATA_DIR / "08_standardized_diet_microbiome_model2.csv",
        index=False,
    )

    diet_m3.to_csv(
        OUT_DATA_DIR / "08_standardized_diet_microbiome_model3.csv",
        index=False,
    )

    primary_overlap.to_csv(
        OUT_REPORT_DIR
        / "08_primary_model2_diet_microbiome_cgm_overlap_species.csv",
        index=False,
    )

    primary_summary.to_csv(
        OUT_REPORT_DIR
        / "08_primary_model2_diet_microbiome_cgm_overlap_summary.csv",
        index=False,
    )

    bmi_overlap.to_csv(
        OUT_REPORT_DIR
        / "08_bmi_model3_diet_microbiome_cgm_overlap_species.csv",
        index=False,
    )

    bmi_summary.to_csv(
        OUT_REPORT_DIR
        / "08_bmi_model3_diet_microbiome_cgm_overlap_summary.csv",
        index=False,
    )

    robust.to_csv(
        OUT_REPORT_DIR
        / "08_primary_vs_bmi_overlap_robustness_species.csv",
        index=False,
    )

    robust_summary.to_csv(
        OUT_REPORT_DIR
        / "08_primary_vs_bmi_overlap_robustness_summary.csv",
        index=False,
    )

    robust_candidates = robust.loc[
        robust["robust_direction_consistent_candidate"]
    ].copy()

    robust_candidates.to_csv(
        OUT_REPORT_DIR
        / "08_robust_direction_consistent_candidates.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print("PRIMARY MODEL 2 — 12 PATH SUMMARY")
    print("=" * 100)
    print(
        primary_summary[
            [
                "diet_score",
                "cgm_outcome",
                "diet_significant_FDR05",
                "cgm_significant_FDR05",
                "overlap_species",
                "expected_overlap_under_random",
                "overlap_fold_enrichment",
                "overlap_hypergeom_p",
                "overlap_hypergeom_FDR12",
                "direction_consistent_with_score_polarity",
                "direction_consistent_fraction",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 100)
    print("MODEL 2 vs MODEL 3 ROBUSTNESS")
    print("=" * 100)
    print(robust_summary.to_string(index=False))

    print()
    print(
        "Robust direction-consistent species-level candidates:",
        len(robust_candidates),
    )

    if len(robust_candidates):
        print()
        print("TOP ROBUST CANDIDATES")
        print("-" * 100)

        view = robust_candidates.copy()
        view["ranking_strength"] = (
            view["primary_beta_product"].abs()
            + view["bmi_beta_product"].abs()
        )

        print(
            view.sort_values(
                "ranking_strength",
                ascending=False,
            )
            .head(30)[
                [
                    "diet_score",
                    "cgm_outcome",
                    "species",
                    "primary_beta_diet",
                    "primary_beta_cgm",
                    "primary_beta_product",
                    "bmi_beta_diet",
                    "bmi_beta_cgm",
                    "bmi_beta_product",
                ]
            ]
            .to_string(index=False)
        )

    print()
    print("=" * 100)
    print("INTERPRETATION RULE")
    print("=" * 100)
    print(
        "AHEI / AMED / hPDI: beta_product < 0 is direction-consistent "
        "with a healthier score relating to a lower adverse CGM phenotype."
    )
    print(
        "rEDIH: beta_product > 0 is direction-consistent with a higher "
        "hyperinsulinemic score relating to a higher adverse CGM phenotype."
    )
    print(
        "These are candidate paths only; overlap and sign consistency do "
        "not establish mediation or causality."
    )

    print()
    print("=" * 100)
    print("SAVED")
    print("=" * 100)

    for p in [
        OUT_REPORT_DIR
        / "08_primary_model2_diet_microbiome_cgm_overlap_summary.csv",
        OUT_REPORT_DIR
        / "08_primary_model2_diet_microbiome_cgm_overlap_species.csv",
        OUT_REPORT_DIR
        / "08_bmi_model3_diet_microbiome_cgm_overlap_summary.csv",
        OUT_REPORT_DIR
        / "08_bmi_model3_diet_microbiome_cgm_overlap_species.csv",
        OUT_REPORT_DIR
        / "08_primary_vs_bmi_overlap_robustness_summary.csv",
        OUT_REPORT_DIR
        / "08_primary_vs_bmi_overlap_robustness_species.csv",
        OUT_REPORT_DIR
        / "08_robust_direction_consistent_candidates.csv",
    ]:
        print(p)

    print()
    print("Diet Model2 source(s):")
    for p in diet_m2_sources:
        print(" ", p)

    print("Diet Model3 source(s):")
    for p in diet_m3_sources:
        print(" ", p)


if __name__ == "__main__":
    main()
