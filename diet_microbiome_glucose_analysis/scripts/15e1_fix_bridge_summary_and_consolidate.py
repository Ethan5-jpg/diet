#!/usr/bin/env python3
"""Step 15e1 — repair bridge reporting split and consolidate new-diet candidates.

Why this exists
---------------
Step15e candidate logic is correct, but the outer Model2/Model3 merge kept
analysis_role metadata only from the Model2 side. Species appearing only in
Model3 therefore had analysis_role=NaN and were printed as separate summary
rows. This is a REPORTING/GROUPING bug, not a robust-candidate bug.

This script:
1) fills context metadata deterministically from diet_score;
2) rebuilds the Model2-vs-Model3 bridge summary so each diet_score×CGM outcome
   appears exactly once;
3) verifies the rebuilt counts against the original Model2 and Model3 overlap
   tables;
4) consolidates the 45 robust paths into exact-species recurrence summaries;
5) does NOT change candidate membership and does NOT run mediation.

No canonical file is overwritten.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
BASE = (
    ROOT / "diet_microbiome_glucose_analysis" / "outputs"
    / "new_diet_extension" / "bridge"
)

M2_OVERLAP = BASE / "15e_model2_overlap_species.csv"
M3_OVERLAP = BASE / "15e_model3_overlap_species.csv"
ROBUST_ALL = BASE / "15e_model2_vs_model3_bridge_robustness_species.csv"
ROBUST_CANDIDATES = BASE / "15e_robust_bridge_candidates.csv"

CORRECTED_ALL = BASE / "15e1_bridge_robustness_species_corrected_metadata.csv"
CORRECTED_SUMMARY = BASE / "15e1_bridge_robustness_summary_corrected.csv"
CANDIDATE_PATHS = BASE / "15e1_robust_candidate_paths.csv"
SPECIES_MASTER = BASE / "15e1_robust_candidate_species_master.csv"
CONTEXT_SUMMARY = BASE / "15e1_robust_candidate_context_summary.csv"
SUMMARY_TXT = BASE / "15e1_bridge_consolidation_summary.txt"

ROLE = {
    "EAT13": "primary_extension",
    "NOVA4": "primary_extension",
    "Carbohydrate_pct": "exploratory_extension",
}

RULE_TYPE = {
    "EAT13": "a_priori_health_polarity",
    "NOVA4": "a_priori_exposure_polarity",
    "Carbohydrate_pct": "observed_total_direction_concordance",
}


def require(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


def truthy(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).eq(1)
    return (
        s.astype(str).str.strip().str.lower()
        .isin({"true", "1", "1.0", "yes", "y", "t"})
    )


def count_by_context(df: pd.DataFrame, present_col: str) -> pd.DataFrame:
    return (
        df.loc[truthy(df[present_col])]
        .groupby(["diet_score", "cgm_outcome"], as_index=False)
        .size()
        .rename(columns={"size": "N"})
    )


def assert_context_counts(
    corrected: pd.DataFrame,
    source: pd.DataFrame,
    present_col: str,
    label: str,
) -> None:
    left = count_by_context(corrected, present_col).rename(
        columns={"N": "corrected_N"}
    )

    source_counts = (
        source.groupby(["diet_score", "cgm_outcome"], as_index=False)
        .size()
        .rename(columns={"size": "source_N"})
    )

    chk = source_counts.merge(
        left,
        on=["diet_score", "cgm_outcome"],
        how="outer",
    ).fillna(0)

    bad = chk.loc[chk["source_N"].ne(chk["corrected_N"])]
    if len(bad):
        raise RuntimeError(
            f"{label} context-count QC failed:\n{bad.to_string(index=False)}"
        )


def terminal_species(full_taxonomy: str) -> str:
    text = str(full_taxonomy)
    if "|s__" in text:
        return text.rsplit("|s__", 1)[-1]
    return text


def main() -> int:
    for p, label in [
        (M2_OVERLAP, "Step15e Model2 overlap"),
        (M3_OVERLAP, "Step15e Model3 overlap"),
        (ROBUST_ALL, "Step15e robustness species table"),
        (ROBUST_CANDIDATES, "Step15e robust candidates"),
    ]:
        require(p, label)

    m2 = pd.read_csv(M2_OVERLAP, low_memory=False)
    m3 = pd.read_csv(M3_OVERLAP, low_memory=False)
    all_df = pd.read_csv(ROBUST_ALL, low_memory=False)
    old_candidates = pd.read_csv(ROBUST_CANDIDATES, low_memory=False)

    required = {
        "diet_score",
        "cgm_outcome",
        "species",
        "present_model2",
        "present_model3",
        "overlap_both_models",
        "robust_direction_consistent_candidate",
    }
    missing = sorted(required - set(all_df.columns))
    if missing:
        raise RuntimeError(f"Robustness table missing columns: {missing}")

    unknown = sorted(set(all_df["diet_score"].dropna()) - set(ROLE))
    if unknown:
        raise RuntimeError(f"Unknown diet_score values: {unknown}")

    # Repair metadata that was absent on right_only Model3 rows.
    all_df["analysis_role"] = all_df["diet_score"].map(ROLE)
    if "direction_rule_type" in all_df.columns:
        all_df["direction_rule_type"] = all_df["diet_score"].map(RULE_TYPE)

    if all_df["analysis_role"].isna().any():
        raise RuntimeError("analysis_role still contains NaN after repair")

    all_df.to_csv(CORRECTED_ALL, index=False)

    # Exact QC: repaired outer merge must recover the source overlap counts.
    assert_context_counts(all_df, m2, "present_model2", "Model2")
    assert_context_counts(all_df, m3, "present_model3", "Model3")

    summary = (
        all_df.groupby(
            ["diet_score", "analysis_role", "cgm_outcome"],
            as_index=False,
        )
        .agg(
            model2_overlap_species=("present_model2", lambda s: int(truthy(s).sum())),
            model3_overlap_species=("present_model3", lambda s: int(truthy(s).sum())),
            overlap_present_both_models=(
                "overlap_both_models", lambda s: int(truthy(s).sum())
            ),
            robust_direction_consistent_candidates=(
                "robust_direction_consistent_candidate",
                lambda s: int(truthy(s).sum()),
            ),
        )
        .sort_values(["diet_score", "cgm_outcome"])
        .reset_index(drop=True)
    )
    summary.to_csv(CORRECTED_SUMMARY, index=False)

    # Candidate membership must remain exactly unchanged.
    candidates = all_df.loc[
        truthy(all_df["robust_direction_consistent_candidate"])
    ].copy()

    old_keys = set(
        map(
            tuple,
            old_candidates[
                ["diet_score", "cgm_outcome", "species"]
            ].astype(str).to_numpy(),
        )
    )
    new_keys = set(
        map(
            tuple,
            candidates[
                ["diet_score", "cgm_outcome", "species"]
            ].astype(str).to_numpy(),
        )
    )
    if old_keys != new_keys:
        raise RuntimeError(
            "Candidate-membership mismatch after metadata repair. "
            f"lost={len(old_keys-new_keys)}, gained={len(new_keys-old_keys)}"
        )

    candidates["species_label"] = candidates["species"].map(terminal_species)
    candidates.to_csv(CANDIDATE_PATHS, index=False)

    context = (
        candidates.groupby(
            ["diet_score", "analysis_role", "cgm_outcome"],
            as_index=False,
        )
        .agg(
            robust_paths=("species", "size"),
            unique_exact_species=("species", "nunique"),
        )
        .sort_values(["diet_score", "cgm_outcome"])
    )
    context.to_csv(CONTEXT_SUMMARY, index=False)

    # Exact full taxonomy string remains the identity key.
    species_master = (
        candidates.groupby("species", as_index=False)
        .agg(
            n_robust_paths=("species", "size"),
            n_diet_exposures=("diet_score", "nunique"),
            n_cgm_outcomes=("cgm_outcome", "nunique"),
            diet_exposures=(
                "diet_score",
                lambda x: ";".join(sorted(set(map(str, x)))),
            ),
            cgm_outcomes=(
                "cgm_outcome",
                lambda x: ";".join(sorted(set(map(str, x)))),
            ),
            contains_primary_extension=(
                "analysis_role",
                lambda x: bool((pd.Series(x) == "primary_extension").any()),
            ),
            contains_exploratory_extension=(
                "analysis_role",
                lambda x: bool((pd.Series(x) == "exploratory_extension").any()),
            ),
        )
    )
    species_master["species_label"] = species_master["species"].map(terminal_species)
    species_master["recurrence_class"] = np.where(
        species_master["n_robust_paths"].ge(2),
        "recurrent_multiple_robust_paths",
        "single_robust_path",
    )
    species_master = species_master.sort_values(
        [
            "n_robust_paths",
            "n_diet_exposures",
            "n_cgm_outcomes",
            "species",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    species_master.insert(0, "candidate_rank", np.arange(1, len(species_master) + 1))
    species_master.to_csv(SPECIES_MASTER, index=False)

    # Cross-exposure overlap of exact species, useful before mediation.
    nova = set(
        candidates.loc[candidates["diet_score"].eq("NOVA4"), "species"].astype(str)
    )
    carb = set(
        candidates.loc[
            candidates["diet_score"].eq("Carbohydrate_pct"), "species"
        ].astype(str)
    )

    recurrent_n = int(species_master["n_robust_paths"].ge(2).sum())
    single_n = int(species_master["n_robust_paths"].eq(1).sum())

    lines = [
        "=== STEP 15e1 BRIDGE REPORT REPAIR + CONSOLIDATION ===",
        "CANDIDATE_MEMBERSHIP_CHANGED=False",
        "REPORTING_NAN_SPLIT_FIXED=True",
        "MODEL2_CONTEXT_COUNT_QC=True",
        "MODEL3_CONTEXT_COUNT_QC=True",
        f"ROBUST_PATHS={len(candidates)}",
        f"UNIQUE_EXACT_SPECIES={candidates['species'].nunique()}",
        f"RECURRENT_SPECIES_N_PATHS_GE_2={recurrent_n}",
        f"SINGLE_PATH_SPECIES={single_n}",
        f"NOVA4_ROBUST_PATHS={int(candidates['diet_score'].eq('NOVA4').sum())}",
        f"CARBOHYDRATE_ROBUST_PATHS={int(candidates['diet_score'].eq('Carbohydrate_pct').sum())}",
        f"NOVA4_UNIQUE_SPECIES={len(nova)}",
        f"CARBOHYDRATE_UNIQUE_SPECIES={len(carb)}",
        f"SHARED_NOVA4_CARBOHYDRATE_EXACT_SPECIES={len(nova & carb)}",
        "",
        "--- CORRECTED ROBUST BRIDGE SUMMARY ---",
        summary.to_string(index=False),
        "",
        "--- ROBUST PATHS BY CONTEXT ---",
        context.to_string(index=False),
        "",
        "IMPORTANT:",
        "- The NaN rows in the original Step15e robust summary were caused by",
        "  Model3-only overlap rows lacking Model2-side analysis_role metadata.",
        "- They did NOT change the 45 robust-candidate paths.",
        "- Full taxonomy string is the exact species identity.",
        "- Recurrence classes here are transparent descriptive labels, not a",
        "  replacement for the canonical old Step09 tier nomenclature.",
        "",
        f"CORRECTED_SUMMARY={CORRECTED_SUMMARY}",
        f"CANDIDATE_PATHS={CANDIDATE_PATHS}",
        f"SPECIES_MASTER={SPECIES_MASTER}",
    ]
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
