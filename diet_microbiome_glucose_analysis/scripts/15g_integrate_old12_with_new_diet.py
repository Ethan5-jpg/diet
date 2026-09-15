#!/usr/bin/env python3
"""Step 15g — integrate original 12-core with final robust new-diet paths.

Purpose
-------
After Step15f3, define the high-robustness new-diet subset as PRIMARY
direction-consistent mediation-style paths that remain direction-consistent in
BOTH:
1) strict diabetes/A10 sensitivity, and
2) protocol-window antibiotic/PPI sensitivity.

This script does NOT redefine the primary result set. It creates an integration
view for biological interpretation.

It compares:
- original AMED/hPDI recurrent 12-core species;
- new high-robustness NOVA4 paths;
- new high-robustness Carbohydrate_pct paths.

For NOVA4 x glucose_cv shared species, it additionally performs a
polarity-aware comparison with the old healthy-diet direction class:
- healthy-diet enriched / lower-CV taxon -> expected NOVA4 diet->species beta < 0
- healthy-diet depleted / higher-CV taxon -> expected NOVA4 diet->species beta > 0

Carbohydrate_pct receives overlap-only interpretation because it has no
predefined healthy/unhealthy polarity.

No model is refit.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
DG = ROOT / "diet_microbiome_glucose_analysis"

NEW_MED = DG / "outputs" / "new_diet_extension" / "mediation"
NEW_BRIDGE = DG / "outputs" / "new_diet_extension" / "bridge"

THREEWAY = NEW_MED / "15f3_threeway_path_robustness.csv"
BRIDGE_PATHS = NEW_BRIDGE / "15e1_robust_candidate_paths.csv"

OUT_DIR = DG / "outputs" / "new_diet_extension" / "integration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATH_OUT = OUT_DIR / "15g_high_robust_new_diet_paths.csv"
SPECIES_OUT = OUT_DIR / "15g_high_robust_new_diet_species.csv"
OVERLAP_OUT = OUT_DIR / "15g_old12_vs_new_species_overlap.csv"
NOVA_POLARITY_OUT = OUT_DIR / "15g_nova4_glucose_cv_old12_polarity_check.csv"
SUMMARY_OUT = OUT_DIR / "15g_old12_new_diet_integration_summary.txt"


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


def terminal_species(x: str) -> str:
    x = str(x)
    return x.rsplit("|s__", 1)[-1] if "|s__" in x else x


def discover_old_core() -> Path:
    candidates = list(
        (DG / "outputs").rglob("13b_12_core_direction_classes.csv")
    )
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        # Fallback to 13a annotation table if 13b exact file was moved.
        candidates = list(
            (DG / "outputs").rglob("13a_12_core_species_biological_annotation.csv")
        )
    if len(candidates) != 1:
        raise RuntimeError(
            "Could not uniquely locate old 12-core annotation table. "
            f"Candidates={candidates}"
        )
    return candidates[0]


def detect_species_col(df: pd.DataFrame) -> str:
    exact = [
        "species",
        "full_taxonomy",
        "species_full_taxonomy",
        "taxonomy",
        "species_taxonomy",
    ]
    for c in exact:
        if c in df.columns:
            return c

    # Last-resort content-based detection: full taxonomy strings contain k__ and |s__.
    for c in df.columns:
        vals = df[c].dropna().astype(str).head(12)
        if len(vals) and vals.str.contains(r"k__.*\|s__", regex=True).mean() > .5:
            return c
    raise RuntimeError(
        f"Could not detect exact species/taxonomy column. Columns={df.columns.tolist()}"
    )


def detect_direction_class_col(df: pd.DataFrame) -> str | None:
    for c in [
        "direction_class",
        "component_direction_class",
        "direction_architecture_class",
    ]:
        if c in df.columns:
            return c
    return None


def main() -> int:
    require(THREEWAY, "Step15f3 three-way robustness")
    require(BRIDGE_PATHS, "Step15e1 robust bridge paths")

    old_core_path = discover_old_core()
    require(old_core_path, "old 12-core table")

    tw = pd.read_csv(THREEWAY, low_memory=False)
    bridge = pd.read_csv(BRIDGE_PATHS, low_memory=False)
    old = pd.read_csv(old_core_path, low_memory=False)

    required_tw = {
        "diet_score",
        "cgm_outcome",
        "species",
        "primary_consistent",
        "strict_consistent",
        "protocol_consistent",
        "primary_consistent_retained_both",
    }
    missing = sorted(required_tw - set(tw.columns))
    if missing:
        raise RuntimeError(f"Three-way table missing columns: {missing}")

    high = tw.loc[
        truthy(tw["primary_consistent_retained_both"])
    ].copy()

    if len(high) != 22:
        raise RuntimeError(
            f"Expected 22 high-robustness paths from Step15f3; found {len(high)}"
        )

    # Bring forward bridge coefficients for direction-aware integration.
    bcols = [
        "diet_score",
        "cgm_outcome",
        "species",
        "model2_beta_diet",
        "model2_beta_cgm",
        "model2_beta_product",
        "model3_beta_diet",
        "model3_beta_cgm",
        "model3_beta_product",
    ]
    bcols = [c for c in bcols if c in bridge.columns]
    high = high.merge(
        bridge[bcols],
        on=["diet_score", "cgm_outcome", "species"],
        how="left",
        validate="one_to_one",
    )
    high["species_label"] = high["species"].map(terminal_species)
    high.to_csv(PATH_OUT, index=False)

    species_new = (
        high.groupby("species", as_index=False)
        .agg(
            species_label=("species_label", "first"),
            high_robust_paths=("species", "size"),
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
            nova4_paths=("diet_score", lambda x: int((pd.Series(x) == "NOVA4").sum())),
            carbohydrate_paths=(
                "diet_score",
                lambda x: int((pd.Series(x) == "Carbohydrate_pct").sum()),
            ),
        )
        .sort_values(
            ["high_robust_paths", "n_diet_exposures", "n_cgm_outcomes", "species"],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
    )
    species_new.to_csv(SPECIES_OUT, index=False)

    old_species_col = detect_species_col(old)
    old = old.copy()
    old["species"] = old[old_species_col].astype(str)

    if old["species"].nunique() != 12:
        raise RuntimeError(
            f"Old core table should have 12 exact species; found "
            f"{old['species'].nunique()} using column {old_species_col}"
        )

    direction_col = detect_direction_class_col(old)

    old_keep = ["species"]
    if direction_col is not None:
        old_keep.append(direction_col)
    for c in ["literature_interpretation", "literature_status", "discussion_priority"]:
        if c in old.columns:
            old_keep.append(c)

    old12 = old[old_keep].drop_duplicates("species").copy()
    old12["species_label"] = old12["species"].map(terminal_species)

    old_set = set(old12["species"])
    new_set = set(species_new["species"])
    nova_set = set(
        high.loc[high["diet_score"].eq("NOVA4"), "species"]
    )
    carb_set = set(
        high.loc[high["diet_score"].eq("Carbohydrate_pct"), "species"]
    )

    overlap = old12.merge(
        species_new,
        on=["species", "species_label"],
        how="left",
        validate="one_to_one",
    )
    overlap["present_in_new_high_robust"] = overlap["species"].isin(new_set)
    overlap["present_in_nova4_high_robust"] = overlap["species"].isin(nova_set)
    overlap["present_in_carbohydrate_high_robust"] = overlap["species"].isin(carb_set)
    overlap.to_csv(OVERLAP_OUT, index=False)

    # Polarity-aware old healthy-diet vs NOVA4 comparison only for glucose_cv.
    nova_gc = high.loc[
        high["diet_score"].eq("NOVA4")
        & high["cgm_outcome"].eq("glucose_cv")
        & high["species"].isin(old_set)
    ].copy()

    if len(nova_gc):
        nova_gc = nova_gc.merge(
            old12,
            on="species",
            how="left",
            validate="one_to_one",
            suffixes=("", "_old"),
        )

        if direction_col is not None and "model2_beta_diet" in nova_gc.columns:
            cls = nova_gc[direction_col].astype(str)

            expected_nova_sign = np.select(
                [
                    cls.str.contains(
                        "healthy_diet_enriched_relative__lower_glucose_cv",
                        regex=False,
                    ),
                    cls.str.contains(
                        "healthy_diet_depleted_relative__higher_glucose_cv_taxon",
                        regex=False,
                    ),
                ],
                [-1, +1],
                default=np.nan,
            )
            nova_gc["expected_NOVA4_diet_species_sign_from_old_class"] = expected_nova_sign
            nova_gc["observed_NOVA4_model2_diet_species_sign"] = np.sign(
                pd.to_numeric(nova_gc["model2_beta_diet"], errors="coerce")
            )
            nova_gc["healthy_vs_NOVA4_polarity_concordant"] = (
                nova_gc[
                    "observed_NOVA4_model2_diet_species_sign"
                ]
                == nova_gc[
                    "expected_NOVA4_diet_species_sign_from_old_class"
                ]
            )
        else:
            nova_gc["healthy_vs_NOVA4_polarity_concordant"] = np.nan

    nova_gc.to_csv(NOVA_POLARITY_OUT, index=False)

    old_new_shared = old_set & new_set
    old_nova_shared = old_set & nova_set
    old_carb_shared = old_set & carb_set

    by_context = (
        high.groupby(["diet_score", "cgm_outcome"], as_index=False)
        .agg(
            high_robust_paths=("species", "size"),
            unique_exact_species=("species", "nunique"),
        )
    )

    polarity_n = (
        int(nova_gc["healthy_vs_NOVA4_polarity_concordant"].eq(True).sum())
        if len(nova_gc) and "healthy_vs_NOVA4_polarity_concordant" in nova_gc.columns
        else 0
    )
    polarity_den = (
        int(nova_gc["healthy_vs_NOVA4_polarity_concordant"].notna().sum())
        if len(nova_gc) and "healthy_vs_NOVA4_polarity_concordant" in nova_gc.columns
        else 0
    )

    lines = [
        "=== STEP 15g OLD 12-CORE vs NEW-DIET HIGH-ROBUSTNESS INTEGRATION ===",
        f"OLD_CORE_SOURCE={old_core_path}",
        "PRIMARY_RESULT_SET_REDEFINED=False",
        "NEW_HIGH_ROBUSTNESS_DEFINITION=primary_consistent retained in BOTH strict and protocol sensitivities",
        f"NEW_HIGH_ROBUST_PATHS={len(high)}",
        f"NEW_HIGH_ROBUST_UNIQUE_EXACT_SPECIES={high['species'].nunique()}",
        f"NOVA4_HIGH_ROBUST_PATHS={int(high['diet_score'].eq('NOVA4').sum())}",
        f"CARBOHYDRATE_HIGH_ROBUST_PATHS={int(high['diet_score'].eq('Carbohydrate_pct').sum())}",
        f"OLD12_SHARED_WITH_ANY_NEW_HIGH_ROBUST_SPECIES={len(old_new_shared)}/12",
        f"OLD12_SHARED_WITH_NOVA4_HIGH_ROBUST_SPECIES={len(old_nova_shared)}/12",
        f"OLD12_SHARED_WITH_CARBOHYDRATE_HIGH_ROBUST_SPECIES={len(old_carb_shared)}/12",
        f"NOVA4_CARBOHYDRATE_SHARED_HIGH_ROBUST_SPECIES={len(nova_set & carb_set)}",
        f"OLD12_SHARED_NOVA4_GLUCOSE_CV_SPECIES={len(nova_gc)}",
        f"OLD_HEALTHY_VS_NOVA4_POLARITY_CONCORDANT={polarity_n}/{polarity_den}",
        "",
        "--- NEW HIGH-ROBUST PATHS BY CONTEXT ---",
        by_context.to_string(index=False),
        "",
        "--- OLD 12-CORE SHARED WITH NEW HIGH-ROBUST SPECIES ---",
        (
            overlap.loc[
                overlap["present_in_new_high_robust"],
                [
                    "species_label",
                    "present_in_nova4_high_robust",
                    "present_in_carbohydrate_high_robust",
                ]
                + ([direction_col] if direction_col is not None else []),
            ].to_string(index=False)
            if len(old_new_shared) else "NONE"
        ),
        "",
        "--- NOVA4 glucose_cv SHARED OLD-CORE POLARITY CHECK ---",
        (
            nova_gc[
                [
                    "species_label",
                    "model2_beta_diet",
                    "model2_beta_cgm",
                ]
                + ([direction_col] if direction_col is not None else [])
                + ["healthy_vs_NOVA4_polarity_concordant"]
            ].to_string(index=False)
            if len(nova_gc) else "NONE"
        ),
        "",
        "INTERPRETATION RULES:",
        "- Exact full taxonomy string is the species identity.",
        "- The 22-path high-robustness subset supplements, but does not replace,",
        "  the 30-path primary mediation-style result set.",
        "- NOVA4 can be polarity-compared with old healthy-diet scores because",
        "  higher NOVA4 is adverse-facing; Carbohydrate_pct cannot.",
        "- Shared species do not establish a shared causal mechanism.",
        "",
        f"PATH_TABLE={PATH_OUT}",
        f"SPECIES_TABLE={SPECIES_OUT}",
        f"OLD12_OVERLAP_TABLE={OVERLAP_OUT}",
        f"NOVA4_POLARITY_TABLE={NOVA_POLARITY_OUT}",
    ]

    SUMMARY_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
