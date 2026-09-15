#!/usr/bin/env python3
"""Step 15g1 — repair/verify old-core vs NOVA4 polarity check.

Why this exists
---------------
Step15g correctly found 8/12 old recurrent core species among the high-robust
NOVA4 glucose-CV paths, but the polarity summary printed 0/0 because the script
did not recognize the exact column name that stores the old Step13b direction
class. The overlap itself is correct.

This script does NOT refit any model and does NOT change species membership.
It detects the Step13b direction-class column by its VALUES rather than by a
hard-coded column name, then verifies the expected mirror pattern:

old healthy-diet class:
  healthy_diet_depleted_relative__higher_glucose_cv_taxon
    -> expected NOVA4 diet->species beta > 0

  healthy_diet_enriched_relative__lower_glucose_cv
    -> expected NOVA4 diet->species beta < 0

Outputs
-------
diet_microbiome_glucose_analysis/outputs/new_diet_extension/integration/
    15g1_nova4_glucose_cv_old12_polarity_check_FIXED.csv
    15g1_nova4_glucose_cv_old12_polarity_summary.txt
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
DG = ROOT / "diet_microbiome_glucose_analysis"

OLD = DG / "outputs" / "reports" / "13b_12_core_direction_classes.csv"
NEW = (
    DG / "outputs" / "new_diet_extension" / "integration"
    / "15g_high_robust_new_diet_paths.csv"
)

OUT_DIR = (
    DG / "outputs" / "new_diet_extension" / "integration"
)
OUT_CSV = OUT_DIR / "15g1_nova4_glucose_cv_old12_polarity_check_FIXED.csv"
OUT_TXT = OUT_DIR / "15g1_nova4_glucose_cv_old12_polarity_summary.txt"

CLASS_ENRICHED = "healthy_diet_enriched_relative__lower_glucose_cv"
CLASS_DEPLETED = "healthy_diet_depleted_relative__higher_glucose_cv_taxon"


def require(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


def terminal_species(x: str) -> str:
    x = str(x)
    return x.rsplit("|s__", 1)[-1] if "|s__" in x else x


def detect_species_col(df: pd.DataFrame) -> str:
    for c in [
        "species",
        "full_taxonomy",
        "species_full_taxonomy",
        "taxonomy",
        "species_taxonomy",
    ]:
        if c in df.columns:
            vals = df[c].dropna().astype(str)
            if vals.nunique() == 12:
                return c

    for c in df.columns:
        vals = df[c].dropna().astype(str)
        if len(vals) and vals.str.contains(r"k__.*\|s__", regex=True).mean() > .5:
            return c

    raise RuntimeError(
        "Could not detect exact species/taxonomy column in old Step13b table. "
        f"Columns={df.columns.tolist()}"
    )


def detect_direction_class_col_by_values(df: pd.DataFrame) -> str:
    hits = []
    for c in df.columns:
        vals = set(df[c].dropna().astype(str).unique().tolist())
        n_match = int(CLASS_ENRICHED in vals) + int(CLASS_DEPLETED in vals)
        if n_match:
            hits.append((n_match, c, vals))

    if not hits:
        # Allow substring match in case a prefix/suffix was added.
        for c in df.columns:
            vals = df[c].dropna().astype(str)
            if (
                vals.str.contains(CLASS_ENRICHED, regex=False).any()
                or vals.str.contains(CLASS_DEPLETED, regex=False).any()
            ):
                hits.append((1, c, set(vals.unique().tolist())))

    if not hits:
        raise RuntimeError(
            "Could not find Step13b direction-class column by values. "
            f"Columns={df.columns.tolist()}"
        )

    hits.sort(reverse=True, key=lambda x: (x[0], x[1]))
    best = [h for h in hits if h[0] == hits[0][0]]
    if len(best) > 1:
        print("WARNING: multiple direction-class candidates found:")
        for _, c, _ in best:
            print(" -", c)

    return best[0][1]


def main() -> int:
    require(OLD, "old Step13b direction classes")
    require(NEW, "Step15g high-robust new-diet paths")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    old = pd.read_csv(OLD, low_memory=False)
    new = pd.read_csv(NEW, low_memory=False)

    species_col = detect_species_col(old)
    class_col = detect_direction_class_col_by_values(old)

    old2 = old[[species_col, class_col]].copy().rename(
        columns={
            species_col: "species",
            class_col: "old_direction_class",
        }
    )
    old2["species"] = old2["species"].astype(str)
    old2 = old2.drop_duplicates("species")

    if old2["species"].nunique() != 12:
        raise RuntimeError(
            f"Expected 12 old core exact species, found {old2['species'].nunique()}"
        )

    required_new = {
        "diet_score",
        "cgm_outcome",
        "species",
        "model2_beta_diet",
        "model2_beta_cgm",
    }
    missing = sorted(required_new - set(new.columns))
    if missing:
        raise RuntimeError(f"Step15g path table missing columns: {missing}")

    x = new.loc[
        new["diet_score"].eq("NOVA4")
        & new["cgm_outcome"].eq("glucose_cv")
    ].copy()

    x = x.merge(
        old2,
        on="species",
        how="inner",
        validate="one_to_one",
    )

    if len(x) != 8:
        raise RuntimeError(
            f"Expected 8 old-core species in high-robust NOVA4 glucose_cv; found {len(x)}"
        )

    x["species_label"] = x["species"].map(terminal_species)

    def expected_sign(cls: str) -> float:
        cls = str(cls)
        if CLASS_DEPLETED in cls:
            return 1.0
        if CLASS_ENRICHED in cls:
            return -1.0
        return np.nan

    x["expected_NOVA4_diet_species_sign"] = x["old_direction_class"].map(
        expected_sign
    )
    x["observed_NOVA4_model2_diet_species_sign"] = np.sign(
        pd.to_numeric(x["model2_beta_diet"], errors="coerce")
    )
    x["observed_species_CGM_sign"] = np.sign(
        pd.to_numeric(x["model2_beta_cgm"], errors="coerce")
    )

    x["healthy_vs_NOVA4_polarity_concordant"] = (
        x["observed_NOVA4_model2_diet_species_sign"]
        == x["expected_NOVA4_diet_species_sign"]
    )

    # Extra internal consistency check with old direction class:
    # enriched/lower-CV -> species->CV expected negative;
    # depleted/higher-CV -> species->CV expected positive.
    x["expected_species_CGM_sign_from_old_class"] = np.where(
        x["old_direction_class"].astype(str).str.contains(
            CLASS_DEPLETED, regex=False
        ),
        1.0,
        np.where(
            x["old_direction_class"].astype(str).str.contains(
                CLASS_ENRICHED, regex=False
            ),
            -1.0,
            np.nan,
        ),
    )
    x["old_vs_new_species_CGM_direction_concordant"] = (
        x["observed_species_CGM_sign"]
        == x["expected_species_CGM_sign_from_old_class"]
    )

    x = x.sort_values("species_label").reset_index(drop=True)
    x.to_csv(OUT_CSV, index=False)

    polarity_n = int(
        x["healthy_vs_NOVA4_polarity_concordant"].eq(True).sum()
    )
    cgm_n = int(
        x["old_vs_new_species_CGM_direction_concordant"].eq(True).sum()
    )

    print("=== STEP 15g1 NOVA4 POLARITY CHECK FIX ===")
    print(f"OLD_SPECIES_COLUMN={species_col}")
    print(f"OLD_DIRECTION_CLASS_COLUMN={class_col}")
    print(f"SHARED_OLD12_NOVA4_GLUCOSE_CV_SPECIES={len(x)}")
    print(f"HEALTHY_VS_NOVA4_POLARITY_CONCORDANT={polarity_n}/{len(x)}")
    print(f"OLD_VS_NEW_SPECIES_CGM_DIRECTION_CONCORDANT={cgm_n}/{len(x)}")

    print("\n--- SPECIES-LEVEL CHECK ---")
    print(
        x[
            [
                "species_label",
                "old_direction_class",
                "model2_beta_diet",
                "expected_NOVA4_diet_species_sign",
                "model2_beta_cgm",
                "healthy_vs_NOVA4_polarity_concordant",
                "old_vs_new_species_CGM_direction_concordant",
            ]
        ].to_string(index=False)
    )

    lines = [
        "=== STEP 15g1 NOVA4 POLARITY CHECK FIX ===",
        f"OLD_SPECIES_COLUMN={species_col}",
        f"OLD_DIRECTION_CLASS_COLUMN={class_col}",
        f"SHARED_OLD12_NOVA4_GLUCOSE_CV_SPECIES={len(x)}",
        f"HEALTHY_VS_NOVA4_POLARITY_CONCORDANT={polarity_n}/{len(x)}",
        f"OLD_VS_NEW_SPECIES_CGM_DIRECTION_CONCORDANT={cgm_n}/{len(x)}",
        "",
        "INTERPRETATION:",
        "- This fixes a metadata-column detection failure in Step15g only.",
        "- Species overlap and model coefficients were not changed.",
        "- Concordance here means the adverse-facing NOVA4 exposure shows the",
        "  mirror diet->species direction expected from the old healthy-diet class.",
        "- It supports a shared compositional association axis; it does not prove",
        "  a shared causal mechanism.",
        "",
        x[
            [
                "species_label",
                "old_direction_class",
                "model2_beta_diet",
                "expected_NOVA4_diet_species_sign",
                "model2_beta_cgm",
                "healthy_vs_NOVA4_polarity_concordant",
                "old_vs_new_species_CGM_direction_concordant",
            ]
        ].to_string(index=False),
        "",
        f"OUTPUT={OUT_CSV}",
    ]
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
