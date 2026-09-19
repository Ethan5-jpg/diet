#!/usr/bin/env python3
"""
Step 19a — Final biological architecture synthesis

Read-only synthesis of the frozen Step18f/18g highest-robustness path set.

Purpose
-------
Summarize the 871 highest-robustness Diet -> Microbiome -> CGM paths into:
1) recurrent taxa across phenotype domains,
2) domain breadth (1 / 2 / 3 domains),
3) diet breadth,
4) CGM-outcome breadth,
5) exposure x phenotype-domain architecture.

No models are refit.
No bootstrap is rerun.
No FDR is recomputed.
No new discoveries are created.

Interpretation
--------------
- "Pan-domain" means a taxon appears in all 3 descriptive phenotype domains.
- "Cross-domain-2" means it appears in exactly 2 domains.
- "Domain-specific" means it appears in exactly 1 domain.
- These are descriptive recurrence labels, not new statistical tests.
- Mean glucose and GMI remain highly dependent phenotypes and are not treated
  as independent replication.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
BASE = ROOT / "diet_microbiome_glucose_analysis" / "4New_cgm" / "outputs"

IN_DIR = BASE / "paper20_mediation_phenotype_domains" / "reports"
IN_FILE = IN_DIR / "18g_threeway_paths_with_domains.csv"

OUT_DIR = BASE / "paper20_final_biological_architecture" / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_HIGHEST_ROBUST = 871

DOMAIN_ORDER = [
    "glycemic_level",
    "glycemic_variability",
    "glycemic_risk_range",
]

DOMAIN_SHORT = {
    "glycemic_level": "level",
    "glycemic_variability": "variability",
    "glycemic_risk_range": "risk_range",
}

DIET_ORDER = [
    "AHEI",
    "AMED",
    "Carbohydrate_pct",
    "EAT13",
    "NOVA4",
    "hPDI",
    "rEDIH",
]


def bool_series(s):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).eq(1)
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "1.0", "yes", "y", "t"})
    )


def semi_unique(series):
    return ";".join(sorted(set(map(str, series.dropna()))))


def breadth_class(n):
    if n == 3:
        return "pan_domain_3"
    if n == 2:
        return "cross_domain_2"
    if n == 1:
        return "domain_specific_1"
    return "none"


def main():
    print("=== STEP 19a FINAL BIOLOGICAL ARCHITECTURE SYNTHESIS ===")
    print("MODELS_FIT=False")
    print("BOOTSTRAP_RERUN=False")
    print("FDR_RECALCULATED=False")
    print("PRIMARY_DISCOVERY_REDEFINED=False")

    if not IN_FILE.is_file():
        raise FileNotFoundError(IN_FILE)

    df = pd.read_csv(IN_FILE, low_memory=False)

    required = {
        "exposure",
        "species",
        "outcome_field",
        "phenotype_domain",
        "highest_robustness",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    df["highest_robustness"] = bool_series(df["highest_robustness"])
    h = df.loc[df["highest_robustness"]].copy()

    n_high = len(h)
    print(f"HIGHEST_ROBUSTNESS_PATHS={n_high}")
    print(f"EXPECTED_871_MATCH={n_high == EXPECTED_HIGHEST_ROBUST}")

    if n_high != EXPECTED_HIGHEST_ROBUST:
        raise RuntimeError(
            f"Frozen highest-robustness path count changed: expected "
            f"{EXPECTED_HIGHEST_ROBUST}, observed {n_high}"
        )

    unknown_domains = sorted(
        set(h["phenotype_domain"].dropna().astype(str)) - set(DOMAIN_ORDER)
    )
    missing_domains = sorted(
        set(DOMAIN_ORDER) - set(h["phenotype_domain"].dropna().astype(str))
    )
    print(f"UNKNOWN_DOMAINS={unknown_domains}")
    print(f"MISSING_EXPECTED_DOMAINS={missing_domains}")

    if unknown_domains or missing_domains:
        raise RuntimeError("Phenotype-domain mapping is not the frozen 3-domain scheme.")

    # ------------------------------------------------------------------
    # 1. Species-level architecture
    # ------------------------------------------------------------------
    species = (
        h.groupby("species", dropna=False)
        .agg(
            highest_robust_paths=("species", "size"),
            diet_exposures=("exposure", "nunique"),
            cgm_outcomes=("outcome_field", "nunique"),
            phenotype_domains=("phenotype_domain", "nunique"),
            exposure_list=("exposure", semi_unique),
            outcome_list=("outcome_field", semi_unique),
            domain_list=("phenotype_domain", semi_unique),
        )
        .reset_index()
    )

    species["breadth_class"] = species["phenotype_domains"].map(breadth_class)
    species["pan_domain"] = species["phenotype_domains"].eq(3)

    # Domain-specific path counts per species.
    dom_counts = (
        h.groupby(["species", "phenotype_domain"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=DOMAIN_ORDER, fill_value=0)
        .rename(columns=lambda c: f"paths_{DOMAIN_SHORT[c]}")
        .reset_index()
    )
    species = species.merge(dom_counts, on="species", how="left")

    # Domain-specific unique outcome counts per species.
    dom_outcomes = (
        h.groupby(["species", "phenotype_domain"])["outcome_field"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(columns=DOMAIN_ORDER, fill_value=0)
        .rename(columns=lambda c: f"outcomes_{DOMAIN_SHORT[c]}")
        .reset_index()
    )
    species = species.merge(dom_outcomes, on="species", how="left")

    # Domain-specific unique diet counts per species.
    dom_diets = (
        h.groupby(["species", "phenotype_domain"])["exposure"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(columns=DOMAIN_ORDER, fill_value=0)
        .rename(columns=lambda c: f"diets_{DOMAIN_SHORT[c]}")
        .reset_index()
    )
    species = species.merge(dom_diets, on="species", how="left")

    species = species.sort_values(
        [
            "phenotype_domains",
            "diet_exposures",
            "cgm_outcomes",
            "highest_robust_paths",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. Pan-domain core taxa
    # ------------------------------------------------------------------
    pan = species.loc[species["pan_domain"]].copy()
    pan = pan.sort_values(
        ["diet_exposures", "cgm_outcomes", "highest_robust_paths"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    pan.insert(0, "pan_domain_rank", np.arange(1, len(pan) + 1))

    # Cross-domain two-domain taxa.
    cross2 = species.loc[species["phenotype_domains"].eq(2)].copy()
    cross2 = cross2.sort_values(
        ["diet_exposures", "cgm_outcomes", "highest_robust_paths"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    # Domain-specific taxa.
    domain1 = species.loc[species["phenotype_domains"].eq(1)].copy()
    domain1 = domain1.sort_values(
        ["diet_exposures", "cgm_outcomes", "highest_robust_paths"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. Diet x phenotype-domain architecture
    # ------------------------------------------------------------------
    exposure_domain = (
        h.groupby(["exposure", "phenotype_domain"])
        .agg(
            highest_robust_paths=("species", "size"),
            unique_species=("species", "nunique"),
            unique_outcomes=("outcome_field", "nunique"),
        )
        .reset_index()
    )
    exposure_domain["exposure"] = pd.Categorical(
        exposure_domain["exposure"], categories=DIET_ORDER, ordered=True
    )
    exposure_domain["phenotype_domain"] = pd.Categorical(
        exposure_domain["phenotype_domain"],
        categories=DOMAIN_ORDER,
        ordered=True,
    )
    exposure_domain = exposure_domain.sort_values(
        ["exposure", "phenotype_domain"]
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 4. Pan-domain taxa x diet matrix (path counts)
    # ------------------------------------------------------------------
    pan_species = set(pan["species"])
    pan_h = h.loc[h["species"].isin(pan_species)].copy()

    pan_diet_matrix = (
        pan_h.groupby(["species", "exposure"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=DIET_ORDER, fill_value=0)
    )
    if not pan.empty:
        pan_diet_matrix = pan_diet_matrix.reindex(pan["species"].tolist())
    pan_diet_matrix = pan_diet_matrix.reset_index()

    # Pan-domain taxa x domain matrix (path counts)
    pan_domain_matrix = (
        pan_h.groupby(["species", "phenotype_domain"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=DOMAIN_ORDER, fill_value=0)
    )
    if not pan.empty:
        pan_domain_matrix = pan_domain_matrix.reindex(pan["species"].tolist())
    pan_domain_matrix = pan_domain_matrix.reset_index()

    # ------------------------------------------------------------------
    # 5. Domain-level recurrent species
    # ------------------------------------------------------------------
    domain_species = (
        h.groupby(["phenotype_domain", "species"])
        .agg(
            highest_robust_paths=("species", "size"),
            diet_exposures=("exposure", "nunique"),
            cgm_outcomes=("outcome_field", "nunique"),
            exposure_list=("exposure", semi_unique),
            outcome_list=("outcome_field", semi_unique),
        )
        .reset_index()
        .sort_values(
            ["phenotype_domain", "diet_exposures", "cgm_outcomes", "highest_robust_paths"],
            ascending=[True, False, False, False],
        )
    )

    # Rank within each domain, descriptively.
    domain_species["domain_rank"] = (
        domain_species.groupby("phenotype_domain").cumcount() + 1
    )

    # ------------------------------------------------------------------
    # 6. Breadth overview
    # ------------------------------------------------------------------
    breadth = (
        species.groupby(["phenotype_domains", "breadth_class"])
        .agg(
            unique_species=("species", "nunique"),
            total_highest_robust_paths=("highest_robust_paths", "sum"),
            median_diet_exposures=("diet_exposures", "median"),
            median_cgm_outcomes=("cgm_outcomes", "median"),
        )
        .reset_index()
        .sort_values("phenotype_domains", ascending=False)
    )

    # ------------------------------------------------------------------
    # Save tables
    # ------------------------------------------------------------------
    paths_out = OUT_DIR / "19a_highest_robust_paths_annotated.csv"
    species_out = OUT_DIR / "19a_species_architecture_summary.csv"
    pan_out = OUT_DIR / "19a_pan_domain_core_taxa.csv"
    cross2_out = OUT_DIR / "19a_cross_domain2_taxa.csv"
    domain1_out = OUT_DIR / "19a_domain_specific_taxa.csv"
    exposure_domain_out = OUT_DIR / "19a_exposure_domain_architecture.csv"
    pan_diet_out = OUT_DIR / "19a_pan_domain_taxa_by_diet.csv"
    pan_domain_out = OUT_DIR / "19a_pan_domain_taxa_by_domain.csv"
    domain_species_out = OUT_DIR / "19a_domain_species_recurrence.csv"
    breadth_out = OUT_DIR / "19a_domain_breadth_overview.csv"

    h.to_csv(paths_out, index=False)
    species.to_csv(species_out, index=False)
    pan.to_csv(pan_out, index=False)
    cross2.to_csv(cross2_out, index=False)
    domain1.to_csv(domain1_out, index=False)
    exposure_domain.to_csv(exposure_domain_out, index=False)
    pan_diet_matrix.to_csv(pan_diet_out, index=False)
    pan_domain_matrix.to_csv(pan_domain_out, index=False)
    domain_species.to_csv(domain_species_out, index=False)
    breadth.to_csv(breadth_out, index=False)

    # ------------------------------------------------------------------
    # Concise terminal output
    # ------------------------------------------------------------------
    print()
    print("--- DOMAIN BREADTH OVERVIEW ---")
    print(breadth.to_string(index=False))

    print()
    print("--- PAN-DOMAIN CORE TAXA: TOP 25 ---")
    show_cols = [
        "pan_domain_rank",
        "species",
        "highest_robust_paths",
        "diet_exposures",
        "cgm_outcomes",
        "phenotype_domains",
        "paths_level",
        "paths_variability",
        "paths_risk_range",
    ]
    print(pan[show_cols].head(25).to_string(index=False))

    print()
    print("--- EXPOSURE x DOMAIN ARCHITECTURE ---")
    print(exposure_domain.to_string(index=False))

    print()
    print("INTERPRETATION RULES:")
    print("- Pan-domain/cross-domain labels are descriptive recurrence summaries, not new significance tests.")
    print("- Primary discovery remains the frozen Step18b result; highest robustness remains the Step18f retained-both subset.")
    print("- Raw path counts across domains are not directly comparable because domains contain unequal numbers of outcomes.")
    print("- Mean glucose and GMI remain highly dependent and are not independent replication.")
    print("- Carbohydrate_pct remains exploratory.")
    print("- These are cross-sectional mediation-style associations, not causal mediation.")

    print()
    print(f"PAN_DOMAIN_SPECIES={len(pan)}")
    print(f"CROSS_DOMAIN2_SPECIES={len(cross2)}")
    print(f"DOMAIN_SPECIFIC_SPECIES={len(domain1)}")
    print(f"TOTAL_UNIQUE_SPECIES={species['species'].nunique()}")
    print(f"HIGHEST_ROBUSTNESS_TOTAL={len(h)}")
    print(f"TOTAL_MATCHES_STEP18F={len(h) == EXPECTED_HIGHEST_ROBUST}")
    print("STEP19A_FINAL_BIOLOGICAL_ARCHITECTURE=PASS")

    print(f"SPECIES_SUMMARY={species_out}")
    print(f"PAN_DOMAIN_CORE={pan_out}")
    print(f"EXPOSURE_DOMAIN={exposure_domain_out}")
    print(f"DOMAIN_SPECIES={domain_species_out}")
    print(f"OUTPUT_DIR={OUT_DIR}")


if __name__ == "__main__":
    main()
