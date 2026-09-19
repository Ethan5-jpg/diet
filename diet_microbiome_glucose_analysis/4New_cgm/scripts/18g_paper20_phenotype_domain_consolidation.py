#!/usr/bin/env python3
"""
Step 18g — Paper-20 CGM phenotype-domain consolidation

Read-only consolidation of Step18f outputs.
No models are fit, no bootstrap is rerun, and no FDR is recomputed.

Domains:
  glycemic_level:
      mean, median, GMI
  glycemic_variability:
      CV, IQR, MAD, MAG, MODD, SD of ROC,
      Time-of-Day SD, Interday SD, Within-Hour SD
  glycemic_risk_range:
      LBGI, HBGI, TAR140, TAR180, ADRR, COGI, GRADE, GRADE eugly

Important:
- Domain summaries are descriptive; outcomes per domain are unequal.
- Mean glucose and GMI are treated as highly dependent outcomes, not
  independent replication.
- Highest robustness is inherited unchanged from Step18f:
  primary + strict + protocol-window ABX/PPI retained.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
BASE = ROOT / "diet_microbiome_glucose_analysis" / "4New_cgm" / "outputs"
IN_DIR = BASE / "paper20_mediation_threeway_robustness" / "reports"
OUT_DIR = BASE / "paper20_mediation_phenotype_domains" / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATH_FILE = IN_DIR / "18f_threeway_path_robustness.csv"
OUTCOME_FILE = IN_DIR / "18f_outcome_robustness_summary.csv"

DOMAIN_MAP = {
    # Glycemic level
    "cgm_mean": "glycemic_level",
    "cgm_median": "glycemic_level",
    "cgm_gmi": "glycemic_level",

    # Glycemic variability / dynamics
    "cgm_cv": "glycemic_variability",
    "cgm_iqr": "glycemic_variability",
    "cgm_mad": "glycemic_variability",
    "cgm_mag": "glycemic_variability",
    "cgm_modd": "glycemic_variability",
    "cgm_sd_roc": "glycemic_variability",
    "cgm_sdhhmm": "glycemic_variability",
    "cgm_sdw": "glycemic_variability",
    "cgm_sdwsh": "glycemic_variability",

    # Glycemic risk / range / composite
    "cgm_lbgi": "glycemic_risk_range",
    "cgm_hbgi": "glycemic_risk_range",
    "cgm_above_140": "glycemic_risk_range",
    "cgm_above_180": "glycemic_risk_range",
    "cgm_adrr": "glycemic_risk_range",
    "cgm_cogi": "glycemic_risk_range",
    "cgm_grade": "glycemic_risk_range",
    "cgm_grade_eugly": "glycemic_risk_range",
}

DISPLAY = {
    "cgm_mean": "Mean glucose",
    "cgm_median": "Median glucose",
    "cgm_gmi": "GMI",
    "cgm_cv": "CV",
    "cgm_iqr": "IQR",
    "cgm_mad": "MAD",
    "cgm_mag": "MAG",
    "cgm_modd": "MODD",
    "cgm_sd_roc": "SD of ROC",
    "cgm_sdhhmm": "Time-of-Day SD (SDhhmm)",
    "cgm_sdw": "Interday SD (SDw)",
    "cgm_sdwsh": "Within-Hour SD (SDwsh)",
    "cgm_lbgi": "LBGI",
    "cgm_hbgi": "HBGI",
    "cgm_above_140": "TAR140",
    "cgm_above_180": "TAR180",
    "cgm_adrr": "ADRR",
    "cgm_cogi": "COGI",
    "cgm_grade": "GRADE",
    "cgm_grade_eugly": "GRADE eugly",
}


def bool_series(s):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).eq(1)
    return s.astype(str).str.strip().str.lower().isin({"true","1","1.0","yes","y","t"})


def main():
    print("=== STEP 18g PAPER-20 CGM PHENOTYPE-DOMAIN CONSOLIDATION ===")
    print("MODELS_FIT=False")
    print("BOOTSTRAP_RERUN=False")
    print("FDR_RECALCULATED=False")
    print("PRIMARY_DISCOVERY_REDEFINED=False")

    if not PATH_FILE.is_file():
        raise FileNotFoundError(PATH_FILE)
    if not OUTCOME_FILE.is_file():
        raise FileNotFoundError(OUTCOME_FILE)

    paths = pd.read_csv(PATH_FILE, low_memory=False)
    outcomes = pd.read_csv(OUTCOME_FILE, low_memory=False)

    required = {
        "exposure", "outcome_field", "species",
        "robust_primary", "primary_retained_strict",
        "primary_retained_protocol", "highest_robustness"
    }
    missing = sorted(required - set(paths.columns))
    if missing:
        raise RuntimeError(f"Missing required Step18f columns: {missing}")

    for c in ["robust_primary", "primary_retained_strict",
              "primary_retained_protocol", "highest_robustness"]:
        paths[c] = bool_series(paths[c])

    paths["phenotype_domain"] = paths["outcome_field"].map(DOMAIN_MAP)
    unknown = sorted(paths.loc[paths["phenotype_domain"].isna(), "outcome_field"].dropna().unique())
    print(f"UNKNOWN_OUTCOME_FIELDS={unknown}")
    if unknown:
        raise RuntimeError("Unmapped outcome fields found; update DOMAIN_MAP before interpretation.")

    paths["outcome_label_domain"] = paths["outcome_field"].map(DISPLAY).fillna(paths["outcome_field"])

    # Domain summary on frozen path set.
    # First aggregate only quantities that can be computed directly within each
    # phenotype-domain group.  Do NOT try to combine Index objects with "&":
    # in pandas that is elementwise logical-and, not set intersection, and can
    # fail when the two Index objects have different lengths.
    domain = (
        paths.groupby("phenotype_domain", dropna=False)
        .agg(
            paths_tested=("species", "size"),
            outcomes_in_frozen_set=("outcome_field", "nunique"),
            unique_species_tested=("species", "nunique"),
            primary_robust=("robust_primary", "sum"),
            retained_strict=("primary_retained_strict", "sum"),
            retained_protocol=("primary_retained_protocol", "sum"),
            retained_both=("highest_robustness", "sum"),
        )
        .reset_index()
    )

    domain["primary_retention_strict"] = domain["retained_strict"] / domain["primary_robust"].replace(0, np.nan)
    domain["primary_retention_protocol"] = domain["retained_protocol"] / domain["primary_robust"].replace(0, np.nan)
    domain["primary_retention_both"] = domain["retained_both"] / domain["primary_robust"].replace(0, np.nan)
    domain["share_of_all_871"] = domain["retained_both"] / int(paths["highest_robustness"].sum())

    # Highest-robustness unique species are calculated on the filtered frame,
    # then merged back by domain.  This is index-safe and equivalent to the
    # intended calculation.
    sp = (
        paths.loc[paths["highest_robustness"]]
        .groupby("phenotype_domain")["species"]
        .nunique()
        .rename("highest_robust_unique_species")
        .reset_index()
    )
    domain = domain.merge(sp, on="phenotype_domain", how="left")
    domain["highest_robust_unique_species"] = (
        domain["highest_robust_unique_species"].fillna(0).astype(int)
    )

    # Outcome-level annotated table.
    outcomes = outcomes.copy()
    outcomes["phenotype_domain"] = outcomes["outcome_field"].map(DOMAIN_MAP)
    outcomes["outcome_label_domain"] = outcomes["outcome_field"].map(DISPLAY).fillna(outcomes["outcome_field"])
    outcomes = outcomes.sort_values(
        ["phenotype_domain", "retained_both", "retained_both_fraction_of_primary"],
        ascending=[True, False, False]
    )

    # Highest-robust species by phenotype domain.
    h = paths.loc[paths["highest_robustness"]].copy()
    species_domain = (
        h.groupby(["phenotype_domain", "species"], dropna=False)
        .agg(
            highest_robust_paths=("species", "size"),
            diet_exposures=("exposure", "nunique"),
            cgm_outcomes=("outcome_field", "nunique"),
            exposure_list=("exposure", lambda s: ";".join(sorted(set(map(str, s))))),
            outcome_list=("outcome_field", lambda s: ";".join(sorted(set(map(str, s))))),
        )
        .reset_index()
        .sort_values(
            ["phenotype_domain", "highest_robust_paths", "diet_exposures", "cgm_outcomes"],
            ascending=[True, False, False, False]
        )
    )

    # Mean vs GMI redundancy audit, using highest-robust paths only.
    def keyset(outcome):
        z = h.loc[h["outcome_field"].eq(outcome), ["exposure", "species"]]
        return set(map(tuple, z.to_numpy()))

    mean_set = keyset("cgm_mean")
    gmi_set = keyset("cgm_gmi")
    union = mean_set | gmi_set
    inter = mean_set & gmi_set
    redundancy = pd.DataFrame([{
        "mean_highest_robust_paths": len(mean_set),
        "gmi_highest_robust_paths": len(gmi_set),
        "shared_exposure_species_pairs": len(inter),
        "union_exposure_species_pairs": len(union),
        "jaccard_mean_vs_gmi": (len(inter) / len(union)) if union else np.nan,
        "interpretation": "Mean glucose and GMI are dependent phenotypes; overlap is descriptive, not replication."
    }])

    # Output files.
    domain_path = OUT_DIR / "18g_domain_robustness_summary.csv"
    outcome_path = OUT_DIR / "18g_outcome_domain_summary.csv"
    species_path = OUT_DIR / "18g_species_by_domain.csv"
    redundancy_path = OUT_DIR / "18g_mean_gmi_redundancy_audit.csv"
    annotated_path = OUT_DIR / "18g_threeway_paths_with_domains.csv"

    domain.to_csv(domain_path, index=False)
    outcomes.to_csv(outcome_path, index=False)
    species_domain.to_csv(species_path, index=False)
    redundancy.to_csv(redundancy_path, index=False)
    paths.to_csv(annotated_path, index=False)

    print()
    print("--- DOMAIN ROBUSTNESS SUMMARY ---")
    cols = [
        "phenotype_domain", "outcomes_in_frozen_set", "primary_robust",
        "retained_strict", "retained_protocol", "retained_both",
        "primary_retention_both", "share_of_all_871",
        "highest_robust_unique_species"
    ]
    print(domain[cols].to_string(index=False))

    print()
    print("--- OUTCOME-LEVEL HIGHEST ROBUSTNESS ---")
    print(outcomes[
        ["phenotype_domain", "outcome_field", "retained_both",
         "retained_both_fraction_of_primary", "unique_species"]
    ].to_string(index=False))

    print()
    print("--- MEAN vs GMI REDUNDANCY AUDIT ---")
    print(redundancy.to_string(index=False))

    print()
    print("INTERPRETATION RULES:")
    print("- Domain grouping is descriptive and does not alter any Step18f inference.")
    print("- Raw domain path counts are not directly comparable because domains contain unequal numbers of outcomes.")
    print("- Mean glucose and GMI must not be described as independent replication.")
    print("- Sparse outcomes such as TAR140/HBGI/GRADE-eugly require denominator-aware interpretation.")
    print("- Highest robustness remains primary + strict + protocol retained; no new discoveries are created here.")

    total = int(paths["highest_robustness"].sum())
    print()
    print(f"HIGHEST_ROBUSTNESS_TOTAL={total}")
    print(f"TOTAL_MATCHES_STEP18F={total == 871}")
    print("STEP18G_PHENOTYPE_DOMAIN_CONSOLIDATION=PASS")
    print(f"DOMAIN_SUMMARY={domain_path}")
    print(f"OUTCOME_SUMMARY={outcome_path}")
    print(f"SPECIES_BY_DOMAIN={species_path}")
    print(f"MEAN_GMI_AUDIT={redundancy_path}")
    print(f"ANNOTATED_PATHS={annotated_path}")
    print(f"OUTPUT_DIR={OUT_DIR}")


if __name__ == "__main__":
    main()
