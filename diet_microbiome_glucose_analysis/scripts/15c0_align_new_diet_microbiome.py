#!/usr/bin/env python3
"""Step 15c0 — build and align frozen new diet exposures to microbiome data.

This starts the new-exposure analysis after the corrected-hPDI audit was closed.

Frozen exposures
----------------
1) EAT13_z
   = modified_eat_lancet13_z
   Higher = greater modified EAT-Lancet-13 adherence.

2) NOVA4_z
   = nova4_total_energy_pct_cov80_z
   Total dietary energy denominator; participant NOVA energy-mapping coverage
   must be >=80%.
   Higher = greater observed NOVA4 energy share.

3) Carbohydrate_pct_z
   = carbohydrate_energy_pct_z
   Higher = greater carbohydrate energy share.
   This is an exploratory macronutrient-composition exposure, not a diet-quality
   score.

Important design rule
---------------------
Each exposure retains its own maximum valid sample. There is NO requirement that
all three new exposures be simultaneously complete.

This script only builds aligned analysis data. It fits no association model.

Outputs
-------
diet_microbiome_glucose_analysis/outputs/new_diet_extension/
    data/
        15c0_new_diet_microbiome_scores_aligned.csv
        15c0_new_diet_microbiome_species_aligned.csv
        15c0_new_diet_microbiome_alpha_aligned.csv
        15c0_exposure_definitions.csv
    reports/
        15c0_exposure_overlap_summary.csv
        15c0_exposure_quintile_counts.csv
        15c0_alignment_summary.txt
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")

CANDIDATE = (
    ROOT / "diet_microbiome_glucose_analysis" / "outputs" / "reports"
    / "new_diet_extension" / "15b_new_diet_extension_candidate_master.csv"
)
ORIGINAL_NEW_SCORE = (
    ROOT / "diet_deal" / "outputs" / "05_diet_scores"
    / "new_diet_indicators" / "diet_indicator_participant_scores.csv"
)

MICRO = ROOT / "gut_microbiome_deal" / "data" / "08_species_clr_zscore.csv"
ALPHA = ROOT / "gut_microbiome_deal" / "data" / "06_alpha_diversity.csv"

OUT_ROOT = (
    ROOT / "diet_microbiome_glucose_analysis" / "outputs"
    / "new_diet_extension"
)
DATA_OUT = OUT_ROOT / "data"
REPORT_OUT = OUT_ROOT / "reports"

SCORES_OUT = DATA_OUT / "15c0_new_diet_microbiome_scores_aligned.csv"
MICRO_OUT = DATA_OUT / "15c0_new_diet_microbiome_species_aligned.csv"
ALPHA_OUT = DATA_OUT / "15c0_new_diet_microbiome_alpha_aligned.csv"
DEFS_OUT = DATA_OUT / "15c0_exposure_definitions.csv"
OVERLAP_OUT = REPORT_OUT / "15c0_exposure_overlap_summary.csv"
QUINTILE_OUT = REPORT_OUT / "15c0_exposure_quintile_counts.csv"
SUMMARY_OUT = REPORT_OUT / "15c0_alignment_summary.txt"

META_COLS = ["participant_id", "cohort", "research_stage", "array_index"]

EXPOSURES = {
    "EAT13": {
        "z_col": "modified_eat_lancet13_z",
        "raw_col": "modified_eat_lancet13_raw_0_39",
        "q_col": "EAT13_quintile",
        "polarity": "higher_more_adherent",
        "analysis_role": "primary_extension",
        "definition": "modified EAT-Lancet-13 (0-39), Z standardized",
    },
    "NOVA4": {
        "z_col": "nova4_total_energy_pct_cov80_z",
        "raw_col": "nova4_total_energy_pct",
        "q_col": "NOVA4_quintile",
        "polarity": "higher_more_ultraprocessed",
        "analysis_role": "primary_extension",
        "definition": (
            "observed NOVA4 energy / total dietary energy, "
            "participant mapping-energy coverage >=80%, Z standardized"
        ),
    },
    "Carbohydrate_pct": {
        "z_col": "carbohydrate_energy_pct_z",
        "raw_col": "carbohydrate_energy_pct",
        "q_col": "Carbohydrate_pct_quintile",
        "polarity": "neutral_macronutrient_composition",
        "analysis_role": "exploratory_extension",
        "definition": "carbohydrate energy percentage, Z standardized",
    },
}


def require(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


def norm_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def distribution_quintile(s: pd.Series) -> pd.Series:
    """Create distribution-based quintiles while keeping tied values together.

    Percentile rank with method='average' means equal raw values receive the
    same quintile. This is preferable to arbitrarily splitting ties.
    """
    x = numeric(s)
    out = pd.Series(pd.NA, index=x.index, dtype="Int64")
    valid = x.notna() & np.isfinite(x)
    if int(valid.sum()) == 0:
        return out
    pct = x.loc[valid].rank(method="average", pct=True)
    q = np.ceil(pct * 5).clip(1, 5).astype(int)
    out.loc[valid] = q.to_numpy()
    return out


def main() -> int:
    for p, label in [
        (CANDIDATE, "Step15b candidate exposure master"),
        (ORIGINAL_NEW_SCORE, "original new-diet score table"),
        (MICRO, "microbiome CLR-Z table"),
        (ALPHA, "alpha diversity table"),
    ]:
        require(p, label)

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.mkdir(parents=True, exist_ok=True)

    cand = pd.read_csv(CANDIDATE, low_memory=False)
    raw = pd.read_csv(
        ORIGINAL_NEW_SCORE,
        usecols=lambda c: c in {
            "participant_id",
            "modified_eat_lancet13_raw_0_39",
        },
        low_memory=False,
    )
    micro = pd.read_csv(MICRO, low_memory=False)
    alpha = pd.read_csv(ALPHA, low_memory=False)

    for label, d in [
        ("candidate", cand),
        ("raw score", raw),
        ("microbiome", micro),
        ("alpha", alpha),
    ]:
        if "participant_id" not in d.columns:
            raise RuntimeError(f"{label} has no participant_id")
        d["participant_id"] = norm_id(d["participant_id"])
        if d["participant_id"].duplicated().any():
            raise RuntimeError(f"{label} has duplicate participant_id")

    species = [c for c in micro.columns if c not in META_COLS]
    if len(species) != 379:
        raise RuntimeError(
            f"Expected 379 microbiome species, found {len(species)}"
        )

    required_cand = {
        "participant_id",
        "modified_eat_lancet13_z",
        "nova4_total_energy_pct",
        "nova4_total_energy_pct_cov80_z",
        "carbohydrate_energy_pct",
        "carbohydrate_energy_pct_z",
        "mean_daily_energy_kcal",
        "participant_nova_mapping_energy_coverage",
    }
    missing = sorted(required_cand - set(cand.columns))
    if missing:
        raise RuntimeError(
            f"Candidate master missing frozen fields: {missing}"
        )

    score = cand.merge(raw, on="participant_id", how="left", validate="one_to_one")

    # Friendly analysis aliases. Keep source columns too for provenance.
    score["EAT13_z"] = numeric(score["modified_eat_lancet13_z"])
    score["EAT13_raw"] = numeric(score["modified_eat_lancet13_raw_0_39"])

    score["NOVA4_z"] = numeric(score["nova4_total_energy_pct_cov80_z"])
    score["NOVA4_raw_pct"] = numeric(score["nova4_total_energy_pct"])

    score["Carbohydrate_pct_z"] = numeric(score["carbohydrate_energy_pct_z"])
    score["Carbohydrate_pct_raw"] = numeric(score["carbohydrate_energy_pct"])

    score["mean_daily_energy_kcal"] = numeric(score["mean_daily_energy_kcal"])
    score["participant_nova_mapping_energy_coverage"] = numeric(
        score["participant_nova_mapping_energy_coverage"]
    )

    # Quintiles are descriptive/diversity-analysis variables only.
    score["EAT13_quintile"] = distribution_quintile(score["EAT13_raw"])
    score["NOVA4_quintile"] = distribution_quintile(
        score["NOVA4_raw_pct"].where(score["NOVA4_z"].notna())
    )
    score["Carbohydrate_pct_quintile"] = distribution_quintile(
        score["Carbohydrate_pct_raw"].where(
            score["Carbohydrate_pct_z"].notna()
        )
    )

    micro_ids = set(micro["participant_id"])
    score_ids = set(score["participant_id"])
    aligned_ids = score_ids & micro_ids

    # Preserve canonical microbiome row order.
    micro_aligned = micro.loc[micro["participant_id"].isin(aligned_ids)].copy()
    aligned_order = micro_aligned["participant_id"].tolist()

    score_aligned = (
        pd.DataFrame({"participant_id": aligned_order})
        .merge(score, on="participant_id", how="left", validate="one_to_one")
    )
    alpha_aligned = (
        pd.DataFrame({"participant_id": aligned_order})
        .merge(alpha, on="participant_id", how="left", validate="one_to_one")
    )

    if score_aligned["participant_id"].tolist() != micro_aligned["participant_id"].tolist():
        raise RuntimeError("Aligned score and microbiome participant order differs")
    if alpha_aligned["participant_id"].tolist() != micro_aligned["participant_id"].tolist():
        raise RuntimeError("Aligned alpha and microbiome participant order differs")

    alpha_required = {"shannon_index", "simpson_index"}
    missing_alpha = sorted(alpha_required - set(alpha_aligned.columns))
    if missing_alpha:
        raise RuntimeError(f"Alpha table missing: {missing_alpha}")

    # Output only stable analysis-facing fields + provenance/QC fields.
    score_cols = [
        "participant_id",
        "EAT13_raw",
        "EAT13_z",
        "EAT13_quintile",
        "NOVA4_raw_pct",
        "NOVA4_z",
        "NOVA4_quintile",
        "participant_nova_mapping_energy_coverage",
        "Carbohydrate_pct_raw",
        "Carbohydrate_pct_z",
        "Carbohydrate_pct_quintile",
        "mean_daily_energy_kcal",
    ]
    score_aligned[score_cols].to_csv(SCORES_OUT, index=False)
    micro_aligned.to_csv(MICRO_OUT, index=False)
    alpha_aligned.to_csv(ALPHA_OUT, index=False)

    definitions = []
    alias_map = {
        "EAT13": ("EAT13_raw", "EAT13_z", "EAT13_quintile"),
        "NOVA4": ("NOVA4_raw_pct", "NOVA4_z", "NOVA4_quintile"),
        "Carbohydrate_pct": (
            "Carbohydrate_pct_raw",
            "Carbohydrate_pct_z",
            "Carbohydrate_pct_quintile",
        ),
    }
    for name, cfg in EXPOSURES.items():
        raw_alias, z_alias, q_alias = alias_map[name]
        definitions.append({
            "exposure": name,
            "raw_column": raw_alias,
            "z_column": z_alias,
            "quintile_column": q_alias,
            "polarity": cfg["polarity"],
            "analysis_role": cfg["analysis_role"],
            "definition": cfg["definition"],
            "model2_model3_extra_covariate": "mean_daily_energy_kcal",
        })
    defs = pd.DataFrame(definitions)
    defs.to_csv(DEFS_OUT, index=False)

    # Exposure-specific N; never use all-three-complete as an analysis rule.
    rows = []
    for name, (_, zcol, qcol) in alias_map.items():
        z = numeric(score_aligned[zcol])
        valid = z.notna() & np.isfinite(z)
        q = score_aligned[qcol]
        rows.append({
            "exposure": name,
            "microbiome_union_rows": len(score_aligned),
            "valid_exposure_with_microbiome_N": int(valid.sum()),
            "mean_daily_energy_available_among_valid_N": int(
                score_aligned.loc[valid, "mean_daily_energy_kcal"].notna().sum()
            ),
            "alpha_shannon_available_among_valid_N": int(
                score_aligned.loc[valid, "participant_id"]
                .isin(
                    set(
                        alpha_aligned.loc[
                            numeric(alpha_aligned["shannon_index"]).notna(),
                            "participant_id",
                        ]
                    )
                )
                .sum()
            ),
            "quintile_nonmissing_N": int(q.notna().sum()),
            "quintile_levels_observed": ",".join(
                map(
                    str,
                    sorted(
                        pd.to_numeric(q, errors="coerce")
                        .dropna()
                        .astype(int)
                        .unique()
                        .tolist()
                    ),
                )
            ),
        })

    overlap = pd.DataFrame(rows)
    overlap.to_csv(OVERLAP_OUT, index=False)

    qrows = []
    for name, (_, zcol, qcol) in alias_map.items():
        for q, n in (
            score_aligned.loc[
                numeric(score_aligned[zcol]).notna(), qcol
            ]
            .value_counts(dropna=False)
            .sort_index()
            .items()
        ):
            qrows.append({
                "exposure": name,
                "quintile": q,
                "N": int(n),
            })
    qcounts = pd.DataFrame(qrows)
    qcounts.to_csv(QUINTILE_OUT, index=False)

    lines = [
        "=== STEP 15c0 NEW DIET -> MICROBIOME ALIGNMENT ===",
        "NO_ASSOCIATION_MODELS_FIT=True",
        "ALL_THREE_COMPLETE_REQUIRED=False",
        f"MICROBIOME_SPECIES={len(species)}",
        f"MICROBIOME_UNION_ROWS={len(micro_aligned)}",
        "",
        "--- EXPOSURE-SPECIFIC OVERLAP ---",
        overlap.to_string(index=False),
        "",
        "--- QUINTILE COUNTS ---",
        qcounts.to_string(index=False),
        "",
        "FROZEN EXPOSURES:",
        "- EAT13_z = modified EAT-Lancet-13, primary extension",
        "- NOVA4_z = total-energy denominator + >=80% mapping coverage, primary extension",
        "- Carbohydrate_pct_z = exploratory macronutrient exposure",
        "",
        "ADJUSTED MODELS:",
        "- Model2-new = original Model2 covariates + mean_daily_energy_kcal",
        "- Model3-new = Model2-new + BMI",
        "",
        f"SCORES_ALIGNED={SCORES_OUT}",
        f"MICROBIOME_ALIGNED={MICRO_OUT}",
        f"ALPHA_ALIGNED={ALPHA_OUT}",
    ]
    SUMMARY_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
