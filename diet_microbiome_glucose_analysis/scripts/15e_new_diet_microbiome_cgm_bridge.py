#!/usr/bin/env python3
"""Step 15e — New diet -> microbiome -> CGM bridge analysis.

This mirrors the canonical Step08 bridge logic, but only for Diet-CGM contexts
that passed the frozen Model2-primary + Model3-BMI robustness gate in Step15d3.

Inputs
------
Diet -> microbiome:
    Step15c4 Model2-new
    Step15c5 Model3-new

Microbiome -> CGM:
    canonical Model2
    canonical Model3 (+BMI)

Diet -> CGM gate:
    Step15d3 bridge-eligible contexts

Candidate logic
---------------
For each eligible exposure x CGM outcome:

1) Model2 overlap:
   species significant (FDR<0.05) in BOTH
       diet -> microbiome Model2
       microbiome -> CGM Model2

2) Model3 overlap:
   species significant (FDR<0.05) in BOTH
       diet -> microbiome Model3
       microbiome -> CGM Model3

3) Robust candidate:
   - present in overlap in BOTH Model2 and Model3;
   - diet->species beta has same sign in Model2/3;
   - species->CGM beta has same sign in Model2/3;
   - beta-product direction is concordant with the frozen context rule.

Direction rules
---------------
EAT13:
    higher = more favorable adherence
    adverse CGM outcomes -> expected beta product < 0.
    (Currently EAT13 has no bridge-eligible context, but the rule is defined.)

NOVA4:
    higher = more ultra-processed-food exposure
    adverse CGM outcomes -> expected beta product > 0.

Carbohydrate_pct:
    no a-priori healthy/unhealthy polarity.
    Exploratory bridge consistency is defined as beta-product having the same
    sign as the already-frozen direct Diet->CGM association from Model2/Model3.
    This is explicitly labelled "observed-total-direction concordance", NOT
    health-polarity consistency.

Important
---------
- Carbohydrate_pct remains exploratory even if robust bridge candidates exist.
- This script performs overlap screening only. It does NOT run mediation.
- No canonical output is modified.
- Exact full taxonomy string remains the species key.

Outputs
-------
diet_microbiome_glucose_analysis/outputs/new_diet_extension/
    bridge/
        15e_model2_overlap_species.csv
        15e_model2_overlap_summary.csv
        15e_model3_overlap_species.csv
        15e_model3_overlap_summary.csv
        15e_model2_vs_model3_bridge_robustness_species.csv
        15e_bridge_robustness_summary.csv
        15e_robust_bridge_candidates.csv
        15e_bridge_summary.txt
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import hypergeom


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
BASE = (
    ROOT / "diet_microbiome_glucose_analysis" / "outputs"
    / "new_diet_extension"
)
MODEL_DIR = BASE / "models"
REPORT_DIR = BASE / "reports"
BRIDGE_DIR = BASE / "bridge"

DIET_M2 = MODEL_DIR / "15c4_MWAS_all_model2.csv"
DIET_M3 = MODEL_DIR / "15c5_MWAS_all_model3.csv"

CGM_M2 = (
    ROOT / "diet_microbiome_glucose_analysis" / "outputs"
    / "models" / "06_microbiome_cgm_all_model2.csv"
)
CGM_M3 = (
    ROOT / "diet_microbiome_glucose_analysis" / "outputs"
    / "models" / "07_microbiome_cgm_all_model3_bmi.csv"
)

GATES = REPORT_DIR / "15d3_bridge_eligible_diet_cgm_contexts.csv"

M2_OVERLAP = BRIDGE_DIR / "15e_model2_overlap_species.csv"
M2_SUMMARY = BRIDGE_DIR / "15e_model2_overlap_summary.csv"
M3_OVERLAP = BRIDGE_DIR / "15e_model3_overlap_species.csv"
M3_SUMMARY = BRIDGE_DIR / "15e_model3_overlap_summary.csv"
ROBUST_ALL = BRIDGE_DIR / "15e_model2_vs_model3_bridge_robustness_species.csv"
ROBUST_SUMMARY = BRIDGE_DIR / "15e_bridge_robustness_summary.csv"
ROBUST_CANDIDATES = BRIDGE_DIR / "15e_robust_bridge_candidates.csv"
SUMMARY_TXT = BRIDGE_DIR / "15e_bridge_summary.txt"

EXPECTED_SPECIES = 379
ADVERSE_OUTCOMES = {"mean_glucose", "glucose_cv", "time_above_140"}

EXPOSURE_RULE = {
    "EAT13": {
        "analysis_role": "primary_extension",
        "rule_type": "a_priori_health_polarity",
        "fixed_product_sign": -1,
        "rule_text": "higher EAT13 is favorable; adverse CGM outcome expects beta_product < 0",
    },
    "NOVA4": {
        "analysis_role": "primary_extension",
        "rule_type": "a_priori_exposure_polarity",
        "fixed_product_sign": +1,
        "rule_text": "higher NOVA4 is adverse-facing; adverse CGM outcome expects beta_product > 0",
    },
    "Carbohydrate_pct": {
        "analysis_role": "exploratory_extension",
        "rule_type": "observed_total_direction_concordance",
        "fixed_product_sign": None,
        "rule_text": (
            "no a-priori health polarity; beta_product must match frozen "
            "direct Diet->CGM beta sign"
        ),
    },
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


def load_diet(path: Path, model: int) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {"exposure", "species", "beta", "FDR", "significant_FDR05"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Diet Model{model} missing columns: {missing}")

    out = df[
        ["exposure", "species", "beta", "FDR", "significant_FDR05"]
    ].copy()
    out = out.rename(
        columns={
            "exposure": "diet_score",
            "beta": "beta_diet",
            "FDR": "FDR_diet",
            "significant_FDR05": "diet_significant_FDR05",
        }
    )
    out["species"] = out["species"].astype(str)
    out["beta_diet"] = pd.to_numeric(out["beta_diet"], errors="coerce")
    out["FDR_diet"] = pd.to_numeric(out["FDR_diet"], errors="coerce")
    out["diet_significant_FDR05"] = truthy(out["diet_significant_FDR05"])

    if out[["beta_diet", "FDR_diet"]].isna().any().any():
        raise RuntimeError(f"Diet Model{model} contains nonnumeric beta/FDR")

    for exposure in out["diet_score"].unique():
        piece = out.loc[out["diet_score"].eq(exposure)]
        if len(piece) != EXPECTED_SPECIES:
            raise RuntimeError(
                f"Diet Model{model} {exposure}: expected {EXPECTED_SPECIES}, got {len(piece)}"
            )
        if piece["species"].duplicated().any():
            raise RuntimeError(f"Diet Model{model} {exposure}: duplicate species")
    return out


def load_cgm(path: Path, model: int) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {
        "outcome_name",
        "species",
        "beta_species",
        "FDR_within_outcome379",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Microbiome-CGM Model{model} missing columns: {missing}")

    out = df[
        [
            "outcome_name",
            "species",
            "beta_species",
            "FDR_within_outcome379",
        ]
    ].copy()
    out = out.rename(
        columns={
            "beta_species": "beta_cgm",
            "FDR_within_outcome379": "FDR_cgm",
        }
    )
    out["species"] = out["species"].astype(str)
    out["beta_cgm"] = pd.to_numeric(out["beta_cgm"], errors="coerce")
    out["FDR_cgm"] = pd.to_numeric(out["FDR_cgm"], errors="coerce")
    out["cgm_significant_FDR05"] = out["FDR_cgm"] < .05

    if out[["beta_cgm", "FDR_cgm"]].isna().any().any():
        raise RuntimeError(f"Microbiome-CGM Model{model} contains nonnumeric beta/FDR")

    for outcome in ADVERSE_OUTCOMES:
        piece = out.loc[out["outcome_name"].eq(outcome)]
        if len(piece) != EXPECTED_SPECIES:
            raise RuntimeError(
                f"Microbiome-CGM Model{model} {outcome}: "
                f"expected {EXPECTED_SPECIES}, got {len(piece)}"
            )
        if piece["species"].duplicated().any():
            raise RuntimeError(
                f"Microbiome-CGM Model{model} {outcome}: duplicate species"
            )
    return out


def get_expected_sign(gate_row: pd.Series) -> tuple[int, str, str]:
    exposure = gate_row["diet_score"]
    outcome = gate_row["outcome_name"]

    if outcome not in ADVERSE_OUTCOMES:
        raise RuntimeError(f"Unexpected CGM outcome: {outcome}")

    rule = EXPOSURE_RULE[exposure]
    if rule["fixed_product_sign"] is not None:
        return (
            int(rule["fixed_product_sign"]),
            rule["rule_type"],
            rule["rule_text"],
        )

    b2 = float(gate_row["beta_diet_model2"])
    b3 = float(gate_row["beta_diet_model3"])
    if b2 == 0 or b3 == 0 or np.sign(b2) != np.sign(b3):
        raise RuntimeError(
            f"{exposure} x {outcome}: direct Diet-CGM direction is not robust"
        )

    sign = int(np.sign(b2))
    return sign, rule["rule_type"], rule["rule_text"]


def analyze_layer(
    diet: pd.DataFrame,
    cgm: pd.DataFrame,
    gates: pd.DataFrame,
    model_label: str,
):
    overlap_rows = []
    summary_rows = []

    for _, gate in gates.iterrows():
        exposure = str(gate["diet_score"])
        outcome = str(gate["outcome_name"])

        d = diet.loc[diet["diet_score"].eq(exposure)].copy()
        g = cgm.loc[cgm["outcome_name"].eq(outcome)].copy()

        merged = d.merge(
            g,
            on="species",
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != EXPECTED_SPECIES:
            raise RuntimeError(
                f"{model_label} {exposure} x {outcome}: merged "
                f"{len(merged)} species, expected {EXPECTED_SPECIES}"
            )

        diet_sig = merged["diet_significant_FDR05"]
        cgm_sig = merged["cgm_significant_FDR05"]
        both = diet_sig & cgm_sig

        candidates = merged.loc[both].copy()
        expected_sign, rule_type, rule_text = get_expected_sign(gate)

        candidates["analysis_layer"] = model_label
        candidates["analysis_role"] = EXPOSURE_RULE[exposure]["analysis_role"]
        candidates["cgm_outcome"] = outcome
        candidates["beta_product"] = (
            candidates["beta_diet"] * candidates["beta_cgm"]
        )
        candidates["abs_beta_product"] = candidates["beta_product"].abs()
        candidates["expected_beta_product_sign_numeric"] = expected_sign
        candidates["expected_beta_product_sign"] = (
            "positive" if expected_sign > 0 else "negative"
        )
        candidates["direction_rule_type"] = rule_type
        candidates["direction_rule_text"] = rule_text
        candidates["direction_concordant"] = (
            np.sign(candidates["beta_product"]) == expected_sign
        )
        candidates["direct_diet_cgm_beta_model2"] = float(
            gate["beta_diet_model2"]
        )
        candidates["direct_diet_cgm_beta_model3"] = float(
            gate["beta_diet_model3"]
        )

        overlap_rows.append(candidates)

        M = EXPECTED_SPECIES
        K = int(diet_sig.sum())
        n = int(cgm_sig.sum())
        k = int(both.sum())
        expected_overlap = K * n / M
        enrichment_p = float(hypergeom.sf(k - 1, M, K, n))

        summary_rows.append({
            "analysis_layer": model_label,
            "diet_score": exposure,
            "analysis_role": EXPOSURE_RULE[exposure]["analysis_role"],
            "cgm_outcome": outcome,
            "diet_sig_species": K,
            "cgm_sig_species": n,
            "observed_overlap_species": k,
            "expected_overlap_species": expected_overlap,
            "overlap_enrichment_fold": (
                k / expected_overlap if expected_overlap > 0 else np.nan
            ),
            "hypergeom_enrichment_p": enrichment_p,
            "direction_rule_type": rule_type,
            "expected_beta_product_sign": (
                "positive" if expected_sign > 0 else "negative"
            ),
            "direction_concordant_overlap_species": int(
                candidates["direction_concordant"].sum()
            ),
        })

    overlap = (
        pd.concat(overlap_rows, ignore_index=True)
        if overlap_rows
        else pd.DataFrame()
    )
    summary = pd.DataFrame(summary_rows)
    return overlap, summary


def compare_layers(m2: pd.DataFrame, m3: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["diet_score", "cgm_outcome", "species"]

    p = m2.copy().rename(
        columns={
            "beta_diet": "model2_beta_diet",
            "beta_cgm": "model2_beta_cgm",
            "beta_product": "model2_beta_product",
            "FDR_diet": "model2_FDR_diet",
            "FDR_cgm": "model2_FDR_cgm",
            "direction_concordant": "model2_direction_concordant",
        }
    )
    s = m3.copy().rename(
        columns={
            "beta_diet": "model3_beta_diet",
            "beta_cgm": "model3_beta_cgm",
            "beta_product": "model3_beta_product",
            "FDR_diet": "model3_FDR_diet",
            "FDR_cgm": "model3_FDR_cgm",
            "direction_concordant": "model3_direction_concordant",
        }
    )

    keep_common = [
        "analysis_role",
        "direction_rule_type",
        "direction_rule_text",
        "expected_beta_product_sign",
        "expected_beta_product_sign_numeric",
        "direct_diet_cgm_beta_model2",
        "direct_diet_cgm_beta_model3",
    ]

    keep_p = keys + keep_common + [
        "model2_beta_diet",
        "model2_beta_cgm",
        "model2_beta_product",
        "model2_FDR_diet",
        "model2_FDR_cgm",
        "model2_direction_concordant",
    ]
    keep_s = keys + [
        "model3_beta_diet",
        "model3_beta_cgm",
        "model3_beta_product",
        "model3_FDR_diet",
        "model3_FDR_cgm",
        "model3_direction_concordant",
    ]

    merged = p[keep_p].merge(
        s[keep_s],
        on=keys,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    merged["present_model2"] = merged["_merge"].isin(["left_only", "both"])
    merged["present_model3"] = merged["_merge"].isin(["right_only", "both"])
    merged["overlap_both_models"] = merged["_merge"].eq("both")

    merged["diet_beta_same_sign_M2_M3"] = np.where(
        merged["overlap_both_models"],
        np.sign(merged["model2_beta_diet"])
        == np.sign(merged["model3_beta_diet"]),
        False,
    )
    merged["cgm_beta_same_sign_M2_M3"] = np.where(
        merged["overlap_both_models"],
        np.sign(merged["model2_beta_cgm"])
        == np.sign(merged["model3_beta_cgm"]),
        False,
    )
    merged["product_same_sign_M2_M3"] = np.where(
        merged["overlap_both_models"],
        np.sign(merged["model2_beta_product"])
        == np.sign(merged["model3_beta_product"]),
        False,
    )

    merged["robust_direction_consistent_candidate"] = (
        merged["overlap_both_models"]
        & merged["model2_direction_concordant"].fillna(False)
        & merged["model3_direction_concordant"].fillna(False)
        & merged["diet_beta_same_sign_M2_M3"]
        & merged["cgm_beta_same_sign_M2_M3"]
    )

    merged = merged.drop(columns="_merge")

    summary = (
        merged.groupby(
            ["diet_score", "analysis_role", "cgm_outcome"],
            as_index=False,
            dropna=False,
        )
        .agg(
            model2_overlap_species=("present_model2", "sum"),
            model3_overlap_species=("present_model3", "sum"),
            overlap_present_both_models=("overlap_both_models", "sum"),
            robust_direction_consistent_candidates=(
                "robust_direction_consistent_candidate",
                "sum",
            ),
        )
    )
    return merged, summary


def main() -> int:
    for p, label in [
        (DIET_M2, "new-diet MWAS Model2"),
        (DIET_M3, "new-diet MWAS Model3"),
        (CGM_M2, "microbiome-CGM Model2"),
        (CGM_M3, "microbiome-CGM Model3"),
        (GATES, "Step15d3 bridge-eligible contexts"),
    ]:
        require(p, label)

    BRIDGE_DIR.mkdir(parents=True, exist_ok=True)

    diet2 = load_diet(DIET_M2, 2)
    diet3 = load_diet(DIET_M3, 3)
    cgm2 = load_cgm(CGM_M2, 2)
    cgm3 = load_cgm(CGM_M3, 3)

    gates = pd.read_csv(GATES, low_memory=False)
    required_gate = {
        "diet_score",
        "analysis_role",
        "outcome_name",
        "beta_diet_model2",
        "beta_diet_model3",
        "bridge_gate_eligible",
    }
    missing = sorted(required_gate - set(gates.columns))
    if missing:
        raise RuntimeError(f"Gate table missing columns: {missing}")

    gates = gates.loc[truthy(gates["bridge_gate_eligible"])].copy()
    if gates.empty:
        raise RuntimeError("No bridge-eligible Diet-CGM contexts")

    unknown = sorted(set(gates["diet_score"]) - set(EXPOSURE_RULE))
    if unknown:
        raise RuntimeError(f"Unknown bridge exposure(s): {unknown}")

    if gates.duplicated(["diet_score", "outcome_name"]).any():
        raise RuntimeError("Duplicate bridge-eligible Diet-CGM contexts")

    print("=== STEP 15e NEW DIET -> MICROBIOME -> CGM BRIDGE ===")
    print(f"ELIGIBLE_CONTEXTS={len(gates)}")
    print("MODEL2=primary")
    print("MODEL3=BMI sensitivity")
    print("MEDIATION_RUN=False")

    print("\n--- BRIDGE-ELIGIBLE CONTEXTS ---")
    print(
        gates[
            [
                "diet_score",
                "analysis_role",
                "outcome_name",
                "beta_diet_model2",
                "beta_diet_model3",
            ]
        ].to_string(index=False)
    )

    m2_overlap, m2_summary = analyze_layer(
        diet2, cgm2, gates, "MODEL2_PRIMARY"
    )
    m3_overlap, m3_summary = analyze_layer(
        diet3, cgm3, gates, "MODEL3_BMI"
    )

    robust_all, robust_summary = compare_layers(m2_overlap, m3_overlap)
    candidates = robust_all.loc[
        robust_all["robust_direction_consistent_candidate"]
    ].copy()

    m2_overlap.to_csv(M2_OVERLAP, index=False)
    m2_summary.to_csv(M2_SUMMARY, index=False)
    m3_overlap.to_csv(M3_OVERLAP, index=False)
    m3_summary.to_csv(M3_SUMMARY, index=False)
    robust_all.to_csv(ROBUST_ALL, index=False)
    robust_summary.to_csv(ROBUST_SUMMARY, index=False)
    candidates.to_csv(ROBUST_CANDIDATES, index=False)

    print("\n--- MODEL2 OVERLAP SUMMARY ---")
    print(m2_summary.to_string(index=False))

    print("\n--- MODEL3 OVERLAP SUMMARY ---")
    print(m3_summary.to_string(index=False))

    print("\n--- ROBUST BRIDGE SUMMARY ---")
    print(robust_summary.to_string(index=False))

    total_robust = len(candidates)
    by_exposure = (
        candidates.groupby(
            ["diet_score", "analysis_role"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "robust_paths"})
    )

    print("\n--- ROBUST CANDIDATES BY EXPOSURE ---")
    print(by_exposure.to_string(index=False))

    lines = [
        "=== STEP 15e NEW DIET -> MICROBIOME -> CGM BRIDGE SUMMARY ===",
        f"ELIGIBLE_CONTEXTS={len(gates)}",
        f"ROBUST_DIRECTION_CONSISTENT_PATHS={total_robust}",
        "MEDIATION_RUN=False",
        "",
        "--- ROBUST BRIDGE SUMMARY ---",
        robust_summary.to_string(index=False),
        "",
        "--- ROBUST CANDIDATES BY EXPOSURE ---",
        by_exposure.to_string(index=False),
        "",
        "DIRECTION RULES:",
        "- EAT13: a-priori favorable polarity; adverse outcomes expect negative product.",
        "- NOVA4: a-priori adverse-facing polarity; adverse outcomes expect positive product.",
        "- Carbohydrate_pct: exploratory only; product must match the frozen direct",
        "  Diet->CGM association direction. This is NOT a health-polarity claim.",
        "",
        "INTERPRETATION:",
        "- Robust bridge candidates are overlap-screening results, not evidence of causality.",
        "- Full taxonomy strings are the authoritative species keys.",
        "- Only these candidates should be considered for the next mediation-style step.",
        "",
        f"ROBUST_CANDIDATES={ROBUST_CANDIDATES}",
    ]
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
