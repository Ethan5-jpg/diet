#!/usr/bin/env python3
"""Step 15b2 — isolated corrected-hPDI rebuild and old-vs-new impact audit.

Purpose
-------
Step 15b showed that the 10 exact food IDs flagged during the new-diet-indicator
review occur in 8,376 baseline events across 1,902 participants. That burden is
too large to dismiss from event counts alone.

This script therefore performs an ISOLATED corrected-hPDI rebuild. It does NOT
overwrite the canonical hPDI mapping/intakes/scores and it does NOT refit any
Diet→Gut / Diet→CGM / mediation model.

Exact corrections frozen for this audit
---------------------------------------
1) 1012933 BENEFIBER supplement
   -> not_applicable to hPDI food groups.

2) 1014956 boiled potato with skin
   -> potatoes.

3) Eight exact plant-alternative products identified in the new-indicator review
   -> not_applicable to hPDI component grams because product grams are not
      ingredient-equivalent legume/nut/dairy grams.

The eight plant alternatives are:
1009259, 1010775, 1009011, 1011847,
1012723, 1009092, 1013950, 1007077.

Outputs
-------
diet_microbiome_glucose_analysis/outputs/reports/new_diet_extension/
    hpdi_correction_audit/
        corrected_hpdi_food_id_mapping.csv
        corrected_intakes/...
        corrected_scores/...
        15b2_hpdi_mapping_changes.csv
        15b2_hpdi_component_intake_change_summary.csv
        15b2_hpdi_score_comparison_metrics.csv
        15b2_hpdi_participant_score_comparison.csv
        15b2_corrected_hpdi_impact_summary.txt

Decision principle
------------------
The audit quantifies impact first. Whether old hPDI downstream models need
rerunning is decided only after seeing score concordance and participant-level
changes.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
DIET = ROOT / "diet_deal"
ANALYSIS = ROOT / "diet_microbiome_glucose_analysis"

HPDI_MAPPING = (
    DIET / "outputs" / "03_score_mapping" / "hpdi" / "hpdi_food_id_mapping.csv"
)
OLD_INTAKES = (
    DIET / "outputs" / "04_score_intakes" / "hpdi"
    / "hpdi_participant_component_intakes.csv"
)
WEIGHT_OVERRIDES = (
    DIET / "outputs" / "04_score_intakes" / "hpdi"
    / "hpdi_missing_weight_review.csv"
)
OLD_SCORES = (
    DIET / "outputs" / "05_diet_scores" / "hpdi"
    / "hpdi_participant_scores.csv"
)
EVENTS = ROOT / "Transfer" / "diet_logging" / "diet_logging_events.csv"
NEW_MAPPING = (
    DIET / "outputs" / "03_score_mapping" / "new_diet_indicators"
    / "new_diet_indicator_food_id_mapping.csv"
)

BUILD_SCRIPT = DIET / "scripts" / "hpdi" / "04_build_hpdi_intakes.py"
SCORE_SCRIPT = DIET / "scripts" / "hpdi" / "05_calculate_hpdi_scores.py"

OUT = (
    ANALYSIS / "outputs" / "reports" / "new_diet_extension"
    / "hpdi_correction_audit"
)
CORRECTED_MAPPING = OUT / "corrected_hpdi_food_id_mapping.csv"
CORRECTED_INTAKE_DIR = OUT / "corrected_intakes"
CORRECTED_SCORE_DIR = OUT / "corrected_scores"

MAPPING_CHANGES = OUT / "15b2_hpdi_mapping_changes.csv"
COMPONENT_SUMMARY = OUT / "15b2_hpdi_component_intake_change_summary.csv"
METRICS = OUT / "15b2_hpdi_score_comparison_metrics.csv"
PARTICIPANT_COMPARISON = OUT / "15b2_hpdi_participant_score_comparison.csv"
SUMMARY = OUT / "15b2_corrected_hpdi_impact_summary.txt"

CORRECTIONS = {
    "1012933": {
        "new_status": "not_applicable",
        "new_component": pd.NA,
        "reason": "BENEFIBER fiber supplement is not a whole-food hPDI component",
    },
    "1014956": {
        "new_status": "mapped",
        "new_component": "potatoes",
        "reason": "boiled potato with skin belongs to potatoes",
    },
    "1009259": {
        "new_status": "not_applicable",
        "new_component": pd.NA,
        "reason": "plant alternative; product grams are not ingredient-equivalent component grams",
    },
    "1010775": {
        "new_status": "not_applicable",
        "new_component": pd.NA,
        "reason": "plant alternative; product grams are not ingredient-equivalent component grams",
    },
    "1009011": {
        "new_status": "not_applicable",
        "new_component": pd.NA,
        "reason": "plant alternative; product grams are not ingredient-equivalent component grams",
    },
    "1011847": {
        "new_status": "not_applicable",
        "new_component": pd.NA,
        "reason": "plant alternative; product grams are not ingredient-equivalent component grams",
    },
    "1012723": {
        "new_status": "not_applicable",
        "new_component": pd.NA,
        "reason": "plant alternative; product grams are not ingredient-equivalent component grams",
    },
    "1009092": {
        "new_status": "not_applicable",
        "new_component": pd.NA,
        "reason": "plant alternative; product grams are not ingredient-equivalent component grams",
    },
    "1013950": {
        "new_status": "not_applicable",
        "new_component": pd.NA,
        "reason": "plant alternative; product grams are not ingredient-equivalent component grams",
    },
    "1007077": {
        "new_status": "not_applicable",
        "new_component": pd.NA,
        "reason": "plant alternative; product grams are not ingredient-equivalent component grams",
    },
}

COMPONENTS = [
    "whole_grains",
    "fruits",
    "vegetables",
    "nuts",
    "legumes",
    "vegetable_oils",
    "tea_coffee",
    "fruit_juice",
    "refined_grains",
    "potatoes",
    "sugar_sweetened_beverages",
    "sweets_desserts",
    "animal_fat",
    "dairy",
    "eggs",
    "fish_seafood",
    "meat",
    "miscellaneous_animal_foods",
]


def require(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


def norm_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def spearman(x: pd.Series, y: pd.Series) -> tuple[int, float]:
    a, b = num(x), num(y)
    ok = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 3:
        return int(ok.sum()), np.nan
    rho = a.loc[ok].rank(method="average").corr(
        b.loc[ok].rank(method="average"),
        method="pearson",
    )
    return int(ok.sum()), float(rho)


def pearson(x: pd.Series, y: pd.Series) -> tuple[int, float]:
    a, b = num(x), num(y)
    ok = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 3:
        return int(ok.sum()), np.nan
    return int(ok.sum()), float(a.loc[ok].corr(b.loc[ok]))


def build_corrected_mapping() -> pd.DataFrame:
    mapping = pd.read_csv(HPDI_MAPPING, dtype={"food_id": "string"}, low_memory=False)
    required_cols = {"food_id", "mapping_status", "hpdi_component"}
    missing = sorted(required_cols - set(mapping.columns))
    if missing:
        raise RuntimeError(
            f"Canonical hPDI mapping missing columns: {missing}"
        )
    mapping["food_id"] = norm_id(mapping["food_id"])
    if mapping["food_id"].duplicated().any():
        raise RuntimeError("Canonical hPDI mapping has duplicate food_id")

    missing_ids = sorted(set(CORRECTIONS) - set(mapping["food_id"]))
    if missing_ids:
        raise RuntimeError(
            "Correction food IDs missing from canonical hPDI mapping: "
            + ", ".join(missing_ids)
        )

    identity = pd.DataFrame()
    if NEW_MAPPING.is_file():
        nm = pd.read_csv(NEW_MAPPING, dtype={"food_id": "string"}, low_memory=False)
        nm["food_id"] = norm_id(nm["food_id"])
        keep = [
            c for c in [
                "food_id",
                "canonical_short_food_name",
                "canonical_product_name",
                "canonical_food_category",
                "source_hpdi_component",
                "source_hpdi_mapping_status",
                "food_identity_check",
            ]
            if c in nm.columns
        ]
        identity = nm.loc[
            nm["food_id"].isin(CORRECTIONS), keep
        ].drop_duplicates("food_id")

    audit_rows = []
    corrected = mapping.copy()
    for food_id, rule in CORRECTIONS.items():
        idx = corrected.index[corrected["food_id"].eq(food_id)]
        if len(idx) != 1:
            raise RuntimeError(
                f"Expected one mapping row for food_id={food_id}, found {len(idx)}"
            )
        i = idx[0]
        old_status = corrected.at[i, "mapping_status"]
        old_component = corrected.at[i, "hpdi_component"]

        corrected.at[i, "mapping_status"] = rule["new_status"]
        corrected.at[i, "hpdi_component"] = rule["new_component"]

        if "mapping_basis" in corrected.columns:
            corrected.at[i, "mapping_basis"] = "step15b2_exact_id_correction"
        if "review_notes" in corrected.columns:
            corrected.at[i, "review_notes"] = rule["reason"]

        audit_rows.append({
            "food_id": food_id,
            "old_mapping_status": old_status,
            "old_hpdi_component": old_component,
            "new_mapping_status": rule["new_status"],
            "new_hpdi_component": rule["new_component"],
            "correction_reason": rule["reason"],
        })

    audit = pd.DataFrame(audit_rows)
    if not identity.empty:
        audit = audit.merge(identity, on="food_id", how="left", validate="one_to_one")

    OUT.mkdir(parents=True, exist_ok=True)
    corrected.to_csv(CORRECTED_MAPPING, index=False, encoding="utf-8-sig")
    audit.to_csv(MAPPING_CHANGES, index=False, encoding="utf-8-sig")
    return audit


def run_isolated_pipeline() -> None:
    CORRECTED_INTAKE_DIR.mkdir(parents=True, exist_ok=True)
    CORRECTED_SCORE_DIR.mkdir(parents=True, exist_ok=True)

    cmd_intake = [
        sys.executable,
        str(BUILD_SCRIPT),
        "--input", str(EVENTS),
        "--mapping", str(CORRECTED_MAPPING),
        "--weight-overrides", str(WEIGHT_OVERRIDES),
        "--output-dir", str(CORRECTED_INTAKE_DIR),
        "--cohort", "10k",
        "--research-stage", "00_00_visit",
    ]
    print("\n[15b2] Running isolated corrected hPDI intake build...")
    subprocess.run(cmd_intake, check=True)

    corrected_intakes = (
        CORRECTED_INTAKE_DIR / "hpdi_participant_component_intakes.csv"
    )
    cmd_score = [
        sys.executable,
        str(SCORE_SCRIPT),
        "--input", str(corrected_intakes),
        "--output-dir", str(CORRECTED_SCORE_DIR),
    ]
    print("\n[15b2] Running isolated corrected hPDI score build...")
    subprocess.run(cmd_score, check=True)


def compare_component_intakes() -> pd.DataFrame:
    new_path = CORRECTED_INTAKE_DIR / "hpdi_participant_component_intakes.csv"
    old = pd.read_csv(OLD_INTAKES, low_memory=False)
    new = pd.read_csv(new_path, low_memory=False)
    for d in (old, new):
        d["participant_id"] = norm_id(d["participant_id"])
    if old["participant_id"].duplicated().any() or new["participant_id"].duplicated().any():
        raise RuntimeError("Duplicate participant_id in hPDI intake comparison")

    rows = []
    for component in COMPONENTS:
        col = f"mean_daily_{component}_g_proxy"
        if col not in old.columns or col not in new.columns:
            continue
        m = old[["participant_id", col]].merge(
            new[["participant_id", col]],
            on="participant_id",
            how="inner",
            suffixes=("_old", "_new"),
            validate="one_to_one",
        )
        o = num(m[f"{col}_old"])
        n = num(m[f"{col}_new"])
        delta = n - o
        valid = o.notna() & n.notna()
        changed = valid & delta.abs().gt(1e-12)
        rows.append({
            "component": component,
            "N_pair": int(valid.sum()),
            "N_changed": int(changed.sum()),
            "median_delta_g_day_among_changed": (
                float(delta.loc[changed].median()) if changed.any() else 0.0
            ),
            "median_abs_delta_g_day_among_changed": (
                float(delta.loc[changed].abs().median()) if changed.any() else 0.0
            ),
            "max_abs_delta_g_day": (
                float(delta.loc[valid].abs().max()) if valid.any() else np.nan
            ),
        })
    out = pd.DataFrame(rows)
    out.to_csv(COMPONENT_SUMMARY, index=False)
    return out


def compare_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    new_path = CORRECTED_SCORE_DIR / "hpdi_participant_scores.csv"
    old = pd.read_csv(OLD_SCORES, low_memory=False)
    new = pd.read_csv(new_path, low_memory=False)
    for d in (old, new):
        d["participant_id"] = norm_id(d["participant_id"])
    if old["participant_id"].duplicated().any() or new["participant_id"].duplicated().any():
        raise RuntimeError("Duplicate participant_id in hPDI score comparison")

    score_cols = [
        "hpdi_score_raw_18_90",
        "hpdi_score_energy_adjusted",
        "hpdi_score_energy_adjusted_z",
        "hpdi_energy_adjusted_quintile",
    ]
    missing_old = [c for c in score_cols if c not in old.columns]
    missing_new = [c for c in score_cols if c not in new.columns]
    if missing_old or missing_new:
        raise RuntimeError(
            f"Missing score columns. old={missing_old}, new={missing_new}"
        )

    m = old[["participant_id", *score_cols]].merge(
        new[["participant_id", *score_cols]],
        on="participant_id",
        how="outer",
        suffixes=("_old", "_new"),
        indicator=True,
        validate="one_to_one",
    )

    metrics = []
    for col in score_cols[:3]:
        o = num(m[f"{col}_old"])
        n = num(m[f"{col}_new"])
        valid = o.notna() & n.notna()
        delta = n - o
        nsp, rsp = spearman(o, n)
        npe, rpe = pearson(o, n)
        metrics.append({
            "metric": col,
            "N_pair": int(valid.sum()),
            "spearman_rho": rsp,
            "pearson_r": rpe,
            "N_changed": int((valid & delta.abs().gt(1e-12)).sum()),
            "median_delta": float(delta.loc[valid].median()) if valid.any() else np.nan,
            "median_abs_delta": float(delta.loc[valid].abs().median()) if valid.any() else np.nan,
            "p95_abs_delta": float(delta.loc[valid].abs().quantile(.95)) if valid.any() else np.nan,
            "max_abs_delta": float(delta.loc[valid].abs().max()) if valid.any() else np.nan,
        })

    q_old = num(m["hpdi_energy_adjusted_quintile_old"])
    q_new = num(m["hpdi_energy_adjusted_quintile_new"])
    q_ok = q_old.notna() & q_new.notna()
    metrics.append({
        "metric": "hpdi_energy_adjusted_quintile",
        "N_pair": int(q_ok.sum()),
        "spearman_rho": spearman(q_old, q_new)[1],
        "pearson_r": pearson(q_old, q_new)[1],
        "N_changed": int((q_ok & q_old.ne(q_new)).sum()),
        "median_delta": float((q_new - q_old).loc[q_ok].median()) if q_ok.any() else np.nan,
        "median_abs_delta": float((q_new - q_old).loc[q_ok].abs().median()) if q_ok.any() else np.nan,
        "p95_abs_delta": float((q_new - q_old).loc[q_ok].abs().quantile(.95)) if q_ok.any() else np.nan,
        "max_abs_delta": float((q_new - q_old).loc[q_ok].abs().max()) if q_ok.any() else np.nan,
    })

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(METRICS, index=False)

    # Compact participant table; useful if downstream rerun is needed.
    keep = [
        "participant_id",
        "hpdi_score_raw_18_90_old",
        "hpdi_score_raw_18_90_new",
        "hpdi_score_energy_adjusted_old",
        "hpdi_score_energy_adjusted_new",
        "hpdi_score_energy_adjusted_z_old",
        "hpdi_score_energy_adjusted_z_new",
        "hpdi_energy_adjusted_quintile_old",
        "hpdi_energy_adjusted_quintile_new",
        "_merge",
    ]
    out = m[keep].copy()
    out["delta_raw"] = (
        num(out["hpdi_score_raw_18_90_new"])
        - num(out["hpdi_score_raw_18_90_old"])
    )
    out["delta_energy_adjusted"] = (
        num(out["hpdi_score_energy_adjusted_new"])
        - num(out["hpdi_score_energy_adjusted_old"])
    )
    out["delta_z"] = (
        num(out["hpdi_score_energy_adjusted_z_new"])
        - num(out["hpdi_score_energy_adjusted_z_old"])
    )
    out["quintile_changed"] = (
        num(out["hpdi_energy_adjusted_quintile_new"])
        .ne(num(out["hpdi_energy_adjusted_quintile_old"]))
        & num(out["hpdi_energy_adjusted_quintile_new"]).notna()
        & num(out["hpdi_energy_adjusted_quintile_old"]).notna()
    )
    out.to_csv(PARTICIPANT_COMPARISON, index=False)
    return metrics_df, out


def main() -> int:
    for path, label in [
        (HPDI_MAPPING, "canonical hPDI mapping"),
        (OLD_INTAKES, "canonical hPDI participant intakes"),
        (WEIGHT_OVERRIDES, "canonical hPDI weight review"),
        (OLD_SCORES, "canonical old hPDI scores"),
        (EVENTS, "diet logging events"),
        (BUILD_SCRIPT, "canonical hPDI intake script"),
        (SCORE_SCRIPT, "canonical hPDI score script"),
    ]:
        require(path, label)

    OUT.mkdir(parents=True, exist_ok=True)
    audit = build_corrected_mapping()

    print("=== STEP 15b2 CORRECTED hPDI ISOLATED AUDIT ===")
    print("CANONICAL_FILES_MODIFIED=False")
    print("DOWNSTREAM_ASSOCIATION_MODELS_REFIT=False")
    print("\n--- EXACT MAPPING CORRECTIONS ---")
    print(audit.to_string(index=False))

    run_isolated_pipeline()
    component = compare_component_intakes()
    metrics, participant = compare_scores()

    zrow = metrics.loc[
        metrics["metric"].eq("hpdi_score_energy_adjusted_z")
    ].iloc[0]
    qrow = metrics.loc[
        metrics["metric"].eq("hpdi_energy_adjusted_quintile")
    ].iloc[0]

    lines = [
        "=== STEP 15b2 CORRECTED hPDI IMPACT SUMMARY ===",
        "CANONICAL_FILES_MODIFIED=False",
        "DOWNSTREAM_ASSOCIATION_MODELS_REFIT=False",
        "",
        f"EXACT_CORRECTIONS={len(CORRECTIONS)}",
        "",
        "--- SCORE COMPARISON ---",
        metrics.to_string(index=False),
        "",
        "--- COMPONENTS WITH CHANGED INTAKE ---",
        component.loc[component["N_changed"].gt(0)].to_string(index=False),
        "",
        "--- KEY DECISION NUMBERS ---",
        f"OLD_VS_CORRECTED_Z_SPEARMAN_RHO={zrow['spearman_rho']:.9f}",
        f"OLD_VS_CORRECTED_Z_PEARSON_R={zrow['pearson_r']:.9f}",
        f"Z_N_CHANGED={int(zrow['N_changed'])}",
        f"Z_MEDIAN_ABS_DELTA={zrow['median_abs_delta']:.9f}",
        f"Z_P95_ABS_DELTA={zrow['p95_abs_delta']:.9f}",
        f"Z_MAX_ABS_DELTA={zrow['max_abs_delta']:.9f}",
        f"QUINTILE_N_CHANGED={int(qrow['N_changed'])}",
        "",
        "INTERPRETATION_RULE:",
        "- Very high score concordance alone is not sufficient if a meaningful",
        "  number of participants cross quintiles or have large |delta_z|.",
        "- Decide whether old hPDI downstream models require rerun only after",
        "  reviewing rho, delta_z distribution, quintile changes and component burden.",
        "",
        f"SUMMARY={SUMMARY}",
    ]
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
