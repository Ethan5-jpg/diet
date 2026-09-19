#!/usr/bin/env python3
"""
Step 14b — freeze the Paper-20 outcome analysis registry after QC.

READ/FREEZE ONLY:
- no association models
- no microbiome models
- no candidate discovery
- no change to the finalized CGM phenotype table

Purpose
-------
Convert Step14a QC into a prespecified analysis contract BEFORE looking at
Diet->CGM Paper-20 association results.

Primary outcome family:
    Paper-20 (20 outcomes)

Already-existing project supplementary outcomes:
    MAGE, TBR70, TIR70_180, SDb, SDdm

Primary Diet->CGM multiplicity plan for the later full analysis:
    existing_primary: AHEI, AMED, corrected hPDI, rEDIH  = 4 x 20 = 80 tests
    new_primary:      EAT13, NOVA4                       = 2 x 20 = 40 tests
    exploratory:      Carbohydrate_pct                   = 1 x 20 = 20 tests

Primary FDR will be BH within each prespecified exposure-role family.
Also report:
    - BH over all 140 Paper-20 Diet->CGM tests (sensitivity)
    - BH within each diet over 20 outcomes (descriptive/sensitivity)

Outcome-model policy
--------------------
All Paper-20 outcomes keep the canonical continuous Z-score Model2/Model3
for comparability with the existing project and source-paper phenotype set.

Distribution-sensitive outcomes additionally require prespecified sensitivity:
    TAR140  : two-part/hurdle sensitivity
    TAR180  : two-part/hurdle sensitivity; NOT an automatic downstream bridge gate
    HBGI    : log1p/raw-scale sensitivity + robust/HC3 check
    GRADE   : log1p/raw-scale sensitivity + robust/HC3 check

The QC flags are review triggers, not exclusions.

Important:
    Mean glucose and GMI are intentionally BOTH retained because the source
    phenotype set contains both, despite essentially deterministic correlation.
    They must not be described as independent replication.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
PACKAGE = ROOT / "cgm_deal" / "cgm论文新增17指标"

EXPOSURE_ROLES = {
    "AHEI": "existing_primary",
    "AMED": "existing_primary",
    "hPDI": "existing_primary_corrected",
    "rEDIH": "existing_primary",
    "EAT13": "new_primary",
    "NOVA4": "new_primary",
    "Carbohydrate_pct": "exploratory",
}

ROLE_FDR_FAMILIES = {
    "existing_primary": ["AHEI", "AMED", "hPDI", "rEDIH"],
    "new_primary": ["EAT13", "NOVA4"],
    "exploratory": ["Carbohydrate_pct"],
}

SPECIAL = {
    "cgm_above_140": {
        "analysis_class": "distribution_sensitive",
        "primary_model": "canonical_z_OLS_Model2_Model3",
        "sensitivity_model": "two_part_any_positive_plus_positive_log1p",
        "downstream_bridge_rule": "primary_gate_requires_sensitivity_direction_check",
        "reason": "QC: right-skewed TAR140; nontrivial zero mass.",
    },
    "cgm_above_180": {
        "analysis_class": "distribution_sensitive_high_priority_review",
        "primary_model": "canonical_z_OLS_Model2_Model3_for_continuity",
        "sensitivity_model": "two_part_any_positive_plus_positive_log1p",
        "downstream_bridge_rule": "do_not_auto_open_bridge_gate_from_OLS_alone",
        "reason": "QC: ~75% zeros and extreme right skew; ordinary OLS alone is insufficient.",
    },
    "cgm_hbgi": {
        "analysis_class": "distribution_sensitive",
        "primary_model": "canonical_z_OLS_Model2_Model3",
        "sensitivity_model": "log1p_clean_scale_plus_HC3",
        "downstream_bridge_rule": "primary_gate_requires_sensitivity_direction_check",
        "reason": "QC: right skew plus >2% robust cleaning burden.",
    },
    "cgm_grade": {
        "analysis_class": "distribution_sensitive",
        "primary_model": "canonical_z_OLS_Model2_Model3",
        "sensitivity_model": "log1p_clean_scale_plus_HC3",
        "downstream_bridge_rule": "primary_gate_requires_sensitivity_direction_check",
        "reason": "QC: >2% robust cleaning burden.",
    },
}

SUPP_SPECIAL = {
    "cgm_below_70": "distribution_sensitive_existing_project_outcome",
    "cgm_in_range_70_180": "distribution_sensitive_existing_project_outcome",
}


def latest_qc():
    xs = []
    for p in (PACKAGE / "outputs").glob("paper20_plus5_qc_*"):
        f = p / "01_outcome_qc_summary.csv"
        if f.is_file():
            xs.append((f.stat().st_mtime, p))
    if not xs:
        raise FileNotFoundError("No completed paper20_plus5_qc_* run found")
    xs.sort(reverse=True)
    return xs[0][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    qc_dir = Path(a.qc_dir).expanduser().resolve() if a.qc_dir else latest_qc()
    qc_path = qc_dir / "01_outcome_qc_summary.csv"
    if not qc_path.is_file():
        raise FileNotFoundError(qc_path)

    qc = pd.read_csv(qc_path, low_memory=False)
    if len(qc) != 25:
        raise RuntimeError(f"Expected 25 QC outcomes, found {len(qc)}")

    paper = qc.loc[qc["family"].eq("paper20_primary")].copy()
    supp = qc.loc[qc["family"].eq("project_supplementary")].copy()
    if len(paper) != 20 or len(supp) != 5:
        raise RuntimeError(
            f"Expected Paper20=20 and Project5=5; got {len(paper)}, {len(supp)}"
        )
    if not paper["z_standardized_ok"].astype(bool).all():
        raise RuntimeError("At least one Paper-20 Z-score failed standardization QC")

    rows = []
    for _, r in paper.iterrows():
        field = r["field"]
        spec = SPECIAL.get(field)
        if spec is None:
            spec = {
                "analysis_class": "standard_continuous",
                "primary_model": "canonical_z_OLS_Model2_Model3",
                "sensitivity_model": "HC3_robust_SE_check",
                "downstream_bridge_rule": "standard_primary_gate",
                "reason": "Step14a QC did not flag major distribution sensitivity.",
            }
        rows.append({
            "outcome_family": "paper20_primary",
            "label": r["label"],
            "field": field,
            "z_column": field + "_z",
            "analysis_class": spec["analysis_class"],
            "primary_model": spec["primary_model"],
            "sensitivity_model": spec["sensitivity_model"],
            "downstream_bridge_rule": spec["downstream_bridge_rule"],
            "qc_flags": r["qc_flags"],
            "raw_zero_fraction": r["raw_zero_fraction"],
            "skewness": r["skewness"],
            "missing_pct": r["missing_pct"],
            "reason": spec["reason"],
            "include_in_primary_paper20": True,
        })

    for _, r in supp.iterrows():
        field = r["field"]
        rows.append({
            "outcome_family": "project_supplementary",
            "label": r["label"],
            "field": field,
            "z_column": field + "_z",
            "analysis_class": SUPP_SPECIAL.get(
                field, "standard_existing_project_outcome"
            ),
            "primary_model": "already_existing_project_analysis",
            "sensitivity_model": "use_existing_frozen_project_rules",
            "downstream_bridge_rule": "do_not_mix_with_new_Paper20_primary_family",
            "qc_flags": r["qc_flags"],
            "raw_zero_fraction": r["raw_zero_fraction"],
            "skewness": r["skewness"],
            "missing_pct": r["missing_pct"],
            "reason": "Existing project supplementary outcome; retained for context/QC.",
            "include_in_primary_paper20": False,
        })

    registry = pd.DataFrame(rows)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = (
        Path(a.out_dir).expanduser().resolve()
        if a.out_dir
        else PACKAGE / "outputs" / f"paper20_analysis_registry_{stamp}"
    )
    out.mkdir(parents=True, exist_ok=False)

    reg_path = out / "01_paper20_outcome_analysis_registry.csv"
    fdr_path = out / "02_diet_cgm_fdr_plan.csv"
    txt_path = out / "03_registry_summary.txt"
    manifest = out / "manifest.json"

    registry.to_csv(reg_path, index=False)

    fdr_rows = []
    for family, diets in ROLE_FDR_FAMILIES.items():
        fdr_rows.append({
            "fdr_family": family,
            "diet_scores": ";".join(diets),
            "n_diets": len(diets),
            "n_outcomes": 20,
            "n_tests": len(diets) * 20,
            "primary_or_sensitivity": "primary",
            "method": "Benjamini-Hochberg",
        })
    fdr_rows += [
        {
            "fdr_family": "global_all_paper20",
            "diet_scores": ";".join(EXPOSURE_ROLES),
            "n_diets": 7,
            "n_outcomes": 20,
            "n_tests": 140,
            "primary_or_sensitivity": "sensitivity",
            "method": "Benjamini-Hochberg",
        },
        {
            "fdr_family": "within_each_diet",
            "diet_scores": "one diet at a time",
            "n_diets": 1,
            "n_outcomes": 20,
            "n_tests": 20,
            "primary_or_sensitivity": "descriptive_sensitivity",
            "method": "Benjamini-Hochberg",
        },
    ]
    pd.DataFrame(fdr_rows).to_csv(fdr_path, index=False)

    n_special = int(registry.loc[
        registry["outcome_family"].eq("paper20_primary"),
        "analysis_class",
    ].str.startswith("distribution_sensitive").sum())

    lines = [
        "=== STEP 14b PAPER-20 ANALYSIS REGISTRY FROZEN ===",
        f"QC_SOURCE={qc_dir}",
        "MODELS_FIT=False",
        "PAPER20_PRIMARY_OUTCOMES=20",
        "PROJECT_SUPPLEMENTARY_OUTCOMES=5",
        f"PAPER20_DISTRIBUTION_SENSITIVE={n_special}",
        "",
        "PRIMARY DIET->CGM FDR FAMILIES:",
        "existing_primary: AHEI + AMED + corrected hPDI + rEDIH = 80 tests",
        "new_primary: EAT13 + NOVA4 = 40 tests",
        "exploratory: Carbohydrate_pct = 20 tests",
        "global 140-test BH-FDR = sensitivity only",
        "within-diet 20-outcome BH-FDR = descriptive sensitivity",
        "",
        "SPECIAL OUTCOME RULES:",
        "TAR140: OLS primary + two-part sensitivity; direction check before bridge use.",
        "TAR180: OLS continuity + mandatory two-part sensitivity; OLS alone cannot open bridge gate.",
        "HBGI: OLS primary + log1p/HC3 sensitivity.",
        "GRADE: OLS primary + log1p/HC3 sensitivity.",
        "",
        "DEPENDENCY NOTE:",
        "Mean glucose and GMI are both retained because the source phenotype set contains both.",
        "Their near-deterministic correlation must not be described as independent replication.",
        "",
        "NEXT:",
        "Run 7-diet x 20-outcome Diet->CGM Model2/Model3 under this frozen registry.",
        f"REGISTRY={reg_path}",
        f"FDR_PLAN={fdr_path}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest.write_text(json.dumps({
        "status": "frozen",
        "models_fit": False,
        "qc_source": str(qc_dir),
        "paper20_primary_outcomes": 20,
        "project_supplementary_outcomes": 5,
        "paper20_distribution_sensitive": n_special,
        "registry": str(reg_path),
        "fdr_plan": str(fdr_path),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
