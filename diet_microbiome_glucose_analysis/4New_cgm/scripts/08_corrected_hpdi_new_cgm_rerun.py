#!/usr/bin/env python3
"""Targeted corrected-hPDI rerun for the existing 4New_cgm branch.

Refits ONLY hPDI -> {MAGE, TAR180, TBR70, TIR70_180}.
Other 6 diet exposures, microbiome models, and CGM phenotypes are unchanged.

Canonical corrected hPDI:
diet_microbiome_glucose_analysis/outputs/reports/new_diet_extension/
hpdi_correction_audit/corrected_scores/hpdi_participant_scores.csv
column = hpdi_score_energy_adjusted_z

The script reconstructs the full planned 7x4 diet table before BH-FDR, so the
original families remain:
  original_four = 16 tests
  new_primary   = 8 tests
  exploratory   = 4 tests
  global        = 28 tests
"""

from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
BRANCH_CANDIDATES = [
    ROOT / "diet_microbiome_glucose_analysis" / "4New_cgm",
    ROOT / "4New_cgm",
]
CORRECTED_HPDI = (
    ROOT / "diet_microbiome_glucose_analysis" / "outputs" / "reports"
    / "new_diet_extension" / "hpdi_correction_audit" / "corrected_scores"
    / "hpdi_participant_scores.csv"
)
CORRECTED_Z = "hpdi_score_energy_adjusted_z"

def require(p, label):
    p = Path(p)
    if not p.is_file():
        raise FileNotFoundError(f"{label} missing: {p}")

def sha256(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def norm_id(s):
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

def truthy(s):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).eq(1)
    return s.astype(str).str.strip().str.lower().isin(
        {"true","1","1.0","yes","y","t"}
    )

def find_branch(explicit=None):
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(p)
        return p
    found = [p for p in BRANCH_CANDIDATES if p.is_dir()]
    if not found:
        raise FileNotFoundError("4New_cgm not found")
    return found[0]

def find_legacy_run(branch, explicit=None):
    if explicit:
        p = Path(explicit).expanduser().resolve()
        require(p/"models"/"diet_cgm_all_models.csv", "legacy diet results")
        return p
    cand = []
    for p in (branch/"outputs").glob("run_*"):
        f = p/"models"/"diet_cgm_all_models.csv"
        if f.is_file():
            cand.append((f.stat().st_mtime, p))
    if not cand:
        raise FileNotFoundError("No prior run_* with diet_cgm_all_models.csv")
    return sorted(cand, reverse=True)[0][1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch-dir")
    ap.add_argument("--legacy-run-dir")
    ap.add_argument("--config")
    args = ap.parse_args()

    branch = find_branch(args.branch_dir)
    scripts = branch/"scripts"
    sys.path.insert(0, str(scripts))
    import new_cgm_data as data
    import new_cgm_models as models

    config_path = Path(args.config).resolve() if args.config else branch/"config"/"analysis.json"
    require(config_path, "config")
    require(CORRECTED_HPDI, "corrected hPDI")
    legacy_run = find_legacy_run(branch, args.legacy_run_dir)
    legacy_path = legacy_run/"models"/"diet_cgm_all_models.csv"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = branch/"outputs"/f"corrected_hpdi_{stamp}"
    (out/"models").mkdir(parents=True)
    (out/"reports").mkdir()

    print("=== CORRECTED hPDI x FOUR NEW CGM RERUN ===")
    print(f"BRANCH_DIR={branch}")
    print(f"LEGACY_RUN={legacy_run}")
    print(f"CORRECTED_HPDI={CORRECTED_HPDI}")
    print(f"CORRECTED_Z_COL={CORRECTED_Z}")
    print("OTHER_DIETS_REFIT=False")
    print("MICROBIOME_REFIT=False")
    print("CGM_RECOMPUTED=False")

    config = data.load_config(config_path)
    diet, _, _, _, _, _ = data.prepare_tables(config)

    corrected = pd.read_csv(CORRECTED_HPDI, low_memory=False)
    if "participant_id" not in corrected or CORRECTED_Z not in corrected:
        raise RuntimeError("Corrected hPDI file missing required columns")
    corrected["participant_id"] = norm_id(corrected["participant_id"])
    if corrected["participant_id"].duplicated().any():
        raise RuntimeError("Corrected hPDI duplicate participant_id")
    corrected[CORRECTED_Z] = pd.to_numeric(corrected[CORRECTED_Z], errors="coerce")

    diet2 = diet.merge(
        corrected[["participant_id", CORRECTED_Z]],
        on="participant_id", how="left", validate="one_to_one"
    )
    diet2["hPDI_z_legacy"] = pd.to_numeric(diet2["hPDI_z"], errors="coerce")
    diet2["hPDI_z"] = pd.to_numeric(diet2[CORRECTED_Z], errors="coerce")

    both = diet2["hPDI_z_legacy"].notna() & diet2["hPDI_z"].notna()
    rho = spearmanr(
        diet2.loc[both,"hPDI_z_legacy"],
        diet2.loc[both,"hPDI_z"]
    ).statistic
    changed = int(
        (diet2.loc[both,"hPDI_z_legacy"] - diet2.loc[both,"hPDI_z"])
        .abs().gt(1e-12).sum()
    )

    print("\n--- SCORE ALIGNMENT QC ---")
    print(f"DIET_COHORT_ROWS={len(diet2)}")
    print(f"LEGACY_HPDI_AVAILABLE={int(diet2['hPDI_z_legacy'].notna().sum())}")
    print(f"CORRECTED_HPDI_AVAILABLE={int(diet2['hPDI_z'].notna().sum())}")
    print(f"BOTH_AVAILABLE={int(both.sum())}")
    print(f"OLD_VS_CORRECTED_SPEARMAN_RHO={rho:.9f}")
    print(f"Z_VALUE_CHANGED_N={changed}")

    # Refit ONLY hPDI rows.
    old_exposures = models.EXPOSURES
    try:
        models.EXPOSURES = {"hPDI": ("hPDI_z", "original_four")}
        hpdi, _, _, _ = models.fit_diet(diet2, config)
    finally:
        models.EXPOSURES = old_exposures

    # Discard hPDI-only q values; they used a 4-test family.
    fdr_cols = [
        "FDR_family","FDR_global","significant_family_05","significant_global_05",
        "family_tests_planned","global_tests_planned"
    ]
    hpdi = hpdi.drop(columns=[c for c in fdr_cols if c in hpdi], errors="ignore")

    legacy = pd.read_csv(legacy_path, low_memory=False)
    old_hpdi = legacy.loc[legacy["exposure"].eq("hPDI")].copy()

    key = ["analysis_set","model","selection_model","exposure","outcome"]
    if set(map(tuple, old_hpdi[key].astype(str).to_numpy())) != set(
        map(tuple, hpdi[key].astype(str).to_numpy())
    ):
        raise RuntimeError("Old/corrected hPDI planned model keys differ")

    other = legacy.loc[~legacy["exposure"].eq("hPDI")].copy()
    other = other.drop(columns=[c for c in fdr_cols if c in other], errors="ignore")
    combined = pd.concat([other, hpdi], ignore_index=True, sort=False)

    # 7 exposures x 4 outcomes = 28 tests for every planned analysis/model set.
    counts = combined.groupby(["analysis_set","model"]).size()
    if not counts.eq(28).all():
        raise RuntimeError(f"Expected 28 planned diet tests per set/model, got {counts.to_dict()}")

    combined = models.correct_fdr(combined, "diet")

    idcols = ["analysis_set","model","selection_model","outcome"]
    vals = [
        "N","sample_sha256","beta","SE","CI95_lower","CI95_upper","p_value",
        "FDR_family","FDR_global","significant_family_05","significant_global_05","status"
    ]
    a = old_hpdi[idcols+vals].rename(columns={c:f"{c}_legacy" for c in vals})
    b = combined.loc[combined.exposure.eq("hPDI"), idcols+vals].rename(
        columns={c:f"{c}_corrected" for c in vals}
    )
    cmp = a.merge(b, on=idcols, validate="one_to_one")

    cmp["same_N"] = cmp["N_legacy"].eq(cmp["N_corrected"])
    cmp["same_sample"] = cmp["sample_sha256_legacy"].eq(cmp["sample_sha256_corrected"])
    cmp["beta_same_sign"] = (
        np.sign(pd.to_numeric(cmp["beta_legacy"], errors="coerce"))
        == np.sign(pd.to_numeric(cmp["beta_corrected"], errors="coerce"))
    )
    cmp["beta_delta"] = (
        pd.to_numeric(cmp["beta_corrected"], errors="coerce")
        - pd.to_numeric(cmp["beta_legacy"], errors="coerce")
    )
    cmp["family_gate_changed"] = (
        truthy(cmp["significant_family_05_legacy"])
        != truthy(cmp["significant_family_05_corrected"])
    )
    cmp["global_gate_changed"] = (
        truthy(cmp["significant_global_05_legacy"])
        != truthy(cmp["significant_global_05_corrected"])
    )

    same = models.comparisons(combined, "diet")
    same = same.loc[same["exposure"].eq("hPDI")].copy()

    beta_valid = cmp[["beta_legacy","beta_corrected"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    beta_rho = spearmanr(
        beta_valid["beta_legacy"], beta_valid["beta_corrected"]
    ).statistic

    hpdi.to_csv(out/"models"/"hpdi_corrected_only_models.csv", index=False)
    combined.to_csv(out/"models"/"diet_cgm_all_models_corrected_hpdi.csv", index=False)
    cmp.to_csv(out/"reports"/"hpdi_old_vs_corrected_comparison.csv", index=False)
    same.to_csv(out/"reports"/"hpdi_corrected_same_cohort_comparisons.csv", index=False)

    focus = cmp.loc[
        cmp["analysis_set"].isin(["primary","strict"]) &
        cmp["model"].isin([2,3])
    ]
    primary = focus.loc[focus.analysis_set.eq("primary")]
    strict = focus.loc[focus.analysis_set.eq("strict")]

    lines = [
        "=== CORRECTED hPDI x FOUR NEW CGM SUMMARY ===",
        "CANONICAL_HPDI_SOURCE=corrected",
        f"CORRECTED_HPDI_FILE={CORRECTED_HPDI}",
        f"CORRECTED_HPDI_COLUMN={CORRECTED_Z}",
        f"LEGACY_RUN={legacy_run}",
        "OTHER_DIETS_REFIT=False",
        "MICROBIOME_REFIT=False",
        "CGM_RECOMPUTED=False",
        "FDR_RECONSTRUCTED_ON_FULL_7X4_PLAN=True",
        "",
        "--- SCORE ALIGNMENT ---",
        f"OLD_VS_CORRECTED_SPEARMAN_RHO={rho:.9f}",
        f"Z_VALUE_CHANGED_N={changed}",
        "",
        "--- MODEL STABILITY ---",
        f"MODEL_ROWS={len(cmp)}",
        f"SAME_N={int(cmp.same_N.sum())}/{len(cmp)}",
        f"SAME_SAMPLE={int(cmp.same_sample.sum())}/{len(cmp)}",
        f"BETA_SAME_SIGN={int(cmp.beta_same_sign.sum())}/{len(cmp)}",
        f"BETA_SPEARMAN_RHO={beta_rho:.9f}",
        f"FAMILY_GATE_CHANGED_N={int(cmp.family_gate_changed.sum())}",
        f"GLOBAL_GATE_CHANGED_N={int(cmp.global_gate_changed.sum())}",
        "",
        "--- PRIMARY M2/M3 ---",
        primary[[
            "model","outcome","N_legacy","N_corrected",
            "beta_legacy","beta_corrected",
            "FDR_family_legacy","FDR_family_corrected",
            "significant_family_05_legacy","significant_family_05_corrected",
            "family_gate_changed"
        ]].to_string(index=False),
        "",
        "--- STRICT M2/M3 ---",
        strict[[
            "model","outcome","N_legacy","N_corrected",
            "beta_legacy","beta_corrected",
            "FDR_family_legacy","FDR_family_corrected",
            "significant_family_05_legacy","significant_family_05_corrected",
            "family_gate_changed"
        ]].to_string(index=False),
        "",
        "DECISION:",
        "- Corrected hPDI is the canonical hPDI for all future new-CGM work.",
        "- The other six dietary exposures are unchanged.",
        "- Microbiome->CGM results are unchanged because hPDI is not in those models.",
        "",
        f"OUTPUT_DIR={out}",
    ]
    (out/"reports"/"hpdi_corrected_summary.txt").write_text(
        "\n".join(lines)+"\n", encoding="utf-8"
    )

    manifest = {
        "status":"completed",
        "canonical_hpdi_file":str(CORRECTED_HPDI),
        "canonical_hpdi_column":CORRECTED_Z,
        "canonical_hpdi_sha256":sha256(CORRECTED_HPDI),
        "legacy_run":str(legacy_run),
        "legacy_result_sha256":sha256(legacy_path),
        "fdr_reconstructed_on_full_7x4_plan":True,
        "other_diets_refit":False,
        "microbiome_refit":False,
        "cgm_recomputed":False,
        "output_dir":str(out),
    }
    (out/"reports"/"manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)+"\n",
        encoding="utf-8"
    )

    print("\n" + "\n".join(lines))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
