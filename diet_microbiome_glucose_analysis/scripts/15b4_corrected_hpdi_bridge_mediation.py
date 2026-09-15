#!/usr/bin/env python3
"""Step 15b4 — corrected hPDI bridge + mediation rerun (isolated).

This step is triggered only after Step 15b3 shows:
- hPDI × glucose_cv Model3 diet-CGM gate is retained;
- hPDI→microbiome Model3 effects remain highly stable.

What this script does
---------------------
1) Rebuild Step08 bridge using:
   - canonical AHEI/AMED/rEDIH Diet→Microbiome Model2/3;
   - corrected hPDI Diet→Microbiome Model2/3 from Step15b3;
   - unchanged canonical Microbiome→CGM Model2/3.

2) Rebuild Step09 candidate ranking in an isolated output directory.

3) Run Step10 mediation ONLY for corrected hPDI robust paths that survive the
   corrected Model3 hPDI→CGM gate. It reuses the canonical Step10 implementation
   and 1000-bootstrap method, but writes to an isolated directory.

4) Compare old vs corrected hPDI mediation:
   - exact tested species set;
   - FDR-significant retained/lost/gained;
   - direction-consistent retained/lost/gained;
   - indirect-effect Spearman rho;
   - indirect sign flips;
   - AMED/hPDI recurrent core retention.

Important
---------
- No canonical file is overwritten.
- AHEI/AMED/rEDIH models are NOT refit.
- Microbiome→CGM models are NOT refit.
- This is still cross-sectional mediation-style analysis, not causal mediation.

Outputs
-------
diet_microbiome_glucose_analysis/outputs/reports/new_diet_extension/
    hpdi_correction_audit/bridge_mediation/
        bridge/
        mediation/models/
        mediation/reports/
        15b4_old_vs_corrected_hpdi_mediation_paths.csv
        15b4_corrected_hpdi_bridge_mediation_summary.txt
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import numpy as np
import pandas as pd


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
DM = ROOT / "diet_microbiome_analysis"
DG = ROOT / "diet_microbiome_glucose_analysis"

STEP08_PATH = DG / "scripts" / "08_diet_microbiome_cgm_overlap.py"
STEP09_PATH = DG / "scripts" / "09_rank_and_consolidate_bridge_candidates.py"
STEP10_PATH = DG / "scripts" / "10_diet_microbiome_cgm_mediation.py"

CORRECTED_MWAS_DIR = (
    DG / "outputs" / "reports" / "new_diet_extension"
    / "hpdi_correction_audit" / "downstream" / "microbiome"
)
CORRECTED_HPDI_M2 = CORRECTED_MWAS_DIR / "15b3_corrected_hPDI_MWAS_model2.csv"
CORRECTED_HPDI_M3 = CORRECTED_MWAS_DIR / "15b3_corrected_hPDI_MWAS_model3.csv"

CORRECTED_CGM_MODEL3 = (
    DG / "outputs" / "reports" / "new_diet_extension"
    / "hpdi_correction_audit" / "downstream" / "cgm"
    / "15b3_primary12_model3_with_corrected_hPDI.csv"
)

CORRECTED_HPDI_SCORES = (
    DG / "outputs" / "reports" / "new_diet_extension"
    / "hpdi_correction_audit" / "corrected_scores"
    / "hpdi_participant_scores.csv"
)
CORRECTED_HPDI_Z = "hpdi_score_energy_adjusted_z"

FORMAL_COHORT = DG / "outputs" / "data" / "00_current4_diet_cgm_cohort.csv"
MICROBIOME = ROOT / "gut_microbiome_deal" / "data" / "08_species_clr_zscore.csv"
COVARIATES = ROOT / "co-variant" / "outputs" / "data" / "02_covariate_master.csv"

CGM_M2 = DG / "outputs" / "models" / "06_microbiome_cgm_all_model2.csv"
CGM_M3 = DG / "outputs" / "models" / "07_microbiome_cgm_all_model3_bmi.csv"

OLD_ROBUST = DG / "outputs" / "reports" / "08_robust_direction_consistent_candidates.csv"
OLD_MEDIATION = DG / "outputs" / "models" / "10_mediation_paths_model3.csv"

OUT = (
    DG / "outputs" / "reports" / "new_diet_extension"
    / "hpdi_correction_audit" / "bridge_mediation"
)
BRIDGE_OUT = OUT / "bridge"
MED_OUT = OUT / "mediation"
MED_MODELS = MED_OUT / "models"
MED_REPORTS = MED_OUT / "reports"

COMBINED_M2 = BRIDGE_OUT / "15b4_combined_diet_microbiome_model2.csv"
COMBINED_M3 = BRIDGE_OUT / "15b4_combined_diet_microbiome_model3.csv"
ROBUST_ALL = BRIDGE_OUT / "15b4_robust_direction_consistent_candidates.csv"
BRIDGE_LONG = BRIDGE_OUT / "15b4_bridge_candidate_paths_long.csv"
BRIDGE_LONG_HPDI = BRIDGE_OUT / "15b4_bridge_candidate_paths_hPDI_only.csv"
CORRECTED_COHORT = MED_OUT / "15b4_formal_cohort_corrected_hPDI.csv"

PATH_COMPARE = OUT / "15b4_old_vs_corrected_hpdi_mediation_paths.csv"
SUMMARY = OUT / "15b4_corrected_hpdi_bridge_mediation_summary.txt"

EXPECTED_SPECIES = 379


def require(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


def norm_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def truthy(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).eq(1)
    return (
        s.astype(str).str.strip().str.lower()
        .isin({"true", "1", "1.0", "yes", "y", "t"})
    )


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def spearman(x: pd.Series, y: pd.Series) -> float:
    a = pd.to_numeric(x, errors="coerce")
    b = pd.to_numeric(y, errors="coerce")
    ok = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 3:
        return np.nan
    return float(
        a.loc[ok].rank(method="average").corr(
            b.loc[ok].rank(method="average"),
            method="pearson",
        )
    )


def corrected_hpdi_piece(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    required = {"species", "beta", "p", "FDR"}
    missing = sorted(required - set(d.columns))
    if missing:
        raise RuntimeError(
            f"Corrected hPDI MWAS missing columns {missing}: {path}"
        )
    if len(d) != EXPECTED_SPECIES:
        raise RuntimeError(
            f"Corrected hPDI MWAS expected {EXPECTED_SPECIES} rows, got {len(d)}"
        )
    if d["species"].duplicated().any():
        raise RuntimeError("Corrected hPDI MWAS has duplicate species")

    out = pd.DataFrame({
        "score": "hPDI",
        "species": d["species"].astype(str),
        "beta_diet": pd.to_numeric(d["beta"], errors="coerce"),
        "FDR_diet": pd.to_numeric(d["FDR"], errors="coerce"),
        "p_diet": pd.to_numeric(d["p"], errors="coerce"),
        "source_file": str(path),
    })
    if out[["beta_diet", "FDR_diet", "p_diet"]].isna().any().any():
        raise RuntimeError("Corrected hPDI MWAS has nonnumeric beta/FDR/p")
    return out


def build_corrected_combined_diet_tables(step08):
    old_m2, _ = step08.load_diet_model(None, model_number=2)
    old_m3, _ = step08.load_diet_model(None, model_number=3)

    hpdi_m2 = corrected_hpdi_piece(CORRECTED_HPDI_M2)
    hpdi_m3 = corrected_hpdi_piece(CORRECTED_HPDI_M3)

    combined_m2 = pd.concat(
        [old_m2.loc[~old_m2["score"].eq("hPDI")].copy(), hpdi_m2],
        ignore_index=True,
        sort=False,
    )
    combined_m3 = pd.concat(
        [old_m3.loc[~old_m3["score"].eq("hPDI")].copy(), hpdi_m3],
        ignore_index=True,
        sort=False,
    )

    combined_m2 = step08.validate_and_finalize_diet_table(
        combined_m2, "Step15b4 corrected combined Diet Model2"
    )
    combined_m3 = step08.validate_and_finalize_diet_table(
        combined_m3, "Step15b4 corrected combined Diet Model3"
    )

    combined_m2.to_csv(COMBINED_M2, index=False)
    combined_m3.to_csv(COMBINED_M3, index=False)
    return combined_m2, combined_m3


def rebuild_bridge(step08, step09, combined_m2, combined_m3):
    cgm_m2 = step08.load_cgm_model(CGM_M2, "Microbiome->CGM Model2")
    cgm_m3 = step08.load_cgm_model(CGM_M3, "Microbiome->CGM Model3")

    primary_overlap, primary_summary = step08.analyze_layer(
        combined_m2, cgm_m2, "PRIMARY_MODEL2"
    )
    bmi_overlap, bmi_summary = step08.analyze_layer(
        combined_m3, cgm_m3, "BMI_SENSITIVITY_MODEL3"
    )

    robust, robust_summary = step08.compare_primary_vs_bmi(
        primary_overlap.rename(
            columns={"score": "diet_score", "outcome_name": "cgm_outcome"}
        ),
        bmi_overlap.rename(
            columns={"score": "diet_score", "outcome_name": "cgm_outcome"}
        ),
    )

    robust_candidates = robust.loc[
        robust["robust_direction_consistent_candidate"]
    ].copy()

    primary_overlap.to_csv(
        BRIDGE_OUT / "15b4_primary_model2_overlap_species.csv", index=False
    )
    primary_summary.to_csv(
        BRIDGE_OUT / "15b4_primary_model2_overlap_summary.csv", index=False
    )
    bmi_overlap.to_csv(
        BRIDGE_OUT / "15b4_bmi_model3_overlap_species.csv", index=False
    )
    bmi_summary.to_csv(
        BRIDGE_OUT / "15b4_bmi_model3_overlap_summary.csv", index=False
    )
    robust.to_csv(
        BRIDGE_OUT / "15b4_primary_vs_bmi_overlap_robustness_species.csv",
        index=False,
    )
    robust_summary.to_csv(
        BRIDGE_OUT / "15b4_primary_vs_bmi_overlap_robustness_summary.csv",
        index=False,
    )
    robust_candidates.to_csv(ROBUST_ALL, index=False)

    # Rebuild Step09 metadata/ranking using the corrected all-score bridge.
    valid = step09.validate_input(robust_candidates)
    long_df = step09.build_long_table(valid)
    master = step09.build_species_master(long_df)
    metadata = master[
        [
            "species",
            "candidate_rank",
            "priority_tier",
            "n_robust_paths",
            "n_diet_scores",
            "n_cgm_outcomes",
        ]
    ]
    long_ranked = (
        long_df.merge(
            metadata,
            on="species",
            how="left",
            validate="many_to_one",
        )
        .sort_values(["candidate_rank", "diet_score", "cgm_outcome"])
        .reset_index(drop=True)
    )

    long_ranked.to_csv(BRIDGE_LONG, index=False)
    long_ranked.loc[long_ranked["diet_score"].eq("hPDI")].to_csv(
        BRIDGE_LONG_HPDI, index=False
    )
    master.to_csv(BRIDGE_OUT / "15b4_bridge_species_master.csv", index=False)
    step09.build_summary(long_df, master).to_csv(
        BRIDGE_OUT / "15b4_bridge_summary.csv", index=False
    )

    old_robust = pd.read_csv(OLD_ROBUST, low_memory=False)
    old_h = old_robust.loc[old_robust["diet_score"].eq("hPDI")].copy()
    new_h = robust_candidates.loc[
        robust_candidates["diet_score"].eq("hPDI")
    ].copy()

    key = ["diet_score", "cgm_outcome", "species"]
    old_set = set(map(tuple, old_h[key].to_numpy()))
    new_set = set(map(tuple, new_h[key].to_numpy()))

    bridge_metrics = {
        "old_hpdi_robust_paths": len(old_h),
        "corrected_hpdi_robust_paths": len(new_h),
        "retained_hpdi_robust_paths": len(old_set & new_set),
        "lost_hpdi_robust_paths": len(old_set - new_set),
        "gained_hpdi_robust_paths": len(new_set - old_set),
    }
    return long_ranked, bridge_metrics


def build_corrected_formal_cohort(step10):
    cohort = pd.read_csv(FORMAL_COHORT, low_memory=False)
    corr = pd.read_csv(
        CORRECTED_HPDI_SCORES,
        usecols=["participant_id", CORRECTED_HPDI_Z],
        low_memory=False,
    )

    pid_col = step10.find_participant_id(cohort, "formal Diet-CGM cohort")
    score_map = step10.identify_score_columns(cohort)
    old_hpdi_col = score_map["hPDI"]

    cohort[pid_col] = norm_id(cohort[pid_col])
    corr["participant_id"] = norm_id(corr["participant_id"])
    if cohort[pid_col].duplicated().any():
        raise RuntimeError("Formal cohort has duplicate participant IDs")
    if corr["participant_id"].duplicated().any():
        raise RuntimeError("Corrected hPDI score has duplicate participant IDs")

    corr = corr.rename(columns={CORRECTED_HPDI_Z: "_corrected_hPDI_z"})
    merged = cohort.merge(
        corr,
        left_on=pid_col,
        right_on="participant_id",
        how="left",
        validate="one_to_one",
    )
    missing = int(merged["_corrected_hPDI_z"].isna().sum())
    if missing:
        raise RuntimeError(
            f"Corrected hPDI missing for {missing} formal cohort participants"
        )

    merged[old_hpdi_col] = pd.to_numeric(
        merged["_corrected_hPDI_z"], errors="coerce"
    )
    merged = merged.drop(columns=["_corrected_hPDI_z"])
    if "participant_id_y" in merged.columns:
        merged = merged.drop(columns=["participant_id_y"])
    if "participant_id_x" in merged.columns:
        merged = merged.rename(columns={"participant_id_x": pid_col})

    merged.to_csv(CORRECTED_COHORT, index=False)
    return old_hpdi_col


def run_isolated_hpdi_mediation(step10):
    old_argv = sys.argv[:]
    old_out_models = step10.OUT_MODELS
    old_out_reports = step10.OUT_REPORTS
    try:
        step10.OUT_MODELS = MED_MODELS
        step10.OUT_REPORTS = MED_REPORTS
        sys.argv = [
            str(STEP10_PATH),
            "--cohort", str(CORRECTED_COHORT),
            "--diet-cgm-model3", str(CORRECTED_CGM_MODEL3),
            "--bridges", str(BRIDGE_LONG_HPDI),
            "--microbiome", str(MICROBIOME),
            "--covariates", str(COVARIATES),
            "--cohort-mode", "primary",
            "--bootstrap", "1000",
            "--seed", "20260904",
        ]
        step10.main()
    finally:
        sys.argv = old_argv
        step10.OUT_MODELS = old_out_models
        step10.OUT_REPORTS = old_out_reports

    out = MED_MODELS / "10_mediation_paths_model3.csv"
    require(out, "isolated corrected hPDI mediation output")
    return out


def compare_mediation(corrected_path: Path):
    old = pd.read_csv(OLD_MEDIATION, low_memory=False)
    new = pd.read_csv(corrected_path, low_memory=False)

    old_h = old.loc[
        old["diet_score"].eq("hPDI")
        & old["cgm_outcome"].eq("glucose_cv")
    ].copy()
    new_h = new.loc[
        new["diet_score"].eq("hPDI")
        & new["cgm_outcome"].eq("glucose_cv")
    ].copy()

    required = {
        "species",
        "indirect_effect",
        "mediation_FDR05_primary",
        "candidate_mediator_consistent",
    }
    for label, d in [("old", old_h), ("corrected", new_h)]:
        missing = sorted(required - set(d.columns))
        if missing:
            raise RuntimeError(f"{label} mediation missing columns: {missing}")
        if d["species"].duplicated().any():
            raise RuntimeError(f"{label} hPDI mediation has duplicate species")

    merged = old_h.merge(
        new_h,
        on="species",
        how="outer",
        suffixes=("_old", "_corrected"),
        indicator=True,
        validate="one_to_one",
    )

    old_sig = truthy(merged["mediation_FDR05_primary_old"])
    new_sig = truthy(merged["mediation_FDR05_primary_corrected"])
    old_con = truthy(merged["candidate_mediator_consistent_old"])
    new_con = truthy(merged["candidate_mediator_consistent_corrected"])

    both = merged["_merge"].eq("both")
    old_ind = pd.to_numeric(merged["indirect_effect_old"], errors="coerce")
    new_ind = pd.to_numeric(merged["indirect_effect_corrected"], errors="coerce")
    sign_flip = (
        both
        & old_ind.notna()
        & new_ind.notna()
        & np.sign(old_ind).ne(np.sign(new_ind))
    )

    merged["old_sig"] = old_sig
    merged["corrected_sig"] = new_sig
    merged["old_consistent"] = old_con
    merged["corrected_consistent"] = new_con
    merged["indirect_sign_flip"] = sign_flip
    merged.to_csv(PATH_COMPARE, index=False)

    # Recurrent core: canonical AMED consistent glucose_cv species vs corrected hPDI.
    old_amed = old.loc[
        old["diet_score"].eq("AMED")
        & old["cgm_outcome"].eq("glucose_cv")
        & truthy(old["candidate_mediator_consistent"])
    ].copy()
    amed_set = set(old_amed["species"].astype(str))
    old_hpdi_cons_set = set(
        old_h.loc[
            truthy(old_h["candidate_mediator_consistent"]),
            "species",
        ].astype(str)
    )
    new_hpdi_cons_set = set(
        new_h.loc[
            truthy(new_h["candidate_mediator_consistent"]),
            "species",
        ].astype(str)
    )

    old_core = amed_set & old_hpdi_cons_set
    corrected_core = amed_set & new_hpdi_cons_set

    metrics = {
        "old_tested_paths": len(old_h),
        "corrected_tested_paths": len(new_h),
        "exact_tested_species_set_match": (
            set(old_h["species"].astype(str))
            == set(new_h["species"].astype(str))
        ),
        "old_significant": int(truthy(old_h["mediation_FDR05_primary"]).sum()),
        "corrected_significant": int(truthy(new_h["mediation_FDR05_primary"]).sum()),
        "old_consistent": int(truthy(old_h["candidate_mediator_consistent"]).sum()),
        "corrected_consistent": int(truthy(new_h["candidate_mediator_consistent"]).sum()),
        "old_significant_retained": int((old_sig & new_sig).sum()),
        "old_consistent_retained": int((old_con & new_con).sum()),
        "indirect_rho_matched_paths": spearman(
            merged.loc[both, "indirect_effect_old"],
            merged.loc[both, "indirect_effect_corrected"],
        ),
        "indirect_sign_flips_matched_paths": int(sign_flip.sum()),
        "old_amed_hpdi_core_n": len(old_core),
        "corrected_amed_hpdi_core_n": len(corrected_core),
        "old_core_retained_n": len(old_core & corrected_core),
        "old_core_lost_n": len(old_core - corrected_core),
        "new_core_gained_n": len(corrected_core - old_core),
    }
    return metrics, old_core, corrected_core


def main() -> int:
    for path, label in [
        (STEP08_PATH, "Step08 canonical script"),
        (STEP09_PATH, "Step09 canonical script"),
        (STEP10_PATH, "Step10 canonical script"),
        (CORRECTED_HPDI_M2, "corrected hPDI MWAS Model2"),
        (CORRECTED_HPDI_M3, "corrected hPDI MWAS Model3"),
        (CORRECTED_CGM_MODEL3, "corrected hPDI combined CGM Model3"),
        (CORRECTED_HPDI_SCORES, "corrected hPDI scores"),
        (FORMAL_COHORT, "formal Diet-CGM cohort"),
        (MICROBIOME, "microbiome CLR-Z"),
        (COVARIATES, "covariate master"),
        (CGM_M2, "Microbiome-CGM Model2"),
        (CGM_M3, "Microbiome-CGM Model3"),
        (OLD_ROBUST, "canonical old robust bridge candidates"),
        (OLD_MEDIATION, "canonical old primary mediation"),
    ]:
        require(path, label)

    for d in [OUT, BRIDGE_OUT, MED_OUT, MED_MODELS, MED_REPORTS]:
        d.mkdir(parents=True, exist_ok=True)

    print("=== STEP 15b4 CORRECTED hPDI BRIDGE + MEDIATION ===")
    print("CANONICAL_FILES_MODIFIED=False")
    print("OTHER_DIET_SCORES_REFIT=False")
    print("MICROBIOME_CGM_REFIT=False")

    step08 = load_module(STEP08_PATH, "step15b4_step08")
    step09 = load_module(STEP09_PATH, "step15b4_step09")
    step10 = load_module(STEP10_PATH, "step15b4_step10")

    combined_m2, combined_m3 = build_corrected_combined_diet_tables(step08)
    _, bridge_metrics = rebuild_bridge(
        step08, step09, combined_m2, combined_m3
    )

    hpdi_bridge = pd.read_csv(BRIDGE_LONG_HPDI, low_memory=False)
    if hpdi_bridge.empty:
        raise RuntimeError("No corrected hPDI robust bridge paths remain")

    old_hpdi_col = build_corrected_formal_cohort(step10)
    corrected_med_path = run_isolated_hpdi_mediation(step10)
    med_metrics, old_core, corrected_core = compare_mediation(corrected_med_path)

    lines = [
        "=== STEP 15b4 CORRECTED hPDI BRIDGE + MEDIATION SUMMARY ===",
        "CANONICAL_FILES_MODIFIED=False",
        "OTHER_DIET_SCORES_REFIT=False",
        "MICROBIOME_CGM_REFIT=False",
        "MEDIATION_BOOTSTRAP=1000",
        "",
        f"CORRECTED_COHORT_HPDI_SOURCE_COLUMN_REPLACED={old_hpdi_col}",
        "",
        "--- BRIDGE STABILITY ---",
    ]
    for k, v in bridge_metrics.items():
        lines.append(f"{k.upper()}={v}")

    lines += [
        "",
        "--- MEDIATION STABILITY ---",
    ]
    for k, v in med_metrics.items():
        if isinstance(v, float):
            lines.append(f"{k.upper()}={v:.9f}")
        else:
            lines.append(f"{k.upper()}={v}")

    lines += [
        "",
        "OLD_AMED_HPDI_CORE_SPECIES:",
        *[f"- {x}" for x in sorted(old_core)],
        "",
        "CORRECTED_AMED_HPDI_CORE_SPECIES:",
        *[f"- {x}" for x in sorted(corrected_core)],
        "",
        "DECISION_RULE:",
        "- If corrected hPDI retains the glucose_cv mediation family, effect",
        "  signs/ranks are highly stable, and the recurrent AMED/hPDI core is",
        "  largely retained, promote corrected hPDI to the canonical definition.",
        "- If material path/core changes occur, inspect the changed paths before",
        "  replacing the old hPDI branch.",
        "",
        f"PATH_COMPARISON={PATH_COMPARE}",
        f"ISOLATED_MEDIATION={corrected_med_path}",
    ]

    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
