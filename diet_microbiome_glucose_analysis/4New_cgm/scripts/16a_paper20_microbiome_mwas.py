#!/usr/bin/env python3
"""
Step 16a — Paper-20 microbiome -> CGM MWAS + exact-species robustness.

Purpose
-------
Extend the existing project-canonical microbiome association pipeline from
the previous CGM outcomes to all 20 frozen Paper-20 CGM phenotypes.

This script PRESERVES the previous MWAS logic:
    Primary MWAS set = Primary Model2
    Model3 = BMI-adjusted robustness
    Strict Model2/3 = diabetes/A10 sensitivity
    Same-M3-sample Model2 = isolates BMI adjustment from sample selection

Planned model sets per outcome:
    primary_m2   : primary cohort, Model2 sample, Model2 fit
    same_m3_m2   : primary cohort, Model3-complete sample, Model2 fit
    primary_m3   : primary cohort, Model3-complete sample, Model3 fit
    strict_m2    : strict cohort, Model2 sample, Model2 fit
    strict_m3    : strict cohort, Model3-complete sample, Model3 fit

20 outcomes x 5 model sets x 379 species = 37,900 species-level rows.

Multiplicity
------------
Primary family FDR:
    BH across 379 species WITHIN each CGM outcome and each model set.

Global FDR:
    BH across 20 x 379 = 7,580 species-outcome tests within a model set,
    reported as sensitivity only.

Highest-robustness species:
    significant (within-outcome FDR<0.05) in
        primary_m2 + primary_m3 + strict_m2 + strict_m3
    AND same beta sign in all four.

Important
---------
- No Diet->CGM model is refit.
- No Diet->microbiome model is refit.
- No bridge or mediation model is run.
- TAR180 is still analyzed as a microbiome association phenotype for
  completeness, but Step15b gives it no primary Diet->CGM bridge gate.
- TAR140/HBGI/GRADE remain distribution-sensitive; canonical MWAS here is
  followed by a separate prespecified outcome-sensitivity audit before those
  outcomes are used in a final bridge claim.
- Mean glucose and GMI are both retained but are not independent replication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
DG = ROOT / "diet_microbiome_glucose_analysis"
CGM_PACKAGE = ROOT / "cgm_deal" / "cgm论文新增17指标"

BRANCH_CANDIDATES = [
    DG / "4New_cgm",
    ROOT / "4New_cgm",
]

EXPECTED_SPECIES = 379

SET_SPECS = {
    "primary_m2": ("primary", 2, 2),
    "same_m3_m2": ("primary", 2, 3),
    "primary_m3": ("primary", 3, 3),
    "strict_m2": ("strict", 2, 2),
    "strict_m3": ("strict", 3, 3),
}

SPECIAL_OUTCOMES = {
    "cgm_above_140": "distribution_sensitive_mwas_sensitivity_required",
    "cgm_above_180": "distribution_sensitive_no_primary_diet_bridge_gate",
    "cgm_hbgi": "distribution_sensitive_mwas_sensitivity_required",
    "cgm_grade": "distribution_sensitive_mwas_sensitivity_required",
}


def require(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def truthy(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).eq(1)
    return (
        s.astype(str).str.strip().str.lower()
        .isin({"true", "1", "1.0", "yes", "y", "t"})
    )


def find_branch(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(p)
        return p
    for p in BRANCH_CANDIDATES:
        if p.is_dir():
            return p
    raise FileNotFoundError("4New_cgm branch not found")


def latest_final_cgm(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        require(p, "finalized Paper-20 CGM")
        return p
    xs = []
    for p in (CGM_PACKAGE / "outputs").glob(
        "run_*/data/07_cgm_paper_extended_phenotypes.csv"
    ):
        if p.is_file():
            xs.append((p.stat().st_mtime, p))
    if not xs:
        raise FileNotFoundError("No finalized Paper-20 CGM table found")
    xs.sort(reverse=True)
    return xs[0][1]


def latest_registry(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        require(p, "Paper-20 registry")
        return p
    xs = []
    for p in (CGM_PACKAGE / "outputs").glob(
        "paper20_analysis_registry_*/01_paper20_outcome_analysis_registry.csv"
    ):
        if p.is_file():
            xs.append((p.stat().st_mtime, p))
    if not xs:
        raise FileNotFoundError("No frozen Paper-20 registry found")
    xs.sort(reverse=True)
    return xs[0][1]


def safe_spearman(a: pd.Series, b: pd.Series):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    m = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 3:
        return np.nan, np.nan
    rho, p = spearmanr(a.loc[m], b.loc[m])
    return float(rho), float(p)


def add_fdr(results: pd.DataFrame, bh_fdr) -> pd.DataFrame:
    x = results.copy()
    x["FDR_family"] = np.nan
    x["FDR_global"] = np.nan

    # Within-outcome family: 379 species.
    for (_, outcome), idx in x.groupby(
        ["set_name", "outcome_field"], sort=False
    ).groups.items():
        idx = list(idx)
        part = x.loc[idx]
        if len(part) != EXPECTED_SPECIES:
            raise RuntimeError(
                f"{outcome}: expected {EXPECTED_SPECIES} species, got {len(part)}"
            )
        good = part["status"].eq("computed") & part["p_value"].notna()
        p = part["p_value"].where(good, 1.0).to_numpy(float)
        q = bh_fdr(p)
        x.loc[idx, "FDR_family"] = np.where(good, q, np.nan)
        x.loc[idx, "family_tests_planned"] = EXPECTED_SPECIES

    # Global 20 x 379 sensitivity family within each set.
    expected_global = 20 * EXPECTED_SPECIES
    for set_name, idx in x.groupby("set_name", sort=False).groups.items():
        idx = list(idx)
        part = x.loc[idx]
        if len(part) != expected_global:
            raise RuntimeError(
                f"{set_name}: expected global {expected_global}, got {len(part)}"
            )
        good = part["status"].eq("computed") & part["p_value"].notna()
        p = part["p_value"].where(good, 1.0).to_numpy(float)
        q = bh_fdr(p)
        x.loc[idx, "FDR_global"] = np.where(good, q, np.nan)
        x.loc[idx, "global_tests_planned"] = expected_global

    x["significant_family_05"] = x["FDR_family"].lt(0.05)
    x["significant_global_05"] = x["FDR_global"].lt(0.05)
    return x


def merged_robustness(results: pd.DataFrame, outcome: str) -> pd.DataFrame:
    pieces = {}
    cols = [
        "species", "N", "sample_sha256", "beta", "SE",
        "CI95_lower", "CI95_upper", "p_value",
        "FDR_family", "FDR_global", "status",
        "significant_family_05",
    ]

    for set_name in SET_SPECS:
        z = results.loc[
            results["set_name"].eq(set_name)
            & results["outcome_field"].eq(outcome),
            cols,
        ].copy()
        if len(z) != EXPECTED_SPECIES:
            raise RuntimeError(
                f"{outcome}/{set_name}: {len(z)} rows, expected {EXPECTED_SPECIES}"
            )
        if z["species"].duplicated().any():
            raise RuntimeError(f"{outcome}/{set_name}: duplicate species")
        z = z.rename(columns={
            c: f"{c}_{set_name}"
            for c in z.columns if c != "species"
        })
        pieces[set_name] = z

    m = pieces["primary_m2"]
    for set_name in ["same_m3_m2", "primary_m3", "strict_m2", "strict_m3"]:
        m = m.merge(
            pieces[set_name],
            on="species",
            how="inner",
            validate="one_to_one",
        )

    if len(m) != EXPECTED_SPECIES:
        raise RuntimeError(f"{outcome}: robustness merge not exact")

    for set_name in SET_SPECS:
        m[f"sig_{set_name}"] = truthy(
            m[f"significant_family_05_{set_name}"]
        )

    def same_sign(a, b):
        aa = pd.to_numeric(m[a], errors="coerce")
        bb = pd.to_numeric(m[b], errors="coerce")
        return (
            aa.notna() & bb.notna()
            & np.isfinite(aa) & np.isfinite(bb)
            & np.sign(aa).eq(np.sign(bb))
        )

    m["same_sign_primary_m2_primary_m3"] = same_sign(
        "beta_primary_m2", "beta_primary_m3"
    )
    m["same_sign_primary_m2_strict_m2"] = same_sign(
        "beta_primary_m2", "beta_strict_m2"
    )
    m["same_sign_primary_m2_strict_m3"] = same_sign(
        "beta_primary_m2", "beta_strict_m3"
    )

    m["highest_robustness"] = (
        m["sig_primary_m2"]
        & m["sig_primary_m3"]
        & m["sig_strict_m2"]
        & m["sig_strict_m3"]
        & m["same_sign_primary_m2_primary_m3"]
        & m["same_sign_primary_m2_strict_m2"]
        & m["same_sign_primary_m2_strict_m3"]
    )
    m.insert(0, "outcome_field", outcome)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch-dir", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--cgm-csv", default=None)
    ap.add_argument("--registry", default=None)
    args = ap.parse_args()

    branch = find_branch(args.branch_dir)
    scripts = branch / "scripts"
    sys.path.insert(0, str(scripts))

    import new_cgm_data as data
    import new_cgm_models as models
    import legacy_microbiome_models as micro_legacy
    import legacy_diet_models as diet_legacy

    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else branch / "config" / "analysis.json"
    )
    require(config_path, "4New_cgm config")

    final_cgm = latest_final_cgm(args.cgm_csv)
    registry_path = latest_registry(args.registry)

    registry = pd.read_csv(registry_path, low_memory=False)
    paper = registry.loc[
        registry["outcome_family"].eq("paper20_primary")
    ].copy()
    if len(paper) != 20:
        raise RuntimeError(f"Expected Paper-20=20, found {len(paper)}")

    outcome_fields = paper["field"].astype(str).tolist()
    outcome_labels = dict(zip(paper["field"], paper["label"]))

    config = data.load_config(config_path)
    _, microbiome, species, _, overlap, _ = data.prepare_tables(config)

    if len(species) != EXPECTED_SPECIES:
        raise RuntimeError(
            f"Expected {EXPECTED_SPECIES} species, found {len(species)}"
        )
    if len(set(species)) != EXPECTED_SPECIES:
        raise RuntimeError("Species list is not unique")

    # Merge finalized Paper-20 Z columns; audit pre-existing overlaps.
    cgm = data.read_table(final_cgm)
    required = ["participant_id"] + [f + "_z" for f in outcome_fields]
    data.require_columns(cgm, required, "finalized Paper-20 CGM")
    cgm_piece = cgm[required].copy()

    for col in required[1:]:
        cgm_piece[col] = data.numeric(cgm_piece[col], col)

    overlap_cols = sorted(
        (set(microbiome.columns) & set(cgm_piece.columns))
        - {"participant_id"}
    )
    print("=== STEP 16a PAPER-20 MICROBIOME->CGM MWAS ===")
    print(f"FINAL_CGM={final_cgm}")
    print(f"REGISTRY={registry_path}")
    print(f"MICROBIOME_UNION_ROWS={len(microbiome)}")
    print(f"SPECIES={len(species)}")
    print("OUTCOMES=20")
    print("MODEL_SETS=5")
    print(f"PLANNED_SPECIES_ROWS={20*5*EXPECTED_SPECIES}")
    print("DIET_MODELS_REFIT=False")
    print("BRIDGE_SCREEN_RUN=False")
    print(
        "PAPER20_EXISTING_COLUMN_OVERLAPS="
        + ",".join(overlap_cols)
    )

    if overlap_cols:
        audit = microbiome[["participant_id"] + overlap_cols].merge(
            cgm_piece[["participant_id"] + overlap_cols],
            on="participant_id",
            how="inner",
            suffixes=("_legacy", "_paper20"),
            validate="one_to_one",
        )
        print("--- OVERLAP VALUE AUDIT ---")
        for col in overlap_cols:
            a = pd.to_numeric(audit[col + "_legacy"], errors="coerce")
            b = pd.to_numeric(audit[col + "_paper20"], errors="coerce")
            both = a.notna() & b.notna()
            mismatch = int((a.notna() != b.notna()).sum())
            changed = int(((a - b).abs().gt(1e-12) & both).sum())
            maxdiff = (
                float((a.loc[both] - b.loc[both]).abs().max())
                if both.any() else np.nan
            )
            print(
                f"{col}: both_N={int(both.sum())} "
                f"availability_mismatch_N={mismatch} "
                f"value_changed_N={changed} "
                f"max_abs_diff={maxdiff}"
            )
        microbiome = microbiome.drop(columns=overlap_cols)

    before = len(microbiome)
    microbiome = microbiome.merge(
        cgm_piece,
        on="participant_id",
        how="left",
        validate="one_to_one",
    )
    if len(microbiome) != before:
        raise RuntimeError("Paper-20 merge changed microbiome row count")

    missing_after = [
        f + "_z" for f in outcome_fields
        if f + "_z" not in microbiome.columns
    ]
    if missing_after:
        raise RuntimeError(
            "Paper-20 columns missing after merge: " + ", ".join(missing_after)
        )

    rows = []
    design_rows = []

    for set_name, (cohort_mode, fit_model, selection_model) in SET_SPECS.items():
        for field in outcome_fields:
            zcol = field + "_z"

            required_cols = [zcol]
            if selection_model >= 2:
                required_cols += data.BASE_COVARIATES
            if selection_model == 3:
                required_cols += ["bmi"]

            eligible = (
                microbiome["strict_eligible"]
                if cohort_mode == "strict"
                else microbiome["primary_eligible"]
            )
            selected = microbiome.loc[
                eligible & models.complete_mask(microbiome, required_cols)
            ].copy()

            N = len(selected)
            sample_sha = data.sample_hash(selected["participant_id"])

            group = []
            base = {
                "set_name": set_name,
                "analysis_set": cohort_mode,
                "model": fit_model,
                "selection_model": selection_model,
                "outcome_field": field,
                "outcome_label": outcome_labels[field],
                "outcome_role": SPECIAL_OUTCOMES.get(
                    field, "standard_inferential"
                ),
                "N": N,
                "sample_sha256": sample_sha,
                "beta": np.nan,
                "SE": np.nan,
                "CI95_lower": np.nan,
                "CI95_upper": np.nan,
                "p_value": np.nan,
                "partial_R2": np.nan,
                "df_resid": np.nan,
                "status": "not_fitted",
                "error": "",
            }
            group = [dict(base, species=s) for s in species]

            try:
                if N < config["minimum_model_n"]:
                    raise RuntimeError(
                        f"Insufficient N={N} < {config['minimum_model_n']}"
                    )
                y = selected[zcol].to_numpy(float)
                if not np.isfinite(y).all() or np.ptp(y) == 0:
                    raise RuntimeError("Invalid/constant outcome")

                C, names, rank, condition, reference = (
                    micro_legacy.build_covariate_design(
                        selected,
                        include_bmi=(fit_model == 3),
                    )
                )

                for j, name in enumerate(names):
                    design_rows.append({
                        "set_name": set_name,
                        "analysis_set": cohort_mode,
                        "model": fit_model,
                        "selection_model": selection_model,
                        "outcome_field": field,
                        "term": name,
                        "N": N,
                        "rank": rank,
                        "condition": condition,
                        "device_reference": reference,
                        "encoded_mean": float(C[:, j].mean()),
                        "encoded_sd_ddof0": float(C[:, j].std()),
                        "sample_sha256": sample_sha,
                    })

                X = selected[species].to_numpy(float)
                residual = micro_legacy.residualize_against_covariates(C, X)
                estimable = np.sum(residual ** 2, axis=0) > 1e-12

                for i in np.flatnonzero(~estimable):
                    group[i].update(
                        status="failed",
                        error="Species has no residual variance after adjustment",
                    )

                if estimable.any():
                    fit = micro_legacy.fit_all_species_adjusted(
                        X[:, estimable], y, C
                    )
                    for j, i in enumerate(np.flatnonzero(estimable)):
                        if (
                            not np.isfinite(fit["p_value"][j])
                            or not np.isfinite(fit["SE"][j])
                            or fit["SE"][j] <= 0
                        ):
                            group[i].update(
                                status="failed",
                                error="Nonfinite/degenerate inference",
                            )
                            continue
                        group[i].update(
                            beta=float(fit["beta"][j]),
                            SE=float(fit["SE"][j]),
                            CI95_lower=float(fit["CI95_lower"][j]),
                            CI95_upper=float(fit["CI95_upper"][j]),
                            p_value=float(fit["p_value"][j]),
                            partial_R2=float(fit["partial_R2"][j]),
                            df_resid=int(fit["df_resid"]),
                            status="computed",
                        )

            except Exception as e:
                for row in group:
                    if row["status"] == "not_fitted":
                        row.update(status="failed", error=repr(e))

            rows.extend(group)
            computed = sum(r["status"] == "computed" for r in group)
            print(
                f"MWAS {set_name:11s} {field:20s} "
                f"N={N:4d} species={len(species)} computed={computed}"
            )

    results = pd.DataFrame(rows)
    expected = 20 * len(SET_SPECS) * EXPECTED_SPECIES
    if len(results) != expected:
        raise RuntimeError(f"Expected {expected} rows, got {len(results)}")

    results = add_fdr(results, diet_legacy.bh_fdr)

    # Exact-species robustness consolidation.
    long_frames = []
    summary_rows = []

    for field in outcome_fields:
        m = merged_robustness(results, field)
        m.insert(1, "outcome_label", outcome_labels[field])
        m.insert(
            2,
            "outcome_role",
            SPECIAL_OUTCOMES.get(field, "standard_inferential"),
        )
        long_frames.append(m)

        pm2 = truthy(m["sig_primary_m2"])
        pm3 = truthy(m["sig_primary_m3"])
        sm2 = truthy(m["sig_strict_m2"])
        sm3 = truthy(m["sig_strict_m3"])

        rho_m2_m3, rho_m2_m3_p = safe_spearman(
            m["beta_primary_m2"], m["beta_primary_m3"]
        )
        rho_pm2_sm2, rho_pm2_sm2_p = safe_spearman(
            m["beta_primary_m2"], m["beta_strict_m2"]
        )
        rho_pm3_sm3, rho_pm3_sm3_p = safe_spearman(
            m["beta_primary_m3"], m["beta_strict_m3"]
        )

        summary_rows.append({
            "outcome_field": field,
            "outcome_label": outcome_labels[field],
            "outcome_role": SPECIAL_OUTCOMES.get(
                field, "standard_inferential"
            ),
            "species_tested": EXPECTED_SPECIES,
            "primary_m2_significant": int(pm2.sum()),
            "primary_m3_significant": int(pm3.sum()),
            "strict_m2_significant": int(sm2.sum()),
            "strict_m3_significant": int(sm3.sum()),
            "primary_m2_retained_primary_m3": int((pm2 & pm3).sum()),
            "primary_m2_lost_primary_m3": int((pm2 & ~pm3).sum()),
            "primary_m2_retained_strict_m2": int((pm2 & sm2).sum()),
            "primary_m2_lost_strict_m2": int((pm2 & ~sm2).sum()),
            "highest_robustness_n": int(m["highest_robustness"].sum()),
            "beta_rho_primary_m2_vs_m3": rho_m2_m3,
            "beta_rho_primary_m2_vs_m3_p": rho_m2_m3_p,
            "beta_rho_primary_m2_vs_strict_m2": rho_pm2_sm2,
            "beta_rho_primary_m2_vs_strict_m2_p": rho_pm2_sm2_p,
            "beta_rho_primary_m3_vs_strict_m3": rho_pm3_sm3,
            "beta_rho_primary_m3_vs_strict_m3_p": rho_pm3_sm3_p,
        })

    robustness = pd.concat(long_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["primary_m2_significant", "highest_robustness_n", "outcome_field"],
        ascending=[False, False, True],
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = branch / "outputs" / f"paper20_microbiome_mwas_{stamp}"
    models_dir = out / "models"
    reports = out / "reports"
    data_dir = out / "data"
    models_dir.mkdir(parents=True, exist_ok=False)
    reports.mkdir()
    data_dir.mkdir()

    model_path = models_dir / "16_paper20_microbiome_all_models.csv"
    design_path = reports / "16_microbiome_design_parameters.csv"
    robustness_path = reports / "16_exact_species_robustness_long.csv"
    summary_path = reports / "16_outcome_robustness_summary.csv"
    highest_path = reports / "16_highest_robustness_species.csv"
    primary_path = reports / "16_primary_m2_significant_species.csv"
    txt_path = reports / "16_paper20_microbiome_mwas_summary.txt"

    results.to_csv(model_path, index=False)
    pd.DataFrame(design_rows).to_csv(design_path, index=False)
    robustness.to_csv(robustness_path, index=False)
    summary.to_csv(summary_path, index=False)
    robustness.loc[robustness["highest_robustness"]].to_csv(
        highest_path, index=False
    )
    robustness.loc[truthy(robustness["sig_primary_m2"])].to_csv(
        primary_path, index=False
    )

    computed = int(results["status"].eq("computed").sum())
    failed = int(results["status"].ne("computed").sum())
    primary_total = int(
        results.loc[
            results["set_name"].eq("primary_m2"),
            "significant_family_05",
        ].sum()
    )
    highest_total = int(robustness["highest_robustness"].sum())

    lines = [
        "=== STEP 16a PAPER-20 MICROBIOME->CGM MWAS SUMMARY ===",
        f"FINAL_CGM={final_cgm}",
        f"REGISTRY={registry_path}",
        f"MICROBIOME_ROWS={len(microbiome)}",
        f"SPECIES={len(species)}",
        "OUTCOMES=20",
        "MODEL_SETS=5",
        f"SPECIES_MODEL_ROWS={len(results)}",
        f"COMPUTED={computed}",
        f"FAILED={failed}",
        "PRIMARY_MWAS_SET=primary_m2",
        "PRIMARY_FDR=BH within each outcome across 379 species",
        "GLOBAL_7580_FDR=sensitivity_only",
        f"PRIMARY_M2_SIGNIFICANT_TOTAL={primary_total}",
        f"HIGHEST_ROBUSTNESS_TOTAL={highest_total}",
        "",
        "--- OUTCOME ROBUSTNESS ---",
        summary.to_string(index=False),
        "",
        "INTERPRETATION RULES:",
        "- Primary MWAS findings remain Primary Model2 within-outcome FDR<0.05.",
        "- Model3/strict/same-sample results are robustness/sensitivity, not a redefinition of discovery.",
        "- Highest-robustness is an annotation subset only.",
        "- TAR180 has no primary Diet->CGM bridge gate from Step15b.",
        "- TAR140/HBGI/GRADE require their prespecified microbiome outcome-sensitivity audit before final bridge interpretation.",
        "- Mean glucose and GMI are highly dependent and must not be described as independent replication.",
        "- No causal interpretation.",
        "",
        f"MODEL_TABLE={model_path}",
        f"ROBUSTNESS={robustness_path}",
        f"SUMMARY={summary_path}",
        f"HIGHEST={highest_path}",
        f"OUTPUT_DIR={out}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    (reports / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed" if failed == 0 else "completed_with_failures",
                "final_cgm": str(final_cgm),
                "final_cgm_sha256": sha256(final_cgm),
                "registry": str(registry_path),
                "registry_sha256": sha256(registry_path),
                "microbiome_rows": int(len(microbiome)),
                "species": int(len(species)),
                "outcomes": 20,
                "model_sets": list(SET_SPECS),
                "species_model_rows": int(len(results)),
                "computed": computed,
                "failed": failed,
                "primary_m2_significant_total": primary_total,
                "highest_robustness_total": highest_total,
                "diet_models_refit": False,
                "bridge_screen_run": False,
                "outputs": {
                    "model_table": str(model_path),
                    "robustness": str(robustness_path),
                    "outcome_summary": str(summary_path),
                    "highest": str(highest_path),
                    "summary": str(txt_path),
                },
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
