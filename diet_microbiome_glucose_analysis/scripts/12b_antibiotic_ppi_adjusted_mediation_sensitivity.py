#!/usr/bin/env python3
"""Step 12b — antibiotic/PPI-adjusted mediation sensitivity.

This wrapper REUSES the canonical Step-10 mediation code unchanged:
    10_diet_microbiome_cgm_mediation.py

It only changes:
    1) covariate input -> 02_covariate_master_antibiotic_ppi_sensitivity.csv
    2) output location -> temporary isolated directories, so canonical Step-10
       results can never be overwritten
    3) after the run, copies sensitivity results to 12b_* filenames
    4) compares the new run directly with canonical primary mediation

No mediation model is redesigned.

Primary comparison targets:
    - exact 31-path set
    - significant path counts
    - direction-consistent path counts
    - retention of primary significant/consistent paths
    - a / b / indirect / total effect stability
    - sign flips
    - AMED/hPDI glucose-CV recurrent core retention
    - complete-case N change

Default:
    1000 bootstrap iterations, primary diabetes/A10 cohort.

Run from:
/home/ec2-user/Desktop/HPP/Data

    python diet_microbiome_glucose_analysis/scripts/12b_antibiotic_ppi_adjusted_mediation_sensitivity.py
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
ANALYSIS_DIR = ROOT / "diet_microbiome_glucose_analysis"
SCRIPT_DIR = ANALYSIS_DIR / "scripts"

CANONICAL_STEP10 = SCRIPT_DIR / "10_diet_microbiome_cgm_mediation.py"
SENS_MASTER = (
    ROOT
    / "co-variant"
    / "outputs"
    / "data"
    / "02_covariate_master_antibiotic_ppi_sensitivity.csv"
)

CANONICAL_PRIMARY = (
    ANALYSIS_DIR
    / "outputs"
    / "models"
    / "10_mediation_paths_model3.csv"
)

MODEL_DIR = ANALYSIS_DIR / "outputs" / "models"
REPORT_DIR = ANALYSIS_DIR / "outputs" / "reports"

TMP_MODEL_DIR = MODEL_DIR / "_12b_antibiotic_ppi_tmp"
TMP_REPORT_DIR = REPORT_DIR / "_12b_antibiotic_ppi_tmp"

OUT_MODEL = MODEL_DIR / "12b_antibiotic_ppi_mediation_paths_model3.csv"
OUT_SUMMARY = REPORT_DIR / "12b_antibiotic_ppi_comparison_summary.csv"
OUT_PATH_COMPARE = REPORT_DIR / "12b_antibiotic_ppi_path_comparison.csv"
OUT_CORE_COMPARE = REPORT_DIR / "12b_antibiotic_ppi_core_comparison.csv"
OUT_COV_AUDIT = REPORT_DIR / "12b_antibiotic_ppi_covariate_audit.csv"
OUT_TXT = REPORT_DIR / "12b_antibiotic_ppi_comparison_summary.txt"

KEY = ["diet_score", "cgm_outcome", "species"]


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "1.0", "yes", "y", "t"})
    )


def sign_concordance(a: pd.Series, b: pd.Series) -> tuple[int, int, float]:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    ok = x.notna() & y.notna()
    if not ok.any():
        return 0, 0, np.nan
    same = np.sign(x[ok].to_numpy()) == np.sign(y[ok].to_numpy())
    return int(same.sum()), int((~same).sum()), float(same.mean())


def spearman(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    ok = x.notna() & y.notna()
    if ok.sum() < 3:
        return np.nan
    return float(x[ok].rank().corr(y[ok].rank(), method="pearson"))


def load_step10_module(path: Path):
    spec = importlib.util.spec_from_file_location("hpp_step10_sensitivity", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import canonical Step10: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_canonical_step10_with_sensitivity_master(bootstrap: int, seed: int):
    if not CANONICAL_STEP10.is_file():
        raise FileNotFoundError(CANONICAL_STEP10)
    if not SENS_MASTER.is_file():
        raise FileNotFoundError(
            f"Sensitivity master missing: {SENS_MASTER}\n"
            "Run Step 12a first."
        )

    # Dedicated directories guarantee canonical files cannot be overwritten.
    for d in (TMP_MODEL_DIR, TMP_REPORT_DIR):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    mod = load_step10_module(CANONICAL_STEP10)
    mod.OUT_MODELS = TMP_MODEL_DIR
    mod.OUT_REPORTS = TMP_REPORT_DIR

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            str(CANONICAL_STEP10),
            "--covariates",
            str(SENS_MASTER),
            "--cohort-mode",
            "primary",
            "--bootstrap",
            str(bootstrap),
            "--seed",
            str(seed),
        ]
        mod.main()
    finally:
        sys.argv = old_argv

    generated_model = TMP_MODEL_DIR / "10_mediation_paths_model3.csv"
    generated_cov_audit = TMP_REPORT_DIR / "10_mediation_covariate_audit.csv"

    if not generated_model.is_file():
        raise RuntimeError(
            "Canonical Step10 completed without expected sensitivity model output."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy2(generated_model, OUT_MODEL)
    if generated_cov_audit.is_file():
        shutil.copy2(generated_cov_audit, OUT_COV_AUDIT)

    # Preserve all auxiliary Step10 sensitivity reports under 12b_* names.
    for src in TMP_REPORT_DIR.glob("10_mediation_*"):
        suffix = src.name.removeprefix("10_mediation_")
        dst = REPORT_DIR / f"12b_antibiotic_ppi_{suffix}"
        shutil.copy2(src, dst)


def compare():
    if not CANONICAL_PRIMARY.is_file():
        raise FileNotFoundError(
            f"Canonical primary mediation file missing: {CANONICAL_PRIMARY}"
        )
    if not OUT_MODEL.is_file():
        raise FileNotFoundError(OUT_MODEL)

    primary = pd.read_csv(CANONICAL_PRIMARY, low_memory=False)
    sens = pd.read_csv(OUT_MODEL, low_memory=False)

    for name, df in (("primary", primary), ("sensitivity", sens)):
        missing = [c for c in KEY if c not in df.columns]
        if missing:
            raise ValueError(f"{name} mediation missing key columns: {missing}")
        if df.duplicated(KEY).any():
            raise ValueError(f"{name} mediation has duplicate exact paths")

    p_keys = set(map(tuple, primary[KEY].astype(str).to_numpy()))
    s_keys = set(map(tuple, sens[KEY].astype(str).to_numpy()))
    exact_match = p_keys == s_keys

    merged = primary.merge(
        sens,
        on=KEY,
        how="outer",
        suffixes=("_primary", "_abxppi"),
        indicator=True,
        validate="one_to_one",
    )

    if not exact_match:
        missing_sens = merged.loc[
            merged["_merge"].eq("left_only"), KEY
        ]
        extra_sens = merged.loc[
            merged["_merge"].eq("right_only"), KEY
        ]
        print("WARNING: sensitivity path set differs from primary.")
        print(f"Missing from sensitivity: {len(missing_sens)}")
        print(f"Extra in sensitivity: {len(extra_sens)}")

    matched = merged.loc[merged["_merge"].eq("both")].copy()

    for prefix in ("primary", "abxppi"):
        for col in (
            "mediation_FDR05_primary",
            "candidate_mediator_consistent",
            "indirect_CI_excludes_zero",
            "indirect_same_sign_as_total",
        ):
            full = f"{col}_{prefix}"
            if full in matched.columns:
                matched[full] = bool_series(matched[full])

    # Path-level status transitions.
    def status(prefix: str) -> pd.Series:
        sig = matched[f"mediation_FDR05_primary_{prefix}"]
        con = matched[f"candidate_mediator_consistent_{prefix}"]
        return np.select(
            [con, sig],
            ["direction_consistent", "significant_inconsistent"],
            default="nonsignificant",
        )

    matched["status_primary"] = status("primary")
    matched["status_abxppi"] = status("abxppi")
    matched["status_transition"] = (
        matched["status_primary"].astype(str)
        + " -> "
        + matched["status_abxppi"].astype(str)
    )

    effect_cols = [
        "a_diet_to_microbiome",
        "b_microbiome_to_cgm",
        "indirect_effect",
        "total_effect",
    ]

    summary_rows = []

    p_sig = bool_series(primary["mediation_FDR05_primary"])
    s_sig = bool_series(sens["mediation_FDR05_primary"])
    p_cons = bool_series(primary["candidate_mediator_consistent"])
    s_cons = bool_series(sens["candidate_mediator_consistent"])

    summary_rows.extend(
        [
            {"metric": "primary_paths", "value": len(primary)},
            {"metric": "abxppi_paths", "value": len(sens)},
            {"metric": "exact_path_set_match", "value": exact_match},
            {"metric": "primary_significant", "value": int(p_sig.sum())},
            {"metric": "abxppi_significant", "value": int(s_sig.sum())},
            {"metric": "primary_direction_consistent", "value": int(p_cons.sum())},
            {"metric": "abxppi_direction_consistent", "value": int(s_cons.sum())},
        ]
    )

    if len(matched):
        primary_sig_retained = int(
            (
                matched["mediation_FDR05_primary_primary"]
                & matched["mediation_FDR05_primary_abxppi"]
            ).sum()
        )
        primary_cons_retained = int(
            (
                matched["candidate_mediator_consistent_primary"]
                & matched["candidate_mediator_consistent_abxppi"]
            ).sum()
        )
        summary_rows.extend(
            [
                {
                    "metric": "primary_significant_retained",
                    "value": primary_sig_retained,
                },
                {
                    "metric": "primary_direction_consistent_retained",
                    "value": primary_cons_retained,
                },
                {
                    "metric": "median_N_primary",
                    "value": float(
                        pd.to_numeric(
                            matched["N_primary"], errors="coerce"
                        ).median()
                    ),
                },
                {
                    "metric": "median_N_abxppi",
                    "value": float(
                        pd.to_numeric(
                            matched["N_abxppi"], errors="coerce"
                        ).median()
                    ),
                },
                {
                    "metric": "median_N_change_abxppi_minus_primary",
                    "value": float(
                        (
                            pd.to_numeric(
                                matched["N_abxppi"], errors="coerce"
                            )
                            - pd.to_numeric(
                                matched["N_primary"], errors="coerce"
                            )
                        ).median()
                    ),
                },
            ]
        )

        for col in effect_cols:
            rho = spearman(
                matched[f"{col}_primary"],
                matched[f"{col}_abxppi"],
            )
            same, flips, frac = sign_concordance(
                matched[f"{col}_primary"],
                matched[f"{col}_abxppi"],
            )
            summary_rows.extend(
                [
                    {"metric": f"{col}_spearman_rho", "value": rho},
                    {"metric": f"{col}_same_sign", "value": same},
                    {"metric": f"{col}_sign_flips", "value": flips},
                    {"metric": f"{col}_sign_concordance_fraction", "value": frac},
                ]
            )

    # Derive recurrent AMED/hPDI glucose-CV core from canonical primary
    # rather than hard-coding species labels.
    p_cons_df = primary.loc[p_cons].copy()
    s_cons_df = sens.loc[s_cons].copy()

    def context_species(df, score):
        return set(
            df.loc[
                df["diet_score"].eq(score)
                & df["cgm_outcome"].eq("glucose_cv"),
                "species",
            ].astype(str)
        )

    p_amed = context_species(p_cons_df, "AMED")
    p_hpdi = context_species(p_cons_df, "hPDI")
    s_amed = context_species(s_cons_df, "AMED")
    s_hpdi = context_species(s_cons_df, "hPDI")

    primary_core = p_amed & p_hpdi
    sens_core = s_amed & s_hpdi

    core_union = sorted(primary_core | sens_core)
    core_rows = []
    for species in core_union:
        core_rows.append(
            {
                "species": species,
                "primary_core": species in primary_core,
                "abxppi_core": species in sens_core,
                "primary_core_retained": (
                    species in primary_core and species in sens_core
                ),
            }
        )
    core_compare = pd.DataFrame(core_rows)

    summary_rows.extend(
        [
            {"metric": "primary_recurrent_core_n", "value": len(primary_core)},
            {"metric": "abxppi_recurrent_core_n", "value": len(sens_core)},
            {
                "metric": "primary_core_retained_n",
                "value": len(primary_core & sens_core),
            },
            {
                "metric": "primary_core_all_retained",
                "value": primary_core.issubset(sens_core),
            },
        ]
    )

    summary = pd.DataFrame(summary_rows)

    OUT_PATH_COMPARE.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(OUT_PATH_COMPARE, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    core_compare.to_csv(OUT_CORE_COMPARE, index=False)

    metric = dict(zip(summary["metric"], summary["value"]))

    lines = [
        "=== STEP 12b ANTIBIOTIC/PPI-ADJUSTED MEDIATION SENSITIVITY ===",
        "",
        "Model logic:",
        "- canonical Step10 code reused unchanged",
        "- primary diabetes/A10 cohort",
        "- Model3 covariates + antibiotic_use + ppi_use",
        "- 1000-bootstrap default",
        "- canonical Step10 outputs were NOT overwritten",
        "",
        f"Primary paths: {metric.get('primary_paths')}",
        f"ABX/PPI paths: {metric.get('abxppi_paths')}",
        f"Exact path-set match: {metric.get('exact_path_set_match')}",
        f"Primary significant: {metric.get('primary_significant')}",
        f"ABX/PPI significant: {metric.get('abxppi_significant')}",
        f"Primary significant retained: {metric.get('primary_significant_retained')}",
        f"Primary direction-consistent: {metric.get('primary_direction_consistent')}",
        f"ABX/PPI direction-consistent: {metric.get('abxppi_direction_consistent')}",
        f"Primary direction-consistent retained: {metric.get('primary_direction_consistent_retained')}",
        "",
        f"Primary recurrent core N: {metric.get('primary_recurrent_core_n')}",
        f"ABX/PPI recurrent core N: {metric.get('abxppi_recurrent_core_n')}",
        f"Primary core retained N: {metric.get('primary_core_retained_n')}",
        f"Primary core all retained: {metric.get('primary_core_all_retained')}",
        "",
        f"Median N primary: {metric.get('median_N_primary')}",
        f"Median N ABX/PPI: {metric.get('median_N_abxppi')}",
        f"Median N change: {metric.get('median_N_change_abxppi_minus_primary')}",
        "",
        "Effect stability:",
    ]

    for col in effect_cols:
        lines.append(
            f"- {col}: rho={metric.get(col + '_spearman_rho')}, "
            f"sign_flips={metric.get(col + '_sign_flips')}"
        )

    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return summary, matched, core_compare


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260904)
    p.add_argument(
        "--skip-run",
        action="store_true",
        help="Only compare already-generated 12b output with canonical primary.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.bootstrap < 20:
        raise ValueError("--bootstrap must be >=20")

    if not args.skip_run:
        print("=" * 100)
        print("STEP 12b — REUSING CANONICAL STEP10 WITH ANTIBIOTIC/PPI COVARIATES")
        print("=" * 100)
        print(f"Canonical Step10: {CANONICAL_STEP10}")
        print(f"Sensitivity master: {SENS_MASTER}")
        print("Canonical primary outputs will NOT be overwritten.")
        print()
        run_canonical_step10_with_sensitivity_master(
            bootstrap=args.bootstrap,
            seed=args.seed,
        )

    summary, matched, core = compare()

    print()
    print("=" * 100)
    print("STEP 12b KEY COMPARISON")
    print("=" * 100)
    print(summary.to_string(index=False))

    print()
    print("--- STATUS TRANSITIONS ---")
    print(
        matched["status_transition"]
        .value_counts(dropna=False)
        .rename_axis("transition")
        .reset_index(name="n_paths")
        .to_string(index=False)
    )

    print()
    print("--- RECURRENT CORE ---")
    if core.empty:
        print("No recurrent core species found.")
    else:
        print(core.to_string(index=False))

    print()
    print(f"SENSITIVITY_MODEL={OUT_MODEL}")
    print(f"COMPARISON={OUT_PATH_COMPARE}")
    print(f"SUMMARY={OUT_TXT}")
    print("CANONICAL_STEP10_OVERWRITTEN=False")


if __name__ == "__main__":
    main()
