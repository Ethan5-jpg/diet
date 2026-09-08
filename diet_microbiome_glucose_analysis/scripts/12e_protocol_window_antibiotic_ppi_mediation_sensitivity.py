#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
PROJECT = ROOT / "diet_microbiome_glucose_analysis"
SCRIPT_DIR = PROJECT / "scripts"

CANONICAL_STEP10 = SCRIPT_DIR / "10_diet_microbiome_cgm_mediation.py"
PROTOCOL_MASTER = (
    ROOT / "co-variant" / "outputs" / "data"
    / "02_covariate_master_antibiotic_ppi_during_logging.csv"
)
CANONICAL_PRIMARY = PROJECT / "outputs" / "models" / "10_mediation_paths_model3.csv"

MODEL_DIR = PROJECT / "outputs" / "models"
REPORT_DIR = PROJECT / "outputs" / "reports"
TMP_MODEL_DIR = MODEL_DIR / "_12e_protocol_window_tmp"
TMP_REPORT_DIR = REPORT_DIR / "_12e_protocol_window_tmp"

OUT_MODEL = MODEL_DIR / "12e_protocol_window_mediation_paths_model3.csv"
OUT_COV_AUDIT = REPORT_DIR / "12e_protocol_window_covariate_audit.csv"
OUT_PATH_COMPARE = REPORT_DIR / "12e_protocol_window_path_comparison.csv"
OUT_CORE_COMPARE = REPORT_DIR / "12e_protocol_window_core_comparison.csv"
OUT_SUMMARY_CSV = REPORT_DIR / "12e_protocol_window_comparison_summary.csv"
OUT_SUMMARY_TXT = REPORT_DIR / "12e_protocol_window_comparison_summary.txt"

KEY = ["diet_score", "cgm_outcome", "species"]


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).eq(1)
    return (
        series.astype(str).str.strip().str.lower()
        .isin({"true", "1", "1.0", "yes", "y", "t"})
    )


def spearman(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    ok = x.notna() & y.notna()
    if ok.sum() < 3:
        return np.nan
    return float(
        x.loc[ok].rank(method="average").corr(
            y.loc[ok].rank(method="average"), method="pearson"
        )
    )


def sign_stats(a: pd.Series, b: pd.Series):
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    ok = x.notna() & y.notna()
    if not ok.any():
        return 0, 0, np.nan
    same = np.sign(x.loc[ok].to_numpy()) == np.sign(y.loc[ok].to_numpy())
    return int(same.sum()), int((~same).sum()), float(same.mean())


def load_step10_module(path: Path):
    spec = importlib.util.spec_from_file_location("hpp_step10_protocol_window", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import canonical Step10: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_covariate_guard(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise RuntimeError(f"Covariate audit missing: {path}")

    audit = pd.read_csv(path, low_memory=False)
    needed = {"logical_name", "source_column", "status"}
    if not needed.issubset(audit.columns):
        raise RuntimeError(f"Unexpected covariate audit schema: {audit.columns.tolist()}")

    failures = []
    for logical in ("antibiotic_use", "ppi_use"):
        hit = audit.loc[audit["logical_name"].eq(logical)]
        if len(hit) != 1:
            failures.append(f"{logical}: expected 1 audit row, found {len(hit)}")
            continue
        row = hit.iloc[0]
        if row["status"] != "optional_found_and_included":
            failures.append(
                f"{logical}: status={row['status']!r}, source_column={row['source_column']!r}"
            )
        if pd.isna(row["source_column"]) or not str(row["source_column"]).strip():
            failures.append(f"{logical}: missing source_column")

    if failures:
        raise RuntimeError(
            "ABX/PPI COVARIATE GUARD FAILED. Do not interpret this run as adjusted.\n- "
            + "\n- ".join(failures)
        )
    return audit


def run_step10(bootstrap: int, seed: int) -> pd.DataFrame:
    for path in (CANONICAL_STEP10, PROTOCOL_MASTER):
        if not path.is_file():
            raise FileNotFoundError(path)

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
            "--covariates", str(PROTOCOL_MASTER),
            "--cohort-mode", "primary",
            "--bootstrap", str(bootstrap),
            "--seed", str(seed),
        ]
        mod.main()
    finally:
        sys.argv = old_argv

    generated_model = TMP_MODEL_DIR / "10_mediation_paths_model3.csv"
    generated_audit = TMP_REPORT_DIR / "10_mediation_covariate_audit.csv"

    audit = assert_covariate_guard(generated_audit)

    if not generated_model.is_file():
        raise RuntimeError("Canonical Step10 produced no mediation model output")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy2(generated_model, OUT_MODEL)
    shutil.copy2(generated_audit, OUT_COV_AUDIT)

    for src in TMP_REPORT_DIR.glob("10_mediation_*"):
        suffix = src.name.removeprefix("10_mediation_")
        shutil.copy2(src, REPORT_DIR / f"12e_protocol_window_{suffix}")

    return audit


def compare():
    if not CANONICAL_PRIMARY.is_file():
        raise FileNotFoundError(CANONICAL_PRIMARY)
    if not OUT_MODEL.is_file():
        raise FileNotFoundError(OUT_MODEL)

    primary = pd.read_csv(CANONICAL_PRIMARY, low_memory=False)
    sens = pd.read_csv(OUT_MODEL, low_memory=False)

    for label, df in (("primary", primary), ("protocol", sens)):
        missing = [c for c in KEY if c not in df.columns]
        if missing:
            raise ValueError(f"{label} mediation missing keys: {missing}")
        if df.duplicated(KEY).any():
            raise ValueError(f"{label} mediation has duplicate exact paths")

    pset = set(map(tuple, primary[KEY].astype(str).to_numpy()))
    sset = set(map(tuple, sens[KEY].astype(str).to_numpy()))
    exact_qc = pset == sset

    merged = primary.merge(
        sens, on=KEY, how="outer",
        suffixes=("_primary", "_protocol"),
        indicator=True, validate="one_to_one",
    )
    matched = merged.loc[merged["_merge"].eq("both")].copy()

    for prefix in ("primary", "protocol"):
        for col in ("mediation_FDR05_primary", "candidate_mediator_consistent"):
            matched[f"{col}_{prefix}"] = bool_series(matched[f"{col}_{prefix}"])

    def status(prefix: str) -> pd.Series:
        sig = matched[f"mediation_FDR05_primary_{prefix}"]
        con = matched[f"candidate_mediator_consistent_{prefix}"]
        return np.select(
            [con, sig],
            ["direction_consistent", "significant_inconsistent"],
            default="nonsignificant",
        )

    matched["status_primary"] = status("primary")
    matched["status_protocol"] = status("protocol")
    matched["status_transition"] = (
        matched["status_primary"].astype(str)
        + " -> "
        + matched["status_protocol"].astype(str)
    )

    p_sig = bool_series(primary["mediation_FDR05_primary"])
    s_sig = bool_series(sens["mediation_FDR05_primary"])
    p_con = bool_series(primary["candidate_mediator_consistent"])
    s_con = bool_series(sens["candidate_mediator_consistent"])

    rows = []
    def add(metric, value):
        rows.append({"metric": metric, "value": value})

    add("pipeline_qc_exact_path_set_match", exact_qc)
    add("primary_paths", len(primary))
    add("protocol_paths", len(sens))
    add("primary_significant", int(p_sig.sum()))
    add("protocol_significant", int(s_sig.sum()))
    add(
        "primary_significant_retained",
        int(
            (
                matched["mediation_FDR05_primary_primary"]
                & matched["mediation_FDR05_primary_protocol"]
            ).sum()
        ),
    )
    add("primary_direction_consistent", int(p_con.sum()))
    add("protocol_direction_consistent", int(s_con.sum()))
    add(
        "primary_direction_consistent_retained",
        int(
            (
                matched["candidate_mediator_consistent_primary"]
                & matched["candidate_mediator_consistent_protocol"]
            ).sum()
        ),
    )

    add("median_N_primary", float(pd.to_numeric(matched["N_primary"], errors="coerce").median()))
    add("median_N_protocol", float(pd.to_numeric(matched["N_protocol"], errors="coerce").median()))
    add(
        "median_N_change_protocol_minus_primary",
        float(
            (
                pd.to_numeric(matched["N_protocol"], errors="coerce")
                - pd.to_numeric(matched["N_primary"], errors="coerce")
            ).median()
        ),
    )

    for effect in (
        "a_diet_to_microbiome",
        "b_microbiome_to_cgm",
        "indirect_effect",
        "total_effect",
    ):
        rho = spearman(matched[f"{effect}_primary"], matched[f"{effect}_protocol"])
        same, flips, frac = sign_stats(
            matched[f"{effect}_primary"], matched[f"{effect}_protocol"]
        )
        add(f"{effect}_spearman_rho", rho)
        add(f"{effect}_same_sign", same)
        add(f"{effect}_sign_flips", flips)
        add(f"{effect}_sign_concordance_fraction", frac)

    def context_species(df: pd.DataFrame, score: str) -> set:
        con = bool_series(df["candidate_mediator_consistent"])
        return set(
            df.loc[
                con
                & df["diet_score"].eq(score)
                & df["cgm_outcome"].eq("glucose_cv"),
                "species",
            ].astype(str)
        )

    p_core = context_species(primary, "AMED") & context_species(primary, "hPDI")
    s_core = context_species(sens, "AMED") & context_species(sens, "hPDI")

    add("primary_recurrent_core_n", len(p_core))
    add("protocol_recurrent_core_n", len(s_core))
    add("primary_core_retained_n", len(p_core & s_core))
    add("primary_core_all_retained", p_core.issubset(s_core))

    core = pd.DataFrame(
        [
            {
                "species": species,
                "primary_core": species in p_core,
                "protocol_core": species in s_core,
                "primary_core_retained": species in p_core and species in s_core,
            }
            for species in sorted(p_core | s_core)
        ]
    )

    summary = pd.DataFrame(rows)
    matched.to_csv(OUT_PATH_COMPARE, index=False)
    core.to_csv(OUT_CORE_COMPARE, index=False)
    summary.to_csv(OUT_SUMMARY_CSV, index=False)

    metric = dict(zip(summary["metric"], summary["value"]))
    lines = [
        "=== STEP 12e PROTOCOL-WINDOW ABX/PPI MEDIATION SENSITIVITY ===",
        "Mandatory antibiotic/PPI covariate guard PASSED.",
        "Exact path-set match is PIPELINE QC only.",
        "",
        f"Pipeline exact path-set QC: {metric['pipeline_qc_exact_path_set_match']}",
        f"Primary significant: {metric['primary_significant']}",
        f"Protocol-window significant: {metric['protocol_significant']}",
        f"Primary significant retained: {metric['primary_significant_retained']}",
        f"Primary direction-consistent: {metric['primary_direction_consistent']}",
        f"Protocol-window direction-consistent: {metric['protocol_direction_consistent']}",
        f"Primary direction-consistent retained: {metric['primary_direction_consistent_retained']}",
        f"Primary recurrent core: {metric['primary_recurrent_core_n']}",
        f"Protocol-window recurrent core: {metric['protocol_recurrent_core_n']}",
        f"Primary core retained: {metric['primary_core_retained_n']}",
        f"Median N primary: {metric['median_N_primary']}",
        f"Median N protocol: {metric['median_N_protocol']}",
        f"Median N change: {metric['median_N_change_protocol_minus_primary']}",
        f"Indirect rho: {metric['indirect_effect_spearman_rho']}",
        f"Indirect sign flips: {metric['indirect_effect_sign_flips']}",
    ]
    OUT_SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary, matched, core


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260904)
    p.add_argument("--skip-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap < 20:
        raise ValueError("--bootstrap must be >=20")

    if not args.skip_run:
        print("=" * 100)
        print("STEP 12e — PROTOCOL-WINDOW ANTIBIOTIC/PPI MEDIATION")
        print("=" * 100)
        print(f"Covariate master: {PROTOCOL_MASTER}")
        print("Canonical Step10 outputs will NOT be overwritten.")
        audit = run_step10(args.bootstrap, args.seed)
        print("\n--- MANDATORY COVARIATE GUARD PASSED ---")
        print(
            audit.loc[
                audit["logical_name"].isin(["antibiotic_use", "ppi_use"])
            ].to_string(index=False)
        )

    summary, matched, core = compare()

    print("\n=== STEP 12e KEY COMPARISON ===")
    print(summary.to_string(index=False))

    print("\n--- STATUS TRANSITIONS ---")
    print(
        matched["status_transition"]
        .value_counts(dropna=False)
        .rename_axis("transition")
        .reset_index(name="n_paths")
        .to_string(index=False)
    )

    print("\n--- RECURRENT CORE ---")
    print(core.to_string(index=False) if len(core) else "<none>")

    print(f"\nSENSITIVITY_MODEL={OUT_MODEL}")
    print(f"COVARIATE_AUDIT={OUT_COV_AUDIT}")
    print(f"SUMMARY={OUT_SUMMARY_TXT}")
    print("CANONICAL_STEP10_OVERWRITTEN=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
