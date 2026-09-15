#!/usr/bin/env python3
"""Step 15f2 — protocol-window antibiotic/PPI mediation sensitivity.

Purpose
-------
Rerun the EXACT frozen 45 new-diet mediation-style paths in the PRIMARY cohort,
adding protocol-window antibiotic and PPI use as covariates.

This is the new-exposure analogue of the earlier protocol-window ABX/PPI
sensitivity used for the original diet scores.

Important
---------
- Candidate path set is NOT re-screened.
- Primary cohort definition is unchanged.
- The only model change is adding protocol-window antibiotic_use and ppi_use.
- Carbohydrate_pct remains exploratory.
- Cross-sectional mediation-style analysis only; no causal mediation claim.
- Outputs are isolated; canonical primary/strict outputs are not overwritten.

Protocol medication master
--------------------------
By default, this script auto-discovers a CSV under the project that:
- has participant_id;
- has antibiotic/PPI protocol-window columns;
- has a filename/path strongly indicating protocol/window medication output.

You can override discovery explicitly:
    --protocol-master /path/to/file.csv

Supported column pairs include:
- antibiotic_use / ppi_use
- antibiotic_use_protocol_window / ppi_use_protocol_window
- protocol_window_antibiotic_use / protocol_window_ppi_use

Outputs
-------
diet_microbiome_glucose_analysis/outputs/new_diet_extension/mediation/
    protocol_abxppi/
        ... rerun path outputs ...
    15f2_primary_vs_protocol_abxppi_path_comparison.csv
    15f2_primary_vs_protocol_abxppi_context_summary.csv
    15f2_primary_vs_protocol_abxppi_summary.txt
"""

from __future__ import annotations

from pathlib import Path
import argparse
import importlib.util
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
DG = ROOT / "diet_microbiome_glucose_analysis"

BASE_MED = (
    DG / "outputs" / "new_diet_extension" / "mediation"
)
PRIMARY_PATHS = BASE_MED / "15f_primary_mediation_paths.csv"
BASE_COHORT = (
    DG / "outputs" / "new_diet_extension" / "data"
    / "15d0_new_diet_cgm_cohort.csv"
)
STEP15F = DG / "scripts" / "15f_new_diet_mediation.py"

SENS_DIR = BASE_MED / "protocol_abxppi"
SENS_COHORT = SENS_DIR / "15f2_protocol_abxppi_cohort.csv"

PATH_COMPARE = (
    BASE_MED / "15f2_primary_vs_protocol_abxppi_path_comparison.csv"
)
CONTEXT_SUMMARY = (
    BASE_MED / "15f2_primary_vs_protocol_abxppi_context_summary.csv"
)
SUMMARY_TXT = (
    BASE_MED / "15f2_primary_vs_protocol_abxppi_summary.txt"
)

KEY = ["diet_score", "cgm_outcome", "species"]

SUPPORTED_COLUMN_PAIRS = [
    ("antibiotic_use", "ppi_use"),
    ("antibiotic_use_protocol_window", "ppi_use_protocol_window"),
    ("protocol_window_antibiotic_use", "protocol_window_ppi_use"),
]


def require(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


def norm_id(s: pd.Series) -> pd.Series:
    return (
        s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    )


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


def identify_med_columns(path: Path):
    cols = set(pd.read_csv(path, nrows=0).columns)
    if "participant_id" not in cols:
        return None
    for abx, ppi in SUPPORTED_COLUMN_PAIRS:
        if abx in cols and ppi in cols:
            return abx, ppi
    return None


def candidate_score(path: Path) -> int:
    s = str(path).lower()
    score = 0
    if "protocol" in s:
        score += 6
    if "window" in s:
        score += 5
    if "antibiotic" in s or "abx" in s:
        score += 3
    if "ppi" in s:
        score += 3
    if "sensitivity" in s:
        score += 2
    if "baseline" in s or "broad" in s:
        score -= 8
    if "candidate" in s:
        score -= 2
    return score


def discover_protocol_master(explicit: str | None):
    if explicit:
        p = Path(explicit)
        require(p, "explicit protocol-window medication master")
        pair = identify_med_columns(p)
        if pair is None:
            raise RuntimeError(
                f"Explicit file does not contain a supported antibiotic/PPI "
                f"column pair: {p}"
            )
        return p, pair

    roots = [
        ROOT / "co-variant" / "outputs",
        DG / "outputs",
    ]

    candidates = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.csv"):
            if p in seen:
                continue
            seen.add(p)
            s = str(p).lower()
            if not (
                "protocol" in s
                or "window" in s
                or "antibiotic" in s
                or "ppi" in s
                or "abx" in s
            ):
                continue
            try:
                pair = identify_med_columns(p)
            except Exception:
                continue
            if pair is not None:
                candidates.append((candidate_score(p), p, pair))

    if not candidates:
        raise RuntimeError(
            "Could not auto-discover protocol-window antibiotic/PPI master. "
            "Rerun with --protocol-master /exact/path/to/file.csv"
        )

    candidates.sort(key=lambda x: (x[0], str(x[1])), reverse=True)
    best_score = candidates[0][0]
    best = [x for x in candidates if x[0] == best_score]

    if len(best) != 1:
        text = "\n".join(
            f"score={sc} | {p} | columns={pair}"
            for sc, p, pair in candidates[:20]
        )
        raise RuntimeError(
            "Protocol master auto-discovery is ambiguous. "
            "Use --protocol-master explicitly.\nCandidates:\n" + text
        )

    _, p, pair = best[0]
    return p, pair


def build_sensitivity_cohort(protocol_master: Path, abx_col: str, ppi_col: str):
    require(BASE_COHORT, "Step15d0 base cohort")

    cohort = pd.read_csv(BASE_COHORT, low_memory=False)
    meds = pd.read_csv(
        protocol_master,
        usecols=["participant_id", abx_col, ppi_col],
        low_memory=False,
    )

    for label, d in [("cohort", cohort), ("protocol master", meds)]:
        d["participant_id"] = norm_id(d["participant_id"])
        if d["participant_id"].duplicated().any():
            raise RuntimeError(f"{label} has duplicate participant_id")

    meds[abx_col] = pd.to_numeric(meds[abx_col], errors="coerce")
    meds[ppi_col] = pd.to_numeric(meds[ppi_col], errors="coerce")

    # Hard semantic guard: only 0/1/NaN accepted.
    for c in [abx_col, ppi_col]:
        vals = set(meds[c].dropna().unique().tolist())
        if not vals.issubset({0, 1, 0.0, 1.0}):
            raise RuntimeError(
                f"{c} contains values outside 0/1/NaN: {sorted(vals)[:20]}"
            )

    meds = meds.rename(
        columns={
            abx_col: "antibiotic_use_protocol_window",
            ppi_col: "ppi_use_protocol_window",
        }
    )

    out = cohort.merge(
        meds,
        on="participant_id",
        how="left",
        validate="one_to_one",
    )

    SENS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(SENS_COHORT, index=False)

    primary = truthy(out["primary_cgm_analysis_eligible"])
    qc = {
        "formal_rows": len(out),
        "primary_eligible": int(primary.sum()),
        "antibiotic_known_primary": int(
            out.loc[primary, "antibiotic_use_protocol_window"].notna().sum()
        ),
        "antibiotic_positive_primary": int(
            out.loc[primary, "antibiotic_use_protocol_window"].eq(1).sum()
        ),
        "ppi_known_primary": int(
            out.loc[primary, "ppi_use_protocol_window"].notna().sum()
        ),
        "ppi_positive_primary": int(
            out.loc[primary, "ppi_use_protocol_window"].eq(1).sum()
        ),
    }
    return qc


def run_protocol_sensitivity(step15f, bootstrap: int, seed: int):
    old_cohort = step15f.COHORT_FILE
    old_out_dir = step15f.OUT_DIR
    old_covariates = list(step15f.COVARIATES)
    old_categorical = set(step15f.CATEGORICAL)
    old_argv = sys.argv[:]

    try:
        step15f.COHORT_FILE = SENS_COHORT
        step15f.OUT_DIR = SENS_DIR
        step15f.COVARIATES = old_covariates + [
            "antibiotic_use_protocol_window",
            "ppi_use_protocol_window",
        ]
        # Keep binary medication variables numeric; no dummy expansion needed.
        step15f.CATEGORICAL = old_categorical

        sys.argv = [
            str(STEP15F),
            "--cohort-mode", "primary",
            "--bootstrap", str(bootstrap),
            "--seed", str(seed),
        ]
        rc = step15f.main()
        if rc not in (None, 0):
            raise RuntimeError(f"Step15f sensitivity run returned {rc}")
    finally:
        step15f.COHORT_FILE = old_cohort
        step15f.OUT_DIR = old_out_dir
        step15f.COVARIATES = old_covariates
        step15f.CATEGORICAL = old_categorical
        sys.argv = old_argv

    out = SENS_DIR / "15f_primary_mediation_paths.csv"
    require(out, "protocol-window mediation paths")
    return out


def compare(primary_path: Path, protocol_path: Path):
    p = pd.read_csv(primary_path, low_memory=False)
    s = pd.read_csv(protocol_path, low_memory=False)

    for label, d in [("primary", p), ("protocol", s)]:
        if len(d) != 45:
            raise RuntimeError(f"{label}: expected 45 paths, got {len(d)}")
        if d.duplicated(KEY).any():
            raise RuntimeError(f"{label}: duplicate path keys")

    p_keys = set(map(tuple, p[KEY].astype(str).to_numpy()))
    s_keys = set(map(tuple, s[KEY].astype(str).to_numpy()))
    if p_keys != s_keys:
        raise RuntimeError(
            f"Path set mismatch: primary-only={len(p_keys-s_keys)}, "
            f"protocol-only={len(s_keys-p_keys)}"
        )

    cols = KEY + [
        "analysis_role",
        "N",
        "a_diet_to_microbiome",
        "b_microbiome_to_cgm",
        "indirect_effect",
        "direct_effect",
        "total_effect",
        "FDR_BH_within_exposure_outcome",
        "mediation_FDR05_primary",
        "candidate_mediator_consistent",
    ]
    cols = [c for c in cols if c in p.columns and c in s.columns]

    m = p[cols].merge(
        s[cols],
        on=KEY,
        suffixes=("_primary", "_protocol"),
        validate="one_to_one",
    )

    p_sig = truthy(m["mediation_FDR05_primary_primary"])
    s_sig = truthy(m["mediation_FDR05_primary_protocol"])
    p_con = truthy(m["candidate_mediator_consistent_primary"])
    s_con = truthy(m["candidate_mediator_consistent_protocol"])

    p_ind = pd.to_numeric(m["indirect_effect_primary"], errors="coerce")
    s_ind = pd.to_numeric(m["indirect_effect_protocol"], errors="coerce")

    m["significant_retained"] = p_sig & s_sig
    m["significant_lost"] = p_sig & ~s_sig
    m["significant_gained"] = ~p_sig & s_sig
    m["consistent_retained"] = p_con & s_con
    m["consistent_lost"] = p_con & ~s_con
    m["consistent_gained"] = ~p_con & s_con
    m["indirect_same_sign"] = np.sign(p_ind) == np.sign(s_ind)

    m.to_csv(PATH_COMPARE, index=False)

    context_rows = []
    for (diet, outcome), g in m.groupby(
        ["diet_score", "cgm_outcome"], sort=True, dropna=False
    ):
        rho = (
            float(
                spearmanr(
                    pd.to_numeric(g["indirect_effect_primary"]),
                    pd.to_numeric(g["indirect_effect_protocol"]),
                ).statistic
            )
            if len(g) >= 3 else np.nan
        )
        context_rows.append({
            "diet_score": diet,
            "cgm_outcome": outcome,
            "paths_tested": len(g),
            "primary_significant": int(
                truthy(g["mediation_FDR05_primary_primary"]).sum()
            ),
            "protocol_significant": int(
                truthy(g["mediation_FDR05_primary_protocol"]).sum()
            ),
            "primary_consistent": int(
                truthy(g["candidate_mediator_consistent_primary"]).sum()
            ),
            "protocol_consistent": int(
                truthy(g["candidate_mediator_consistent_protocol"]).sum()
            ),
            "primary_consistent_retained": int(
                g["consistent_retained"].sum()
            ),
            "primary_consistent_lost": int(g["consistent_lost"].sum()),
            "protocol_consistent_gained": int(g["consistent_gained"].sum()),
            "median_N_primary": float(
                pd.to_numeric(g["N_primary"]).median()
            ),
            "median_N_protocol": float(
                pd.to_numeric(g["N_protocol"]).median()
            ),
            "indirect_rho": rho,
            "indirect_sign_flips": int((~g["indirect_same_sign"]).sum()),
        })

    context = pd.DataFrame(context_rows)
    context.to_csv(CONTEXT_SUMMARY, index=False)

    rho_all = float(spearmanr(p_ind, s_ind).statistic)

    exposure = (
        m.groupby("diet_score", as_index=False)
        .agg(
            paths_tested=("species", "size"),
            primary_significant=(
                "mediation_FDR05_primary_primary",
                lambda x: int(truthy(x).sum()),
            ),
            protocol_significant=(
                "mediation_FDR05_primary_protocol",
                lambda x: int(truthy(x).sum()),
            ),
            primary_consistent=(
                "candidate_mediator_consistent_primary",
                lambda x: int(truthy(x).sum()),
            ),
            protocol_consistent=(
                "candidate_mediator_consistent_protocol",
                lambda x: int(truthy(x).sum()),
            ),
            primary_consistent_retained=("consistent_retained", "sum"),
            primary_consistent_lost=("consistent_lost", "sum"),
            protocol_consistent_gained=("consistent_gained", "sum"),
        )
    )

    metrics = {
        "exact_45_path_set_match": True,
        "primary_significant": int(p_sig.sum()),
        "protocol_significant": int(s_sig.sum()),
        "primary_significant_retained": int((p_sig & s_sig).sum()),
        "primary_significant_lost": int((p_sig & ~s_sig).sum()),
        "protocol_significant_gained": int((~p_sig & s_sig).sum()),
        "primary_consistent": int(p_con.sum()),
        "protocol_consistent": int(s_con.sum()),
        "primary_consistent_retained": int((p_con & s_con).sum()),
        "primary_consistent_lost": int((p_con & ~s_con).sum()),
        "protocol_consistent_gained": int((~p_con & s_con).sum()),
        "indirect_rho_primary_vs_protocol": rho_all,
        "indirect_sign_flips": int((~m["indirect_same_sign"]).sum()),
    }

    return m, context, exposure, metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-master", default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()

    for p, label in [
        (PRIMARY_PATHS, "primary mediation paths"),
        (BASE_COHORT, "new-diet CGM cohort"),
        (STEP15F, "Step15f mediation script"),
    ]:
        require(p, label)

    protocol_master, (abx_col, ppi_col) = discover_protocol_master(
        args.protocol_master
    )

    print("=== STEP 15f2 PROTOCOL-WINDOW ANTIBIOTIC/PPI SENSITIVITY ===")
    print(f"PROTOCOL_MASTER={protocol_master}")
    print(f"ANTIBIOTIC_COLUMN={abx_col}")
    print(f"PPI_COLUMN={ppi_col}")
    print("PATH_SET_FIXED=45")
    print("COHORT_MODE=primary")
    print("BRIDGE_RESCREEN=False")

    qc = build_sensitivity_cohort(protocol_master, abx_col, ppi_col)

    print("\n--- MEDICATION COVERAGE QC ---")
    for k, v in qc.items():
        print(f"{k}={v}")

    step15f = load_module(STEP15F, "step15f2_base_mediation")
    protocol_paths = run_protocol_sensitivity(
        step15f,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )

    m, context, exposure, metrics = compare(
        PRIMARY_PATHS,
        protocol_paths,
    )

    print("\n--- PRIMARY vs PROTOCOL ABX/PPI ---")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k.upper()}={v:.9f}")
        else:
            print(f"{k.upper()}={v}")

    print("\n--- EXPOSURE SUMMARY ---")
    print(exposure.to_string(index=False))

    print("\n--- CONTEXT SUMMARY ---")
    print(context.to_string(index=False))

    lost = m.loc[m["consistent_lost"]].copy()
    gained = m.loc[m["consistent_gained"]].copy()

    print("\n--- LOST PRIMARY-CONSISTENT PATHS ---")
    if len(lost):
        print(
            lost[
                [
                    "diet_score",
                    "cgm_outcome",
                    "species",
                    "indirect_effect_primary",
                    "indirect_effect_protocol",
                    "FDR_BH_within_exposure_outcome_primary",
                    "FDR_BH_within_exposure_outcome_protocol",
                ]
            ].to_string(index=False)
        )
    else:
        print("NONE")

    print("\n--- PROTOCOL-GAINED CONSISTENT PATHS ---")
    if len(gained):
        print(
            gained[
                [
                    "diet_score",
                    "cgm_outcome",
                    "species",
                    "indirect_effect_primary",
                    "indirect_effect_protocol",
                    "FDR_BH_within_exposure_outcome_primary",
                    "FDR_BH_within_exposure_outcome_protocol",
                ]
            ].to_string(index=False)
        )
    else:
        print("NONE")

    lines = [
        "=== STEP 15f2 PROTOCOL-WINDOW ANTIBIOTIC/PPI SENSITIVITY ===",
        f"PROTOCOL_MASTER={protocol_master}",
        f"ANTIBIOTIC_COLUMN={abx_col}",
        f"PPI_COLUMN={ppi_col}",
        "PATH_SET_FIXED=45",
        "COHORT_MODE=primary",
        "BRIDGE_RESCREEN=False",
        "",
        "--- MEDICATION COVERAGE QC ---",
        *[f"{k}={v}" for k, v in qc.items()],
        "",
        "--- PRIMARY vs PROTOCOL ABX/PPI ---",
        *[
            f"{k.upper()}={v:.9f}" if isinstance(v, float)
            else f"{k.upper()}={v}"
            for k, v in metrics.items()
        ],
        "",
        "--- EXPOSURE SUMMARY ---",
        exposure.to_string(index=False),
        "",
        "--- CONTEXT SUMMARY ---",
        context.to_string(index=False),
        "",
        "--- LOST PRIMARY-CONSISTENT PATHS ---",
        (
            lost[
                [
                    "diet_score",
                    "cgm_outcome",
                    "species",
                    "indirect_effect_primary",
                    "indirect_effect_protocol",
                    "FDR_BH_within_exposure_outcome_primary",
                    "FDR_BH_within_exposure_outcome_protocol",
                ]
            ].to_string(index=False)
            if len(lost) else "NONE"
        ),
        "",
        "--- PROTOCOL-GAINED CONSISTENT PATHS ---",
        (
            gained[
                [
                    "diet_score",
                    "cgm_outcome",
                    "species",
                    "indirect_effect_primary",
                    "indirect_effect_protocol",
                    "FDR_BH_within_exposure_outcome_primary",
                    "FDR_BH_within_exposure_outcome_protocol",
                ]
            ].to_string(index=False)
            if len(gained) else "NONE"
        ),
        "",
        f"PATH_COMPARISON={PATH_COMPARE}",
        f"CONTEXT_SUMMARY={CONTEXT_SUMMARY}",
        f"SENSITIVITY_OUTPUT_DIR={SENS_DIR}",
    ]
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
