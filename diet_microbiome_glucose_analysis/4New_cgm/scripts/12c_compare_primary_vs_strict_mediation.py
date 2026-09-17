#!/usr/bin/env python3
"""Step 12c — consolidate primary vs strict new-CGM mediation results.

This is a READ/COMPARE step only:
- no model refit
- no bootstrap rerun
- no candidate re-screening

It compares the exact frozen 73 Step12b paths between:
    primary cohort
    strict known-nondiabetes/A10 cohort

Primary inference remains the primary Step12b result set.
The strict-retained subset is a robustness annotation only.

Outputs:
    primary_vs_strict_path_comparison.csv
    primary_vs_strict_context_summary.csv
    strict_retained_mediation_paths.csv
    primary_lost_in_strict.csv
    strict_gained_vs_primary.csv
    step12c_primary_vs_strict_summary.txt
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
BRANCH_CANDIDATES = [
    ROOT / "diet_microbiome_glucose_analysis" / "4New_cgm",
    ROOT / "4New_cgm",
]

EXPECTED_PATHS = 73
EXPECTED_CONTEXTS = 7
EXPECTED_SPECIES = 36

KEY = ["diet_score", "cgm_outcome", "species"]


def truthy(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).eq(1)
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "1.0", "yes", "y", "t"})
    )


def find_branch(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(p)
        return p
    found = [p for p in BRANCH_CANDIDATES if p.is_dir()]
    if not found:
        raise FileNotFoundError("4New_cgm branch not found")
    return found[0]


def find_latest_mediation_dir(branch: Path, cohort_mode: str) -> Path:
    candidates = []
    pattern = f"mediation_newcgm_*_{cohort_mode}"
    for p in (branch / "outputs").glob(pattern):
        if "smoke" in p.name:
            continue
        model_name = (
            "12b_newcgm_primary_mediation_paths.csv"
            if cohort_mode == "primary"
            else "12b_newcgm_strict_mediation_paths.csv"
        )
        f = p / "models" / model_name
        if f.is_file():
            candidates.append((f.stat().st_mtime, p))
    if not candidates:
        raise FileNotFoundError(
            f"No completed non-smoke {cohort_mode} mediation run found"
        )
    candidates.sort(reverse=True)
    return candidates[0][1]


def model_file(run_dir: Path, cohort_mode: str) -> Path:
    name = (
        "12b_newcgm_primary_mediation_paths.csv"
        if cohort_mode == "primary"
        else "12b_newcgm_strict_mediation_paths.csv"
    )
    return run_dir / "models" / name


def load_result(path: Path, cohort_mode: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)

    d = pd.read_csv(path, low_memory=False)

    required = {
        *KEY,
        "N",
        "indirect_effect",
        "bootstrap_p_indirect",
        "FDR_BH_within_score_outcome",
        "mediation_FDR05_primary",
        "candidate_mediator_consistent",
        "indirect_CI95_lower",
        "indirect_CI95_upper",
        "indirect_same_sign_as_total",
        "indirect_CI_excludes_zero",
    }
    missing = sorted(required - set(d.columns))
    if missing:
        raise RuntimeError(
            f"{cohort_mode} mediation output missing columns: {missing}"
        )

    if len(d) != EXPECTED_PATHS:
        raise RuntimeError(
            f"{cohort_mode}: expected {EXPECTED_PATHS} rows, got {len(d)}"
        )
    if d.duplicated(KEY).any():
        raise RuntimeError(f"{cohort_mode}: duplicate exact paths")

    for c in [
        "N", "indirect_effect", "bootstrap_p_indirect",
        "FDR_BH_within_score_outcome",
        "indirect_CI95_lower", "indirect_CI95_upper",
    ]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d["mediation_sig"] = truthy(d["mediation_FDR05_primary"])
    d["consistent"] = truthy(d["candidate_mediator_consistent"])

    if "bridge_highest_priority" in d.columns:
        d["highest_priority"] = truthy(d["bridge_highest_priority"])
    elif "highest_priority_bridge_candidate" in d.columns:
        d["highest_priority"] = truthy(
            d["highest_priority_bridge_candidate"]
        )
    else:
        d["highest_priority"] = False

    return d


def safe_spearman(x, y):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan
    rho, p = spearmanr(x.loc[mask], y.loc[mask])
    return float(rho), float(p)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-dir", default=None)
    parser.add_argument("--primary-dir", default=None)
    parser.add_argument("--strict-dir", default=None)
    args = parser.parse_args()

    branch = find_branch(args.branch_dir)

    primary_dir = (
        Path(args.primary_dir).expanduser().resolve()
        if args.primary_dir
        else find_latest_mediation_dir(branch, "primary")
    )
    strict_dir = (
        Path(args.strict_dir).expanduser().resolve()
        if args.strict_dir
        else find_latest_mediation_dir(branch, "strict")
    )

    p = load_result(model_file(primary_dir, "primary"), "primary")
    s = load_result(model_file(strict_dir, "strict"), "strict")

    p_keys = set(map(tuple, p[KEY].to_numpy()))
    s_keys = set(map(tuple, s[KEY].to_numpy()))
    if p_keys != s_keys:
        raise RuntimeError(
            "Primary/strict exact 73-path sets differ: "
            f"primary_only={len(p_keys-s_keys)}, strict_only={len(s_keys-p_keys)}"
        )

    if p[KEY].drop_duplicates().shape[0] != EXPECTED_PATHS:
        raise RuntimeError("Exact path count validation failed")

    if p[["diet_score", "cgm_outcome"]].drop_duplicates().shape[0] != EXPECTED_CONTEXTS:
        raise RuntimeError("Expected 7 Diet-CGM contexts")

    if p["species"].nunique() != EXPECTED_SPECIES:
        raise RuntimeError("Expected 36 unique exact species")

    keep_cols = KEY + [
        "N",
        "indirect_effect",
        "bootstrap_p_indirect",
        "FDR_BH_within_score_outcome",
        "mediation_sig",
        "consistent",
        "highest_priority",
        "indirect_CI95_lower",
        "indirect_CI95_upper",
    ]

    merged = p[keep_cols].rename(
        columns={c: f"{c}_primary" for c in keep_cols if c not in KEY}
    ).merge(
        s[keep_cols].rename(
            columns={c: f"{c}_strict" for c in keep_cols if c not in KEY}
        ),
        on=KEY,
        how="inner",
        validate="one_to_one",
    )

    merged["indirect_same_sign_primary_strict"] = (
        np.sign(merged["indirect_effect_primary"])
        == np.sign(merged["indirect_effect_strict"])
    )

    merged["primary_sig_retained_strict"] = (
        merged["mediation_sig_primary"]
        & merged["mediation_sig_strict"]
    )
    merged["primary_sig_lost_strict"] = (
        merged["mediation_sig_primary"]
        & ~merged["mediation_sig_strict"]
    )
    merged["strict_sig_gained_vs_primary"] = (
        ~merged["mediation_sig_primary"]
        & merged["mediation_sig_strict"]
    )

    merged["primary_consistent_retained_strict"] = (
        merged["consistent_primary"]
        & merged["consistent_strict"]
        & merged["indirect_same_sign_primary_strict"]
    )
    merged["primary_consistent_lost_strict"] = (
        merged["consistent_primary"]
        & ~merged["consistent_strict"]
    )
    merged["strict_consistent_gained_vs_primary"] = (
        ~merged["consistent_primary"]
        & merged["consistent_strict"]
    )

    merged["strict_retained_highest_priority"] = (
        merged["primary_consistent_retained_strict"]
        & merged["highest_priority_primary"]
        & merged["highest_priority_strict"]
    )

    # Overall correlations.
    rho_all, rho_all_p = safe_spearman(
        merged["indirect_effect_primary"],
        merged["indirect_effect_strict"],
    )
    rho_primary_sig, rho_primary_sig_p = safe_spearman(
        merged.loc[merged["mediation_sig_primary"], "indirect_effect_primary"],
        merged.loc[merged["mediation_sig_primary"], "indirect_effect_strict"],
    )

    # Context-level exact retention.
    rows = []
    for (diet, outcome), g in merged.groupby(["diet_score", "cgm_outcome"]):
        r_all, r_all_p = safe_spearman(
            g["indirect_effect_primary"],
            g["indirect_effect_strict"],
        )
        rows.append({
            "diet_score": diet,
            "cgm_outcome": outcome,
            "paths_tested": len(g),
            "primary_significant": int(g["mediation_sig_primary"].sum()),
            "strict_significant": int(g["mediation_sig_strict"].sum()),
            "primary_sig_retained_strict": int(
                g["primary_sig_retained_strict"].sum()
            ),
            "primary_sig_lost_strict": int(
                g["primary_sig_lost_strict"].sum()
            ),
            "strict_sig_gained_vs_primary": int(
                g["strict_sig_gained_vs_primary"].sum()
            ),
            "primary_consistent": int(g["consistent_primary"].sum()),
            "strict_consistent": int(g["consistent_strict"].sum()),
            "primary_consistent_retained_strict": int(
                g["primary_consistent_retained_strict"].sum()
            ),
            "primary_consistent_lost_strict": int(
                g["primary_consistent_lost_strict"].sum()
            ),
            "strict_consistent_gained_vs_primary": int(
                g["strict_consistent_gained_vs_primary"].sum()
            ),
            "highest_priority_paths": int(g["highest_priority_primary"].sum()),
            "strict_retained_highest_priority": int(
                g["strict_retained_highest_priority"].sum()
            ),
            "indirect_rho_primary_vs_strict": r_all,
            "indirect_rho_p": r_all_p,
            "sign_flips_primary_vs_strict": int(
                (~g["indirect_same_sign_primary_strict"]).sum()
            ),
            "median_N_primary": float(g["N_primary"].median()),
            "median_N_strict": float(g["N_strict"].median()),
        })

    context = pd.DataFrame(rows).sort_values(
        ["cgm_outcome", "diet_score"]
    )

    primary_sig = int(merged["mediation_sig_primary"].sum())
    strict_sig = int(merged["mediation_sig_strict"].sum())
    retained = int(merged["primary_sig_retained_strict"].sum())
    lost = int(merged["primary_sig_lost_strict"].sum())
    gained = int(merged["strict_sig_gained_vs_primary"].sum())

    primary_cons = int(merged["consistent_primary"].sum())
    strict_cons = int(merged["consistent_strict"].sum())
    retained_cons = int(
        merged["primary_consistent_retained_strict"].sum()
    )
    lost_cons = int(
        merged["primary_consistent_lost_strict"].sum()
    )
    gained_cons = int(
        merged["strict_consistent_gained_vs_primary"].sum()
    )

    sign_flips_all = int(
        (~merged["indirect_same_sign_primary_strict"]).sum()
    )
    sign_flips_primary_sig = int(
        (
            merged["mediation_sig_primary"]
            & ~merged["indirect_same_sign_primary_strict"]
        ).sum()
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = branch / "outputs" / f"mediation_consolidation_{stamp}"
    reports = out / "reports"
    reports.mkdir(parents=True, exist_ok=False)

    path_table = reports / "primary_vs_strict_path_comparison.csv"
    context_table = reports / "primary_vs_strict_context_summary.csv"
    retained_table = reports / "strict_retained_mediation_paths.csv"
    lost_table = reports / "primary_lost_in_strict.csv"
    gained_table = reports / "strict_gained_vs_primary.csv"
    summary_txt = reports / "step12c_primary_vs_strict_summary.txt"
    manifest = reports / "manifest.json"

    merged.to_csv(path_table, index=False)
    context.to_csv(context_table, index=False)
    merged.loc[
        merged["primary_consistent_retained_strict"]
    ].to_csv(retained_table, index=False)
    merged.loc[
        merged["primary_consistent_lost_strict"]
    ].to_csv(lost_table, index=False)
    merged.loc[
        merged["strict_consistent_gained_vs_primary"]
    ].to_csv(gained_table, index=False)

    lines = [
        "=== STEP 12c PRIMARY vs STRICT MEDIATION CONSOLIDATION ===",
        f"PRIMARY_DIR={primary_dir}",
        f"STRICT_DIR={strict_dir}",
        "MODELS_REFIT=False",
        "BOOTSTRAP_RERUN=False",
        f"EXACT_73_PATH_SET_MATCH=True",
        "",
        f"PRIMARY_SIGNIFICANT={primary_sig}",
        f"STRICT_SIGNIFICANT={strict_sig}",
        f"PRIMARY_SIG_RETAINED_STRICT={retained}",
        f"PRIMARY_SIG_LOST_STRICT={lost}",
        f"STRICT_SIG_GAINED_VS_PRIMARY={gained}",
        "",
        f"PRIMARY_CONSISTENT={primary_cons}",
        f"STRICT_CONSISTENT={strict_cons}",
        f"PRIMARY_CONSISTENT_RETAINED_STRICT={retained_cons}",
        f"PRIMARY_CONSISTENT_LOST_STRICT={lost_cons}",
        f"STRICT_CONSISTENT_GAINED_VS_PRIMARY={gained_cons}",
        "",
        f"INDIRECT_RHO_ALL73={rho_all:.9f}",
        f"INDIRECT_RHO_ALL73_P={rho_all_p:.6g}",
        f"INDIRECT_RHO_PRIMARY_SIG={rho_primary_sig:.9f}",
        f"INDIRECT_RHO_PRIMARY_SIG_P={rho_primary_sig_p:.6g}",
        f"INDIRECT_SIGN_FLIPS_ALL73={sign_flips_all}",
        f"INDIRECT_SIGN_FLIPS_PRIMARY_SIG={sign_flips_primary_sig}",
        "",
        "--- CONTEXT SUMMARY ---",
        context.to_string(index=False),
        "",
        "--- STRICT-RETAINED PRIMARY CONSISTENT PATHS ---",
        merged.loc[
            merged["primary_consistent_retained_strict"],
            KEY + [
                "indirect_effect_primary",
                "FDR_BH_within_score_outcome_primary",
                "indirect_effect_strict",
                "FDR_BH_within_score_outcome_strict",
                "highest_priority_primary",
                "strict_retained_highest_priority",
            ],
        ].to_string(index=False),
        "",
        "INTERPRETATION RULES:",
        "- Primary Step12b remains the primary inference set.",
        "- Strict retention is a sensitivity/robustness annotation only.",
        "- Strict-only gains are not promoted into the primary discovery set.",
        "- Exact path identity is Diet x CGM x full taxonomy string.",
        "- These remain cross-sectional mediation-style associations, not causal mediation.",
        "",
        f"PATH_TABLE={path_table}",
        f"CONTEXT_TABLE={context_table}",
        f"RETAINED={retained_table}",
        f"LOST={lost_table}",
        f"GAINED={gained_table}",
    ]
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest.write_text(
        json.dumps({
            "status": "completed",
            "models_refit": False,
            "bootstrap_rerun": False,
            "exact_73_path_set_match": True,
            "primary_significant": primary_sig,
            "strict_significant": strict_sig,
            "primary_sig_retained_strict": retained,
            "primary_sig_lost_strict": lost,
            "strict_sig_gained_vs_primary": gained,
            "primary_consistent_retained_strict": retained_cons,
            "indirect_rho_all73": rho_all,
            "indirect_sign_flips_all73": sign_flips_all,
            "outputs": {
                "path_table": str(path_table),
                "context_table": str(context_table),
                "retained": str(retained_table),
                "lost": str(lost_table),
                "gained": str(gained_table),
                "summary": str(summary_txt),
            },
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
