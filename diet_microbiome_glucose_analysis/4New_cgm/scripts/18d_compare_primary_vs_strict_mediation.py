#!/usr/bin/env python3
"""
Step 18d — Consolidate Paper-20 PRIMARY vs STRICT mediation robustness.

Read-only comparison of completed Step18b primary and Step18c strict mediation
runs. No models are refit, no bootstrap is rerun, and no FDR is recalculated.
Primary discovery remains Step18b. Strict results are sensitivity/robustness
annotations only.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
DG = ROOT / "diet_microbiome_glucose_analysis"
BRANCH_CANDIDATES = [DG / "4New_cgm", ROOT / "4New_cgm"]

def find_branch(explicit=None):
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(p)
        return p
    for p in BRANCH_CANDIDATES:
        if p.is_dir():
            return p
    raise FileNotFoundError("4New_cgm branch not found")

def latest_dir(parent: Path, pattern: str, rel: str):
    found = []
    for p in parent.glob(pattern):
        f = p / rel
        if f.is_file():
            found.append((f.stat().st_mtime, p))
    if not found:
        raise FileNotFoundError(f"No {pattern} containing {rel}")
    found.sort(reverse=True)
    return found[0][1]

def truthy(s):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).eq(1)
    return s.astype(str).str.strip().str.lower().isin(
        {"true", "1", "1.0", "yes", "y", "t"}
    )

def safe_spearman(a, b):
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    ok = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or x[ok].nunique() < 2 or y[ok].nunique() < 2:
        return np.nan
    return float(x[ok].corr(y[ok], method="spearman"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch-dir", default=None)
    ap.add_argument("--primary-dir", default=None)
    ap.add_argument("--strict-dir", default=None)
    args = ap.parse_args()

    branch = find_branch(args.branch_dir)
    outputs = branch / "outputs"

    primary_dir = (
        Path(args.primary_dir).expanduser().resolve()
        if args.primary_dir
        else latest_dir(
            outputs, "paper20_mediation_*_primary",
            "models/18b_primary_mediation_paths.csv"
        )
    )
    strict_dir = (
        Path(args.strict_dir).expanduser().resolve()
        if args.strict_dir
        else latest_dir(
            outputs, "paper20_mediation_*_strict",
            "models/18c_strict_mediation_paths.csv"
        )
    )

    pfile = primary_dir / "models" / "18b_primary_mediation_paths.csv"
    sfile = strict_dir / "models" / "18c_strict_mediation_paths.csv"
    if not pfile.is_file():
        raise FileNotFoundError(pfile)
    if not sfile.is_file():
        raise FileNotFoundError(sfile)

    p = pd.read_csv(pfile, low_memory=False)
    s = pd.read_csv(sfile, low_memory=False)

    keys = ["exposure", "outcome_field", "species"]
    for label, df in [("primary", p), ("strict", s)]:
        missing = [c for c in keys if c not in df.columns]
        if missing:
            raise RuntimeError(f"{label} missing keys: {missing}")
        if df.duplicated(keys).any():
            raise RuntimeError(f"{label} contains duplicate path keys")

    pkeys = set(map(tuple, p[keys].to_numpy()))
    skeys = set(map(tuple, s[keys].to_numpy()))
    exact_match = pkeys == skeys
    if not exact_match:
        raise RuntimeError(
            f"Primary/strict path mismatch: primary={len(pkeys)} "
            f"strict={len(skeys)} primary_only={len(pkeys-skeys)} "
            f"strict_only={len(skeys-pkeys)}"
        )

    keep = [
        *keys, "role", "outcome_label", "N", "sample_sha256",
        "indirect_effect", "direct_effect", "total_effect",
        "bootstrap_p_indirect", "FDR_BH_within_score_outcome",
        "FDR_BH_global_all_mediation", "mediation_FDR05_primary",
        "candidate_mediator_consistent",
        "bridge_direction_consistent_mediator",
        "highest_priority_bridge_candidate",
        "highest_priority_significant_mediation",
        "highest_priority_direction_consistent_mediation",
        "mediated_proportion", "mediated_proportion_outside_0_1",
        "proportion_potentially_unstable",
    ]
    pkeep = [c for c in keep if c in p.columns]
    skeep = [c for c in keep if c in s.columns]

    m = p[pkeep].merge(
        s[skeep], on=keys, how="inner", validate="one_to_one",
        suffixes=("_primary", "_strict")
    )

    role_col = "role_primary" if "role_primary" in m.columns else "role_strict"
    label_col = "outcome_label_primary" if "outcome_label_primary" in m.columns else "outcome_label_strict"
    m["role"] = m[role_col]
    m["outcome_label"] = m[label_col]

    p_sig = truthy(m["mediation_FDR05_primary_primary"])
    s_sig = truthy(m["mediation_FDR05_primary_strict"])
    p_cons = truthy(m["bridge_direction_consistent_mediator_primary"])
    s_cons = truthy(m["bridge_direction_consistent_mediator_strict"])

    m["primary_significant"] = p_sig
    m["strict_significant"] = s_sig
    m["primary_consistent"] = p_cons
    m["strict_consistent"] = s_cons

    m["significance_transition"] = np.select(
        [p_sig & s_sig, p_sig & ~s_sig, ~p_sig & s_sig],
        ["retained", "lost_in_strict", "gained_in_strict"],
        default="nonsignificant_both",
    )
    m["consistency_transition"] = np.select(
        [p_cons & s_cons, p_cons & ~s_cons, ~p_cons & s_cons],
        ["retained", "lost_in_strict", "gained_in_strict"],
        default="nonconsistent_both",
    )

    p_ind = pd.to_numeric(m["indirect_effect_primary"], errors="coerce")
    s_ind = pd.to_numeric(m["indirect_effect_strict"], errors="coerce")
    m["indirect_sign_flip"] = (
        p_ind.notna() & s_ind.notna() & (np.sign(p_ind) != np.sign(s_ind))
    )
    m["N_lost_strict"] = (
        pd.to_numeric(m["N_primary"], errors="coerce")
        - pd.to_numeric(m["N_strict"], errors="coerce")
    )

    hp_col = "highest_priority_bridge_candidate_primary"
    m["highest_priority"] = truthy(m[hp_col]) if hp_col in m.columns else False

    primary_sig_n = int(p_sig.sum())
    strict_sig_n = int(s_sig.sum())
    retained_sig_n = int((p_sig & s_sig).sum())
    lost_sig_n = int((p_sig & ~s_sig).sum())
    gained_sig_n = int((~p_sig & s_sig).sum())

    primary_cons_n = int(p_cons.sum())
    strict_cons_n = int(s_cons.sum())
    retained_cons_n = int((p_cons & s_cons).sum())
    lost_cons_n = int((p_cons & ~s_cons).sum())
    gained_cons_n = int((~p_cons & s_cons).sum())

    rho_all = safe_spearman(p_ind, s_ind)
    rho_primary_sig = safe_spearman(p_ind[p_sig], s_ind[p_sig])
    sign_flips_all = int(m["indirect_sign_flip"].sum())
    sign_flips_primary_sig = int(m.loc[p_sig, "indirect_sign_flip"].sum())

    union_sig = int((p_sig | s_sig).sum())
    jaccard_sig = retained_sig_n / union_sig if union_sig else np.nan

    context = (
        m.groupby(
            ["exposure", "role", "outcome_field", "outcome_label"],
            as_index=False, dropna=False
        )
        .agg(
            paths_tested=("species", "size"),
            primary_sig=("primary_significant", "sum"),
            strict_sig=("strict_significant", "sum"),
            retained_sig=("significance_transition", lambda x: int((x == "retained").sum())),
            lost_sig=("significance_transition", lambda x: int((x == "lost_in_strict").sum())),
            gained_sig=("significance_transition", lambda x: int((x == "gained_in_strict").sum())),
            primary_consistent=("primary_consistent", "sum"),
            strict_consistent=("strict_consistent", "sum"),
            retained_consistent=("consistency_transition", lambda x: int((x == "retained").sum())),
            sign_flips=("indirect_sign_flip", "sum"),
            median_N_primary=("N_primary", "median"),
            median_N_strict=("N_strict", "median"),
        )
        .sort_values(["exposure", "outcome_field"])
    )
    context["primary_sig_retention_fraction"] = np.where(
        context["primary_sig"] > 0,
        context["retained_sig"] / context["primary_sig"],
        np.nan
    )

    exposure = (
        m.groupby(["exposure", "role"], as_index=False, dropna=False)
        .agg(
            paths_tested=("species", "size"),
            primary_sig=("primary_significant", "sum"),
            strict_sig=("strict_significant", "sum"),
            retained_sig=("significance_transition", lambda x: int((x == "retained").sum())),
            lost_sig=("significance_transition", lambda x: int((x == "lost_in_strict").sum())),
            gained_sig=("significance_transition", lambda x: int((x == "gained_in_strict").sum())),
            primary_consistent=("primary_consistent", "sum"),
            strict_consistent=("strict_consistent", "sum"),
            retained_consistent=("consistency_transition", lambda x: int((x == "retained").sum())),
            sign_flips=("indirect_sign_flip", "sum"),
            outcomes=("outcome_field", "nunique"),
            unique_species=("species", "nunique"),
        )
        .sort_values("exposure")
    )
    exposure["primary_sig_retention_fraction"] = np.where(
        exposure["primary_sig"] > 0,
        exposure["retained_sig"] / exposure["primary_sig"],
        np.nan
    )

    retained = m.loc[p_cons & s_cons].copy()
    if retained.empty:
        species = pd.DataFrame()
    else:
        species = (
            retained.groupby("species", as_index=False)
            .agg(
                retained_paths=("species", "size"),
                diets=("exposure", "nunique"),
                outcomes=("outcome_field", "nunique"),
                exposure_list=("exposure", lambda x: ";".join(sorted(set(map(str, x))))),
                outcome_list=("outcome_field", lambda x: ";".join(sorted(set(map(str, x))))),
                highest_priority_retained_paths=("highest_priority", "sum"),
            )
            .sort_values(
                ["retained_paths", "diets", "outcomes", "species"],
                ascending=[False, False, False, True]
            )
        )

    unstable_p = truthy(m["proportion_potentially_unstable_primary"]) if "proportion_potentially_unstable_primary" in m.columns else pd.Series(False, index=m.index)
    unstable_s = truthy(m["proportion_potentially_unstable_strict"]) if "proportion_potentially_unstable_strict" in m.columns else pd.Series(False, index=m.index)

    out = outputs / "paper20_mediation_primary_strict_consolidation"
    out.mkdir(parents=True, exist_ok=True)
    reports = out / "reports"
    reports.mkdir(exist_ok=True)

    path_out = reports / "18d_primary_vs_strict_path_comparison.csv"
    ctx_out = reports / "18d_context_robustness_summary.csv"
    exp_out = reports / "18d_exposure_robustness_summary.csv"
    sp_out = reports / "18d_retained_species_recurrence.csv"
    txt_out = reports / "18d_primary_vs_strict_summary.txt"

    m.to_csv(path_out, index=False)
    context.to_csv(ctx_out, index=False)
    exposure.to_csv(exp_out, index=False)
    species.to_csv(sp_out, index=False)

    lines = [
        "=== STEP 18d PRIMARY vs STRICT MEDIATION CONSOLIDATION ===",
        f"PRIMARY_DIR={primary_dir}",
        f"STRICT_DIR={strict_dir}",
        "MODELS_REFIT=False",
        "BOOTSTRAP_RERUN=False",
        "FDR_RECALCULATED=False",
        f"EXACT_1490_PATH_SET_MATCH={exact_match}",
        f"PATHS_COMPARED={len(m)}",
        "",
        f"PRIMARY_SIGNIFICANT={primary_sig_n}",
        f"STRICT_SIGNIFICANT={strict_sig_n}",
        f"PRIMARY_SIGNIFICANT_RETAINED={retained_sig_n}",
        f"PRIMARY_SIGNIFICANT_LOST={lost_sig_n}",
        f"STRICT_SIGNIFICANT_GAINED={gained_sig_n}",
        f"SIGNIFICANT_SET_JACCARD={jaccard_sig:.6f}",
        f"PRIMARY_SIG_RETENTION_FRACTION={(retained_sig_n/primary_sig_n if primary_sig_n else np.nan):.6f}",
        "",
        f"PRIMARY_CONSISTENT={primary_cons_n}",
        f"STRICT_CONSISTENT={strict_cons_n}",
        f"PRIMARY_CONSISTENT_RETAINED={retained_cons_n}",
        f"PRIMARY_CONSISTENT_LOST={lost_cons_n}",
        f"STRICT_CONSISTENT_GAINED={gained_cons_n}",
        "",
        f"INDIRECT_RHO_ALL1490={rho_all:.9f}",
        f"INDIRECT_RHO_PRIMARY_SIG={rho_primary_sig:.9f}",
        f"INDIRECT_SIGN_FLIPS_ALL1490={sign_flips_all}",
        f"INDIRECT_SIGN_FLIPS_PRIMARY_SIG={sign_flips_primary_sig}",
        "",
        f"PRIMARY_UNSTABLE_PROPORTION_N={int(unstable_p.sum())}",
        f"STRICT_UNSTABLE_PROPORTION_N={int(unstable_s.sum())}",
        "",
        "--- EXPOSURE ROBUSTNESS ---",
        exposure.to_string(index=False),
        "",
        "--- CONTEXT ROBUSTNESS ---",
        context.to_string(index=False),
        "",
        "--- TOP RETAINED RECURRENT SPECIES ---",
        ("NONE" if species.empty else species.head(30).to_string(index=False)),
        "",
        "INTERPRETATION RULES:",
        "- Step18b primary remains the discovery/inference set.",
        "- Step18c strict is a sensitivity analysis and does not redefine primary discovery.",
        "- retained means a primary FDR-significant path remains FDR-significant in strict on the identical frozen path set.",
        "- gained means strict-only significance; it is not promoted into primary discovery.",
        "- indirect-effect Spearman rho and sign flips assess effect-direction stability independent of FDR membership changes.",
        "- These remain cross-sectional mediation-style associations, not causal mediation.",
        "",
        f"PATH_TABLE={path_out}",
        f"CONTEXT_TABLE={ctx_out}",
        f"EXPOSURE_TABLE={exp_out}",
        f"SPECIES_TABLE={sp_out}",
        f"OUTPUT_DIR={out}",
    ]
    txt_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
