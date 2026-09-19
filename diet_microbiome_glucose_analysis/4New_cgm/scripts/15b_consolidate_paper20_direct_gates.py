#!/usr/bin/env python3
"""
Step 15b — consolidate Paper-20 Diet->CGM direct-association gates.

READ/COMPARE ONLY:
- no model refit
- no microbiome model
- no bridge screen
- no new FDR calculation

This consumes the completed Step15 run and applies the already-frozen
Step14b interpretation rules at the exact Diet x CGM context level.

Primary bridge-eligible context:
    1) Primary Model3 role-family BH-FDR < 0.05
    2) Same context remains role-FDR significant in strict Model3
    3) Primary and strict beta have the same sign
    4) Primary Model3 HC3 role-FDR < 0.05
    5) Special-outcome sensitivity rule passes:
         TAR140/HBGI/GRADE:
             no FDR-significant sensitivity component contradicts OLS
         TAR180:
             at least one FDR-significant sensitivity component is
             direction-concordant AND none is significantly opposite
             (OLS alone can never open the gate)

Highest-robustness annotation additionally requires:
    - strict Model3 HC3 role-FDR < 0.05
    - the same special-outcome rule in strict

Mean glucose and GMI are both retained but tagged as a dependent pair;
they are not independent replication.

Sensitivity-only TAR180 signals are reported but are NOT promoted into the
primary bridge set unless the primary OLS gate also exists.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
BRANCH_CANDIDATES = [
    ROOT / "diet_microbiome_glucose_analysis" / "4New_cgm",
    ROOT / "4New_cgm",
]


def truthy(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).eq(1)
    return (
        s.astype(str).str.strip().str.lower()
        .isin({"true", "1", "1.0", "yes", "y", "t"})
    )


def branch_dir(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(p)
        return p
    for p in BRANCH_CANDIDATES:
        if p.is_dir():
            return p
    raise FileNotFoundError("4New_cgm branch not found")


def latest_step15(branch: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(p)
        return p
    xs = []
    for p in (branch / "outputs").glob("paper20_diet_cgm_*"):
        f = p / "models" / "15_paper20_diet_cgm_all_models.csv"
        s = p / "models" / "15_paper20_distribution_sensitivities.csv"
        if f.is_file() and s.is_file():
            xs.append((max(f.stat().st_mtime, s.stat().st_mtime), p))
    if not xs:
        raise FileNotFoundError("No completed Step15 paper20_diet_cgm_* run found")
    xs.sort(reverse=True)
    return xs[0][1]


def sensitivity_aggregate(sens: pd.DataFrame, analysis_set: str) -> pd.DataFrame:
    x = sens.loc[
        sens["analysis_set"].eq(analysis_set)
        & sens["model"].eq(3)
    ].copy()

    x["computed"] = x["status"].eq("computed")
    x["sig"] = truthy(x["significant_role_05"])
    x["same"] = truthy(x["same_direction_as_OLS"])

    x["sig_same"] = x["computed"] & x["sig"] & x["same"]
    x["sig_opposite"] = x["computed"] & x["sig"] & ~x["same"]
    x["computed_same"] = x["computed"] & x["same"]

    agg = (
        x.groupby(
            ["exposure", "role", "source_outcome_field"],
            as_index=False,
        )
        .agg(
            sensitivity_components=("sensitivity_id", "size"),
            sensitivity_computed=("computed", "sum"),
            sensitivity_direction_same=("computed_same", "sum"),
            sensitivity_significant_same=("sig_same", "sum"),
            sensitivity_significant_opposite=("sig_opposite", "sum"),
        )
    )
    return agg.rename(
        columns={
            "source_outcome_field": "outcome_field",
            **{
                c: c + f"_{analysis_set}"
                for c in [
                    "sensitivity_components",
                    "sensitivity_computed",
                    "sensitivity_direction_same",
                    "sensitivity_significant_same",
                    "sensitivity_significant_opposite",
                ]
            },
        }
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch-dir", default=None)
    ap.add_argument("--step15-dir", default=None)
    args = ap.parse_args()

    branch = branch_dir(args.branch_dir)
    run = latest_step15(branch, args.step15_dir)

    model_path = run / "models" / "15_paper20_diet_cgm_all_models.csv"
    sens_path = run / "models" / "15_paper20_distribution_sensitivities.csv"

    models = pd.read_csv(model_path, low_memory=False)
    sens = pd.read_csv(sens_path, low_memory=False)

    m3 = models.loc[models["model"].eq(3)].copy()
    primary = m3.loc[m3["analysis_set"].eq("primary")].copy()
    strict = m3.loc[m3["analysis_set"].eq("strict")].copy()

    if len(primary) != 140 or len(strict) != 140:
        raise RuntimeError(
            f"Expected 140 primary + 140 strict Model3 rows, got "
            f"{len(primary)} + {len(strict)}"
        )

    key = ["exposure", "role", "outcome_field", "outcome_label"]

    keep = key + [
        "N", "beta", "p_value", "FDR_role",
        "significant_role_05",
        "HC3_p_value", "HC3_FDR_role",
        "HC3_significant_role_05",
        "analysis_class",
    ]
    p = primary[keep].copy()
    s = strict[keep].copy()

    p = p.rename(columns={
        c: c + "_primary"
        for c in keep
        if c not in key
    })
    s = s.rename(columns={
        c: c + "_strict"
        for c in keep
        if c not in key
    })

    comp = p.merge(s, on=key, how="inner", validate="one_to_one")
    if len(comp) != 140:
        raise RuntimeError(f"Expected exact 140 matched contexts, got {len(comp)}")

    for c in [
        "significant_role_05_primary",
        "HC3_significant_role_05_primary",
        "significant_role_05_strict",
        "HC3_significant_role_05_strict",
    ]:
        comp[c] = truthy(comp[c])

    comp["same_beta_direction_primary_strict"] = (
        np.sign(pd.to_numeric(comp["beta_primary"], errors="coerce"))
        == np.sign(pd.to_numeric(comp["beta_strict"], errors="coerce"))
    )

    # Merge special-outcome sensitivity summaries.
    spa = sensitivity_aggregate(sens, "primary")
    ssa = sensitivity_aggregate(sens, "strict")
    comp = comp.merge(
        spa,
        on=["exposure", "role", "outcome_field"],
        how="left",
        validate="one_to_one",
    ).merge(
        ssa,
        on=["exposure", "role", "outcome_field"],
        how="left",
        validate="one_to_one",
    )

    sensitivity_cols = [
        c for c in comp.columns if c.startswith("sensitivity_")
    ]
    for c in sensitivity_cols:
        comp[c] = pd.to_numeric(comp[c], errors="coerce").fillna(0).astype(int)

    special = comp["outcome_field"].isin(
        ["cgm_above_140", "cgm_above_180", "cgm_hbgi", "cgm_grade"]
    )
    tar180 = comp["outcome_field"].eq("cgm_above_180")

    # Standard outcomes automatically pass the special check.
    comp["special_rule_primary_ok"] = True
    comp["special_rule_strict_ok"] = True

    # TAR140/HBGI/GRADE: block only if a statistically significant
    # sensitivity component points opposite to canonical OLS.
    non_tar180_special = special & ~tar180
    comp.loc[non_tar180_special, "special_rule_primary_ok"] = (
        comp.loc[
            non_tar180_special,
            "sensitivity_significant_opposite_primary",
        ].eq(0)
        & comp.loc[
            non_tar180_special,
            "sensitivity_computed_primary",
        ].gt(0)
    )
    comp.loc[non_tar180_special, "special_rule_strict_ok"] = (
        comp.loc[
            non_tar180_special,
            "sensitivity_significant_opposite_strict",
        ].eq(0)
        & comp.loc[
            non_tar180_special,
            "sensitivity_computed_strict",
        ].gt(0)
    )

    # TAR180 requires affirmative hurdle support; OLS alone cannot gate.
    comp.loc[tar180, "special_rule_primary_ok"] = (
        comp.loc[
            tar180,
            "sensitivity_significant_same_primary",
        ].ge(1)
        & comp.loc[
            tar180,
            "sensitivity_significant_opposite_primary",
        ].eq(0)
    )
    comp.loc[tar180, "special_rule_strict_ok"] = (
        comp.loc[
            tar180,
            "sensitivity_significant_same_strict",
        ].ge(1)
        & comp.loc[
            tar180,
            "sensitivity_significant_opposite_strict",
        ].eq(0)
    )

    comp["primary_direct_sig"] = comp["significant_role_05_primary"]
    comp["strict_direct_retained"] = (
        comp["significant_role_05_strict"]
        & comp["same_beta_direction_primary_strict"]
    )
    comp["primary_hc3_retained"] = comp["HC3_significant_role_05_primary"]
    comp["strict_hc3_retained"] = comp["HC3_significant_role_05_strict"]

    comp["bridge_context_eligible"] = (
        comp["primary_direct_sig"]
        & comp["strict_direct_retained"]
        & comp["primary_hc3_retained"]
        & comp["special_rule_primary_ok"]
    )

    comp["highest_robustness_context"] = (
        comp["bridge_context_eligible"]
        & comp["strict_hc3_retained"]
        & comp["special_rule_strict_ok"]
    )

    comp["tar180_sensitivity_only"] = (
        tar180
        & ~comp["primary_direct_sig"]
        & comp["special_rule_primary_ok"]
    )

    # Dependency annotation only; do not remove either outcome.
    comp["dependency_group"] = "independent_or_partial_overlap"
    comp.loc[
        comp["outcome_field"].isin(["cgm_mean", "cgm_gmi"]),
        "dependency_group",
    ] = "mean_glucose_GMI_near_deterministic_pair"

    # Reason for noneligibility.
    def reasons(row):
        out = []
        if not row["primary_direct_sig"]:
            out.append("primary_roleFDR_not_sig")
        if row["primary_direct_sig"] and not row["strict_direct_retained"]:
            out.append("not_retained_strict")
        if row["primary_direct_sig"] and not row["primary_hc3_retained"]:
            out.append("not_retained_HC3")
        if not row["special_rule_primary_ok"]:
            out.append("special_sensitivity_rule_failed")
        return ";".join(out) if out else "eligible"

    comp["bridge_gate_reason"] = comp.apply(reasons, axis=1)

    # Summaries.
    diet_summary = (
        comp.groupby(["exposure", "role"], as_index=False)
        .agg(
            contexts_tested=("outcome_field", "size"),
            primary_sig=("primary_direct_sig", "sum"),
            strict_retained=("strict_direct_retained", "sum"),
            hc3_retained=("primary_hc3_retained", "sum"),
            bridge_eligible=("bridge_context_eligible", "sum"),
            highest_robustness=("highest_robustness_context", "sum"),
        )
        .sort_values(["role", "exposure"])
    )

    outcome_summary = (
        comp.groupby(["outcome_field", "outcome_label"], as_index=False)
        .agg(
            diets_tested=("exposure", "size"),
            primary_sig=("primary_direct_sig", "sum"),
            strict_retained=("strict_direct_retained", "sum"),
            hc3_retained=("primary_hc3_retained", "sum"),
            bridge_eligible=("bridge_context_eligible", "sum"),
            highest_robustness=("highest_robustness_context", "sum"),
        )
        .sort_values(
            ["bridge_eligible", "highest_robustness", "outcome_field"],
            ascending=[False, False, True],
        )
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = branch / "outputs" / f"paper20_direct_gate_{stamp}"
    reports = out / "reports"
    reports.mkdir(parents=True, exist_ok=False)

    comp_path = reports / "15b_all140_context_gate_audit.csv"
    eligible_path = reports / "15b_bridge_eligible_contexts.csv"
    highest_path = reports / "15b_highest_robustness_contexts.csv"
    blocked_path = reports / "15b_primary_significant_but_not_bridge_eligible.csv"
    tar180_path = reports / "15b_tar180_sensitivity_only.csv"
    diet_path = reports / "15b_gate_summary_by_diet.csv"
    outcome_path = reports / "15b_gate_summary_by_outcome.csv"
    txt_path = reports / "15b_direct_gate_summary.txt"

    comp.to_csv(comp_path, index=False)
    comp.loc[comp["bridge_context_eligible"]].to_csv(
        eligible_path, index=False
    )
    comp.loc[comp["highest_robustness_context"]].to_csv(
        highest_path, index=False
    )
    comp.loc[
        comp["primary_direct_sig"] & ~comp["bridge_context_eligible"]
    ].to_csv(blocked_path, index=False)
    comp.loc[comp["tar180_sensitivity_only"]].to_csv(
        tar180_path, index=False
    )
    diet_summary.to_csv(diet_path, index=False)
    outcome_summary.to_csv(outcome_path, index=False)

    primary_sig = int(comp["primary_direct_sig"].sum())
    strict_ret = int(
        (comp["primary_direct_sig"] & comp["strict_direct_retained"]).sum()
    )
    hc3_ret = int(
        (comp["primary_direct_sig"] & comp["primary_hc3_retained"]).sum()
    )
    eligible_n = int(comp["bridge_context_eligible"].sum())
    highest_n = int(comp["highest_robustness_context"].sum())
    tar180_only = int(comp["tar180_sensitivity_only"].sum())
    flips = int((~comp["same_beta_direction_primary_strict"]).sum())

    lines = [
        "=== STEP 15b PAPER-20 DIRECT-GATE CONSOLIDATION ===",
        f"STEP15_SOURCE={run}",
        "MODELS_REFIT=False",
        "FDR_RECALCULATED=False",
        "MICROBIOME_MODELS_FIT=False",
        "BRIDGE_SCREEN_RUN=False",
        "",
        f"CONTEXTS_TESTED=140",
        f"PRIMARY_MODEL3_ROLE_FDR05={primary_sig}",
        f"PRIMARY_SIG_RETAINED_STRICT={strict_ret}",
        f"PRIMARY_SIG_RETAINED_HC3={hc3_ret}",
        f"BRIDGE_ELIGIBLE_CONTEXTS={eligible_n}",
        f"HIGHEST_ROBUSTNESS_CONTEXTS={highest_n}",
        f"TAR180_SENSITIVITY_ONLY_CONTEXTS={tar180_only}",
        f"BETA_DIRECTION_FLIPS_PRIMARY_VS_STRICT={flips}",
        "",
        "--- BY DIET ---",
        diet_summary.to_string(index=False),
        "",
        "--- BY OUTCOME ---",
        outcome_summary.to_string(index=False),
        "",
        "--- PRIMARY-SIGNIFICANT BUT BLOCKED FROM BRIDGE ---",
        (
            "NONE"
            if not (comp["primary_direct_sig"] & ~comp["bridge_context_eligible"]).any()
            else comp.loc[
                comp["primary_direct_sig"] & ~comp["bridge_context_eligible"],
                [
                    "exposure", "outcome_field", "outcome_label",
                    "FDR_role_primary", "FDR_role_strict",
                    "HC3_FDR_role_primary",
                    "bridge_gate_reason",
                ],
            ].to_string(index=False)
        ),
        "",
        "--- TAR180 SENSITIVITY-ONLY ---",
        (
            "NONE"
            if tar180_only == 0
            else comp.loc[
                comp["tar180_sensitivity_only"],
                [
                    "exposure", "outcome_field",
                    "sensitivity_significant_same_primary",
                    "sensitivity_significant_opposite_primary",
                ],
            ].to_string(index=False)
        ),
        "",
        "INTERPRETATION RULES:",
        "- Step15 primary Model3 role-FDR results remain the primary direct-association set.",
        "- Bridge eligibility is a robustness gate, not a redefinition of the primary results.",
        "- Strict-only or HC3-only gains are not promoted into primary discovery.",
        "- TAR180 sensitivity-only signals remain supplementary and do not automatically enter the bridge screen.",
        "- Mean glucose and GMI remain separate Paper-20 outcomes but are not independent replication.",
        "- Next microbiome->CGM MWAS may cover all 20 outcomes; bridge construction should use the eligible context table.",
        "",
        f"ALL140={comp_path}",
        f"ELIGIBLE={eligible_path}",
        f"HIGHEST={highest_path}",
        f"BLOCKED={blocked_path}",
        f"TAR180_ONLY={tar180_path}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    (reports / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "models_refit": False,
                "fdr_recalculated": False,
                "contexts_tested": 140,
                "primary_model3_role_fdr05": primary_sig,
                "bridge_eligible_contexts": eligible_n,
                "highest_robustness_contexts": highest_n,
                "tar180_sensitivity_only_contexts": tar180_only,
                "beta_direction_flips_primary_vs_strict": flips,
                "outputs": {
                    "all140": str(comp_path),
                    "eligible": str(eligible_path),
                    "highest": str(highest_path),
                    "blocked": str(blocked_path),
                    "tar180_only": str(tar180_path),
                    "diet_summary": str(diet_path),
                    "outcome_summary": str(outcome_path),
                    "summary": str(txt_path),
                },
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
