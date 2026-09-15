#!/usr/bin/env python3
"""Step 15f1 — consolidate primary vs strict mediation robustness.

Compares the exact frozen 45 new-diet mediation-style paths between:
- primary cohort
- strict known-nondiabetes/A10-negative cohort

No model is refit.

Outputs
-------
diet_microbiome_glucose_analysis/outputs/new_diet_extension/mediation/
    15f1_primary_vs_strict_path_comparison.csv
    15f1_primary_vs_strict_context_summary.csv
    15f1_primary_vs_strict_species_summary.csv
    15f1_primary_vs_strict_summary.txt
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
MED_DIR = (
    ROOT / "diet_microbiome_glucose_analysis" / "outputs"
    / "new_diet_extension" / "mediation"
)

PRIMARY = MED_DIR / "15f_primary_mediation_paths.csv"
STRICT = MED_DIR / "15f_strict_mediation_paths.csv"

PATH_OUT = MED_DIR / "15f1_primary_vs_strict_path_comparison.csv"
CONTEXT_OUT = MED_DIR / "15f1_primary_vs_strict_context_summary.csv"
SPECIES_OUT = MED_DIR / "15f1_primary_vs_strict_species_summary.csv"
SUMMARY_OUT = MED_DIR / "15f1_primary_vs_strict_summary.txt"

KEY = ["diet_score", "cgm_outcome", "species"]


def require(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


def truthy(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).eq(1)
    return (
        s.astype(str).str.strip().str.lower()
        .isin({"true", "1", "1.0", "yes", "y", "t"})
    )


def status(sig: bool, con: bool) -> str:
    if con:
        return "direction_consistent"
    if sig:
        return "significant_not_consistent"
    return "nonsignificant"


def main() -> int:
    require(PRIMARY, "primary mediation paths")
    require(STRICT, "strict mediation paths")

    p = pd.read_csv(PRIMARY, low_memory=False)
    s = pd.read_csv(STRICT, low_memory=False)

    required = {
        *KEY,
        "analysis_role",
        "N",
        "indirect_effect",
        "bootstrap_p_indirect",
        "FDR_BH_within_exposure_outcome",
        "mediation_FDR05_primary",
        "candidate_mediator_consistent",
    }
    for label, df in [("primary", p), ("strict", s)]:
        missing = sorted(required - set(df.columns))
        if missing:
            raise RuntimeError(f"{label} missing columns: {missing}")
        if df.duplicated(KEY).any():
            raise RuntimeError(f"{label} has duplicate path keys")
        if len(df) != 45:
            raise RuntimeError(f"{label}: expected 45 paths, got {len(df)}")

    p_keys = set(map(tuple, p[KEY].astype(str).to_numpy()))
    s_keys = set(map(tuple, s[KEY].astype(str).to_numpy()))
    exact_set = p_keys == s_keys
    if not exact_set:
        raise RuntimeError(
            f"Path-set mismatch: primary-only={len(p_keys-s_keys)}, "
            f"strict-only={len(s_keys-p_keys)}"
        )

    cols = KEY + [
        "analysis_role",
        "N",
        "a_diet_to_microbiome",
        "b_microbiome_to_cgm",
        "indirect_effect",
        "direct_effect",
        "total_effect",
        "bootstrap_p_indirect",
        "FDR_BH_within_exposure_outcome",
        "mediation_FDR05_primary",
        "candidate_mediator_consistent",
    ]
    cols = [c for c in cols if c in p.columns and c in s.columns]

    m = p[cols].merge(
        s[cols],
        on=KEY,
        how="inner",
        suffixes=("_primary", "_strict"),
        validate="one_to_one",
    )

    p_sig = truthy(m["mediation_FDR05_primary_primary"])
    s_sig = truthy(m["mediation_FDR05_primary_strict"])
    p_con = truthy(m["candidate_mediator_consistent_primary"])
    s_con = truthy(m["candidate_mediator_consistent_strict"])

    m["primary_status"] = [
        status(a, b) for a, b in zip(p_sig, p_con)
    ]
    m["strict_status"] = [
        status(a, b) for a, b in zip(s_sig, s_con)
    ]

    m["significant_retained"] = p_sig & s_sig
    m["significant_lost"] = p_sig & ~s_sig
    m["significant_gained"] = ~p_sig & s_sig

    m["consistent_retained"] = p_con & s_con
    m["consistent_lost"] = p_con & ~s_con
    m["consistent_gained"] = ~p_con & s_con

    p_ind = pd.to_numeric(m["indirect_effect_primary"], errors="coerce")
    s_ind = pd.to_numeric(m["indirect_effect_strict"], errors="coerce")
    m["indirect_same_sign"] = np.sign(p_ind) == np.sign(s_ind)
    m["indirect_abs_change"] = (s_ind - p_ind).abs()

    if "a_diet_to_microbiome_primary" in m.columns:
        m["a_same_sign"] = (
            np.sign(pd.to_numeric(m["a_diet_to_microbiome_primary"], errors="coerce"))
            == np.sign(pd.to_numeric(m["a_diet_to_microbiome_strict"], errors="coerce"))
        )
    if "b_microbiome_to_cgm_primary" in m.columns:
        m["b_same_sign"] = (
            np.sign(pd.to_numeric(m["b_microbiome_to_cgm_primary"], errors="coerce"))
            == np.sign(pd.to_numeric(m["b_microbiome_to_cgm_strict"], errors="coerce"))
        )

    m.to_csv(PATH_OUT, index=False)

    # Overall effect stability.
    rho_ind, _ = spearmanr(p_ind, s_ind)

    def context_row(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "paths_tested": len(g),
            "primary_significant": int(g["mediation_FDR05_primary_primary"].pipe(truthy).sum()),
            "strict_significant": int(g["mediation_FDR05_primary_strict"].pipe(truthy).sum()),
            "primary_consistent": int(g["candidate_mediator_consistent_primary"].pipe(truthy).sum()),
            "strict_consistent": int(g["candidate_mediator_consistent_strict"].pipe(truthy).sum()),
            "primary_significant_retained": int(g["significant_retained"].sum()),
            "primary_significant_lost": int(g["significant_lost"].sum()),
            "strict_significant_gained": int(g["significant_gained"].sum()),
            "primary_consistent_retained": int(g["consistent_retained"].sum()),
            "primary_consistent_lost": int(g["consistent_lost"].sum()),
            "strict_consistent_gained": int(g["consistent_gained"].sum()),
            "median_N_primary": float(pd.to_numeric(g["N_primary"]).median()),
            "median_N_strict": float(pd.to_numeric(g["N_strict"]).median()),
            "indirect_sign_flips": int((~g["indirect_same_sign"]).sum()),
            "indirect_rho": float(
                spearmanr(
                    pd.to_numeric(g["indirect_effect_primary"]),
                    pd.to_numeric(g["indirect_effect_strict"]),
                ).statistic
            ) if len(g) >= 3 else np.nan,
        })

    # Pandas compatibility: avoid include_groups= (not supported in older pandas).
    context_rows = []
    for (diet_score, cgm_outcome), g in m.groupby(
        ["diet_score", "cgm_outcome"], sort=True, dropna=False
    ):
        row = context_row(g).to_dict()
        row["diet_score"] = diet_score
        row["cgm_outcome"] = cgm_outcome
        context_rows.append(row)

    context = pd.DataFrame(context_rows)
    context = context[
        ["diet_score", "cgm_outcome"]
        + [c for c in context.columns if c not in {"diet_score", "cgm_outcome"}]
    ]
    context.to_csv(CONTEXT_OUT, index=False)

    # Species-level recurrence / robustness.
    species = (
        m.groupby("species", as_index=False)
        .agg(
            tested_paths=("species", "size"),
            primary_consistent_paths=("candidate_mediator_consistent_primary", lambda x: int(truthy(x).sum())),
            strict_consistent_paths=("candidate_mediator_consistent_strict", lambda x: int(truthy(x).sum())),
            consistent_retained_paths=("consistent_retained", "sum"),
            n_diet_exposures=("diet_score", "nunique"),
            n_cgm_outcomes=("cgm_outcome", "nunique"),
            diet_exposures=("diet_score", lambda x: ";".join(sorted(set(map(str, x))))),
            cgm_outcomes=("cgm_outcome", lambda x: ";".join(sorted(set(map(str, x))))),
        )
        .sort_values(
            ["consistent_retained_paths", "tested_paths", "species"],
            ascending=[False, False, True],
        )
    )
    species.to_csv(SPECIES_OUT, index=False)

    transitions = (
        m.groupby(["primary_status", "strict_status"], as_index=False)
        .size()
        .rename(columns={"size": "N"})
    )

    # Per-exposure totals.
    exposure_summary = (
        m.groupby("diet_score", as_index=False)
        .agg(
            paths_tested=("species", "size"),
            primary_significant=("mediation_FDR05_primary_primary", lambda x: int(truthy(x).sum())),
            strict_significant=("mediation_FDR05_primary_strict", lambda x: int(truthy(x).sum())),
            primary_consistent=("candidate_mediator_consistent_primary", lambda x: int(truthy(x).sum())),
            strict_consistent=("candidate_mediator_consistent_strict", lambda x: int(truthy(x).sum())),
            primary_consistent_retained=("consistent_retained", "sum"),
            primary_consistent_lost=("consistent_lost", "sum"),
            strict_consistent_gained=("consistent_gained", "sum"),
        )
    )

    lines = [
        "=== STEP 15f1 PRIMARY vs STRICT MEDIATION ROBUSTNESS ===",
        f"EXACT_45_PATH_SET_MATCH={exact_set}",
        f"PRIMARY_SIGNIFICANT={int(p_sig.sum())}",
        f"STRICT_SIGNIFICANT={int(s_sig.sum())}",
        f"PRIMARY_SIGNIFICANT_RETAINED={int((p_sig & s_sig).sum())}/{int(p_sig.sum())}",
        f"PRIMARY_SIGNIFICANT_LOST={int((p_sig & ~s_sig).sum())}",
        f"STRICT_SIGNIFICANT_GAINED={int((~p_sig & s_sig).sum())}",
        f"PRIMARY_CONSISTENT={int(p_con.sum())}",
        f"STRICT_CONSISTENT={int(s_con.sum())}",
        f"PRIMARY_CONSISTENT_RETAINED={int((p_con & s_con).sum())}/{int(p_con.sum())}",
        f"PRIMARY_CONSISTENT_LOST={int((p_con & ~s_con).sum())}",
        f"STRICT_CONSISTENT_GAINED={int((~p_con & s_con).sum())}",
        f"INDIRECT_RHO_PRIMARY_VS_STRICT={rho_ind:.9f}",
        f"INDIRECT_SIGN_FLIPS={int((~m['indirect_same_sign']).sum())}",
        "",
        "--- EXPOSURE SUMMARY ---",
        exposure_summary.to_string(index=False),
        "",
        "--- CONTEXT SUMMARY ---",
        context.to_string(index=False),
        "",
        "--- STATUS TRANSITIONS ---",
        transitions.to_string(index=False),
        "",
        "--- LOST PRIMARY-CONSISTENT PATHS ---",
        (
            m.loc[
                m["consistent_lost"],
                [
                    "diet_score",
                    "cgm_outcome",
                    "species",
                    "indirect_effect_primary",
                    "indirect_effect_strict",
                    "FDR_BH_within_exposure_outcome_primary",
                    "FDR_BH_within_exposure_outcome_strict",
                ],
            ].to_string(index=False)
            if int(m["consistent_lost"].sum()) else "NONE"
        ),
        "",
        "--- STRICT-GAINED CONSISTENT PATHS ---",
        (
            m.loc[
                m["consistent_gained"],
                [
                    "diet_score",
                    "cgm_outcome",
                    "species",
                    "indirect_effect_primary",
                    "indirect_effect_strict",
                    "FDR_BH_within_exposure_outcome_primary",
                    "FDR_BH_within_exposure_outcome_strict",
                ],
            ].to_string(index=False)
            if int(m["consistent_gained"].sum()) else "NONE"
        ),
        "",
        f"PATH_COMPARISON={PATH_OUT}",
        f"CONTEXT_SUMMARY={CONTEXT_OUT}",
        f"SPECIES_SUMMARY={SPECIES_OUT}",
    ]

    SUMMARY_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
