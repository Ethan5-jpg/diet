#!/usr/bin/env python3
"""
10c_primary_vs_strict_mediation_robustness.py

Path-by-path robustness comparison of the corrected PRIMARY Step-10 mediation
analysis versus the STRICT known-nondiabetes/non-A10 sensitivity analysis.

This script does NOT refit any mediation model. It compares the two completed
Step-10 result tables on the exact same Diet -> microbiome -> CGM paths.

Default inputs
--------------
outputs/models/10_mediation_paths_model3.csv
outputs/models/10_strict_mediation_paths_model3.csv

Main questions
--------------
1. Are the exact candidate path sets identical between PRIMARY and STRICT?
2. How many PRIMARY FDR-significant / direction-consistent paths survive STRICT?
3. Are a-path, b-path, total-effect and indirect-effect directions preserved?
4. How strongly are PRIMARY and STRICT effect estimates correlated?
5. Do the 12 PRIMARY AMED/hPDI shared glucose-CV core species remain supported?
6. Which paths are lost, gained, or show sign changes?

Outputs
-------
outputs/reports/10c_primary_vs_strict_path_comparison.csv
outputs/reports/10c_primary_vs_strict_effect_stability.csv
outputs/reports/10c_primary_vs_strict_transition_counts.csv
outputs/reports/10c_primary_shared_core_strict_robustness.csv
outputs/reports/10c_primary_vs_strict_summary.csv
outputs/reports/10c_primary_vs_strict_summary.txt

Dependencies
------------
numpy, pandas only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_PRIMARY = (
    PROJECT_ROOT / "outputs" / "models" / "10_mediation_paths_model3.csv"
)
DEFAULT_STRICT = (
    PROJECT_ROOT / "outputs" / "models" / "10_strict_mediation_paths_model3.csv"
)
DEFAULT_REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

KEYS = ["diet_score", "cgm_outcome", "species"]

BOOL_COLS = [
    "mediation_FDR05_primary",
    "candidate_mediator_consistent",
    "indirect_same_sign_as_total",
    "indirect_CI_excludes_zero",
]

NUMERIC_COLS = [
    "N",
    "a_diet_to_microbiome",
    "b_microbiome_to_cgm",
    "indirect_effect",
    "total_effect",
    "bootstrap_p_indirect",
    "FDR_BH_within_score_outcome",
    "indirect_CI95_lower",
    "indirect_CI95_upper",
    "mediated_proportion",
]


def truthy_series(s: pd.Series) -> pd.Series:
    """Parse bool / 0-1 / string booleans robustly."""
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)

    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        bad = x.notna() & ~x.isin([0, 1])
        if bad.any():
            examples = sorted(x.loc[bad].astype(str).unique().tolist())[:10]
            raise RuntimeError(
                f"Could not parse boolean numeric values in {s.name}: {examples}"
            )
        return x.fillna(0).eq(1)

    mapped = (
        s.astype("string")
        .str.strip()
        .str.lower()
        .replace(
            {
                "true": "1",
                "false": "0",
                "yes": "1",
                "no": "0",
                "y": "1",
                "n": "0",
                "1.0": "1",
                "0.0": "0",
            }
        )
    )
    bad = mapped.notna() & ~mapped.isin(["1", "0", "<na>", "nan", "none", ""])
    if bad.any():
        counts = mapped.loc[bad].value_counts(dropna=False).head(10).to_dict()
        raise RuntimeError(f"Could not parse boolean values in {s.name}: {counts}")
    return mapped.eq("1")


def ensure_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{label} is missing required columns: {', '.join(missing)}"
        )


def prepare(df: pd.DataFrame, label: str) -> pd.DataFrame:
    required = KEYS + BOOL_COLS + [
        "N",
        "a_diet_to_microbiome",
        "b_microbiome_to_cgm",
        "indirect_effect",
        "total_effect",
        "bootstrap_p_indirect",
        "FDR_BH_within_score_outcome",
        "indirect_CI95_lower",
        "indirect_CI95_upper",
    ]
    ensure_columns(df, required, label)

    out = df.copy()

    for c in BOOL_COLS:
        out[c] = truthy_series(out[c])

    for c in NUMERIC_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    for c in KEYS:
        out[c] = out[c].astype("string").str.strip()

    if out.duplicated(KEYS).any():
        dup = out.loc[out.duplicated(KEYS, keep=False), KEYS]
        raise RuntimeError(
            f"{label} has duplicate path keys. First duplicates:\n"
            + dup.head(20).to_string(index=False)
        )

    # Display label only; exact `species` remains the identity.
    if "species_label" not in out.columns:
        out["species_label"] = out["species"]
    else:
        out["species_label"] = (
            out["species_label"]
            .astype("string")
            .fillna(out["species"])
            .str.strip()
        )

    return out


def sign_series(x: pd.Series, tol: float = 1e-12) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    arr = np.where(
        x.isna(),
        np.nan,
        np.where(x > tol, 1.0, np.where(x < -tol, -1.0, 0.0)),
    )
    return pd.Series(arr, index=x.index)


def paired_spearman(x: pd.Series, y: pd.Series) -> tuple[int, float]:
    """Spearman rho using rank + Pearson, so scipy is not required."""
    a = pd.to_numeric(x, errors="coerce")
    b = pd.to_numeric(y, errors="coerce")
    ok = a.notna() & b.notna()
    n = int(ok.sum())
    if n < 3:
        return n, np.nan

    ar = a.loc[ok].rank(method="average")
    br = b.loc[ok].rank(method="average")
    if ar.nunique() < 2 or br.nunique() < 2:
        return n, np.nan
    return n, float(ar.corr(br, method="pearson"))


def pearson(x: pd.Series, y: pd.Series) -> tuple[int, float]:
    a = pd.to_numeric(x, errors="coerce")
    b = pd.to_numeric(y, errors="coerce")
    ok = a.notna() & b.notna()
    n = int(ok.sum())
    if n < 3 or a.loc[ok].nunique() < 2 or b.loc[ok].nunique() < 2:
        return n, np.nan
    return n, float(a.loc[ok].corr(b.loc[ok], method="pearson"))


def relative_change(primary: pd.Series, strict: pd.Series) -> pd.Series:
    p = pd.to_numeric(primary, errors="coerce")
    s = pd.to_numeric(strict, errors="coerce")
    denom = p.abs()
    return np.where(denom > 1e-12, (s - p) / denom, np.nan)


def classify_transition(row: pd.Series) -> str:
    p_sig = bool(row["primary_mediation_FDR05_primary"])
    s_sig = bool(row["strict_mediation_FDR05_primary"])
    p_cons = bool(row["primary_candidate_mediator_consistent"])
    s_cons = bool(row["strict_candidate_mediator_consistent"])
    sign_flip = bool(row["indirect_sign_flip"])

    if p_cons and s_cons:
        return "stable_direction_consistent"
    if p_cons and not s_cons:
        if not s_sig:
            return "primary_consistent_lost_FDR"
        if sign_flip:
            return "primary_consistent_strict_indirect_sign_flip"
        return "primary_consistent_became_inconsistent"
    if (not p_cons) and s_cons:
        return "gained_direction_consistent_in_strict"
    if p_sig and s_sig:
        return "stable_significant_not_direction_consistent"
    if p_sig and not s_sig:
        return "primary_significant_lost_FDR"
    if (not p_sig) and s_sig:
        return "gained_significant_in_strict"
    return "stable_not_significant"


def run_self_test() -> None:
    p = pd.DataFrame(
        {
            "diet_score": ["AMED", "hPDI", "AHEI"],
            "cgm_outcome": ["glucose_cv", "glucose_cv", "mean_glucose"],
            "species": ["s1", "s1", "s2"],
            "species_label": ["S1", "S1", "S2"],
            "N": [100, 100, 100],
            "a_diet_to_microbiome": [-0.2, -0.3, 0.1],
            "b_microbiome_to_cgm": [0.2, 0.2, 0.1],
            "indirect_effect": [-0.04, -0.06, 0.01],
            "total_effect": [-0.2, -0.3, -0.1],
            "bootstrap_p_indirect": [0.01, 0.01, 0.5],
            "FDR_BH_within_score_outcome": [0.02, 0.02, 0.5],
            "indirect_CI95_lower": [-0.06, -0.08, -0.01],
            "indirect_CI95_upper": [-0.02, -0.04, 0.03],
            "mediation_FDR05_primary": [1, 1, 0],
            "candidate_mediator_consistent": [1, 1, 0],
            "indirect_same_sign_as_total": [1, 1, 0],
            "indirect_CI_excludes_zero": [1, 1, 0],
        }
    )
    s = p.copy()
    s["N"] = [90, 90, 90]
    s.loc[1, "mediation_FDR05_primary"] = 0
    s.loc[1, "candidate_mediator_consistent"] = 0

    p = prepare(p, "PRIMARY")
    s = prepare(s, "STRICT")

    m = p.merge(s, on=KEYS, suffixes=("_primary", "_strict"), validate="one_to_one")
    assert len(m) == 3
    assert set(m["species"]) == {"s1", "s2"}

    print("SELF TEST: PASS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    ap.add_argument("--strict", type=Path, default=DEFAULT_STRICT)
    ap.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        run_self_test()
        return

    primary_path = args.primary.resolve()
    strict_path = args.strict.resolve()
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    if not primary_path.exists():
        raise FileNotFoundError(f"PRIMARY input not found: {primary_path}")
    if not strict_path.exists():
        raise FileNotFoundError(f"STRICT input not found: {strict_path}")

    primary = prepare(pd.read_csv(primary_path), "PRIMARY")
    strict = prepare(pd.read_csv(strict_path), "STRICT")

    pkeys = set(map(tuple, primary[KEYS].astype(str).to_numpy()))
    skeys = set(map(tuple, strict[KEYS].astype(str).to_numpy()))
    p_only = pkeys - skeys
    s_only = skeys - pkeys
    exact_path_set_match = (len(p_only) == 0 and len(s_only) == 0)

    # Outer merge so path-set problems are visible rather than silently dropped.
    merged = primary.merge(
        strict,
        on=KEYS,
        how="outer",
        suffixes=("_primary", "_strict"),
        indicator=True,
        validate="one_to_one",
    )

    merged["path_present_primary"] = merged["_merge"].isin(["both", "left_only"])
    merged["path_present_strict"] = merged["_merge"].isin(["both", "right_only"])

    # Prefer PRIMARY display/tier metadata, but preserve both if available.
    if "species_label_primary" in merged.columns:
        merged["species_label"] = merged["species_label_primary"]
        if "species_label_strict" in merged.columns:
            merged["species_label"] = merged["species_label"].fillna(
                merged["species_label_strict"]
            )
    else:
        merged["species_label"] = merged["species"]

    for c in ["priority_tier", "candidate_rank"]:
        pc = f"{c}_primary"
        sc = f"{c}_strict"
        if pc in merged.columns:
            merged[c] = merged[pc]
            if sc in merged.columns:
                merged[c] = merged[c].fillna(merged[sc])

    both = merged["_merge"].eq("both")

    # Sign and change diagnostics.
    for effect in [
        "a_diet_to_microbiome",
        "b_microbiome_to_cgm",
        "indirect_effect",
        "total_effect",
    ]:
        pc = f"{effect}_primary"
        sc = f"{effect}_strict"
        if pc in merged.columns and sc in merged.columns:
            ps = sign_series(merged[pc])
            ss = sign_series(merged[sc])
            merged[f"{effect}_same_sign"] = both & ps.eq(ss) & ps.notna() & ss.notna()
            merged[f"{effect}_sign_flip"] = (
                both & ps.mul(ss).lt(0) & ps.notna() & ss.notna()
            )
            merged[f"{effect}_absolute_change"] = merged[sc] - merged[pc]
            merged[f"{effect}_relative_change_vs_abs_primary"] = relative_change(
                merged[pc], merged[sc]
            )

    merged["indirect_sign_flip"] = merged.get(
        "indirect_effect_sign_flip", pd.Series(False, index=merged.index)
    ).fillna(False)

    # N loss.
    merged["N_change_strict_minus_primary"] = (
        pd.to_numeric(merged.get("N_strict"), errors="coerce")
        - pd.to_numeric(merged.get("N_primary"), errors="coerce")
    )
    merged["N_retained_fraction"] = (
        pd.to_numeric(merged.get("N_strict"), errors="coerce")
        / pd.to_numeric(merged.get("N_primary"), errors="coerce")
    )

    # Make boolean fields false for path-absent rows.
    for stem in BOOL_COLS:
        for prefix in ["primary", "strict"]:
            c = f"{stem}_{prefix}"
            # merge suffix order is source-column + suffix, not prefix + source-column
            alt = f"{prefix}_{stem}"
            if c in merged.columns and alt not in merged.columns:
                merged[alt] = merged[c].fillna(False).astype(bool)

    # Explicit aliases with readable prefix naming.
    for stem in BOOL_COLS:
        pc = f"{stem}_primary"
        sc = f"{stem}_strict"
        if pc in merged.columns:
            merged[f"primary_{stem}"] = merged[pc].fillna(False).astype(bool)
        if sc in merged.columns:
            merged[f"strict_{stem}"] = merged[sc].fillna(False).astype(bool)

    merged["transition_class"] = merged.apply(classify_transition, axis=1)

    # ------------------------------------------------------------------
    # Effect stability table.
    # ------------------------------------------------------------------
    stability_rows = []
    effect_pairs = [
        ("a_diet_to_microbiome", "a path"),
        ("b_microbiome_to_cgm", "b path"),
        ("indirect_effect", "indirect effect"),
        ("total_effect", "total effect"),
    ]

    paired = merged.loc[both].copy()

    for col, label in effect_pairs:
        x = paired[f"{col}_primary"]
        y = paired[f"{col}_strict"]
        n_s, rho = paired_spearman(x, y)
        n_p, r = pearson(x, y)

        sx = sign_series(x)
        sy = sign_series(y)
        valid_sign = sx.notna() & sy.notna()
        same_sign = (sx.loc[valid_sign] == sy.loc[valid_sign])
        sign_concordance = float(same_sign.mean()) if len(same_sign) else np.nan
        sign_flips = int((sx.loc[valid_sign] * sy.loc[valid_sign] < 0).sum())

        abs_change = (pd.to_numeric(y, errors="coerce") - pd.to_numeric(x, errors="coerce")).abs()
        rel = pd.Series(relative_change(x, y), index=x.index).abs()

        stability_rows.append(
            {
                "effect": label,
                "column": col,
                "N_paired": int(n_s),
                "spearman_rho": rho,
                "pearson_r": r if n_p == n_s else r,
                "sign_concordance_fraction": sign_concordance,
                "sign_flips": sign_flips,
                "median_absolute_change": float(np.nanmedian(abs_change)),
                "median_abs_relative_change_vs_abs_primary": float(np.nanmedian(rel)),
            }
        )

    stability = pd.DataFrame(stability_rows)

    # ------------------------------------------------------------------
    # Transition counts.
    # ------------------------------------------------------------------
    transition_counts = (
        merged["transition_class"]
        .value_counts(dropna=False)
        .rename_axis("transition_class")
        .reset_index(name="n_paths")
    )

    # ------------------------------------------------------------------
    # PRIMARY recurrent AMED/hPDI glucose-CV core and STRICT preservation.
    # ------------------------------------------------------------------
    p_cons_cv = primary.loc[
        primary["candidate_mediator_consistent"]
        & primary["cgm_outcome"].eq("glucose_cv")
        & primary["diet_score"].isin(["AMED", "hPDI"])
    ].copy()

    amed_species = set(
        p_cons_cv.loc[p_cons_cv["diet_score"].eq("AMED"), "species"].astype(str)
    )
    hpdi_species = set(
        p_cons_cv.loc[p_cons_cv["diet_score"].eq("hPDI"), "species"].astype(str)
    )
    primary_shared_core = sorted(amed_species & hpdi_species)

    core_rows = []
    for sp in primary_shared_core:
        label_rows = primary.loc[primary["species"].astype(str).eq(sp), "species_label"]
        label = str(label_rows.iloc[0]) if len(label_rows) else sp

        row = {
            "species": sp,
            "species_label": label,
            "primary_core_member": True,
        }

        strict_both_sig = True
        strict_both_cons = True
        all_indirect_same_sign = True

        for score in ["AMED", "hPDI"]:
            p_row = primary.loc[
                primary["diet_score"].eq(score)
                & primary["cgm_outcome"].eq("glucose_cv")
                & primary["species"].astype(str).eq(sp)
            ]
            s_row = strict.loc[
                strict["diet_score"].eq(score)
                & strict["cgm_outcome"].eq("glucose_cv")
                & strict["species"].astype(str).eq(sp)
            ]

            if len(p_row) != 1:
                raise RuntimeError(
                    f"PRIMARY core lookup expected one row for {score}/{sp}, got {len(p_row)}"
                )

            row[f"{score}_primary_FDR"] = float(
                p_row.iloc[0]["FDR_BH_within_score_outcome"]
            )
            row[f"{score}_primary_indirect"] = float(
                p_row.iloc[0]["indirect_effect"]
            )

            if len(s_row) == 1:
                s0 = s_row.iloc[0]
                row[f"{score}_strict_present"] = True
                row[f"{score}_strict_FDR"] = float(
                    s0["FDR_BH_within_score_outcome"]
                )
                row[f"{score}_strict_significant"] = bool(
                    s0["mediation_FDR05_primary"]
                )
                row[f"{score}_strict_consistent"] = bool(
                    s0["candidate_mediator_consistent"]
                )
                row[f"{score}_strict_indirect"] = float(s0["indirect_effect"])

                p_ind = float(p_row.iloc[0]["indirect_effect"])
                s_ind = float(s0["indirect_effect"])
                same = np.sign(p_ind) == np.sign(s_ind)
                row[f"{score}_indirect_same_sign_primary_strict"] = bool(same)

                strict_both_sig &= bool(s0["mediation_FDR05_primary"])
                strict_both_cons &= bool(s0["candidate_mediator_consistent"])
                all_indirect_same_sign &= bool(same)
            else:
                row[f"{score}_strict_present"] = False
                row[f"{score}_strict_FDR"] = np.nan
                row[f"{score}_strict_significant"] = False
                row[f"{score}_strict_consistent"] = False
                row[f"{score}_strict_indirect"] = np.nan
                row[f"{score}_indirect_same_sign_primary_strict"] = False
                strict_both_sig = False
                strict_both_cons = False
                all_indirect_same_sign = False

        row["strict_both_scores_significant"] = strict_both_sig
        row["strict_both_scores_direction_consistent"] = strict_both_cons
        row["indirect_same_sign_in_both_scores_primary_vs_strict"] = (
            all_indirect_same_sign
        )
        core_rows.append(row)

    core = pd.DataFrame(core_rows)
    if len(core):
        core = core.sort_values(
            [
                "strict_both_scores_direction_consistent",
                "strict_both_scores_significant",
                "species_label",
            ],
            ascending=[False, False, True],
        ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Overall summary.
    # ------------------------------------------------------------------
    p_sig = int(primary["mediation_FDR05_primary"].sum())
    s_sig = int(strict["mediation_FDR05_primary"].sum())
    p_cons = int(primary["candidate_mediator_consistent"].sum())
    s_cons = int(strict["candidate_mediator_consistent"].sum())

    p_cons_keys = set(
        map(
            tuple,
            primary.loc[primary["candidate_mediator_consistent"], KEYS]
            .astype(str)
            .to_numpy(),
        )
    )
    s_cons_keys = set(
        map(
            tuple,
            strict.loc[strict["candidate_mediator_consistent"], KEYS]
            .astype(str)
            .to_numpy(),
        )
    )
    p_sig_keys = set(
        map(
            tuple,
            primary.loc[primary["mediation_FDR05_primary"], KEYS]
            .astype(str)
            .to_numpy(),
        )
    )
    s_sig_keys = set(
        map(
            tuple,
            strict.loc[strict["mediation_FDR05_primary"], KEYS]
            .astype(str)
            .to_numpy(),
        )
    )

    primary_consistent_retained = len(p_cons_keys & s_cons_keys)
    primary_sig_retained = len(p_sig_keys & s_sig_keys)

    indirect_row = stability.loc[stability["column"].eq("indirect_effect")].iloc[0]

    core_n = len(core)
    core_both_sig = (
        int(core["strict_both_scores_significant"].sum()) if core_n else 0
    )
    core_both_cons = (
        int(core["strict_both_scores_direction_consistent"].sum()) if core_n else 0
    )
    core_same_sign = (
        int(
            core[
                "indirect_same_sign_in_both_scores_primary_vs_strict"
            ].sum()
        )
        if core_n
        else 0
    )

    summary = pd.DataFrame(
        [
            {
                "primary_paths": len(primary),
                "strict_paths": len(strict),
                "exact_path_set_match": exact_path_set_match,
                "primary_only_paths": len(p_only),
                "strict_only_paths": len(s_only),
                "primary_significant_FDR05": p_sig,
                "strict_significant_FDR05": s_sig,
                "primary_significant_retained_in_strict": primary_sig_retained,
                "primary_significant_retention_fraction": (
                    primary_sig_retained / p_sig if p_sig else np.nan
                ),
                "primary_direction_consistent": p_cons,
                "strict_direction_consistent": s_cons,
                "primary_consistent_retained_in_strict": primary_consistent_retained,
                "primary_consistent_retention_fraction": (
                    primary_consistent_retained / p_cons if p_cons else np.nan
                ),
                "indirect_effect_spearman_rho": indirect_row["spearman_rho"],
                "indirect_effect_sign_concordance_fraction": indirect_row[
                    "sign_concordance_fraction"
                ],
                "indirect_effect_sign_flips": int(indirect_row["sign_flips"]),
                "median_N_primary": float(np.nanmedian(primary["N"])),
                "median_N_strict": float(np.nanmedian(strict["N"])),
                "median_N_retained_fraction": float(
                    np.nanmedian(merged.loc[both, "N_retained_fraction"])
                ),
                "primary_AMED_hPDI_shared_glucose_cv_core_species": core_n,
                "core_species_strict_both_scores_significant": core_both_sig,
                "core_species_strict_both_scores_direction_consistent": core_both_cons,
                "core_species_indirect_same_sign_both_scores": core_same_sign,
            }
        ]
    )

    # Sort path comparison for inspection.
    class_order = {
        "stable_direction_consistent": 1,
        "primary_consistent_lost_FDR": 2,
        "primary_consistent_became_inconsistent": 3,
        "primary_consistent_strict_indirect_sign_flip": 4,
        "gained_direction_consistent_in_strict": 5,
        "stable_significant_not_direction_consistent": 6,
        "primary_significant_lost_FDR": 7,
        "gained_significant_in_strict": 8,
        "stable_not_significant": 9,
    }
    merged["_class_order"] = merged["transition_class"].map(class_order).fillna(99)
    merged = merged.sort_values(
        [
            "_class_order",
            "diet_score",
            "cgm_outcome",
            "FDR_BH_within_score_outcome_primary",
            "species_label",
        ],
        ascending=[True, True, True, True, True],
    ).drop(columns=["_class_order", "_merge"])

    # ------------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------------
    out_paths = report_dir / "10c_primary_vs_strict_path_comparison.csv"
    out_stability = report_dir / "10c_primary_vs_strict_effect_stability.csv"
    out_transitions = report_dir / "10c_primary_vs_strict_transition_counts.csv"
    out_core = report_dir / "10c_primary_shared_core_strict_robustness.csv"
    out_summary = report_dir / "10c_primary_vs_strict_summary.csv"
    out_txt = report_dir / "10c_primary_vs_strict_summary.txt"

    merged.to_csv(out_paths, index=False)
    stability.to_csv(out_stability, index=False)
    transition_counts.to_csv(out_transitions, index=False)
    core.to_csv(out_core, index=False)
    summary.to_csv(out_summary, index=False)

    # ------------------------------------------------------------------
    # Console / text report.
    # ------------------------------------------------------------------
    lines = []
    lines.append("=" * 104)
    lines.append("10c — PRIMARY VS STRICT MEDIATION ROBUSTNESS")
    lines.append("=" * 104)
    lines.append(f"PRIMARY: {primary_path}")
    lines.append(f"STRICT : {strict_path}")
    lines.append("")

    lines.append("PATH-SET QC")
    lines.append("-" * 104)
    lines.append(f"PRIMARY paths: {len(primary)}")
    lines.append(f"STRICT paths : {len(strict)}")
    lines.append(f"Exact path-set match: {exact_path_set_match}")
    lines.append(f"PRIMARY-only paths: {len(p_only)}")
    lines.append(f"STRICT-only paths : {len(s_only)}")
    if p_only:
        lines.append("PRIMARY-only examples:")
        for x in sorted(p_only)[:10]:
            lines.append("  " + " | ".join(x))
    if s_only:
        lines.append("STRICT-only examples:")
        for x in sorted(s_only)[:10]:
            lines.append("  " + " | ".join(x))
    lines.append("")

    lines.append("SIGNIFICANCE / CANDIDATE RETENTION")
    lines.append("-" * 104)
    lines.append(
        f"PRIMARY significant FDR<0.05: {p_sig} | STRICT: {s_sig} | "
        f"PRIMARY retained: {primary_sig_retained}/{p_sig}"
    )
    lines.append(
        f"PRIMARY direction-consistent: {p_cons} | STRICT: {s_cons} | "
        f"PRIMARY retained: {primary_consistent_retained}/{p_cons}"
    )
    lines.append("")
    lines.append("TRANSITION CLASSES")
    lines.append(transition_counts.to_string(index=False))
    lines.append("")

    lines.append("EFFECT-STABILITY QC")
    lines.append("-" * 104)
    lines.append(stability.to_string(index=False))
    lines.append("")

    lines.append("PRIMARY AMED/hPDI SHARED GLUCOSE-CV CORE")
    lines.append("-" * 104)
    lines.append(f"PRIMARY shared core species: {core_n}")
    lines.append(
        "STRICT significant in BOTH AMED and hPDI paths: "
        f"{core_both_sig}/{core_n}"
    )
    lines.append(
        "STRICT direction-consistent in BOTH AMED and hPDI paths: "
        f"{core_both_cons}/{core_n}"
    )
    lines.append(
        "Indirect effect same sign PRIMARY vs STRICT in BOTH score paths: "
        f"{core_same_sign}/{core_n}"
    )
    if core_n:
        show_cols = [
            "species_label",
            "AMED_strict_FDR",
            "hPDI_strict_FDR",
            "strict_both_scores_significant",
            "strict_both_scores_direction_consistent",
            "indirect_same_sign_in_both_scores_primary_vs_strict",
        ]
        lines.append("")
        lines.append(core[show_cols].to_string(index=False))
    lines.append("")

    lines.append("INTERPRETATION")
    lines.append("-" * 104)
    lines.append(
        "The STRICT analysis is a fixed-candidate-set sensitivity analysis: "
        "the Step-09/Model-3 candidate gate is held fixed and mediation is re-estimated "
        "after excluding participants with unknown diabetes/A10 status."
    )
    lines.append(
        "Therefore robustness should be judged by paired path retention, effect-size "
        "stability, sign concordance and core-species preservation, not only by raw "
        "counts of FDR-significant paths."
    )
    lines.append(
        "This remains a cross-sectional mediation-style observational analysis; "
        "these regressions do not establish temporal or causal mediation."
    )

    txt = "\n".join(lines)
    out_txt.write_text(txt + "\n", encoding="utf-8")

    print(txt)
    print()
    print("=" * 104)
    print("SAVED")
    print("=" * 104)
    for p in [
        out_paths,
        out_stability,
        out_transitions,
        out_core,
        out_summary,
        out_txt,
    ]:
        print(p)


if __name__ == "__main__":
    main()
