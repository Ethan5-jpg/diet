#!/usr/bin/env python3
"""
10b_mediation_qc_and_recurrence.py

QC and recurrence consolidation for Step 10 mediation-style results.

Purpose
-------
This script does NOT refit mediation models. It audits the completed Step 10
results and answers the questions needed before strict-sensitivity analyses:

1. Why did significant paths fail/pass the direction-consistency criterion?
2. Are selected paths really Tier-1 recurrent bridge candidates?
3. Which species recur across diet scores / CGM outcomes?
4. Which glucose-CV mediators are shared between AMED and hPDI?
5. Are mediated proportions numerically unstable or outside [0, 1]?
6. Are there bootstrap-FDR / bootstrap-CI discrepancies?

Input
-----
outputs/models/10_mediation_paths_model3.csv

Outputs
-------
outputs/reports/10b_mediation_path_audit.csv
outputs/reports/10b_mediation_priority_tier_qc.csv
outputs/reports/10b_mediation_species_recurrence.csv
outputs/reports/10b_glucose_cv_amed_hpdi_overlap.csv
outputs/reports/10b_mediation_proportion_qc.csv
outputs/reports/10b_mediation_qc_summary.csv
outputs/reports/10b_mediation_qc_summary.txt

Dependencies
------------
numpy, pandas only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "models" / "10_mediation_paths_model3.csv"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"


def truthy_series(s: pd.Series) -> pd.Series:
    """Robust bool parser accepting bool, 0/1, 0.0/1.0, and strings."""
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)

    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="coerce")
        bad = vals.notna() & ~vals.isin([0, 1])
        if bad.any():
            examples = sorted(vals.loc[bad].astype(str).unique().tolist())[:10]
            raise RuntimeError(
                f"Could not parse boolean numeric values in {s.name}: {examples}"
            )
        return vals.fillna(0).eq(1)

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
        raise RuntimeError(
            f"Could not parse boolean values in {s.name}: {counts}"
        )
    return mapped.eq("1")


def ensure_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            "Required columns are missing from Step 10 results: "
            + ", ".join(missing)
        )


def clean_species_label(row: pd.Series) -> str:
    label = row.get("species_label", np.nan)
    if pd.notna(label) and str(label).strip():
        return str(label).strip()

    species = str(row["species"])
    # Preserve exact taxonomy in `species`; this is display-only fallback.
    if "|s__" in species:
        return species.split("|s__", 1)[1]
    if "s__" in species:
        return species.rsplit("s__", 1)[-1]
    return species


def classify_path(row: pd.Series) -> str:
    if not bool(row["mediation_FDR05_primary"]):
        return "not_significant_primary_FDR"

    if bool(row["candidate_mediator_consistent"]):
        return "significant_direction_consistent_candidate"

    same_sign = bool(row["indirect_same_sign_as_total"])
    ci_excludes = bool(row["indirect_CI_excludes_zero"])

    if (not same_sign) and ci_excludes:
        return "significant_inconsistent_or_suppression"
    if same_sign and (not ci_excludes):
        return "significant_FDR_but_CI_includes_zero"
    if (not same_sign) and (not ci_excludes):
        return "significant_inconsistent_and_CI_includes_zero"

    return "significant_other"


def semicolon_unique(values) -> str:
    vals = sorted({str(v) for v in values if pd.notna(v) and str(v) != ""})
    return ";".join(vals)


def run_self_test() -> None:
    x = pd.Series([True, False, 1, 0, "1.0", "0.0", "true", "false"])
    got = truthy_series(x).tolist()
    exp = [True, False, True, False, True, False, True, False]
    assert got == exp, (got, exp)

    test = pd.DataFrame(
        {
            "species": ["s1", "s1", "s2", "s3"],
            "species_label": ["S1", "S1", "S2", "S3"],
            "diet_score": ["AMED", "hPDI", "AMED", "AHEI"],
            "cgm_outcome": ["glucose_cv", "glucose_cv", "mean_glucose", "mean_glucose"],
            "mediation_FDR05_primary": [1, 1, 1, 0],
            "candidate_mediator_consistent": [1, 1, 0, 0],
            "indirect_same_sign_as_total": [1, 1, 0, 1],
            "indirect_CI_excludes_zero": [1, 1, 1, 0],
            "priority_tier": [
                "Tier1_cross_context_recurrent",
                "Tier1_cross_context_recurrent",
                "Tier1_cross_context_recurrent",
                "Tier3_single_robust_path",
            ],
            "mediated_proportion": [0.1, 0.2, -0.1, 0.0],
            "mediated_proportion_outside_0_1": [0, 0, 1, 0],
            "proportion_potentially_unstable": [0, 0, 0, 0],
            "bootstrap_total_crosses_zero_fraction": [0, 0, 0, 0],
            "bootstrap_p_indirect": [0.01, 0.02, 0.01, 0.8],
            "FDR_BH_within_score_outcome": [0.02, 0.03, 0.02, 0.8],
            "FDR_BH_global_all_mediation": [0.03, 0.04, 0.03, 0.9],
            "N": [100, 100, 100, 100],
            "a_diet_to_microbiome": [0.2, 0.2, 0.3, 0.1],
            "b_microbiome_to_cgm": [0.3, 0.3, -0.2, 0.1],
            "indirect_effect": [0.06, 0.06, -0.06, 0.01],
            "total_effect": [0.5, 0.5, 0.4, 0.2],
            "indirect_CI95_lower": [0.01, 0.01, -0.10, -0.01],
            "indirect_CI95_upper": [0.10, 0.10, -0.02, 0.03],
        }
    )
    for c in [
        "mediation_FDR05_primary",
        "candidate_mediator_consistent",
        "indirect_same_sign_as_total",
        "indirect_CI_excludes_zero",
        "mediated_proportion_outside_0_1",
        "proportion_potentially_unstable",
    ]:
        test[c] = truthy_series(test[c])

    classes = test.apply(classify_path, axis=1).tolist()
    assert classes[0] == "significant_direction_consistent_candidate"
    assert classes[2] == "significant_inconsistent_or_suppression"
    assert classes[3] == "not_significant_primary_FDR"

    print("SELF TEST: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Step 10 mediation-path CSV.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Output report directory.",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    inp = args.input.resolve()
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp}")

    df = pd.read_csv(inp)

    required = [
        "diet_score",
        "cgm_outcome",
        "species",
        "N",
        "mediation_FDR05_primary",
        "candidate_mediator_consistent",
        "indirect_same_sign_as_total",
        "indirect_CI_excludes_zero",
        "bootstrap_p_indirect",
        "FDR_BH_within_score_outcome",
        "a_diet_to_microbiome",
        "b_microbiome_to_cgm",
        "indirect_effect",
        "total_effect",
        "indirect_CI95_lower",
        "indirect_CI95_upper",
        "mediated_proportion",
    ]
    ensure_columns(df, required)

    bool_cols = [
        "mediation_FDR05_primary",
        "candidate_mediator_consistent",
        "indirect_same_sign_as_total",
        "indirect_CI_excludes_zero",
    ]
    for c in [
        "mediated_proportion_outside_0_1",
        "proportion_potentially_unstable",
    ]:
        if c in df.columns:
            bool_cols.append(c)

    for c in bool_cols:
        df[c] = truthy_series(df[c])

    if "species_label" not in df.columns:
        df["species_label"] = np.nan
    df["species_label"] = df.apply(clean_species_label, axis=1)

    if "priority_tier" not in df.columns:
        df["priority_tier"] = "MISSING_PRIORITY_TIER"

    if "FDR_BH_global_all_mediation" not in df.columns:
        df["FDR_BH_global_all_mediation"] = np.nan

    if "bootstrap_total_crosses_zero_fraction" not in df.columns:
        df["bootstrap_total_crosses_zero_fraction"] = np.nan

    if "mediated_proportion_outside_0_1" not in df.columns:
        df["mediated_proportion_outside_0_1"] = (
            np.isfinite(df["mediated_proportion"])
            & (
                (df["mediated_proportion"] < 0)
                | (df["mediated_proportion"] > 1)
            )
        )

    if "proportion_potentially_unstable" not in df.columns:
        df["proportion_potentially_unstable"] = (
            ~np.isfinite(df["mediated_proportion"])
            | (
                pd.to_numeric(
                    df["bootstrap_total_crosses_zero_fraction"],
                    errors="coerce",
                )
                > 0.05
            )
        )

    # ------------------------------------------------------------------
    # 1. Row-level audit and explicit classification.
    # ------------------------------------------------------------------
    audit = df.copy()
    audit["mediation_interpretation_class"] = audit.apply(classify_path, axis=1)
    audit["is_inconsistent_significant"] = (
        audit["mediation_FDR05_primary"]
        & ~audit["indirect_same_sign_as_total"]
    )
    audit["fdr_significant_but_ci_includes_zero"] = (
        audit["mediation_FDR05_primary"]
        & ~audit["indirect_CI_excludes_zero"]
    )

    # Rank rows without creating an arbitrary biological "importance score".
    # Ordering is only for inspection: consistent -> recurrence tier -> FDR -> p.
    tier_order = {
        "Tier1_cross_context_recurrent": 1,
        "Tier2_repeated": 2,
        "Tier3_single_robust_path": 3,
    }
    audit["_tier_order"] = audit["priority_tier"].map(tier_order).fillna(99)
    audit = audit.sort_values(
        [
            "candidate_mediator_consistent",
            "mediation_FDR05_primary",
            "_tier_order",
            "FDR_BH_within_score_outcome",
            "bootstrap_p_indirect",
            "diet_score",
            "cgm_outcome",
            "species_label",
        ],
        ascending=[False, False, True, True, True, True, True, True],
    ).drop(columns="_tier_order")

    # ------------------------------------------------------------------
    # 2. Priority-tier QC.
    # ------------------------------------------------------------------
    priority = (
        audit.groupby("priority_tier", dropna=False)
        .agg(
            paths_tested=("species", "size"),
            unique_species=("species", "nunique"),
            significant_primary_FDR05=("mediation_FDR05_primary", "sum"),
            direction_consistent_candidates=("candidate_mediator_consistent", "sum"),
            inconsistent_significant=("is_inconsistent_significant", "sum"),
        )
        .reset_index()
        .sort_values(
            ["direction_consistent_candidates", "paths_tested"],
            ascending=[False, False],
        )
    )

    # ------------------------------------------------------------------
    # 3. Species recurrence among direction-consistent candidates.
    # ------------------------------------------------------------------
    cons = audit.loc[audit["candidate_mediator_consistent"]].copy()

    if len(cons):
        recurrence = (
            cons.groupby(["species", "species_label"], dropna=False)
            .agg(
                n_consistent_paths=("species", "size"),
                n_diet_scores=("diet_score", "nunique"),
                n_cgm_outcomes=("cgm_outcome", "nunique"),
                diet_scores=("diet_score", semicolon_unique),
                cgm_outcomes=("cgm_outcome", semicolon_unique),
                priority_tiers=("priority_tier", semicolon_unique),
                median_N=("N", "median"),
                min_primary_FDR=("FDR_BH_within_score_outcome", "min"),
                min_bootstrap_p=("bootstrap_p_indirect", "min"),
                median_abs_indirect_effect=(
                    "indirect_effect",
                    lambda x: float(np.nanmedian(np.abs(pd.to_numeric(x, errors="coerce")))),
                ),
                median_mediated_proportion=("mediated_proportion", "median"),
                any_proportion_outside_0_1=("mediated_proportion_outside_0_1", "max"),
                any_proportion_unstable=("proportion_potentially_unstable", "max"),
            )
            .reset_index()
        )
        recurrence["cross_diet_recurrent"] = recurrence["n_diet_scores"] >= 2
        recurrence["cross_outcome_recurrent"] = recurrence["n_cgm_outcomes"] >= 2
        recurrence = recurrence.sort_values(
            [
                "n_cgm_outcomes",
                "n_diet_scores",
                "n_consistent_paths",
                "min_primary_FDR",
                "species_label",
            ],
            ascending=[False, False, False, True, True],
        ).reset_index(drop=True)
        recurrence.insert(0, "recurrence_rank", np.arange(1, len(recurrence) + 1))
    else:
        recurrence = pd.DataFrame()

    # ------------------------------------------------------------------
    # 4. AMED vs hPDI overlap specifically for glucose CV.
    # ------------------------------------------------------------------
    cv = cons.loc[cons["cgm_outcome"].eq("glucose_cv")].copy()
    amed = cv.loc[cv["diet_score"].eq("AMED"), ["species", "species_label"]].drop_duplicates()
    hpdi = cv.loc[cv["diet_score"].eq("hPDI"), ["species", "species_label"]].drop_duplicates()

    amed_set = set(amed["species"])
    hpdi_set = set(hpdi["species"])
    shared_set = amed_set & hpdi_set
    union_set = amed_set | hpdi_set

    overlap_rows = []
    for species in sorted(union_set):
        labels = cv.loc[cv["species"].eq(species), "species_label"].dropna().astype(str)
        label = labels.iloc[0] if len(labels) else species

        rows = cv.loc[cv["species"].eq(species)]
        amed_row = rows.loc[rows["diet_score"].eq("AMED")]
        hpdi_row = rows.loc[rows["diet_score"].eq("hPDI")]

        def one_val(d, col):
            return d.iloc[0][col] if len(d) else np.nan

        overlap_rows.append(
            {
                "species": species,
                "species_label": label,
                "in_AMED_glucose_cv": species in amed_set,
                "in_hPDI_glucose_cv": species in hpdi_set,
                "shared_AMED_hPDI_glucose_cv": species in shared_set,
                "AMED_indirect_effect": one_val(amed_row, "indirect_effect"),
                "hPDI_indirect_effect": one_val(hpdi_row, "indirect_effect"),
                "AMED_primary_FDR": one_val(amed_row, "FDR_BH_within_score_outcome"),
                "hPDI_primary_FDR": one_val(hpdi_row, "FDR_BH_within_score_outcome"),
                "AMED_mediated_proportion": one_val(amed_row, "mediated_proportion"),
                "hPDI_mediated_proportion": one_val(hpdi_row, "mediated_proportion"),
            }
        )

    overlap = pd.DataFrame(overlap_rows)
    if len(overlap):
        overlap = overlap.sort_values(
            [
                "shared_AMED_hPDI_glucose_cv",
                "species_label",
            ],
            ascending=[False, True],
        ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 5. Mediated-proportion / numerical QC.
    # ------------------------------------------------------------------
    prop_cols = [
        "diet_score",
        "cgm_outcome",
        "species",
        "species_label",
        "priority_tier",
        "N",
        "mediation_FDR05_primary",
        "candidate_mediator_consistent",
        "indirect_effect",
        "total_effect",
        "mediated_proportion",
        "mediated_proportion_outside_0_1",
        "proportion_potentially_unstable",
        "bootstrap_total_crosses_zero_fraction",
        "bootstrap_p_indirect",
        "FDR_BH_within_score_outcome",
    ]
    for c in [
        "mediated_proportion_CI95_lower",
        "mediated_proportion_CI95_upper",
        "bootstrap_valid_proportion_n",
    ]:
        if c in audit.columns:
            prop_cols.append(c)

    proportion_qc = audit[prop_cols].copy()
    proportion_qc["needs_proportion_review"] = (
        proportion_qc["mediated_proportion_outside_0_1"]
        | proportion_qc["proportion_potentially_unstable"]
    )
    proportion_qc = proportion_qc.sort_values(
        [
            "needs_proportion_review",
            "candidate_mediator_consistent",
            "FDR_BH_within_score_outcome",
        ],
        ascending=[False, False, True],
    )

    # ------------------------------------------------------------------
    # 6. Compact overall summary.
    # ------------------------------------------------------------------
    n_paths = len(audit)
    n_sig = int(audit["mediation_FDR05_primary"].sum())
    n_cons = int(audit["candidate_mediator_consistent"].sum())
    n_inconsistent_sig = int(audit["is_inconsistent_significant"].sum())
    n_fdr_ci_disagree = int(audit["fdr_significant_but_ci_includes_zero"].sum())

    tier1_mask = audit["priority_tier"].eq("Tier1_cross_context_recurrent")
    n_tier1 = int(tier1_mask.sum())
    n_tier1_cons = int(
        (tier1_mask & audit["candidate_mediator_consistent"]).sum()
    )

    amed_n = len(amed_set)
    hpdi_n = len(hpdi_set)
    shared_n = len(shared_set)
    union_n = len(union_set)
    jaccard = shared_n / union_n if union_n else np.nan

    summary = pd.DataFrame(
        [
            {
                "paths_tested": n_paths,
                "unique_species_tested": int(audit["species"].nunique()),
                "primary_FDR05_paths": n_sig,
                "direction_consistent_candidates": n_cons,
                "significant_inconsistent_or_suppression": n_inconsistent_sig,
                "FDR_significant_but_CI_includes_zero": n_fdr_ci_disagree,
                "Tier1_paths_tested": n_tier1,
                "Tier1_direction_consistent_candidates": n_tier1_cons,
                "consistent_unique_species": int(cons["species"].nunique()) if len(cons) else 0,
                "AMED_glucose_cv_consistent_species": amed_n,
                "hPDI_glucose_cv_consistent_species": hpdi_n,
                "AMED_hPDI_glucose_cv_shared_species": shared_n,
                "AMED_hPDI_glucose_cv_union_species": union_n,
                "AMED_hPDI_glucose_cv_jaccard": jaccard,
                "paths_proportion_outside_0_1": int(
                    audit["mediated_proportion_outside_0_1"].sum()
                ),
                "paths_proportion_potentially_unstable": int(
                    audit["proportion_potentially_unstable"].sum()
                ),
            }
        ]
    )

    # ------------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------------
    out_audit = report_dir / "10b_mediation_path_audit.csv"
    out_priority = report_dir / "10b_mediation_priority_tier_qc.csv"
    out_recur = report_dir / "10b_mediation_species_recurrence.csv"
    out_overlap = report_dir / "10b_glucose_cv_amed_hpdi_overlap.csv"
    out_prop = report_dir / "10b_mediation_proportion_qc.csv"
    out_summary = report_dir / "10b_mediation_qc_summary.csv"
    out_txt = report_dir / "10b_mediation_qc_summary.txt"

    audit.to_csv(out_audit, index=False)
    priority.to_csv(out_priority, index=False)
    recurrence.to_csv(out_recur, index=False)
    overlap.to_csv(out_overlap, index=False)
    proportion_qc.to_csv(out_prop, index=False)
    summary.to_csv(out_summary, index=False)

    lines = []
    lines.append("=" * 100)
    lines.append("10b — MEDIATION QC AND RECURRENCE")
    lines.append("=" * 100)
    lines.append(f"Input: {inp}")
    lines.append("")
    lines.append("CORE COUNTS")
    lines.append("-" * 100)
    lines.append(f"Paths tested: {n_paths}")
    lines.append(f"Unique species tested: {audit['species'].nunique()}")
    lines.append(f"Primary FDR<0.05 paths: {n_sig}")
    lines.append(f"Direction-consistent candidates: {n_cons}")
    lines.append(
        "Significant but direction-inconsistent / suppression-like paths: "
        f"{n_inconsistent_sig}"
    )
    lines.append(
        "FDR-significant but bootstrap CI includes zero: "
        f"{n_fdr_ci_disagree}"
    )
    lines.append("")

    lines.append("SIGNIFICANT-PATH INTERPRETATION CLASSES")
    lines.append("-" * 100)
    class_counts = (
        audit["mediation_interpretation_class"]
        .value_counts(dropna=False)
        .rename_axis("class")
        .reset_index(name="n")
    )
    lines.append(class_counts.to_string(index=False))
    lines.append("")

    lines.append("PRIORITY-TIER QC")
    lines.append("-" * 100)
    lines.append(priority.to_string(index=False))
    lines.append("")

    lines.append("AMED vs hPDI — DIRECTION-CONSISTENT GLUCOSE-CV MEDIATORS")
    lines.append("-" * 100)
    lines.append(f"AMED species: {amed_n}")
    lines.append(f"hPDI species: {hpdi_n}")
    lines.append(f"Shared species: {shared_n}")
    lines.append(f"Union species: {union_n}")
    lines.append(
        "Jaccard overlap: "
        + (f"{jaccard:.4f}" if np.isfinite(jaccard) else "NA")
    )
    if shared_n:
        shared_display = overlap.loc[
            overlap["shared_AMED_hPDI_glucose_cv"],
            ["species_label", "AMED_primary_FDR", "hPDI_primary_FDR"],
        ]
        lines.append("")
        lines.append("Shared species:")
        lines.append(shared_display.to_string(index=False))
    else:
        lines.append("No shared AMED/hPDI glucose-CV mediator species.")
    lines.append("")

    lines.append("SPECIES RECURRENCE AMONG DIRECTION-CONSISTENT CANDIDATES")
    lines.append("-" * 100)
    if len(recurrence):
        show = [
            "recurrence_rank",
            "species_label",
            "n_consistent_paths",
            "n_diet_scores",
            "n_cgm_outcomes",
            "diet_scores",
            "cgm_outcomes",
            "min_primary_FDR",
        ]
        lines.append(recurrence[show].to_string(index=False))
    else:
        lines.append("None.")
    lines.append("")

    lines.append("MEDIATED-PROPORTION QC")
    lines.append("-" * 100)
    lines.append(
        "Outside [0,1]: "
        f"{int(audit['mediated_proportion_outside_0_1'].sum())}"
    )
    lines.append(
        "Potentially unstable: "
        f"{int(audit['proportion_potentially_unstable'].sum())}"
    )
    lines.append(
        "These are QC flags only. They do not by themselves invalidate an "
        "indirect effect; mediated proportion becomes difficult to interpret "
        "when total effects are small or unstable."
    )
    lines.append("")

    lines.append("INTERPRETATION RULE")
    lines.append("-" * 100)
    lines.append(
        "Primary candidate mediators = primary mediation FDR<0.05 + indirect "
        "effect has the same sign as the total effect + bootstrap CI excludes zero."
    )
    lines.append(
        "Significant indirect effects with the opposite sign to the total effect "
        "are retained separately as inconsistent/suppression-like paths, not "
        "counted as the primary mediation bridge."
    )
    lines.append(
        "This remains a mediation-style observational analysis; temporal/causal "
        "mediation is not established by these regressions alone."
    )
    lines.append("")

    lines.append("SAVED")
    lines.append("-" * 100)
    for p in [
        out_audit,
        out_priority,
        out_recur,
        out_overlap,
        out_prop,
        out_summary,
        out_txt,
    ]:
        lines.append(str(p))

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
