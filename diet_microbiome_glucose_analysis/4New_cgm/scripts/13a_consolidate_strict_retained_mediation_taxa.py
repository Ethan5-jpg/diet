#!/usr/bin/env python3
"""Step 13a — consolidate strict-retained new-CGM mediation taxa.

READ/COMPARE ONLY:
- no model refit
- no bootstrap
- no new candidate discovery

Primary inference remains the Step12b primary set (46 significant paths).
This step characterizes the robustness subset:
    primary-consistent AND strict-consistent AND same indirect-effect sign

Expected from Step12c:
    38 strict-retained primary-consistent paths.

It summarizes:
- exact retained paths
- recurrent taxa across diets/outcomes
- overlap with the old AMED/hPDI 12-core species
- old12 polarity concordance where available
- a literature-annotation shortlist

Outputs are descriptive/prioritization tables only.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
DG = ROOT / "diet_microbiome_glucose_analysis"

BRANCH_CANDIDATES = [
    DG / "4New_cgm",
    ROOT / "4New_cgm",
]

OLD12_CANDIDATES = [
    DG / "outputs" / "reports" / "13b_12_core_direction_classes.csv",
    DG / "outputs" / "reports" / "13a_12_core_species_biological_annotation.csv",
]

EXPECTED_RETAINED = 38


def truthy(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
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
    found = [p for p in BRANCH_CANDIDATES if p.is_dir()]
    if not found:
        raise FileNotFoundError("4New_cgm branch not found")
    return found[0]


def latest_consolidation(branch: Path) -> Path:
    candidates = []
    for p in (branch / "outputs").glob("mediation_consolidation_*"):
        f = p / "reports" / "strict_retained_mediation_paths.csv"
        if f.is_file():
            candidates.append((f.stat().st_mtime, p))
    if not candidates:
        raise FileNotFoundError("No mediation_consolidation_* output found")
    candidates.sort(reverse=True)
    return candidates[0][1]


def find_old12() -> Path:
    for p in OLD12_CANDIDATES:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Could not find old 12-core table. Tried:\n"
        + "\n".join(map(str, OLD12_CANDIDATES))
    )


def species_label(full_taxonomy: str) -> str:
    s = str(full_taxonomy)
    # Prefer terminal species token after |s__
    if "|s__" in s:
        return s.split("|s__")[-1]
    if "s__" in s:
        return s.split("s__")[-1]
    return s


def load_old12(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)

    species_col = None
    for c in ["species", "full_taxonomy", "taxon"]:
        if c in d.columns:
            species_col = c
            break
    if species_col is None:
        raise RuntimeError(
            f"Old12 table has no exact taxonomy column. Columns={d.columns.tolist()}"
        )

    keep = [species_col]
    for c in [
        "species_label",
        "project_direction_class",
        "biological_evidence_category",
        "primary_core",
        "protocol_core",
        "primary_core_retained",
    ]:
        if c in d.columns:
            keep.append(c)

    out = d[keep].copy()
    out = out.rename(columns={species_col: "species"})
    out["species"] = out["species"].astype(str)
    out = out.drop_duplicates("species")

    if "species_label" not in out.columns:
        out["species_label"] = out["species"].map(species_label)

    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-dir", default=None)
    parser.add_argument("--consolidation-dir", default=None)
    parser.add_argument("--expected-retained", type=int, default=EXPECTED_RETAINED)
    args = parser.parse_args()

    branch = find_branch(args.branch_dir)
    cons = (
        Path(args.consolidation_dir).expanduser().resolve()
        if args.consolidation_dir
        else latest_consolidation(branch)
    )

    retained_path = cons / "reports" / "strict_retained_mediation_paths.csv"
    if not retained_path.is_file():
        raise FileNotFoundError(retained_path)

    retained = pd.read_csv(retained_path, low_memory=False)

    required = {
        "diet_score", "cgm_outcome", "species",
        "indirect_effect_primary",
        "FDR_BH_within_score_outcome_primary",
        "indirect_effect_strict",
        "FDR_BH_within_score_outcome_strict",
    }
    missing = sorted(required - set(retained.columns))
    if missing:
        raise RuntimeError(f"Retained path table missing columns: {missing}")

    if len(retained) != args.expected_retained:
        raise RuntimeError(
            f"Expected {args.expected_retained} retained paths, found {len(retained)}"
        )

    if retained.duplicated(["diet_score", "cgm_outcome", "species"]).any():
        raise RuntimeError("Duplicate retained exact paths")

    retained["species"] = retained["species"].astype(str)
    retained["species_label"] = retained["species"].map(species_label)

    if "highest_priority_primary" in retained.columns:
        retained["highest_priority_primary"] = truthy(
            retained["highest_priority_primary"]
        )
    else:
        retained["highest_priority_primary"] = False

    if "strict_retained_highest_priority" in retained.columns:
        retained["strict_retained_highest_priority"] = truthy(
            retained["strict_retained_highest_priority"]
        )
    else:
        retained["strict_retained_highest_priority"] = False

    # Recurrence summary.
    recurrence = (
        retained.groupby("species", as_index=False)
        .agg(
            species_label=("species_label", "first"),
            retained_paths=("species", "size"),
            diet_exposures=("diet_score", "nunique"),
            cgm_outcomes=("cgm_outcome", "nunique"),
            diets=("diet_score", lambda x: ";".join(sorted(set(map(str, x))))),
            outcomes=("cgm_outcome", lambda x: ";".join(sorted(set(map(str, x))))),
            highest_priority_retained_paths=(
                "strict_retained_highest_priority", "sum"
            ),
            mean_abs_indirect_primary=(
                "indirect_effect_primary",
                lambda x: float(np.mean(np.abs(pd.to_numeric(x, errors="coerce"))))
            ),
            mean_abs_indirect_strict=(
                "indirect_effect_strict",
                lambda x: float(np.mean(np.abs(pd.to_numeric(x, errors="coerce"))))
            ),
            best_primary_FDR=(
                "FDR_BH_within_score_outcome_primary", "min"
            ),
            best_strict_FDR=(
                "FDR_BH_within_score_outcome_strict", "min"
            ),
        )
    )

    # Descriptive annotation classes, not new inference.
    recurrence["recurrence_class"] = np.select(
        [
            recurrence["cgm_outcomes"].ge(2),
            recurrence["diet_exposures"].ge(3),
            recurrence["diet_exposures"].eq(2),
        ],
        [
            "cross_outcome_recurrent",
            "multi_diet_recurrent_ge3",
            "two_diet_recurrent",
        ],
        default="single_context",
    )

    # Old 12-core overlap.
    old12_path = find_old12()
    old12 = load_old12(old12_path)
    recurrence = recurrence.merge(
        old12.add_prefix("old12_").rename(
            columns={"old12_species": "species"}
        ),
        on="species",
        how="left",
        validate="one_to_one",
    )
    recurrence["present_in_old12_core"] = recurrence[
        "old12_species_label"
    ].notna()

    # Literature annotation shortlist:
    # all recurrent taxa + all old12-overlap taxa + all retained highest-priority taxa.
    recurrence["literature_annotation_shortlist"] = (
        recurrence["diet_exposures"].ge(2)
        | recurrence["cgm_outcomes"].ge(2)
        | recurrence["present_in_old12_core"]
        | recurrence["highest_priority_retained_paths"].gt(0)
    )

    recurrence = recurrence.sort_values(
        [
            "cgm_outcomes",
            "diet_exposures",
            "retained_paths",
            "highest_priority_retained_paths",
            "species_label",
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)

    old_overlap = recurrence.loc[
        recurrence["present_in_old12_core"]
    ].copy()

    shortlist = recurrence.loc[
        recurrence["literature_annotation_shortlist"]
    ].copy()

    # Context summary.
    context = (
        retained.groupby(["diet_score", "cgm_outcome"], as_index=False)
        .agg(
            retained_paths=("species", "size"),
            unique_species=("species", "nunique"),
            highest_priority_retained=(
                "strict_retained_highest_priority", "sum"
            ),
            median_primary_FDR=(
                "FDR_BH_within_score_outcome_primary", "median"
            ),
            median_strict_FDR=(
                "FDR_BH_within_score_outcome_strict", "median"
            ),
        )
        .sort_values(["cgm_outcome", "diet_score"])
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = branch / "outputs" / f"final_taxa_consolidation_{stamp}"
    reports = out / "reports"
    reports.mkdir(parents=True, exist_ok=False)

    retained_out = reports / "13a_strict_retained_38_paths.csv"
    recurrence_out = reports / "13a_retained_species_recurrence.csv"
    old12_out = reports / "13a_old12_overlap.csv"
    shortlist_out = reports / "13a_literature_annotation_shortlist.csv"
    context_out = reports / "13a_retained_context_summary.csv"
    summary_out = reports / "13a_taxa_consolidation_summary.txt"
    manifest_out = reports / "manifest.json"

    retained.to_csv(retained_out, index=False)
    recurrence.to_csv(recurrence_out, index=False)
    old_overlap.to_csv(old12_out, index=False)
    shortlist.to_csv(shortlist_out, index=False)
    context.to_csv(context_out, index=False)

    lines = [
        "=== STEP 13a STRICT-RETAINED MEDIATION TAXA CONSOLIDATION ===",
        f"CONSOLIDATION_SOURCE={cons}",
        f"OLD12_SOURCE={old12_path}",
        "MODELS_REFIT=False",
        "BOOTSTRAP_RERUN=False",
        f"STRICT_RETAINED_PATHS={len(retained)}",
        f"STRICT_RETAINED_UNIQUE_SPECIES={retained['species'].nunique()}",
        f"OLD12_SHARED_SPECIES={int(recurrence['present_in_old12_core'].sum())}",
        f"CROSS_OUTCOME_RECURRENT_SPECIES={int(recurrence['cgm_outcomes'].ge(2).sum())}",
        f"MULTI_DIET_GE2_SPECIES={int(recurrence['diet_exposures'].ge(2).sum())}",
        f"LITERATURE_SHORTLIST_SPECIES={len(shortlist)}",
        "",
        "--- CONTEXT SUMMARY ---",
        context.to_string(index=False),
        "",
        "--- TOP RECURRENT TAXA ---",
        recurrence.head(30)[
            [
                "species_label",
                "retained_paths",
                "diet_exposures",
                "cgm_outcomes",
                "diets",
                "outcomes",
                "highest_priority_retained_paths",
                "present_in_old12_core",
                "recurrence_class",
            ]
        ].to_string(index=False),
        "",
        "--- OLD 12-CORE OVERLAP ---",
        (
            "NONE"
            if old_overlap.empty
            else old_overlap[
                [
                    "species_label",
                    "retained_paths",
                    "diet_exposures",
                    "cgm_outcomes",
                    "diets",
                    "outcomes",
                    "old12_project_direction_class",
                ]
                if "old12_project_direction_class" in old_overlap.columns
                else [
                    "species_label",
                    "retained_paths",
                    "diet_exposures",
                    "cgm_outcomes",
                    "diets",
                    "outcomes",
                ]
            ].to_string(index=False)
        ),
        "",
        "INTERPRETATION RULES:",
        "- Primary Step12b remains the primary inference set (46 paths).",
        "- These 38 paths are a strict-retained robustness subset.",
        "- Recurrence is descriptive prioritization, not a new significance test.",
        "- Old12 overlap is exact full-taxonomy overlap.",
        "- Literature annotation does not convert cross-sectional mediation-style associations into causal mechanisms.",
        "",
        f"RETAINED_PATHS={retained_out}",
        f"RECURRENCE={recurrence_out}",
        f"OLD12_OVERLAP={old12_out}",
        f"LITERATURE_SHORTLIST={shortlist_out}",
        f"CONTEXT={context_out}",
    ]
    summary_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest_out.write_text(
        json.dumps(
            {
                "status": "completed",
                "models_refit": False,
                "bootstrap_rerun": False,
                "primary_inference_redefined": False,
                "strict_retained_paths": int(len(retained)),
                "strict_retained_unique_species": int(retained["species"].nunique()),
                "old12_shared_species": int(
                    recurrence["present_in_old12_core"].sum()
                ),
                "literature_shortlist_species": int(len(shortlist)),
                "outputs": {
                    "retained_paths": str(retained_out),
                    "recurrence": str(recurrence_out),
                    "old12_overlap": str(old12_out),
                    "literature_shortlist": str(shortlist_out),
                    "context": str(context_out),
                    "summary": str(summary_out),
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
