#!/usr/bin/env python3
"""Step 10 — exact-species robustness consolidation for 4New_cgm.

Purpose
-------
Do NOT refit any microbiome model.

Read the already-completed:
    4New_cgm/outputs/run_*/models/microbiome_cgm_all_models.csv

Then consolidate exact species-level robustness for:
    MAGE
    TAR180
    TBR70
    TIR70_180

The script explicitly separates:
1) Primary Model2 -> Primary Model3
2) Primary Model2 -> Strict Model2
3) Primary Model3 -> Strict Model3
4) Same-M3-sample Model2 -> Primary Model3
   (isolates BMI adjustment from sample selection)

Primary findings are NOT redefined by sensitivity analyses.
A "highest_robustness" subset is reported separately:
    significant in primary M2 + primary M3 + strict M2 + strict M3
    AND same beta sign in all four models.

TIR70_180 remains supportive.
TAR180 remains a microbiome association result only; Diet->TAR180 does not
gain a primary bridge gate merely because a hurdle sensitivity was positive.

Dependencies
------------
numpy, pandas, scipy only.

Outputs
-------
4New_cgm/outputs/species_robustness_<timestamp>/
    reports/
        exact_species_long.csv
        outcome_robustness_summary.csv
        same_cohort_bmi_decomposition.csv
        status_transition_counts.csv
        highest_robustness_species.csv
        primary_m2_significant_species.csv
        step10_species_robustness_summary.txt
        manifest.json
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")

CANDIDATE_BRANCH_DIRS = [
    ROOT / "diet_microbiome_glucose_analysis" / "4New_cgm",
    ROOT / "4New_cgm",
]

EXPECTED_SPECIES = 379
OUTCOME_ORDER = ["MAGE", "TAR180", "TBR70", "TIR70_180"]

OUTCOME_ROLE = {
    "MAGE": "inferential",
    "TAR180": "inferential_distribution_sensitive",
    "TBR70": "inferential",
    "TIR70_180": "supportive",
}

SET_SPECS = {
    "primary_m2": ("primary", 2),
    "same_m3_m2": ("same_m3", 2),
    "primary_m3": ("primary", 3),
    "strict_m2": ("strict", 2),
    "strict_m3": ("strict", 3),
}


def require(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_branch_dir(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"4New_cgm directory missing: {p}")
        return p

    found = [p for p in CANDIDATE_BRANCH_DIRS if p.is_dir()]
    if not found:
        raise FileNotFoundError(
            "Could not find 4New_cgm. Tried:\n"
            + "\n".join(str(p) for p in CANDIDATE_BRANCH_DIRS)
        )
    return found[0]


def find_latest_completed_association_run(branch: Path) -> Path:
    candidates = []

    for p in (branch / "outputs").glob("run_*"):
        table = p / "models" / "microbiome_cgm_all_models.csv"
        manifest = p / "reports" / "manifest.json"

        if not table.is_file():
            continue

        status = ""
        if manifest.is_file():
            try:
                status = json.loads(
                    manifest.read_text(encoding="utf-8")
                ).get("status", "")
            except Exception:
                status = ""

        acceptable = status in {
            "completed",
            "completed_with_model_failures",
            "completed_with_visualization_failure",
        }
        candidates.append(
            (
                1 if acceptable else 0,
                table.stat().st_mtime,
                p,
                status,
            )
        )

    if not candidates:
        raise FileNotFoundError(
            f"No run_* containing microbiome_cgm_all_models.csv under "
            f"{branch / 'outputs'}"
        )

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


def safe_spearman(x, y):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan
    rho, p = spearmanr(x.loc[mask], y.loc[mask])
    return float(rho), float(p)


def sign_equal(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return (
        a.notna()
        & b.notna()
        & np.isfinite(a)
        & np.isfinite(b)
        & np.sign(a).eq(np.sign(b))
    )


def subset_for(result: pd.DataFrame, outcome: str, set_name: str) -> pd.DataFrame:
    analysis_set, model = SET_SPECS[set_name]
    x = result.loc[
        result["analysis_set"].eq(analysis_set)
        & result["model"].eq(model)
        & result["outcome"].eq(outcome)
    ].copy()

    if len(x) != EXPECTED_SPECIES:
        raise RuntimeError(
            f"{outcome}/{set_name}: expected {EXPECTED_SPECIES} rows, "
            f"found {len(x)}"
        )

    if x["species"].duplicated().any():
        dup = x.loc[x["species"].duplicated(), "species"].head(20).tolist()
        raise RuntimeError(
            f"{outcome}/{set_name}: duplicate species rows: {dup}"
        )

    x["computed"] = x["status"].eq("computed") & x["p_value"].notna()
    x["sig"] = (
        x["computed"]
        & pd.to_numeric(x["FDR_family"], errors="coerce").lt(0.05)
    )

    cols = [
        "species",
        "N",
        "sample_sha256",
        "beta",
        "SE",
        "CI95_lower",
        "CI95_upper",
        "p_value",
        "FDR_family",
        "FDR_global",
        "status",
        "computed",
        "sig",
    ]
    missing = [c for c in cols if c not in x.columns]
    if missing:
        raise RuntimeError(
            f"{outcome}/{set_name}: missing columns {missing}"
        )

    return x[cols].copy()


def renamed(x: pd.DataFrame, label: str) -> pd.DataFrame:
    return x.rename(
        columns={
            c: f"{c}_{label}"
            for c in x.columns
            if c != "species"
        }
    )


def transition_table(long_df: pd.DataFrame, left: str, right: str):
    a = truthy(long_df[f"sig_{left}"])
    b = truthy(long_df[f"sig_{right}"])

    status = np.select(
        [
            a & b,
            a & ~b,
            ~a & b,
            ~a & ~b,
        ],
        [
            "retained_significant",
            "lost_significance",
            "gained_significance",
            "nonsignificant_both",
        ],
        default="unknown",
    )

    return pd.Series(status, index=long_df.index)


def comparison_metrics(
    df: pd.DataFrame,
    left: str,
    right: str,
) -> dict:
    left_sig = truthy(df[f"sig_{left}"])
    right_sig = truthy(df[f"sig_{right}"])

    retained = left_sig & right_sig
    lost = left_sig & ~right_sig
    gained = ~left_sig & right_sig

    rho_all, rho_all_p = safe_spearman(
        df[f"beta_{left}"],
        df[f"beta_{right}"],
    )

    rho_primary, rho_primary_p = safe_spearman(
        df.loc[left_sig, f"beta_{left}"],
        df.loc[left_sig, f"beta_{right}"],
    )

    same_all = sign_equal(
        df[f"beta_{left}"],
        df[f"beta_{right}"],
    )

    same_retained = sign_equal(
        df.loc[retained, f"beta_{left}"],
        df.loc[retained, f"beta_{right}"],
    )

    return {
        "left_set": left,
        "right_set": right,
        "left_significant": int(left_sig.sum()),
        "right_significant": int(right_sig.sum()),
        "retained_significant": int(retained.sum()),
        "lost_significance": int(lost.sum()),
        "gained_significance": int(gained.sum()),
        "same_direction_all_species": int(same_all.sum()),
        "sign_flips_all_species": int((~same_all).sum()),
        "same_direction_retained": int(same_retained.sum()),
        "sign_flips_retained": int((~same_retained).sum()),
        "beta_spearman_all_species": rho_all,
        "beta_spearman_all_species_p": rho_all_p,
        "beta_spearman_left_sig": rho_primary,
        "beta_spearman_left_sig_p": rho_primary_p,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-dir", default=None)
    parser.add_argument("--association-run-dir", default=None)
    args = parser.parse_args()

    branch = find_branch_dir(args.branch_dir)

    run_dir = (
        Path(args.association_run_dir).expanduser().resolve()
        if args.association_run_dir
        else find_latest_completed_association_run(branch)
    )

    model_path = (
        run_dir / "models" / "microbiome_cgm_all_models.csv"
    )
    require(model_path, "microbiome model table")

    print("=== STEP 10 EXACT-SPECIES MICROBIOME ROBUSTNESS ===")
    print(f"ASSOCIATION_RUN={run_dir}")
    print(f"MODEL_TABLE={model_path}")
    print("NO_MODELS_REFIT=True")
    print("PRIMARY_RESULT_REDEFINED=False")
    print("TIR70_180_ROLE=supportive")
    print("TAR180_ROLE=inferential_distribution_sensitive")

    result = pd.read_csv(model_path, low_memory=False)

    required = {
        "analysis_set",
        "model",
        "outcome",
        "species",
        "N",
        "sample_sha256",
        "beta",
        "SE",
        "CI95_lower",
        "CI95_upper",
        "p_value",
        "FDR_family",
        "FDR_global",
        "status",
    }
    missing = sorted(required - set(result.columns))
    if missing:
        raise RuntimeError(
            f"microbiome model table missing columns: {missing}"
        )

    outcomes_found = set(result["outcome"].dropna().astype(str))
    if not set(OUTCOME_ORDER).issubset(outcomes_found):
        raise RuntimeError(
            f"Missing expected outcomes. Found={sorted(outcomes_found)}"
        )

    # Require exact set availability for all intended robustness comparisons.
    long_frames = []
    summary_rows = []
    transition_rows = []
    same_cohort_rows = []

    for outcome in OUTCOME_ORDER:
        pieces = {}

        for set_name in SET_SPECS:
            pieces[set_name] = subset_for(
                result, outcome, set_name
            )

        reference_species = set(pieces["primary_m2"]["species"])
        if len(reference_species) != EXPECTED_SPECIES:
            raise RuntimeError(
                f"{outcome}: primary_m2 has "
                f"{len(reference_species)} unique species"
            )

        for set_name, x in pieces.items():
            current = set(x["species"])
            if current != reference_species:
                raise RuntimeError(
                    f"{outcome}: exact species set mismatch in {set_name}; "
                    f"missing={len(reference_species-current)}, "
                    f"extra={len(current-reference_species)}"
                )

        merged = renamed(
            pieces["primary_m2"], "primary_m2"
        )

        for set_name in [
            "same_m3_m2",
            "primary_m3",
            "strict_m2",
            "strict_m3",
        ]:
            merged = merged.merge(
                renamed(pieces[set_name], set_name),
                on="species",
                how="inner",
                validate="one_to_one",
            )

        if len(merged) != EXPECTED_SPECIES:
            raise RuntimeError(
                f"{outcome}: merged exact species table has "
                f"{len(merged)} rows"
            )

        merged.insert(0, "outcome", outcome)
        merged.insert(1, "outcome_role", OUTCOME_ROLE[outcome])

        # Primary/sensitivity status.
        p2 = truthy(merged["sig_primary_m2"])
        pm3 = truthy(merged["sig_primary_m3"])
        sm2 = truthy(merged["sig_strict_m2"])
        sm3 = truthy(merged["sig_strict_m3"])
        same_m3_m2_sig = truthy(merged["sig_same_m3_m2"])

        merged["primary_m2_significant"] = p2
        merged["retained_primary_m3"] = p2 & pm3
        merged["retained_strict_m2"] = p2 & sm2
        merged["retained_strict_m3"] = p2 & sm3

        same_sign_p2_pm3 = sign_equal(
            merged["beta_primary_m2"],
            merged["beta_primary_m3"],
        )
        same_sign_p2_sm2 = sign_equal(
            merged["beta_primary_m2"],
            merged["beta_strict_m2"],
        )
        same_sign_p2_sm3 = sign_equal(
            merged["beta_primary_m2"],
            merged["beta_strict_m3"],
        )

        merged["direction_concordant_primary_m3"] = (
            same_sign_p2_pm3
        )
        merged["direction_concordant_strict_m2"] = (
            same_sign_p2_sm2
        )
        merged["direction_concordant_strict_m3"] = (
            same_sign_p2_sm3
        )

        merged["highest_robustness"] = (
            p2
            & pm3
            & sm2
            & sm3
            & same_sign_p2_pm3
            & same_sign_p2_sm2
            & same_sign_p2_sm3
        )

        # Transparent tiering only; primary result set remains pM2.
        merged["robustness_tier"] = np.select(
            [
                merged["highest_robustness"],
                p2 & pm3 & sm2,
                p2 & pm3,
                p2,
            ],
            [
                "highest_robustness_all_models",
                "retained_primary_m3_and_strict_m2",
                "retained_primary_m3",
                "primary_m2_only_or_other_sensitivity_loss",
            ],
            default="not_primary_m2_significant",
        )

        long_frames.append(merged)

        # Core pairwise comparisons.
        pairs = [
            ("primary_m2", "primary_m3"),
            ("primary_m2", "strict_m2"),
            ("primary_m3", "strict_m3"),
            ("primary_m2", "same_m3_m2"),
            ("same_m3_m2", "primary_m3"),
        ]

        pair_metrics = {}

        for left, right in pairs:
            metrics = comparison_metrics(
                merged, left, right
            )
            metrics.update(
                outcome=outcome,
                outcome_role=OUTCOME_ROLE[outcome],
            )
            pair_metrics[(left, right)] = metrics

            transition = transition_table(
                merged, left, right
            )
            counts = (
                transition
                .value_counts(dropna=False)
                .rename_axis("transition")
                .reset_index(name="N")
            )
            counts.insert(0, "right_set", right)
            counts.insert(0, "left_set", left)
            counts.insert(0, "outcome", outcome)
            transition_rows.append(counts)

        primary_sig = int(p2.sum())
        pm3_sig = int(pm3.sum())
        strict2_sig = int(sm2.sum())
        strict3_sig = int(sm3.sum())
        high = int(merged["highest_robustness"].sum())

        # Outcome-level summary.
        row = {
            "outcome": outcome,
            "outcome_role": OUTCOME_ROLE[outcome],
            "species_tested": EXPECTED_SPECIES,
            "primary_m2_significant": primary_sig,
            "primary_m3_significant": pm3_sig,
            "strict_m2_significant": strict2_sig,
            "strict_m3_significant": strict3_sig,
            "primary_m2_retained_m3":
                int((p2 & pm3).sum()),
            "primary_m2_lost_m3":
                int((p2 & ~pm3).sum()),
            "primary_m3_gained_vs_m2":
                int((~p2 & pm3).sum()),
            "primary_m2_retained_strict_m2":
                int((p2 & sm2).sum()),
            "primary_m2_lost_strict_m2":
                int((p2 & ~sm2).sum()),
            "strict_m2_gained_vs_primary_m2":
                int((~p2 & sm2).sum()),
            "highest_robustness_n": high,
            "highest_robustness_fraction_of_primary_m2":
                (high / primary_sig if primary_sig else np.nan),
            "primary_m2_vs_m3_beta_rho_all379":
                pair_metrics[
                    ("primary_m2", "primary_m3")
                ]["beta_spearman_all_species"],
            "primary_m2_vs_strict_m2_beta_rho_all379":
                pair_metrics[
                    ("primary_m2", "strict_m2")
                ]["beta_spearman_all_species"],
            "same_m3_m2_vs_m3_beta_rho_all379":
                pair_metrics[
                    ("same_m3_m2", "primary_m3")
                ]["beta_spearman_all_species"],
            "primary_m2_vs_m3_sign_flips_all379":
                pair_metrics[
                    ("primary_m2", "primary_m3")
                ]["sign_flips_all_species"],
            "primary_m2_vs_strict_m2_sign_flips_all379":
                pair_metrics[
                    ("primary_m2", "strict_m2")
                ]["sign_flips_all_species"],
            "same_m3_m2_vs_m3_sign_flips_all379":
                pair_metrics[
                    ("same_m3_m2", "primary_m3")
                ]["sign_flips_all_species"],
        }
        summary_rows.append(row)

        # Same-cohort BMI decomposition table.
        selection_metrics = pair_metrics[
            ("primary_m2", "same_m3_m2")
        ].copy()
        selection_metrics.update(
            outcome=outcome,
            decomposition_component="sample_selection_primaryM2_to_M3completecase",
        )

        adjustment_metrics = pair_metrics[
            ("same_m3_m2", "primary_m3")
        ].copy()
        adjustment_metrics.update(
            outcome=outcome,
            decomposition_component="BMI_adjustment_same_sample",
        )

        same_cohort_rows += [
            selection_metrics,
            adjustment_metrics,
        ]

    exact_long = pd.concat(
        long_frames,
        ignore_index=True,
    )
    outcome_summary = pd.DataFrame(summary_rows)
    transitions = pd.concat(
        transition_rows,
        ignore_index=True,
    )
    same_cohort = pd.DataFrame(same_cohort_rows)

    # Important species lists.
    primary_species = exact_long.loc[
        exact_long["primary_m2_significant"]
    ].copy()

    highest = exact_long.loc[
        exact_long["highest_robustness"]
    ].copy()

    # Sanity: every primary significant species is exact taxonomy string,
    # and no duplicate within an outcome.
    if primary_species.duplicated(
        ["outcome", "species"]
    ).any():
        raise RuntimeError(
            "Duplicate primary significant outcome/species rows"
        )

    if highest.duplicated(
        ["outcome", "species"]
    ).any():
        raise RuntimeError(
            "Duplicate highest-robustness outcome/species rows"
        )

    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    out = (
        branch
        / "outputs"
        / f"species_robustness_{stamp}"
    )
    reports = out / "reports"
    reports.mkdir(
        parents=True,
        exist_ok=False,
    )

    exact_path = reports / "exact_species_long.csv"
    summary_path = reports / "outcome_robustness_summary.csv"
    same_path = reports / "same_cohort_bmi_decomposition.csv"
    transition_path = reports / "status_transition_counts.csv"
    primary_path = reports / "primary_m2_significant_species.csv"
    high_path = reports / "highest_robustness_species.csv"
    txt_path = reports / "step10_species_robustness_summary.txt"
    manifest_path = reports / "manifest.json"

    exact_long.to_csv(
        exact_path,
        index=False,
    )
    outcome_summary.to_csv(
        summary_path,
        index=False,
    )
    same_cohort.to_csv(
        same_path,
        index=False,
    )
    transitions.to_csv(
        transition_path,
        index=False,
    )
    primary_species.to_csv(
        primary_path,
        index=False,
    )
    highest.to_csv(
        high_path,
        index=False,
    )

    # Print compact exact-species lists for MAGE and TBR70.
    def compact_species(outcome):
        x = highest.loc[
            highest["outcome"].eq(outcome),
            [
                "species",
                "beta_primary_m2",
                "FDR_family_primary_m2",
                "beta_primary_m3",
                "FDR_family_primary_m3",
                "beta_strict_m2",
                "FDR_family_strict_m2",
                "beta_strict_m3",
                "FDR_family_strict_m3",
            ],
        ].copy()
        return (
            "NONE"
            if x.empty
            else x.to_string(index=False)
        )

    lines = [
        "=== STEP 10 EXACT-SPECIES MICROBIOME ROBUSTNESS SUMMARY ===",
        f"ASSOCIATION_RUN={run_dir}",
        "NO_MODELS_REFIT=True",
        "EXACT_SPECIES_SET_MATCH_ALL_COMPARISONS=True",
        f"SPECIES_TESTED_PER_OUTCOME={EXPECTED_SPECIES}",
        "PRIMARY_RESULT_REDEFINED=False",
        "HIGHEST_ROBUSTNESS_DEFINITION="
        "primaryM2+primaryM3+strictM2+strictM3 FDR<0.05 and same sign",
        "TIR70_180_ROLE=supportive",
        "TAR180_ROLE=inferential_distribution_sensitive",
        "",
        "--- OUTCOME ROBUSTNESS ---",
        outcome_summary.to_string(index=False),
        "",
        "--- SAME-COHORT M2->M3 DECOMPOSITION ---",
        same_cohort.to_string(index=False),
        "",
        "--- HIGHEST-ROBUSTNESS MAGE SPECIES ---",
        compact_species("MAGE"),
        "",
        "--- HIGHEST-ROBUSTNESS TBR70 SPECIES ---",
        compact_species("TBR70"),
        "",
        "INTERPRETATION RULES:",
        "- Primary Model2 significant species remain the primary MWAS result set.",
        "- Sensitivity losses/gains are reported; they do not silently redefine the primary result.",
        "- 'highest_robustness' is a secondary robustness subset, not a new discovery family.",
        "- Same-M3-sample Model2 vs Model3 isolates BMI adjustment from BMI-complete-case selection.",
        "- TIR70_180 remains supportive and is not counted as an independent TBR/TAR replication.",
        "- TAR180 microbiome associations do not create a Diet->TAR180 bridge gate by themselves.",
        "- No bridge or mediation is performed in Step10.",
        "",
        f"EXACT_LONG={exact_path}",
        f"OUTCOME_SUMMARY={summary_path}",
        f"SAME_COHORT={same_path}",
        f"TRANSITIONS={transition_path}",
        f"PRIMARY_SPECIES={primary_path}",
        f"HIGHEST_ROBUSTNESS_SPECIES={high_path}",
    ]

    txt_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "status": "completed",
        "purpose": "exact-species microbiome robustness before bridge",
        "association_run": str(run_dir),
        "source_model_table": str(model_path),
        "source_model_sha256": sha256(model_path),
        "models_refit": False,
        "expected_species_per_outcome": EXPECTED_SPECIES,
        "exact_species_set_match_all_comparisons": True,
        "primary_result_redefined": False,
        "highest_robustness_definition": (
            "FDR<0.05 in primary M2, primary M3, strict M2, strict M3 "
            "with same beta sign in all four"
        ),
        "outcome_roles": OUTCOME_ROLE,
        "bridge_run": False,
        "mediation_run": False,
        "outputs": {
            "exact_species_long": str(exact_path),
            "outcome_robustness_summary": str(summary_path),
            "same_cohort_bmi_decomposition": str(same_path),
            "status_transition_counts": str(transition_path),
            "primary_m2_significant_species": str(primary_path),
            "highest_robustness_species": str(high_path),
            "summary": str(txt_path),
        },
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
