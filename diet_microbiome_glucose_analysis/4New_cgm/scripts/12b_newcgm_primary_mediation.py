#!/usr/bin/env python3
"""Step 12b — primary mediation-style analysis for the 73 robust new-CGM bridges.

This script REUSES the current canonical old-project mediation implementation:
    diet_microbiome_glucose_analysis/scripts/
    10_diet_microbiome_cgm_mediation.py

It does not reimplement the statistical core.  In particular it reuses:
    - run_one_path()
    - mediation_point_estimate()
    - bootstrap_mediation()
    - encode_covariates()
    - bh_fdr()

New-CGM adaptation
------------------
Primary candidate set:
    Step11 robust_bridge_candidates.csv
    expected frozen set = 73 exact Diet -> species -> CGM paths
    across 7 Diet x CGM contexts.

Outcomes:
    MAGE
    TBR70

Candidate membership:
    Defined upstream in Step11.  Step12 does NOT rescreen/redefine paths.

Model:
    M ~ Diet + covariates
    Y ~ Diet + M + covariates
    Y ~ Diet + covariates

    indirect = a * b
    mediated proportion = indirect / total

Bootstrap:
    Canonical nonparametric percentile bootstrap
    default n = 1000
    canonical two-sided sign p-value (+1 correction)

Multiplicity:
    Primary BH-FDR within each Diet x CGM candidate-mediator family.
    Global BH-FDR across all 73 paths is also reported as sensitivity.

Covariates:
    Reuse canonical Model3 covariate list:
      age, sex, education, smoking, sleep, physical activity,
      vitamin use, hormone use, BMI, CGM device type
    plus canonical optional antibiotic/PPI only if present.

    New-score prespecified adjustment:
      EAT13, NOVA4, Carbohydrate_pct additionally adjust for
      mean_daily_energy_kcal, matching the frozen new-score Model2/3 analyses.

Eligibility:
    primary: primary_eligible
    strict:  strict_eligible
Default is primary.

Interpretation:
    Cross-sectional statistical mediation-style patterns only.
    NOT causal or temporal mediation.

Dependencies:
    numpy, pandas, scipy only.  No statsmodels required.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
ANALYSIS_DIR = ROOT / "diet_microbiome_glucose_analysis"
CANONICAL_STEP10 = (
    ANALYSIS_DIR / "scripts" / "10_diet_microbiome_cgm_mediation.py"
)

BRANCH_CANDIDATES = [
    ANALYSIS_DIR / "4New_cgm",
    ROOT / "4New_cgm",
]

NEW_ENERGY_ADJUSTED_SCORES = {
    "EAT13",
    "NOVA4",
    "Carbohydrate_pct",
}

EXPECTED_PRIMARY_PATHS = 73
EXPECTED_DIRECT_CONTEXTS = 7
EXPECTED_UNIQUE_SPECIES = 36
ALLOWED_OUTCOMES = {"MAGE", "TBR70"}


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


def find_branch(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"4New_cgm branch missing: {p}")
        return p

    found = [p for p in BRANCH_CANDIDATES if p.is_dir()]
    if not found:
        raise FileNotFoundError(
            "Could not find 4New_cgm. Tried:\n"
            + "\n".join(str(p) for p in BRANCH_CANDIDATES)
        )
    return found[0]


def latest_bridge_dir(branch: Path) -> Path:
    candidates = []
    for p in (branch / "outputs").glob("bridge_screen_*"):
        f = p / "reports" / "robust_bridge_candidates.csv"
        if f.is_file():
            candidates.append((f.stat().st_mtime, p))
    if not candidates:
        raise FileNotFoundError(
            f"No bridge_screen_* with robust_bridge_candidates.csv under "
            f"{branch / 'outputs'}"
        )
    candidates.sort(reverse=True)
    return candidates[0][1]


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_candidates(
    bridge_dir: Path,
    expected_paths: int,
    max_paths: int | None,
) -> pd.DataFrame:
    path = bridge_dir / "reports" / "robust_bridge_candidates.csv"
    require(path, "Step11 robust bridge candidates")
    d = pd.read_csv(path, low_memory=False)

    required = {"exposure", "outcome", "species"}
    missing = sorted(required - set(d.columns))
    if missing:
        raise RuntimeError(
            f"Step11 candidate table missing columns: {missing}"
        )

    # robust_bridge_candidates.csv should already be filtered, but if the
    # boolean is present, enforce it again.
    if "robust_bridge_candidate" in d.columns:
        d = d.loc[truthy(d["robust_bridge_candidate"])].copy()

    d["exposure"] = d["exposure"].astype(str)
    d["outcome"] = d["outcome"].astype(str)
    d["species"] = d["species"].astype(str)

    unexpected_outcomes = sorted(set(d["outcome"]) - ALLOWED_OUTCOMES)
    if unexpected_outcomes:
        raise RuntimeError(
            "Primary new-CGM mediation must contain only MAGE/TBR70. "
            f"Unexpected={unexpected_outcomes}"
        )

    if d.duplicated(["exposure", "outcome", "species"]).any():
        dup = d.loc[
            d.duplicated(["exposure", "outcome", "species"], keep=False),
            ["exposure", "outcome", "species"],
        ]
        raise RuntimeError(
            "Duplicate exact mediation candidate paths:\n"
            + dup.head(30).to_string(index=False)
        )

    if max_paths is None:
        if len(d) != expected_paths:
            raise RuntimeError(
                f"Frozen Step11 path set mismatch: expected {expected_paths}, "
                f"found {len(d)}. Do not silently change the primary family."
            )
        n_contexts = d[["exposure", "outcome"]].drop_duplicates().shape[0]
        if n_contexts != EXPECTED_DIRECT_CONTEXTS:
            raise RuntimeError(
                f"Expected {EXPECTED_DIRECT_CONTEXTS} Diet-CGM contexts, "
                f"found {n_contexts}"
            )
        n_species = d["species"].nunique()
        if n_species != EXPECTED_UNIQUE_SPECIES:
            raise RuntimeError(
                f"Expected {EXPECTED_UNIQUE_SPECIES} unique exact species, "
                f"found {n_species}"
            )

    # Rename to canonical Step10 path-row keys.
    d = d.rename(
        columns={
            "exposure": "diet_score",
            "outcome": "cgm_outcome",
        }
    )

    # Deterministic order; highest-priority tags are annotations only.
    if "highest_priority_bridge_candidate" in d.columns:
        d["_priority"] = truthy(
            d["highest_priority_bridge_candidate"]
        ).astype(int)
    else:
        d["_priority"] = 0

    d = d.sort_values(
        ["_priority", "cgm_outcome", "diet_score", "species"],
        ascending=[False, True, True, True],
    ).drop(columns="_priority").reset_index(drop=True)

    if max_paths is not None:
        if max_paths < 1:
            raise ValueError("--max-paths must be >=1")
        d = d.head(max_paths).copy()

    return d


def patch_covariates(canonical):
    """Extend canonical get_covariate_list only for the three new scores."""
    original = canonical.get_covariate_list

    def patched(score, available_columns):
        covars = list(original(score, available_columns))

        if score in NEW_ENERGY_ADJUSTED_SCORES:
            energy = "mean_daily_energy_kcal"
            if energy not in available_columns:
                raise RuntimeError(
                    f"{score} mediation requires {energy}, matching its "
                    "frozen Model2/3 adjustment, but the column is absent."
                )
            if energy not in covars:
                covars.append(energy)

        return covars

    canonical.get_covariate_list = patched


def prepare_new_cgm_data(branch: Path, config_path: Path):
    scripts = branch / "scripts"
    require(scripts / "new_cgm_data.py", "new_cgm_data.py")

    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))

    import new_cgm_data as data

    config = data.load_config(config_path)
    diet, microbiome, species, cgm_audit, overlap, fingerprints = (
        data.prepare_tables(config)
    )

    # Exact project identity.
    if diet["participant_id"].duplicated().any():
        raise RuntimeError("Diet table has duplicate participant_id")
    if microbiome["participant_id"].duplicated().any():
        raise RuntimeError("Microbiome table has duplicate participant_id")

    return data, diet, microbiome, species, cgm_audit, overlap, fingerprints


def candidate_metadata_for_merge(candidates: pd.DataFrame) -> pd.DataFrame:
    key = ["diet_score", "cgm_outcome", "species"]

    # Keep all useful Step11 annotations except huge/duplicative helper columns
    # only when names do not collide with canonical mediation outputs.
    wanted = [
        "exposure_role",
        "cgm_highest_robustness",
        "highest_priority_bridge_candidate",
        "m2_beta_diet",
        "m2_FDR_diet",
        "m2_beta_cgm",
        "m2_FDR_cgm",
        "m2_beta_product",
        "m2_direction_concordant",
        "m3_beta_diet",
        "m3_FDR_diet",
        "m3_beta_cgm",
        "m3_FDR_cgm",
        "m3_beta_product",
        "m3_direction_concordant",
        "direct_beta_m2",
        "direct_beta_m3",
        "direction_rule_type",
    ]
    cols = key + [c for c in wanted if c in candidates.columns]
    return candidates[cols].copy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-dir", default=None)
    parser.add_argument("--bridge-dir", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--cohort-mode",
        choices=["primary", "strict"],
        default="primary",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260904,
    )
    parser.add_argument(
        "--expected-paths",
        type=int,
        default=EXPECTED_PRIMARY_PATHS,
    )
    parser.add_argument(
        "--max-paths",
        type=int,
        default=None,
        help="Smoke-test only. If set, frozen 73-path count is checked before truncation.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run only the canonical mediation synthetic self-test.",
    )
    args = parser.parse_args()

    if args.bootstrap < 20:
        raise ValueError("--bootstrap must be >=20")

    branch = find_branch(args.branch_dir)
    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else branch / "config" / "analysis.json"
    )
    require(config_path, "4New_cgm analysis config")
    require(CANONICAL_STEP10, "canonical old-project mediation script")

    # Make imports used by the canonical script resolvable.
    old_script_dir = CANONICAL_STEP10.parent
    if str(old_script_dir) not in sys.path:
        sys.path.insert(0, str(old_script_dir))

    canonical = load_module(
        CANONICAL_STEP10,
        "hpp_canonical_mediation_newcgm_adapter",
    )

    required_helpers = [
        "run_one_path",
        "mediation_point_estimate",
        "bootstrap_mediation",
        "bh_fdr",
        "get_covariate_list",
    ]
    missing_helpers = [
        h for h in required_helpers if not hasattr(canonical, h)
    ]
    if missing_helpers:
        raise RuntimeError(
            f"Canonical Step10 missing expected helpers: {missing_helpers}"
        )

    if args.self_test:
        if not hasattr(canonical, "self_test"):
            raise RuntimeError("Canonical Step10 has no self_test()")
        canonical.self_test()
        return 0

    bridge_dir = (
        Path(args.bridge_dir).expanduser().resolve()
        if args.bridge_dir
        else latest_bridge_dir(branch)
    )

    # Always validate the full frozen candidate family first.
    full_candidates = load_candidates(
        bridge_dir,
        expected_paths=args.expected_paths,
        max_paths=None,
    )
    if args.max_paths is None:
        candidates = full_candidates.copy()
    else:
        candidates = full_candidates.head(args.max_paths).copy()

    data, diet, microbiome, species, cgm_audit, overlap, fingerprints = (
        prepare_new_cgm_data(branch, config_path)
    )

    if args.cohort_mode == "primary":
        flag = "primary_eligible"
    else:
        flag = "strict_eligible"

    if flag not in diet.columns:
        raise RuntimeError(f"Eligibility flag missing: {flag}")

    cohort_cov = diet.loc[truthy(diet[flag])].copy()

    # Patch ONLY the three new diet scores to add their prespecified energy
    # covariate.  Everything else inside run_one_path remains canonical.
    patch_covariates(canonical)

    # Validate exposure/outcome mappings.
    missing_scores = sorted(
        set(candidates["diet_score"]) - set(data.EXPOSURES)
    )
    missing_outcomes = sorted(
        set(candidates["cgm_outcome"]) - set(data.OUTCOMES)
    )
    if missing_scores or missing_outcomes:
        raise RuntimeError(
            f"Mapping failure: scores={missing_scores}; "
            f"outcomes={missing_outcomes}"
        )

    candidate_species = set(candidates["species"])
    missing_species = sorted(candidate_species - set(microbiome.columns))
    if missing_species:
        raise RuntimeError(
            f"{len(missing_species)} exact candidate species are absent from "
            f"the frozen microbiome matrix. Examples={missing_species[:10]}"
        )

    # Output location is isolated; no old or prior new-CGM result can be overwritten.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = (
        f"_{args.cohort_mode}"
        + (f"_smoke{args.max_paths}" if args.max_paths is not None else "")
    )
    out = branch / "outputs" / f"mediation_newcgm_{stamp}{suffix}"
    model_dir = out / "models"
    report_dir = out / "reports"
    model_dir.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir()

    print("=== STEP 12b NEW-CGM MEDIATION-STYLE ANALYSIS ===")
    print(f"CANONICAL_ENGINE={CANONICAL_STEP10}")
    print("CANONICAL_STATISTICAL_CORE_REUSED=True")
    print("STATSMODELS_REQUIRED=False")
    print(f"BRIDGE_DIR={bridge_dir}")
    print(f"COHORT_MODE={args.cohort_mode}")
    print(f"BOOTSTRAP={args.bootstrap}")
    print(f"BASE_SEED={args.seed}")
    print(f"FROZEN_FULL_PATH_SET={len(full_candidates)}")
    print(f"PATHS_THIS_RUN={len(candidates)}")
    print(
        "UNIQUE_SPECIES_THIS_RUN="
        f"{candidates['species'].nunique()}"
    )
    print(
        "DIET_CGM_CONTEXTS_THIS_RUN="
        f"{candidates[['diet_score','cgm_outcome']].drop_duplicates().shape[0]}"
    )
    print(
        "NEW_SCORE_EXTRA_COVARIATE="
        "mean_daily_energy_kcal"
    )
    print("TAR180_INCLUDED=False")
    print("TIR70_180_INCLUDED=False")
    print("PRIMARY_RESULT_SET_REDEFINED=False")

    print("\n--- CANDIDATE COUNTS ---")
    print(
        candidates.groupby(
            ["diet_score", "cgm_outcome"],
            dropna=False,
        )
        .size()
        .rename("n_paths")
        .reset_index()
        .sort_values(["cgm_outcome", "diet_score"])
        .to_string(index=False)
    )

    results = []
    start_all = time.time()
    total = len(candidates)

    for i, (_, row) in enumerate(candidates.iterrows(), start=1):
        score = str(row["diet_score"])
        outcome = str(row["cgm_outcome"])
        taxon = str(row["species"])

        score_col = data.EXPOSURES[score][0]
        outcome_col = data.OUTCOMES[outcome]

        t0 = time.time()
        res = canonical.run_one_path(
            path_row=row,
            cohort_cov=cohort_cov,
            microbiome=microbiome,
            outcome_col=outcome_col,
            score_col=score_col,
            bootstrap=args.bootstrap,
            base_seed=args.seed,
        )
        res["cohort_mode"] = args.cohort_mode
        res["elapsed_seconds"] = time.time() - t0
        results.append(res)

        print(
            f"[{i:02d}/{total}] {score} -> {taxon} -> {outcome} | "
            f"N={res['N']} | "
            f"a={res['a_diet_to_microbiome']:+.5f} | "
            f"b={res['b_microbiome_to_cgm']:+.5f} | "
            f"indirect={res['indirect_effect']:+.6f} "
            f"[{res['indirect_CI95_lower']:+.6f},"
            f"{res['indirect_CI95_upper']:+.6f}] | "
            f"p={res['bootstrap_p_indirect']:.4g}"
        )

    results = pd.DataFrame(results)

    # Canonical primary multiplicity family.
    results["FDR_BH_within_score_outcome"] = np.nan
    for _, idx in results.groupby(
        ["diet_score", "cgm_outcome"]
    ).groups.items():
        idx = list(idx)
        results.loc[
            idx, "FDR_BH_within_score_outcome"
        ] = canonical.bh_fdr(
            results.loc[idx, "bootstrap_p_indirect"]
        )

    # Canonical global sensitivity family.
    results["FDR_BH_global_all_mediation"] = canonical.bh_fdr(
        results["bootstrap_p_indirect"]
    )

    results["mediation_FDR05_primary"] = (
        results["FDR_BH_within_score_outcome"] < 0.05
    )

    # EXACT canonical interpretation flag.
    results["candidate_mediator_consistent"] = (
        results["mediation_FDR05_primary"]
        & truthy(results["indirect_same_sign_as_total"])
        & truthy(results["indirect_CI_excludes_zero"])
    )

    # Add Step11 robustness annotations without changing membership/inference.
    meta = candidate_metadata_for_merge(candidates)
    results = results.merge(
        meta,
        on=["diet_score", "cgm_outcome", "species"],
        how="left",
        validate="one_to_one",
    )

    if "highest_priority_bridge_candidate" in results.columns:
        results["bridge_highest_priority"] = truthy(
            results["highest_priority_bridge_candidate"]
        )
    else:
        results["bridge_highest_priority"] = False

    results["significant_and_highest_priority_bridge"] = (
        truthy(results["mediation_FDR05_primary"])
        & results["bridge_highest_priority"]
    )
    results["consistent_and_highest_priority_bridge"] = (
        truthy(results["candidate_mediator_consistent"])
        & results["bridge_highest_priority"]
    )

    results = results.sort_values(
        [
            "candidate_mediator_consistent",
            "mediation_FDR05_primary",
            "bridge_highest_priority",
            "FDR_BH_within_score_outcome",
            "bootstrap_p_indirect",
        ],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)

    # Primary output.
    prefix = (
        "12b_newcgm_primary"
        if args.cohort_mode == "primary"
        else "12b_newcgm_strict"
    )
    if args.max_paths is not None:
        prefix += f"_smoke{args.max_paths}"

    model_path = model_dir / f"{prefix}_mediation_paths.csv"
    sig_path = report_dir / f"{prefix}_significant.csv"
    consistent_path = report_dir / f"{prefix}_consistent.csv"
    summary_path = report_dir / f"{prefix}_context_summary.csv"
    cov_path = report_dir / f"{prefix}_covariate_audit.csv"
    txt_path = report_dir / f"{prefix}_summary.txt"
    manifest_path = report_dir / "manifest.json"

    results.to_csv(model_path, index=False)
    results.loc[
        truthy(results["mediation_FDR05_primary"])
    ].to_csv(sig_path, index=False)
    results.loc[
        truthy(results["candidate_mediator_consistent"])
    ].to_csv(consistent_path, index=False)

    summary = (
        results.groupby(
            ["diet_score", "cgm_outcome"],
            dropna=False,
        )
        .agg(
            paths_tested=("species", "size"),
            unique_species=("species", "nunique"),
            median_N=("N", "median"),
            significant_mediation_FDR05=(
                "mediation_FDR05_primary", "sum"
            ),
            direction_consistent_candidates=(
                "candidate_mediator_consistent", "sum"
            ),
            highest_priority_bridge_paths=(
                "bridge_highest_priority", "sum"
            ),
            significant_highest_priority=(
                "significant_and_highest_priority_bridge", "sum"
            ),
            consistent_highest_priority=(
                "consistent_and_highest_priority_bridge", "sum"
            ),
            min_bootstrap_p=("bootstrap_p_indirect", "min"),
            min_FDR_primary=(
                "FDR_BH_within_score_outcome", "min"
            ),
        )
        .reset_index()
        .sort_values(["cgm_outcome", "diet_score"])
    )
    summary.to_csv(summary_path, index=False)

    # Explicit covariate audit for every score represented.
    cov_rows = []
    for score in sorted(candidates["diet_score"].unique()):
        covars = canonical.get_covariate_list(
            score,
            cohort_cov.columns,
        )
        for order, covar in enumerate(covars, start=1):
            cov_rows.append({
                "diet_score": score,
                "order": order,
                "covariate": covar,
                "source": (
                    "new_score_prespecified_energy_adjustment"
                    if covar == "mean_daily_energy_kcal"
                    else "canonical_step10_covariate_rule"
                ),
            })
    cov_audit = pd.DataFrame(cov_rows)
    cov_audit.to_csv(cov_path, index=False)

    elapsed = time.time() - start_all

    n_sig = int(truthy(results["mediation_FDR05_primary"]).sum())
    n_cons = int(truthy(results["candidate_mediator_consistent"]).sum())
    n_sig_hp = int(results["significant_and_highest_priority_bridge"].sum())
    n_cons_hp = int(results["consistent_and_highest_priority_bridge"].sum())

    unstable = int(
        truthy(results["proportion_potentially_unstable"]).sum()
    )
    outside = int(
        truthy(results["mediated_proportion_outside_0_1"]).sum()
    )

    lines = [
        "=== STEP 12b NEW-CGM MEDIATION-STYLE SUMMARY ===",
        f"CANONICAL_ENGINE={CANONICAL_STEP10}",
        "CANONICAL_STATISTICAL_CORE_REUSED=True",
        f"COHORT_MODE={args.cohort_mode}",
        f"BOOTSTRAP={args.bootstrap}",
        f"PATHS_TESTED={len(results)}",
        f"UNIQUE_SPECIES_TESTED={results['species'].nunique()}",
        (
            "DIET_CGM_CONTEXTS="
            f"{results[['diet_score','cgm_outcome']].drop_duplicates().shape[0]}"
        ),
        f"MEDIATION_FDR05_PRIMARY={n_sig}",
        f"DIRECTION_CONSISTENT_CANDIDATES={n_cons}",
        f"SIGNIFICANT_HIGHEST_PRIORITY_BRIDGE={n_sig_hp}",
        f"CONSISTENT_HIGHEST_PRIORITY_BRIDGE={n_cons_hp}",
        f"MEDIATED_PROPORTION_OUTSIDE_0_1={outside}",
        f"POTENTIALLY_UNSTABLE_PROPORTION={unstable}",
        f"ELAPSED_SECONDS={elapsed:.1f}",
        "",
        "--- CONTEXT SUMMARY ---",
        summary.to_string(index=False),
        "",
        "--- COVARIATE AUDIT ---",
        cov_audit.to_string(index=False),
        "",
        "INTERPRETATION RULES:",
        "- These are cross-sectional statistical mediation-style patterns.",
        "- They do NOT establish causal or temporal mediation.",
        "- Primary BH-FDR is within each frozen Diet x CGM candidate family.",
        "- Global BH-FDR is sensitivity only.",
        "- The full 73-path Step11 robust set is the primary mediation family.",
        "- Step11 highest-priority status is annotation/prioritization only.",
        "- EAT13/NOVA4/Carbohydrate_pct include mean_daily_energy_kcal.",
        "- TAR180 and TIR70_180 are not part of this primary mediation run.",
        "",
        f"MODEL={model_path}",
        f"SIGNIFICANT={sig_path}",
        f"CONSISTENT={consistent_path}",
        f"CONTEXT_SUMMARY={summary_path}",
        f"COVARIATE_AUDIT={cov_path}",
    ]
    txt_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "status": "completed",
        "canonical_engine": str(CANONICAL_STEP10),
        "canonical_statistical_core_reused": True,
        "bridge_dir": str(bridge_dir),
        "cohort_mode": args.cohort_mode,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "full_frozen_candidate_paths": len(full_candidates),
        "paths_tested": len(results),
        "unique_species_tested": int(results["species"].nunique()),
        "diet_cgm_contexts": int(
            results[["diet_score", "cgm_outcome"]]
            .drop_duplicates()
            .shape[0]
        ),
        "mediation_fdr05_primary": n_sig,
        "direction_consistent_candidates": n_cons,
        "significant_highest_priority_bridge": n_sig_hp,
        "consistent_highest_priority_bridge": n_cons_hp,
        "primary_result_redefined": False,
        "tar180_included": False,
        "tir70_180_included": False,
        "new_score_extra_covariate": "mean_daily_energy_kcal",
        "outputs": {
            "model": str(model_path),
            "significant": str(sig_path),
            "consistent": str(consistent_path),
            "context_summary": str(summary_path),
            "covariate_audit": str(cov_path),
            "summary": str(txt_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
