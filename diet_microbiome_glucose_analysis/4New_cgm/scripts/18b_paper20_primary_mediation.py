#!/usr/bin/env python3
"""
Step 18b — Paper-20 PRIMARY cross-sectional mediation-style analysis.

This is the formal PRIMARY mediation run on the frozen Step17 robust bridge set.

Frozen candidate set
--------------------
ALL Step17 robust_bridge_candidate paths are tested.
The highest-priority subset is carried forward as an annotation only and does
not redefine the primary mediation family.

Statistical core
----------------
The canonical project Step10 statistical core is reused directly:
    mediator: M ~ Diet + covariates
    outcome:  Y ~ Diet + M + covariates
    total:    Y ~ Diet + covariates
    indirect = a * b
    95% percentile nonparametric bootstrap CI
    two-sided bootstrap sign p-value

Multiplicity
------------
Primary:
    BH-FDR within each frozen Diet x CGM mediator family.

Sensitivity:
    BH-FDR globally across all frozen mediation paths.

Primary participant cohort
--------------------------
Uses the same primary diabetes/A10 eligibility logic audited in Step18a.

Covariates
----------
AHEI / AMED / rEDIH:
    age, sex, education, smoking, sleep, physical activity,
    vitamin use, hormone use, BMI, CGM device type.

Corrected hPDI:
    same + alcohol_intake_g_day.

EAT13 / NOVA4 / Carbohydrate_pct:
    same + mean_daily_energy_kcal.

Antibiotic/PPI are deliberately excluded here and remain a later
protocol-window sensitivity analysis.

QC
--
For every fitted path, N and the exact complete-case participant hash must
match the Step18a PRIMARY audit. A mismatch aborts the run.

Interpretation
--------------
Cross-sectional statistical mediation-style patterns only.
No temporal or causal mediation is established.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import multiprocessing as mp
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
DG = ROOT / "diet_microbiome_glucose_analysis"

BRANCH_CANDIDATES = [
    DG / "4New_cgm",
    ROOT / "4New_cgm",
]

CANONICAL_ENGINE = DG / "scripts" / "10_diet_microbiome_cgm_mediation.py"

CURRENT4_COHORT = (
    DG / "outputs" / "data" / "00_current4_diet_cgm_cohort.csv"
)

NEW_DIET_COHORT = (
    DG / "outputs" / "new_diet_extension"
    / "data" / "15d0_new_diet_cgm_cohort.csv"
)

CORRECTED_HPDI = (
    DG / "outputs" / "reports" / "new_diet_extension"
    / "hpdi_correction_audit" / "corrected_scores"
    / "hpdi_participant_scores.csv"
)

MICROBIOME = (
    ROOT / "gut_microbiome_deal"
    / "data" / "08_species_clr_zscore.csv"
)

COVARIATES = (
    ROOT / "co-variant"
    / "outputs" / "data" / "02_covariate_master.csv"
)

CGM_PACKAGE = ROOT / "cgm_deal" / "cgm论文新增17指标"

BASE_COVARIATES = [
    "age_years",
    "sex",
    "education_level",
    "smoking_status",
    "sleep_duration_hours_day",
    "physical_activity_met_h_week",
    "vitamin_use",
    "hormone_use",
    "bmi",
    "cgm_device_type",
]

CATEGORICAL = {
    "sex",
    "education_level",
    "smoking_status",
    "cgm_device_type",
}

SCORE_CANDIDATES = {
    "AHEI": [
        "AHEI_z", "ahei_z", "ahei_score_z",
        "ahei_energy_adjusted_z",
        "ahei_score_energy_adjusted_z",
        "mahei7_score_energy_adjusted_z",
        "mahei7_z",
    ],
    "AMED": [
        "AMED_z", "amed_z", "amed_score_z",
        "amed_energy_adjusted_z",
        "amed_score_energy_adjusted_z",
    ],
    "rEDIH": [
        "rEDIH_z", "redih_z", "redih_score_z",
        "redih_energy_adjusted_z",
        "redih_score_energy_adjusted_z",
        "redih_energy_adjusted_score_z",
    ],
}

NEW_SCORE_COLUMNS = {
    "EAT13": "EAT13_z",
    "NOVA4": "NOVA4_z",
    "Carbohydrate_pct": "Carbohydrate_pct_z",
}

HPDI_SCORE_COLUMN = "hpdi_score_energy_adjusted_z"


def require(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def norm_id(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def truthy(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).eq(1)
    return (
        s.astype(str).str.strip().str.lower()
        .isin({"true", "1", "1.0", "yes", "y", "t"})
    )


def require_unique(df: pd.DataFrame, label: str) -> None:
    if "participant_id" not in df.columns:
        raise RuntimeError(f"{label} missing participant_id")
    if df["participant_id"].duplicated().any():
        raise RuntimeError(f"{label} contains duplicate participant_id")


def find_branch(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(p)
        return p
    for p in BRANCH_CANDIDATES:
        if p.is_dir():
            return p
    raise FileNotFoundError("4New_cgm branch not found")


def latest_dir(parent: Path, pattern: str, required_rel: str) -> Path:
    xs = []
    for p in parent.glob(pattern):
        f = p / required_rel
        if f.is_file():
            xs.append((f.stat().st_mtime, p))
    if not xs:
        raise FileNotFoundError(
            f"No {pattern} containing {required_rel} under {parent}"
        )
    xs.sort(reverse=True)
    return xs[0][1]


def latest_final_cgm() -> Path:
    xs = []
    for p in (CGM_PACKAGE / "outputs").glob(
        "run_*/data/07_cgm_paper_extended_phenotypes.csv"
    ):
        if p.is_file():
            xs.append((p.stat().st_mtime, p))
    if not xs:
        raise FileNotFoundError("No finalized Paper-20 CGM table found")
    xs.sort(reverse=True)
    return xs[0][1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sample_hash(ids: pd.Series) -> str:
    vals = sorted(norm_id(ids).astype(str).tolist())
    return hashlib.sha256(
        "\n".join(vals).encode("utf-8")
    ).hexdigest()


def find_column(columns, candidates, label):
    cols = list(columns)
    exact = {str(c): c for c in cols}
    lower = {str(c).lower(): c for c in cols}
    for cand in candidates:
        if cand in exact:
            return exact[cand]
        if cand.lower() in lower:
            return lower[cand.lower()]
    raise RuntimeError(f"Could not find {label}; tried {candidates}")


def valid_numeric(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x.notna() & np.isfinite(x.to_numpy(float))


def valid_covariate(s: pd.Series, logical: str) -> pd.Series:
    if logical in CATEGORICAL:
        return (
            s.notna()
            & s.astype(str).str.strip().ne("")
            & s.astype(str).str.strip().str.lower().ne("nan")
        )
    return valid_numeric(s)


def build_score_table() -> pd.DataFrame:
    require(CURRENT4_COHORT, "current4 formal cohort")
    require(NEW_DIET_COHORT, "new-diet formal cohort")
    require(CORRECTED_HPDI, "corrected hPDI")

    old = pd.read_csv(CURRENT4_COHORT, low_memory=False)
    new = pd.read_csv(NEW_DIET_COHORT, low_memory=False)
    hpdi = pd.read_csv(CORRECTED_HPDI, low_memory=False)

    for label, df in [
        ("current4", old), ("new-diet", new), ("hPDI", hpdi)
    ]:
        df["participant_id"] = norm_id(df["participant_id"])
        require_unique(df, label)

    ids = pd.DataFrame({
        "participant_id": sorted(
            set(old["participant_id"])
            | set(new["participant_id"])
            | set(hpdi["participant_id"])
        )
    })
    out = ids.copy()

    for exposure in ["AHEI", "AMED", "rEDIH"]:
        src = find_column(
            old.columns, SCORE_CANDIDATES[exposure],
            f"{exposure} score"
        )
        x = old[["participant_id", src]].copy()
        x = x.rename(columns={src: f"{exposure}_z"})
        x[f"{exposure}_z"] = pd.to_numeric(
            x[f"{exposure}_z"], errors="coerce"
        )
        out = out.merge(
            x, on="participant_id", how="left", validate="one_to_one"
        )

    if HPDI_SCORE_COLUMN not in hpdi.columns:
        raise RuntimeError(f"hPDI missing {HPDI_SCORE_COLUMN}")
    h = hpdi[
        ["participant_id", HPDI_SCORE_COLUMN]
    ].rename(columns={HPDI_SCORE_COLUMN: "hPDI_z"}).copy()
    h["hPDI_z"] = pd.to_numeric(h["hPDI_z"], errors="coerce")
    out = out.merge(
        h, on="participant_id", how="left", validate="one_to_one"
    )

    needed = [
        "participant_id", "mean_daily_energy_kcal",
        *NEW_SCORE_COLUMNS.values(),
    ]
    miss = sorted(set(needed) - set(new.columns))
    if miss:
        raise RuntimeError(f"New-diet cohort missing {miss}")

    n = new[needed].copy().rename(columns={
        NEW_SCORE_COLUMNS[k]: f"{k}_z"
        for k in NEW_SCORE_COLUMNS
    })
    for c in [
        "EAT13_z", "NOVA4_z", "Carbohydrate_pct_z",
        "mean_daily_energy_kcal",
    ]:
        n[c] = pd.to_numeric(n[c], errors="coerce")
    out = out.merge(
        n, on="participant_id", how="left", validate="one_to_one"
    )

    require_unique(out, "assembled score table")
    return out


def build_master(final_cgm: Path):
    for p, label in [
        (MICROBIOME, "microbiome"),
        (COVARIATES, "covariates"),
        (final_cgm, "final CGM"),
    ]:
        require(p, label)

    scores = build_score_table()
    micro = pd.read_csv(MICROBIOME, low_memory=False)
    cov = pd.read_csv(COVARIATES, low_memory=False)
    cgm = pd.read_csv(final_cgm, low_memory=False)

    for label, df in [
        ("microbiome", micro),
        ("covariates", cov),
        ("CGM", cgm),
    ]:
        df["participant_id"] = norm_id(df["participant_id"])
        require_unique(df, label)

    required_cov = {
        "participant_id",
        "exclude_diabetes_or_a10",
        *BASE_COVARIATES,
    }
    miss = sorted(required_cov - set(cov.columns))
    if miss:
        raise RuntimeError(f"Covariate master missing {miss}")

    alcohol_col = None
    for c in [
        "alcohol_intake_g_day",
        "alcohol_g_day",
        "alcohol_intake",
    ]:
        if c in cov.columns:
            alcohol_col = c
            break
    if alcohol_col is None:
        raise RuntimeError("No recognized alcohol intake column")

    cov2 = cov[
        [
            "participant_id",
            "exclude_diabetes_or_a10",
            *BASE_COVARIATES,
            alcohol_col,
        ]
    ].copy()
    cov2 = cov2.rename(
        columns={alcohol_col: "alcohol_intake_g_day"}
    )

    e = pd.to_numeric(
        cov2["exclude_diabetes_or_a10"], errors="coerce"
    )
    cov2["primary_eligible"] = ~e.eq(1)

    master = scores.merge(
        cov2.drop(columns=["exclude_diabetes_or_a10"]),
        on="participant_id", how="outer", validate="one_to_one"
    )
    master = master.merge(
        cgm,
        on="participant_id", how="outer",
        validate="one_to_one", suffixes=("", "_cgm")
    )
    return master, micro


def path_covariates(exposure: str) -> list[str]:
    covs = list(BASE_COVARIATES)
    if exposure == "hPDI":
        covs.append("alcohol_intake_g_day")
    if exposure in {"EAT13", "NOVA4", "Carbohydrate_pct"}:
        covs.append("mean_daily_energy_kcal")
    return covs


def prepare_path_data(
    row: pd.Series,
    master: pd.DataFrame,
    micro: pd.DataFrame,
):
    exposure = str(row["exposure"])
    outcome = str(row["outcome_field"])
    species = str(row["species"])

    score_col = f"{exposure}_z"
    outcome_col = f"{outcome}_z"
    covs = path_covariates(exposure)

    need = [
        "participant_id", "primary_eligible",
        score_col, outcome_col, *covs
    ]
    miss = [c for c in need if c not in master.columns]
    if miss:
        raise RuntimeError(
            f"{exposure}/{outcome}: master missing {miss}"
        )
    if species not in micro.columns:
        raise RuntimeError(f"Species absent: {species}")

    d = master[need].merge(
        micro[["participant_id", species]],
        on="participant_id",
        how="inner",
        validate="one_to_one",
    )

    ok = d["primary_eligible"].fillna(False).astype(bool)
    ok &= valid_numeric(d[score_col])
    ok &= valid_numeric(d[outcome_col])
    ok &= valid_numeric(d[species])
    for c in covs:
        ok &= valid_covariate(d[c], c)

    d = d.loc[ok].copy()

    d["_X"] = pd.to_numeric(d[score_col], errors="raise")
    d["_Y"] = pd.to_numeric(d[outcome_col], errors="raise")
    d["_M"] = pd.to_numeric(d[species], errors="raise")

    return d, covs



# Linux/fork worker globals. The large participant tables are built once in
# the parent, then inherited copy-on-write by workers. Statistical functions
# still come from the canonical Step10 engine unchanged.
_WORKER_MASTER = None
_WORKER_MICRO = None
_WORKER_CANONICAL = None


def _run_one_path_parallel(payload):
    idx, row_dict, bootstrap, base_seed = payload
    row = pd.Series(row_dict)

    d, covs = prepare_path_data(
        row,
        _WORKER_MASTER,
        _WORKER_MICRO,
    )

    n = len(d)
    h = sample_hash(d["participant_id"])
    audit_n = int(row["audit_N"])
    audit_h = str(row["audit_sample_sha256"])

    if n != audit_n or h != audit_h:
        raise RuntimeError(
            "Step18a sample mismatch for "
            f"{row['exposure']} / {row['outcome_field']} / "
            f"{row['species']}: fit N/hash={n}/{h}, "
            f"audit={audit_n}/{audit_h}"
        )

    cov_df = _WORKER_CANONICAL.encode_covariates(d, covs)
    x = d["_X"].to_numpy(float)
    m = d["_M"].to_numpy(float)
    y = d["_Y"].to_numpy(float)
    cov = cov_df.to_numpy(float)

    point = _WORKER_CANONICAL.mediation_point_estimate(
        x, m, y, cov
    )

    seed_offset = sum(
        ord(ch)
        for ch in (
            f"{row['exposure']}|"
            f"{row['outcome_field']}|"
            f"{row['species']}"
        )
    )
    boot = _WORKER_CANONICAL.bootstrap_mediation(
        x, m, y, cov,
        n_boot=bootstrap,
        seed=(base_seed + seed_offset) % (2**32 - 1),
    )

    out = {
        "exposure": row["exposure"],
        "role": row.get("role", np.nan),
        "outcome_field": row["outcome_field"],
        "outcome_label": row.get("outcome_label", np.nan),
        "species": row["species"],
        "N": n,
        "sample_sha256": h,
        "n_covariate_columns_after_dummy_encoding": cov.shape[1],
        "covariates": ";".join(covs),
        **point,
        **boot,
    }

    carry = [
        "direct_beta_primary",
        "direct_beta_strict",
        "direct_context_highest",
        "m2_beta_diet",
        "m2_FDR_diet",
        "m2_beta_cgm",
        "m2_FDR_cgm",
        "m3_beta_diet",
        "m3_FDR_diet",
        "m3_beta_cgm",
        "m3_FDR_cgm",
        "cgm_highest_robustness",
        "special_rule_ok_effective",
        "special_high_support_effective",
        "robust_bridge_candidate",
        "highest_priority_bridge_candidate",
    ]
    for c in carry:
        if c in row.index:
            out[c] = row[c]

    out["indirect_same_sign_as_total"] = bool(
        np.isfinite(out["total_effect"])
        and np.sign(out["indirect_effect"])
        == np.sign(out["total_effect"])
    )
    out["indirect_CI_excludes_zero"] = bool(
        out["indirect_CI95_lower"] > 0
        or out["indirect_CI95_upper"] < 0
    )

    direct_beta = pd.to_numeric(
        pd.Series([row.get("direct_beta_primary", np.nan)]),
        errors="coerce",
    ).iloc[0]
    out["indirect_same_sign_as_frozen_direct"] = bool(
        np.isfinite(direct_beta)
        and np.sign(out["indirect_effect"])
        == np.sign(direct_beta)
    )

    out["mediated_proportion_outside_0_1"] = bool(
        np.isfinite(out["mediated_proportion"])
        and (
            out["mediated_proportion"] < 0
            or out["mediated_proportion"] > 1
        )
    )
    out["proportion_potentially_unstable"] = bool(
        out["bootstrap_total_crosses_zero_fraction"] > 0.05
        or not np.isfinite(out["mediated_proportion"])
    )

    return idx, out

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch-dir", default=None)
    ap.add_argument("--step17-dir", default=None)
    ap.add_argument("--step18a-dir", default=None)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Path-level worker processes. Default: 4. Use 1 for sequential.",
    )
    ap.add_argument(
        "--max-paths",
        type=int,
        default=None,
        help=(
            "DEBUG ONLY. Runs only first N paths. FDR from a debug subset "
            "is not a valid final result."
        ),
    )
    args = ap.parse_args()

    if args.bootstrap < 20:
        raise RuntimeError("bootstrap must be >=20")
    if args.workers < 1:
        raise RuntimeError("workers must be >=1")

    branch = find_branch(args.branch_dir)
    outputs = branch / "outputs"

    step17 = (
        Path(args.step17_dir).expanduser().resolve()
        if args.step17_dir
        else latest_dir(
            outputs,
            "paper20_bridge_screen_*",
            "reports/17_robust_bridge_candidates.csv",
        )
    )
    step18a = (
        Path(args.step18a_dir).expanduser().resolve()
        if args.step18a_dir
        else latest_dir(
            outputs,
            "paper20_mediation_interface_audit_*",
            "data/18a_path_complete_case_audit.csv",
        )
    )

    robust_path = (
        step17 / "reports" / "17_robust_bridge_candidates.csv"
    )
    audit_path = (
        step18a / "data" / "18a_path_complete_case_audit.csv"
    )
    require(robust_path, "Step17 robust paths")
    require(audit_path, "Step18a path audit")
    require(CANONICAL_ENGINE, "canonical Step10 mediation engine")

    final_cgm = latest_final_cgm()

    canonical = load_module(
        CANONICAL_ENGINE,
        "canonical_step10_mediation_engine",
    )

    robust = pd.read_csv(robust_path, low_memory=False)
    robust = robust.loc[
        truthy(robust["robust_bridge_candidate"])
    ].copy()

    audit = pd.read_csv(audit_path, low_memory=False)
    audit = audit.loc[
        audit["cohort_mode"].astype(str).eq("primary")
    ].copy()

    keys = ["exposure", "outcome_field", "species"]

    if robust.duplicated(keys).any():
        raise RuntimeError("Duplicate Step17 robust path")
    if audit.duplicated(keys).any():
        raise RuntimeError("Duplicate Step18a primary audit path")

    if set(map(tuple, robust[keys].to_numpy())) != set(
        map(tuple, audit[keys].to_numpy())
    ):
        raise RuntimeError(
            "Step17 robust path identities do not exactly match "
            "Step18a primary audit"
        )

    audit_check = audit[
        keys + ["N", "sample_sha256", "ready"]
    ].copy()
    audit_check = audit_check.rename(columns={
        "N": "audit_N",
        "sample_sha256": "audit_sample_sha256",
        "ready": "audit_ready",
    })

    robust = robust.merge(
        audit_check,
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if not truthy(robust["audit_ready"]).all():
        raise RuntimeError(
            "Step18a contains primary path(s) not ready for mediation"
        )

    # Deterministic order; highest-priority first only affects progress order.
    if "highest_priority_bridge_candidate" in robust.columns:
        robust["_hp"] = truthy(
            robust["highest_priority_bridge_candidate"]
        ).astype(int)
    else:
        robust["_hp"] = 0

    robust = (
        robust.sort_values(
            ["_hp", "exposure", "outcome_field", "species"],
            ascending=[False, True, True, True],
        )
        .drop(columns="_hp")
        .reset_index(drop=True)
    )

    full_n = len(robust)
    debug_subset = args.max_paths is not None
    if args.max_paths is not None:
        robust = robust.head(args.max_paths).copy()

    master, micro = build_master(final_cgm)

    print("=== STEP 18b PAPER-20 PRIMARY MEDIATION ===")
    print(f"CANONICAL_ENGINE={CANONICAL_ENGINE}")
    print("CANONICAL_STATISTICAL_CORE_REUSED=True")
    print(f"STEP17_SOURCE={step17}")
    print(f"STEP18A_SOURCE={step18a}")
    print(f"FINAL_CGM={final_cgm}")
    print(f"FROZEN_FULL_PATH_SET={full_n}")
    print(f"PATHS_THIS_RUN={len(robust)}")
    print(f"BOOTSTRAP={args.bootstrap}")
    print(f"BASE_SEED={args.seed}")
    print(f"WORKERS={args.workers}")
    print("PATH_LEVEL_MULTIPROCESSING=True")
    print("COHORT_MODE=primary")
    print("ANTIBIOTIC_PPI_INCLUDED=False")
    print(f"DEBUG_SUBSET={debug_subset}")
    print("FDR_PRIMARY_FAMILY=within Diet x CGM frozen mediator family")
    print("GLOBAL_FDR=sensitivity_only")

    results = []
    t_all = time.time()

    # Prepare deterministic tasks once. The statistical core and bootstrap
    # rules are unchanged; only independent paths are distributed across
    # processes.
    tasks = [
        (
            i,
            row.to_dict(),
            args.bootstrap,
            args.seed,
        )
        for i, (_, row) in enumerate(
            robust.iterrows(),
            start=1,
        )
    ]

    global _WORKER_MASTER, _WORKER_MICRO, _WORKER_CANONICAL
    _WORKER_MASTER = master
    _WORKER_MICRO = micro
    _WORKER_CANONICAL = canonical

    records = []
    completed = 0

    if args.workers == 1:
        iterator = map(_run_one_path_parallel, tasks)
        pool = None
    else:
        # This project runs on Linux. fork preserves the already-loaded
        # read-only participant tables efficiently via copy-on-write.
        ctx = mp.get_context("fork")
        pool = ctx.Pool(processes=args.workers)
        iterator = pool.imap_unordered(
            _run_one_path_parallel,
            tasks,
            chunksize=1,
        )

    try:
        for idx, out in iterator:
            records.append((idx, out))
            completed += 1

            if (
                completed == 1
                or completed % 25 == 0
                or completed == len(records) == len(tasks)
                or completed == len(tasks)
            ):
                elapsed = time.time() - t_all
                rate = elapsed / completed
                remain = rate * (len(tasks) - completed)
                print(
                    f"[{completed}/{len(tasks)} completed] "
                    f"last={out['exposure']} -> species -> "
                    f"{out['outcome_field']} | "
                    f"N={out['N']} "
                    f"p_boot={out['bootstrap_p_indirect']:.4g} | "
                    f"elapsed={elapsed:.1f}s "
                    f"ETA={remain/60:.1f}min"
                )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    records.sort(key=lambda x: x[0])
    results = [out for _, out in records]

    results = pd.DataFrame(results)

    # DEBUG subset must never be mistaken for the full primary inferential run.
    if debug_subset:
        results["FDR_BH_within_score_outcome"] = np.nan
        results["FDR_BH_global_all_mediation"] = np.nan
        results["mediation_FDR05_primary"] = False
        results["candidate_mediator_consistent"] = False
        results["bridge_direction_consistent_mediator"] = False
    else:
        results["FDR_BH_within_score_outcome"] = np.nan
        for _, idx in results.groupby(
            ["exposure", "outcome_field"]
        ).groups.items():
            idx = list(idx)
            results.loc[
                idx, "FDR_BH_within_score_outcome"
            ] = canonical.bh_fdr(
                results.loc[idx, "bootstrap_p_indirect"]
            )

        results["FDR_BH_global_all_mediation"] = (
            canonical.bh_fdr(
                results["bootstrap_p_indirect"]
            )
        )

        results["mediation_FDR05_primary"] = (
            results["FDR_BH_within_score_outcome"] < 0.05
        )

        # Canonical Step10 interpretation flag.
        results["candidate_mediator_consistent"] = (
            results["mediation_FDR05_primary"]
            & results["indirect_same_sign_as_total"]
            & results["indirect_CI_excludes_zero"]
        )

        # Additional Paper-20 bridge-direction annotation.
        results["bridge_direction_consistent_mediator"] = (
            results["candidate_mediator_consistent"]
            & results["indirect_same_sign_as_frozen_direct"]
        )

    if "highest_priority_bridge_candidate" in results.columns:
        hp = truthy(results["highest_priority_bridge_candidate"])
    else:
        hp = pd.Series(False, index=results.index)

    results["highest_priority_significant_mediation"] = (
        hp & results["mediation_FDR05_primary"]
    )
    results["highest_priority_direction_consistent_mediation"] = (
        hp & results["bridge_direction_consistent_mediator"]
    )

    results = results.sort_values(
        [
            "mediation_FDR05_primary",
            "bridge_direction_consistent_mediator",
            "FDR_BH_within_score_outcome",
            "bootstrap_p_indirect",
        ],
        ascending=[False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)

    summary = (
        results.groupby(
            ["exposure", "role", "outcome_field", "outcome_label"],
            as_index=False,
            dropna=False,
        )
        .agg(
            paths_tested=("species", "size"),
            unique_species=("species", "nunique"),
            median_N=("N", "median"),
            significant_mediation_FDR05=(
                "mediation_FDR05_primary", "sum"
            ),
            canonical_direction_consistent=(
                "candidate_mediator_consistent", "sum"
            ),
            bridge_direction_consistent=(
                "bridge_direction_consistent_mediator", "sum"
            ),
            highest_priority_significant=(
                "highest_priority_significant_mediation", "sum"
            ),
            highest_priority_direction_consistent=(
                "highest_priority_direction_consistent_mediation", "sum"
            ),
            min_bootstrap_p=("bootstrap_p_indirect", "min"),
            min_FDR_primary=(
                "FDR_BH_within_score_outcome", "min"
            ),
        )
        .sort_values(["exposure", "outcome_field"])
    )

    if debug_subset:
        sig = results.iloc[0:0].copy()
        consistent = results.iloc[0:0].copy()
    else:
        sig = results.loc[
            results["mediation_FDR05_primary"]
        ].copy()
        consistent = results.loc[
            results["bridge_direction_consistent_mediator"]
        ].copy()

    recurrence = (
        consistent.groupby("species", as_index=False)
        .agg(
            significant_paths=("species", "size"),
            diets=("exposure", "nunique"),
            outcomes=("outcome_field", "nunique"),
            exposure_list=(
                "exposure",
                lambda x: ";".join(sorted(set(map(str, x))))
            ),
            outcome_list=(
                "outcome_field",
                lambda x: ";".join(sorted(set(map(str, x))))
            ),
            highest_priority_paths=(
                "highest_priority_direction_consistent_mediation",
                "sum",
            ),
        )
        .sort_values(
            ["significant_paths", "diets", "outcomes", "species"],
            ascending=[False, False, False, True],
        )
        if len(consistent)
        else pd.DataFrame(
            columns=[
                "species", "significant_paths", "diets", "outcomes",
                "exposure_list", "outcome_list",
                "highest_priority_paths",
            ]
        )
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = (
        f"_debug{len(results)}_b{args.bootstrap}"
        if debug_subset else "_primary"
    )
    outdir = outputs / f"paper20_mediation_{stamp}{suffix}"
    models_dir = outdir / "models"
    reports = outdir / "reports"
    models_dir.mkdir(parents=True, exist_ok=False)
    reports.mkdir()

    model_out = models_dir / "18b_primary_mediation_paths.csv"
    summary_out = reports / "18b_primary_mediation_context_summary.csv"
    sig_out = reports / "18b_primary_significant.csv"
    consistent_out = (
        reports / "18b_primary_direction_consistent_significant.csv"
    )
    recurrence_out = reports / "18b_primary_species_recurrence.csv"
    txt_out = reports / "18b_primary_mediation_summary.txt"

    results.to_csv(model_out, index=False)
    summary.to_csv(summary_out, index=False)
    sig.to_csv(sig_out, index=False)
    consistent.to_csv(consistent_out, index=False)
    recurrence.to_csv(recurrence_out, index=False)

    elapsed = time.time() - t_all

    lines = [
        "=== STEP 18b PAPER-20 PRIMARY MEDIATION SUMMARY ===",
        f"CANONICAL_ENGINE={CANONICAL_ENGINE}",
        "CANONICAL_STATISTICAL_CORE_REUSED=True",
        f"STEP17_SOURCE={step17}",
        f"STEP18A_SOURCE={step18a}",
        f"FINAL_CGM={final_cgm}",
        f"DEBUG_SUBSET={debug_subset}",
        f"BOOTSTRAP={args.bootstrap}",
        f"WORKERS={args.workers}",
        "PATH_LEVEL_MULTIPROCESSING=True",
        f"FROZEN_FULL_PATH_SET={full_n}",
        f"PATHS_TESTED={len(results)}",
        f"DIET_CGM_FAMILIES_TESTED={results[['exposure','outcome_field']].drop_duplicates().shape[0]}",
        "COHORT_MODE=primary",
        "ANTIBIOTIC_PPI_INCLUDED=False",
        "PRIMARY_FDR=BH within frozen Diet x CGM mediator family",
        "GLOBAL_FDR=sensitivity only",
        f"AUDIT_N_HASH_MATCH_ALL=True",
        f"ELAPSED_SECONDS={elapsed:.1f}",
    ]

    if debug_subset:
        lines += [
            "PRIMARY_INFERENCE_VALID=False",
            "REASON=debug subset does not contain complete mediator families",
        ]
    else:
        lines += [
            "PRIMARY_INFERENCE_VALID=True",
            f"MEDIATION_FDR05={int(results['mediation_FDR05_primary'].sum())}",
            f"CANONICAL_DIRECTION_CONSISTENT={int(results['candidate_mediator_consistent'].sum())}",
            f"BRIDGE_DIRECTION_CONSISTENT={int(results['bridge_direction_consistent_mediator'].sum())}",
            f"HIGHEST_PRIORITY_SIGNIFICANT={int(results['highest_priority_significant_mediation'].sum())}",
            f"HIGHEST_PRIORITY_DIRECTION_CONSISTENT={int(results['highest_priority_direction_consistent_mediation'].sum())}",
            f"MEDIATED_PROPORTION_OUTSIDE_0_1={int(results['mediated_proportion_outside_0_1'].sum())}",
            f"POTENTIALLY_UNSTABLE_PROPORTION={int(results['proportion_potentially_unstable'].sum())}",
        ]

    lines += [
        "",
        "--- CONTEXT SUMMARY ---",
        summary.to_string(index=False),
        "",
        "--- TOP RECURRENT DIRECTION-CONSISTENT MEDIATORS ---",
        (
            "NONE"
            if recurrence.empty
            else recurrence.head(30).to_string(index=False)
        ),
        "",
        "INTERPRETATION RULES:",
        "- The full Step17 robust path set is the frozen primary mediation family.",
        "- Highest-priority bridge status is annotation only.",
        "- Primary BH-FDR is within each frozen Diet x CGM mediator family.",
        "- Global mediation BH-FDR is sensitivity only.",
        "- candidate_mediator_consistent preserves the canonical Step10 rule: FDR<0.05 + indirect CI excludes zero + indirect same sign as fitted total effect.",
        "- bridge_direction_consistent_mediator additionally requires the indirect effect to match the frozen direct Diet->CGM direction.",
        "- Antibiotic/PPI are not included here; they remain a later protocol-window sensitivity.",
        "- Carbohydrate_pct remains exploratory.",
        "- Mean glucose and GMI are highly dependent and are not independent replication.",
        "- These are cross-sectional mediation-style associations, not causal or temporal mediation.",
        "",
        f"MODEL_TABLE={model_out}",
        f"CONTEXT_SUMMARY={summary_out}",
        f"SIGNIFICANT={sig_out}",
        f"DIRECTION_CONSISTENT={consistent_out}",
        f"RECURRENCE={recurrence_out}",
        f"OUTPUT_DIR={outdir}",
    ]

    txt_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "status": (
            "debug_subset"
            if debug_subset else "completed"
        ),
        "canonical_engine": str(CANONICAL_ENGINE),
        "canonical_engine_sha256": sha256(CANONICAL_ENGINE),
        "canonical_statistical_core_reused": True,
        "step17_source": str(step17),
        "step18a_source": str(step18a),
        "final_cgm": str(final_cgm),
        "bootstrap": int(args.bootstrap),
        "workers": int(args.workers),
        "path_level_multiprocessing": True,
        "seed": int(args.seed),
        "frozen_full_path_set": int(full_n),
        "paths_tested": int(len(results)),
        "cohort_mode": "primary",
        "antibiotic_ppi_included": False,
        "debug_subset": bool(debug_subset),
        "primary_inference_valid": bool(not debug_subset),
        "outputs": {
            "model_table": str(model_out),
            "context_summary": str(summary_out),
            "significant": str(sig_out),
            "direction_consistent": str(consistent_out),
            "recurrence": str(recurrence_out),
            "summary": str(txt_out),
        },
    }
    if not debug_subset:
        manifest.update({
            "mediation_fdr05": int(
                results["mediation_FDR05_primary"].sum()
            ),
            "canonical_direction_consistent": int(
                results["candidate_mediator_consistent"].sum()
            ),
            "bridge_direction_consistent": int(
                results["bridge_direction_consistent_mediator"].sum()
            ),
        })

    (reports / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
