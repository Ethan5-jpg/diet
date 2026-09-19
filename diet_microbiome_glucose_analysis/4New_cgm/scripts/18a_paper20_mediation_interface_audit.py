#!/usr/bin/env python3
"""
Step 18a — Paper-20 mediation interface / complete-case audit.

Purpose
-------
Audit the frozen Step17 robust bridge set BEFORE any mediation model is fit.

This script:
- fits NO models;
- runs NO bootstrap;
- recalculates NO FDR;
- creates NO new bridge candidates;
- uses ALL Step17 robust_bridge_candidate paths, not only the highest-priority subset.

For each frozen Diet -> species -> CGM path it verifies:
1. exact diet score availability;
2. exact full-taxonomy species availability in the CLR-Z table;
3. Paper-20 CGM Z outcome availability;
4. primary / strict diabetes-A10 eligibility;
5. prespecified mediation covariates;
6. complete-case N in both primary and strict cohorts;
7. non-zero variance of X, M and Y;
8. deterministic sample hashes for later primary/strict mediation QC.

Covariate rules
---------------
Existing scores AHEI / AMED / rEDIH:
    age, sex, education, smoking, sleep, physical activity,
    vitamin use, hormone use, BMI, CGM device type.

Corrected hPDI:
    same canonical covariates + alcohol intake.

New scores EAT13 / NOVA4 / Carbohydrate_pct:
    same canonical covariates + mean_daily_energy_kcal.

Antibiotic/PPI are NOT part of this primary audit; they remain a later
protocol-window sensitivity, consistent with the frozen analysis plan.

Interpretation
--------------
This is only an interface and sample-selection audit. Passing this step means
the 1490-path frozen candidate set is technically ready for the primary and
strict cross-sectional mediation-style analyses. It is not evidence of
mediation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

MIN_N = 100

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
        "AHEI_z",
        "ahei_z",
        "ahei_score_z",
        "ahei_energy_adjusted_z",
        "ahei_score_energy_adjusted_z",
        "mahei7_score_energy_adjusted_z",
        "mahei7_z",
    ],
    "AMED": [
        "AMED_z",
        "amed_z",
        "amed_score_z",
        "amed_energy_adjusted_z",
        "amed_score_energy_adjusted_z",
    ],
    "rEDIH": [
        "rEDIH_z",
        "redih_z",
        "redih_score_z",
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


def norm_id(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def require_unique(df: pd.DataFrame, label: str) -> None:
    if "participant_id" not in df.columns:
        raise RuntimeError(f"{label} missing participant_id")
    if df["participant_id"].duplicated().any():
        ex = (
            df.loc[
                df["participant_id"].duplicated(keep=False),
                "participant_id",
            ]
            .head(20)
            .tolist()
        )
        raise RuntimeError(
            f"{label} has duplicate participant_id values: {ex}"
        )


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


def latest_final_cgm(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        require(p, "finalized Paper-20 CGM")
        return p

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
    payload = "\n".join(vals).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def find_column(columns, candidates, label):
    cols = list(columns)
    exact = {str(c): c for c in cols}
    lower = {str(c).lower(): c for c in cols}
    for cand in candidates:
        if cand in exact:
            return exact[cand]
        if cand.lower() in lower:
            return lower[cand.lower()]
    raise RuntimeError(
        f"Could not find {label}. Tried {candidates}"
    )


def valid_numeric(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x.notna() & np.isfinite(x.to_numpy(float))


def valid_raw_covariate(s: pd.Series, logical: str) -> pd.Series:
    if logical in CATEGORICAL:
        return (
            s.notna()
            & s.astype(str).str.strip().ne("")
            & s.astype(str).str.strip().str.lower().ne("nan")
        )
    return valid_numeric(s)


def reconstruct_eligibility(cov: pd.DataFrame) -> pd.DataFrame:
    if "exclude_diabetes_or_a10" not in cov.columns:
        raise RuntimeError(
            "Covariate master missing exclude_diabetes_or_a10"
        )
    e = pd.to_numeric(
        cov["exclude_diabetes_or_a10"], errors="coerce"
    )
    out = cov[["participant_id"]].copy()
    out["primary_eligible"] = ~e.eq(1)
    out["strict_eligible"] = e.eq(0)
    out["eligibility_status_known"] = e.notna()
    out["known_positive_excluded"] = e.eq(1)
    return out


def build_score_table() -> tuple[pd.DataFrame, list[dict]]:
    require(CURRENT4_COHORT, "current4 formal cohort")
    require(NEW_DIET_COHORT, "new-diet formal cohort")
    require(CORRECTED_HPDI, "corrected hPDI scores")

    old = pd.read_csv(CURRENT4_COHORT, low_memory=False)
    new = pd.read_csv(NEW_DIET_COHORT, low_memory=False)
    hpdi = pd.read_csv(CORRECTED_HPDI, low_memory=False)

    for label, df in [
        ("current4 cohort", old),
        ("new-diet cohort", new),
        ("corrected hPDI", hpdi),
    ]:
        df["participant_id"] = norm_id(df["participant_id"])
        require_unique(df, label)

    score_parts = [old[["participant_id"]].copy()]
    audit = []

    # Existing AHEI / AMED / rEDIH.
    for exposure in ["AHEI", "AMED", "rEDIH"]:
        source_col = find_column(
            old.columns,
            SCORE_CANDIDATES[exposure],
            f"{exposure} score",
        )
        x = old[["participant_id", source_col]].copy()
        x = x.rename(columns={source_col: f"{exposure}_z"})
        x[f"{exposure}_z"] = pd.to_numeric(
            x[f"{exposure}_z"], errors="coerce"
        )
        score_parts.append(x)
        audit.append({
            "exposure": exposure,
            "source_file": str(CURRENT4_COHORT),
            "source_column": source_col,
            "extra_covariate": "none",
        })

    # Corrected hPDI must override any legacy hPDI in current4.
    if HPDI_SCORE_COLUMN not in hpdi.columns:
        raise RuntimeError(
            f"Corrected hPDI missing {HPDI_SCORE_COLUMN}"
        )
    h = hpdi[
        ["participant_id", HPDI_SCORE_COLUMN]
    ].copy().rename(
        columns={HPDI_SCORE_COLUMN: "hPDI_z"}
    )
    h["hPDI_z"] = pd.to_numeric(
        h["hPDI_z"], errors="coerce"
    )
    score_parts.append(h)
    audit.append({
        "exposure": "hPDI",
        "source_file": str(CORRECTED_HPDI),
        "source_column": HPDI_SCORE_COLUMN,
        "extra_covariate": "alcohol_intake_g_day",
    })

    # New exposures + mean energy.
    needed = [
        "participant_id",
        "mean_daily_energy_kcal",
        *NEW_SCORE_COLUMNS.values(),
    ]
    missing = sorted(set(needed) - set(new.columns))
    if missing:
        raise RuntimeError(
            f"New-diet cohort missing: {missing}"
        )
    n = new[needed].copy()
    rename = {
        NEW_SCORE_COLUMNS[k]: f"{k}_z"
        for k in NEW_SCORE_COLUMNS
    }
    n = n.rename(columns=rename)
    for col in [
        "EAT13_z", "NOVA4_z", "Carbohydrate_pct_z",
        "mean_daily_energy_kcal",
    ]:
        n[col] = pd.to_numeric(n[col], errors="coerce")
    score_parts.append(n)

    for exposure, source_col in NEW_SCORE_COLUMNS.items():
        audit.append({
            "exposure": exposure,
            "source_file": str(NEW_DIET_COHORT),
            "source_column": source_col,
            "extra_covariate": "mean_daily_energy_kcal",
        })

    # Merge onto union of IDs from old/new/hPDI.
    ids = pd.DataFrame({
        "participant_id": sorted(
            set(old["participant_id"])
            | set(new["participant_id"])
            | set(hpdi["participant_id"])
        )
    })
    scores = ids.copy()
    for part in score_parts[1:]:
        scores = scores.merge(
            part,
            on="participant_id",
            how="left",
            validate="one_to_one",
        )

    require_unique(scores, "assembled score table")
    return scores, audit


def build_master(final_cgm: Path):
    require(MICROBIOME, "microbiome CLR-Z")
    require(COVARIATES, "covariate master")
    require(final_cgm, "final Paper-20 CGM")

    scores, score_audit = build_score_table()

    micro = pd.read_csv(MICROBIOME, low_memory=False)
    cov = pd.read_csv(COVARIATES, low_memory=False)
    cgm = pd.read_csv(final_cgm, low_memory=False)

    for label, df in [
        ("microbiome", micro),
        ("covariates", cov),
        ("Paper-20 CGM", cgm),
    ]:
        df["participant_id"] = norm_id(df["participant_id"])
        require_unique(df, label)

    # Required covariates.
    required_cov = {
        "participant_id",
        "exclude_diabetes_or_a10",
        *BASE_COVARIATES,
    }
    missing_cov = sorted(required_cov - set(cov.columns))
    if missing_cov:
        raise RuntimeError(
            f"Covariate master missing required columns: {missing_cov}"
        )

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
        raise RuntimeError(
            "hPDI primary mediation requires alcohol intake, but no "
            "recognized alcohol column was found"
        )

    cov_keep = [
        "participant_id",
        "exclude_diabetes_or_a10",
        *BASE_COVARIATES,
        alcohol_col,
    ]
    cov2 = cov[cov_keep].copy()
    cov2 = cov2.rename(
        columns={alcohol_col: "alcohol_intake_g_day"}
    )

    elig = reconstruct_eligibility(cov2)

    # Keep all CGM columns because exact needed outcomes are determined by
    # the frozen Step17 candidate table.
    master = scores.merge(
        cov2.drop(columns=["exclude_diabetes_or_a10"]),
        on="participant_id",
        how="outer",
        validate="one_to_one",
    )
    master = master.merge(
        elig,
        on="participant_id",
        how="outer",
        validate="one_to_one",
    )
    master = master.merge(
        cgm,
        on="participant_id",
        how="outer",
        validate="one_to_one",
        suffixes=("", "_cgm"),
    )

    # Microbiome is intentionally kept separate to avoid a huge duplicate
    # master in memory; path-specific species is merged by participant_id.
    return master, micro, score_audit


def path_covariates(exposure: str) -> list[str]:
    covs = list(BASE_COVARIATES)
    if exposure == "hPDI":
        covs.append("alcohol_intake_g_day")
    if exposure in {"EAT13", "NOVA4", "Carbohydrate_pct"}:
        covs.append("mean_daily_energy_kcal")
    return covs


def path_score_col(exposure: str) -> str:
    return f"{exposure}_z"


def audit_path(
    row: pd.Series,
    master: pd.DataFrame,
    micro: pd.DataFrame,
    cohort_mode: str,
) -> dict:
    exposure = str(row["exposure"])
    outcome = str(row["outcome_field"])
    species = str(row["species"])

    score_col = path_score_col(exposure)
    outcome_col = outcome + "_z"
    covs = path_covariates(exposure)
    elig_col = (
        "primary_eligible"
        if cohort_mode == "primary"
        else "strict_eligible"
    )

    missing_master = [
        c for c in [score_col, outcome_col, elig_col, *covs]
        if c not in master.columns
    ]
    if missing_master:
        raise RuntimeError(
            f"{exposure}/{outcome}: master missing {missing_master}"
        )
    if species not in micro.columns:
        raise RuntimeError(
            f"Exact species absent from microbiome table: {species}"
        )

    left = master[
        ["participant_id", score_col, outcome_col, elig_col, *covs]
    ].copy()
    med = micro[["participant_id", species]].copy()

    d = left.merge(
        med,
        on="participant_id",
        how="inner",
        validate="one_to_one",
    )

    ok = d[elig_col].fillna(False).astype(bool)
    ok &= valid_numeric(d[score_col])
    ok &= valid_numeric(d[outcome_col])
    ok &= valid_numeric(d[species])

    for cov in covs:
        ok &= valid_raw_covariate(d[cov], cov)

    cc = d.loc[ok].copy()

    x = pd.to_numeric(cc[score_col], errors="coerce").to_numpy(float)
    m = pd.to_numeric(cc[species], errors="coerce").to_numpy(float)
    y = pd.to_numeric(cc[outcome_col], errors="coerce").to_numpy(float)

    def var_ok(a):
        return (
            len(a) >= 2
            and np.isfinite(a).all()
            and float(np.nanstd(a, ddof=1)) > 0
        )

    return {
        "exposure": exposure,
        "role": row.get("role", np.nan),
        "outcome_field": outcome,
        "outcome_label": row.get("outcome_label", np.nan),
        "species": species,
        "cohort_mode": cohort_mode,
        "score_column": score_col,
        "outcome_column": outcome_col,
        "covariates": ";".join(covs),
        "N": int(len(cc)),
        "sample_sha256": sample_hash(cc["participant_id"]),
        "x_nonzero_variance": bool(var_ok(x)),
        "m_nonzero_variance": bool(var_ok(m)),
        "y_nonzero_variance": bool(var_ok(y)),
        "N_ge_minimum": bool(len(cc) >= MIN_N),
        "ready": bool(
            len(cc) >= MIN_N
            and var_ok(x)
            and var_ok(m)
            and var_ok(y)
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch-dir", default=None)
    ap.add_argument("--step17-dir", default=None)
    ap.add_argument("--cgm-csv", default=None)
    args = ap.parse_args()

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
    robust_path = (
        step17 / "reports" / "17_robust_bridge_candidates.csv"
    )
    require(robust_path, "Step17 robust bridge candidates")

    final_cgm = latest_final_cgm(args.cgm_csv)

    robust = pd.read_csv(robust_path, low_memory=False)
    required_bridge = {
        "exposure",
        "outcome_field",
        "species",
        "robust_bridge_candidate",
    }
    missing = sorted(required_bridge - set(robust.columns))
    if missing:
        raise RuntimeError(
            f"Step17 robust table missing: {missing}"
        )

    keep = robust["robust_bridge_candidate"]
    if not pd.api.types.is_bool_dtype(keep):
        keep = (
            keep.astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "1.0", "yes", "y", "t"})
        )
    robust = robust.loc[keep].copy()

    if robust.empty:
        raise RuntimeError("Frozen Step17 robust path set is empty")
    if robust.duplicated(
        ["exposure", "outcome_field", "species"]
    ).any():
        raise RuntimeError("Duplicate frozen bridge path")

    master, micro, score_audit = build_master(final_cgm)

    outcomes = sorted(
        robust["outcome_field"].astype(str).unique()
    )
    missing_outcomes = [
        o for o in outcomes if o + "_z" not in master.columns
    ]
    if missing_outcomes:
        raise RuntimeError(
            "Final CGM missing Z columns for: "
            + ",".join(missing_outcomes)
        )

    species = sorted(robust["species"].astype(str).unique())
    missing_species = [s for s in species if s not in micro.columns]
    if missing_species:
        raise RuntimeError(
            f"{len(missing_species)} frozen exact species absent from "
            f"microbiome table; first={missing_species[:5]}"
        )

    print("=== STEP 18a PAPER-20 MEDIATION INTERFACE AUDIT ===")
    print(f"STEP17_SOURCE={step17}")
    print(f"FINAL_CGM={final_cgm}")
    print(f"FROZEN_ROBUST_PATHS={len(robust)}")
    print(f"FROZEN_UNIQUE_EXACT_SPECIES={len(species)}")
    print(
        f"FROZEN_DIET_CGM_CONTEXTS="
        f"{robust[['exposure','outcome_field']].drop_duplicates().shape[0]}"
    )
    print("MODELS_FIT=False")
    print("BOOTSTRAP_RUN=False")
    print("FDR_RECALCULATED=False")
    print("HIGHEST_PRIORITY_ONLY=False")
    print("ANTIBIOTIC_PPI_INCLUDED=False")
    print("PRIMARY_AND_STRICT_SAME_FROZEN_PATH_SET=True")

    rows = []
    for i, (_, row) in enumerate(robust.iterrows(), start=1):
        if i == 1 or i % 100 == 0 or i == len(robust):
            print(f"AUDIT_PATH={i}/{len(robust)}")
        rows.append(audit_path(row, master, micro, "primary"))
        rows.append(audit_path(row, master, micro, "strict"))

    audit = pd.DataFrame(rows)

    primary = audit.loc[audit["cohort_mode"].eq("primary")].copy()
    strict = audit.loc[audit["cohort_mode"].eq("strict")].copy()

    # Exact same frozen path keys must exist in both cohort audits.
    keys = ["exposure", "outcome_field", "species"]
    pkeys = set(map(tuple, primary[keys].to_numpy()))
    skeys = set(map(tuple, strict[keys].to_numpy()))
    exact_path_set_match = pkeys == skeys

    comparison = primary[
        keys + ["N", "sample_sha256", "ready"]
    ].merge(
        strict[keys + ["N", "sample_sha256", "ready"]],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_primary", "_strict"),
    )
    comparison["N_lost_strict"] = (
        comparison["N_primary"] - comparison["N_strict"]
    )
    comparison["strict_subset_N_ok"] = (
        comparison["N_strict"] <= comparison["N_primary"]
    )

    context = (
        audit.groupby(
            ["cohort_mode", "exposure", "role",
             "outcome_field", "outcome_label"],
            as_index=False,
            dropna=False,
        )
        .agg(
            paths=("species", "size"),
            unique_species=("species", "nunique"),
            min_N=("N", "min"),
            median_N=("N", "median"),
            max_N=("N", "max"),
            ready_paths=("ready", "sum"),
            unique_sample_hashes=("sample_sha256", "nunique"),
        )
        .sort_values(
            ["cohort_mode", "exposure", "outcome_field"]
        )
    )

    by_exposure = (
        audit.groupby(
            ["cohort_mode", "exposure", "role"],
            as_index=False,
            dropna=False,
        )
        .agg(
            paths=("species", "size"),
            min_N=("N", "min"),
            median_N=("N", "median"),
            max_N=("N", "max"),
            ready_paths=("ready", "sum"),
            outcomes=("outcome_field", "nunique"),
            unique_species=("species", "nunique"),
        )
        .sort_values(["cohort_mode", "exposure"])
    )

    failed = audit.loc[~audit["ready"]].copy()
    strict_subset_fail = int(
        (~comparison["strict_subset_N_ok"]).sum()
    )

    ready = bool(
        failed.empty
        and exact_path_set_match
        and strict_subset_fail == 0
        and len(primary) == len(robust)
        and len(strict) == len(robust)
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = outputs / f"paper20_mediation_interface_audit_{stamp}"
    reports = out / "reports"
    data_dir = out / "data"
    reports.mkdir(parents=True, exist_ok=False)
    data_dir.mkdir()

    audit_path_out = data_dir / "18a_path_complete_case_audit.csv"
    comparison_path = reports / "18a_primary_vs_strict_N_comparison.csv"
    context_path = reports / "18a_context_complete_case_summary.csv"
    exposure_path = reports / "18a_exposure_complete_case_summary.csv"
    source_path = reports / "18a_source_score_mapping.csv"
    failed_path = reports / "18a_not_ready_paths.csv"
    txt_path = reports / "18a_mediation_interface_summary.txt"

    audit.to_csv(audit_path_out, index=False)
    comparison.to_csv(comparison_path, index=False)
    context.to_csv(context_path, index=False)
    by_exposure.to_csv(exposure_path, index=False)
    pd.DataFrame(score_audit).to_csv(source_path, index=False)
    failed.to_csv(failed_path, index=False)

    source_files = [
        robust_path,
        CURRENT4_COHORT,
        NEW_DIET_COHORT,
        CORRECTED_HPDI,
        MICROBIOME,
        COVARIATES,
        final_cgm,
    ]

    lines = [
        "=== STEP 18a PAPER-20 MEDIATION INTERFACE AUDIT SUMMARY ===",
        f"STEP17_SOURCE={step17}",
        f"FINAL_CGM={final_cgm}",
        f"FROZEN_ROBUST_PATHS={len(robust)}",
        f"FROZEN_UNIQUE_EXACT_SPECIES={len(species)}",
        (
            "FROZEN_DIET_CGM_CONTEXTS="
            f"{robust[['exposure','outcome_field']].drop_duplicates().shape[0]}"
        ),
        "MODELS_FIT=False",
        "BOOTSTRAP_RUN=False",
        "FDR_RECALCULATED=False",
        "HIGHEST_PRIORITY_ONLY=False",
        "ANTIBIOTIC_PPI_INCLUDED=False",
        "PRIMARY_AND_STRICT_SAME_FROZEN_PATH_SET=True",
        "",
        f"PRIMARY_PATHS_AUDITED={len(primary)}",
        f"STRICT_PATHS_AUDITED={len(strict)}",
        f"EXACT_PRIMARY_STRICT_PATH_SET_MATCH={exact_path_set_match}",
        f"PRIMARY_NOT_READY={int((~primary['ready']).sum())}",
        f"STRICT_NOT_READY={int((~strict['ready']).sum())}",
        f"STRICT_SUBSET_N_VIOLATIONS={strict_subset_fail}",
        f"READY_FOR_MEDIATION={ready}",
        "",
        "--- BY EXPOSURE ---",
        by_exposure.to_string(index=False),
        "",
        "--- CONTEXT COMPLETE-CASE SUMMARY ---",
        context.to_string(index=False),
        "",
        "--- SCORE SOURCE MAP ---",
        pd.DataFrame(score_audit).to_string(index=False),
        "",
        "INTERPRETATION RULES:",
        "- The Step17 robust 1490-path set is frozen; this audit cannot add/drop paths for statistical reasons.",
        "- Primary and strict analyses use the exact same frozen path identities.",
        "- Existing AHEI/AMED/rEDIH use canonical mediation covariates.",
        "- Corrected hPDI uses corrected score plus hPDI-specific alcohol adjustment.",
        "- EAT13/NOVA4/Carbohydrate_pct additionally adjust for mean_daily_energy_kcal.",
        "- Antibiotic/PPI remain a later protocol-window sensitivity.",
        "- Passing this audit is technical readiness only, not evidence of causal mediation.",
        "",
        f"PATH_AUDIT={audit_path_out}",
        f"PRIMARY_STRICT_COMPARISON={comparison_path}",
        f"CONTEXT_SUMMARY={context_path}",
        f"EXPOSURE_SUMMARY={exposure_path}",
        f"FAILED_PATHS={failed_path}",
        f"OUTPUT_DIR={out}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "status": "ready" if ready else "audit_failed",
        "models_fit": False,
        "bootstrap_run": False,
        "fdr_recalculated": False,
        "highest_priority_only": False,
        "antibiotic_ppi_included": False,
        "frozen_robust_paths": int(len(robust)),
        "frozen_unique_exact_species": int(len(species)),
        "frozen_diet_cgm_contexts": int(
            robust[["exposure", "outcome_field"]]
            .drop_duplicates().shape[0]
        ),
        "primary_paths_audited": int(len(primary)),
        "strict_paths_audited": int(len(strict)),
        "exact_primary_strict_path_set_match": bool(exact_path_set_match),
        "primary_not_ready": int((~primary["ready"]).sum()),
        "strict_not_ready": int((~strict["ready"]).sum()),
        "strict_subset_N_violations": int(strict_subset_fail),
        "ready_for_mediation": bool(ready),
        "source_sha256": {
            str(p): sha256(p) for p in source_files
        },
        "outputs": {
            "path_audit": str(audit_path_out),
            "primary_strict_comparison": str(comparison_path),
            "context_summary": str(context_path),
            "exposure_summary": str(exposure_path),
            "failed_paths": str(failed_path),
            "summary": str(txt_path),
        },
    }
    (
        reports / "manifest.json"
    ).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
