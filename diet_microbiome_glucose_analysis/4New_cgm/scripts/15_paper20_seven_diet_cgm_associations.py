#!/usr/bin/env python3
"""
Step 15 — seven diets x Paper-20 Diet->CGM association analysis.

This is the first inferential step after Step14a/14b froze the 20-outcome
phenotype family and the analysis registry.

What this script does
---------------------
1. Reuses the CURRENT 4New_cgm cohort/covariate/design machinery.
2. Replaces legacy hPDI with the already-corrected canonical hPDI.
3. Merges the finalized Paper-20 CGM Z/raw/clean columns onto the frozen cohort.
4. Fits all 7 diets x 20 outcomes in primary + strict cohorts, Model2 + Model3.
5. Reports:
      - conventional OLS inference (project-canonical)
      - HC3 robust inference on the exact same design/sample
6. Applies the frozen Step14b BH-FDR plan:
      existing_primary: AHEI, AMED, corrected hPDI, rEDIH = 80 tests
      new_primary:      EAT13, NOVA4                       = 40 tests
      exploratory:      Carbohydrate_pct                   = 20 tests
   separately within each cohort/model.
   Also reports global 140-test BH and within-diet 20-test BH as sensitivity.
7. Runs the prespecified distribution-sensitive checks:
      TAR140  : any-positive logistic + positive log1p HC3
      TAR180  : any-positive logistic + positive log1p HC3
      HBGI    : log1p(clean) HC3
      GRADE   : log1p(clean) HC3

No microbiome model, bridge screen, or mediation model is run here.
No Paper-20 outcome is removed because of correlation or QC flags.

Dependencies: numpy, pandas, scipy only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm, t as t_dist


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
DG = ROOT / "diet_microbiome_glucose_analysis"

BRANCH_CANDIDATES = [
    DG / "4New_cgm",
    ROOT / "4New_cgm",
]

CGM_PACKAGE = ROOT / "cgm_deal" / "cgm论文新增17指标"

CORRECTED_HPDI = (
    DG / "outputs" / "reports" / "new_diet_extension"
    / "hpdi_correction_audit" / "corrected_scores"
    / "hpdi_participant_scores.csv"
)
CORRECTED_HPDI_COL = "hpdi_score_energy_adjusted_z"

ROLE_BY_DIET = {
    "AHEI": "existing_primary",
    "AMED": "existing_primary",
    "hPDI": "existing_primary",
    "rEDIH": "existing_primary",
    "EAT13": "new_primary",
    "NOVA4": "new_primary",
    "Carbohydrate_pct": "exploratory",
}

EXPECTED_ROLE_TESTS = {
    "existing_primary": 80,
    "new_primary": 40,
    "exploratory": 20,
}

SPECIAL_RULES = {
    "cgm_above_140": "two_part",
    "cgm_above_180": "two_part",
    "cgm_hbgi": "log1p_hc3",
    "cgm_grade": "log1p_hc3",
}


def require(path: Path, label: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def norm_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def truthy(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
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
        raise FileNotFoundError("4New_cgm not found")
    return found[0]


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


def latest_registry(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        require(p, "Paper-20 analysis registry")
        return p
    xs = []
    for p in (CGM_PACKAGE / "outputs").glob(
        "paper20_analysis_registry_*/01_paper20_outcome_analysis_registry.csv"
    ):
        if p.is_file():
            xs.append((p.stat().st_mtime, p))
    if not xs:
        raise FileNotFoundError("No frozen Paper-20 analysis registry found")
    xs.sort(reverse=True)
    return xs[0][1]


def finite_numeric(s: pd.Series, name: str) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    bad = x.notna() & ~np.isfinite(x)
    if bad.any():
        raise RuntimeError(f"{name} contains non-finite values")
    return x


def hc3_fit(X: np.ndarray, y: np.ndarray) -> dict:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    xtx = X.T @ X
    try:
        inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(xtx)

    beta = inv @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, inv, X)
    denom = np.clip(1.0 - h, 1e-12, None)
    meat = X.T @ (((resid / denom) ** 2)[:, None] * X)
    cov = inv @ meat @ inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    df = n - k
    stat = beta / se
    p = 2 * t_dist.sf(np.abs(stat), df=df)
    return {"coef": beta, "se": se, "p": p, "df_resid": df}


def loglik_logistic(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    eta = X @ beta
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))


def logistic_irls(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 100,
    step_tol: float = 1e-8,
    score_tol: float = 1e-6,
) -> dict:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    vals = set(np.unique(y).tolist())
    if not vals.issubset({0.0, 1.0}) or len(vals) < 2:
        raise RuntimeError("Logistic outcome must contain both 0 and 1")

    beta = np.zeros(X.shape[1], dtype=float)
    ll = loglik_logistic(X, y, beta)
    converged = False
    message = "max_iter reached"

    for iteration in range(1, max_iter + 1):
        mu = expit(X @ beta)
        w = np.clip(mu * (1 - mu), 1e-12, None)
        score = X.T @ (y - mu)
        fisher = X.T @ (w[:, None] * X)
        try:
            step = np.linalg.solve(fisher, score)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(fisher) @ score

        scale = 1.0
        accepted = False
        for _ in range(30):
            candidate = beta + scale * step
            ll_new = loglik_logistic(X, y, candidate)
            if np.isfinite(ll_new) and ll_new >= ll - 1e-10:
                accepted = True
                break
            scale *= 0.5

        if not accepted:
            message = "Newton step-halving failed"
            break

        beta = candidate
        ll = ll_new
        mu = expit(X @ beta)
        score_new = X.T @ (y - mu)

        if (
            np.max(np.abs(scale * step)) < step_tol
            or np.max(np.abs(score_new)) < score_tol
        ):
            converged = True
            message = f"converged in {iteration} iterations"
            break

        if np.max(np.abs(beta)) > 50:
            message = "possible separation: |beta| > 50"
            break

    mu = expit(X @ beta)
    w = np.clip(mu * (1 - mu), 1e-12, None)
    fisher = X.T @ (w[:, None] * X)
    try:
        cov = np.linalg.inv(fisher)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(fisher)

    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    z = beta / se
    p = 2 * norm.sf(np.abs(z))
    finite = (
        np.isfinite(beta).all()
        and np.isfinite(se).all()
        and np.isfinite(p).all()
        and (se > 0).all()
    )

    return {
        "coef": beta,
        "se": se,
        "p": p,
        "converged": bool(converged and finite),
        "message": message,
        "iterations": iteration,
    }


def z_within(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = float(x.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise RuntimeError("Cannot standardize transformed outcome")
    return (x - x.mean()) / sd


def apply_bh_with_failed_as_one(
    frame: pd.DataFrame,
    p_col: str,
    status_col: str,
    legacy,
    group_cols: list[str],
    out_col: str,
) -> None:
    frame[out_col] = np.nan
    for _, idx in frame.groupby(group_cols, sort=False).groups.items():
        idx = list(idx)
        part = frame.loc[idx]
        valid = part[status_col].eq("computed") & part[p_col].notna()
        p = part[p_col].where(valid, 1.0).to_numpy(float)
        q = legacy.bh_fdr(p)
        frame.loc[idx, out_col] = np.where(valid, q, np.nan)


def add_primary_fdr(results: pd.DataFrame, legacy) -> pd.DataFrame:
    x = results.copy()

    # Frozen role-family FDR.
    x["FDR_role"] = np.nan
    x["role_tests_planned"] = np.nan
    for (aset, model, role), idx in x.groupby(
        ["analysis_set", "model", "role"], sort=False
    ).groups.items():
        idx = list(idx)
        part = x.loc[idx]
        expected = EXPECTED_ROLE_TESTS[role]
        if len(part) != expected:
            raise RuntimeError(
                f"FDR family size mismatch {aset}/M{model}/{role}: "
                f"{len(part)} != {expected}"
            )
        good = part["status"].eq("computed") & part["p_value"].notna()
        q = legacy.bh_fdr(
            part["p_value"].where(good, 1.0).to_numpy(float)
        )
        x.loc[idx, "FDR_role"] = np.where(good, q, np.nan)
        x.loc[idx, "role_tests_planned"] = len(part)

    # Global 140-test sensitivity.
    x["FDR_global140"] = np.nan
    for (aset, model), idx in x.groupby(
        ["analysis_set", "model"], sort=False
    ).groups.items():
        idx = list(idx)
        part = x.loc[idx]
        if len(part) != 140:
            raise RuntimeError(
                f"Global Paper20 family mismatch {aset}/M{model}: "
                f"{len(part)} != 140"
            )
        good = part["status"].eq("computed") & part["p_value"].notna()
        q = legacy.bh_fdr(
            part["p_value"].where(good, 1.0).to_numpy(float)
        )
        x.loc[idx, "FDR_global140"] = np.where(good, q, np.nan)

    # Within-diet 20-outcome descriptive sensitivity.
    x["FDR_within_diet20"] = np.nan
    for _, idx in x.groupby(
        ["analysis_set", "model", "exposure"], sort=False
    ).groups.items():
        idx = list(idx)
        part = x.loc[idx]
        if len(part) != 20:
            raise RuntimeError(
                f"Within-diet Paper20 family has {len(part)} != 20 tests"
            )
        good = part["status"].eq("computed") & part["p_value"].notna()
        q = legacy.bh_fdr(
            part["p_value"].where(good, 1.0).to_numpy(float)
        )
        x.loc[idx, "FDR_within_diet20"] = np.where(good, q, np.nan)

    x["significant_role_05"] = x["FDR_role"].lt(0.05)
    x["significant_global140_05"] = x["FDR_global140"].lt(0.05)
    x["significant_within_diet20_05"] = x["FDR_within_diet20"].lt(0.05)
    return x


def add_hc3_fdr(results: pd.DataFrame, legacy) -> pd.DataFrame:
    x = results.copy()

    # Same exact planned families, using HC3 p-values.
    x["HC3_FDR_role"] = np.nan
    for _, idx in x.groupby(
        ["analysis_set", "model", "role"], sort=False
    ).groups.items():
        idx = list(idx)
        part = x.loc[idx]
        good = part["status"].eq("computed") & part["HC3_p_value"].notna()
        q = legacy.bh_fdr(
            part["HC3_p_value"].where(good, 1.0).to_numpy(float)
        )
        x.loc[idx, "HC3_FDR_role"] = np.where(good, q, np.nan)

    x["HC3_FDR_global140"] = np.nan
    for _, idx in x.groupby(
        ["analysis_set", "model"], sort=False
    ).groups.items():
        idx = list(idx)
        part = x.loc[idx]
        good = part["status"].eq("computed") & part["HC3_p_value"].notna()
        q = legacy.bh_fdr(
            part["HC3_p_value"].where(good, 1.0).to_numpy(float)
        )
        x.loc[idx, "HC3_FDR_global140"] = np.where(good, q, np.nan)

    x["HC3_significant_role_05"] = x["HC3_FDR_role"].lt(0.05)
    x["HC3_role_gate_changed"] = (
        x["significant_role_05"] != x["HC3_significant_role_05"]
    )
    return x


def add_sensitivity_fdr(sens: pd.DataFrame, legacy) -> pd.DataFrame:
    x = sens.copy()
    x["FDR_role"] = np.nan
    x["FDR_global7"] = np.nan

    # Each sensitivity_id is an independent confirmatory diagnostic family:
    # 4 existing-primary + 2 new-primary + 1 exploratory exposure tests.
    for (_, _, sid, role), idx in x.groupby(
        ["analysis_set", "model", "sensitivity_id", "role"], sort=False
    ).groups.items():
        idx = list(idx)
        part = x.loc[idx]
        expected = {"existing_primary": 4, "new_primary": 2, "exploratory": 1}[role]
        if len(part) != expected:
            raise RuntimeError(
                f"Sensitivity role-family mismatch {sid}/{role}: "
                f"{len(part)} != {expected}"
            )
        good = part["status"].eq("computed") & part["p_value"].notna()
        q = legacy.bh_fdr(
            part["p_value"].where(good, 1.0).to_numpy(float)
        )
        x.loc[idx, "FDR_role"] = np.where(good, q, np.nan)

    for _, idx in x.groupby(
        ["analysis_set", "model", "sensitivity_id"], sort=False
    ).groups.items():
        idx = list(idx)
        part = x.loc[idx]
        if len(part) != 7:
            raise RuntimeError(
                f"Sensitivity global family {part['sensitivity_id'].iloc[0]} "
                f"has {len(part)} != 7 tests"
            )
        good = part["status"].eq("computed") & part["p_value"].notna()
        q = legacy.bh_fdr(
            part["p_value"].where(good, 1.0).to_numpy(float)
        )
        x.loc[idx, "FDR_global7"] = np.where(good, q, np.nan)

    x["significant_role_05"] = x["FDR_role"].lt(0.05)
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch-dir", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--cgm-csv", default=None)
    ap.add_argument("--registry", default=None)
    args = ap.parse_args()

    branch = find_branch(args.branch_dir)
    scripts = branch / "scripts"
    sys.path.insert(0, str(scripts))

    import new_cgm_data as data
    import new_cgm_models as models
    import legacy_diet_models as legacy

    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else branch / "config" / "analysis.json"
    )
    require(config_path, "4New_cgm config")
    require(CORRECTED_HPDI, "corrected hPDI")

    final_cgm = latest_final_cgm(args.cgm_csv)
    registry_path = latest_registry(args.registry)

    registry = pd.read_csv(registry_path, low_memory=False)
    paper = registry.loc[
        registry["outcome_family"].eq("paper20_primary")
    ].copy()
    if len(paper) != 20:
        raise RuntimeError(f"Frozen registry Paper-20 count is {len(paper)}, expected 20")
    if paper["field"].duplicated().any():
        raise RuntimeError("Duplicate Paper-20 field in registry")

    outcome_fields = paper["field"].astype(str).tolist()
    outcome_labels = dict(zip(paper["field"], paper["label"]))
    outcome_classes = dict(zip(paper["field"], paper["analysis_class"]))

    # Build the exact existing project cohort/design source.
    config = data.load_config(config_path)
    diet, _, _, _, overlap, fingerprints = data.prepare_tables(config)

    # Merge finalized Paper-20 values.
    cgm = data.read_table(final_cgm)
    if len(cgm) != 7493:
        raise RuntimeError(f"Finalized Paper-20 CGM N={len(cgm)}, expected 7493")
    required_cgm = ["participant_id"]
    for field in outcome_fields:
        required_cgm.append(field + "_z")
        if field in SPECIAL_RULES:
            required_cgm += [field + "_raw", field + "_clean"]
    data.require_columns(cgm, required_cgm, "finalized Paper-20 CGM")

    keep = list(dict.fromkeys(required_cgm))
    cgm_piece = cgm[keep].copy()
    for col in keep:
        if col != "participant_id":
            cgm_piece[col] = data.numeric(cgm_piece[col], col)

    # Some outcomes already exist in the legacy 4New_cgm cohort.
    # In particular TAR180 (cgm_above_180_raw/clean/z) is present in
    # data.prepare_tables().  A plain pandas merge would rename duplicate
    # columns to *_x / *_y, after which cgm_above_180_z no longer exists.
    #
    # Audit any overlaps, then let the newly finalized Paper-20 table be the
    # canonical source for those columns.
    overlap_cols = sorted(
        (set(diet.columns) & set(cgm_piece.columns)) - {"participant_id"}
    )
    print("PAPER20_EXISTING_COLUMN_OVERLAPS=" + ",".join(overlap_cols))

    if overlap_cols:
        audit = diet[["participant_id"] + overlap_cols].merge(
            cgm_piece[["participant_id"] + overlap_cols],
            on="participant_id",
            how="inner",
            validate="one_to_one",
            suffixes=("_legacy", "_paper20"),
        )
        print("--- OVERLAP VALUE AUDIT ---")
        for col in overlap_cols:
            a = pd.to_numeric(audit[col + "_legacy"], errors="coerce")
            b = pd.to_numeric(audit[col + "_paper20"], errors="coerce")
            both = a.notna() & b.notna()
            availability_mismatch = int((a.notna() != b.notna()).sum())
            changed = int(
                ((a - b).abs().gt(1e-12) & both).sum()
            )
            max_abs_diff = (
                float((a.loc[both] - b.loc[both]).abs().max())
                if both.any()
                else np.nan
            )
            print(
                f"{col}: both_N={int(both.sum())} "
                f"availability_mismatch_N={availability_mismatch} "
                f"value_changed_N={changed} "
                f"max_abs_diff={max_abs_diff}"
            )

        # IMPORTANT: the finalized Paper-20 file wins by construction.
        diet = diet.drop(columns=overlap_cols)

    before_n = len(diet)
    diet = diet.merge(
        cgm_piece,
        on="participant_id",
        how="left",
        validate="one_to_one",
    )
    if len(diet) != before_n:
        raise RuntimeError("Paper-20 merge changed diet cohort row count")

    # Hard post-merge schema gate: no pandas duplicate suffixes and all
    # required Paper-20 fields must retain their canonical names.
    bad_suffixes = [
        c for c in diet.columns if c.endswith("_x") or c.endswith("_y")
    ]
    if bad_suffixes:
        raise RuntimeError(
            "Unexpected merge suffix columns remain: "
            + ", ".join(bad_suffixes[:20])
        )
    missing_after_merge = [
        c for c in required_cgm
        if c != "participant_id" and c not in diet.columns
    ]
    if missing_after_merge:
        raise RuntimeError(
            "Paper-20 columns missing after merge: "
            + ", ".join(missing_after_merge)
        )

    # Replace hPDI with the canonical corrected score.
    hpdi = pd.read_csv(CORRECTED_HPDI, low_memory=False)
    if "participant_id" not in hpdi or CORRECTED_HPDI_COL not in hpdi:
        raise RuntimeError("Corrected hPDI file missing required columns")
    hpdi["participant_id"] = norm_id(hpdi["participant_id"])
    if hpdi["participant_id"].duplicated().any():
        raise RuntimeError("Corrected hPDI duplicate participant_id")
    hpdi[CORRECTED_HPDI_COL] = finite_numeric(
        hpdi[CORRECTED_HPDI_COL], CORRECTED_HPDI_COL
    )
    diet = diet.merge(
        hpdi[["participant_id", CORRECTED_HPDI_COL]],
        on="participant_id",
        how="left",
        validate="one_to_one",
    )
    diet["hPDI_z_legacy"] = pd.to_numeric(diet["hPDI_z"], errors="coerce")
    diet["hPDI_z"] = pd.to_numeric(
        diet[CORRECTED_HPDI_COL], errors="coerce"
    )

    both = diet["hPDI_z_legacy"].notna() & diet["hPDI_z"].notna()
    if not both.any():
        raise RuntimeError("No old/corrected hPDI overlap")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = branch / "outputs" / f"paper20_diet_cgm_{stamp}"
    (out / "models").mkdir(parents=True, exist_ok=False)
    (out / "reports").mkdir()
    (out / "data").mkdir()

    print("=== STEP 15 SEVEN-DIET x PAPER-20 DIET->CGM ===")
    print(f"FINAL_CGM={final_cgm}")
    print(f"REGISTRY={registry_path}")
    print(f"DIET_COHORT_ROWS={len(diet)}")
    print("CORRECTED_HPDI=True")
    print("PAPER20_OUTCOMES=20")
    print("DIETS=7")
    print("ANALYSIS_SETS=primary,strict")
    print("MODELS=2,3")
    print("PLANNED_CLASSICAL_FITS=560")
    print("STATSMODELS_USED=False")
    print("MICROBIOME_MODELS_FIT=False")
    print("BRIDGE_SCREEN_RUN=False")

    rows = []
    sensitivity_rows = []

    for analysis_set in ["primary", "strict"]:
        eligible = (
            diet["strict_eligible"]
            if analysis_set == "strict"
            else diet["primary_eligible"]
        )

        for model in [2, 3]:
            for score in data.EXPOSURES:
                score_col, old_family = data.EXPOSURES[score]
                role = ROLE_BY_DIET[score]

                for field in outcome_fields:
                    zcol = field + "_z"

                    required = [score_col, zcol]
                    if model >= 2:
                        required += data.BASE_COVARIATES
                        if model == 3:
                            required += ["bmi"]
                        if score == "hPDI":
                            required += ["alcohol_intake_g_day"]
                        if old_family != "original_four":
                            required += ["mean_daily_energy_kcal"]

                        prefix = "hpdi" if score == "hPDI" else "amed"
                        flag = prefix + f"_cgm_model{model}_covariates_complete"
                        mask = eligible & data.boolean(diet[flag], flag)
                    else:
                        mask = eligible.copy()

                    selected = diet.loc[
                        mask & models.complete_mask(diet, required)
                    ].copy()

                    row = {
                        "analysis_set": analysis_set,
                        "model": model,
                        "exposure": score,
                        "role": role,
                        "outcome_field": field,
                        "outcome_label": outcome_labels[field],
                        "analysis_class": outcome_classes[field],
                        "N": len(selected),
                        "sample_sha256": data.sample_hash(
                            selected["participant_id"]
                        ) if len(selected) else "",
                        "beta": np.nan,
                        "SE": np.nan,
                        "CI95_lower": np.nan,
                        "CI95_upper": np.nan,
                        "p_value": np.nan,
                        "HC3_SE": np.nan,
                        "HC3_p_value": np.nan,
                        "status": "not_fitted",
                        "error": "",
                    }

                    try:
                        if len(selected) < config["minimum_model_n"]:
                            raise RuntimeError(
                                f"Insufficient N={len(selected)}"
                            )

                        y = selected[zcol].to_numpy(float)
                        if not np.isfinite(y).all() or np.ptp(y) == 0:
                            raise RuntimeError("Invalid/constant outcome")

                        X, names, _, condition = models.diet_design(
                            selected, score, model
                        )
                        j = names.index(score_col)

                        fit = legacy.fit_ols(X, y)
                        hc3 = hc3_fit(X, y)

                        row.update(
                            beta=float(fit["coef"][j]),
                            SE=float(fit["se"][j]),
                            CI95_lower=float(fit["ci_lower"][j]),
                            CI95_upper=float(fit["ci_upper"][j]),
                            p_value=float(fit["p"][j]),
                            R2=float(fit["r2"]),
                            df_resid=int(fit["df_resid"]),
                            design_condition=float(condition),
                            HC3_SE=float(hc3["se"][j]),
                            HC3_p_value=float(hc3["p"][j]),
                            status="computed",
                        )
                    except Exception as e:
                        row.update(status="failed", error=repr(e))

                    rows.append(row)

                    # Distribution-sensitive sensitivity models are run on the
                    # exact same complete-case base selected for primary OLS,
                    # with additional nonmissing raw/clean requirements below.
                    if field not in SPECIAL_RULES or row["status"] != "computed":
                        continue

                    X, names, _, _ = models.diet_design(
                        selected, score, model
                    )
                    j = names.index(score_col)

                    if SPECIAL_RULES[field] == "two_part":
                        rawcol = field + "_raw"
                        raw = pd.to_numeric(
                            selected[rawcol], errors="coerce"
                        )
                        valid = raw.notna() & np.isfinite(raw)
                        sel = selected.loc[valid].copy()
                        raw = raw.loc[valid].astype(float)

                        Xs, names_s, _, _ = models.diet_design(
                            sel, score, model
                        )
                        js = names_s.index(score_col)
                        y_any = raw.gt(0).astype(float).to_numpy()

                        srow = {
                            "analysis_set": analysis_set,
                            "model": model,
                            "exposure": score,
                            "role": role,
                            "source_outcome_field": field,
                            "source_outcome_label": outcome_labels[field],
                            "sensitivity_id": field + "_any_positive",
                            "part": "any_positive_logistic",
                            "N": len(sel),
                            "events": int(y_any.sum()),
                            "beta": np.nan,
                            "SE": np.nan,
                            "p_value": np.nan,
                            "status": "not_fitted",
                            "error": "",
                        }
                        try:
                            lg = logistic_irls(Xs, y_any)
                            if not lg["converged"]:
                                raise RuntimeError(lg["message"])
                            srow.update(
                                beta=float(lg["coef"][js]),
                                SE=float(lg["se"][js]),
                                p_value=float(lg["p"][js]),
                                odds_ratio_per_1SD_exposure=float(
                                    np.exp(lg["coef"][js])
                                ),
                                status="computed",
                                convergence_message=lg["message"],
                            )
                        except Exception as e:
                            srow.update(status="failed", error=repr(e))
                        sensitivity_rows.append(srow)

                        pos = sel.loc[raw.gt(0)].copy()
                        srow = {
                            "analysis_set": analysis_set,
                            "model": model,
                            "exposure": score,
                            "role": role,
                            "source_outcome_field": field,
                            "source_outcome_label": outcome_labels[field],
                            "sensitivity_id": field + "_positive_log1p",
                            "part": "positive_log1p_HC3",
                            "N": len(pos),
                            "events": len(pos),
                            "beta": np.nan,
                            "SE": np.nan,
                            "p_value": np.nan,
                            "status": "not_fitted",
                            "error": "",
                        }
                        try:
                            raw_pos = pd.to_numeric(
                                pos[rawcol], errors="raise"
                            ).to_numpy(float)
                            y_pos = z_within(np.log1p(raw_pos))
                            Xp, namesp, _, _ = models.diet_design(
                                pos, score, model
                            )
                            jp = namesp.index(score_col)
                            fp = hc3_fit(Xp, y_pos)
                            srow.update(
                                beta=float(fp["coef"][jp]),
                                SE=float(fp["se"][jp]),
                                p_value=float(fp["p"][jp]),
                                status="computed",
                            )
                        except Exception as e:
                            srow.update(status="failed", error=repr(e))
                        sensitivity_rows.append(srow)

                    elif SPECIAL_RULES[field] == "log1p_hc3":
                        cleancol = field + "_clean"
                        clean = pd.to_numeric(
                            selected[cleancol], errors="coerce"
                        )
                        valid = (
                            clean.notna()
                            & np.isfinite(clean)
                            & clean.ge(0)
                        )
                        sel = selected.loc[valid].copy()

                        srow = {
                            "analysis_set": analysis_set,
                            "model": model,
                            "exposure": score,
                            "role": role,
                            "source_outcome_field": field,
                            "source_outcome_label": outcome_labels[field],
                            "sensitivity_id": field + "_log1p",
                            "part": "log1p_clean_HC3",
                            "N": len(sel),
                            "events": np.nan,
                            "beta": np.nan,
                            "SE": np.nan,
                            "p_value": np.nan,
                            "status": "not_fitted",
                            "error": "",
                        }
                        try:
                            ylog = np.log1p(
                                pd.to_numeric(
                                    sel[cleancol], errors="raise"
                                ).to_numpy(float)
                            )
                            ylog = z_within(ylog)
                            Xs, names_s, _, _ = models.diet_design(
                                sel, score, model
                            )
                            js = names_s.index(score_col)
                            fs = hc3_fit(Xs, ylog)
                            srow.update(
                                beta=float(fs["coef"][js]),
                                SE=float(fs["se"][js]),
                                p_value=float(fs["p"][js]),
                                status="computed",
                            )
                        except Exception as e:
                            srow.update(status="failed", error=repr(e))
                        sensitivity_rows.append(srow)

    results = pd.DataFrame(rows)
    if len(results) != 560:
        raise RuntimeError(f"Expected 560 classical rows, found {len(results)}")

    results = add_primary_fdr(results, legacy)
    results = add_hc3_fdr(results, legacy)

    sensitivity = pd.DataFrame(sensitivity_rows)
    # 4 sensitivity IDs x 7 diets x 2 cohorts x 2 models = 112 rows
    # TAR140/TAR180 each contribute 2 IDs; HBGI/GRADE each 1:
    # total IDs = 6 -> 168 rows.
    if len(sensitivity) != 168:
        raise RuntimeError(
            f"Expected 168 sensitivity rows, found {len(sensitivity)}"
        )
    sensitivity = add_sensitivity_fdr(sensitivity, legacy)

    # Join OLS direction for audit only. This does NOT create a bridge gate.
    ols_key = [
        "analysis_set", "model", "exposure", "role", "outcome_field"
    ]
    ols_small = results[ols_key + [
        "beta", "FDR_role", "significant_role_05"
    ]].rename(columns={
        "outcome_field": "source_outcome_field",
        "beta": "OLS_beta",
        "FDR_role": "OLS_FDR_role",
        "significant_role_05": "OLS_significant_role_05",
    })
    sensitivity = sensitivity.merge(
        ols_small,
        on=[
            "analysis_set", "model", "exposure", "role",
            "source_outcome_field"
        ],
        how="left",
        validate="many_to_one",
    )
    sensitivity["same_direction_as_OLS"] = (
        np.sign(pd.to_numeric(sensitivity["beta"], errors="coerce"))
        == np.sign(pd.to_numeric(sensitivity["OLS_beta"], errors="coerce"))
    )

    # Save full models.
    model_path = out / "models" / "15_paper20_diet_cgm_all_models.csv"
    sens_path = out / "models" / "15_paper20_distribution_sensitivities.csv"
    results.to_csv(model_path, index=False)
    sensitivity.to_csv(sens_path, index=False)

    primary_m3 = results.loc[
        results["analysis_set"].eq("primary")
        & results["model"].eq(3)
    ].copy()
    strict_m3 = results.loc[
        results["analysis_set"].eq("strict")
        & results["model"].eq(3)
    ].copy()

    sig_primary_m3 = primary_m3.loc[
        primary_m3["significant_role_05"]
    ].copy()
    sig_primary_m3.to_csv(
        out / "reports" / "15_primary_model3_roleFDR_significant.csv",
        index=False,
    )

    # Context/outcome summaries.
    outcome_summary = (
        primary_m3.groupby(
            ["outcome_field", "outcome_label", "analysis_class"],
            as_index=False,
        )
        .agg(
            diets_tested=("exposure", "size"),
            roleFDR05=("significant_role_05", "sum"),
            HC3_roleFDR05=("HC3_significant_role_05", "sum"),
            HC3_gate_changes=("HC3_role_gate_changed", "sum"),
            min_p=("p_value", "min"),
            min_FDR_role=("FDR_role", "min"),
        )
        .sort_values(
            ["roleFDR05", "min_FDR_role"],
            ascending=[False, True],
        )
    )
    outcome_summary.to_csv(
        out / "reports" / "15_primary_model3_outcome_summary.csv",
        index=False,
    )

    diet_summary = (
        primary_m3.groupby(
            ["exposure", "role"], as_index=False
        )
        .agg(
            outcomes_tested=("outcome_field", "size"),
            roleFDR05=("significant_role_05", "sum"),
            global140_FDR05=("significant_global140_05", "sum"),
            within_diet20_FDR05=(
                "significant_within_diet20_05", "sum"
            ),
            HC3_roleFDR05=("HC3_significant_role_05", "sum"),
        )
        .sort_values(["role", "exposure"])
    )
    diet_summary.to_csv(
        out / "reports" / "15_primary_model3_diet_summary.csv",
        index=False,
    )

    # Exact primary-vs-strict Model3 comparison, same 140 planned paths.
    comp_cols = [
        "exposure", "role", "outcome_field", "outcome_label",
        "N", "beta", "FDR_role", "significant_role_05",
        "HC3_FDR_role", "HC3_significant_role_05",
    ]
    a = primary_m3[comp_cols].rename(
        columns={
            c: c + "_primary"
            for c in comp_cols
            if c not in ["exposure", "role", "outcome_field", "outcome_label"]
        }
    )
    b = strict_m3[comp_cols].rename(
        columns={
            c: c + "_strict"
            for c in comp_cols
            if c not in ["exposure", "role", "outcome_field", "outcome_label"]
        }
    )
    comp = a.merge(
        b,
        on=["exposure", "role", "outcome_field", "outcome_label"],
        validate="one_to_one",
    )
    comp["same_beta_direction"] = (
        np.sign(comp["beta_primary"]) == np.sign(comp["beta_strict"])
    )
    comp["primary_sig_retained_strict"] = (
        truthy(comp["significant_role_05_primary"])
        & truthy(comp["significant_role_05_strict"])
    )
    comp["primary_sig_lost_strict"] = (
        truthy(comp["significant_role_05_primary"])
        & ~truthy(comp["significant_role_05_strict"])
    )
    comp["strict_sig_gained"] = (
        ~truthy(comp["significant_role_05_primary"])
        & truthy(comp["significant_role_05_strict"])
    )
    comp.to_csv(
        out / "reports" / "15_primary_vs_strict_model3.csv",
        index=False,
    )

    # Sensitivity summary for primary Model3 special outcomes.
    sens_pm3 = sensitivity.loc[
        sensitivity["analysis_set"].eq("primary")
        & sensitivity["model"].eq(3)
    ].copy()
    sens_summary = (
        sens_pm3.groupby(
            ["source_outcome_field", "source_outcome_label", "sensitivity_id"],
            as_index=False,
        )
        .agg(
            fits=("exposure", "size"),
            computed=("status", lambda x: int((x == "computed").sum())),
            sensitivity_roleFDR05=("significant_role_05", "sum"),
            OLS_roleFDR05=("OLS_significant_role_05", "sum"),
            same_direction_as_OLS=(
                "same_direction_as_OLS", "sum"
            ),
        )
    )
    sens_summary.to_csv(
        out / "reports" / "15_special_outcome_sensitivity_summary.csv",
        index=False,
    )

    # Important checks.
    computed = int(results["status"].eq("computed").sum())
    failed = len(results) - computed
    primary_sig = int(primary_m3["significant_role_05"].sum())
    strict_sig = int(strict_m3["significant_role_05"].sum())
    primary_hc3_sig = int(primary_m3["HC3_significant_role_05"].sum())
    hc3_changes = int(primary_m3["HC3_role_gate_changed"].sum())
    retained = int(comp["primary_sig_retained_strict"].sum())
    lost = int(comp["primary_sig_lost_strict"].sum())
    gained = int(comp["strict_sig_gained"].sum())
    sign_flips = int((~comp["same_beta_direction"]).sum())

    summary_lines = [
        "=== STEP 15 PAPER-20 DIET->CGM SUMMARY ===",
        f"FINAL_CGM={final_cgm}",
        f"REGISTRY={registry_path}",
        f"CORRECTED_HPDI={CORRECTED_HPDI}",
        "MICROBIOME_MODELS_FIT=False",
        "BRIDGE_SCREEN_RUN=False",
        "",
        f"CLASSICAL_ROWS={len(results)}",
        f"CLASSICAL_COMPUTED={computed}",
        f"CLASSICAL_FAILED={failed}",
        f"SENSITIVITY_ROWS={len(sensitivity)}",
        "",
        f"PRIMARY_MODEL3_ROLE_FDR05={primary_sig}",
        f"STRICT_MODEL3_ROLE_FDR05={strict_sig}",
        f"PRIMARY_MODEL3_HC3_ROLE_FDR05={primary_hc3_sig}",
        f"PRIMARY_MODEL3_HC3_GATE_CHANGES={hc3_changes}",
        "",
        f"PRIMARY_M3_SIG_RETAINED_STRICT={retained}",
        f"PRIMARY_M3_SIG_LOST_STRICT={lost}",
        f"STRICT_M3_SIG_GAINED={gained}",
        f"MODEL3_BETA_DIRECTION_FLIPS_ALL140={sign_flips}",
        "",
        "--- PRIMARY MODEL3 BY DIET ---",
        diet_summary.to_string(index=False),
        "",
        "--- PRIMARY MODEL3 BY OUTCOME ---",
        outcome_summary.to_string(index=False),
        "",
        "--- SPECIAL OUTCOME SENSITIVITY ---",
        sens_summary.to_string(index=False),
        "",
        "INTERPRETATION RULES:",
        "- Paper-20 remains the primary outcome family; none are removed post hoc.",
        "- Primary BH-FDR is the frozen exposure-role family FDR.",
        "- Global 140-test and within-diet 20-test BH are sensitivity/descriptive only.",
        "- HC3 is robustness inference on the exact same OLS design/sample.",
        "- TAR140/TAR180/HBGI/GRADE sensitivity outputs are diagnostic/robustness analyses.",
        "- TAR180 ordinary OLS alone must NOT be used to open a downstream bridge gate.",
        "- Mean glucose and GMI are highly dependent and are not independent replication.",
        "- Results are cross-sectional associations, not causal effects.",
        "",
        f"MODELS={model_path}",
        f"SENSITIVITIES={sens_path}",
        f"OUTPUT_DIR={out}",
    ]

    summary_path = out / "reports" / "15_paper20_diet_cgm_summary.txt"
    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "status": "completed",
        "final_cgm": str(final_cgm),
        "final_cgm_sha256": sha256(final_cgm),
        "registry": str(registry_path),
        "registry_sha256": sha256(registry_path),
        "corrected_hpdi": str(CORRECTED_HPDI),
        "corrected_hpdi_sha256": sha256(CORRECTED_HPDI),
        "paper20_outcomes": 20,
        "diets": 7,
        "analysis_sets": ["primary", "strict"],
        "models": [2, 3],
        "classical_rows": int(len(results)),
        "classical_computed": computed,
        "classical_failed": failed,
        "sensitivity_rows": int(len(sensitivity)),
        "primary_model3_role_fdr05": primary_sig,
        "strict_model3_role_fdr05": strict_sig,
        "primary_model3_hc3_role_fdr05": primary_hc3_sig,
        "primary_model3_hc3_gate_changes": hc3_changes,
        "primary_m3_sig_retained_strict": retained,
        "primary_m3_sig_lost_strict": lost,
        "strict_m3_sig_gained": gained,
        "model3_beta_direction_flips_all140": sign_flips,
        "microbiome_models_fit": False,
        "bridge_screen_run": False,
        "outputs": {
            "models": str(model_path),
            "sensitivities": str(sens_path),
            "summary": str(summary_path),
        },
    }
    (out / "reports" / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n" + "\n".join(summary_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
