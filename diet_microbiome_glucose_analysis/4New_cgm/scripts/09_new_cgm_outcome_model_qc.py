#!/usr/bin/env python3
"""Step 09c — four-new-CGM outcome-model QC, exact-project-design version.

NO statsmodels dependency.

Key fix vs 09b
--------------
The 4New_cgm project already creates:
    primary_eligible
    strict_eligible
inside new_cgm_data.prepare_tables().

Rather than guessing flag names or rebuilding categorical coding ourselves,
this version directly reuses the project's own:
    new_cgm_models.select_diet()
    new_cgm_models.diet_design()
    legacy_diet_models.fit_ols()

Therefore the classical OLS reconstruction uses the EXACT same sample
selection and design matrix as the existing 4New_cgm pipeline.

Purpose
-------
1) Reconstruct canonical corrected-hPDI primary/strict Model2/Model3 models.
2) Require exact N and near-exact exposure beta match.
3) Add HC3 robust-SE sensitivity using the exact existing design matrix.
4) Add a two-part TAR180 sensitivity because TAR180 is strongly zero-inflated:
   - any TAR180 > 0: logistic regression
   - positive TAR180 only: log1p(TAR180), OLS + HC3
5) Do NOT run microbiome, bridge, or mediation.

Dependencies
------------
numpy, pandas, scipy only.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2, jarque_bera, t as t_dist


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")

CANDIDATE_BRANCH_DIRS = [
    ROOT / "diet_microbiome_glucose_analysis" / "4New_cgm",
    ROOT / "4New_cgm",
]

CORRECTED_HPDI = (
    ROOT
    / "diet_microbiome_glucose_analysis"
    / "outputs"
    / "reports"
    / "new_diet_extension"
    / "hpdi_correction_audit"
    / "corrected_scores"
    / "hpdi_participant_scores.csv"
)
CORRECTED_HPDI_COL = "hpdi_score_energy_adjusted_z"

RAW_TAR180_COL = "cgm_above_180_raw"


def require(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


def norm_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def truthy(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).eq(1)
    return (
        s.astype(str).str.strip().str.lower()
        .isin({"true", "1", "1.0", "yes", "y", "t"})
    )


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


def find_latest_corrected_hpdi_run(branch: Path) -> Path:
    candidates = []
    outroot = branch / "outputs"

    for p in outroot.glob("corrected_hpdi_*"):
        table = p / "models" / "diet_cgm_all_models_corrected_hpdi.csv"
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
                pass

        candidates.append(
            (status == "completed", table.stat().st_mtime, p)
        )

    if not candidates:
        raise FileNotFoundError(
            f"No corrected_hpdi_* run found under {outroot}"
        )

    candidates.sort(reverse=True)
    return candidates[0][2]


def hc3_from_exact_design(X: np.ndarray, y: np.ndarray) -> dict:
    """Classical OLS + HC3 using the already-frozen project design matrix."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n, k = X.shape
    if n <= k:
        raise RuntimeError(f"N={n} <= K={k}")

    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    fitted = X @ beta
    resid = y - fitted

    df_resid = n - k
    sse = float(resid @ resid)
    mse = sse / df_resid

    se_classical = np.sqrt(np.diag(xtx_inv) * mse)
    t_classical = beta / se_classical
    p_classical = 2 * t_dist.sf(np.abs(t_classical), df=df_resid)

    # HC3: (X'X)^-1 X' diag(e_i^2/(1-h_ii)^2) X (X'X)^-1
    h = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    one_minus_h = np.clip(1.0 - h, 1e-12, None)
    adjusted_sq_resid = (resid / one_minus_h) ** 2
    meat = X.T @ (adjusted_sq_resid[:, None] * X)
    cov_hc3 = xtx_inv @ meat @ xtx_inv
    se_hc3 = np.sqrt(np.clip(np.diag(cov_hc3), 0, None))

    # Use residual-df t reference for a conservative finite-sample HC3 test.
    t_hc3 = beta / se_hc3
    p_hc3 = 2 * t_dist.sf(np.abs(t_hc3), df=df_resid)
    tcrit = float(t_dist.ppf(0.975, df=df_resid))

    # Breusch-Pagan LM diagnostic.
    e2 = resid ** 2
    gamma = np.linalg.lstsq(X, e2, rcond=None)[0]
    fitted_e2 = X @ gamma
    ss_tot = float(((e2 - e2.mean()) ** 2).sum())
    ss_res = float(((e2 - fitted_e2) ** 2).sum())

    if ss_tot > 0:
        r2_aux = max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
        lm_stat = n * r2_aux
        bp_df = max(1, k - 1)
        bp_p = float(chi2.sf(lm_stat, bp_df))
    else:
        bp_p = np.nan

    try:
        jb = jarque_bera(resid)
        jb_p = float(jb.pvalue)
    except Exception:
        jb_p = np.nan

    sd = float(resid.std(ddof=0))
    if sd > 0:
        z = (resid - resid.mean()) / sd
        resid_skew = float(np.mean(z ** 3))
        resid_kurtosis = float(np.mean(z ** 4))
    else:
        resid_skew = np.nan
        resid_kurtosis = np.nan

    return {
        "coef": beta,
        "se_classical": se_classical,
        "p_classical": p_classical,
        "ci_lower_classical": beta - tcrit * se_classical,
        "ci_upper_classical": beta + tcrit * se_classical,
        "se_hc3": se_hc3,
        "p_hc3": p_hc3,
        "ci_lower_hc3": beta - tcrit * se_hc3,
        "ci_upper_hc3": beta + tcrit * se_hc3,
        "df_resid": df_resid,
        "bp_p": bp_p,
        "jb_p": jb_p,
        "resid_skew": resid_skew,
        "resid_kurtosis": resid_kurtosis,
    }


def logistic_exact_design(X: np.ndarray, y: np.ndarray) -> dict:
    """Unpenalized logistic regression using scipy.optimize only."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    def nll(beta):
        eta = X @ beta
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    def grad(beta):
        mu = expit(X @ beta)
        return X.T @ (mu - y)

    result = minimize(
        nll,
        np.zeros(X.shape[1], dtype=float),
        jac=grad,
        method="BFGS",
        options={"maxiter": 500, "gtol": 1e-8},
    )

    beta = result.x
    mu = expit(X @ beta)
    w = np.clip(mu * (1.0 - mu), 1e-10, None)
    fisher = X.T @ (w[:, None] * X)

    try:
        cov = np.linalg.inv(fisher)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(fisher)

    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    z = beta / se

    # Normal reference is standard for logistic Wald inference.
    from scipy.stats import norm
    p = 2 * norm.sf(np.abs(z))

    return {
        "coef": beta,
        "se": se,
        "p": p,
        "converged": bool(result.success),
        "message": str(result.message),
    }


def rebuild_fdr(models_module, frame: pd.DataFrame) -> pd.DataFrame:
    """Use the project's own BH-FDR implementation."""
    return models_module.correct_fdr(frame, "diet")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-dir", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--canonical-run-dir", default=None)
    args = parser.parse_args()

    branch = find_branch_dir(args.branch_dir)
    scripts = branch / "scripts"

    require(scripts / "new_cgm_data.py", "new_cgm_data.py")
    require(scripts / "new_cgm_models.py", "new_cgm_models.py")
    require(scripts / "legacy_diet_models.py", "legacy_diet_models.py")
    require(CORRECTED_HPDI, "corrected hPDI")

    sys.path.insert(0, str(scripts))
    import new_cgm_data as data
    import new_cgm_models as models
    import legacy_diet_models as legacy

    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else branch / "config" / "analysis.json"
    )
    require(config_path, "analysis config")

    canonical_run = (
        Path(args.canonical_run_dir).expanduser().resolve()
        if args.canonical_run_dir
        else find_latest_corrected_hpdi_run(branch)
    )
    canonical_table = (
        canonical_run / "models" / "diet_cgm_all_models_corrected_hpdi.csv"
    )
    require(canonical_table, "canonical corrected-hPDI 7x4 model table")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = branch / "outputs" / f"outcome_qc_exact_{stamp}"
    (out / "models").mkdir(parents=True, exist_ok=False)
    (out / "reports").mkdir()

    print("=== STEP 09c FOUR-NEW-CGM OUTCOME QC: EXACT PROJECT DESIGN ===")
    print("STATSMODELS_USED=False")
    print("DEPENDENCIES=numpy,pandas,scipy")
    print(f"BRANCH_DIR={branch}")
    print(f"CANONICAL_CORRECTED_HPDI_RUN={canonical_run}")
    print("SAMPLE_SELECTION=models.select_diet")
    print("DESIGN_MATRIX=models.diet_design")
    print("MICROBIOME_REFIT=False")
    print("BRIDGE_RUN=False")
    print("MEDIATION_RUN=False")

    config = data.load_config(config_path)
    diet, microbiome, species, cgm_audit, overlap, fingerprints = data.prepare_tables(config)

    # Sanity-check the actual project-defined flags rather than guessing names.
    for col in ["primary_eligible", "strict_eligible"]:
        if col not in diet.columns:
            raise RuntimeError(
                f"Expected project-defined eligibility column missing: {col}"
            )

    # Replace only hPDI with the already-frozen corrected canonical score.
    hpdi = pd.read_csv(CORRECTED_HPDI, low_memory=False)
    hpdi["participant_id"] = norm_id(hpdi["participant_id"])
    if hpdi["participant_id"].duplicated().any():
        raise RuntimeError("Corrected hPDI contains duplicate participant_id")

    diet["participant_id"] = norm_id(diet["participant_id"])
    diet = diet.merge(
        hpdi[["participant_id", CORRECTED_HPDI_COL]],
        on="participant_id",
        how="left",
        validate="one_to_one",
    )
    diet["hPDI_z"] = pd.to_numeric(
        diet[CORRECTED_HPDI_COL], errors="coerce"
    )

    if RAW_TAR180_COL not in diet.columns:
        raise RuntimeError(
            f"{RAW_TAR180_COL} is missing from prepared diet table. "
            "The frozen 4New_cgm input contract normally includes it."
        )

    canonical = pd.read_csv(canonical_table, low_memory=False)
    canonical_sub = canonical.loc[
        canonical["analysis_set"].isin(["primary", "strict"])
        & canonical["model"].isin([2, 3])
    ].copy()

    expected = 2 * 2 * 7 * 4
    if len(canonical_sub) != expected:
        raise RuntimeError(
            f"Expected {expected} canonical primary/strict M2/M3 rows, "
            f"found {len(canonical_sub)}"
        )

    # ------------------------------------------------------------
    # A. Exact classical reconstruction + HC3
    # ------------------------------------------------------------
    rows = []

    for analysis_set in ["primary", "strict"]:
        for model in [2, 3]:
            for exposure in data.EXPOSURES:
                for outcome in data.OUTCOMES:
                    # EXACT project sample selection.
                    selected = models.select_diet(
                        diet,
                        exposure,
                        outcome,
                        analysis_set,
                        model,
                    )

                    # EXACT project design matrix.
                    X, names, parameters, condition = models.diet_design(
                        selected,
                        exposure,
                        model,
                    )
                    y = selected[data.OUTCOMES[outcome]].to_numpy(float)

                    # Existing project classical model.
                    legacy_fit = legacy.fit_ols(X, y)

                    # Manual HC3 on the identical X/y.
                    robust = hc3_from_exact_design(X, y)

                    exposure_col = data.EXPOSURES[exposure][0]
                    if names[1] != exposure_col:
                        raise RuntimeError(
                            f"Unexpected exposure term position for {exposure}: "
                            f"{names[:3]}"
                        )

                    rows.append({
                        "analysis_set": analysis_set,
                        "model": model,
                        "exposure": exposure,
                        "exposure_column": exposure_col,
                        "family": data.EXPOSURES[exposure][1],
                        "outcome": outcome,
                        "outcome_column": data.OUTCOMES[outcome],
                        "analysis_role": (
                            "supportive"
                            if outcome == "TIR70_180"
                            else "inferential"
                        ),
                        "N": len(selected),
                        "design_columns": X.shape[1],
                        "design_condition": condition,
                        "beta": float(legacy_fit["coef"][1]),
                        "SE_classical": float(legacy_fit["se"][1]),
                        "p_classical": float(legacy_fit["p"][1]),
                        "CI95_lower_classical": float(legacy_fit["ci_lower"][1]),
                        "CI95_upper_classical": float(legacy_fit["ci_upper"][1]),
                        "SE_HC3": float(robust["se_hc3"][1]),
                        "p_HC3": float(robust["p_hc3"][1]),
                        "CI95_lower_HC3": float(robust["ci_lower_hc3"][1]),
                        "CI95_upper_HC3": float(robust["ci_upper_hc3"][1]),
                        "bp_lm_p": robust["bp_p"],
                        "jb_p": robust["jb_p"],
                        "resid_skew": robust["resid_skew"],
                        "resid_kurtosis": robust["resid_kurtosis"],
                        "status": "computed",
                        "error": "",
                    })

    base = pd.DataFrame(rows)

    # Rebuild classical and HC3 FDR using the project's exact FDR function.
    classical_input = base[
        [
            "analysis_set", "model", "exposure", "family",
            "outcome", "N", "beta", "status", "error",
        ]
    ].copy()
    classical_input["p_value"] = base["p_classical"].to_numpy()
    classical_fdr = rebuild_fdr(models, classical_input)

    hc3_input = base[
        [
            "analysis_set", "model", "exposure", "family",
            "outcome", "N", "beta", "status", "error",
        ]
    ].copy()
    hc3_input["p_value"] = base["p_HC3"].to_numpy()
    hc3_fdr = rebuild_fdr(models, hc3_input)

    key = ["analysis_set", "model", "exposure", "outcome"]

    classical_fdr = classical_fdr[key + [
        "FDR_family", "FDR_global",
        "significant_family_05", "significant_global_05",
    ]].rename(columns={
        "FDR_family": "classical_FDR_family",
        "FDR_global": "classical_FDR_global",
        "significant_family_05": "classical_family_sig05",
        "significant_global_05": "classical_global_sig05",
    })

    hc3_fdr = hc3_fdr[key + [
        "FDR_family", "FDR_global",
        "significant_family_05", "significant_global_05",
    ]].rename(columns={
        "FDR_family": "HC3_FDR_family",
        "FDR_global": "HC3_FDR_global",
        "significant_family_05": "HC3_family_sig05",
        "significant_global_05": "HC3_global_sig05",
    })

    base = (
        base
        .merge(classical_fdr, on=key, validate="one_to_one")
        .merge(hc3_fdr, on=key, validate="one_to_one")
    )

    # ------------------------------------------------------------
    # B. Hard reconstruction check against corrected canonical table
    # ------------------------------------------------------------
    can_cols = key + [
        "N", "beta", "p_value", "FDR_family", "FDR_global",
        "significant_family_05", "significant_global_05",
    ]
    missing = [c for c in can_cols if c not in canonical_sub.columns]
    if missing:
        raise RuntimeError(
            f"Canonical corrected-hPDI table missing columns: {missing}"
        )

    recon = base.merge(
        canonical_sub[can_cols],
        on=key,
        how="left",
        validate="one_to_one",
        suffixes=("_reconstructed", "_canonical"),
    )

    recon["N_match"] = recon["N_reconstructed"].eq(recon["N_canonical"])
    recon["beta_abs_diff"] = (
        pd.to_numeric(recon["beta_reconstructed"], errors="coerce")
        - pd.to_numeric(recon["beta_canonical"], errors="coerce")
    ).abs()
    recon["beta_match"] = recon["beta_abs_diff"].lt(1e-10)

    if not recon["N_match"].all() or not recon["beta_match"].all():
        bad = recon.loc[
            ~(recon["N_match"] & recon["beta_match"]),
            key + [
                "N_reconstructed", "N_canonical",
                "beta_reconstructed", "beta_canonical",
                "beta_abs_diff",
            ],
        ]
        failed_path = out / "reports" / "classical_reconstruction_FAILED.csv"
        bad.to_csv(failed_path, index=False)

        raise RuntimeError(
            "Exact-project reconstruction unexpectedly differs from canonical. "
            f"See {failed_path}. Stop before interpreting HC3."
        )

    recon["HC3_family_gate_changed"] = (
        truthy(recon["significant_family_05"])
        != truthy(recon["HC3_family_sig05"])
    )
    recon["HC3_global_gate_changed"] = (
        truthy(recon["significant_global_05"])
        != truthy(recon["HC3_global_sig05"])
    )

    # ------------------------------------------------------------
    # C. TAR180 two-part sensitivity
    # ------------------------------------------------------------
    hurdle_rows = []

    for analysis_set in ["primary", "strict"]:
        for model in [2, 3]:
            for exposure in data.EXPOSURES:
                # Start from the EXACT canonical TAR180 complete-case sample.
                selected = models.select_diet(
                    diet,
                    exposure,
                    "TAR180",
                    analysis_set,
                    model,
                )

                raw = pd.to_numeric(
                    selected[RAW_TAR180_COL], errors="coerce"
                )
                if raw.isna().any():
                    raise RuntimeError(
                        f"Raw TAR180 missing inside selected TAR180 cohort: "
                        f"{analysis_set}/M{model}/{exposure}"
                    )

                X, names, _, condition = models.diet_design(
                    selected,
                    exposure,
                    model,
                )
                exposure_col = data.EXPOSURES[exposure][0]
                j = names.index(exposure_col)

                # Part 1: occurrence of any TAR180.
                y_any = raw.gt(0).astype(float).to_numpy()
                try:
                    logit = logistic_exact_design(X, y_any)
                    b = float(logit["coef"][j])
                    se = float(logit["se"][j])
                    p = float(logit["p"][j])
                    status = "computed" if logit["converged"] else "failed"
                    error = "" if logit["converged"] else logit["message"]
                except Exception as e:
                    b = se = p = np.nan
                    status = "failed"
                    error = repr(e)

                hurdle_rows.append({
                    "analysis_set": analysis_set,
                    "model": model,
                    "exposure": exposure,
                    "family": data.EXPOSURES[exposure][1],
                    "outcome": "TAR180_any",
                    "part": "any_TAR180_logistic",
                    "N": len(selected),
                    "events": int(y_any.sum()),
                    "beta": b,
                    "SE": se,
                    "p_value": p,
                    "odds_ratio_per_1SD_exposure": (
                        float(np.exp(b)) if np.isfinite(b) else np.nan
                    ),
                    "status": status,
                    "error": error,
                })

                # Part 2: severity among positive TAR180 participants.
                pos = selected.loc[raw.gt(0)].copy()
                raw_pos = pd.to_numeric(
                    pos[RAW_TAR180_COL], errors="coerce"
                )
                if len(pos) < 100:
                    hurdle_rows.append({
                        "analysis_set": analysis_set,
                        "model": model,
                        "exposure": exposure,
                        "family": data.EXPOSURES[exposure][1],
                        "outcome": "TAR180_positive",
                        "part": "positive_TAR180_log1p_HC3",
                        "N": len(pos),
                        "events": len(pos),
                        "beta": np.nan,
                        "SE": np.nan,
                        "p_value": np.nan,
                        "odds_ratio_per_1SD_exposure": np.nan,
                        "status": "failed",
                        "error": "positive-part N < 100",
                    })
                    continue

                logy = np.log1p(raw_pos.to_numpy(float))
                sd = float(logy.std(ddof=1))
                if not np.isfinite(sd) or sd <= 0:
                    raise RuntimeError(
                        f"Invalid positive TAR180 log1p SD: "
                        f"{analysis_set}/M{model}/{exposure}"
                    )
                y_pos = (logy - logy.mean()) / sd

                # Rebuild design in the positive-only sample so continuous
                # covariates are standardized exactly as this new model requires.
                Xp, namesp, _, conditionp = models.diet_design(
                    pos,
                    exposure,
                    model,
                )
                jp = namesp.index(exposure_col)

                try:
                    fitp = hc3_from_exact_design(Xp, y_pos)
                    b = float(fitp["coef"][jp])
                    se = float(fitp["se_hc3"][jp])
                    p = float(fitp["p_hc3"][jp])
                    status = "computed"
                    error = ""
                except Exception as e:
                    b = se = p = np.nan
                    status = "failed"
                    error = repr(e)

                hurdle_rows.append({
                    "analysis_set": analysis_set,
                    "model": model,
                    "exposure": exposure,
                    "family": data.EXPOSURES[exposure][1],
                    "outcome": "TAR180_positive",
                    "part": "positive_TAR180_log1p_HC3",
                    "N": len(pos),
                    "events": len(pos),
                    "beta": b,
                    "SE": se,
                    "p_value": p,
                    "odds_ratio_per_1SD_exposure": np.nan,
                    "status": status,
                    "error": error,
                })

    hurdle = pd.DataFrame(hurdle_rows)
    hurdle = rebuild_fdr(models, hurdle)

    # Compare each hurdle component to the continuity OLS TAR180 direction.
    can_tar = canonical_sub.loc[
        canonical_sub["outcome"].eq("TAR180"),
        key + ["beta", "FDR_family", "significant_family_05"],
    ].rename(columns={
        "beta": "canonical_TAR180_OLS_beta",
        "FDR_family": "canonical_TAR180_OLS_FDR_family",
        "significant_family_05": "canonical_TAR180_OLS_family_sig05",
    })

    hurdle = hurdle.merge(
        can_tar,
        on=["analysis_set", "model", "exposure"],
        how="left",
        validate="many_to_one",
    )

    hurdle["same_direction_as_OLS"] = (
        np.sign(pd.to_numeric(hurdle["beta"], errors="coerce"))
        == np.sign(
            pd.to_numeric(
                hurdle["canonical_TAR180_OLS_beta"],
                errors="coerce",
            )
        )
    )

    hurdle["hurdle_only_family_signal"] = (
        truthy(hurdle["significant_family_05"])
        & ~truthy(hurdle["canonical_TAR180_OLS_family_sig05"])
    )

    # ------------------------------------------------------------
    # D. Outputs
    # ------------------------------------------------------------
    hc3_path = out / "models" / "hc3_full_7x4_models.csv"
    hurdle_path = out / "models" / "tar180_hurdle_models.csv"
    recon_path = out / "reports" / "classical_reconstruction_qc.csv"
    summary_path = out / "reports" / "outcome_model_qc_summary.txt"
    manifest_path = out / "reports" / "manifest.json"

    base.to_csv(hc3_path, index=False)
    hurdle.to_csv(hurdle_path, index=False)
    recon.to_csv(recon_path, index=False)

    raw_all = pd.to_numeric(diet[RAW_TAR180_COL], errors="coerce")

    primary_m2 = recon.loc[
        recon["analysis_set"].eq("primary")
        & recon["model"].eq(2),
        [
            "exposure",
            "outcome",
            "N_canonical",
            "beta_canonical",
            "FDR_family",
            "HC3_FDR_family",
            "significant_family_05",
            "HC3_family_sig05",
            "HC3_family_gate_changed",
            "bp_lm_p",
        ],
    ].sort_values(["outcome", "exposure"])

    hurdle_primary_m2 = hurdle.loc[
        hurdle["analysis_set"].eq("primary")
        & hurdle["model"].eq(2),
        [
            "exposure",
            "part",
            "N",
            "events",
            "beta",
            "p_value",
            "FDR_family",
            "significant_family_05",
            "canonical_TAR180_OLS_beta",
            "canonical_TAR180_OLS_FDR_family",
            "same_direction_as_OLS",
            "hurdle_only_family_signal",
            "status",
        ],
    ].sort_values(["part", "exposure"])

    summary_lines = [
        "=== STEP 09c FOUR-NEW-CGM OUTCOME MODEL QC SUMMARY ===",
        "STATSMODELS_USED=False",
        "DEPENDENCIES=numpy,pandas,scipy",
        "EXACT_PROJECT_SAMPLE_SELECTION=True",
        "EXACT_PROJECT_DESIGN_MATRIX=True",
        "CLASSICAL_RECONSTRUCTION_EXACT=True",
        f"RECON_ROWS={len(recon)}",
        f"N_MATCH={int(recon['N_match'].sum())}/{len(recon)}",
        f"BETA_MATCH={int(recon['beta_match'].sum())}/{len(recon)}",
        "CORRECTED_HPDI_CANONICAL=True",
        "TIR70_180_ROLE=supportive",
        "",
        "--- HC3 OVERALL ---",
        f"HC3_FAMILY_GATE_CHANGED_N="
        f"{int(recon['HC3_family_gate_changed'].sum())}",
        f"HC3_GLOBAL_GATE_CHANGED_N="
        f"{int(recon['HC3_global_gate_changed'].sum())}",
        "",
        "--- PRIMARY MODEL2 HC3 DETAIL ---",
        primary_m2.to_string(index=False),
        "",
        "--- TAR180 RAW DISTRIBUTION ---",
        f"RAW_TAR180_COLUMN={RAW_TAR180_COL}",
        f"VALID_N={int(raw_all.notna().sum())}",
        f"ZERO_N={int(raw_all.eq(0).sum())}",
        f"ZERO_FRACTION={float(raw_all.eq(0).mean()):.9f}",
        f"POSITIVE_N={int(raw_all.gt(0).sum())}",
        "",
        "--- TAR180 HURDLE PRIMARY MODEL2 DETAIL ---",
        hurdle_primary_m2.to_string(index=False),
        "",
        "DECISION RULES:",
        "- HC3 uses exactly the existing project's selected cohort and design matrix.",
        "- TIR70_180 remains supportive and is not independent replication of TBR/TAR.",
        "- TAR180 OLS remains the continuity model.",
        "- Hurdle-only significance is distribution-sensitive/exploratory;",
        "  it does not automatically create a primary bridge gate.",
        "- Bridge remains blocked until exact-species microbiome robustness is complete.",
        "",
        f"HC3_MODELS={hc3_path}",
        f"HURDLE_MODELS={hurdle_path}",
        f"RECON_QC={recon_path}",
    ]

    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )
    print("\n" + "\n".join(summary_lines))

    manifest = {
        "status": "completed",
        "statsmodels_used": False,
        "dependencies": ["numpy", "pandas", "scipy"],
        "sample_selection": "new_cgm_models.select_diet",
        "design_matrix": "new_cgm_models.diet_design",
        "canonical_corrected_hpdi_run": str(canonical_run),
        "canonical_table": str(canonical_table),
        "canonical_table_sha256": sha256(canonical_table),
        "classical_reconstruction_exact": True,
        "rows_reconstructed": len(recon),
        "n_match": int(recon["N_match"].sum()),
        "beta_match": int(recon["beta_match"].sum()),
        "hc3_family_gate_changed_n": int(
            recon["HC3_family_gate_changed"].sum()
        ),
        "hc3_global_gate_changed_n": int(
            recon["HC3_global_gate_changed"].sum()
        ),
        "raw_tar180_column": RAW_TAR180_COL,
        "tar180_zero_fraction": float(raw_all.eq(0).mean()),
        "tir70_180_role": "supportive",
        "microbiome_refit": False,
        "bridge_run": False,
        "mediation_run": False,
        "outputs": {
            "hc3_models": str(hc3_path),
            "tar180_hurdle_models": str(hurdle_path),
            "reconstruction_qc": str(recon_path),
            "summary": str(summary_path),
        },
    }

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
