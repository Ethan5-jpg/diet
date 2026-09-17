#!/usr/bin/env python3
"""Step 09d — finalize Step09 TAR180 hurdle QC without statsmodels.

Why this patch exists
---------------------
Step09c successfully completed the exact-project OLS reconstruction and HC3
analysis, but scipy BFGS returned non-success convergence statuses for all
TAR180 logistic occurrence models. Because the project FDR function treats
status != "computed" as failed, those otherwise finite logistic estimates had
NaN FDR values.

This patch:
1) DOES NOT refit the already-valid HC3 analysis.
2) Reads the latest completed outcome_qc_exact_* result.
3) Prints every HC3 family-gate change for audit.
4) Rebuilds only the TAR180 two-part sensitivity using:
   - damped Newton/IRLS logistic regression for any TAR180 > 0;
   - exact-project design + manual HC3 for positive TAR180 severity.
5) Recomputes the pre-specified hurdle FDR families with the project's own
   correct_fdr() implementation.

Dependencies: numpy, pandas, scipy only.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm, t as t_dist


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
        raise FileNotFoundError("Could not find 4New_cgm")
    return found[0]


def find_latest_exact_run(branch: Path) -> Path:
    candidates = []
    for p in (branch / "outputs").glob("outcome_qc_exact_*"):
        recon = p / "reports" / "classical_reconstruction_qc.csv"
        hc3 = p / "models" / "hc3_full_7x4_models.csv"
        if recon.is_file() and hc3.is_file():
            candidates.append((recon.stat().st_mtime, p))
    if not candidates:
        raise FileNotFoundError("No outcome_qc_exact_* run found")
    candidates.sort(reverse=True)
    return candidates[0][1]


def find_latest_corrected_hpdi_run(branch: Path) -> Path:
    candidates = []
    for p in (branch / "outputs").glob("corrected_hpdi_*"):
        table = p / "models" / "diet_cgm_all_models_corrected_hpdi.csv"
        if table.is_file():
            candidates.append((table.stat().st_mtime, p))
    if not candidates:
        raise FileNotFoundError("No corrected_hpdi_* run found")
    candidates.sort(reverse=True)
    return candidates[0][1]


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
    """Unpenalized logistic regression via damped Newton/IRLS.

    No ridge penalty is used, so this estimates the same ordinary logistic
    model intended in Step09c.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if not set(np.unique(y)).issubset({0.0, 1.0}) or len(np.unique(y)) < 2:
        raise RuntimeError("Logistic outcome must contain both 0 and 1")

    beta = np.zeros(X.shape[1], dtype=float)
    ll = loglik_logistic(X, y, beta)
    converged = False
    message = "max_iter reached"

    for iteration in range(1, max_iter + 1):
        eta = X @ beta
        mu = expit(eta)
        w = np.clip(mu * (1.0 - mu), 1e-12, None)

        score = X.T @ (y - mu)
        fisher = X.T @ (w[:, None] * X)

        try:
            step = np.linalg.solve(fisher, score)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(fisher) @ score

        # Damped Newton: only accept a likelihood-improving move.
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

        eta = X @ beta
        mu = expit(eta)
        score_new = X.T @ (y - mu)

        if (
            np.max(np.abs(scale * step)) < step_tol
            or np.max(np.abs(score_new)) < score_tol
        ):
            converged = True
            message = f"converged in {iteration} iterations"
            break

        # Guard against practical separation / explosive coefficients.
        if np.max(np.abs(beta)) > 50:
            message = "possible separation: |beta| > 50"
            break

    mu = expit(X @ beta)
    w = np.clip(mu * (1.0 - mu), 1e-12, None)
    fisher = X.T @ (w[:, None] * X)

    try:
        cov = np.linalg.inv(fisher)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(fisher)

    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    z = beta / se
    p = 2 * norm.sf(np.abs(z))

    condition = float(np.linalg.cond(fisher))
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
        "loglik": ll,
        "fisher_condition": condition,
        "max_abs_score": float(np.max(np.abs(X.T @ (y - mu)))),
        "max_abs_beta": float(np.max(np.abs(beta))),
    }


def hc3_exact(X: np.ndarray, y: np.ndarray) -> dict:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape

    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    df = n - k

    h = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    denom = np.clip(1.0 - h, 1e-12, None)
    meat = X.T @ (((resid / denom) ** 2)[:, None] * X)
    cov = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))

    t = beta / se
    p = 2 * t_dist.sf(np.abs(t), df=df)

    return {"coef": beta, "se": se, "p": p}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-dir", default=None)
    parser.add_argument("--exact-run-dir", default=None)
    parser.add_argument("--canonical-run-dir", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    branch = find_branch_dir(args.branch_dir)
    scripts = branch / "scripts"
    sys.path.insert(0, str(scripts))

    require(scripts / "new_cgm_data.py", "new_cgm_data.py")
    require(scripts / "new_cgm_models.py", "new_cgm_models.py")
    require(CORRECTED_HPDI, "corrected hPDI")

    import new_cgm_data as data
    import new_cgm_models as models

    exact_run = (
        Path(args.exact_run_dir).expanduser().resolve()
        if args.exact_run_dir else find_latest_exact_run(branch)
    )
    canonical_run = (
        Path(args.canonical_run_dir).expanduser().resolve()
        if args.canonical_run_dir else find_latest_corrected_hpdi_run(branch)
    )

    recon_path = exact_run / "reports" / "classical_reconstruction_qc.csv"
    hc3_path = exact_run / "models" / "hc3_full_7x4_models.csv"
    canonical_path = canonical_run / "models" / "diet_cgm_all_models_corrected_hpdi.csv"

    require(recon_path, "Step09c reconstruction QC")
    require(hc3_path, "Step09c HC3 models")
    require(canonical_path, "canonical corrected-hPDI models")

    recon = pd.read_csv(recon_path, low_memory=False)

    if not truthy(recon["N_match"]).all() or not truthy(recon["beta_match"]).all():
        raise RuntimeError("Step09c reconstruction was not exact; stop")

    print("=== STEP 09d TAR180 HURDLE FINALIZATION ===")
    print("STATSMODELS_USED=False")
    print(f"SOURCE_EXACT_RUN={exact_run}")
    print("STEP09C_CLASSICAL_RECONSTRUCTION_EXACT=True")
    print(f"N_MATCH={int(truthy(recon['N_match']).sum())}/{len(recon)}")
    print(f"BETA_MATCH={int(truthy(recon['beta_match']).sum())}/{len(recon)}")

    # Audit all HC3 family gate changes before moving on.
    changes = recon.loc[
        truthy(recon["HC3_family_gate_changed"])
    ].copy()

    print("\n--- HC3 FAMILY GATE CHANGES ---")
    if changes.empty:
        print("NONE")
    else:
        cols = [
            "analysis_set", "model", "exposure", "outcome",
            "beta_canonical", "FDR_family", "HC3_FDR_family",
            "significant_family_05", "HC3_family_sig05",
        ]
        print(changes[cols].to_string(index=False))

    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config else branch / "config" / "analysis.json"
    )
    config = data.load_config(config_path)
    diet, microbiome, species, cgm_audit, overlap, fingerprints = data.prepare_tables(config)

    # canonical corrected hPDI
    hpdi = pd.read_csv(CORRECTED_HPDI, low_memory=False)
    hpdi["participant_id"] = norm_id(hpdi["participant_id"])
    diet["participant_id"] = norm_id(diet["participant_id"])
    diet = diet.merge(
        hpdi[["participant_id", CORRECTED_HPDI_COL]],
        on="participant_id",
        how="left",
        validate="one_to_one",
    )
    diet["hPDI_z"] = pd.to_numeric(diet[CORRECTED_HPDI_COL], errors="coerce")

    if RAW_TAR180_COL not in diet.columns:
        raise RuntimeError(f"Missing {RAW_TAR180_COL}")

    canonical = pd.read_csv(canonical_path, low_memory=False)
    canonical_tar = canonical.loc[
        canonical["analysis_set"].isin(["primary", "strict"])
        & canonical["model"].isin([2, 3])
        & canonical["outcome"].eq("TAR180")
    ].copy()

    rows = []

    for analysis_set in ["primary", "strict"]:
        for model in [2, 3]:
            for exposure in data.EXPOSURES:
                selected = models.select_diet(
                    diet, exposure, "TAR180", analysis_set, model
                )
                X, names, _, _ = models.diet_design(
                    selected, exposure, model
                )
                exposure_col = data.EXPOSURES[exposure][0]
                j = names.index(exposure_col)

                raw = pd.to_numeric(
                    selected[RAW_TAR180_COL], errors="raise"
                )
                y_any = raw.gt(0).astype(float).to_numpy()

                # Part 1: occurrence.
                try:
                    logit = logistic_irls(X, y_any)
                    b = float(logit["coef"][j])
                    se = float(logit["se"][j])
                    p = float(logit["p"][j])
                    status = "computed" if logit["converged"] else "failed"
                    error = "" if logit["converged"] else logit["message"]
                    conv_message = logit["message"]
                    max_score = logit["max_abs_score"]
                    fisher_condition = logit["fisher_condition"]
                except Exception as e:
                    b = se = p = np.nan
                    status = "failed"
                    error = repr(e)
                    conv_message = repr(e)
                    max_score = np.nan
                    fisher_condition = np.nan

                rows.append({
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
                    "convergence_message": conv_message,
                    "max_abs_score": max_score,
                    "fisher_condition": fisher_condition,
                })

                # Part 2: severity among positives.
                pos = selected.loc[raw.gt(0)].copy()
                raw_pos = pd.to_numeric(
                    pos[RAW_TAR180_COL], errors="raise"
                ).to_numpy(float)

                logy = np.log1p(raw_pos)
                sd = float(logy.std(ddof=1))
                if not np.isfinite(sd) or sd <= 0:
                    raise RuntimeError(
                        f"Invalid positive TAR180 log1p SD: "
                        f"{analysis_set}/M{model}/{exposure}"
                    )
                y_pos = (logy - logy.mean()) / sd

                Xp, namesp, _, _ = models.diet_design(
                    pos, exposure, model
                )
                jp = namesp.index(exposure_col)

                try:
                    fitp = hc3_exact(Xp, y_pos)
                    b = float(fitp["coef"][jp])
                    se = float(fitp["se"][jp])
                    p = float(fitp["p"][jp])
                    status = "computed"
                    error = ""
                except Exception as e:
                    b = se = p = np.nan
                    status = "failed"
                    error = repr(e)

                rows.append({
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
                    "convergence_message": "",
                    "max_abs_score": np.nan,
                    "fisher_condition": np.nan,
                })

    hurdle = pd.DataFrame(rows)

    # Use project-defined planned-test FDR handling.
    hurdle = models.correct_fdr(hurdle, "diet")

    # Compare against continuity OLS.
    key = ["analysis_set", "model", "exposure"]
    can = canonical_tar[key + [
        "beta", "FDR_family", "significant_family_05"
    ]].rename(columns={
        "beta": "canonical_TAR180_OLS_beta",
        "FDR_family": "canonical_TAR180_OLS_FDR_family",
        "significant_family_05": "canonical_TAR180_OLS_family_sig05",
    })

    hurdle = hurdle.merge(
        can, on=key, how="left", validate="many_to_one"
    )

    hurdle["same_direction_as_OLS"] = (
        np.sign(pd.to_numeric(hurdle["beta"], errors="coerce"))
        == np.sign(pd.to_numeric(
            hurdle["canonical_TAR180_OLS_beta"], errors="coerce"
        ))
    )
    hurdle["hurdle_only_family_signal"] = (
        truthy(hurdle["significant_family_05"])
        & ~truthy(hurdle["canonical_TAR180_OLS_family_sig05"])
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = branch / "outputs" / f"outcome_qc_hurdle_fixed_{stamp}"
    (out / "models").mkdir(parents=True, exist_ok=False)
    (out / "reports").mkdir()

    hurdle_path = out / "models" / "tar180_hurdle_models_fixed.csv"
    changes_path = out / "reports" / "hc3_family_gate_changes.csv"
    summary_path = out / "reports" / "step09_final_summary.txt"

    hurdle.to_csv(hurdle_path, index=False)
    changes.to_csv(changes_path, index=False)

    primary_m2 = hurdle.loc[
        hurdle["analysis_set"].eq("primary")
        & hurdle["model"].eq(2),
        [
            "exposure", "part", "N", "events",
            "beta", "p_value", "FDR_family",
            "significant_family_05",
            "canonical_TAR180_OLS_beta",
            "canonical_TAR180_OLS_FDR_family",
            "same_direction_as_OLS",
            "hurdle_only_family_signal",
            "status",
            "convergence_message",
        ],
    ].sort_values(["part", "exposure"])

    failed_logit = hurdle.loc[
        hurdle["part"].eq("any_TAR180_logistic")
        & ~hurdle["status"].eq("computed")
    ]

    summary_lines = [
        "=== STEP 09 FINAL OUTCOME-QC SUMMARY ===",
        "CLASSICAL_RECONSTRUCTION_EXACT=True",
        f"HC3_FAMILY_GATE_CHANGED_N={len(changes)}",
        f"TAR180_LOGISTIC_FAILED_N={len(failed_logit)}",
        f"TAR180_HURDLE_ONLY_FAMILY_SIGNAL_N="
        f"{int(truthy(hurdle['hurdle_only_family_signal']).sum())}",
        "",
        "--- HC3 FAMILY GATE CHANGES ---",
        (
            "NONE"
            if changes.empty
            else changes[
                [
                    "analysis_set", "model", "exposure", "outcome",
                    "beta_canonical", "FDR_family", "HC3_FDR_family",
                    "significant_family_05", "HC3_family_sig05",
                ]
            ].to_string(index=False)
        ),
        "",
        "--- TAR180 HURDLE PRIMARY MODEL2 ---",
        primary_m2.to_string(index=False),
        "",
        "INTERPRETATION RULES:",
        "- TIR70_180 stays supportive.",
        "- TAR180 hurdle-only signals remain distribution-sensitive/exploratory.",
        "- No hurdle-only signal is automatically promoted into a bridge gate.",
        "- Exact-species microbiome robustness remains the next required gate.",
        "",
        f"HURDLE_FIXED={hurdle_path}",
        f"HC3_GATE_CHANGES={changes_path}",
    ]

    summary_path.write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    print("\n" + "\n".join(summary_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
