#!/usr/bin/env python3
"""
Step 16b — distribution-sensitivity audit for Paper-20 microbiome->CGM MWAS.

This is a confirmatory robustness audit of ALREADY DISCOVERED canonical
Primary-Model2 MWAS species for the distribution-sensitive outcomes.

It does NOT create new MWAS discoveries.

Outcomes
--------
TAR140 (cgm_above_140):
    1) any-positive logistic sensitivity
    2) positive-only log1p outcome + HC3 OLS

HBGI (cgm_hbgi):
    log1p(clean) outcome + HC3 OLS

GRADE (cgm_grade):
    log1p(clean) outcome + HC3 OLS

TAR180 (cgm_above_180):
    same two-part audit, but SUPPLEMENTARY ONLY because Step15b provides
    no primary Diet->TAR180 bridge gate.

Candidate species
-----------------
Only species significant in Step16a canonical Primary Model2
(within-outcome BH-FDR < 0.05) are audited.

Model sets
----------
primary_m2, primary_m3, strict_m2, strict_m3

Sensitivity p-values are BH-adjusted within each:
    outcome x sensitivity component x model set
over the preselected canonical candidate species only.

Interpretation
--------------
Primary MWAS discoveries remain the Step16a canonical results.

special_rule_ok:
    no FDR-significant sensitivity component points opposite to the
    corresponding canonical MWAS beta, and the required sensitivity fits
    are computable.

sensitivity_high_support:
    in primary_m2 and strict_m2, at least one sensitivity component remains
    candidate-FDR significant in the same direction, with no significant
    opposite component.

For HBGI/GRADE there is one sensitivity component.
For TAR140/TAR180 there are two.

No microbiome bridge or mediation model is run here.
Dependencies: numpy, pandas, scipy only.
"""

from __future__ import annotations

import argparse
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
CGM_PACKAGE = ROOT / "cgm_deal" / "cgm论文新增17指标"

BRANCH_CANDIDATES = [
    DG / "4New_cgm",
    ROOT / "4New_cgm",
]

SPECIAL = {
    "cgm_above_140": "two_part",
    "cgm_hbgi": "log1p_hc3",
    "cgm_grade": "log1p_hc3",
    "cgm_above_180": "two_part_supplementary_only",
}

MODEL_SETS = {
    "primary_m2": ("primary", 2, 2),
    "primary_m3": ("primary", 3, 3),
    "strict_m2": ("strict", 2, 2),
    "strict_m3": ("strict", 3, 3),
}


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
    for p in BRANCH_CANDIDATES:
        if p.is_dir():
            return p
    raise FileNotFoundError("4New_cgm branch not found")


def latest_step16a(branch: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(p)
        return p
    xs = []
    for p in (branch / "outputs").glob("paper20_microbiome_mwas_*"):
        f = p / "models" / "16_paper20_microbiome_all_models.csv"
        if f.is_file():
            xs.append((f.stat().st_mtime, p))
    if not xs:
        raise FileNotFoundError("No completed Step16a run found")
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
        raise FileNotFoundError("No finalized Paper-20 CGM file found")
    xs.sort(reverse=True)
    return xs[0][1]


def z_within(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = float(x.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise RuntimeError("Transformed outcome has zero/nonfinite SD")
    return (x - x.mean()) / sd


def hc3_ols(C: np.ndarray, species_x: np.ndarray, y: np.ndarray) -> dict:
    X = np.column_stack([C, species_x]).astype(float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape

    if n <= k + 2:
        raise RuntimeError("Insufficient residual degrees of freedom")
    if np.linalg.matrix_rank(X) != k:
        raise RuntimeError("Sensitivity design is rank deficient")

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
    j = k - 1
    if not np.isfinite(se[j]) or se[j] <= 0:
        raise RuntimeError("Degenerate HC3 SE")

    stat = beta[j] / se[j]
    p = 2 * t_dist.sf(abs(stat), df=n - k)
    return {
        "beta": float(beta[j]),
        "SE": float(se[j]),
        "p_value": float(p),
        "N": int(n),
    }


def logistic_loglik(X, y, beta):
    eta = X @ beta
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))


def logistic_irls(C: np.ndarray, species_x: np.ndarray, y: np.ndarray) -> dict:
    X = np.column_stack([C, species_x]).astype(float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape

    vals = set(np.unique(y).tolist())
    if not vals.issubset({0.0, 1.0}) or len(vals) < 2:
        raise RuntimeError("Logistic outcome must contain both 0 and 1")
    if np.linalg.matrix_rank(X) != k:
        raise RuntimeError("Logistic design is rank deficient")

    beta = np.zeros(k, dtype=float)
    ll = logistic_loglik(X, y, beta)
    converged = False
    message = "max_iter reached"

    for iteration in range(1, 101):
        mu = expit(X @ beta)
        w = np.clip(mu * (1.0 - mu), 1e-12, None)
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
            ll_new = logistic_loglik(X, y, candidate)
            if np.isfinite(ll_new) and ll_new >= ll - 1e-10:
                accepted = True
                break
            scale *= 0.5

        if not accepted:
            message = "step-halving failed"
            break

        beta = candidate
        ll = ll_new
        mu = expit(X @ beta)
        score_new = X.T @ (y - mu)

        if (
            np.max(np.abs(scale * step)) < 1e-8
            or np.max(np.abs(score_new)) < 1e-6
        ):
            converged = True
            message = f"converged in {iteration} iterations"
            break

        if np.max(np.abs(beta)) > 50:
            message = "possible separation"
            break

    mu = expit(X @ beta)
    w = np.clip(mu * (1 - mu), 1e-12, None)
    fisher = X.T @ (w[:, None] * X)
    try:
        cov = np.linalg.inv(fisher)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(fisher)

    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    j = k - 1

    if (
        not converged
        or not np.isfinite(beta[j])
        or not np.isfinite(se[j])
        or se[j] <= 0
    ):
        raise RuntimeError(message)

    z = beta[j] / se[j]
    p = 2 * norm.sf(abs(z))
    return {
        "beta": float(beta[j]),
        "SE": float(se[j]),
        "p_value": float(p),
        "odds_ratio": float(np.exp(beta[j])),
        "N": int(n),
        "events": int(y.sum()),
        "message": message,
    }


def add_candidate_fdr(frame: pd.DataFrame, bh_fdr) -> pd.DataFrame:
    x = frame.copy()
    x["FDR_candidate_family"] = np.nan

    for _, idx in x.groupby(
        ["set_name", "outcome_field", "sensitivity_component"],
        sort=False,
    ).groups.items():
        idx = list(idx)
        part = x.loc[idx]
        good = part["status"].eq("computed") & part["p_value"].notna()
        p = part["p_value"].where(good, 1.0).to_numpy(float)
        q = bh_fdr(p)
        x.loc[idx, "FDR_candidate_family"] = np.where(good, q, np.nan)
        x.loc[idx, "candidate_tests_planned"] = len(part)

    x["sensitivity_FDR05"] = x["FDR_candidate_family"].lt(0.05)
    x["same_direction_as_canonical"] = (
        np.sign(pd.to_numeric(x["beta"], errors="coerce"))
        == np.sign(pd.to_numeric(x["canonical_beta"], errors="coerce"))
    )
    x["significant_same"] = (
        x["status"].eq("computed")
        & x["sensitivity_FDR05"]
        & x["same_direction_as_canonical"]
    )
    x["significant_opposite"] = (
        x["status"].eq("computed")
        & x["sensitivity_FDR05"]
        & ~x["same_direction_as_canonical"]
    )
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch-dir", default=None)
    ap.add_argument("--step16a-dir", default=None)
    args = ap.parse_args()

    branch = find_branch(args.branch_dir)
    scripts = branch / "scripts"
    sys.path.insert(0, str(scripts))

    import new_cgm_data as data
    import new_cgm_models as models
    import legacy_microbiome_models as micro_legacy
    import legacy_diet_models as diet_legacy

    step16a = latest_step16a(branch, args.step16a_dir)
    model_path = step16a / "models" / "16_paper20_microbiome_all_models.csv"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    canonical = pd.read_csv(model_path, low_memory=False)
    primary = canonical.loc[
        canonical["set_name"].eq("primary_m2")
        & canonical["outcome_field"].isin(SPECIAL)
    ].copy()
    primary["sig"] = truthy(primary["significant_family_05"])

    candidates = primary.loc[primary["sig"]].copy()
    candidate_counts = (
        candidates.groupby("outcome_field")["species"]
        .nunique()
        .reindex(SPECIAL.keys(), fill_value=0)
    )

    config = data.load_config(branch / "config" / "analysis.json")
    _, microbiome, species, _, _, _ = data.prepare_tables(config)

    final_cgm = latest_final_cgm()
    cgm = data.read_table(final_cgm)

    needed = ["participant_id"]
    for field in SPECIAL:
        needed += [field + "_raw", field + "_clean"]
    data.require_columns(cgm, needed, "finalized Paper-20 special outcomes")

    cgm_piece = cgm[needed].copy()
    for col in needed[1:]:
        cgm_piece[col] = data.numeric(cgm_piece[col], col)

    overlaps = sorted(
        (set(microbiome.columns) & set(cgm_piece.columns))
        - {"participant_id"}
    )
    if overlaps:
        microbiome = microbiome.drop(columns=overlaps)

    microbiome = microbiome.merge(
        cgm_piece,
        on="participant_id",
        how="left",
        validate="one_to_one",
    )

    print("=== STEP 16b SPECIAL-OUTCOME MICROBIOME SENSITIVITY ===")
    print(f"STEP16A_SOURCE={step16a}")
    print(f"FINAL_CGM={final_cgm}")
    print("NEW_DISCOVERY_ALLOWED=False")
    print("BRIDGE_SCREEN_RUN=False")
    print("MODEL_SETS=primary_m2,primary_m3,strict_m2,strict_m3")
    print("CANDIDATE_COUNTS:")
    for field, n in candidate_counts.items():
        print(f"  {field}={int(n)}")

    rows = []

    for field, rule in SPECIAL.items():
        cand_species = sorted(
            candidates.loc[
                candidates["outcome_field"].eq(field),
                "species",
            ].unique().tolist()
        )
        if not cand_species:
            continue

        components = (
            ["any_positive_logistic", "positive_log1p_HC3"]
            if rule.startswith("two_part")
            else ["log1p_clean_HC3"]
        )

        for set_name, (cohort_mode, fit_model, selection_model) in MODEL_SETS.items():
            required = list(data.BASE_COVARIATES)
            if selection_model == 3:
                required += ["bmi"]

            eligible = (
                microbiome["strict_eligible"]
                if cohort_mode == "strict"
                else microbiome["primary_eligible"]
            )

            rawcol = field + "_raw"
            cleancol = field + "_clean"

            for sp in cand_species:
                can = canonical.loc[
                    canonical["set_name"].eq(set_name)
                    & canonical["outcome_field"].eq(field)
                    & canonical["species"].eq(sp)
                ]
                if len(can) != 1:
                    raise RuntimeError(
                        f"Canonical lookup mismatch: {set_name}/{field}/{sp}"
                    )
                canonical_beta = float(can.iloc[0]["beta"])
                canonical_fdr = float(can.iloc[0]["FDR_family"])

                for component in components:
                    row = {
                        "set_name": set_name,
                        "analysis_set": cohort_mode,
                        "model": fit_model,
                        "outcome_field": field,
                        "species": sp,
                        "sensitivity_component": component,
                        "canonical_beta": canonical_beta,
                        "canonical_FDR_family": canonical_fdr,
                        "N": np.nan,
                        "events": np.nan,
                        "beta": np.nan,
                        "SE": np.nan,
                        "p_value": np.nan,
                        "status": "not_fitted",
                        "error": "",
                    }

                    try:
                        extra = [sp]
                        if component == "any_positive_logistic":
                            extra += [rawcol]
                        elif component == "positive_log1p_HC3":
                            extra += [rawcol]
                        else:
                            extra += [cleancol]

                        selected = microbiome.loc[
                            eligible
                            & models.complete_mask(
                                microbiome, required + extra
                            )
                        ].copy()

                        if component == "positive_log1p_HC3":
                            raw = pd.to_numeric(
                                selected[rawcol], errors="coerce"
                            )
                            selected = selected.loc[raw.gt(0)].copy()
                        elif component == "log1p_clean_HC3":
                            clean = pd.to_numeric(
                                selected[cleancol], errors="coerce"
                            )
                            selected = selected.loc[clean.ge(0)].copy()

                        if len(selected) < config["minimum_model_n"]:
                            raise RuntimeError(
                                f"Insufficient N={len(selected)}"
                            )

                        C, _, _, _, _ = (
                            micro_legacy.build_covariate_design(
                                selected,
                                include_bmi=(fit_model == 3),
                            )
                        )
                        sx = selected[sp].to_numpy(float)

                        if component == "any_positive_logistic":
                            y = (
                                pd.to_numeric(
                                    selected[rawcol], errors="raise"
                                )
                                .gt(0)
                                .astype(float)
                                .to_numpy()
                            )
                            fit = logistic_irls(C, sx, y)
                            row.update(
                                N=fit["N"],
                                events=fit["events"],
                                beta=fit["beta"],
                                SE=fit["SE"],
                                p_value=fit["p_value"],
                                odds_ratio=fit["odds_ratio"],
                                status="computed",
                                convergence_message=fit["message"],
                            )
                        elif component == "positive_log1p_HC3":
                            yraw = pd.to_numeric(
                                selected[rawcol], errors="raise"
                            ).to_numpy(float)
                            y = z_within(np.log1p(yraw))
                            fit = hc3_ols(C, sx, y)
                            row.update(
                                N=fit["N"],
                                beta=fit["beta"],
                                SE=fit["SE"],
                                p_value=fit["p_value"],
                                status="computed",
                            )
                        else:
                            yclean = pd.to_numeric(
                                selected[cleancol], errors="raise"
                            ).to_numpy(float)
                            y = z_within(np.log1p(yclean))
                            fit = hc3_ols(C, sx, y)
                            row.update(
                                N=fit["N"],
                                beta=fit["beta"],
                                SE=fit["SE"],
                                p_value=fit["p_value"],
                                status="computed",
                            )

                    except Exception as e:
                        row.update(status="failed", error=repr(e))

                    rows.append(row)

    sens = pd.DataFrame(rows)
    if sens.empty:
        raise RuntimeError("No special-outcome canonical candidates found")

    sens = add_candidate_fdr(sens, diet_legacy.bh_fdr)

    # Candidate-level consolidation.
    records = []
    for (field, sp), g in sens.groupby(
        ["outcome_field", "species"], sort=False
    ):
        rule = SPECIAL[field]
        rec = {
            "outcome_field": field,
            "species": sp,
            "rule": rule,
            "canonical_primary_m2_FDR": float(
                candidates.loc[
                    candidates["outcome_field"].eq(field)
                    & candidates["species"].eq(sp),
                    "FDR_family",
                ].iloc[0]
            ),
        }

        all_computed = g["status"].eq("computed").all()
        rec["all_sensitivity_fits_computed"] = bool(all_computed)

        for set_name in MODEL_SETS:
            z = g.loc[g["set_name"].eq(set_name)]
            rec[f"{set_name}_components"] = len(z)
            rec[f"{set_name}_significant_same"] = int(
                z["significant_same"].sum()
            )
            rec[f"{set_name}_significant_opposite"] = int(
                z["significant_opposite"].sum()
            )
            rec[f"{set_name}_direction_same"] = int(
                (
                    z["status"].eq("computed")
                    & z["same_direction_as_canonical"]
                ).sum()
            )

        # Canonical candidate is blocked only by a significant opposite
        # sensitivity result or an uncomputable required audit.
        any_sig_opposite = bool(sens.loc[
            sens["outcome_field"].eq(field)
            & sens["species"].eq(sp),
            "significant_opposite",
        ].any())

        rec["special_rule_ok"] = bool(
            all_computed and not any_sig_opposite
        )

        # Stronger annotation: primary_m2 and strict_m2 each have at least
        # one FDR-significant concordant sensitivity component and no
        # significant opposite component.
        p2 = g.loc[g["set_name"].eq("primary_m2")]
        s2 = g.loc[g["set_name"].eq("strict_m2")]
        rec["sensitivity_high_support"] = bool(
            all_computed
            and int(p2["significant_same"].sum()) >= 1
            and int(s2["significant_same"].sum()) >= 1
            and int(p2["significant_opposite"].sum()) == 0
            and int(s2["significant_opposite"].sum()) == 0
        )

        rec["bridge_role"] = (
            "supplementary_no_primary_diet_gate"
            if field == "cgm_above_180"
            else "eligible_for_special_outcome_bridge_audit"
        )
        records.append(rec)

    gate = pd.DataFrame(records)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = branch / "outputs" / f"paper20_microbiome_special_sensitivity_{stamp}"
    reports = out / "reports"
    models_dir = out / "models"
    reports.mkdir(parents=True, exist_ok=False)
    models_dir.mkdir()

    sens_path = models_dir / "16b_special_outcome_sensitivity_models.csv"
    gate_path = reports / "16b_special_species_gate.csv"
    blocked_path = reports / "16b_special_species_blocked.csv"
    high_path = reports / "16b_special_species_high_support.csv"
    txt_path = reports / "16b_special_outcome_sensitivity_summary.txt"

    sens.to_csv(sens_path, index=False)
    gate.to_csv(gate_path, index=False)
    gate.loc[~gate["special_rule_ok"]].to_csv(blocked_path, index=False)
    gate.loc[gate["sensitivity_high_support"]].to_csv(high_path, index=False)

    summary = (
        gate.groupby(["outcome_field", "bridge_role"], as_index=False)
        .agg(
            canonical_candidates=("species", "size"),
            special_rule_ok=("special_rule_ok", "sum"),
            blocked=("special_rule_ok", lambda x: int((~x).sum())),
            high_support=("sensitivity_high_support", "sum"),
        )
    )

    failed = int(sens["status"].ne("computed").sum())
    sig_opp = int(sens["significant_opposite"].sum())

    lines = [
        "=== STEP 16b SPECIAL-OUTCOME MICROBIOME SENSITIVITY SUMMARY ===",
        f"STEP16A_SOURCE={step16a}",
        f"FINAL_CGM={final_cgm}",
        "NEW_DISCOVERY_ALLOWED=False",
        "BRIDGE_SCREEN_RUN=False",
        f"SENSITIVITY_MODEL_ROWS={len(sens)}",
        f"FAILED={failed}",
        f"SIGNIFICANT_OPPOSITE_COMPONENTS={sig_opp}",
        "",
        "--- CANDIDATE COUNTS FROM CANONICAL PRIMARY M2 ---",
    ]
    for field, n in candidate_counts.items():
        lines.append(f"{field}={int(n)}")

    lines += [
        "",
        "--- SPECIAL SPECIES GATE SUMMARY ---",
        summary.to_string(index=False),
        "",
        "INTERPRETATION RULES:",
        "- Step16a canonical Primary Model2 MWAS remains the discovery set.",
        "- Sensitivity-only gains never create new species discoveries.",
        "- special_rule_ok requires all prespecified fits to compute and no candidate-FDR-significant opposite-direction sensitivity result.",
        "- sensitivity_high_support is a stronger annotation, not a redefinition of discovery.",
        "- TAR180 remains supplementary because Step15b has no primary Diet->TAR180 bridge gate.",
        "- TAR140/HBGI/GRADE special_rule_ok can be used in the next bridge construction.",
        "",
        f"SENSITIVITY_MODELS={sens_path}",
        f"SPECIES_GATE={gate_path}",
        f"BLOCKED={blocked_path}",
        f"HIGH_SUPPORT={high_path}",
        f"OUTPUT_DIR={out}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    (reports / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed" if failed == 0 else "completed_with_failures",
                "new_discovery_allowed": False,
                "bridge_screen_run": False,
                "sensitivity_model_rows": int(len(sens)),
                "failed": failed,
                "significant_opposite_components": sig_opp,
                "candidate_counts": {
                    str(k): int(v) for k, v in candidate_counts.items()
                },
                "outputs": {
                    "sensitivity_models": str(sens_path),
                    "species_gate": str(gate_path),
                    "blocked": str(blocked_path),
                    "high_support": str(high_path),
                    "summary": str(txt_path),
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
