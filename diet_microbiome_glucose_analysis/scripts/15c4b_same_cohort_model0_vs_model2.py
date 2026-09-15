#!/usr/bin/env python3
"""Step 15c4b — same-cohort Model0 vs Model2 QC for new diet MWAS.

Purpose
-------
Step 15c4 showed fewer FDR-significant species after Model2 adjustment, but
Model2 also uses a smaller complete-case sample. This QC separates:

A) sample-selection effect:
   full-sample Model0  vs same-cohort Model0

B) covariate-adjustment effect:
   same-cohort Model0 vs Model2

For each exposure, same-cohort Model0 is refit on EXACTLY the participants used
by Step15c4 Model2.

No canonical file is modified.

Outputs
-------
diet_microbiome_glucose_analysis/outputs/new_diet_extension/
    models/
        15c4b_samecohort_MWAS_EAT13_model0.csv
        15c4b_samecohort_MWAS_NOVA4_model0.csv
        15c4b_samecohort_MWAS_Carbohydrate_pct_model0.csv
    reports/
        15c4b_samecohort_model0_vs_model2_summary.csv
        15c4b_samecohort_model0_vs_model2_summary.txt
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist, spearmanr


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
BASE = (
    ROOT / "diet_microbiome_glucose_analysis" / "outputs"
    / "new_diet_extension"
)
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "models"
REPORT_DIR = BASE / "reports"

SCORE_FILE = DATA_DIR / "15c0_new_diet_microbiome_scores_aligned.csv"
MICRO_FILE = DATA_DIR / "15c0_new_diet_microbiome_species_aligned.csv"
COV_FILE = ROOT / "co-variant" / "outputs" / "data" / "02_covariate_master.csv"

FULL_MODEL0 = MODEL_DIR / "15c3_MWAS_all_model0.csv"
MODEL2 = MODEL_DIR / "15c4_MWAS_all_model2.csv"

SUMMARY_CSV = REPORT_DIR / "15c4b_samecohort_model0_vs_model2_summary.csv"
SUMMARY_TXT = REPORT_DIR / "15c4b_samecohort_model0_vs_model2_summary.txt"

META_COLS = ["participant_id", "cohort", "research_stage", "array_index"]
COMMON_FLAG = "amed_microbiome_model2_covariates_complete"
ENERGY_COL = "mean_daily_energy_kcal"

EXPOSURES = {
    "EAT13": "EAT13_z",
    "NOVA4": "NOVA4_z",
    "Carbohydrate_pct": "Carbohydrate_pct_z",
}


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


def bh_fdr(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    if not np.isfinite(p).all():
        raise RuntimeError("BH-FDR received non-finite p-values")
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


def fit_univariate_ols(x, Y):
    x = np.asarray(x, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    xc = x - x.mean()
    Y_mean = Y.mean(axis=0)
    Yc = Y - Y_mean

    sxx = np.sum(xc ** 2)
    if sxx <= 0:
        raise RuntimeError("Exposure has zero variance")

    beta = (xc[:, None] * Yc).sum(axis=0) / sxx
    intercept = Y_mean - beta * x.mean()
    resid = Y - (intercept[None, :] + x[:, None] * beta[None, :])

    n = len(x)
    df = n - 2
    sse = np.sum(resid ** 2, axis=0)
    mse = sse / df
    se = np.sqrt(mse / sxx)
    t_stat = beta / se
    p = 2 * t_dist.sf(np.abs(t_stat), df=df)

    sst = np.sum(Yc ** 2, axis=0)
    r2 = 1 - sse / sst
    tcrit = t_dist.ppf(.975, df=df)

    return {
        "beta": beta,
        "SE": se,
        "t": t_stat,
        "p": p,
        "R2": r2,
        "CI_lower": beta - tcrit * se,
        "CI_upper": beta + tcrit * se,
        "df_resid": df,
    }


def compare_pair(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name: str,
    right_name: str,
) -> dict:
    cols = ["species", "beta", "FDR", "significant_FDR05"]
    m = left[cols].merge(
        right[cols],
        on="species",
        how="inner",
        suffixes=(f"_{left_name}", f"_{right_name}"),
        validate="one_to_one",
    )
    if len(m) != 379:
        raise RuntimeError(
            f"{left_name} vs {right_name}: expected 379 species, got {len(m)}"
        )

    l_sig = truthy(m[f"significant_FDR05_{left_name}"])
    r_sig = truthy(m[f"significant_FDR05_{right_name}"])

    l_beta = pd.to_numeric(m[f"beta_{left_name}"], errors="coerce")
    r_beta = pd.to_numeric(m[f"beta_{right_name}"], errors="coerce")
    rho, rho_p = spearmanr(l_beta, r_beta)

    sign_flip = np.sign(l_beta.to_numpy()) != np.sign(r_beta.to_numpy())
    both = l_sig & r_sig
    same_both = int(
        (
            np.sign(l_beta.loc[both].to_numpy())
            == np.sign(r_beta.loc[both].to_numpy())
        ).sum()
    )

    return {
        f"{left_name}_significant": int(l_sig.sum()),
        f"{right_name}_significant": int(r_sig.sum()),
        f"{left_name}_sig_retained_in_{right_name}": int((l_sig & r_sig).sum()),
        f"{left_name}_sig_lost_in_{right_name}": int((l_sig & ~r_sig).sum()),
        f"{right_name}_sig_gained_vs_{left_name}": int((~l_sig & r_sig).sum()),
        f"same_direction_among_both_sig_{left_name}_vs_{right_name}": same_both,
        f"opposite_direction_among_both_sig_{left_name}_vs_{right_name}": int(both.sum()) - same_both,
        f"beta_rho_{left_name}_vs_{right_name}": float(rho),
        f"beta_rho_p_{left_name}_vs_{right_name}": float(rho_p),
        f"beta_sign_flips_{left_name}_vs_{right_name}": int(sign_flip.sum()),
    }


def main() -> int:
    for p, label in [
        (SCORE_FILE, "aligned scores"),
        (MICRO_FILE, "aligned microbiome"),
        (COV_FILE, "covariate master"),
        (FULL_MODEL0, "full-sample Model0 results"),
        (MODEL2, "Model2 results"),
    ]:
        require(p, label)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(SCORE_FILE, low_memory=False)
    micro = pd.read_csv(MICRO_FILE, low_memory=False)
    cov = pd.read_csv(
        COV_FILE,
        usecols=lambda c: c in {"participant_id", COMMON_FLAG},
        low_memory=False,
    )
    full0 = pd.read_csv(FULL_MODEL0, low_memory=False)
    model2 = pd.read_csv(MODEL2, low_memory=False)

    for label, d in [("scores", scores), ("microbiome", micro), ("covariates", cov)]:
        if "participant_id" not in d.columns:
            raise RuntimeError(f"{label} missing participant_id")
        d["participant_id"] = norm_id(d["participant_id"])
        if d["participant_id"].duplicated().any():
            raise RuntimeError(f"{label} duplicate participant_id")

    if scores["participant_id"].tolist() != micro["participant_id"].tolist():
        raise RuntimeError("Score/microbiome participant order differs")

    if COMMON_FLAG not in cov.columns:
        raise RuntimeError(f"Covariate master missing {COMMON_FLAG}")

    species = [c for c in micro.columns if c not in META_COLS]
    if len(species) != 379:
        raise RuntimeError(f"Expected 379 species, found {len(species)}")

    Y = micro[species].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(Y).all():
        raise RuntimeError("Microbiome CLR-Z contains NaN/Inf")

    merged = scores.merge(cov, on="participant_id", how="left", validate="one_to_one")
    base_complete = truthy(merged[COMMON_FLAG])
    energy = pd.to_numeric(merged[ENERGY_COL], errors="coerce")
    energy_ok = energy.notna() & np.isfinite(energy.to_numpy(float))

    print("=== STEP 15c4b SAME-COHORT MODEL0 vs MODEL2 QC ===")
    print("PURPOSE=separate sample-selection from covariate-adjustment")
    print("ALL_THREE_COMPLETE_REQUIRED=False")

    rows = []

    for exposure, col in EXPOSURES.items():
        x = pd.to_numeric(merged[col], errors="coerce")
        exposure_ok = x.notna() & np.isfinite(x.to_numpy(float))

        valid = base_complete & energy_ok & exposure_ok
        idx = np.flatnonzero(valid.to_numpy())

        x_use = x.loc[valid].to_numpy(float)
        Y_use = Y[idx, :]

        fit = fit_univariate_ols(x_use, Y_use)
        fdr = bh_fdr(fit["p"])

        same0 = pd.DataFrame({
            "exposure": exposure,
            "species": species,
            "N": len(x_use),
            "model": "same_cohort_model0",
            "beta": fit["beta"],
            "SE": fit["SE"],
            "t": fit["t"],
            "p": fit["p"],
            "FDR": fdr,
            "R2": fit["R2"],
            "CI_lower": fit["CI_lower"],
            "CI_upper": fit["CI_upper"],
            "df_resid": fit["df_resid"],
        })
        same0["significant_FDR05"] = same0["FDR"] < .05
        same0["direction"] = np.where(same0["beta"] > 0, "positive", "negative")

        out_path = MODEL_DIR / f"15c4b_samecohort_MWAS_{exposure}_model0.csv"
        same0.to_csv(out_path, index=False)

        full0_e = full0.loc[full0["exposure"].eq(exposure)].copy()
        model2_e = model2.loc[model2["exposure"].eq(exposure)].copy()

        if len(full0_e) != 379 or len(model2_e) != 379:
            raise RuntimeError(f"{exposure}: expected 379 rows in prior models")

        sel = compare_pair(
            full0_e,
            same0,
            "full0",
            "same0",
        )
        adj = compare_pair(
            same0,
            model2_e,
            "same0",
            "model2",
        )

        row = {
            "exposure": exposure,
            "full_model0_N": int(pd.to_numeric(full0_e["N"]).median()),
            "same_cohort_N": len(x_use),
            "model2_N": int(pd.to_numeric(model2_e["N"]).median()),
            **sel,
            **adj,
        }
        rows.append(row)

        print("\n" + "=" * 100)
        print(f"{exposure}")
        print("=" * 100)
        print(
            f"N: full0={row['full_model0_N']} "
            f"same0={row['same_cohort_N']} "
            f"model2={row['model2_N']}"
        )
        print(
            "Sample selection | "
            f"sig {row['full0_significant']} -> {row['same0_significant']} | "
            f"retained={row['full0_sig_retained_in_same0']} | "
            f"lost={row['full0_sig_lost_in_same0']} | "
            f"gained={row['same0_sig_gained_vs_full0']} | "
            f"beta rho={row['beta_rho_full0_vs_same0']:.6f} | "
            f"sign flips={row['beta_sign_flips_full0_vs_same0']}"
        )
        print(
            "Adjustment only | "
            f"sig {row['same0_significant']} -> {row['model2_significant']} | "
            f"retained={row['same0_sig_retained_in_model2']} | "
            f"lost={row['same0_sig_lost_in_model2']} | "
            f"gained={row['model2_sig_gained_vs_same0']} | "
            f"beta rho={row['beta_rho_same0_vs_model2']:.6f} | "
            f"sign flips={row['beta_sign_flips_same0_vs_model2']}"
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False)

    print("\n--- SAME-COHORT DECOMPOSITION SUMMARY ---")
    display_cols = [
        "exposure",
        "full_model0_N",
        "same_cohort_N",
        "full0_significant",
        "same0_significant",
        "model2_significant",
        "full0_sig_lost_in_same0",
        "same0_sig_lost_in_model2",
        "beta_rho_full0_vs_same0",
        "beta_rho_same0_vs_model2",
        "beta_sign_flips_full0_vs_same0",
        "beta_sign_flips_same0_vs_model2",
    ]
    print(summary[display_cols].to_string(index=False))

    lines = [
        "=== STEP 15c4b SAME-COHORT MODEL0 vs MODEL2 QC ===",
        "PURPOSE=separate sample-selection effect from covariate-adjustment effect",
        "",
        summary.to_string(index=False),
        "",
        "INTERPRETATION:",
        "- full0 -> same0 quantifies the effect of restricting to the Model2",
        "  complete-case/energy-available cohort without covariate adjustment.",
        "- same0 -> model2 quantifies covariate adjustment on the exact same sample.",
        "- Do not attribute the full full0->model2 difference solely to confounding.",
        "",
        f"SUMMARY_CSV={SUMMARY_CSV}",
    ]
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
