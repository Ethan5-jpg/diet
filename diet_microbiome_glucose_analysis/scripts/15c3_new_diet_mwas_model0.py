#!/usr/bin/env python3
"""Step 15c3 — microbiome-wide association Model0 for frozen new diet exposures.

Mirrors canonical 03_microbiome_wide_unadjusted.py:
    species CLR-Z ~ dietary exposure Z

No covariates.
BH-FDR is applied separately within each exposure across 379 species, exactly
as in the canonical old-score MWAS.

Frozen exposures
----------------
EAT13             primary extension
NOVA4             primary extension
Carbohydrate_pct  exploratory extension

Each exposure uses its own maximum valid microbiome sample. There is no
all-three-complete requirement.

Outputs
-------
diet_microbiome_glucose_analysis/outputs/new_diet_extension/
    models/
        15c3_MWAS_EAT13_model0.csv
        15c3_MWAS_NOVA4_model0.csv
        15c3_MWAS_Carbohydrate_pct_model0.csv
        15c3_MWAS_all_model0.csv
    reports/
        15c3_MWAS_model0_summary.csv
        15c3_MWAS_model0_pairwise_overlap.csv
        15c3_MWAS_model0_summary.txt
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist


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

ALL_OUT = MODEL_DIR / "15c3_MWAS_all_model0.csv"
SUMMARY_CSV = REPORT_DIR / "15c3_MWAS_model0_summary.csv"
OVERLAP_CSV = REPORT_DIR / "15c3_MWAS_model0_pairwise_overlap.csv"
SUMMARY_TXT = REPORT_DIR / "15c3_MWAS_model0_summary.txt"

META_COLS = ["participant_id", "cohort", "research_stage", "array_index"]

EXPOSURES = {
    "EAT13": {
        "col": "EAT13_z",
        "role": "primary_extension",
        "polarity": "higher_more_adherent",
    },
    "NOVA4": {
        "col": "NOVA4_z",
        "role": "primary_extension",
        "polarity": "higher_more_ultraprocessed",
    },
    "Carbohydrate_pct": {
        "col": "Carbohydrate_pct_z",
        "role": "exploratory_extension",
        "polarity": "neutral_macronutrient_composition",
    },
}


def require(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


def normalize_id(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def bh_fdr(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    if not np.isfinite(p).all():
        raise RuntimeError("BH-FDR received non-finite p-values.")
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out


def fit_univariate_ols(x, Y):
    """Canonical Model0 vectorized OLS: Y_j = intercept + beta_j*x."""
    x = np.asarray(x, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    xc = x - x.mean()
    Y_mean = Y.mean(axis=0)
    Yc = Y - Y_mean

    sxx = np.sum(xc ** 2)
    if sxx <= 0:
        raise RuntimeError("Exposure has zero variance.")

    beta = (xc[:, None] * Yc).sum(axis=0) / sxx
    intercept = Y_mean - beta * x.mean()

    fitted = intercept[None, :] + x[:, None] * beta[None, :]
    resid = Y - fitted

    n = len(x)
    df = n - 2
    if df <= 0:
        raise RuntimeError("Insufficient residual degrees of freedom.")

    sse = np.sum(resid ** 2, axis=0)
    mse = sse / df
    se = np.sqrt(mse / sxx)
    t_stat = beta / se
    p = 2 * t_dist.sf(np.abs(t_stat), df=df)

    sst = np.sum(Yc ** 2, axis=0)
    r2 = 1 - sse / sst

    return beta, se, t_stat, p, r2


def pairwise_overlap(all_results: pd.DataFrame) -> pd.DataFrame:
    names = list(EXPOSURES)
    rows = []
    sig_sets = {
        name: set(
            all_results.loc[
                all_results["exposure"].eq(name)
                & all_results["significant_FDR05"],
                "species",
            ].astype(str)
        )
        for name in names
    }

    beta_wide = all_results.pivot(
        index="species", columns="exposure", values="beta"
    )

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            A, B = sig_sets[a], sig_sets[b]
            both = A & B
            union = A | B

            same_direction = 0
            opposite_direction = 0
            if both:
                sub = beta_wide.loc[list(both), [a, b]]
                same_direction = int(
                    (
                        np.sign(sub[a].to_numpy(float))
                        == np.sign(sub[b].to_numpy(float))
                    ).sum()
                )
                opposite_direction = len(both) - same_direction

            rows.append({
                "exposure_1": a,
                "exposure_2": b,
                "sig_1": len(A),
                "sig_2": len(B),
                "shared_significant_species": len(both),
                "union_significant_species": len(union),
                "jaccard": len(both) / len(union) if union else np.nan,
                "same_direction_among_shared": same_direction,
                "opposite_direction_among_shared": opposite_direction,
            })
    return pd.DataFrame(rows)


def main() -> int:
    require(SCORE_FILE, "Step15c0 aligned score table")
    require(MICRO_FILE, "Step15c0 aligned microbiome table")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(SCORE_FILE, low_memory=False)
    micro = pd.read_csv(MICRO_FILE, low_memory=False)

    for label, d in [("scores", scores), ("microbiome", micro)]:
        if "participant_id" not in d.columns:
            raise RuntimeError(f"{label} missing participant_id")
        d["participant_id"] = normalize_id(d["participant_id"])
        if d["participant_id"].duplicated().any():
            raise RuntimeError(f"{label} has duplicate participant_id")

    if scores["participant_id"].tolist() != micro["participant_id"].tolist():
        raise RuntimeError("Score and microbiome participant order differs.")

    missing_exposure_cols = sorted(
        {cfg["col"] for cfg in EXPOSURES.values()} - set(scores.columns)
    )
    if missing_exposure_cols:
        raise RuntimeError(
            "Aligned score table missing exposure columns: "
            + ", ".join(missing_exposure_cols)
        )

    species = [c for c in micro.columns if c not in META_COLS]
    if len(species) != 379:
        raise RuntimeError(f"Expected 379 species, found {len(species)}")

    Y = (
        micro[species]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64)
    )
    if not np.isfinite(Y).all():
        raise RuntimeError("Species CLR-Z matrix contains NaN/Inf.")

    print("=== STEP 15c3 NEW DIET -> MICROBIOME MWAS MODEL0 ===")
    print(f"UNION_PARTICIPANTS={len(scores)}")
    print(f"SPECIES={len(species)}")
    print("MODEL=species CLR-Z ~ exposure Z")
    print("COVARIATES=NONE")
    print("FDR_FAMILY=379 species separately within each exposure")
    print("ALL_THREE_COMPLETE_REQUIRED=False")

    all_results = []
    summary_rows = []

    for exposure, cfg in EXPOSURES.items():
        x = pd.to_numeric(scores[cfg["col"]], errors="coerce").to_numpy(float)
        valid = np.isfinite(x)
        x_use = x[valid]
        Y_use = Y[valid, :]

        if len(x_use) < 3:
            raise RuntimeError(f"{exposure}: <3 valid participants")

        beta, se, t_stat, p, r2 = fit_univariate_ols(x_use, Y_use)
        fdr = bh_fdr(p)

        result = pd.DataFrame({
            "exposure": exposure,
            "analysis_role": cfg["role"],
            "polarity": cfg["polarity"],
            "species": species,
            "N": len(x_use),
            "model": 0,
            "beta": beta,
            "SE": se,
            "t": t_stat,
            "p": p,
            "FDR": fdr,
            "R2": r2,
        })

        # Keep canonical Model0 CI convention for direct comparability.
        result["CI_lower"] = result["beta"] - 1.96 * result["SE"]
        result["CI_upper"] = result["beta"] + 1.96 * result["SE"]
        result["significant_FDR05"] = result["FDR"] < .05
        result["direction"] = np.where(
            result["beta"] > 0, "positive", "negative"
        )

        sig = result.loc[result["significant_FDR05"]].copy()
        n_pos = int((sig["beta"] > 0).sum())
        n_neg = int((sig["beta"] < 0).sum())

        out_path = MODEL_DIR / f"15c3_MWAS_{exposure}_model0.csv"
        result.to_csv(out_path, index=False)
        all_results.append(result)

        summary_rows.append({
            "exposure": exposure,
            "analysis_role": cfg["role"],
            "N": len(x_use),
            "species_tested": len(species),
            "significant_FDR05": len(sig),
            "positive_FDR05": n_pos,
            "negative_FDR05": n_neg,
            "min_p": float(result["p"].min()),
            "min_FDR": float(result["FDR"].min()),
            "max_positive_beta": float(result["beta"].max()),
            "min_negative_beta": float(result["beta"].min()),
        })

        print("\n" + "=" * 96)
        print(f"{exposure} | {cfg['role']} | N={len(x_use)}")
        print("=" * 96)
        print(
            f"FDR<0.05={len(sig)} | "
            f"positive={n_pos} | negative={n_neg}"
        )

        print("\nTOP POSITIVE")
        print(
            result.nlargest(10, "beta")[
                ["species", "beta", "p", "FDR"]
            ].to_string(index=False)
        )

        print("\nTOP NEGATIVE")
        print(
            result.nsmallest(10, "beta")[
                ["species", "beta", "p", "FDR"]
            ].to_string(index=False)
        )

    combined = pd.concat(all_results, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    overlap = pairwise_overlap(combined)

    combined.to_csv(ALL_OUT, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    overlap.to_csv(OVERLAP_CSV, index=False)

    print("\n--- MODEL0 SUMMARY ---")
    print(summary.to_string(index=False))

    print("\n--- PAIRWISE FDR-SIGNIFICANT SPECIES OVERLAP ---")
    print(overlap.to_string(index=False))

    lines = [
        "=== STEP 15c3 NEW DIET -> MICROBIOME MWAS MODEL0 SUMMARY ===",
        "MODEL=species CLR-Z ~ exposure Z",
        "COVARIATES=NONE",
        "FDR_FAMILY=379 species within each exposure",
        "ALL_THREE_COMPLETE_REQUIRED=False",
        "",
        "--- SUMMARY ---",
        summary.to_string(index=False),
        "",
        "--- PAIRWISE OVERLAP ---",
        overlap.to_string(index=False),
        "",
        "INTERPRETATION NOTES:",
        "- EAT13 and NOVA4 are primary extensions.",
        "- Carbohydrate_pct is exploratory.",
        "- NOVA4 polarity is adverse-facing: beta>0 means greater relative abundance",
        "  with higher ultra-processed-food energy share.",
        "- Carbohydrate_pct has no healthy/unhealthy polarity.",
        "- These are unadjusted microbiome-wide associations, not causal effects.",
        "",
        f"ALL_RESULTS={ALL_OUT}",
        f"SUMMARY_CSV={SUMMARY_CSV}",
        f"OVERLAP_CSV={OVERLAP_CSV}",
    ]
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
