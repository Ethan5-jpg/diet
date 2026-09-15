#!/usr/bin/env python3
"""Step 15c1 — alpha diversity analysis for frozen new diet exposures.

This mirrors the canonical old-score alpha-diversity analysis:
- quintile descriptives;
- Kruskal-Wallis across five quintiles;
- approximate epsilon-squared effect size;
- continuous exposure vs diversity Spearman correlation.

No covariate-adjusted model is introduced here because the canonical alpha
analysis was an unadjusted quintile comparison / continuous sanity check.

Multiple testing
----------------
There are 3 exposures × 2 alpha metrics = 6 tests.
Raw Kruskal-Wallis P is retained as the primary descriptive comparison, and
BH-FDR across all 6 tests is reported as an additional QC.

Important
---------
- Exposure-specific maximum N is used.
- All-three-exposures-complete is NOT required.
- Carbohydrate_pct remains exploratory and is not interpreted as a
  diet-quality score.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
BASE = (
    ROOT / "diet_microbiome_glucose_analysis" / "outputs"
    / "new_diet_extension"
)
DATA_DIR = BASE / "data"
REPORT_DIR = BASE / "reports"

SCORE_FILE = DATA_DIR / "15c0_new_diet_microbiome_scores_aligned.csv"
ALPHA_FILE = DATA_DIR / "15c0_new_diet_microbiome_alpha_aligned.csv"

DESC_OUT = REPORT_DIR / "15c1_alpha_diversity_by_quintile.csv"
KW_OUT = REPORT_DIR / "15c1_alpha_diversity_kruskal_wallis.csv"
SP_OUT = REPORT_DIR / "15c1_alpha_diversity_continuous_spearman.csv"
TABLE_OUT = DATA_DIR / "15c1_new_diet_alpha_analysis_table.csv"
SUMMARY_OUT = REPORT_DIR / "15c1_alpha_diversity_summary.txt"

EXPOSURES = {
    "EAT13": {
        "z": "EAT13_z",
        "q": "EAT13_quintile",
        "role": "primary_extension",
        "polarity": "higher_modified_EAT_Lancet_adherence",
    },
    "NOVA4": {
        "z": "NOVA4_z",
        "q": "NOVA4_quintile",
        "role": "primary_extension",
        "polarity": "higher_ultra_processed_food_energy_share",
    },
    "Carbohydrate_pct": {
        "z": "Carbohydrate_pct_z",
        "q": "Carbohydrate_pct_quintile",
        "role": "exploratory_extension",
        "polarity": "neutral_macronutrient_composition",
    },
}

METRICS = ["shannon_index", "simpson_index"]


def require(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


def normalize_id(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def normalize_quintile(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def bh_fdr(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    if len(p) == 0:
        return p
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


def main() -> int:
    require(SCORE_FILE, "Step15c0 aligned score file")
    require(ALPHA_FILE, "Step15c0 aligned alpha file")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(SCORE_FILE, low_memory=False)
    alpha = pd.read_csv(ALPHA_FILE, low_memory=False)

    for label, d in [("scores", scores), ("alpha", alpha)]:
        if "participant_id" not in d.columns:
            raise RuntimeError(f"{label} missing participant_id")
        d["participant_id"] = normalize_id(d["participant_id"])
        if d["participant_id"].duplicated().any():
            raise RuntimeError(f"{label} has duplicate participant_id")

    if scores["participant_id"].tolist() != alpha["participant_id"].tolist():
        raise RuntimeError(
            "Score and alpha participant order differs. Do not continue."
        )

    for metric in METRICS:
        if metric not in alpha.columns:
            raise RuntimeError(f"Alpha metric missing: {metric}")

    needed_score_cols = ["participant_id"]
    for cfg in EXPOSURES.values():
        needed_score_cols.extend([cfg["z"], cfg["q"]])
    missing_score = sorted(set(needed_score_cols) - set(scores.columns))
    if missing_score:
        raise RuntimeError(
            f"Aligned score table missing columns: {missing_score}"
        )

    df = scores[needed_score_cols].merge(
        alpha[["participant_id", *METRICS]],
        on="participant_id",
        how="inner",
        validate="one_to_one",
    )

    desc_rows = []
    kw_rows = []
    sp_rows = []

    print("=== STEP 15c1 NEW DIET ALPHA DIVERSITY ===")
    print(f"UNION_PARTICIPANTS={len(df)}")
    print("ALL_THREE_COMPLETE_REQUIRED=False")

    for exposure, cfg in EXPOSURES.items():
        df[cfg["q"]] = normalize_quintile(df[cfg["q"]])
        q_levels = sorted(
            pd.to_numeric(df[cfg["q"]], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        if q_levels != [1, 2, 3, 4, 5]:
            raise RuntimeError(
                f"{exposure}: expected quintiles 1..5, found {q_levels}"
            )

        print("\n" + "=" * 88)
        print(f"{exposure} | {cfg['role']}")
        print("=" * 88)

        for metric in METRICS:
            temp = df[[cfg["q"], cfg["z"], metric]].copy()
            temp[cfg["z"]] = pd.to_numeric(temp[cfg["z"]], errors="coerce")
            temp[cfg["q"]] = pd.to_numeric(temp[cfg["q"]], errors="coerce")
            temp[metric] = pd.to_numeric(temp[metric], errors="coerce")

            # Group comparison uses valid quintile + metric.
            qt = temp[[cfg["q"], metric]].dropna()
            groups = []

            print(f"\n{metric}")
            for q in range(1, 6):
                values = qt.loc[qt[cfg["q"]].eq(q), metric].dropna()
                if len(values) == 0:
                    raise RuntimeError(
                        f"{exposure} {metric}: Q{q} is empty"
                    )
                groups.append(values.to_numpy())

                q25 = float(values.quantile(.25))
                med = float(values.median())
                q75 = float(values.quantile(.75))

                desc_rows.append({
                    "exposure": exposure,
                    "analysis_role": cfg["role"],
                    "metric": metric,
                    "quintile": q,
                    "N": len(values),
                    "mean": float(values.mean()),
                    "sd": float(values.std()),
                    "median": med,
                    "q25": q25,
                    "q75": q75,
                    "min": float(values.min()),
                    "max": float(values.max()),
                })

                print(
                    f"Q{q}: N={len(values):4d} | "
                    f"median={med:.6f} | IQR={q25:.6f}-{q75:.6f}"
                )

            kw = kruskal(*groups)
            n_total = sum(len(g) for g in groups)
            k = len(groups)
            epsilon_sq = max(
                0.0,
                float((kw.statistic - k + 1) / (n_total - k)),
            )

            kw_rows.append({
                "exposure": exposure,
                "analysis_role": cfg["role"],
                "metric": metric,
                "N": n_total,
                "H": float(kw.statistic),
                "df": k - 1,
                "p_raw": float(kw.pvalue),
                "epsilon_squared": epsilon_sq,
            })

            # Continuous sanity check uses valid Z + metric.
            cont = temp[[cfg["z"], metric]].dropna()
            finite = (
                np.isfinite(cont[cfg["z"]].to_numpy(float))
                & np.isfinite(cont[metric].to_numpy(float))
            )
            cont = cont.loc[finite]
            if len(cont) < 3:
                raise RuntimeError(
                    f"{exposure} {metric}: fewer than 3 continuous pairs"
                )

            sp = spearmanr(cont[cfg["z"]], cont[metric])
            sp_rows.append({
                "exposure": exposure,
                "analysis_role": cfg["role"],
                "metric": metric,
                "N": len(cont),
                "rho": float(sp.statistic),
                "p_raw": float(sp.pvalue),
                "polarity": cfg["polarity"],
            })

            print(
                f"Kruskal-Wallis H={kw.statistic:.6f}, "
                f"P={kw.pvalue:.8g}, epsilon2={epsilon_sq:.6g}"
            )
            print(
                f"Continuous Spearman rho={sp.statistic:.5f}, "
                f"P={sp.pvalue:.8g}"
            )

    desc = pd.DataFrame(desc_rows)
    kw = pd.DataFrame(kw_rows)
    sp = pd.DataFrame(sp_rows)

    if len(kw) != 6 or len(sp) != 6:
        raise RuntimeError(
            f"Expected 6 alpha tests, got KW={len(kw)}, Spearman={len(sp)}"
        )

    kw["p_fdr_6tests"] = bh_fdr(kw["p_raw"].to_numpy())
    sp["p_fdr_6tests"] = bh_fdr(sp["p_raw"].to_numpy())

    kw["raw_P05"] = kw["p_raw"] < .05
    kw["FDR_6tests_05"] = kw["p_fdr_6tests"] < .05
    sp["raw_P05"] = sp["p_raw"] < .05
    sp["FDR_6tests_05"] = sp["p_fdr_6tests"] < .05

    desc.to_csv(DESC_OUT, index=False)
    kw.to_csv(KW_OUT, index=False)
    sp.to_csv(SP_OUT, index=False)
    df.to_csv(TABLE_OUT, index=False)

    print("\n--- KRUSKAL-WALLIS SUMMARY ---")
    print(
        kw[
            [
                "exposure", "metric", "N", "H", "p_raw",
                "p_fdr_6tests", "epsilon_squared",
                "raw_P05", "FDR_6tests_05",
            ]
        ].to_string(index=False)
    )

    print("\n--- CONTINUOUS SPEARMAN SUMMARY ---")
    print(
        sp[
            [
                "exposure", "metric", "N", "rho",
                "p_raw", "p_fdr_6tests",
                "raw_P05", "FDR_6tests_05",
            ]
        ].to_string(index=False)
    )

    lines = [
        "=== STEP 15c1 NEW DIET ALPHA DIVERSITY SUMMARY ===",
        "METHOD=canonical-style quintile Kruskal-Wallis + continuous Spearman",
        "ALL_THREE_COMPLETE_REQUIRED=False",
        "MULTIPLE_TEST_FAMILY=3 exposures x 2 alpha metrics = 6 tests",
        "",
        "--- KRUSKAL-WALLIS ---",
        kw.to_string(index=False),
        "",
        "--- CONTINUOUS SPEARMAN ---",
        sp.to_string(index=False),
        "",
        "INTERPRETATION NOTES:",
        "- Unequal quintile N for EAT13 is expected because the discrete score has ties;",
        "  equal raw values were deliberately kept in the same quintile.",
        "- Raw Kruskal-Wallis P preserves comparability with the old canonical alpha analysis.",
        "- BH-FDR across 6 tests is reported as an additional QC.",
        "- Carbohydrate_pct is an exploratory macronutrient-composition exposure.",
        "",
        f"KW_TABLE={KW_OUT}",
        f"SPEARMAN_TABLE={SP_OUT}",
    ]
    SUMMARY_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
