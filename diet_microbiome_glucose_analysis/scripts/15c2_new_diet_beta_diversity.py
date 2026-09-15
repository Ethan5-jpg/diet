#!/usr/bin/env python3
"""Step 15c2 — beta diversity for frozen new diet exposures.

This intentionally reuses the canonical old-score beta-diversity implementation:
- Bray-Curtis on half-minimum-imputed abundance (NOT CLR-Z);
- PCoA from the Bray-Curtis squared-distance matrix;
- one-factor PERMANOVA across exposure quintiles;
- 999 permutations, seed 42;
- pseudo-F and R² calculated exactly as in canonical 02_beta_diversity.py.

New-exposure rule
-----------------
Each exposure uses its own maximum valid sample. A union distance matrix is
built once, then each PERMANOVA is run on the exposure-specific valid subset.

Exposures
---------
EAT13             primary extension
NOVA4             primary extension; total-energy denominator, >=80% mapping coverage
Carbohydrate_pct  exploratory macronutrient-composition exposure

No adjusted association model is introduced here; this mirrors the old
descriptive beta-diversity step.

Outputs
-------
diet_microbiome_glucose_analysis/outputs/new_diet_extension/
    data/15c2_pcoa_coordinates.csv
    reports/15c2_beta_permanova.csv
    reports/15c2_beta_diversity_summary.txt
    figures/15c2_pcoa_<exposure>_Q1_Q5.png
    cache/15c2_bray_curtis_squared_float32.dat
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import gc
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
OLD_BETA_SCRIPT = (
    ROOT / "diet_microbiome_analysis" / "scripts" / "02_beta_diversity.py"
)

BASE = (
    ROOT / "diet_microbiome_glucose_analysis" / "outputs"
    / "new_diet_extension"
)
DATA_DIR = BASE / "data"
REPORT_DIR = BASE / "reports"
FIGURE_DIR = BASE / "figures"
CACHE_DIR = BASE / "cache"

SCORE_FILE = DATA_DIR / "15c0_new_diet_microbiome_scores_aligned.csv"
ABUNDANCE_FILE = ROOT / "gut_microbiome_deal" / "data" / "05_species_half_min_imputed.csv"

D2_FILE = CACHE_DIR / "15c2_bray_curtis_squared_float32.dat"
CACHE_META_FILE = CACHE_DIR / "15c2_bray_curtis_cache_meta.json"
PCOA_FILE = DATA_DIR / "15c2_pcoa_coordinates.csv"
PERMANOVA_FILE = REPORT_DIR / "15c2_beta_permanova.csv"
SUMMARY_FILE = REPORT_DIR / "15c2_beta_diversity_summary.txt"

META_COLS = ["participant_id", "cohort", "research_stage", "array_index"]

EXPOSURES = {
    "EAT13": {
        "z": "EAT13_z",
        "q": "EAT13_quintile",
        "role": "primary_extension",
    },
    "NOVA4": {
        "z": "NOVA4_z",
        "q": "NOVA4_quintile",
        "role": "primary_extension",
    },
    "Carbohydrate_pct": {
        "z": "Carbohydrate_pct_z",
        "q": "Carbohydrate_pct_quintile",
        "role": "exploratory_extension",
    },
}


def require(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


def norm_id(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    for path, label in [
        (OLD_BETA_SCRIPT, "canonical beta-diversity script"),
        (SCORE_FILE, "Step15c0 aligned new-diet scores"),
        (ABUNDANCE_FILE, "half-minimum-imputed abundance"),
    ]:
        require(path, label)

    for d in [DATA_DIR, REPORT_DIR, FIGURE_DIR, CACHE_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    beta = load_module(OLD_BETA_SCRIPT, "step15c2_canonical_beta")

    # Redirect canonical helper outputs to isolated new-exposure directories.
    beta.ABUNDANCE_FILE = ABUNDANCE_FILE
    beta.D2_FILE = D2_FILE
    beta.CACHE_META_FILE = CACHE_META_FILE
    beta.PCOA_FILE = PCOA_FILE
    beta.FIGURE_DIR = FIGURE_DIR
    beta.PERMUTATIONS = 999
    beta.RANDOM_SEED = 42
    beta.SCORES = {k: v["q"] for k, v in EXPOSURES.items()}

    scores = pd.read_csv(SCORE_FILE, low_memory=False)
    abundance = pd.read_csv(ABUNDANCE_FILE, low_memory=False)

    for label, d in [("scores", scores), ("abundance", abundance)]:
        if "participant_id" not in d.columns:
            raise RuntimeError(f"{label} missing participant_id")
        d["participant_id"] = norm_id(d["participant_id"])
        if d["participant_id"].duplicated().any():
            raise RuntimeError(f"{label} has duplicate participant_id")

    for exposure, cfg in EXPOSURES.items():
        missing = [c for c in [cfg["z"], cfg["q"]] if c not in scores.columns]
        if missing:
            raise RuntimeError(
                f"{exposure}: aligned score file missing {missing}"
            )

    # Step15c0 score file is already in canonical microbiome row order.
    ids = scores["participant_id"].tolist()
    abundance_ids = set(abundance["participant_id"])
    missing_abundance = [pid for pid in ids if pid not in abundance_ids]
    if missing_abundance:
        raise RuntimeError(
            f"{len(missing_abundance)} aligned participants missing from abundance table"
        )

    aligned_abundance = (
        pd.DataFrame({"participant_id": ids})
        .merge(abundance, on="participant_id", how="left", validate="one_to_one")
    )

    species = [c for c in aligned_abundance.columns if c not in META_COLS]
    if len(species) != 379:
        raise RuntimeError(f"Expected 379 species, found {len(species)}")

    X = (
        aligned_abundance[species]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64)
    )
    if not np.isfinite(X).all():
        raise RuntimeError("Abundance contains NaN/Inf after alignment")
    if (X < 0).any():
        raise RuntimeError("Negative abundance encountered")

    print("=== STEP 15c2 NEW DIET BETA DIVERSITY ===")
    print(f"UNION_PARTICIPANTS={len(ids)}")
    print(f"SPECIES={len(species)}")
    print("METHOD=canonical Bray-Curtis + PCoA + PERMANOVA")
    print("PERMUTATIONS=999")
    print("ALL_THREE_COMPLETE_REQUIRED=False")
    print(f"NUMBA_ACCELERATION={beta.HAVE_NUMBA}")

    row_sums = X.sum(axis=1)
    print("\n--- ABUNDANCE ROW-SUM QC ---")
    print(
        pd.Series(row_sums)
        .describe(percentiles=[.01, .50, .99])
        .to_string()
    )

    # One union matrix; exact exposure-specific submatrices are used below.
    d2 = beta.build_bray_curtis_squared(X, ids, species)

    # Union PCoA is only a visualization coordinate system. PERMANOVA itself
    # is always computed on exposure-specific exact subsets.
    pcoa, eigenvalues, trace_prop, total_inertia = beta.compute_pcoa(d2, ids)

    # Q1-vs-Q5 plots, following the canonical visual design but with isolated filenames.
    plot_df = pcoa.merge(
        scores[["participant_id"] + [cfg["q"] for cfg in EXPOSURES.values()]],
        on="participant_id",
        how="left",
        validate="one_to_one",
    )
    for exposure, cfg in EXPOSURES.items():
        qplot = beta.normalize_quintile(plot_df[cfg["q"]])
        mask = qplot.isin([1, 5])
        one = plot_df.loc[mask].copy()
        q_one = qplot.loc[mask]
        fig, ax = beta.plt.subplots(figsize=(7, 6))
        for level in [1, 5]:
            m = q_one.eq(level)
            ax.scatter(
                one.loc[m, "PC1"],
                one.loc[m, "PC2"],
                s=10,
                alpha=0.35,
                label=f"{exposure} Q{level}",
            )
        ax.set_xlabel(f"PC1 ({100*trace_prop[0]:.2f}% trace inertia)")
        ax.set_ylabel(f"PC2 ({100*trace_prop[1]:.2f}% trace inertia)")
        ax.set_title(f"Bray-Curtis PCoA: {exposure} Q1 vs Q5")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"15c2_pcoa_{exposure}_Q1_Q5.png", dpi=180)
        beta.plt.close(fig)

    results = []

    print("\n=== EXPOSURE-SPECIFIC PERMANOVA ===")
    for exposure, cfg in EXPOSURES.items():
        z = pd.to_numeric(scores[cfg["z"]], errors="coerce")
        q = beta.normalize_quintile(scores[cfg["q"]])

        valid = (
            z.notna()
            & np.isfinite(z.to_numpy(float))
            & q.notna()
            & q.isin([1, 2, 3, 4, 5])
        )
        idx = np.flatnonzero(valid.to_numpy())
        q_valid = q.loc[valid].reset_index(drop=True)

        levels = sorted(q_valid.astype(int).unique().tolist())
        if levels != [1, 2, 3, 4, 5]:
            raise RuntimeError(
                f"{exposure}: expected quintiles 1..5, found {levels}"
            )

        print("\n" + "-" * 96)
        print(f"{exposure} | {cfg['role']} | N={len(idx)}")
        print("-" * 96)

        # Materialize one subset at a time (~200-250 MiB at current N).
        d2_sub = np.asarray(
            d2[np.ix_(idx, idx)],
            dtype=np.float32,
        )

        n_sub = len(idx)
        total_ss_sub = float(
            d2_sub.sum(dtype=np.float64) / (2.0 * n_sub)
        )

        result = beta.run_permanova(
            d2_sub,
            exposure,
            q_valid,
            total_ss_sub,
        )
        result["analysis_role"] = cfg["role"]
        result["union_distance_N"] = len(ids)
        result["subset_total_SS"] = total_ss_sub
        results.append(result)

        print(
            f"pseudo-F={result['pseudo_F']:.6f} | "
            f"R2={result['R2_percent']:.6f}% | "
            f"P={result['p_permutation']:.6g}"
        )

        del d2_sub
        gc.collect()

    result_df = pd.DataFrame(results)

    # Three PERMANOVA tests; raw permutation P retains comparability with the
    # canonical old-score analysis. BH-FDR is an added extension QC.
    p = result_df["p_permutation"].to_numpy(float)
    order = np.argsort(p)
    ranked = p[order]
    qvals = ranked * len(p) / np.arange(1, len(p) + 1)
    qvals = np.minimum.accumulate(qvals[::-1])[::-1]
    fdr = np.empty_like(qvals)
    fdr[order] = np.clip(qvals, 0, 1)
    result_df["p_fdr_3tests"] = fdr
    result_df["raw_P05"] = result_df["p_permutation"] < .05
    result_df["FDR_3tests_05"] = result_df["p_fdr_3tests"] < .05

    result_df.to_csv(PERMANOVA_FILE, index=False)

    print("\n--- PERMANOVA SUMMARY ---")
    print(
        result_df[
            [
                "score",
                "analysis_role",
                "N",
                "q1_N",
                "q2_N",
                "q3_N",
                "q4_N",
                "q5_N",
                "pseudo_F",
                "R2_percent",
                "p_permutation",
                "p_fdr_3tests",
                "FDR_3tests_05",
            ]
        ].to_string(index=False)
    )

    lines = [
        "=== STEP 15c2 NEW DIET BETA DIVERSITY SUMMARY ===",
        "METHOD=canonical Bray-Curtis + union PCoA + exposure-specific PERMANOVA",
        "PERMUTATIONS=999",
        "ALL_THREE_COMPLETE_REQUIRED=False",
        f"UNION_PARTICIPANTS={len(ids)}",
        f"SPECIES={len(species)}",
        f"PC1_TRACE_PERCENT={100*trace_prop[0]:.6f}",
        f"PC2_TRACE_PERCENT={100*trace_prop[1]:.6f}",
        "",
        "--- PERMANOVA ---",
        result_df.to_string(index=False),
        "",
        "INTERPRETATION:",
        "- PERMANOVA R2 is the primary beta-diversity effect-size quantity.",
        "- Small P values with tiny R2 should be described as statistically",
        "  detectable but small community-composition differences.",
        "- Carbohydrate_pct remains exploratory and should not be described",
        "  as a healthier/worse dietary-quality score.",
        "",
        f"PCOA={PCOA_FILE}",
        f"PERMANOVA={PERMANOVA_FILE}",
        f"DISTANCE_CACHE={D2_FILE}",
    ]
    SUMMARY_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSUMMARY={SUMMARY_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
