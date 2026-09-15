#!/usr/bin/env python3
"""Step 15c4 — microbiome-wide association Model2 for frozen new diet exposures.

Model specification
-------------------
species CLR-Z ~ exposure Z
              + age
              + sex
              + education
              + smoking
              + sleep duration
              + physical activity
              + vitamin use
              + hormone use
              + mean daily total energy

This deliberately reuses the canonical mwas_adjusted_common.py encoding:
- continuous covariates are Z-standardized within the selected analysis sample;
- vitamin/hormone are binary 0/1;
- sex reference = female;
- education reference = high;
- smoking reference = never;
- BH-FDR separately across 379 species within each exposure.

The only extension to canonical Model2 is the prespecified
mean_daily_energy_kcal covariate.

Important
---------
- EAT13 and NOVA4 are primary extensions.
- Carbohydrate_pct remains exploratory.
- No all-three-complete requirement.
- hPDI-specific alcohol adjustment is NOT applied to these new exposures.
- No BMI in Model2; BMI is added in Step15c5 Model3.

Outputs
-------
diet_microbiome_glucose_analysis/outputs/new_diet_extension/
    models/
        15c4_MWAS_EAT13_model2.csv
        15c4_MWAS_NOVA4_model2.csv
        15c4_MWAS_Carbohydrate_pct_model2.csv
        15c4_MWAS_all_model2.csv
    reports/
        15c4_MWAS_model2_summary.csv
        15c4_model0_vs_model2_comparison.csv
        15c4_MWAS_model2_pairwise_overlap.csv
        15c4_model2_covariate_parameters.csv
        15c4_MWAS_model2_summary.txt
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


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
COMMON_PATH = (
    ROOT / "diet_microbiome_analysis" / "scripts" / "mwas_adjusted_common.py"
)

MODEL0_ALL = MODEL_DIR / "15c3_MWAS_all_model0.csv"

ALL_OUT = MODEL_DIR / "15c4_MWAS_all_model2.csv"
SUMMARY_CSV = REPORT_DIR / "15c4_MWAS_model2_summary.csv"
COMPARE_CSV = REPORT_DIR / "15c4_model0_vs_model2_comparison.csv"
OVERLAP_CSV = REPORT_DIR / "15c4_MWAS_model2_pairwise_overlap.csv"
PARAM_CSV = REPORT_DIR / "15c4_model2_covariate_parameters.csv"
SUMMARY_TXT = REPORT_DIR / "15c4_MWAS_model2_summary.txt"

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

COMMON_FLAG = "amed_microbiome_model2_covariates_complete"
ENERGY_COL = "mean_daily_energy_kcal"


def require(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


def normalize_id(s: pd.Series) -> pd.Series:
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


def truthy(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).eq(1)
    return (
        s.astype(str).str.strip().str.lower()
        .isin({"true", "1", "1.0", "yes", "y", "t"})
    )


def pairwise_overlap(all_results: pd.DataFrame) -> pd.DataFrame:
    names = list(EXPOSURES)
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

    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            A, B = sig_sets[a], sig_sets[b]
            shared = A & B
            union = A | B
            same = 0
            opposite = 0
            if shared:
                sub = beta_wide.loc[list(shared), [a, b]]
                same = int(
                    (
                        np.sign(sub[a].to_numpy(float))
                        == np.sign(sub[b].to_numpy(float))
                    ).sum()
                )
                opposite = len(shared) - same

            rows.append({
                "exposure_1": a,
                "exposure_2": b,
                "sig_1": len(A),
                "sig_2": len(B),
                "shared_significant_species": len(shared),
                "union_significant_species": len(union),
                "jaccard": len(shared) / len(union) if union else np.nan,
                "same_direction_among_shared": same,
                "opposite_direction_among_shared": opposite,
            })
    return pd.DataFrame(rows)


def compare_model0_model2(model0: pd.DataFrame, model2: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for exposure in EXPOSURES:
        old = model0.loc[model0["exposure"].eq(exposure)].copy()
        new = model2.loc[model2["exposure"].eq(exposure)].copy()
        m = old[
            ["species", "beta", "FDR", "significant_FDR05"]
        ].merge(
            new[
                ["species", "beta", "FDR", "significant_FDR05"]
            ],
            on="species",
            suffixes=("_model0", "_model2"),
            validate="one_to_one",
        )
        if len(m) != 379:
            raise RuntimeError(
                f"{exposure}: expected 379 species in Model0-vs-Model2 comparison"
            )

        old_sig = truthy(m["significant_FDR05_model0"])
        new_sig = truthy(m["significant_FDR05_model2"])
        both = old_sig & new_sig

        rho, rho_p = spearmanr(
            pd.to_numeric(m["beta_model0"], errors="coerce"),
            pd.to_numeric(m["beta_model2"], errors="coerce"),
        )
        sign_flip_all = (
            np.sign(pd.to_numeric(m["beta_model0"], errors="coerce").to_numpy())
            != np.sign(pd.to_numeric(m["beta_model2"], errors="coerce").to_numpy())
        )

        same_both = int(
            (
                np.sign(
                    pd.to_numeric(
                        m.loc[both, "beta_model0"], errors="coerce"
                    ).to_numpy()
                )
                == np.sign(
                    pd.to_numeric(
                        m.loc[both, "beta_model2"], errors="coerce"
                    ).to_numpy()
                )
            ).sum()
        )

        rows.append({
            "exposure": exposure,
            "model0_significant_FDR05": int(old_sig.sum()),
            "model2_significant_FDR05": int(new_sig.sum()),
            "retained_significant": int((old_sig & new_sig).sum()),
            "lost_after_adjustment": int((old_sig & ~new_sig).sum()),
            "gained_after_adjustment": int((~old_sig & new_sig).sum()),
            "same_direction_among_both_significant": same_both,
            "opposite_direction_among_both_significant": int(both.sum()) - same_both,
            "beta_spearman_rho_all379": float(rho),
            "beta_spearman_p_all379": float(rho_p),
            "beta_sign_flips_all379": int(sign_flip_all.sum()),
        })
    return pd.DataFrame(rows)


def main() -> int:
    for path, label in [
        (SCORE_FILE, "Step15c0 aligned score table"),
        (MICRO_FILE, "Step15c0 aligned microbiome table"),
        (COV_FILE, "covariate master"),
        (COMMON_PATH, "canonical MWAS adjusted helper"),
        (MODEL0_ALL, "Step15c3 Model0 results"),
    ]:
        require(path, label)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    common = load_module(COMMON_PATH, "step15c4_mwas_common")

    # Freeze exact canonical encoding + one prespecified energy covariate.
    original_base_cont = list(common.BASE_CONT)
    if ENERGY_COL in original_base_cont:
        raise RuntimeError(
            f"{ENERGY_COL} unexpectedly already exists in canonical BASE_CONT"
        )
    common.BASE_CONT = original_base_cont + [ENERGY_COL]

    scores = pd.read_csv(SCORE_FILE, low_memory=False)
    micro = pd.read_csv(MICRO_FILE, low_memory=False)

    needed_cov = {
        "participant_id",
        "age_years",
        "sleep_duration_hours_day",
        "physical_activity_met_h_week",
        "vitamin_use",
        "hormone_use",
        "sex",
        "education_level",
        "smoking_status",
        COMMON_FLAG,
    }
    cov = pd.read_csv(
        COV_FILE,
        usecols=lambda c: c in needed_cov,
        low_memory=False,
    )

    for label, d in [("scores", scores), ("microbiome", micro), ("covariates", cov)]:
        if "participant_id" not in d.columns:
            raise RuntimeError(f"{label} missing participant_id")
        d["participant_id"] = normalize_id(d["participant_id"])
        if d["participant_id"].duplicated().any():
            raise RuntimeError(f"{label} has duplicate participant_id")

    missing_cov = sorted(needed_cov - set(cov.columns))
    if missing_cov:
        raise RuntimeError(
            "Covariate master missing required Model2 columns: "
            + ", ".join(missing_cov)
        )

    if scores["participant_id"].tolist() != micro["participant_id"].tolist():
        raise RuntimeError("Score and microbiome participant order differs")

    species = [c for c in micro.columns if c not in META_COLS]
    if len(species) != 379:
        raise RuntimeError(f"Expected 379 species, found {len(species)}")

    Y = (
        micro[species]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )
    if not np.isfinite(Y).all():
        raise RuntimeError("Species CLR-Z matrix contains NaN/Inf")

    merged = scores.merge(
        cov, on="participant_id", how="left", validate="one_to_one"
    )

    base_complete = common.to_bool(merged[COMMON_FLAG])
    energy = pd.to_numeric(merged[ENERGY_COL], errors="coerce")
    energy_ok = energy.notna() & np.isfinite(energy.to_numpy(float))

    print("=== STEP 15c4 NEW DIET -> MICROBIOME MWAS MODEL2 ===")
    print("MODEL=canonical Model2 + mean_daily_energy_kcal")
    print("BMI_INCLUDED=False")
    print("ALCOHOL_INCLUDED=False")
    print("FDR_FAMILY=379 species separately within each exposure")
    print("ALL_THREE_COMPLETE_REQUIRED=False")
    print(f"COMMON_COMPLETE_FLAG={COMMON_FLAG}")

    all_results = []
    summary_rows = []
    param_rows = []

    try:
        for exposure, cfg in EXPOSURES.items():
            score = pd.to_numeric(merged[cfg["col"]], errors="coerce")
            exposure_ok = score.notna() & np.isfinite(score.to_numpy(float))

            valid = base_complete & energy_ok & exposure_ok
            selected = merged.loc[valid].copy()
            idx = np.flatnonzero(valid.to_numpy())
            Y_use = Y[idx]

            if len(selected) < 3:
                raise RuntimeError(f"{exposure}: <3 complete cases")

            X, design_names, ptab, cond = common.build_design(
                selected,
                exposure,
                cfg["col"],
                2,
            )

            # Hard guard: exactly one total-energy term must be present.
            energy_terms = [
                n for n in design_names
                if n == f"{ENERGY_COL}_z"
            ]
            if len(energy_terms) != 1:
                raise RuntimeError(
                    f"{exposure}: expected exactly one {ENERGY_COL}_z term; "
                    f"design={design_names}"
                )
            if "bmi_z" in design_names:
                raise RuntimeError(f"{exposure}: BMI entered Model2 unexpectedly")
            if "alcohol_intake_g_day_z" in design_names:
                raise RuntimeError(
                    f"{exposure}: hPDI-specific alcohol entered new-exposure Model2"
                )

            ptab["analysis_role"] = cfg["role"]
            ptab["N"] = len(selected)
            ptab["design_condition_number"] = cond
            param_rows.append(ptab)

            beta, se, t_stat, p, r2, lo, hi, df_resid = common.fit_all_species(
                X, Y_use
            )
            fdr = common.bh_fdr(p)

            result = pd.DataFrame({
                "exposure": exposure,
                "analysis_role": cfg["role"],
                "polarity": cfg["polarity"],
                "species": species,
                "N": len(selected),
                "model": 2,
                "beta": beta,
                "SE": se,
                "t": t_stat,
                "p": p,
                "FDR": fdr,
                "R2_full_model": r2,
                "CI_lower": lo,
                "CI_upper": hi,
                "df_resid": df_resid,
                "n_parameters": X.shape[1],
                "design_condition_number": cond,
            })
            result["significant_FDR05"] = result["FDR"] < .05
            result["direction"] = np.where(
                result["beta"] > 0, "positive", "negative"
            )

            sig = result.loc[result["significant_FDR05"]]
            n_pos = int((sig["beta"] > 0).sum())
            n_neg = int((sig["beta"] < 0).sum())

            result.to_csv(
                MODEL_DIR / f"15c4_MWAS_{exposure}_model2.csv",
                index=False,
            )
            all_results.append(result)

            summary_rows.append({
                "exposure": exposure,
                "analysis_role": cfg["role"],
                "N": len(selected),
                "species_tested": len(species),
                "n_parameters": X.shape[1],
                "design_condition_number": cond,
                "significant_FDR05": len(sig),
                "positive_FDR05": n_pos,
                "negative_FDR05": n_neg,
                "min_p": float(result["p"].min()),
                "min_FDR": float(result["FDR"].min()),
                "max_positive_beta": float(result["beta"].max()),
                "min_negative_beta": float(result["beta"].min()),
            })

            print("\n" + "=" * 96)
            print(f"{exposure} | {cfg['role']} | N={len(selected)}")
            print("=" * 96)
            print(
                f"parameters={X.shape[1]} | condition={cond:.4f} | "
                f"FDR<0.05={len(sig)} | positive={n_pos} | negative={n_neg}"
            )

            print("\nTOP POSITIVE")
            print(
                result.nlargest(8, "beta")[
                    ["species", "beta", "p", "FDR"]
                ].to_string(index=False)
            )
            print("\nTOP NEGATIVE")
            print(
                result.nsmallest(8, "beta")[
                    ["species", "beta", "p", "FDR"]
                ].to_string(index=False)
            )
    finally:
        common.BASE_CONT = original_base_cont

    combined = pd.concat(all_results, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    params = pd.concat(param_rows, ignore_index=True)

    model0 = pd.read_csv(MODEL0_ALL, low_memory=False)
    comparison = compare_model0_model2(model0, combined)
    overlap = pairwise_overlap(combined)

    combined.to_csv(ALL_OUT, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    comparison.to_csv(COMPARE_CSV, index=False)
    overlap.to_csv(OVERLAP_CSV, index=False)
    params.to_csv(PARAM_CSV, index=False)

    print("\n--- MODEL2 SUMMARY ---")
    print(summary.to_string(index=False))

    print("\n--- MODEL0 vs MODEL2 ---")
    print(comparison.to_string(index=False))

    print("\n--- MODEL2 PAIRWISE FDR-SIGNIFICANT SPECIES OVERLAP ---")
    print(overlap.to_string(index=False))

    lines = [
        "=== STEP 15c4 NEW DIET -> MICROBIOME MWAS MODEL2 SUMMARY ===",
        "MODEL=canonical Model2 + mean_daily_energy_kcal",
        "BMI_INCLUDED=False",
        "ALCOHOL_INCLUDED=False",
        "FDR_FAMILY=379 species within each exposure",
        "ALL_THREE_COMPLETE_REQUIRED=False",
        "",
        "--- MODEL2 SUMMARY ---",
        summary.to_string(index=False),
        "",
        "--- MODEL0 vs MODEL2 ---",
        comparison.to_string(index=False),
        "",
        "--- MODEL2 PAIRWISE OVERLAP ---",
        overlap.to_string(index=False),
        "",
        "INTERPRETATION NOTES:",
        "- A drop in FDR-significant counts from Model0 to Model2 can reflect both",
        "  covariate adjustment and complete-case sample selection.",
        "- Do not label that change as a pure confounder-adjustment effect without",
        "  a same-cohort Model0-vs-Model2 QC.",
        "- EAT13 and NOVA4 are primary extensions; Carbohydrate_pct is exploratory.",
        "",
        f"ALL_RESULTS={ALL_OUT}",
        f"SUMMARY_CSV={SUMMARY_CSV}",
        f"COMPARISON_CSV={COMPARE_CSV}",
        f"OVERLAP_CSV={OVERLAP_CSV}",
        f"PARAMETERS={PARAM_CSV}",
    ]
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
