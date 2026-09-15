#!/usr/bin/env python3
"""Step 15c5 — microbiome-wide association Model3 (+BMI) for frozen new diet exposures.

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
              + BMI

This reuses canonical mwas_adjusted_common.py encoding and adds the
prespecified mean_daily_energy_kcal term to the canonical continuous covariates.

FDR
---
BH-FDR separately across 379 species within each exposure, matching the
canonical old-score MWAS family definition.

Important
---------
- EAT13 and NOVA4 are primary extensions.
- Carbohydrate_pct is exploratory.
- No all-three-complete requirement.
- No hPDI-specific alcohol term.
- This step compares Model2 vs Model3. If Model3 complete-case selection causes
  a meaningful sample drop and large changes, a same-cohort Model2-vs-Model3
  QC should be run before attributing differences to BMI adjustment.

Outputs
-------
diet_microbiome_glucose_analysis/outputs/new_diet_extension/
    models/
        15c5_MWAS_EAT13_model3.csv
        15c5_MWAS_NOVA4_model3.csv
        15c5_MWAS_Carbohydrate_pct_model3.csv
        15c5_MWAS_all_model3.csv
    reports/
        15c5_MWAS_model3_summary.csv
        15c5_model2_vs_model3_comparison.csv
        15c5_MWAS_model3_pairwise_overlap.csv
        15c5_model3_covariate_parameters.csv
        15c5_MWAS_model3_summary.txt
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

MODEL2_ALL = MODEL_DIR / "15c4_MWAS_all_model2.csv"

ALL_OUT = MODEL_DIR / "15c5_MWAS_all_model3.csv"
SUMMARY_CSV = REPORT_DIR / "15c5_MWAS_model3_summary.csv"
COMPARE_CSV = REPORT_DIR / "15c5_model2_vs_model3_comparison.csv"
OVERLAP_CSV = REPORT_DIR / "15c5_MWAS_model3_pairwise_overlap.csv"
PARAM_CSV = REPORT_DIR / "15c5_model3_covariate_parameters.csv"
SUMMARY_TXT = REPORT_DIR / "15c5_MWAS_model3_summary.txt"

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

COMMON_FLAG = "amed_microbiome_model3_covariates_complete"
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


def compare_model2_model3(model2: pd.DataFrame, model3: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for exposure in EXPOSURES:
        m2 = model2.loc[model2["exposure"].eq(exposure)].copy()
        m3 = model3.loc[model3["exposure"].eq(exposure)].copy()

        m = m2[
            ["species", "N", "beta", "FDR", "significant_FDR05"]
        ].merge(
            m3[
                ["species", "N", "beta", "FDR", "significant_FDR05"]
            ],
            on="species",
            suffixes=("_model2", "_model3"),
            validate="one_to_one",
        )
        if len(m) != 379:
            raise RuntimeError(
                f"{exposure}: expected 379 species in Model2-vs-Model3 comparison"
            )

        m2_sig = truthy(m["significant_FDR05_model2"])
        m3_sig = truthy(m["significant_FDR05_model3"])
        both = m2_sig & m3_sig

        b2 = pd.to_numeric(m["beta_model2"], errors="coerce")
        b3 = pd.to_numeric(m["beta_model3"], errors="coerce")
        rho, rho_p = spearmanr(b2, b3)

        sign_flips = np.sign(b2.to_numpy()) != np.sign(b3.to_numpy())

        same_both = int(
            (
                np.sign(b2.loc[both].to_numpy())
                == np.sign(b3.loc[both].to_numpy())
            ).sum()
        )

        rows.append({
            "exposure": exposure,
            "model2_N": int(pd.to_numeric(m["N_model2"]).median()),
            "model3_N": int(pd.to_numeric(m["N_model3"]).median()),
            "model2_significant_FDR05": int(m2_sig.sum()),
            "model3_significant_FDR05": int(m3_sig.sum()),
            "retained_significant": int((m2_sig & m3_sig).sum()),
            "lost_in_model3": int((m2_sig & ~m3_sig).sum()),
            "gained_in_model3": int((~m2_sig & m3_sig).sum()),
            "same_direction_among_both_significant": same_both,
            "opposite_direction_among_both_significant": int(both.sum()) - same_both,
            "beta_spearman_rho_all379": float(rho),
            "beta_spearman_p_all379": float(rho_p),
            "beta_sign_flips_all379": int(sign_flips.sum()),
        })

    return pd.DataFrame(rows)


def main() -> int:
    for path, label in [
        (SCORE_FILE, "Step15c0 aligned score table"),
        (MICRO_FILE, "Step15c0 aligned microbiome table"),
        (COV_FILE, "covariate master"),
        (COMMON_PATH, "canonical MWAS adjusted helper"),
        (MODEL2_ALL, "Step15c4 Model2 results"),
    ]:
        require(path, label)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    common = load_module(COMMON_PATH, "step15c5_mwas_common")

    # Canonical Model3 already adds BMI through build_design(model=3).
    # We append only the prespecified total-energy covariate.
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
        "bmi",
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
            "Covariate master missing required Model3 columns: "
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

    print("=== STEP 15c5 NEW DIET -> MICROBIOME MWAS MODEL3 ===")
    print("MODEL=canonical Model3 + mean_daily_energy_kcal")
    print("BMI_INCLUDED=True")
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
                3,
            )

            # Hard model guards.
            if design_names.count(f"{ENERGY_COL}_z") != 1:
                raise RuntimeError(
                    f"{exposure}: expected exactly one {ENERGY_COL}_z term"
                )
            if design_names.count("bmi_z") != 1:
                raise RuntimeError(
                    f"{exposure}: expected exactly one bmi_z term in Model3"
                )
            if "alcohol_intake_g_day_z" in design_names:
                raise RuntimeError(
                    f"{exposure}: hPDI-specific alcohol entered Model3"
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
                "model": 3,
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
                MODEL_DIR / f"15c5_MWAS_{exposure}_model3.csv",
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

    finally:
        common.BASE_CONT = original_base_cont

    combined = pd.concat(all_results, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    params = pd.concat(param_rows, ignore_index=True)

    model2 = pd.read_csv(MODEL2_ALL, low_memory=False)
    comparison = compare_model2_model3(model2, combined)
    overlap = pairwise_overlap(combined)

    combined.to_csv(ALL_OUT, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    comparison.to_csv(COMPARE_CSV, index=False)
    overlap.to_csv(OVERLAP_CSV, index=False)
    params.to_csv(PARAM_CSV, index=False)

    print("\n--- MODEL3 SUMMARY ---")
    print(summary.to_string(index=False))

    print("\n--- MODEL2 vs MODEL3 ---")
    print(comparison.to_string(index=False))

    print("\n--- MODEL3 PAIRWISE FDR-SIGNIFICANT SPECIES OVERLAP ---")
    print(overlap.to_string(index=False))

    lines = [
        "=== STEP 15c5 NEW DIET -> MICROBIOME MWAS MODEL3 SUMMARY ===",
        "MODEL=canonical Model3 + mean_daily_energy_kcal",
        "BMI_INCLUDED=True",
        "ALCOHOL_INCLUDED=False",
        "FDR_FAMILY=379 species within each exposure",
        "ALL_THREE_COMPLETE_REQUIRED=False",
        "",
        "--- MODEL3 SUMMARY ---",
        summary.to_string(index=False),
        "",
        "--- MODEL2 vs MODEL3 ---",
        comparison.to_string(index=False),
        "",
        "--- MODEL3 PAIRWISE OVERLAP ---",
        overlap.to_string(index=False),
        "",
        "INTERPRETATION NOTES:",
        "- Model3 is the BMI sensitivity model, not a replacement for Model2 primary.",
        "- If Model2->Model3 changes materially while N also drops, run an exact",
        "  same-cohort Model2-vs-Model3 decomposition before attributing changes to BMI.",
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
