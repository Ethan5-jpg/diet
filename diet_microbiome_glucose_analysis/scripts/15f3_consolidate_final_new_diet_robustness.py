#!/usr/bin/env python3
"""Step 15f3 — final robustness consolidation for new-diet mediation-style results.

Combines:
- primary cohort
- strict known-nondiabetes/A10-negative cohort
- primary + protocol-window antibiotic/PPI covariate sensitivity

Also performs a medication-missingness QC to explain whether unchanged
primary-vs-protocol model N is expected.

No models are refit.

Outputs
-------
diet_microbiome_glucose_analysis/outputs/new_diet_extension/mediation/
    15f3_threeway_path_robustness.csv
    15f3_threeway_context_robustness.csv
    15f3_threeway_species_robustness.csv
    15f3_protocol_medication_missingness_qc.csv
    15f3_final_robustness_summary.txt
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
BASE = (
    ROOT / "diet_microbiome_glucose_analysis" / "outputs"
    / "new_diet_extension"
)
MED = BASE / "mediation"

PRIMARY = MED / "15f_primary_mediation_paths.csv"
STRICT = MED / "15f_strict_mediation_paths.csv"
PROTOCOL = (
    MED / "protocol_abxppi" / "15f_primary_mediation_paths.csv"
)
PROTOCOL_COHORT = (
    MED / "protocol_abxppi" / "15f2_protocol_abxppi_cohort.csv"
)
MICRO = ROOT / "gut_microbiome_deal" / "data" / "08_species_clr_zscore.csv"

PATH_OUT = MED / "15f3_threeway_path_robustness.csv"
CONTEXT_OUT = MED / "15f3_threeway_context_robustness.csv"
SPECIES_OUT = MED / "15f3_threeway_species_robustness.csv"
MEDQC_OUT = MED / "15f3_protocol_medication_missingness_qc.csv"
SUMMARY_OUT = MED / "15f3_final_robustness_summary.txt"

KEY = ["diet_score", "cgm_outcome", "species"]

EXPOSURE_COL = {
    "NOVA4": "NOVA4_z",
    "Carbohydrate_pct": "Carbohydrate_pct_z",
}
OUTCOME_COL = {
    "mean_glucose": "cgm_mean_z",
    "glucose_cv": "cgm_cv_z",
    "time_above_140": "cgm_above_140_z",
}

BASE_COVARIATES = [
    "age_years",
    "sex",
    "education_level",
    "smoking_status",
    "sleep_duration_hours_day",
    "physical_activity_met_h_week",
    "vitamin_use",
    "hormone_use",
    "bmi",
    "cgm_device_type",
    "mean_daily_energy_kcal",
]

MED_COLS = [
    "antibiotic_use_protocol_window",
    "ppi_use_protocol_window",
]


def require(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


def truthy(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).eq(1)
    return (
        s.astype(str).str.strip().str.lower()
        .isin({"true", "1", "1.0", "yes", "y", "t"})
    )


def norm_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def load_paths(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {
        *KEY,
        "analysis_role",
        "N",
        "indirect_effect",
        "mediation_FDR05_primary",
        "candidate_mediator_consistent",
        "FDR_BH_within_exposure_outcome",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{label} missing columns: {missing}")
    if len(df) != 45:
        raise RuntimeError(f"{label}: expected 45 paths, found {len(df)}")
    if df.duplicated(KEY).any():
        raise RuntimeError(f"{label}: duplicate path keys")
    return df


def model_complete_mask(df: pd.DataFrame, exposure: str, outcome: str, add_meds: bool):
    cols = [
        EXPOSURE_COL[exposure],
        OUTCOME_COL[outcome],
        *BASE_COVARIATES,
    ]
    if add_meds:
        cols += MED_COLS

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Protocol cohort missing columns: {missing}")

    mask = truthy(df["primary_cgm_analysis_eligible"])

    for c in cols:
        if c in {"sex", "education_level", "smoking_status", "cgm_device_type"}:
            mask &= df[c].notna()
        else:
            x = pd.to_numeric(df[c], errors="coerce")
            mask &= x.notna() & np.isfinite(x.to_numpy(float))
    return mask


def main() -> int:
    for p, label in [
        (PRIMARY, "primary paths"),
        (STRICT, "strict paths"),
        (PROTOCOL, "protocol ABX/PPI paths"),
        (PROTOCOL_COHORT, "protocol sensitivity cohort"),
        (MICRO, "microbiome CLR-Z"),
    ]:
        require(p, label)

    p = load_paths(PRIMARY, "primary")
    s = load_paths(STRICT, "strict")
    a = load_paths(PROTOCOL, "protocol")

    p_keys = set(map(tuple, p[KEY].astype(str).to_numpy()))
    s_keys = set(map(tuple, s[KEY].astype(str).to_numpy()))
    a_keys = set(map(tuple, a[KEY].astype(str).to_numpy()))
    if not (p_keys == s_keys == a_keys):
        raise RuntimeError(
            "Three-way exact path-set mismatch: "
            f"primary={len(p_keys)}, strict={len(s_keys)}, protocol={len(a_keys)}"
        )

    keep = KEY + [
        "analysis_role",
        "N",
        "indirect_effect",
        "FDR_BH_within_exposure_outcome",
        "mediation_FDR05_primary",
        "candidate_mediator_consistent",
    ]

    m = (
        p[keep]
        .merge(
            s[keep],
            on=KEY,
            suffixes=("_primary", "_strict"),
            validate="one_to_one",
        )
    )

    a2 = a[keep].rename(
        columns={
            c: f"{c}_protocol"
            for c in keep if c not in KEY
        }
    )
    m = m.merge(a2, on=KEY, validate="one_to_one")

    p_sig = truthy(m["mediation_FDR05_primary_primary"])
    s_sig = truthy(m["mediation_FDR05_primary_strict"])
    a_sig = truthy(m["mediation_FDR05_primary_protocol"])

    p_con = truthy(m["candidate_mediator_consistent_primary"])
    s_con = truthy(m["candidate_mediator_consistent_strict"])
    a_con = truthy(m["candidate_mediator_consistent_protocol"])

    p_ind = pd.to_numeric(m["indirect_effect_primary"], errors="coerce")
    s_ind = pd.to_numeric(m["indirect_effect_strict"], errors="coerce")
    a_ind = pd.to_numeric(m["indirect_effect_protocol"], errors="coerce")

    m["primary_significant"] = p_sig
    m["strict_significant"] = s_sig
    m["protocol_significant"] = a_sig
    m["primary_consistent"] = p_con
    m["strict_consistent"] = s_con
    m["protocol_consistent"] = a_con

    m["primary_consistent_retained_strict"] = p_con & s_con
    m["primary_consistent_retained_protocol"] = p_con & a_con
    m["primary_consistent_retained_both"] = p_con & s_con & a_con

    m["indirect_same_sign_primary_strict"] = (
        np.sign(p_ind) == np.sign(s_ind)
    )
    m["indirect_same_sign_primary_protocol"] = (
        np.sign(p_ind) == np.sign(a_ind)
    )
    m["indirect_same_sign_all_three"] = (
        m["indirect_same_sign_primary_strict"]
        & m["indirect_same_sign_primary_protocol"]
    )

    m.to_csv(PATH_OUT, index=False)

    # Context summary.
    context_rows = []
    for (diet, outcome), g in m.groupby(
        ["diet_score", "cgm_outcome"], sort=True
    ):
        rho_s = (
            float(spearmanr(
                pd.to_numeric(g["indirect_effect_primary"]),
                pd.to_numeric(g["indirect_effect_strict"]),
            ).statistic)
            if len(g) >= 3 else np.nan
        )
        rho_a = (
            float(spearmanr(
                pd.to_numeric(g["indirect_effect_primary"]),
                pd.to_numeric(g["indirect_effect_protocol"]),
            ).statistic)
            if len(g) >= 3 else np.nan
        )
        context_rows.append({
            "diet_score": diet,
            "cgm_outcome": outcome,
            "paths_tested": len(g),
            "primary_consistent": int(g["primary_consistent"].sum()),
            "strict_consistent": int(g["strict_consistent"].sum()),
            "protocol_consistent": int(g["protocol_consistent"].sum()),
            "primary_consistent_retained_strict": int(
                g["primary_consistent_retained_strict"].sum()
            ),
            "primary_consistent_retained_protocol": int(
                g["primary_consistent_retained_protocol"].sum()
            ),
            "primary_consistent_retained_both": int(
                g["primary_consistent_retained_both"].sum()
            ),
            "indirect_rho_primary_strict": rho_s,
            "indirect_rho_primary_protocol": rho_a,
            "sign_flips_primary_strict": int(
                (~g["indirect_same_sign_primary_strict"]).sum()
            ),
            "sign_flips_primary_protocol": int(
                (~g["indirect_same_sign_primary_protocol"]).sum()
            ),
        })
    context = pd.DataFrame(context_rows)
    context.to_csv(CONTEXT_OUT, index=False)

    species = (
        m.groupby("species", as_index=False)
        .agg(
            paths_tested=("species", "size"),
            primary_consistent_paths=("primary_consistent", "sum"),
            strict_consistent_paths=("strict_consistent", "sum"),
            protocol_consistent_paths=("protocol_consistent", "sum"),
            retained_both_sensitivities_paths=(
                "primary_consistent_retained_both", "sum"
            ),
            n_diet_exposures=("diet_score", "nunique"),
            n_cgm_outcomes=("cgm_outcome", "nunique"),
            diet_exposures=(
                "diet_score",
                lambda x: ";".join(sorted(set(map(str, x)))),
            ),
            cgm_outcomes=(
                "cgm_outcome",
                lambda x: ";".join(sorted(set(map(str, x)))),
            ),
        )
        .sort_values(
            [
                "retained_both_sensitivities_paths",
                "primary_consistent_paths",
                "paths_tested",
                "species",
            ],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
    )
    species.to_csv(SPECIES_OUT, index=False)

    # Medication missingness / N QC.
    cohort = pd.read_csv(PROTOCOL_COHORT, low_memory=False)
    micro = pd.read_csv(MICRO, usecols=["participant_id"], low_memory=False)

    cohort["participant_id"] = norm_id(cohort["participant_id"])
    micro["participant_id"] = norm_id(micro["participant_id"])
    micro_ids = set(micro["participant_id"])

    if cohort["participant_id"].duplicated().any():
        raise RuntimeError("Protocol cohort duplicate participant_id")

    cohort["_has_microbiome"] = cohort["participant_id"].isin(micro_ids)

    medqc_rows = []
    for exposure in EXPOSURE_COL:
        for outcome in OUTCOME_COL:
            base = model_complete_mask(
                cohort, exposure, outcome, add_meds=False
            ) & cohort["_has_microbiome"]
            prot = model_complete_mask(
                cohort, exposure, outcome, add_meds=True
            ) & cohort["_has_microbiome"]

            medqc_rows.append({
                "diet_score": exposure,
                "cgm_outcome": outcome,
                "base_complete_N_reconstructed": int(base.sum()),
                "protocol_complete_N_reconstructed": int(prot.sum()),
                "N_lost_due_to_protocol_med_missingness": int(
                    base.sum() - prot.sum()
                ),
                "antibiotic_missing_among_base_complete": int(
                    cohort.loc[
                        base, "antibiotic_use_protocol_window"
                    ].isna().sum()
                ),
                "ppi_missing_among_base_complete": int(
                    cohort.loc[
                        base, "ppi_use_protocol_window"
                    ].isna().sum()
                ),
            })

    medqc = pd.DataFrame(medqc_rows)

    # Compare reconstructed N to actual path N medians.
    actual_primary = (
        p.groupby(["diet_score", "cgm_outcome"], as_index=False)
        .agg(actual_primary_N=("N", "median"))
    )
    actual_protocol = (
        a.groupby(["diet_score", "cgm_outcome"], as_index=False)
        .agg(actual_protocol_N=("N", "median"))
    )
    medqc = (
        medqc.merge(
            actual_primary,
            on=["diet_score", "cgm_outcome"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            actual_protocol,
            on=["diet_score", "cgm_outcome"],
            how="left",
            validate="one_to_one",
        )
    )
    medqc["primary_N_matches_reconstruction"] = (
        medqc["base_complete_N_reconstructed"]
        == medqc["actual_primary_N"]
    )
    medqc["protocol_N_matches_reconstruction"] = (
        medqc["protocol_complete_N_reconstructed"]
        == medqc["actual_protocol_N"]
    )
    medqc.to_csv(MEDQC_OUT, index=False)

    rho_strict = float(spearmanr(p_ind, s_ind).statistic)
    rho_protocol = float(spearmanr(p_ind, a_ind).statistic)

    primary_consistent_n = int(p_con.sum())
    retained_both_n = int((p_con & s_con & a_con).sum())

    by_exposure = (
        m.groupby("diet_score", as_index=False)
        .agg(
            primary_consistent=("primary_consistent", "sum"),
            strict_consistent=("strict_consistent", "sum"),
            protocol_consistent=("protocol_consistent", "sum"),
            primary_retained_strict=(
                "primary_consistent_retained_strict", "sum"
            ),
            primary_retained_protocol=(
                "primary_consistent_retained_protocol", "sum"
            ),
            primary_retained_both=(
                "primary_consistent_retained_both", "sum"
            ),
        )
    )

    lines = [
        "=== STEP 15f3 FINAL THREE-WAY MEDIATION ROBUSTNESS ===",
        "EXACT_45_PATH_SET_MATCH_ALL_THREE=True",
        f"PRIMARY_SIGNIFICANT={int(p_sig.sum())}",
        f"STRICT_SIGNIFICANT={int(s_sig.sum())}",
        f"PROTOCOL_ABXPPI_SIGNIFICANT={int(a_sig.sum())}",
        f"PRIMARY_CONSISTENT={primary_consistent_n}",
        f"STRICT_CONSISTENT={int(s_con.sum())}",
        f"PROTOCOL_ABXPPI_CONSISTENT={int(a_con.sum())}",
        f"PRIMARY_CONSISTENT_RETAINED_STRICT={int((p_con & s_con).sum())}/{primary_consistent_n}",
        f"PRIMARY_CONSISTENT_RETAINED_PROTOCOL={int((p_con & a_con).sum())}/{primary_consistent_n}",
        f"PRIMARY_CONSISTENT_RETAINED_BOTH={retained_both_n}/{primary_consistent_n}",
        f"INDIRECT_RHO_PRIMARY_VS_STRICT={rho_strict:.9f}",
        f"INDIRECT_RHO_PRIMARY_VS_PROTOCOL={rho_protocol:.9f}",
        f"INDIRECT_SIGN_FLIPS_PRIMARY_VS_STRICT={int((~m['indirect_same_sign_primary_strict']).sum())}",
        f"INDIRECT_SIGN_FLIPS_PRIMARY_VS_PROTOCOL={int((~m['indirect_same_sign_primary_protocol']).sum())}",
        "",
        "--- EXPOSURE ROBUSTNESS ---",
        by_exposure.to_string(index=False),
        "",
        "--- CONTEXT ROBUSTNESS ---",
        context.to_string(index=False),
        "",
        "--- PROTOCOL MEDICATION MISSINGNESS / N QC ---",
        medqc.to_string(index=False),
        "",
        "INTERPRETATION:",
        "- retained_both means a primary direction-consistent path remains",
        "  direction-consistent in BOTH strict diabetes/A10 and protocol-window",
        "  antibiotic/PPI sensitivity analyses.",
        "- Sensitivity-only gains are not promoted into the primary result set.",
        "- No sign flips means effect direction is stable even when FDR membership changes.",
        "",
        f"PATH_TABLE={PATH_OUT}",
        f"CONTEXT_TABLE={CONTEXT_OUT}",
        f"SPECIES_TABLE={SPECIES_OUT}",
        f"MEDICATION_QC={MEDQC_OUT}",
    ]

    SUMMARY_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
