#!/usr/bin/env python3
"""
Step 17 — Paper-20 robust Diet -> microbiome -> CGM bridge screen.

READ / OVERLAP / DIRECTION SCREEN ONLY
--------------------------------------
No model refit.
No new FDR calculation.
No mediation.

Inputs
------
A) Step15b:
   exact Diet x CGM contexts already eligible for bridge construction.

B) Existing Diet -> microbiome MWAS:
   Model2 + Model3 for the seven diet exposures.

C) Step16a:
   microbiome -> Paper-20 CGM MWAS, primary_m2 + primary_m3,
   plus exact-species highest-robustness annotation.

D) Step16b:
   special-outcome microbiome sensitivity gate for TAR140/HBGI/GRADE.
   TAR180 remains supplementary because Step15b has no primary direct gate.

Primary robust bridge candidate
-------------------------------
For one exact Diet x CGM context and one exact species:

1. Context is Step15b bridge_context_eligible.
2. Diet -> species FDR < 0.05 in BOTH Model2 and Model3.
3. Species -> CGM within-outcome FDR < 0.05 in BOTH
   primary_m2 and primary_m3.
4. Diet -> species beta keeps the same sign M2/M3.
5. Species -> CGM beta keeps the same sign primary_m2/primary_m3.
6. beta(Diet->species) * beta(species->CGM) has the same sign as the
   frozen direct Diet->CGM beta.
7. For distribution-sensitive outcomes, Step16b special_rule_ok=True.

Highest-priority annotation
---------------------------
Primary robust candidate +
- Step15b highest_robustness_context=True
- Step16a CGM-species highest_robustness=True
- for special outcomes, Step16b sensitivity_high_support=True

Highest-priority is only a prioritization tag and does not redefine the
primary robust bridge set.

Exact full taxonomy string is the species identity key.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
DG = ROOT / "diet_microbiome_glucose_analysis"
DM = ROOT / "diet_microbiome_analysis"

BRANCH_CANDIDATES = [
    DG / "4New_cgm",
    ROOT / "4New_cgm",
]

# New-diet EAT13/NOVA4/Carbohydrate outputs and the corrected-hPDI
# targeted rerun live under two different historical output roots.
NEW_DIET_BASE = DG / "outputs" / "new_diet_extension"
HPDI_CORRECTION_BASE = (
    DG / "outputs" / "reports" / "new_diet_extension"
    / "hpdi_correction_audit" / "downstream"
)
EXPECTED_SPECIES = 379

OLD_DIET_MODEL = {
    2: {
        "AHEI": DM / "reports" / "04_MWAS_AHEI_model2.csv",
        "AMED": DM / "reports" / "04_MWAS_AMED_model2.csv",
        "rEDIH": DM / "reports" / "04_MWAS_rEDIH_model2.csv",
    },
    3: {
        "AHEI": DM / "reports" / "05_MWAS_AHEI_model3.csv",
        "AMED": DM / "reports" / "05_MWAS_AMED_model3.csv",
        "rEDIH": DM / "reports" / "05_MWAS_rEDIH_model3.csv",
    },
}

CORRECTED_HPDI_MODEL = {
    2: (
        HPDI_CORRECTION_BASE
        / "microbiome" / "15b3_corrected_hPDI_MWAS_model2.csv"
    ),
    3: (
        HPDI_CORRECTION_BASE
        / "microbiome" / "15b3_corrected_hPDI_MWAS_model3.csv"
    ),
}

NEW_DIET_MODEL = {
    2: NEW_DIET_BASE / "models" / "15c4_MWAS_all_model2.csv",
    3: NEW_DIET_BASE / "models" / "15c5_MWAS_all_model3.csv",
}

SPECIAL_OUTCOMES = {
    "cgm_above_140",
    "cgm_hbgi",
    "cgm_grade",
    "cgm_above_180",
}


def require(path: Path, label: str):
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


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


def latest_dir(parent: Path, pattern: str, required_rel: str) -> Path:
    xs = []
    for p in parent.glob(pattern):
        f = p / required_rel
        if f.is_file():
            xs.append((f.stat().st_mtime, p))
    if not xs:
        raise FileNotFoundError(
            f"No {pattern} containing {required_rel} under {parent}"
        )
    xs.sort(reverse=True)
    return xs[0][1]


def load_one_diet_mwas(
    path: Path,
    exposure: str,
    model: int,
) -> pd.DataFrame:
    require(path, f"{exposure} Diet->microbiome Model{model}")
    df = pd.read_csv(path, low_memory=False)

    exposure_col = None
    for c in ["exposure", "score", "diet_score"]:
        if c in df.columns:
            exposure_col = c
            break

    required = {"species", "beta", "FDR"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{path}: missing {missing}")

    if exposure_col is not None:
        vals = set(df[exposure_col].astype(str).unique())
        if len(vals) > 1:
            df = df.loc[
                df[exposure_col].astype(str).eq(exposure)
            ].copy()

    if len(df) != EXPECTED_SPECIES:
        raise RuntimeError(
            f"{exposure} Model{model}: expected {EXPECTED_SPECIES}, got {len(df)}"
        )
    if df["species"].duplicated().any():
        raise RuntimeError(f"{exposure} Model{model}: duplicate species")

    out = df[["species", "beta", "FDR"]].copy()
    out["species"] = out["species"].astype(str)
    out["beta"] = pd.to_numeric(out["beta"], errors="coerce")
    out["FDR"] = pd.to_numeric(out["FDR"], errors="coerce")
    if out[["beta", "FDR"]].isna().any().any():
        raise RuntimeError(f"{exposure} Model{model}: invalid beta/FDR")

    out["sig"] = out["FDR"].lt(0.05)
    out.insert(0, "exposure", exposure)
    out.insert(1, "model", model)
    return out


def load_new_diet_mwas(model: int) -> pd.DataFrame:
    path = NEW_DIET_MODEL[model]
    require(path, f"new-diet MWAS Model{model}")
    df = pd.read_csv(path, low_memory=False)

    required = {"exposure", "species", "beta", "FDR"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{path}: missing {missing}")

    frames = []
    for exposure in ["EAT13", "NOVA4", "Carbohydrate_pct"]:
        x = df.loc[
            df["exposure"].astype(str).eq(exposure)
        ].copy()
        if len(x) != EXPECTED_SPECIES:
            raise RuntimeError(
                f"{exposure} Model{model}: expected {EXPECTED_SPECIES}, got {len(x)}"
            )
        if x["species"].duplicated().any():
            raise RuntimeError(
                f"{exposure} Model{model}: duplicate species"
            )
        x = x[["exposure", "species", "beta", "FDR"]].copy()
        x["model"] = model
        x["species"] = x["species"].astype(str)
        x["beta"] = pd.to_numeric(x["beta"], errors="coerce")
        x["FDR"] = pd.to_numeric(x["FDR"], errors="coerce")
        x["sig"] = x["FDR"].lt(0.05)
        frames.append(x)

    return pd.concat(frames, ignore_index=True)


def load_selected_diet_mwas(
    model: int,
    exposures: list[str],
) -> pd.DataFrame:
    requested = list(dict.fromkeys(map(str, exposures)))
    frames = []

    for exposure in ["AHEI", "AMED", "rEDIH"]:
        if exposure in requested:
            frames.append(
                load_one_diet_mwas(
                    OLD_DIET_MODEL[model][exposure],
                    exposure,
                    model,
                )
            )

    if "hPDI" in requested:
        frames.append(
            load_one_diet_mwas(
                CORRECTED_HPDI_MODEL[model],
                "hPDI",
                model,
            )
        )

    new_requested = [
        x for x in ["EAT13", "NOVA4", "Carbohydrate_pct"]
        if x in requested
    ]
    if new_requested:
        new_all = load_new_diet_mwas(model)
        frames.append(
            new_all.loc[
                new_all["exposure"].isin(new_requested)
            ].copy()
        )

    if not frames:
        raise RuntimeError(f"No Diet MWAS requested for Model{model}")

    out = pd.concat(frames, ignore_index=True)
    expected = len(requested) * EXPECTED_SPECIES

    if len(out) != expected:
        raise RuntimeError(
            f"Model{model} Diet MWAS: expected {expected}, got {len(out)}"
        )
    if out.duplicated(["exposure", "species"]).any():
        raise RuntimeError(
            f"Model{model} Diet MWAS duplicate exposure/species"
        )
    return out


def load_direct_gate(step15b: Path) -> pd.DataFrame:
    path = step15b / "reports" / "15b_bridge_eligible_contexts.csv"
    require(path, "Step15b eligible contexts")
    x = pd.read_csv(path, low_memory=False)

    required = {
        "exposure", "role", "outcome_field", "outcome_label",
        "beta_primary", "beta_strict",
        "bridge_context_eligible",
        "highest_robustness_context",
    }
    missing = sorted(required - set(x.columns))
    if missing:
        raise RuntimeError(f"Step15b gate missing {missing}")

    x["bridge_context_eligible"] = truthy(
        x["bridge_context_eligible"]
    )
    x["highest_robustness_context"] = truthy(
        x["highest_robustness_context"]
    )

    if not x["bridge_context_eligible"].all():
        raise RuntimeError(
            "15b_bridge_eligible_contexts.csv contains ineligible rows"
        )
    if x.duplicated(["exposure", "outcome_field"]).any():
        raise RuntimeError("Duplicate Step15b Diet x outcome context")

    x["beta_primary"] = pd.to_numeric(
        x["beta_primary"], errors="coerce"
    )
    x["beta_strict"] = pd.to_numeric(
        x["beta_strict"], errors="coerce"
    )
    if x[["beta_primary", "beta_strict"]].isna().any().any():
        raise RuntimeError("Step15b direct beta missing")

    same = (
        np.sign(x["beta_primary"])
        == np.sign(x["beta_strict"])
    )
    if not same.all():
        raise RuntimeError(
            "Step15b eligible context contains primary/strict sign mismatch"
        )
    return x


def load_cgm_mwas(step16a: Path, set_name: str) -> pd.DataFrame:
    path = step16a / "models" / "16_paper20_microbiome_all_models.csv"
    require(path, "Step16a microbiome MWAS")
    df = pd.read_csv(path, low_memory=False)

    required = {
        "set_name", "outcome_field", "outcome_label",
        "species", "beta", "FDR_family", "status",
        "significant_family_05",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Step16a MWAS missing {missing}")

    x = df.loc[df["set_name"].eq(set_name)].copy()
    if len(x) != 20 * EXPECTED_SPECIES:
        raise RuntimeError(
            f"{set_name}: expected {20*EXPECTED_SPECIES}, got {len(x)}"
        )

    x["species"] = x["species"].astype(str)
    x["beta"] = pd.to_numeric(x["beta"], errors="coerce")
    x["FDR_family"] = pd.to_numeric(
        x["FDR_family"], errors="coerce"
    )
    x["sig"] = (
        x["status"].eq("computed")
        & truthy(x["significant_family_05"])
    )

    if x.duplicated(["outcome_field", "species"]).any():
        raise RuntimeError(
            f"{set_name}: duplicate outcome/species"
        )

    return x[
        [
            "outcome_field", "outcome_label", "species",
            "beta", "FDR_family", "sig",
        ]
    ].copy()


def load_cgm_highest(step16a: Path) -> pd.DataFrame:
    path = step16a / "reports" / "16_exact_species_robustness_long.csv"
    require(path, "Step16a exact-species robustness")
    df = pd.read_csv(path, low_memory=False)

    required = {"outcome_field", "species", "highest_robustness"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            f"Step16a robustness missing {missing}"
        )

    x = df[["outcome_field", "species", "highest_robustness"]].copy()
    x["species"] = x["species"].astype(str)
    x["cgm_highest_robustness"] = truthy(
        x["highest_robustness"]
    )
    x = x.drop(columns="highest_robustness")

    if x.duplicated(["outcome_field", "species"]).any():
        raise RuntimeError("Duplicate Step16a robustness outcome/species")
    return x


def load_special_gate(step16b: Path) -> pd.DataFrame:
    path = step16b / "reports" / "16b_special_species_gate.csv"
    require(path, "Step16b special species gate")
    df = pd.read_csv(path, low_memory=False)

    required = {
        "outcome_field", "species",
        "special_rule_ok", "sensitivity_high_support",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Step16b gate missing {missing}")

    x = df[
        [
            "outcome_field", "species",
            "special_rule_ok", "sensitivity_high_support",
        ]
    ].copy()
    x["species"] = x["species"].astype(str)
    x["special_rule_ok"] = truthy(x["special_rule_ok"])
    x["sensitivity_high_support"] = truthy(
        x["sensitivity_high_support"]
    )
    if x.duplicated(["outcome_field", "species"]).any():
        raise RuntimeError(
            "Duplicate Step16b outcome/species"
        )
    return x


def expected_sign(gate_row: pd.Series) -> int:
    b = float(gate_row["beta_primary"])
    if not np.isfinite(b) or b == 0:
        raise RuntimeError(
            f"Invalid direct beta for "
            f"{gate_row['exposure']} x {gate_row['outcome_field']}"
        )
    return int(np.sign(b))


def analyze_layer(
    diet: pd.DataFrame,
    cgm: pd.DataFrame,
    gates: pd.DataFrame,
    model: int,
):
    overlap_rows = []
    summary_rows = []

    for _, gate in gates.iterrows():
        exposure = str(gate["exposure"])
        outcome = str(gate["outcome_field"])

        d = diet.loc[diet["exposure"].eq(exposure)].copy()
        g = cgm.loc[cgm["outcome_field"].eq(outcome)].copy()

        m = d.merge(
            g,
            on="species",
            how="inner",
            validate="one_to_one",
            suffixes=("_diet", "_cgm"),
        )
        if len(m) != EXPECTED_SPECIES:
            raise RuntimeError(
                f"M{model} {exposure} x {outcome}: "
                f"expected {EXPECTED_SPECIES}, got {len(m)}"
            )

        both = truthy(m["sig_diet"]) & truthy(m["sig_cgm"])
        cand = m.loc[both].copy()

        es = expected_sign(gate)

        cand["model"] = model
        cand["role"] = gate["role"]
        cand["outcome_field"] = outcome
        cand["outcome_label"] = gate["outcome_label"]
        cand["direct_beta_primary"] = float(gate["beta_primary"])
        cand["direct_beta_strict"] = float(gate["beta_strict"])
        cand["direct_context_highest"] = bool(
            gate["highest_robustness_context"]
        )
        cand["beta_product"] = (
            cand["beta_diet"] * cand["beta_cgm"]
        )
        cand["expected_product_sign"] = es
        cand["direction_concordant"] = (
            np.sign(cand["beta_product"]) == es
        )
        overlap_rows.append(cand)

        M = EXPECTED_SPECIES
        K = int(truthy(m["sig_diet"]).sum())
        n = int(truthy(m["sig_cgm"]).sum())
        k = int(both.sum())
        expected = K * n / M
        enrich_p = float(hypergeom.sf(k - 1, M, K, n))

        summary_rows.append({
            "model": model,
            "exposure": exposure,
            "role": gate["role"],
            "outcome_field": outcome,
            "outcome_label": gate["outcome_label"],
            "diet_sig_species": K,
            "cgm_sig_species": n,
            "observed_overlap_species": k,
            "expected_overlap_species": expected,
            "overlap_enrichment_fold": (
                k / expected if expected > 0 else np.nan
            ),
            "hypergeom_enrichment_p": enrich_p,
            "direction_concordant_overlap_species": int(
                truthy(cand["direction_concordant"]).sum()
            ),
        })

    overlap = (
        pd.concat(overlap_rows, ignore_index=True)
        if overlap_rows else pd.DataFrame()
    )
    return overlap, pd.DataFrame(summary_rows)


def compare_layers(
    m2: pd.DataFrame,
    m3: pd.DataFrame,
    cgm_highest: pd.DataFrame,
    special_gate: pd.DataFrame,
):
    keys = ["exposure", "outcome_field", "species"]

    if m2.empty or m3.empty:
        raise RuntimeError("Model2 or Model3 overlap table is empty")

    a = m2.rename(columns={
        "beta_diet": "m2_beta_diet",
        "FDR": "m2_FDR_diet",
        "beta_cgm": "m2_beta_cgm",
        "FDR_family": "m2_FDR_cgm",
        "beta_product": "m2_beta_product",
        "direction_concordant": "m2_direction_concordant",
    })
    b = m3.rename(columns={
        "beta_diet": "m3_beta_diet",
        "FDR": "m3_FDR_diet",
        "beta_cgm": "m3_beta_cgm",
        "FDR_family": "m3_FDR_cgm",
        "beta_product": "m3_beta_product",
        "direction_concordant": "m3_direction_concordant",
    })

    keep_a = keys + [
        "role", "outcome_label",
        "direct_beta_primary", "direct_beta_strict",
        "direct_context_highest",
        "m2_beta_diet", "m2_FDR_diet",
        "m2_beta_cgm", "m2_FDR_cgm",
        "m2_beta_product", "m2_direction_concordant",
        "expected_product_sign",
    ]
    keep_b = keys + [
        "m3_beta_diet", "m3_FDR_diet",
        "m3_beta_cgm", "m3_FDR_cgm",
        "m3_beta_product", "m3_direction_concordant",
    ]

    z = a[keep_a].merge(
        b[keep_b],
        on=keys,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    z["present_m2"] = z["_merge"].isin(["left_only", "both"])
    z["present_m3"] = z["_merge"].isin(["right_only", "both"])
    z["present_both_models"] = z["_merge"].eq("both")
    z = z.drop(columns="_merge")

    z["diet_beta_same_sign_m2_m3"] = np.where(
        z["present_both_models"],
        np.sign(z["m2_beta_diet"])
        == np.sign(z["m3_beta_diet"]),
        False,
    )
    z["cgm_beta_same_sign_m2_m3"] = np.where(
        z["present_both_models"],
        np.sign(z["m2_beta_cgm"])
        == np.sign(z["m3_beta_cgm"]),
        False,
    )

    z["robust_pre_special"] = (
        z["present_both_models"]
        & truthy(z["m2_direction_concordant"])
        & truthy(z["m3_direction_concordant"])
        & truthy(z["diet_beta_same_sign_m2_m3"])
        & truthy(z["cgm_beta_same_sign_m2_m3"])
    )

    z = z.merge(
        cgm_highest,
        on=["outcome_field", "species"],
        how="left",
        validate="many_to_one",
    )
    z["cgm_highest_robustness"] = (
        z["cgm_highest_robustness"]
        .fillna(False)
        .astype(bool)
    )

    z = z.merge(
        special_gate,
        on=["outcome_field", "species"],
        how="left",
        validate="many_to_one",
    )

    is_special = z["outcome_field"].isin(SPECIAL_OUTCOMES)

    z["special_rule_ok_effective"] = True
    z["special_high_support_effective"] = True

    z.loc[is_special, "special_rule_ok_effective"] = (
        z.loc[is_special, "special_rule_ok"]
        .fillna(False)
        .astype(bool)
    )
    z.loc[is_special, "special_high_support_effective"] = (
        z.loc[is_special, "sensitivity_high_support"]
        .fillna(False)
        .astype(bool)
    )

    z["robust_bridge_candidate"] = (
        z["robust_pre_special"]
        & z["special_rule_ok_effective"]
    )

    z["highest_priority_bridge_candidate"] = (
        z["robust_bridge_candidate"]
        & z["cgm_highest_robustness"]
        & truthy(z["direct_context_highest"])
        & z["special_high_support_effective"]
    )

    summary = (
        z.groupby(
            [
                "exposure", "role",
                "outcome_field", "outcome_label",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            model2_overlap_species=("present_m2", "sum"),
            model3_overlap_species=("present_m3", "sum"),
            overlap_present_both_models=("present_both_models", "sum"),
            robust_pre_special=("robust_pre_special", "sum"),
            robust_bridge_candidates=("robust_bridge_candidate", "sum"),
            highest_priority_candidates=(
                "highest_priority_bridge_candidate", "sum"
            ),
        )
        .sort_values(
            ["outcome_field", "exposure"]
        )
    )
    return z, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch-dir", default=None)
    ap.add_argument("--step15b-dir", default=None)
    ap.add_argument("--step16a-dir", default=None)
    ap.add_argument("--step16b-dir", default=None)
    args = ap.parse_args()

    branch = find_branch(args.branch_dir)
    outputs = branch / "outputs"

    step15b = (
        Path(args.step15b_dir).expanduser().resolve()
        if args.step15b_dir
        else latest_dir(
            outputs,
            "paper20_direct_gate_*",
            "reports/15b_bridge_eligible_contexts.csv",
        )
    )
    step16a = (
        Path(args.step16a_dir).expanduser().resolve()
        if args.step16a_dir
        else latest_dir(
            outputs,
            "paper20_microbiome_mwas_*",
            "models/16_paper20_microbiome_all_models.csv",
        )
    )
    step16b = (
        Path(args.step16b_dir).expanduser().resolve()
        if args.step16b_dir
        else latest_dir(
            outputs,
            "paper20_microbiome_special_sensitivity_*",
            "reports/16b_special_species_gate.csv",
        )
    )

    print("=== STEP 17 PAPER-20 ROBUST BRIDGE SCREEN ===")
    print(f"STEP15B={step15b}")
    print(f"STEP16A={step16a}")
    print(f"STEP16B={step16b}")
    print("MODELS_REFIT=False")
    print("FDR_RECALCULATED=False")
    print("MEDIATION_RUN=False")

    gates = load_direct_gate(step15b)
    print(f"ELIGIBLE_DIRECT_CONTEXTS={len(gates)}")

    eligible_exposures = sorted(
        gates["exposure"].astype(str).unique().tolist()
    )
    print(
        "ELIGIBLE_DIET_MWAS_EXPOSURES="
        + ",".join(eligible_exposures)
    )

    print(f"NEW_DIET_MWAS_BASE={NEW_DIET_BASE}")
    print(f"CORRECTED_HPDI_MWAS_BASE={HPDI_CORRECTION_BASE}")
    if "hPDI" in eligible_exposures:
        print(f"CORRECTED_HPDI_MODEL2={CORRECTED_HPDI_MODEL[2]}")
        print(f"CORRECTED_HPDI_MODEL3={CORRECTED_HPDI_MODEL[3]}")

    diet2 = load_selected_diet_mwas(2, eligible_exposures)
    diet3 = load_selected_diet_mwas(3, eligible_exposures)

    cgm2 = load_cgm_mwas(step16a, "primary_m2")
    cgm3 = load_cgm_mwas(step16a, "primary_m3")

    highest = load_cgm_highest(step16a)
    special = load_special_gate(step16b)

    m2, m2_summary = analyze_layer(
        diet2, cgm2, gates, 2
    )
    m3, m3_summary = analyze_layer(
        diet3, cgm3, gates, 3
    )

    all_bridge, context_summary = compare_layers(
        m2, m3, highest, special
    )

    robust = all_bridge.loc[
        truthy(all_bridge["robust_bridge_candidate"])
    ].copy()

    priority = all_bridge.loc[
        truthy(all_bridge["highest_priority_bridge_candidate"])
    ].copy()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = outputs / f"paper20_bridge_screen_{stamp}"
    reports = out / "reports"
    reports.mkdir(parents=True, exist_ok=False)

    gate_path = reports / "17_bridge_eligible_diet_cgm_contexts.csv"
    m2_path = reports / "17_model2_overlap_species.csv"
    m3_path = reports / "17_model3_overlap_species.csv"
    all_path = reports / "17_model2_vs_model3_bridge_all.csv"
    context_path = reports / "17_bridge_context_summary.csv"
    robust_path = reports / "17_robust_bridge_candidates.csv"
    priority_path = reports / "17_highest_priority_bridge_candidates.csv"
    recurrence_path = reports / "17_species_recurrence.csv"
    txt_path = reports / "17_paper20_bridge_summary.txt"

    gates.to_csv(gate_path, index=False)
    m2.to_csv(m2_path, index=False)
    m3.to_csv(m3_path, index=False)
    all_bridge.to_csv(all_path, index=False)
    context_summary.to_csv(context_path, index=False)
    robust.to_csv(robust_path, index=False)
    priority.to_csv(priority_path, index=False)

    recurrence = (
        robust.groupby("species", as_index=False)
        .agg(
            robust_paths=("species", "size"),
            diet_exposures=("exposure", "nunique"),
            cgm_outcomes=("outcome_field", "nunique"),
            exposures=(
                "exposure",
                lambda x: ";".join(sorted(set(map(str, x))))
            ),
            outcomes=(
                "outcome_field",
                lambda x: ";".join(sorted(set(map(str, x))))
            ),
            highest_priority_paths=(
                "highest_priority_bridge_candidate", "sum"
            ),
        )
        .sort_values(
            [
                "robust_paths", "diet_exposures",
                "cgm_outcomes", "species",
            ],
            ascending=[False, False, False, True],
        )
    )
    recurrence.to_csv(recurrence_path, index=False)

    by_diet = (
        robust.groupby(["exposure", "role"], as_index=False)
        .agg(
            robust_paths=("species", "size"),
            unique_species=("species", "nunique"),
            cgm_outcomes=("outcome_field", "nunique"),
            highest_priority_paths=(
                "highest_priority_bridge_candidate", "sum"
            ),
        )
        .sort_values(["role", "exposure"])
    )

    by_outcome = (
        robust.groupby(
            ["outcome_field", "outcome_label"], as_index=False
        )
        .agg(
            robust_paths=("species", "size"),
            unique_species=("species", "nunique"),
            diets=("exposure", "nunique"),
            highest_priority_paths=(
                "highest_priority_bridge_candidate", "sum"
            ),
        )
        .sort_values(
            ["robust_paths", "unique_species", "outcome_field"],
            ascending=[False, False, True],
        )
    )

    special_blocked = all_bridge.loc[
        truthy(all_bridge["robust_pre_special"])
        & ~truthy(all_bridge["special_rule_ok_effective"])
    ].copy()

    lines = [
        "=== STEP 17 PAPER-20 ROBUST BRIDGE SUMMARY ===",
        f"STEP15B={step15b}",
        f"STEP16A={step16a}",
        f"STEP16B={step16b}",
        "MODELS_REFIT=False",
        "FDR_RECALCULATED=False",
        "MEDIATION_RUN=False",
        "",
        f"ELIGIBLE_DIRECT_CONTEXTS={len(gates)}",
        f"ROBUST_BRIDGE_PATHS={len(robust)}",
        f"ROBUST_UNIQUE_EXACT_SPECIES={robust['species'].nunique()}",
        f"HIGHEST_PRIORITY_BRIDGE_PATHS={len(priority)}",
        f"HIGHEST_PRIORITY_UNIQUE_SPECIES={priority['species'].nunique()}",
        f"SPECIAL_SENSITIVITY_BLOCKED_PATHS={len(special_blocked)}",
        "",
        "--- ROBUST PATHS BY DIET ---",
        (
            "NONE"
            if by_diet.empty
            else by_diet.to_string(index=False)
        ),
        "",
        "--- ROBUST PATHS BY CGM OUTCOME ---",
        (
            "NONE"
            if by_outcome.empty
            else by_outcome.to_string(index=False)
        ),
        "",
        "--- ROBUST SPECIES RECURRENCE TOP 30 ---",
        (
            "NONE"
            if recurrence.empty
            else recurrence.head(30).to_string(index=False)
        ),
        "",
        "--- SPECIAL-SENSITIVITY BLOCKED PATHS ---",
        (
            "NONE"
            if special_blocked.empty
            else special_blocked[
                [
                    "exposure", "outcome_field", "species",
                    "m2_FDR_diet", "m2_FDR_cgm",
                    "m3_FDR_diet", "m3_FDR_cgm",
                ]
            ].to_string(index=False)
        ),
        "",
        "INTERPRETATION RULES:",
        "- Primary robust bridge membership requires Model2+Model3 robustness on both association legs.",
        "- Product direction must agree with the frozen direct Diet->CGM direction.",
        "- Step16b special_rule_ok is required for distribution-sensitive outcomes.",
        "- Highest-priority is a robustness/prioritization subset only.",
        "- Carbohydrate_pct remains exploratory even if a path is robust.",
        "- Mean glucose and GMI are highly dependent and are not independent replication.",
        "- These are cross-sectional bridge candidates, not causal mediation.",
        "",
        f"GATES={gate_path}",
        f"ROBUST_CANDIDATES={robust_path}",
        f"HIGHEST_PRIORITY={priority_path}",
        f"CONTEXT_SUMMARY={context_path}",
        f"RECURRENCE={recurrence_path}",
        f"OUTPUT_DIR={out}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    (reports / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "models_refit": False,
                "fdr_recalculated": False,
                "mediation_run": False,
                "eligible_direct_contexts": int(len(gates)),
                "robust_bridge_paths": int(len(robust)),
                "robust_unique_exact_species": int(
                    robust["species"].nunique()
                ),
                "highest_priority_bridge_paths": int(len(priority)),
                "highest_priority_unique_species": int(
                    priority["species"].nunique()
                ),
                "special_sensitivity_blocked_paths": int(
                    len(special_blocked)
                ),
                "outputs": {
                    "gate": str(gate_path),
                    "robust_candidates": str(robust_path),
                    "highest_priority": str(priority_path),
                    "context_summary": str(context_path),
                    "recurrence": str(recurrence_path),
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
