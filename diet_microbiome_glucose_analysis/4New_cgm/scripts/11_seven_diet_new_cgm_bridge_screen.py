#!/usr/bin/env python3
"""Step 11 — seven-diet -> microbiome -> new-CGM robust bridge screen.

This is an overlap/direction screen only. It does NOT run mediation.

It combines:
A) direct Diet -> CGM robustness from Step09
B) Diet -> microbiome MWAS Model2/Model3
C) microbiome -> new CGM MWAS primary Model2/Model3\nOnly Diet->microbiome MWAS tables for direct-gate-eligible diets are loaded.
D) Step10 strict/highest-robustness labels for annotation only

Bridge contexts
---------------
A Diet x CGM context is eligible only when ALL are true:
- outcome is inferential and prespecified for bridge:
      MAGE or TBR70
- primary Model2 classical family FDR < 0.05
- primary Model2 HC3 family FDR < 0.05
- primary Model3 classical family FDR < 0.05
- primary Model3 HC3 family FDR < 0.05
- direct Diet->CGM beta has the same sign in Model2 and Model3

TAR180 is NOT promoted from hurdle-only sensitivity into a primary bridge gate.
TIR70_180 is supportive and is NOT used as an independent bridge family.

Species-level robust bridge candidate
-------------------------------------
Within an eligible Diet x CGM context, an exact species is a robust candidate
only if:
- Diet->species FDR<0.05 in BOTH Model2 and Model3
- species->CGM FDR<0.05 in BOTH primary Model2 and primary Model3
- Diet->species beta keeps the same sign M2/M3
- species->CGM beta keeps the same sign M2/M3
- beta-product direction matches the frozen exposure/outcome direction rule

Direction rule
--------------
For every diet/outcome context:
    expected sign(beta_diet->species * beta_species->CGM)
    = sign(observed robust direct Diet->CGM beta)

The direct Diet->CGM sign must agree in primary Model2 and Model3 before the
context is bridge-eligible. Healthy/adverse labels are interpretation only;
they are never used to decide bridge membership.

Important
---------
- Primary bridge membership uses primary Model2+Model3 robustness only,
  consistent with the earlier bridge pipeline.
- Step10 strict/highest-robustness status is added as a PRIORITIZATION TAG,
  not used to redefine the primary candidate set.
- Exact full taxonomy string is the species identity key.
- No model is refit.
- No mediation is run.

Dependencies: numpy, pandas, scipy.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import hashlib

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")

BRANCH_CANDIDATES = [
    ROOT / "diet_microbiome_glucose_analysis" / "4New_cgm",
    ROOT / "4New_cgm",
]

DG = ROOT / "diet_microbiome_glucose_analysis"
DM = ROOT / "diet_microbiome_analysis"

NEW_DIET_BASE = DG / "outputs" / "new_diet_extension"

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
        NEW_DIET_BASE / "hpdi_correction_audit" / "downstream"
        / "microbiome" / "15b3_corrected_hPDI_MWAS_model2.csv"
    ),
    3: (
        NEW_DIET_BASE / "hpdi_correction_audit" / "downstream"
        / "microbiome" / "15b3_corrected_hPDI_MWAS_model3.csv"
    ),
}

NEW_DIET_MODEL = {
    2: NEW_DIET_BASE / "models" / "15c4_MWAS_all_model2.csv",
    3: NEW_DIET_BASE / "models" / "15c5_MWAS_all_model3.csv",
}

EXPECTED_SPECIES = 379

EXPOSURE_ROLE = {
    "AHEI": "existing_primary",
    "AMED": "existing_primary",
    "hPDI": "existing_primary_corrected",
    "rEDIH": "existing_primary",
    "EAT13": "primary_extension",
    "NOVA4": "primary_extension",
    "Carbohydrate_pct": "exploratory_extension",
}

BRIDGE_OUTCOMES = {"MAGE", "TBR70"}


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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_branch(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(p)
        return p
    found = [p for p in BRANCH_CANDIDATES if p.is_dir()]
    if not found:
        raise FileNotFoundError("4New_cgm branch not found")
    return found[0]


def latest_dir(parent: Path, pattern: str, required_rel: str) -> Path:
    candidates = []
    for p in parent.glob(pattern):
        f = p / required_rel
        if f.is_file():
            candidates.append((f.stat().st_mtime, p))
    if not candidates:
        raise FileNotFoundError(
            f"No {pattern} containing {required_rel} under {parent}"
        )
    candidates.sort(reverse=True)
    return candidates[0][1]


def load_one_diet_mwas(
    path: Path,
    exposure: str,
    model: int,
) -> pd.DataFrame:
    require(path, f"{exposure} diet->microbiome Model{model}")
    df = pd.read_csv(path, low_memory=False)

    # Old/corrected hPDI files use score; new files use exposure.
    exposure_col = None
    for c in ["exposure", "score", "diet_score"]:
        if c in df.columns:
            exposure_col = c
            break

    required = {"species", "beta", "FDR"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{path}: missing columns {missing}")

    if exposure_col is not None:
        # Corrected hPDI and old single-score files may have a score column.
        vals = set(df[exposure_col].astype(str).unique())
        if len(vals) > 1:
            df = df.loc[df[exposure_col].astype(str).eq(exposure)].copy()

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
        raise RuntimeError(f"{path}: missing columns {missing}")

    frames = []
    for exposure in ["EAT13", "NOVA4", "Carbohydrate_pct"]:
        x = df.loc[df["exposure"].astype(str).eq(exposure)].copy()
        if len(x) != EXPECTED_SPECIES:
            raise RuntimeError(
                f"{exposure} Model{model}: expected {EXPECTED_SPECIES}, got {len(x)}"
            )
        if x["species"].duplicated().any():
            raise RuntimeError(f"{exposure} Model{model}: duplicate species")
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
    """Load Diet->microbiome MWAS only for direct-gate-eligible exposures.

    This is intentional rather than a workaround:
    a Diet->species table is needed only when that diet already passed the
    prespecified robust Diet->CGM bridge gate for at least one target outcome.

    Therefore ineligible diets (for example corrected hPDI in the current
    MAGE/TBR70 bridge screen) must not block the bridge analysis merely because
    an unused historical MWAS file is stored under a different path.
    """
    requested = list(dict.fromkeys(map(str, exposures)))
    allowed = set(EXPOSURE_ROLE)
    unknown = sorted(set(requested) - allowed)
    if unknown:
        raise RuntimeError(f"Unknown requested exposures: {unknown}")

    frames = []

    # Existing original diet scores.
    for exposure in ["AHEI", "AMED", "rEDIH"]:
        if exposure in requested:
            frames.append(
                load_one_diet_mwas(
                    OLD_DIET_MODEL[model][exposure],
                    exposure,
                    model,
                )
            )

    # Corrected hPDI is loaded only if hPDI actually passed the direct gate.
    if "hPDI" in requested:
        frames.append(
            load_one_diet_mwas(
                CORRECTED_HPDI_MODEL[model],
                "hPDI",
                model,
            )
        )

    # New diet extensions share one combined MWAS file.
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
        raise RuntimeError(
            f"No Diet->microbiome MWAS tables requested for Model{model}"
        )

    out = pd.concat(frames, ignore_index=True)

    expected_rows = len(requested) * EXPECTED_SPECIES
    if len(out) != expected_rows:
        counts = out.groupby("exposure").size().to_dict()
        raise RuntimeError(
            f"Selected Diet MWAS Model{model}: expected {expected_rows} rows "
            f"for {requested}, got {len(out)}; counts={counts}"
        )

    observed = set(out["exposure"].astype(str))
    if observed != set(requested):
        raise RuntimeError(
            f"Selected Diet MWAS Model{model}: requested={sorted(requested)}, "
            f"observed={sorted(observed)}"
        )

    if out.duplicated(["exposure", "species"]).any():
        raise RuntimeError(
            f"Selected Diet MWAS Model{model}: duplicate exposure/species"
        )

    return out


def load_direct_gate(step09_dir: Path) -> pd.DataFrame:
    path = step09_dir / "models" / "hc3_full_7x4_models.csv"
    require(path, "Step09 HC3 models")

    df = pd.read_csv(path, low_memory=False)
    required = {
        "analysis_set",
        "model",
        "exposure",
        "outcome",
        "beta",
        "classical_FDR_family",
        "HC3_FDR_family",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Step09 HC3 table missing {missing}")

    x = df.loc[
        df["analysis_set"].eq("primary")
        & df["model"].isin([2, 3])
        & df["outcome"].isin(BRIDGE_OUTCOMES)
    ].copy()

    # One row per exposure/outcome/model.
    if x.duplicated(["model", "exposure", "outcome"]).any():
        raise RuntimeError("Duplicate direct Diet-CGM rows")

    wide = x.pivot(
        index=["exposure", "outcome"],
        columns="model",
        values=[
            "beta",
            "classical_FDR_family",
            "HC3_FDR_family",
        ],
    )
    wide.columns = [
        f"{a}_m{b}" for a, b in wide.columns
    ]
    wide = wide.reset_index()

    for c in [
        "beta_m2", "beta_m3",
        "classical_FDR_family_m2", "classical_FDR_family_m3",
        "HC3_FDR_family_m2", "HC3_FDR_family_m3",
    ]:
        wide[c] = pd.to_numeric(wide[c], errors="coerce")

    wide["same_direct_sign_m2_m3"] = (
        np.sign(wide["beta_m2"]) == np.sign(wide["beta_m3"])
    )

    wide["bridge_context_eligible"] = (
        wide["classical_FDR_family_m2"].lt(0.05)
        & wide["HC3_FDR_family_m2"].lt(0.05)
        & wide["classical_FDR_family_m3"].lt(0.05)
        & wide["HC3_FDR_family_m3"].lt(0.05)
        & wide["same_direct_sign_m2_m3"]
    )

    wide["exposure_role"] = wide["exposure"].map(EXPOSURE_ROLE)

    return wide


def load_cgm_mwas(run_dir: Path, model: int) -> pd.DataFrame:
    path = run_dir / "models" / "microbiome_cgm_all_models.csv"
    require(path, "4New_cgm microbiome model table")

    df = pd.read_csv(path, low_memory=False)
    required = {
        "analysis_set", "model", "outcome", "species",
        "beta", "FDR_family", "status",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"CGM MWAS missing columns {missing}")

    x = df.loc[
        df["analysis_set"].eq("primary")
        & df["model"].eq(model)
        & df["outcome"].isin(BRIDGE_OUTCOMES)
    ].copy()

    x["species"] = x["species"].astype(str)
    x["beta"] = pd.to_numeric(x["beta"], errors="coerce")
    x["FDR_family"] = pd.to_numeric(x["FDR_family"], errors="coerce")
    x["sig"] = (
        x["status"].eq("computed")
        & x["FDR_family"].lt(0.05)
    )

    for outcome in BRIDGE_OUTCOMES:
        piece = x.loc[x["outcome"].eq(outcome)]
        if len(piece) != EXPECTED_SPECIES:
            raise RuntimeError(
                f"{outcome} CGM MWAS Model{model}: "
                f"expected {EXPECTED_SPECIES}, got {len(piece)}"
            )
        if piece["species"].duplicated().any():
            raise RuntimeError(
                f"{outcome} CGM MWAS Model{model}: duplicate species"
            )

    return x[
        ["outcome", "species", "beta", "FDR_family", "sig"]
    ].copy()


def expected_product_sign(gate_row: pd.Series) -> tuple[int, str]:
    """Bridge sign gate follows the observed robust direct Diet->CGM effect."""
    exposure = str(gate_row["exposure"])
    outcome = str(gate_row["outcome"])
    b2 = float(gate_row["beta_m2"])
    b3 = float(gate_row["beta_m3"])

    if (
        not np.isfinite(b2)
        or not np.isfinite(b3)
        or b2 == 0
        or b3 == 0
        or np.sign(b2) != np.sign(b3)
    ):
        raise RuntimeError(
            f"{exposure} x {outcome}: direct Diet->CGM sign is not robust "
            f"(M2={b2}, M3={b3})"
        )

    return int(np.sign(b2)), "observed_direct_diet_cgm_direction"


def analyze_layer(
    diet: pd.DataFrame,
    cgm: pd.DataFrame,
    gates: pd.DataFrame,
    model: int,
):
    overlap_rows = []
    summary_rows = []

    eligible = gates.loc[gates["bridge_context_eligible"]].copy()

    for _, gate in eligible.iterrows():
        exposure = str(gate["exposure"])
        outcome = str(gate["outcome"])

        d = diet.loc[diet["exposure"].eq(exposure)].copy()
        g = cgm.loc[cgm["outcome"].eq(outcome)].copy()

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

        expected_sign, rule_type = expected_product_sign(gate)

        cand["model"] = model
        cand["exposure_role"] = EXPOSURE_ROLE[exposure]
        cand["beta_product"] = cand["beta_diet"] * cand["beta_cgm"]
        cand["expected_product_sign_numeric"] = expected_sign
        cand["direction_rule_type"] = rule_type
        cand["direction_concordant"] = (
            np.sign(cand["beta_product"]) == expected_sign
        )
        cand["direct_beta_m2"] = float(gate["beta_m2"])
        cand["direct_beta_m3"] = float(gate["beta_m3"])

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
            "exposure_role": EXPOSURE_ROLE[exposure],
            "outcome": outcome,
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
    summary = pd.DataFrame(summary_rows)
    return overlap, summary


def load_step10_highest(step10_dir: Path) -> pd.DataFrame:
    path = step10_dir / "reports" / "highest_robustness_species.csv"
    require(path, "Step10 highest-robustness species")
    df = pd.read_csv(path, low_memory=False)
    if "outcome" not in df.columns or "species" not in df.columns:
        raise RuntimeError("Step10 highest robustness file missing outcome/species")
    x = df[["outcome", "species"]].copy()
    x["species"] = x["species"].astype(str)
    x["cgm_highest_robustness"] = True
    return x


def compare_layers(
    m2: pd.DataFrame,
    m3: pd.DataFrame,
    step10_high: pd.DataFrame,
):
    keys = ["exposure", "outcome", "species"]

    if m2.empty or m3.empty:
        return pd.DataFrame(), pd.DataFrame()

    # beta and sig collide during the merge, so pandas adds _diet/_cgm.
    # FDR and FDR_family do NOT collide, so those names remain unchanged.
    required_m2 = {
        "exposure", "outcome", "species",
        "beta_diet", "FDR", "beta_cgm", "FDR_family",
        "beta_product", "direction_concordant",
    }
    required_m3 = set(required_m2)

    missing_m2 = sorted(required_m2 - set(m2.columns))
    missing_m3 = sorted(required_m3 - set(m3.columns))
    if missing_m2 or missing_m3:
        raise RuntimeError(
            "Unexpected overlap-table schema before Model2/Model3 comparison. "
            f"missing_m2={missing_m2}; missing_m3={missing_m3}; "
            f"m2_columns={m2.columns.tolist()}; m3_columns={m3.columns.tolist()}"
        )

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
        "exposure_role",
        "m2_beta_diet", "m2_FDR_diet",
        "m2_beta_cgm", "m2_FDR_cgm",
        "m2_beta_product", "m2_direction_concordant",
        "expected_product_sign_numeric",
        "direction_rule_type",
        "direct_beta_m2", "direct_beta_m3",
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

    z["diet_beta_same_sign_m2_m3"] = np.where(
        z["present_both_models"],
        np.sign(z["m2_beta_diet"]) == np.sign(z["m3_beta_diet"]),
        False,
    )
    z["cgm_beta_same_sign_m2_m3"] = np.where(
        z["present_both_models"],
        np.sign(z["m2_beta_cgm"]) == np.sign(z["m3_beta_cgm"]),
        False,
    )

    z["robust_bridge_candidate"] = (
        z["present_both_models"]
        & truthy(z["m2_direction_concordant"])
        & truthy(z["m3_direction_concordant"])
        & truthy(z["diet_beta_same_sign_m2_m3"])
        & truthy(z["cgm_beta_same_sign_m2_m3"])
    )

    z = z.drop(columns="_merge")

    z = z.merge(
        step10_high,
        on=["outcome", "species"],
        how="left",
        validate="many_to_one",
    )
    z["cgm_highest_robustness"] = (
        z["cgm_highest_robustness"].fillna(False).astype(bool)
    )

    z["highest_priority_bridge_candidate"] = (
        z["robust_bridge_candidate"]
        & z["cgm_highest_robustness"]
    )

    summary = (
        z.groupby(
            ["exposure", "exposure_role", "outcome"],
            as_index=False,
            dropna=False,
        )
        .agg(
            model2_overlap_species=("present_m2", "sum"),
            model3_overlap_species=("present_m3", "sum"),
            overlap_present_both_models=("present_both_models", "sum"),
            robust_bridge_candidates=("robust_bridge_candidate", "sum"),
            highest_priority_candidates=(
                "highest_priority_bridge_candidate", "sum"
            ),
        )
        .sort_values(["outcome", "exposure"])
    )

    return z, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-dir", default=None)
    parser.add_argument("--association-run-dir", default=None)
    parser.add_argument("--step09-dir", default=None)
    parser.add_argument("--step10-dir", default=None)
    args = parser.parse_args()

    branch = find_branch(args.branch_dir)

    assoc_run = (
        Path(args.association_run_dir).expanduser().resolve()
        if args.association_run_dir
        else latest_dir(
            branch / "outputs",
            "run_*",
            "models/microbiome_cgm_all_models.csv",
        )
    )

    step09 = (
        Path(args.step09_dir).expanduser().resolve()
        if args.step09_dir
        else latest_dir(
            branch / "outputs",
            "outcome_qc_exact_*",
            "models/hc3_full_7x4_models.csv",
        )
    )

    step10 = (
        Path(args.step10_dir).expanduser().resolve()
        if args.step10_dir
        else latest_dir(
            branch / "outputs",
            "species_robustness_*",
            "reports/highest_robustness_species.csv",
        )
    )

    print("=== STEP 11 SEVEN-DIET x NEW-CGM ROBUST BRIDGE SCREEN ===")
    print(f"ASSOCIATION_RUN={assoc_run}")
    print(f"STEP09={step09}")
    print(f"STEP10={step10}")
    print("NO_MODELS_REFIT=True")
    print("MEDIATION_RUN=False")
    print("TAR180_BRIDGE_GATE=False")
    print("TIR70_180_BRIDGE_GATE=False")

    gate = load_direct_gate(step09)

    print("\n--- DIRECT DIET-CGM BRIDGE GATE ---")
    print(
        gate[
            [
                "exposure", "exposure_role", "outcome",
                "beta_m2", "beta_m3",
                "classical_FDR_family_m2", "HC3_FDR_family_m2",
                "classical_FDR_family_m3", "HC3_FDR_family_m3",
                "same_direct_sign_m2_m3",
                "bridge_context_eligible",
            ]
        ].sort_values(["outcome", "exposure"]).to_string(index=False)
    )

    eligible = gate.loc[gate["bridge_context_eligible"]].copy()
    if eligible.empty:
        raise RuntimeError("No robust direct Diet-CGM bridge contexts")

    eligible_exposures = sorted(
        eligible["exposure"].astype(str).unique().tolist()
    )
    print(
        "\nELIGIBLE_DIET_MWAS_EXPOSURES="
        + ",".join(eligible_exposures)
    )
    print(
        "INELIGIBLE_DIETS_NOT_LOADED="
        + ",".join(
            sorted(set(EXPOSURE_ROLE) - set(eligible_exposures))
        )
    )
    print("BRIDGE_DIRECTION_RULE=observed_direct_Diet_to_CGM_sign")

    # Only direct-gate-eligible diets require Diet->species MWAS data.
    # This prevents an unused historical file path from blocking the analysis.
    diet2 = load_selected_diet_mwas(2, eligible_exposures)
    diet3 = load_selected_diet_mwas(3, eligible_exposures)
    cgm2 = load_cgm_mwas(assoc_run, 2)
    cgm3 = load_cgm_mwas(assoc_run, 3)
    step10_high = load_step10_highest(step10)

    m2_overlap, m2_summary = analyze_layer(
        diet2, cgm2, gate, 2
    )
    m3_overlap, m3_summary = analyze_layer(
        diet3, cgm3, gate, 3
    )

    all_bridge, bridge_summary = compare_layers(
        m2_overlap,
        m3_overlap,
        step10_high,
    )

    robust = all_bridge.loc[
        truthy(all_bridge["robust_bridge_candidate"])
    ].copy()

    priority = robust.loc[
        truthy(robust["cgm_highest_robustness"])
    ].copy()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = branch / "outputs" / f"bridge_screen_{stamp}"
    reports = out / "reports"
    reports.mkdir(parents=True, exist_ok=False)

    gate_path = reports / "bridge_eligible_diet_cgm_contexts.csv"
    m2_path = reports / "model2_overlap_species.csv"
    m2sum_path = reports / "model2_overlap_summary.csv"
    m3_path = reports / "model3_overlap_species.csv"
    m3sum_path = reports / "model3_overlap_summary.csv"
    all_path = reports / "model2_vs_model3_bridge_all.csv"
    summary_path = reports / "bridge_context_summary.csv"
    robust_path = reports / "robust_bridge_candidates.csv"
    priority_path = reports / "highest_priority_bridge_candidates.csv"
    txt_path = reports / "step11_bridge_summary.txt"
    manifest_path = reports / "manifest.json"

    eligible.to_csv(gate_path, index=False)
    m2_overlap.to_csv(m2_path, index=False)
    m2_summary.to_csv(m2sum_path, index=False)
    m3_overlap.to_csv(m3_path, index=False)
    m3_summary.to_csv(m3sum_path, index=False)
    all_bridge.to_csv(all_path, index=False)
    bridge_summary.to_csv(summary_path, index=False)
    robust.to_csv(robust_path, index=False)
    priority.to_csv(priority_path, index=False)

    species_recurrence = (
        robust.groupby("species", as_index=False)
        .agg(
            robust_paths=("species", "size"),
            diet_exposures=("exposure", "nunique"),
            cgm_outcomes=("outcome", "nunique"),
            exposures=("exposure", lambda x: ";".join(sorted(set(map(str, x))))),
            outcomes=("outcome", lambda x: ";".join(sorted(set(map(str, x))))),
            highest_priority_any=(
                "cgm_highest_robustness",
                lambda x: bool(pd.Series(x).fillna(False).any()),
            ),
        )
        .sort_values(
            ["robust_paths", "diet_exposures", "cgm_outcomes", "species"],
            ascending=[False, False, False, True],
        )
    )

    lines = [
        "=== STEP 11 SEVEN-DIET x NEW-CGM ROBUST BRIDGE SUMMARY ===",
        f"ELIGIBLE_DIRECT_CONTEXTS={len(eligible)}",
        f"ROBUST_BRIDGE_PATHS={len(robust)}",
        f"ROBUST_UNIQUE_EXACT_SPECIES={robust['species'].nunique()}",
        f"HIGHEST_PRIORITY_BRIDGE_PATHS={len(priority)}",
        f"HIGHEST_PRIORITY_UNIQUE_SPECIES={priority['species'].nunique()}",
        "",
        "--- ELIGIBLE DIRECT DIET-CGM CONTEXTS ---",
        eligible[
            [
                "exposure", "exposure_role", "outcome",
                "beta_m2", "beta_m3",
                "HC3_FDR_family_m2", "HC3_FDR_family_m3",
            ]
        ].sort_values(["outcome", "exposure"]).to_string(index=False),
        "",
        "--- BRIDGE CONTEXT SUMMARY ---",
        bridge_summary.to_string(index=False),
        "",
        "--- ROBUST CANDIDATE RECURRENCE TOP 30 ---",
        (
            "NONE"
            if species_recurrence.empty
            else species_recurrence.head(30).to_string(index=False)
        ),
        "",
        "INTERPRETATION RULES:",
        "- Primary bridge candidates require Model2+Model3 robustness on both association legs.",
        "- Step10 strict/highest robustness is a prioritization tag only.",
        "- TAR180 hurdle-only Diet->CGM signals do not create a primary bridge context.",
        "- TIR70_180 remains supportive and is excluded from independent bridge discovery.",
        "- Carbohydrate_pct remains exploratory even when bridge candidates are robust.",
        "- These are cross-sectional overlap/direction candidates, not causal mediation.",
        "",
        f"GATES={gate_path}",
        f"ROBUST_CANDIDATES={robust_path}",
        f"HIGHEST_PRIORITY={priority_path}",
        f"CONTEXT_SUMMARY={summary_path}",
    ]

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))

    manifest = {
        "status": "completed",
        "purpose": "seven-diet new-CGM robust bridge screening",
        "models_refit": False,
        "mediation_run": False,
        "association_run": str(assoc_run),
        "step09": str(step09),
        "step10": str(step10),
        "eligible_direct_contexts": len(eligible),
        "robust_bridge_paths": len(robust),
        "robust_unique_exact_species": int(robust["species"].nunique()),
        "highest_priority_bridge_paths": len(priority),
        "highest_priority_unique_species": int(priority["species"].nunique()),
        "rules": {
            "bridge_outcomes": sorted(BRIDGE_OUTCOMES),
            "tar180_primary_bridge": False,
            "tir70_180_primary_bridge": False,
            "carbohydrate_role": "exploratory_extension",
            "strict_step10_used_for_membership": False,
        },
        "outputs": {
            "gate": str(gate_path),
            "model2_overlap": str(m2_path),
            "model3_overlap": str(m3_path),
            "all_bridge": str(all_path),
            "robust_candidates": str(robust_path),
            "highest_priority": str(priority_path),
            "context_summary": str(summary_path),
            "summary": str(txt_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
