#!/usr/bin/env python3
"""Step 14a — assemble manuscript-ready final result tables (READ-ONLY).

This step does NOT refit any model.

It consolidates the already-finalized outputs into a small set of tables for
manuscript writing and figure construction.

Inputs
------
outputs/models/
    10_mediation_paths_model3.csv
    10_strict_mediation_paths_model3.csv
    12b_antibiotic_ppi_mediation_paths_model3.csv
    12e_protocol_window_mediation_paths_model3.csv

outputs/reports/
    12c_final_robustness_summary.csv
    13c_12_core_discussion_priority_table.csv

Optional:
    12e_protocol_window_comparison_summary.csv

Outputs
-------
outputs/final/
    Table14A_mediation_overview.csv
    Table14B_robustness_overview.csv
    Table14C_12_core_species.csv
    Table14D_primary_31_paths.csv
    Step14A_final_tables_summary.txt

Design rules
------------
- Full taxonomy string is preserved as species identity.
- Exact 31-path match is treated as pipeline QC only.
- Mediation is described as mediation-style / indirect association, not causal.
- 12-core refers specifically to recurrent AMED/hPDI × glucose_cv species.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
PROJECT = ROOT / "diet_microbiome_glucose_analysis"
MODEL_DIR = PROJECT / "outputs" / "models"
REPORT_DIR = PROJECT / "outputs" / "reports"
FINAL_DIR = PROJECT / "outputs" / "final"

PRIMARY = MODEL_DIR / "10_mediation_paths_model3.csv"
STRICT = MODEL_DIR / "10_strict_mediation_paths_model3.csv"
BROAD = MODEL_DIR / "12b_antibiotic_ppi_mediation_paths_model3.csv"
PROTOCOL = MODEL_DIR / "12e_protocol_window_mediation_paths_model3.csv"

ROBUST = REPORT_DIR / "12c_final_robustness_summary.csv"
CORE = REPORT_DIR / "13c_12_core_discussion_priority_table.csv"
PROTOCOL_SUMMARY = REPORT_DIR / "12e_protocol_window_comparison_summary.csv"

OUT_A = FINAL_DIR / "Table14A_mediation_overview.csv"
OUT_B = FINAL_DIR / "Table14B_robustness_overview.csv"
OUT_C = FINAL_DIR / "Table14C_12_core_species.csv"
OUT_D = FINAL_DIR / "Table14D_primary_31_paths.csv"
OUT_SUMMARY = FINAL_DIR / "Step14A_final_tables_summary.txt"

KEY = ["diet_score", "cgm_outcome", "species"]


def truthy(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).eq(1)
    return (
        s.astype(str).str.strip().str.lower()
        .isin({"true", "1", "1.0", "yes", "y", "t"})
    )


def read_required(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    return pd.read_csv(path, low_memory=False)


def validate_mediation(df: pd.DataFrame, label: str) -> pd.DataFrame:
    required = {
        "diet_score",
        "cgm_outcome",
        "species",
        "N",
        "a_diet_to_microbiome",
        "b_microbiome_to_cgm",
        "indirect_effect",
        "mediation_FDR05_primary",
        "candidate_mediator_consistent",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            f"{label} mediation file missing columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    out = df.copy()
    for c in KEY:
        out[c] = out[c].astype(str).str.strip()
    if out.duplicated(KEY).any():
        raise RuntimeError(f"{label} mediation file has duplicate exact paths")
    out["mediation_FDR05_primary"] = truthy(out["mediation_FDR05_primary"])
    out["candidate_mediator_consistent"] = truthy(
        out["candidate_mediator_consistent"]
    )
    return out


def label_from_species(species: str) -> str:
    species = str(species)
    return species.split("|s__")[-1] if "|s__" in species else species


def status_of(df: pd.DataFrame) -> pd.Series:
    sig = df["mediation_FDR05_primary"]
    con = df["candidate_mediator_consistent"]
    return np.select(
        [con, sig],
        ["direction_consistent", "significant_inconsistent"],
        default="nonsignificant",
    )


def context_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (score, outcome), g in df.groupby(["diet_score", "cgm_outcome"], sort=True):
        rows.append(
            {
                "diet_score": score,
                "cgm_outcome": outcome,
                "tested_paths": len(g),
                "FDR_significant_paths": int(g["mediation_FDR05_primary"].sum()),
                "direction_consistent_paths": int(
                    g["candidate_mediator_consistent"].sum()
                ),
                "median_model_N": float(
                    pd.to_numeric(g["N"], errors="coerce").median()
                ),
            }
        )
    return pd.DataFrame(rows)


def metric_dict(path: Path) -> dict:
    if not path.is_file():
        return {}
    d = pd.read_csv(path, low_memory=False)
    if not {"metric", "value"}.issubset(d.columns):
        raise RuntimeError(
            f"Expected metric/value schema in {path}; "
            f"found {d.columns.tolist()}"
        )
    return dict(zip(d["metric"].astype(str), d["value"]))


def maybe_number(x):
    if x is None:
        return np.nan
    try:
        return float(x)
    except Exception:
        return x


def main() -> int:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    primary = validate_mediation(read_required(PRIMARY, "Primary"), "Primary")
    strict = validate_mediation(read_required(STRICT, "Strict"), "Strict")
    broad = validate_mediation(read_required(BROAD, "Broad ABX/PPI"), "Broad ABX/PPI")
    protocol = validate_mediation(
        read_required(PROTOCOL, "Protocol-window ABX/PPI"),
        "Protocol-window ABX/PPI",
    )

    # Hard QC: fixed candidate set should be the same. This is pipeline QC only.
    pset = set(map(tuple, primary[KEY].to_numpy()))
    sset = set(map(tuple, strict[KEY].to_numpy()))
    bset = set(map(tuple, broad[KEY].to_numpy()))
    wset = set(map(tuple, protocol[KEY].to_numpy()))
    exact_qc = pset == sset == bset == wset

    # ------------------------------------------------------------------
    # Table A: primary mediation overview by score × CGM context.
    # ------------------------------------------------------------------
    overview = context_summary(primary)
    overview["analysis"] = "Primary Model3 mediation-style analysis"
    overview["interpretation_scope"] = (
        "Cross-sectional indirect association; not causal mediation"
    )
    overview = overview[
        [
            "analysis",
            "diet_score",
            "cgm_outcome",
            "tested_paths",
            "FDR_significant_paths",
            "direction_consistent_paths",
            "median_model_N",
            "interpretation_scope",
        ]
    ]
    overview.to_csv(OUT_A, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # Table B: robustness overview.
    # ------------------------------------------------------------------
    def run_summary(name: str, df: pd.DataFrame) -> dict:
        return {
            "analysis": name,
            "tested_paths": len(df),
            "FDR_significant_paths": int(df["mediation_FDR05_primary"].sum()),
            "direction_consistent_paths": int(
                df["candidate_mediator_consistent"].sum()
            ),
            "median_model_N": float(
                pd.to_numeric(df["N"], errors="coerce").median()
            ),
        }

    rob_rows = [
        run_summary("Primary", primary),
        run_summary("Strict diabetes/A10", strict),
        run_summary("Broad baseline antibiotic/PPI", broad),
        run_summary("Protocol-window antibiotic/PPI", protocol),
    ]
    robustness = pd.DataFrame(rob_rows)

    # Add pairwise effect-stability metrics that are already finalized.
    # Strict and broad values are read from Step12c when available.
    r12c = metric_dict(ROBUST)
    r12e = metric_dict(PROTOCOL_SUMMARY)

    stability_map = {
        "Primary": {
            "indirect_rho_vs_primary": 1.0,
            "indirect_sign_flips_vs_primary": 0,
        },
        "Strict diabetes/A10": {
            "indirect_rho_vs_primary": maybe_number(
                r12c.get("strict_indirect_effect_spearman_rho_vs_primary")
            ),
            "indirect_sign_flips_vs_primary": maybe_number(
                r12c.get("strict_indirect_effect_sign_flips_vs_primary")
            ),
        },
        "Broad baseline antibiotic/PPI": {
            "indirect_rho_vs_primary": maybe_number(
                r12c.get("abxppi_indirect_effect_spearman_rho_vs_primary")
            ),
            "indirect_sign_flips_vs_primary": maybe_number(
                r12c.get("abxppi_indirect_effect_sign_flips_vs_primary")
            ),
        },
        "Protocol-window antibiotic/PPI": {
            "indirect_rho_vs_primary": maybe_number(
                r12e.get("indirect_effect_spearman_rho")
            ),
            "indirect_sign_flips_vs_primary": maybe_number(
                r12e.get("indirect_effect_sign_flips")
            ),
        },
    }

    robustness["indirect_rho_vs_primary"] = robustness["analysis"].map(
        lambda x: stability_map[x]["indirect_rho_vs_primary"]
    )
    robustness["indirect_sign_flips_vs_primary"] = robustness["analysis"].map(
        lambda x: stability_map[x]["indirect_sign_flips_vs_primary"]
    )

    # Compute primary retained counts directly rather than trusting a secondary table.
    primary_sig = set(
        map(
            tuple,
            primary.loc[primary["mediation_FDR05_primary"], KEY].to_numpy(),
        )
    )
    primary_con = set(
        map(
            tuple,
            primary.loc[primary["candidate_mediator_consistent"], KEY].to_numpy(),
        )
    )

    retained_sig = {
        "Primary": len(primary_sig),
        "Strict diabetes/A10": len(
            primary_sig
            & set(
                map(
                    tuple,
                    strict.loc[strict["mediation_FDR05_primary"], KEY].to_numpy(),
                )
            )
        ),
        "Broad baseline antibiotic/PPI": len(
            primary_sig
            & set(
                map(
                    tuple,
                    broad.loc[broad["mediation_FDR05_primary"], KEY].to_numpy(),
                )
            )
        ),
        "Protocol-window antibiotic/PPI": len(
            primary_sig
            & set(
                map(
                    tuple,
                    protocol.loc[
                        protocol["mediation_FDR05_primary"], KEY
                    ].to_numpy(),
                )
            )
        ),
    }
    retained_con = {
        "Primary": len(primary_con),
        "Strict diabetes/A10": len(
            primary_con
            & set(
                map(
                    tuple,
                    strict.loc[
                        strict["candidate_mediator_consistent"], KEY
                    ].to_numpy(),
                )
            )
        ),
        "Broad baseline antibiotic/PPI": len(
            primary_con
            & set(
                map(
                    tuple,
                    broad.loc[
                        broad["candidate_mediator_consistent"], KEY
                    ].to_numpy(),
                )
            )
        ),
        "Protocol-window antibiotic/PPI": len(
            primary_con
            & set(
                map(
                    tuple,
                    protocol.loc[
                        protocol["candidate_mediator_consistent"], KEY
                    ].to_numpy(),
                )
            )
        ),
    }
    robustness["primary_significant_retained"] = robustness["analysis"].map(
        retained_sig
    )
    robustness["primary_direction_consistent_retained"] = robustness[
        "analysis"
    ].map(retained_con)
    robustness["pipeline_qc_exact_fixed_path_set"] = exact_qc
    robustness.to_csv(OUT_B, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # Table C: final 12-core biological interpretation table.
    # ------------------------------------------------------------------
    core = read_required(CORE, "Step13c core priority table")
    required_core = {
        "species",
        "species_label",
        "project_direction_class",
        "biological_evidence_category",
        "discussion_priority_tier",
        "discussion_headline",
        "AMED_a",
        "AMED_b",
        "AMED_indirect",
        "hPDI_a",
        "hPDI_b",
        "hPDI_indirect",
    }
    missing_core = sorted(required_core - set(core.columns))
    if missing_core:
        raise RuntimeError(
            "Step13c core table missing required columns: "
            + ", ".join(missing_core)
            + f"\nAvailable columns: {core.columns.tolist()}"
        )
    if len(core) != 12:
        raise RuntimeError(f"Expected 12 core species; found {len(core)}")

    core_cols = [
        "discussion_priority_tier",
        "species_label",
        "species",
        "project_direction_class",
        "biological_evidence_category",
        "AMED_a",
        "AMED_b",
        "AMED_indirect",
        "hPDI_a",
        "hPDI_b",
        "hPDI_indirect",
        "discussion_headline",
    ]
    core_out = core[core_cols].copy()
    core_out["scope_note"] = (
        "Recurrent AMED/hPDI × glucose_cv core; CLR-Z relative abundance"
    )
    core_out.to_csv(OUT_C, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # Table D: all 31 primary paths with manuscript-facing status.
    # ------------------------------------------------------------------
    primary_out = primary.copy()
    primary_out["species_label"] = primary_out["species"].map(label_from_species)
    primary_out["path_status"] = status_of(primary_out)
    primary_out["a_sign"] = np.sign(
        pd.to_numeric(
            primary_out["a_diet_to_microbiome"], errors="coerce"
        )
    )
    primary_out["b_sign"] = np.sign(
        pd.to_numeric(
            primary_out["b_microbiome_to_cgm"], errors="coerce"
        )
    )
    primary_out["indirect_sign"] = np.sign(
        pd.to_numeric(primary_out["indirect_effect"], errors="coerce")
    )

    preferred_cols = [
        "diet_score",
        "cgm_outcome",
        "species_label",
        "species",
        "N",
        "a_diet_to_microbiome",
        "b_microbiome_to_cgm",
        "indirect_effect",
        "indirect_ci_low",
        "indirect_ci_high",
        "FDR_BH_within_score_outcome",
        "mediation_FDR05_primary",
        "candidate_mediator_consistent",
        "path_status",
        "a_sign",
        "b_sign",
        "indirect_sign",
    ]
    primary_out[
        [c for c in preferred_cols if c in primary_out.columns]
    ].to_csv(OUT_D, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # Human-readable summary.
    # ------------------------------------------------------------------
    primary_core = set(
        core.loc[:, "species"].astype(str)
    )
    if len(primary_core) != 12:
        raise RuntimeError("Core species identity is not unique/complete")

    lines = [
        "=== STEP 14A FINAL TABLE ASSEMBLY ===",
        "NO_MODELS_REFIT=True",
        f"PIPELINE_QC_EXACT_FIXED_PATH_SET={exact_qc}",
        "",
        f"PRIMARY_PATHS={len(primary)}",
        f"PRIMARY_SIGNIFICANT={int(primary['mediation_FDR05_primary'].sum())}",
        f"PRIMARY_DIRECTION_CONSISTENT={int(primary['candidate_mediator_consistent'].sum())}",
        f"CORE_SPECIES={len(primary_core)}",
        "",
        "--- ROBUSTNESS ---",
        robustness.to_string(index=False),
        "",
        "--- PRIMARY CONTEXTS ---",
        overview.to_string(index=False),
        "",
        "OUTPUTS:",
        f"TABLE_A={OUT_A}",
        f"TABLE_B={OUT_B}",
        f"TABLE_C={OUT_C}",
        f"TABLE_D={OUT_D}",
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
