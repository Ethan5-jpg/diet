#!/usr/bin/env python3
"""Step 12c — consolidate final mediation robustness across three analyses.

READ-ONLY consolidation only. No models are refit and no canonical outputs are
modified.

Inputs
------
Primary:
    outputs/models/10_mediation_paths_model3.csv
Strict diabetes/A10 sensitivity:
    outputs/models/10_strict_mediation_paths_model3.csv
Antibiotic/PPI-adjusted sensitivity:
    outputs/models/12b_antibiotic_ppi_mediation_paths_model3.csv

Outputs
-------
outputs/reports/
    12c_final_robustness_path_table.csv
    12c_final_robustness_summary.csv
    12c_final_robustness_summary.txt
    12c_recurrent_core_robustness.csv
    12c_primary_significant_not_retained.csv

The script treats the full taxonomy string in `species` as the species ID.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
PROJECT = ROOT / "diet_microbiome_glucose_analysis"
MODEL_DIR = PROJECT / "outputs" / "models"
REPORT_DIR = PROJECT / "outputs" / "reports"

PRIMARY = MODEL_DIR / "10_mediation_paths_model3.csv"
STRICT = MODEL_DIR / "10_strict_mediation_paths_model3.csv"
ABXPPI = MODEL_DIR / "12b_antibiotic_ppi_mediation_paths_model3.csv"

OUT_PATH = REPORT_DIR / "12c_final_robustness_path_table.csv"
OUT_SUMMARY = REPORT_DIR / "12c_final_robustness_summary.csv"
OUT_TXT = REPORT_DIR / "12c_final_robustness_summary.txt"
OUT_CORE = REPORT_DIR / "12c_recurrent_core_robustness.csv"
OUT_LOST = REPORT_DIR / "12c_primary_significant_not_retained.csv"

KEY = ["diet_score", "cgm_outcome", "species"]
BOOL = ["mediation_FDR05_primary", "candidate_mediator_consistent"]
NUM = [
    "N",
    "a_diet_to_microbiome",
    "b_microbiome_to_cgm",
    "indirect_effect",
    "total_effect",
    "FDR_BH_within_score_outcome",
]


def truthy(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).eq(1)
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "1.0", "yes", "y", "t"})
    )


def load(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    df = pd.read_csv(path, low_memory=False)

    required = KEY + BOOL + NUM
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")

    for c in KEY:
        df[c] = df[c].astype(str).str.strip()
    for c in BOOL:
        df[c] = truthy(df[c])
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if df.duplicated(KEY).any():
        raise ValueError(f"{label} has duplicate exact path keys")

    if "species_label" not in df.columns:
        df["species_label"] = (
            df["species"].astype(str).str.split("|s__").str[-1]
        )

    return df


def path_status(df: pd.DataFrame) -> pd.Series:
    sig = df["mediation_FDR05_primary"]
    con = df["candidate_mediator_consistent"]
    return np.select(
        [con, sig],
        ["direction_consistent", "significant_inconsistent"],
        default="nonsignificant",
    )


def spearman(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    ok = x.notna() & y.notna()
    if ok.sum() < 3:
        return np.nan
    xr = x.loc[ok].rank(method="average")
    yr = y.loc[ok].rank(method="average")
    return float(xr.corr(yr, method="pearson"))


def sign_stats(x: pd.Series, y: pd.Series):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    ok = x.notna() & y.notna()
    if not ok.any():
        return np.nan, np.nan
    sx = np.sign(x.loc[ok].to_numpy())
    sy = np.sign(y.loc[ok].to_numpy())
    flips = int((sx * sy < 0).sum())
    concord = float((sx == sy).mean())
    return flips, concord


def keyset(df: pd.DataFrame):
    return set(map(tuple, df[KEY].astype(str).to_numpy()))


def select_for_merge(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = KEY + ["species_label"] + BOOL + NUM
    out = df[cols].copy()
    rename = {
        c: f"{prefix}_{c}"
        for c in cols
        if c not in KEY
    }
    return out.rename(columns=rename)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    p = load(PRIMARY, "PRIMARY")
    s = load(STRICT, "STRICT")
    a = load(ABXPPI, "ABX/PPI")

    p["status"] = path_status(p)
    s["status"] = path_status(s)
    a["status"] = path_status(a)

    pset, sset, aset = keyset(p), keyset(s), keyset(a)
    exact_all = pset == sset == aset

    merged = (
        select_for_merge(p, "primary")
        .merge(
            select_for_merge(s, "strict"),
            on=KEY,
            how="outer",
            validate="one_to_one",
        )
        .merge(
            select_for_merge(a, "abxppi"),
            on=KEY,
            how="outer",
            validate="one_to_one",
        )
    )

    # Add compact status columns.
    for prefix in ["primary", "strict", "abxppi"]:
        sig = merged[f"{prefix}_mediation_FDR05_primary"].fillna(False)
        con = merged[f"{prefix}_candidate_mediator_consistent"].fillna(False)
        merged[f"{prefix}_status"] = np.select(
            [con, sig],
            ["direction_consistent", "significant_inconsistent"],
            default="nonsignificant",
        )

    # Cross-analysis preservation.
    merged["significant_in_all_three"] = (
        merged["primary_mediation_FDR05_primary"].fillna(False)
        & merged["strict_mediation_FDR05_primary"].fillna(False)
        & merged["abxppi_mediation_FDR05_primary"].fillna(False)
    )
    merged["direction_consistent_in_all_three"] = (
        merged["primary_candidate_mediator_consistent"].fillna(False)
        & merged["strict_candidate_mediator_consistent"].fillna(False)
        & merged["abxppi_candidate_mediator_consistent"].fillna(False)
    )

    for variant in ["strict", "abxppi"]:
        for effect in [
            "a_diet_to_microbiome",
            "b_microbiome_to_cgm",
            "indirect_effect",
            "total_effect",
        ]:
            pcol = f"primary_{effect}"
            vcol = f"{variant}_{effect}"
            merged[f"{variant}_{effect}_same_sign_vs_primary"] = (
                np.sign(pd.to_numeric(merged[pcol], errors="coerce"))
                == np.sign(pd.to_numeric(merged[vcol], errors="coerce"))
            )

    merged.to_csv(OUT_PATH, index=False)

    # ------------------------------------------------------------------
    # Summary metrics.
    # ------------------------------------------------------------------
    rows = []

    def add(metric, value):
        rows.append({"metric": metric, "value": value})

    add("primary_paths", len(p))
    add("strict_paths", len(s))
    add("abxppi_paths", len(a))
    add("exact_path_set_match_all_three", exact_all)

    for label, df in [("primary", p), ("strict", s), ("abxppi", a)]:
        add(f"{label}_significant", int(df["mediation_FDR05_primary"].sum()))
        add(
            f"{label}_direction_consistent",
            int(df["candidate_mediator_consistent"].sum()),
        )
        add(f"{label}_median_N", float(df["N"].median()))

    p_sig_keys = keyset(p.loc[p["mediation_FDR05_primary"]])
    s_sig_keys = keyset(s.loc[s["mediation_FDR05_primary"]])
    a_sig_keys = keyset(a.loc[a["mediation_FDR05_primary"]])

    p_con_keys = keyset(p.loc[p["candidate_mediator_consistent"]])
    s_con_keys = keyset(s.loc[s["candidate_mediator_consistent"]])
    a_con_keys = keyset(a.loc[a["candidate_mediator_consistent"]])

    add("primary_significant_retained_strict", len(p_sig_keys & s_sig_keys))
    add("primary_significant_retained_abxppi", len(p_sig_keys & a_sig_keys))
    add(
        "primary_significant_retained_both_sensitivities",
        len(p_sig_keys & s_sig_keys & a_sig_keys),
    )

    add("primary_consistent_retained_strict", len(p_con_keys & s_con_keys))
    add("primary_consistent_retained_abxppi", len(p_con_keys & a_con_keys))
    add(
        "primary_consistent_retained_both_sensitivities",
        len(p_con_keys & s_con_keys & a_con_keys),
    )

    for variant, df in [("strict", s), ("abxppi", a)]:
        pair = p.merge(df, on=KEY, suffixes=("_primary", f"_{variant}"))
        for effect in [
            "a_diet_to_microbiome",
            "b_microbiome_to_cgm",
            "indirect_effect",
            "total_effect",
        ]:
            x = pair[f"{effect}_primary"]
            y = pair[f"{effect}_{variant}"]
            rho = spearman(x, y)
            flips, concord = sign_stats(x, y)
            add(f"{variant}_{effect}_spearman_rho_vs_primary", rho)
            add(f"{variant}_{effect}_sign_flips_vs_primary", flips)
            add(
                f"{variant}_{effect}_sign_concordance_vs_primary",
                concord,
            )

    # ------------------------------------------------------------------
    # Shared AMED/hPDI glucose-CV core derived from PRIMARY.
    # ------------------------------------------------------------------
    p_cv = p.loc[
        p["candidate_mediator_consistent"]
        & p["cgm_outcome"].eq("glucose_cv")
        & p["diet_score"].isin(["AMED", "hPDI"])
    ].copy()

    p_amed = set(
        p_cv.loc[p_cv["diet_score"].eq("AMED"), "species"].astype(str)
    )
    p_hpdi = set(
        p_cv.loc[p_cv["diet_score"].eq("hPDI"), "species"].astype(str)
    )
    primary_core = sorted(p_amed & p_hpdi)

    core_rows = []
    for species in primary_core:
        label_rows = p.loc[p["species"].eq(species), "species_label"]
        species_label = (
            str(label_rows.iloc[0]) if len(label_rows) else species
        )
        row = {
            "species": species,
            "species_label": species_label,
            "primary_core": True,
        }

        for variant, df in [
            ("primary", p),
            ("strict", s),
            ("abxppi", a),
        ]:
            both_sig = True
            both_cons = True
            both_present = True

            for score in ["AMED", "hPDI"]:
                hit = df.loc[
                    df["diet_score"].eq(score)
                    & df["cgm_outcome"].eq("glucose_cv")
                    & df["species"].eq(species)
                ]
                if len(hit) != 1:
                    both_present = False
                    both_sig = False
                    both_cons = False
                    row[f"{variant}_{score}_present"] = False
                    row[f"{variant}_{score}_significant"] = False
                    row[f"{variant}_{score}_consistent"] = False
                else:
                    h = hit.iloc[0]
                    row[f"{variant}_{score}_present"] = True
                    row[f"{variant}_{score}_significant"] = bool(
                        h["mediation_FDR05_primary"]
                    )
                    row[f"{variant}_{score}_consistent"] = bool(
                        h["candidate_mediator_consistent"]
                    )
                    row[f"{variant}_{score}_indirect"] = float(
                        h["indirect_effect"]
                    )
                    both_sig &= bool(h["mediation_FDR05_primary"])
                    both_cons &= bool(h["candidate_mediator_consistent"])

            row[f"{variant}_both_scores_present"] = both_present
            row[f"{variant}_both_scores_significant"] = both_sig
            row[f"{variant}_both_scores_consistent"] = both_cons

        row["retained_as_core_in_strict"] = bool(
            row["strict_both_scores_consistent"]
        )
        row["retained_as_core_in_abxppi"] = bool(
            row["abxppi_both_scores_consistent"]
        )
        row["retained_as_core_in_both_sensitivities"] = bool(
            row["retained_as_core_in_strict"]
            and row["retained_as_core_in_abxppi"]
        )
        core_rows.append(row)

    core = pd.DataFrame(core_rows)
    core.to_csv(OUT_CORE, index=False)

    add("primary_recurrent_core_n", len(primary_core))
    add(
        "core_retained_strict_n",
        int(core["retained_as_core_in_strict"].sum()) if len(core) else 0,
    )
    add(
        "core_retained_abxppi_n",
        int(core["retained_as_core_in_abxppi"].sum()) if len(core) else 0,
    )
    add(
        "core_retained_both_sensitivities_n",
        int(
            core["retained_as_core_in_both_sensitivities"].sum()
        )
        if len(core)
        else 0,
    )

    # Primary significant paths that fail at least one sensitivity.
    lost = merged.loc[
        merged["primary_mediation_FDR05_primary"].fillna(False)
        & ~(
            merged["strict_mediation_FDR05_primary"].fillna(False)
            & merged["abxppi_mediation_FDR05_primary"].fillna(False)
        )
    ].copy()
    lost.to_csv(OUT_LOST, index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_SUMMARY, index=False)
    metric = dict(zip(summary["metric"], summary["value"]))

    lines = [
        "=== STEP 12c FINAL MEDIATION ROBUSTNESS CONSOLIDATION ===",
        "NO MODELS REFIT; CANONICAL OUTPUTS MODIFIED=False",
        "",
        f"Exact 31-path set match across all three: {metric['exact_path_set_match_all_three']}",
        "",
        "Significance:",
        f"- Primary: {metric['primary_significant']}/{metric['primary_paths']}",
        f"- Strict diabetes/A10: {metric['strict_significant']}/{metric['strict_paths']}",
        f"- Antibiotic/PPI-adjusted: {metric['abxppi_significant']}/{metric['abxppi_paths']}",
        f"- Primary significant retained in BOTH sensitivities: "
        f"{metric['primary_significant_retained_both_sensitivities']}/"
        f"{metric['primary_significant']}",
        "",
        "Direction-consistent primary story:",
        f"- Primary: {metric['primary_direction_consistent']}",
        f"- Strict: {metric['strict_direction_consistent']}",
        f"- Antibiotic/PPI-adjusted: {metric['abxppi_direction_consistent']}",
        f"- Primary consistent retained in BOTH sensitivities: "
        f"{metric['primary_consistent_retained_both_sensitivities']}/"
        f"{metric['primary_direction_consistent']}",
        "",
        "Recurrent AMED/hPDI glucose-CV core:",
        f"- Primary core: {metric['primary_recurrent_core_n']}",
        f"- Retained strict: {metric['core_retained_strict_n']}",
        f"- Retained antibiotic/PPI-adjusted: {metric['core_retained_abxppi_n']}",
        f"- Retained in BOTH sensitivities: "
        f"{metric['core_retained_both_sensitivities_n']}",
        "",
        "Indirect-effect stability vs Primary:",
        f"- Strict rho: {metric['strict_indirect_effect_spearman_rho_vs_primary']}",
        f"- Strict sign flips: {metric['strict_indirect_effect_sign_flips_vs_primary']}",
        f"- Antibiotic/PPI rho: {metric['abxppi_indirect_effect_spearman_rho_vs_primary']}",
        f"- Antibiotic/PPI sign flips: {metric['abxppi_indirect_effect_sign_flips_vs_primary']}",
        "",
        "Primary significant paths not significant in both sensitivities:",
        f"- N={len(lost)}",
    ]

    if len(lost):
        for _, r in lost.iterrows():
            label = r.get("primary_species_label", r["species"])
            lines.append(
                f"  * {r['diet_score']} -> {label} -> {r['cgm_outcome']}: "
                f"primary={r['primary_status']}, "
                f"strict={r['strict_status']}, "
                f"abxppi={r['abxppi_status']}"
            )

    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== STEP 12c FINAL ROBUSTNESS SUMMARY ===")
    print(f"EXACT_PATH_SET_MATCH_ALL_THREE={exact_all}")
    print(
        "SIGNIFICANT: "
        f"primary={int(p['mediation_FDR05_primary'].sum())}, "
        f"strict={int(s['mediation_FDR05_primary'].sum())}, "
        f"abxppi={int(a['mediation_FDR05_primary'].sum())}"
    )
    print(
        "PRIMARY_SIGNIFICANT_RETAINED_BOTH="
        f"{len(p_sig_keys & s_sig_keys & a_sig_keys)}/{len(p_sig_keys)}"
    )
    print(
        "DIRECTION_CONSISTENT: "
        f"primary={int(p['candidate_mediator_consistent'].sum())}, "
        f"strict={int(s['candidate_mediator_consistent'].sum())}, "
        f"abxppi={int(a['candidate_mediator_consistent'].sum())}"
    )
    print(
        "PRIMARY_CONSISTENT_RETAINED_BOTH="
        f"{len(p_con_keys & s_con_keys & a_con_keys)}/{len(p_con_keys)}"
    )
    print(
        "CORE_RETAINED_BOTH="
        f"{int(core['retained_as_core_in_both_sensitivities'].sum()) if len(core) else 0}/"
        f"{len(primary_core)}"
    )
    print(
        "INDIRECT_RHO_VS_PRIMARY: "
        f"strict={metric['strict_indirect_effect_spearman_rho_vs_primary']:.6f}, "
        f"abxppi={metric['abxppi_indirect_effect_spearman_rho_vs_primary']:.6f}"
    )
    print(
        "INDIRECT_SIGN_FLIPS: "
        f"strict={metric['strict_indirect_effect_sign_flips_vs_primary']}, "
        f"abxppi={metric['abxppi_indirect_effect_sign_flips_vs_primary']}"
    )

    print("\n--- PRIMARY SIGNIFICANT NOT RETAINED IN BOTH ---")
    if len(lost) == 0:
        print("None")
    else:
        display_cols = [
            "diet_score",
            "cgm_outcome",
            "primary_species_label",
            "primary_indirect_effect",
            "strict_indirect_effect",
            "abxppi_indirect_effect",
            "primary_status",
            "strict_status",
            "abxppi_status",
        ]
        print(
            lost[[c for c in display_cols if c in lost.columns]]
            .to_string(index=False)
        )

    print("\nOUTPUTS:")
    print(f"PATH_TABLE={OUT_PATH}")
    print(f"CORE_TABLE={OUT_CORE}")
    print(f"SUMMARY={OUT_TXT}")
    print("CANONICAL_OUTPUTS_MODIFIED=False")


if __name__ == "__main__":
    main()
