#!/usr/bin/env python3
"""Step 15a — pre-analysis audit for the three newly added diet indicators.

READ-ONLY. No association model is fit.

Audits:
1) score distributions and valid N;
2) correlation with the existing four diet scores;
3) NOVA energy-mapping coverage and denominator sensitivity;
4) overlap with microbiome and formal/primary Diet-CGM cohorts;
5) mean daily energy correlations;
6) EAT-Lancet component score distributions.

This step is intended to decide whether each exposure is ready to enter the
existing Diet -> Gut / Diet -> CGM / bridge / mediation pipeline.

Outputs are written to:
    diet_microbiome_glucose_analysis/outputs/reports/new_diet_extension/
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/ec2-user/Desktop/HPP/Data")

NEW_SCORE = (
    ROOT / "diet_deal" / "outputs" / "05_diet_scores"
    / "new_diet_indicators" / "diet_indicator_participant_scores.csv"
)
OLD_SCORE_CANDIDATES = [
    ROOT / "diet_microbiome_analysis" / "data" / "00_current_four_scores_all_participants.csv",
    ROOT / "diet_microbiome_analysis" / "data" / "00_aligned_four_scores.csv",
]
MICRO = ROOT / "gut_microbiome_deal" / "data" / "08_species_clr_zscore.csv"
FORMAL_CGM = (
    ROOT / "diet_microbiome_glucose_analysis" / "outputs" / "data"
    / "00_current4_diet_cgm_cohort.csv"
)

OUT = (
    ROOT / "diet_microbiome_glucose_analysis" / "outputs" / "reports"
    / "new_diet_extension"
)

NEW_VARS = {
    "EAT13": "modified_eat_lancet13_z",
    "NOVA4_classified": "nova4_classified_energy_pct_z",
    "Carbohydrate_pct": "carbohydrate_energy_pct_z",
}
RAW_VARS = {
    "EAT13": "modified_eat_lancet13_raw_0_39",
    "NOVA4_classified": "nova4_classified_energy_pct",
    "Carbohydrate_pct": "carbohydrate_energy_pct",
}
OLD_VARS = ["AHEI_z", "AMED_z", "hPDI_z", "rEDIH_z"]


def norm_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def truthy(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).eq(1)
    return (
        s.astype(str).str.strip().str.lower()
        .isin({"true", "1", "1.0", "yes", "y", "t"})
    )


def require(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def resolve_old_scores() -> Path:
    for p in OLD_SCORE_CANDIDATES:
        if p.is_file():
            d = pd.read_csv(p, nrows=2, low_memory=False)
            if {"participant_id", *OLD_VARS}.issubset(d.columns):
                return p
    raise FileNotFoundError(
        "Could not find a current-four-score table containing "
        f"{OLD_VARS}. Tried: {OLD_SCORE_CANDIDATES}"
    )


def numeric_summary(name: str, s: pd.Series) -> dict:
    x = pd.to_numeric(s, errors="coerce")
    v = x.dropna()
    out = {
        "indicator": name,
        "N_total_rows": len(x),
        "N_valid": int(v.size),
        "missing": int(x.isna().sum()),
    }
    if len(v):
        q = v.quantile([0, .01, .05, .25, .50, .75, .95, .99, 1])
        out.update({
            "min": float(q.loc[0.00]),
            "p01": float(q.loc[0.01]),
            "p05": float(q.loc[0.05]),
            "p25": float(q.loc[0.25]),
            "median": float(q.loc[0.50]),
            "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else np.nan,
            "p75": float(q.loc[0.75]),
            "p95": float(q.loc[0.95]),
            "p99": float(q.loc[0.99]),
            "max": float(q.loc[1.00]),
        })
    return out


def spearman_pair(x: pd.Series, y: pd.Series) -> tuple[int, float]:
    a = pd.to_numeric(x, errors="coerce")
    b = pd.to_numeric(y, errors="coerce")
    ok = a.notna() & b.notna()
    if ok.sum() < 3:
        return int(ok.sum()), np.nan
    rho = a.loc[ok].rank(method="average").corr(
        b.loc[ok].rank(method="average"), method="pearson"
    )
    return int(ok.sum()), float(rho)


def id_set(path: Path, label: str) -> set[str]:
    require(path, label)
    d = pd.read_csv(path, usecols=["participant_id"], low_memory=False)
    return set(norm_id(d["participant_id"]).dropna())


def main() -> int:
    require(NEW_SCORE, "new diet score file")
    require(MICRO, "microbiome file")
    require(FORMAL_CGM, "formal Diet-CGM cohort")
    old_path = resolve_old_scores()

    OUT.mkdir(parents=True, exist_ok=True)

    new = pd.read_csv(NEW_SCORE, low_memory=False)
    old = pd.read_csv(old_path, low_memory=False)

    for label, d in [("new", new), ("old", old)]:
        if "participant_id" not in d.columns:
            raise RuntimeError(f"{label} table has no participant_id")
        d["participant_id"] = norm_id(d["participant_id"])
        if d["participant_id"].duplicated().any():
            raise RuntimeError(f"{label} table has duplicate participant_id")

    missing_new = sorted(
        set(NEW_VARS.values()) | set(RAW_VARS.values()) - set(new.columns)
    )
    # Python set precedence is not visually obvious; re-check explicitly.
    required_new = set(NEW_VARS.values()) | set(RAW_VARS.values())
    missing_new = sorted(required_new - set(new.columns))
    if missing_new:
        raise RuntimeError(
            f"New score table missing required columns: {missing_new}"
        )

    missing_old = sorted(set(OLD_VARS) - set(old.columns))
    if missing_old:
        raise RuntimeError(
            f"Old score table missing required columns: {missing_old}"
        )

    # ------------------------------------------------------------
    # 1. Basic score summaries
    # ------------------------------------------------------------
    rows = []
    for name, col in RAW_VARS.items():
        rows.append(numeric_summary(name, new[col]))
    basic = pd.DataFrame(rows)
    basic.to_csv(OUT / "15a_new_indicator_basic_summary.csv", index=False)

    # ------------------------------------------------------------
    # 2. Correlations: new vs old, new vs new, new vs energy
    # ------------------------------------------------------------
    merged = new.merge(
        old[["participant_id", *OLD_VARS]],
        on="participant_id",
        how="left",
        validate="one_to_one",
    )

    corr_rows = []
    all_new = list(NEW_VARS.items())

    for new_name, new_col in all_new:
        for old_col in OLD_VARS:
            n, rho = spearman_pair(merged[new_col], merged[old_col])
            corr_rows.append({
                "variable_1": new_name,
                "variable_2": old_col,
                "N_pair": n,
                "spearman_rho": rho,
                "comparison_type": "new_vs_existing",
            })

    for i in range(len(all_new)):
        for j in range(i + 1, len(all_new)):
            n1, c1 = all_new[i]
            n2, c2 = all_new[j]
            n, rho = spearman_pair(merged[c1], merged[c2])
            corr_rows.append({
                "variable_1": n1,
                "variable_2": n2,
                "N_pair": n,
                "spearman_rho": rho,
                "comparison_type": "new_vs_new",
            })

    if "mean_daily_energy_kcal" in new.columns:
        for new_name, new_col in all_new:
            n, rho = spearman_pair(
                new[new_col], new["mean_daily_energy_kcal"]
            )
            corr_rows.append({
                "variable_1": new_name,
                "variable_2": "mean_daily_energy_kcal",
                "N_pair": n,
                "spearman_rho": rho,
                "comparison_type": "new_vs_energy",
            })

    correlations = pd.DataFrame(corr_rows)
    correlations.to_csv(
        OUT / "15a_new_indicator_correlations.csv", index=False
    )

    # ------------------------------------------------------------
    # 3. NOVA coverage audit
    # ------------------------------------------------------------
    coverage_col = "participant_nova_mapping_energy_coverage"
    if coverage_col not in new.columns:
        raise RuntimeError(
            f"NOVA audit requires {coverage_col}, but it is absent."
        )

    cov = pd.to_numeric(new[coverage_col], errors="coerce")
    nova_qc = []
    q = cov.dropna().quantile([0, .01, .05, .10, .25, .50, .75, .90, .95, .99, 1])
    for quantile, val in q.items():
        nova_qc.append({
            "section": "coverage_quantile",
            "metric": f"q{quantile:g}",
            "value": float(val),
        })

    for threshold in [0.50, 0.70, 0.80, 0.90, 0.95, 0.98]:
        nova_qc.append({
            "section": "coverage_threshold",
            "metric": f"N_coverage_ge_{threshold:.2f}",
            "value": int(cov.ge(threshold).sum()),
        })
        nova_qc.append({
            "section": "coverage_threshold",
            "metric": f"fraction_coverage_ge_{threshold:.2f}",
            "value": float(cov.ge(threshold).mean()),
        })

    # Compare classified-energy denominator with total-resolved-energy denominator
    # if the diagnostic lower-bound variable was retained.
    if "nova4_energy_pct" in new.columns:
        classified = pd.to_numeric(
            new["nova4_classified_energy_pct"], errors="coerce"
        )
        totalden = pd.to_numeric(new["nova4_energy_pct"], errors="coerce")
        n, rho = spearman_pair(classified, totalden)
        nova_qc += [
            {
                "section": "denominator_comparison",
                "metric": "N_classified_vs_totalden_pair",
                "value": n,
            },
            {
                "section": "denominator_comparison",
                "metric": "spearman_classified_vs_totalden",
                "value": rho,
            },
        ]
        for threshold in [0.80, 0.90, 0.95]:
            use = cov.ge(threshold) & classified.notna() & totalden.notna()
            if use.sum() >= 3:
                n2, rho2 = spearman_pair(
                    classified.loc[use], totalden.loc[use]
                )
                diff = classified.loc[use] - totalden.loc[use]
                nova_qc += [
                    {
                        "section": "denominator_comparison",
                        "metric": f"rho_at_coverage_ge_{threshold:.2f}",
                        "value": rho2,
                    },
                    {
                        "section": "denominator_comparison",
                        "metric": f"median_pctpoint_difference_at_coverage_ge_{threshold:.2f}",
                        "value": float(diff.median()),
                    },
                ]

    pd.DataFrame(nova_qc).to_csv(
        OUT / "15a_nova_coverage_qc.csv", index=False
    )

    # ------------------------------------------------------------
    # 4. EAT component score distributions
    # ------------------------------------------------------------
    component_cols = [
        c for c in new.columns if c.endswith("_score_0_3")
    ]
    comp_rows = []
    for col in component_cols:
        s = pd.to_numeric(new[col], errors="coerce")
        for level in [0, 1, 2, 3]:
            comp_rows.append({
                "component_score_column": col,
                "score_level": level,
                "N": int(s.eq(level).sum()),
                "fraction_among_nonmissing": (
                    float(s.eq(level).sum() / s.notna().sum())
                    if s.notna().sum() else np.nan
                ),
            })
    pd.DataFrame(comp_rows).to_csv(
        OUT / "15a_eat_component_score_distribution.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # 5. Analysis overlap counts
    # ------------------------------------------------------------
    micro_ids = id_set(MICRO, "microbiome")
    formal = pd.read_csv(FORMAL_CGM, low_memory=False)
    if "participant_id" not in formal.columns:
        raise RuntimeError("Formal Diet-CGM cohort has no participant_id")
    formal["participant_id"] = norm_id(formal["participant_id"])
    formal_ids = set(formal["participant_id"])

    primary_ids = set()
    if "primary_cgm_analysis_eligible" in formal.columns:
        keep = truthy(formal["primary_cgm_analysis_eligible"])
        primary_ids = set(formal.loc[keep, "participant_id"])

    overlap_rows = []
    for name, col in NEW_VARS.items():
        valid_ids = set(
            new.loc[
                pd.to_numeric(new[col], errors="coerce").notna(),
                "participant_id",
            ]
        )
        overlap_rows.append({
            "indicator": name,
            "valid_score_N": len(valid_ids),
            "with_microbiome_N": len(valid_ids & micro_ids),
            "with_formal_diet_cgm_N": len(valid_ids & formal_ids),
            "with_primary_diet_cgm_N": (
                len(valid_ids & primary_ids) if primary_ids else np.nan
            ),
        })

    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(OUT / "15a_analysis_overlap_counts.csv", index=False)

    # ------------------------------------------------------------
    # 6. Human-readable summary
    # ------------------------------------------------------------
    novacov = pd.to_numeric(new[coverage_col], errors="coerce")
    lines = [
        "=== STEP 15a NEW DIET INDICATOR PRE-ANALYSIS AUDIT ===",
        f"NEW_SCORE_FILE={NEW_SCORE}",
        f"OLD_SCORE_FILE={old_path}",
        "",
        "--- BASIC SUMMARY ---",
        basic.to_string(index=False),
        "",
        "--- ANALYSIS OVERLAP ---",
        overlap.to_string(index=False),
        "",
        "--- NEW vs EXISTING SCORE CORRELATIONS ---",
        correlations.loc[
            correlations["comparison_type"].eq("new_vs_existing")
        ].to_string(index=False),
        "",
        "--- NEW vs NEW / ENERGY CORRELATIONS ---",
        correlations.loc[
            ~correlations["comparison_type"].eq("new_vs_existing")
        ].to_string(index=False),
        "",
        "--- NOVA MAPPING COVERAGE ---",
        f"N with coverage: {int(novacov.notna().sum())}",
        f"median coverage: {float(novacov.median()):.4f}",
        f"N >=80%: {int(novacov.ge(.80).sum())}",
        f"N >=90%: {int(novacov.ge(.90).sum())}",
        f"N >=95%: {int(novacov.ge(.95).sum())}",
        "",
        "INTERPRETATION:",
        "- Do not run downstream models before reviewing NOVA coverage.",
        "- EAT-Lancet must remain labelled modified EAT-Lancet-13.",
        "- Carbohydrate % is an exploratory macronutrient exposure, not a diet-quality score.",
        "- New-score models should include mean daily total energy as a prespecified covariate.",
    ]
    summary_path = OUT / "15a_new_diet_indicator_preanalysis_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSUMMARY={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
