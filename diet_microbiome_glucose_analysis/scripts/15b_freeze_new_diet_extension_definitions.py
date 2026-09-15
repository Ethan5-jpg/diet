#!/usr/bin/env python3
"""Step 15b — freeze audit for new diet indicators before downstream models.

READ-ONLY. No Gut/CGM association model is fit and no canonical score file is
modified.

This step resolves the three issues identified after Step 15a:

1) NOVA denominator / mapping-coverage decision
   - compare classified-energy denominator vs total-energy denominator;
   - construct candidate primary NOVA exposure using total-energy denominator
     with >=80% participant energy-mapping coverage;
   - construct >=90% and classified-denominator sensitivity candidates.

2) Carbohydrate N discrepancy
   - explain raw-valid N vs standardized-valid N;
   - identify participants excluded from Z standardization and their QC reason.

3) Historical hPDI mapping issue triage
   - quantify events/participants/weight/energy attached to the exact food IDs
     flagged during the new-indicator review;
   - DO NOT silently rewrite hPDI, because a corrected hPDI-specific mapping
     decision has not yet been frozen.

Outputs
-------
diet_microbiome_glucose_analysis/outputs/reports/new_diet_extension/
    15b_nova_denominator_comparison.csv
    15b_nova_candidate_exposure_counts.csv
    15b_carbohydrate_raw_vs_z_discrepancy.csv
    15b_hpdi_mapping_issue_triage.csv
    15b_new_diet_extension_candidate_master.csv
    15b_new_diet_extension_freeze_summary.txt

Important
---------
The candidate master is an isolated extension file. It does NOT overwrite the
source diet-indicator score file or the old four-score files.
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
NEW_INTAKE = (
    ROOT / "diet_deal" / "outputs" / "04_score_intakes"
    / "new_diet_indicators" / "diet_indicator_participant_intakes.csv"
)
NEW_DAILY = (
    ROOT / "diet_deal" / "outputs" / "04_score_intakes"
    / "new_diet_indicators" / "diet_indicator_daily_intakes.csv"
)
NEW_MAPPING = (
    ROOT / "diet_deal" / "outputs" / "03_score_mapping"
    / "new_diet_indicators" / "new_diet_indicator_food_id_mapping.csv"
)

EVENT_CANDIDATES = [
    ROOT / "Transfer" / "diet_logging" / "diet_logging_events.csv",
    ROOT / "csv" / "diet_logging" / "diet_logging_events.csv",
]
HPDI_SCORE_CANDIDATES = [
    ROOT / "diet_deal" / "outputs" / "05_diet_scores"
    / "hpdi" / "hpdi_participant_scores.csv",
    ROOT / "diet_deal" / "outputs" / "05_diet_scores"
    / "hpdi" / "hpdi_participant_scores" / "hpdi_participant_scores.csv",
]

OUT = (
    ROOT / "diet_microbiome_glucose_analysis" / "outputs" / "reports"
    / "new_diet_extension"
)

COHORT = "10k"
STAGE = "00_00_visit"

# Exact identities documented by the new-indicator pipeline. These are used only
# for impact triage, not to infer a corrected hPDI component.
AFFECTED_FOOD_IDS = {
    "1012933": "fiber_supplement_not_whole_grain",
    "1014956": "boiled_potato_component_misclassification",
    "1009259": "plant_alternative_requires_hpdi_specific_review",
    "1010775": "plant_alternative_requires_hpdi_specific_review",
    "1009011": "plant_alternative_requires_hpdi_specific_review",
    "1011847": "plant_alternative_requires_hpdi_specific_review",
    "1012723": "plant_alternative_requires_hpdi_specific_review",
    "1009092": "plant_alternative_requires_hpdi_specific_review",
    "1013950": "plant_alternative_requires_hpdi_specific_review",
    "1007077": "plant_alternative_requires_hpdi_specific_review",
}


def require(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def resolve_first(paths: list[Path], label: str, required: bool = True) -> Path | None:
    for p in paths:
        if p.is_file():
            return p
    if required:
        raise FileNotFoundError(
            f"{label} not found. Tried:\n" + "\n".join(str(x) for x in paths)
        )
    return None


def norm_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def truthy(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return numeric(s).fillna(0).eq(1)
    return (
        s.astype(str).str.strip().str.lower()
        .isin({"true", "1", "1.0", "yes", "y", "t"})
    )


def standardize(values: pd.Series, eligible: pd.Series) -> tuple[pd.Series, dict]:
    x = numeric(values)
    use = eligible.fillna(False) & x.notna() & np.isfinite(x)
    n = int(use.sum())
    if n < 2:
        raise RuntimeError("Cannot standardize candidate exposure with <2 valid rows")
    mean = float(x.loc[use].mean())
    sd = float(x.loc[use].std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise RuntimeError("Candidate exposure has invalid/zero SD")
    z = pd.Series(np.nan, index=x.index, dtype="float64")
    z.loc[use] = (x.loc[use] - mean) / sd
    return z, {"N": n, "mean": mean, "sd": sd}


def spearman_pair(x: pd.Series, y: pd.Series, mask: pd.Series | None = None) -> tuple[int, float]:
    a = numeric(x)
    b = numeric(y)
    ok = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        ok &= mask.fillna(False)
    n = int(ok.sum())
    if n < 3:
        return n, np.nan
    rho = a.loc[ok].rank(method="average").corr(
        b.loc[ok].rank(method="average"), method="pearson"
    )
    return n, float(rho)


def comparison_stats(
    classified: pd.Series,
    totalden: pd.Series,
    coverage: pd.Series,
    threshold: float | None,
) -> dict:
    c = numeric(classified)
    t = numeric(totalden)
    cov = numeric(coverage)
    mask = c.notna() & t.notna()
    label = "all_pairs"
    if threshold is not None:
        mask &= cov.ge(threshold)
        label = f"coverage_ge_{threshold:.2f}"
    n, rho = spearman_pair(c, t, mask)
    diff = (c - t).loc[mask]
    absdiff = diff.abs()
    return {
        "subset": label,
        "N_pair": n,
        "spearman_rho": rho,
        "median_classified_minus_total_pctpoint": (
            float(diff.median()) if len(diff) else np.nan
        ),
        "p95_abs_difference_pctpoint": (
            float(absdiff.quantile(.95)) if len(absdiff) else np.nan
        ),
        "max_abs_difference_pctpoint": (
            float(absdiff.max()) if len(absdiff) else np.nan
        ),
    }


def main() -> int:
    for p, label in [
        (NEW_SCORE, "new score table"),
        (NEW_INTAKE, "new participant intake table"),
        (NEW_MAPPING, "new indicator mapping"),
    ]:
        require(p, label)

    event_path = resolve_first(EVENT_CANDIDATES, "diet event table", required=True)
    hpdi_score_path = resolve_first(
        HPDI_SCORE_CANDIDATES, "old hPDI participant score table", required=False
    )

    OUT.mkdir(parents=True, exist_ok=True)

    score = pd.read_csv(NEW_SCORE, low_memory=False)
    intake = pd.read_csv(NEW_INTAKE, low_memory=False)

    required_score = {
        "participant_id",
        "modified_eat_lancet13_z",
        "nova4_classified_energy_pct",
        "nova4_classified_energy_pct_z",
        "nova4_energy_pct",
        "participant_nova_mapping_energy_coverage",
        "nova4_classified_valid_day_count",
        "nova4_valid_day_count",
        "carbohydrate_energy_pct",
        "carbohydrate_energy_pct_z",
        "carbohydrate_valid_day_count",
        "carbohydrate_pct_out_of_range_day_count",
        "mean_daily_energy_kcal",
    }
    missing = sorted(required_score - set(score.columns))
    if missing:
        raise RuntimeError(
            f"New score table missing columns: {missing}\n"
            f"Available columns: {score.columns.tolist()}"
        )

    score["participant_id"] = norm_id(score["participant_id"])
    if score["participant_id"].duplicated().any():
        raise RuntimeError("New score table has duplicate participant_id")

    # ============================================================
    # A. NOVA denominator / coverage freeze audit
    # ============================================================
    classified = numeric(score["nova4_classified_energy_pct"])
    totalden = numeric(score["nova4_energy_pct"])
    coverage = numeric(score["participant_nova_mapping_energy_coverage"])
    classified_days = numeric(score["nova4_classified_valid_day_count"])
    totalden_days = numeric(score["nova4_valid_day_count"])

    nova_compare = pd.DataFrame(
        [
            comparison_stats(classified, totalden, coverage, None),
            comparison_stats(classified, totalden, coverage, .80),
            comparison_stats(classified, totalden, coverage, .90),
            comparison_stats(classified, totalden, coverage, .95),
        ]
    )

    # How strongly does denominator inflation depend on mapping coverage?
    diff = classified - totalden
    n_covdiff, rho_covdiff = spearman_pair(coverage, diff)
    n_covabs, rho_covabs = spearman_pair(coverage, diff.abs())
    nova_compare["coverage_vs_difference_N"] = n_covdiff
    nova_compare["coverage_vs_difference_rho"] = rho_covdiff
    nova_compare["coverage_vs_abs_difference_N"] = n_covabs
    nova_compare["coverage_vs_abs_difference_rho"] = rho_covabs
    nova_compare.to_csv(
        OUT / "15b_nova_denominator_comparison.csv", index=False
    )

    # Candidate definitions. Primary is intentionally isolated here; final
    # approval happens after reading this audit.
    total80_eligible = (
        totalden.between(0, 100)
        & totalden_days.gt(0)
        & coverage.ge(.80)
    )
    total90_eligible = (
        totalden.between(0, 100)
        & totalden_days.gt(0)
        & coverage.ge(.90)
    )
    classified80_eligible = (
        classified.between(0, 100)
        & classified_days.gt(0)
        & coverage.ge(.80)
    )

    total80_z, total80_par = standardize(totalden, total80_eligible)
    total90_z, total90_par = standardize(totalden, total90_eligible)
    classified80_z, classified80_par = standardize(
        classified, classified80_eligible
    )

    nova_count_rows = [
        {
            "candidate": "PRIMARY_CANDIDATE_total_energy_denominator_cov80",
            "denominator": "all_resolved_dietary_energy",
            "minimum_participant_mapping_energy_coverage": .80,
            **total80_par,
        },
        {
            "candidate": "SENSITIVITY_total_energy_denominator_cov90",
            "denominator": "all_resolved_dietary_energy",
            "minimum_participant_mapping_energy_coverage": .90,
            **total90_par,
        },
        {
            "candidate": "SENSITIVITY_classified_energy_denominator_cov80",
            "denominator": "classified_food_energy_only",
            "minimum_participant_mapping_energy_coverage": .80,
            **classified80_par,
        },
        {
            "candidate": "LEGACY_CURRENT_classified_energy_denominator_no_threshold",
            "denominator": "classified_food_energy_only",
            "minimum_participant_mapping_energy_coverage": 0.0,
            "N": int(numeric(score["nova4_classified_energy_pct_z"]).notna().sum()),
            "mean": np.nan,
            "sd": np.nan,
        },
    ]
    nova_counts = pd.DataFrame(nova_count_rows)
    nova_counts.to_csv(
        OUT / "15b_nova_candidate_exposure_counts.csv", index=False
    )

    # ============================================================
    # B. Carbohydrate raw-valid vs Z-valid discrepancy
    # ============================================================
    carb_raw = numeric(score["carbohydrate_energy_pct"])
    carb_z = numeric(score["carbohydrate_energy_pct_z"])
    raw_valid = carb_raw.notna() & np.isfinite(carb_raw)
    z_valid = carb_z.notna() & np.isfinite(carb_z)
    raw_not_z = raw_valid & ~z_valid

    carb_detail_cols = [
        "participant_id",
        "carbohydrate_energy_pct",
        "carbohydrate_energy_pct_z",
        "carbohydrate_valid_day_count",
        "carbohydrate_pct_out_of_range_day_count",
    ]
    if "all_days_carbohydrate_complete" in score.columns:
        carb_detail_cols.append("all_days_carbohydrate_complete")

    carb_detail = score.loc[raw_not_z, carb_detail_cols].copy()
    carb_detail["raw_in_0_100"] = numeric(
        carb_detail["carbohydrate_energy_pct"]
    ).between(0, 100)
    carb_detail["exclusion_reason"] = np.select(
        [
            numeric(carb_detail["carbohydrate_pct_out_of_range_day_count"]).gt(0),
            ~carb_detail["raw_in_0_100"],
            numeric(carb_detail["carbohydrate_valid_day_count"]).le(0),
        ],
        [
            "one_or_more_daily_carbohydrate_pct_out_of_range",
            "participant_mean_out_of_0_100",
            "no_valid_carbohydrate_day",
        ],
        default="other_or_unexplained",
    )

    # If daily file is present, add min/max daily ratios for excluded people.
    if NEW_DAILY.is_file() and len(carb_detail):
        daily = pd.read_csv(
            NEW_DAILY,
            usecols=lambda c: c in {
                "participant_id",
                "carbohydrate_energy_pct",
                "carbohydrate_energy_pct_out_of_range",
            },
            low_memory=False,
        )
        if "participant_id" in daily.columns:
            daily["participant_id"] = norm_id(daily["participant_id"])
            dd = (
                daily.loc[daily["participant_id"].isin(carb_detail["participant_id"])]
                .groupby("participant_id", as_index=False)
                .agg(
                    min_daily_carbohydrate_pct=("carbohydrate_energy_pct", "min"),
                    max_daily_carbohydrate_pct=("carbohydrate_energy_pct", "max"),
                    daily_out_of_range_count=(
                        "carbohydrate_energy_pct_out_of_range",
                        lambda s: int(truthy(s).sum()),
                    ),
                )
            )
            carb_detail = carb_detail.merge(
                dd, on="participant_id", how="left", validate="one_to_one"
            )

    carb_detail.to_csv(
        OUT / "15b_carbohydrate_raw_vs_z_discrepancy.csv", index=False
    )
    carb_fully_explained = (
        len(carb_detail) > 0
        and carb_detail["exclusion_reason"]
        .ne("other_or_unexplained")
        .all()
    )

    # ============================================================
    # C. Historical hPDI mapping issue triage
    # ============================================================
    mapping = pd.read_csv(NEW_MAPPING, dtype={"food_id": "string"}, low_memory=False)
    mapping["food_id"] = norm_id(mapping["food_id"])

    map_cols = [
        c for c in [
            "food_id",
            "canonical_short_food_name",
            "canonical_product_name",
            "canonical_food_category",
            "source_hpdi_component",
            "source_hpdi_mapping_status",
            "eat_lancet_component",
            "eat_lancet_mapping_status",
            "food_identity_check",
        ]
        if c in mapping.columns
    ]

    affected_map = mapping.loc[
        mapping["food_id"].isin(AFFECTED_FOOD_IDS),
        map_cols,
    ].copy()
    affected_map["issue_type"] = affected_map["food_id"].map(AFFECTED_FOOD_IDS)

    event_header = pd.read_csv(event_path, nrows=0).columns.tolist()
    wanted_events = [
        c for c in [
            "participant_id",
            "cohort",
            "research_stage",
            "food_id",
            "weight_g",
            "calories_kcal",
        ]
        if c in event_header
    ]
    if not {"participant_id", "food_id"}.issubset(wanted_events):
        raise RuntimeError(
            f"Diet event table lacks participant_id/food_id: {event_path}"
        )

    events = pd.read_csv(
        event_path,
        usecols=wanted_events,
        dtype={"participant_id": "string", "food_id": "string"},
        low_memory=False,
    )
    events["participant_id"] = norm_id(events["participant_id"])
    events["food_id"] = norm_id(events["food_id"])
    if "cohort" in events.columns:
        events = events.loc[events["cohort"].astype(str).eq(COHORT)].copy()
    if "research_stage" in events.columns:
        events = events.loc[
            events["research_stage"].astype(str).eq(STAGE)
        ].copy()

    aff = events.loc[events["food_id"].isin(AFFECTED_FOOD_IDS)].copy()
    if "weight_g" in aff.columns:
        aff["weight_g"] = numeric(aff["weight_g"])
    if "calories_kcal" in aff.columns:
        aff["calories_kcal"] = numeric(aff["calories_kcal"])

    if len(aff):
        agg_spec = {
            "event_count": ("food_id", "size"),
            "participant_count": ("participant_id", "nunique"),
        }
        if "weight_g" in aff.columns:
            agg_spec["weight_g_sum"] = ("weight_g", "sum")
            agg_spec["weight_g_nonmissing"] = ("weight_g", "count")
        if "calories_kcal" in aff.columns:
            agg_spec["calories_kcal_sum"] = ("calories_kcal", "sum")
            agg_spec["calories_kcal_nonmissing"] = ("calories_kcal", "count")
        aff_summary = aff.groupby("food_id", as_index=False).agg(**agg_spec)
    else:
        aff_summary = pd.DataFrame(
            columns=["food_id", "event_count", "participant_count"]
        )

    hpdi_triage = affected_map.merge(
        aff_summary, on="food_id", how="outer", validate="one_to_one"
    )
    hpdi_triage["issue_type"] = hpdi_triage["issue_type"].fillna(
        hpdi_triage["food_id"].map(AFFECTED_FOOD_IDS)
    )

    # Old hPDI score coverage among affected participants, if available.
    affected_participants = set(aff["participant_id"]) if len(aff) else set()
    hpdi_valid_affected = np.nan
    hpdi_score_col = ""
    if hpdi_score_path is not None:
        hpdi = pd.read_csv(hpdi_score_path, low_memory=False)
        if "participant_id" in hpdi.columns:
            hpdi["participant_id"] = norm_id(hpdi["participant_id"])
            for candidate in [
                "hpdi_score_energy_adjusted_z",
                "hPDI_z",
                "hpdi_score_energy_adjusted",
                "hpdi_score_raw_18_90",
            ]:
                if candidate in hpdi.columns:
                    hpdi_score_col = candidate
                    valid_hpdi = set(
                        hpdi.loc[
                            numeric(hpdi[candidate]).notna(),
                            "participant_id",
                        ]
                    )
                    hpdi_valid_affected = len(valid_hpdi & affected_participants)
                    break

    hpdi_triage.to_csv(
        OUT / "15b_hpdi_mapping_issue_triage.csv", index=False
    )

    # ============================================================
    # D. Isolated candidate exposure master for future models
    # ============================================================
    candidate = pd.DataFrame({
        "participant_id": score["participant_id"],
        "modified_eat_lancet13_z": numeric(
            score["modified_eat_lancet13_z"]
        ),
        "nova4_total_energy_pct": totalden,
        "nova4_total_energy_pct_cov80_z": total80_z,
        "nova4_total_energy_pct_cov90_z": total90_z,
        "nova4_classified_energy_pct": classified,
        "nova4_classified_energy_pct_cov80_z": classified80_z,
        "carbohydrate_energy_pct": carb_raw,
        "carbohydrate_energy_pct_z": carb_z,
        "mean_daily_energy_kcal": numeric(score["mean_daily_energy_kcal"]),
        "participant_nova_mapping_energy_coverage": coverage,
    })
    candidate.to_csv(
        OUT / "15b_new_diet_extension_candidate_master.csv", index=False
    )

    # ============================================================
    # E. Summary / freeze status
    # ============================================================
    total_affected_events = int(len(aff))
    total_affected_people = int(len(affected_participants))
    total_events = int(len(events))
    affected_event_fraction = (
        total_affected_events / total_events if total_events else np.nan
    )

    lines = [
        "=== STEP 15b NEW DIET EXTENSION FREEZE AUDIT ===",
        "NO_ASSOCIATION_MODELS_FIT=True",
        "CANONICAL_SCORE_FILES_MODIFIED=False",
        "",
        "--- NOVA DENOMINATOR COMPARISON ---",
        nova_compare.to_string(index=False),
        "",
        "--- NOVA CANDIDATE COUNTS ---",
        nova_counts.to_string(index=False),
        "",
        "PROVISIONAL_NOVA_PRIMARY=",
        "  nova4_total_energy_pct_cov80_z",
        "PROVISIONAL_NOVA_SENSITIVITIES=",
        "  nova4_total_energy_pct_cov90_z",
        "  nova4_classified_energy_pct_cov80_z",
        "",
        "--- CARBOHYDRATE RAW-vs-Z N ---",
        f"RAW_VALID_N={int(raw_valid.sum())}",
        f"Z_VALID_N={int(z_valid.sum())}",
        f"RAW_VALID_BUT_Z_MISSING_N={int(raw_not_z.sum())}",
        f"DISCREPANCY_FULLY_EXPLAINED_BY_RECORDED_QC={carb_fully_explained}",
    ]
    if len(carb_detail):
        lines += [
            "",
            carb_detail.to_string(index=False),
        ]

    lines += [
        "",
        "--- hPDI HISTORICAL MAPPING ISSUE TRIAGE ---",
        f"FLAGGED_EXACT_FOOD_IDS={len(AFFECTED_FOOD_IDS)}",
        f"FLAGGED_IDS_FOUND_IN_NEW_MAPPING={int(hpdi_triage['food_id'].notna().sum())}",
        f"AFFECTED_BASELINE_EVENTS={total_affected_events}",
        f"AFFECTED_BASELINE_PARTICIPANTS={total_affected_people}",
        f"AFFECTED_EVENT_FRACTION={affected_event_fraction:.8f}",
        f"OLD_HPDI_SCORE_FILE={hpdi_score_path or 'NOT_FOUND'}",
        f"OLD_HPDI_SCORE_COLUMN_USED={hpdi_score_col or 'NONE'}",
        f"AFFECTED_PARTICIPANTS_WITH_VALID_OLD_HPDI={hpdi_valid_affected}",
        "",
        "hPDI_DECISION_STATUS=IMPACT_TRIAGE_ONLY",
        "Do not recompute hPDI until hPDI-specific corrections are explicitly frozen.",
        "",
        "--- NEXT DECISION RULE ---",
        "1) If carbohydrate discrepancy is fully explained, keep the existing",
        "   carbohydrate_energy_pct_z (the smaller Z-valid N is the analysis N).",
        "2) Review NOVA denominator agreement and sample counts, then freeze",
        "   primary/sensitivity definitions before Diet->Gut / Diet->CGM.",
        "3) Review hPDI affected-event/participant burden. If non-negligible,",
        "   perform an isolated corrected-hPDI rebuild + old-vs-new score audit",
        "   before deciding whether old hPDI downstream results require rerun.",
        "",
        f"CANDIDATE_MASTER={OUT / '15b_new_diet_extension_candidate_master.csv'}",
    ]

    summary_path = OUT / "15b_new_diet_extension_freeze_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSUMMARY={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
