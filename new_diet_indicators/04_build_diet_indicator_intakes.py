"""Aggregate HPP events for EAT-Lancet-13, NOVA4, and carbohydrate energy %."""

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


SCRIPT_VERSION = "2026-09-12-new-diet-indicator-intakes-v2"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR.parent
DEFAULT_INPUT_CSV = DATA_DIR / "Transfer" / "diet_logging" / "diet_logging_events.csv"
DEFAULT_MAPPING_CSV = (
    PROJECT_DIR
    / "outputs"
    / "03_score_mapping"
    / "new_diet_indicators"
    / "new_diet_indicator_food_id_mapping.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "outputs" / "04_score_intakes" / "new_diet_indicators"
)
DEFAULT_COHORT = "10k"
DEFAULT_RESEARCH_STAGE = "00_00_visit"

GROUP_COLUMNS = [
    "participant_id",
    "cohort",
    "research_stage",
    "logging_day",
    "collection_date",
]
TEXT_COLUMNS = ["participant_id", "cohort", "research_stage", "food_id"]
NUMERIC_COLUMNS = [
    "weight_g",
    "calories_kcal",
    "carbohydrate_g",
    "lipid_g",
    "protein_g",
]
REQUIRED_EVENT_COLUMNS = [*GROUP_COLUMNS, "food_id", *NUMERIC_COLUMNS]
REQUIRED_MAPPING_COLUMNS = [
    "food_id",
    "eat_lancet_component",
    "eat_lancet_mapping_status",
    "nova_group",
    "nova_mapping_status",
]
EAT_COMPONENTS = [
    "vegetables",
    "fruits",
    "unsaturated_oils",
    "legumes",
    "nuts",
    "whole_grains",
    "fish",
    "beef_and_lamb",
    "pork",
    "poultry",
    "eggs",
    "dairy",
    "tubers",
]
VALID_EAT_STATUSES = {"mapped", "not_applicable", "unmapped_missing_labels"}
VALID_NOVA_STATUSES = {"mapped", "unmapped_missing_labels", "review_required"}


def normalize_text(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip()
    return normalized.mask(normalized.eq(""), pd.NA)


def validate_columns(columns: List[str], required: List[str], source_name: str) -> None:
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(
            "{}缺少字段：{}".format(source_name, ", ".join(missing))
        )


def load_mapping(path: Path) -> pd.DataFrame:
    if not Path(path).is_file():
        raise FileNotFoundError("找不到新饮食指标映射：{}".format(path))
    mapping = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=REQUIRED_MAPPING_COLUMNS,
        dtype={"food_id": "string"},
        low_memory=False,
    )
    for column in [
        "food_id",
        "eat_lancet_component",
        "eat_lancet_mapping_status",
        "nova_mapping_status",
    ]:
        mapping[column] = normalize_text(mapping[column])
    mapping["nova_group"] = pd.to_numeric(
        mapping["nova_group"], errors="coerce"
    ).astype("Int64")
    if mapping["food_id"].isna().any() or mapping["food_id"].duplicated().any():
        raise ValueError("新饮食指标映射food_id必须非缺失且唯一。")
    invalid_eat = ~mapping["eat_lancet_mapping_status"].isin(VALID_EAT_STATUSES)
    if invalid_eat.any():
        count = int(invalid_eat.sum())
        raise ValueError(
            "EAT-Lancet映射仍有 {:,} 个food_id未审核；先完成第03步override。".format(
                count
            )
        )
    invalid_nova = ~mapping["nova_mapping_status"].isin(VALID_NOVA_STATUSES)
    if invalid_nova.any():
        raise ValueError("NOVA映射包含无效状态。")
    mapped_eat = mapping["eat_lancet_mapping_status"].eq("mapped")
    if (~mapping.loc[mapped_eat, "eat_lancet_component"].isin(EAT_COMPONENTS)).any():
        raise ValueError("EAT-Lancet mapped记录含无效组件。")
    mapped_nova = mapping["nova_mapping_status"].eq("mapped")
    if (~mapping.loc[mapped_nova, "nova_group"].isin([1, 2, 3, 4])).any():
        raise ValueError("NOVA mapped记录含无效NOVA组。")
    return mapping


def aggregate_chunk(
    chunk: pd.DataFrame,
    mapping: pd.DataFrame,
    cohort: str,
    research_stage: str,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    validate_columns(chunk.columns.tolist(), REQUIRED_EVENT_COLUMNS, "饮食事件表")
    chunk = chunk.loc[:, REQUIRED_EVENT_COLUMNS].copy()
    input_rows = len(chunk)
    for column in TEXT_COLUMNS:
        chunk[column] = normalize_text(chunk[column])
    selected = chunk["cohort"].eq(cohort) & chunk["research_stage"].eq(
        research_stage
    )
    chunk = chunk.loc[selected].copy()
    metrics = {
        "input_rows": input_rows,
        "selected_rows": len(chunk),
        "excluded_other_cohort_or_stage_rows": input_rows - len(chunk),
    }
    if chunk.empty:
        return pd.DataFrame(), metrics

    chunk["logging_day"] = pd.to_numeric(
        chunk["logging_day"], errors="coerce"
    ).astype("Int64")
    dates = pd.to_datetime(chunk["collection_date"], errors="coerce")
    chunk["collection_date"] = dates.dt.strftime("%Y-%m-%d")
    missing_keys = chunk[GROUP_COLUMNS].isna().any(axis=1)
    if missing_keys.any():
        raise ValueError(
            "目标事件有 {:,} 行缺少参与者-日键。".format(int(missing_keys.sum()))
        )

    for column in NUMERIC_COLUMNS:
        original = chunk[column]
        numeric = pd.to_numeric(original, errors="coerce")
        metrics["invalid_numeric_{}".format(column)] = int(
            (original.notna() & numeric.isna()).sum()
        )
        chunk[column] = numeric

    mapping_for_merge = mapping.loc[:, REQUIRED_MAPPING_COLUMNS].copy()
    chunk = chunk.merge(
        mapping_for_merge, on="food_id", how="left", validate="many_to_one"
    )
    missing_food_id = chunk["food_id"].isna()
    unmatched_food_id = chunk["eat_lancet_mapping_status"].isna() & ~missing_food_id
    metrics["missing_food_id_rows"] = int(missing_food_id.sum())
    metrics["unmatched_food_id_rows"] = int(unmatched_food_id.sum())
    if missing_food_id.any() or unmatched_food_id.any():
        raise ValueError(
            "目标事件有缺失food_id或未进入第03步映射的food_id：missing={:,}, unmatched={:,}。"
            "请先用同一事件表重建食物字典和新指标映射。".format(
                int(missing_food_id.sum()), int(unmatched_food_id.sum())
            )
        )

    observed_energy = chunk["calories_kcal"].notna() & chunk[
        "calories_kcal"
    ].ge(0)
    macro_columns = ["carbohydrate_g", "lipid_g", "protein_g"]
    valid_macros = pd.DataFrame(index=chunk.index)
    for column in macro_columns:
        valid_macros[column] = chunk[column].where(chunk[column].ge(0))
    macros_complete = valid_macros.notna().all(axis=1)
    formula_energy = (
        4.0 * valid_macros["carbohydrate_g"]
        + 9.0 * valid_macros["lipid_g"]
        + 4.0 * valid_macros["protein_g"]
    )
    formula_energy_rows = ~observed_energy & macros_complete
    chunk["__resolved_energy_kcal"] = chunk["calories_kcal"].where(
        observed_energy
    )
    chunk.loc[formula_energy_rows, "__resolved_energy_kcal"] = formula_energy.loc[
        formula_energy_rows
    ]
    unresolved_energy = chunk["__resolved_energy_kcal"].isna()
    metrics["observed_energy_rows"] = int(observed_energy.sum())
    metrics["formula_energy_rows"] = int(formula_energy_rows.sum())
    metrics["unresolved_energy_rows"] = int(unresolved_energy.sum())

    valid_carb = chunk["carbohydrate_g"].notna() & chunk[
        "carbohydrate_g"
    ].ge(0)
    chunk["__valid_carbohydrate_g"] = chunk["carbohydrate_g"].where(valid_carb)
    metrics["unresolved_carbohydrate_rows"] = int((~valid_carb).sum())

    eat_mapped = chunk["eat_lancet_mapping_status"].eq("mapped").fillna(False)
    valid_weight = chunk["weight_g"].notna() & chunk["weight_g"].ge(0)
    for component in EAT_COMPONENTS:
        component_event = (
            eat_mapped
            & chunk["eat_lancet_component"].eq(component).fillna(False)
        )
        chunk["__{}_mapped_event".format(component)] = component_event.astype(
            "int64"
        )
        chunk["__{}_missing_weight_event".format(component)] = (
            component_event & ~valid_weight
        ).astype("int64")
        chunk["__{}_observed_g".format(component)] = chunk["weight_g"].where(
            component_event & valid_weight, 0.0
        )

    nova_known = (
        chunk["nova_mapping_status"].eq("mapped").fillna(False)
        & chunk["nova_group"].isin([1, 2, 3, 4]).fillna(False)
    )
    nova4 = nova_known & chunk["nova_group"].eq(4).fillna(False)
    chunk["__nova_classified_event"] = nova_known.astype("int64")
    chunk["__nova_classified_missing_energy_event"] = (
        nova_known & unresolved_energy
    ).astype("int64")
    chunk["__nova_review_required_event"] = chunk[
        "nova_mapping_status"
    ].eq("review_required").fillna(False).astype("int64")
    chunk["__nova_unmapped_missing_labels_event"] = chunk[
        "nova_mapping_status"
    ].eq("unmapped_missing_labels").fillna(False).astype("int64")
    chunk["__nova_classified_kcal"] = chunk["__resolved_energy_kcal"].where(
        nova_known, 0.0
    )
    chunk["__nova4_kcal"] = chunk["__resolved_energy_kcal"].where(nova4, 0.0)
    chunk["__resolved_energy_event"] = (~unresolved_energy).astype("int64")
    chunk["__valid_carb_event"] = valid_carb.astype("int64")

    grouped = chunk.groupby(GROUP_COLUMNS, sort=False, dropna=False)
    daily = grouped.size().rename("event_count").to_frame()
    sum_columns = [
        "__resolved_energy_kcal",
        "__valid_carbohydrate_g",
        "__resolved_energy_event",
        "__valid_carb_event",
        "__nova_classified_event",
        "__nova_classified_missing_energy_event",
        "__nova_review_required_event",
        "__nova_unmapped_missing_labels_event",
        "__nova_classified_kcal",
        "__nova4_kcal",
    ]
    for component in EAT_COMPONENTS:
        sum_columns.extend(
            [
                "__{}_mapped_event".format(component),
                "__{}_missing_weight_event".format(component),
                "__{}_observed_g".format(component),
            ]
        )
    sums = grouped[sum_columns].sum(min_count=1)
    daily = daily.join(sums).reset_index()
    return daily, metrics


def combine_daily_partials(partials: List[pd.DataFrame]) -> pd.DataFrame:
    if not partials:
        raise ValueError("筛选后没有可用饮食事件。")
    combined = pd.concat(partials, ignore_index=True)
    grouped = combined.groupby(GROUP_COLUMNS, sort=True, dropna=False)
    sum_columns = [column for column in combined.columns if column.startswith("__")]
    daily = grouped[sum_columns].sum(min_count=1)
    daily["event_count"] = grouped["event_count"].sum()
    daily = daily.reset_index()

    daily = daily.rename(
        columns={
            "__resolved_energy_kcal": "total_resolved_energy_kcal",
            "__valid_carbohydrate_g": "total_observed_carbohydrate_g",
            "__resolved_energy_event": "resolved_energy_event_count",
            "__valid_carb_event": "valid_carbohydrate_event_count",
            "__nova_classified_event": "nova_classified_event_count",
            "__nova_classified_missing_energy_event": "nova_classified_missing_energy_event_count",
            "__nova_review_required_event": "nova_review_required_event_count",
            "__nova_unmapped_missing_labels_event": "nova_unmapped_missing_labels_event_count",
            "__nova_classified_kcal": "nova_classified_energy_kcal",
            "__nova4_kcal": "nova4_energy_kcal",
        }
    )
    daily["energy_complete"] = daily["resolved_energy_event_count"].eq(
        daily["event_count"]
    )
    daily["carbohydrate_complete"] = daily[
        "valid_carbohydrate_event_count"
    ].eq(daily["event_count"])
    positive_energy = daily["total_resolved_energy_kcal"].gt(0)
    daily["nova_mapping_event_coverage"] = (
        daily["nova_classified_event_count"] / daily["event_count"]
    )
    daily["nova_mapping_energy_coverage"] = (
        daily["nova_classified_energy_kcal"]
        / daily["total_resolved_energy_kcal"].where(positive_energy)
    )
    daily["nova4_energy_pct"] = (
        100.0
        * daily["nova4_energy_kcal"]
        / daily["total_resolved_energy_kcal"].where(positive_energy)
    )
    daily.loc[~daily["energy_complete"], "nova4_energy_pct"] = pd.NA
    # User-approved exclusion applies only to NOVA. Preserve the original-day
    # universe and all events for EAT/carbohydrate and exclusion accounting.
    daily["nova_excluded_event_count"] = (
        daily["event_count"] - daily["nova_classified_event_count"]
    )
    daily["nova_excluded_resolved_energy_kcal"] = (
        daily["total_resolved_energy_kcal"] - daily["nova_classified_energy_kcal"]
    )
    daily["nova_excluded_resolved_energy_pct"] = (
        100.0 * daily["nova_excluded_resolved_energy_kcal"]
        / daily["total_resolved_energy_kcal"].where(positive_energy)
    )
    classified_valid = (
        daily["nova_classified_energy_kcal"].gt(0)
        & daily["nova_classified_missing_energy_event_count"].eq(0)
    )
    daily["nova4_classified_energy_pct"] = (
        100.0 * daily["nova4_energy_kcal"]
        / daily["nova_classified_energy_kcal"].where(classified_valid)
    )
    daily["carbohydrate_energy_pct"] = (
        100.0
        * 4.0
        * daily["total_observed_carbohydrate_g"]
        / daily["total_resolved_energy_kcal"].where(positive_energy)
    )
    daily.loc[
        ~(daily["energy_complete"] & daily["carbohydrate_complete"]),
        "carbohydrate_energy_pct",
    ] = pd.NA
    daily["carbohydrate_energy_pct_out_of_range"] = daily[
        "carbohydrate_energy_pct"
    ].notna() & ~daily["carbohydrate_energy_pct"].between(0, 100)

    for component in EAT_COMPONENTS:
        daily = daily.rename(
            columns={
                "__{}_mapped_event".format(component): "{}_mapped_event_count".format(
                    component
                ),
                "__{}_missing_weight_event".format(
                    component
                ): "{}_missing_weight_event_count".format(component),
                "__{}_observed_g".format(component): "{}_g".format(component),
            }
        )
        daily["{}_complete".format(component)] = daily[
            "{}_missing_weight_event_count".format(component)
        ].eq(0)
    return daily.sort_values(GROUP_COLUMNS).reset_index(drop=True)


def build_participant_summary(daily: pd.DataFrame) -> pd.DataFrame:
    keys = ["participant_id", "cohort", "research_stage"]
    grouped = daily.groupby(keys, sort=True, dropna=False)
    participants = grouped.size().rename("observed_diet_days").to_frame()
    participants["mean_daily_energy_kcal"] = grouped[
        "total_resolved_energy_kcal"
    ].mean()
    participants["energy_complete_day_count"] = grouped["energy_complete"].sum()
    participants["all_days_energy_complete"] = grouped["energy_complete"].all()
    for component in EAT_COMPONENTS:
        participants["mean_daily_{}_g".format(component)] = grouped[
            "{}_g".format(component)
        ].mean()
        participants["{}_complete".format(component)] = grouped[
            "{}_complete".format(component)
        ].all()
        participants["{}_mapped_event_count".format(component)] = grouped[
            "{}_mapped_event_count".format(component)
        ].sum()
        participants["{}_missing_weight_event_count".format(component)] = grouped[
            "{}_missing_weight_event_count".format(component)
        ].sum()
    participants["eat_lancet13_intake_complete"] = participants[
        ["{}_complete".format(component) for component in EAT_COMPONENTS]
    ].all(axis=1)

    participants["nova4_valid_day_count"] = grouped["nova4_energy_pct"].count()
    participants["nova4_energy_pct"] = grouped["nova4_energy_pct"].mean()
    participants["nova4_classified_valid_day_count"] = grouped["nova4_classified_energy_pct"].count()
    participants["nova4_classified_energy_pct"] = grouped["nova4_classified_energy_pct"].mean()
    participants["nova_excluded_event_count"] = grouped["nova_excluded_event_count"].sum()
    participants["nova_excluded_resolved_energy_kcal"] = grouped["nova_excluded_resolved_energy_kcal"].sum(min_count=1)
    participants["nova_classified_missing_energy_event_count"] = grouped["nova_classified_missing_energy_event_count"].sum()
    participants["total_resolved_energy_kcal"] = grouped[
        "total_resolved_energy_kcal"
    ].sum(min_count=1)
    participants["total_nova_classified_energy_kcal"] = grouped[
        "nova_classified_energy_kcal"
    ].sum(min_count=1)
    participants["participant_nova_mapping_energy_coverage"] = (
        participants["total_nova_classified_energy_kcal"]
        / participants["total_resolved_energy_kcal"].where(
            participants["total_resolved_energy_kcal"].gt(0)
        )
    )
    participants["mean_nova_mapping_event_coverage"] = grouped[
        "nova_mapping_event_coverage"
    ].mean()
    participants["participant_nova_excluded_resolved_energy_pct"] = (
        100.0 * participants["nova_excluded_resolved_energy_kcal"]
        / participants["total_resolved_energy_kcal"].where(participants["total_resolved_energy_kcal"].gt(0))
    )
    participants["mean_nova_mapping_energy_coverage"] = grouped[
        "nova_mapping_energy_coverage"
    ].mean()
    participants["minimum_daily_nova_mapping_energy_coverage"] = grouped[
        "nova_mapping_energy_coverage"
    ].min()
    participants["nova_review_required_event_count"] = grouped[
        "nova_review_required_event_count"
    ].sum()
    participants["nova_unmapped_missing_labels_event_count"] = grouped[
        "nova_unmapped_missing_labels_event_count"
    ].sum()

    participants["carbohydrate_valid_day_count"] = grouped[
        "carbohydrate_energy_pct"
    ].count()
    participants["carbohydrate_energy_pct"] = grouped[
        "carbohydrate_energy_pct"
    ].mean()
    participants["carbohydrate_pct_out_of_range_day_count"] = grouped[
        "carbohydrate_energy_pct_out_of_range"
    ].sum()
    participants["all_days_carbohydrate_complete"] = grouped[
        "carbohydrate_complete"
    ].all()
    participants = participants.reset_index()
    return participants.sort_values(keys).reset_index(drop=True)


def build_qc(
    daily: pd.DataFrame,
    participants: pd.DataFrame,
    metrics: Dict[str, int],
    cohort: str,
    research_stage: str,
) -> pd.DataFrame:
    rows = [
        ("protocol", "script_version", SCRIPT_VERSION),
        ("protocol", "cohort", cohort),
        ("protocol", "research_stage", research_stage),
        ("protocol", "energy_fallback", "4*carbohydrate+9*fat+4*protein"),
        ("protocol", "nova_analysis_variable", "nova4_classified_energy_pct"),
        ("protocol", "nova_denominator", "classified_food_energy_only; unknowns excluded from NOVA only"),
        ("nova4", "excluded_event_count", int(daily["nova_excluded_event_count"].sum())),
        ("nova4", "excluded_missing_labels_event_count", int(daily["nova_unmapped_missing_labels_event_count"].sum())),
        ("nova4", "excluded_insufficient_evidence_event_count", int(daily["nova_review_required_event_count"].sum())),
        ("nova4", "excluded_resolved_energy_kcal", daily["nova_excluded_resolved_energy_kcal"].sum(min_count=1)),
        ("nova4", "classified_valid_participant_count", int(participants["nova4_classified_energy_pct"].notna().sum())),
        ("overall", "participant_count", len(participants)),
        ("overall", "participant_day_count", len(daily)),
        (
            "eat_lancet",
            "complete_participant_count",
            int(participants["eat_lancet13_intake_complete"].sum()),
        ),
        (
            "nova4",
            "participants_with_valid_percentage",
            int(participants["nova4_energy_pct"].notna().sum()),
        ),
        (
            "nova4",
            "mean_participant_energy_mapping_coverage",
            participants["participant_nova_mapping_energy_coverage"].mean(),
        ),
        (
            "carbohydrate",
            "participants_with_valid_percentage",
            int(participants["carbohydrate_energy_pct"].notna().sum()),
        ),
        (
            "carbohydrate",
            "out_of_range_participant_day_count",
            int(daily["carbohydrate_energy_pct_out_of_range"].sum()),
        ),
    ]
    for key, value in sorted(metrics.items()):
        rows.append(("events", key, value))
    for component in EAT_COMPONENTS:
        rows.append(
            (
                "eat_lancet_component",
                "{}_complete_participant_count".format(component),
                int(participants["{}_complete".format(component)].sum()),
            )
        )
    return pd.DataFrame(rows, columns=["section", "metric", "value"])


def build_diet_indicator_intakes(
    input_csv: Path = DEFAULT_INPUT_CSV,
    mapping_csv: Path = DEFAULT_MAPPING_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    chunk_size: int = 200000,
    cohort: str = DEFAULT_COHORT,
    research_stage: str = DEFAULT_RESEARCH_STAGE,
) -> Tuple[Path, Path, Path]:
    if not Path(input_csv).is_file():
        raise FileNotFoundError("找不到饮食事件表：{}".format(input_csv))
    mapping = load_mapping(mapping_csv)
    header = pd.read_csv(input_csv, encoding="utf-8-sig", nrows=0)
    validate_columns(header.columns.tolist(), REQUIRED_EVENT_COLUMNS, "饮食事件表")
    partials = []
    totals: Dict[str, int] = {}
    for chunk in pd.read_csv(
        input_csv,
        encoding="utf-8-sig",
        usecols=REQUIRED_EVENT_COLUMNS,
        dtype={column: "string" for column in TEXT_COLUMNS},
        chunksize=chunk_size,
        low_memory=False,
    ):
        partial, metrics = aggregate_chunk(chunk, mapping, cohort, research_stage)
        if not partial.empty:
            partials.append(partial)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0) + int(value)
    daily = combine_daily_partials(partials)
    participants = build_participant_summary(daily)
    qc = build_qc(daily, participants, totals, cohort, research_stage)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_path = output_dir / "diet_indicator_daily_intakes.csv"
    participant_path = output_dir / "diet_indicator_participant_intakes.csv"
    qc_path = output_dir / "diet_indicator_intake_qc.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    participants.to_csv(participant_path, index=False, encoding="utf-8-sig")
    qc.to_csv(qc_path, index=False, encoding="utf-8-sig")
    print("SCRIPT_VERSION={}".format(SCRIPT_VERSION))
    print("participant_days={:,}".format(len(daily)))
    print("participants={:,}".format(len(participants)))
    print(
        "eat_lancet_complete_participants={:,}".format(
            int(participants["eat_lancet13_intake_complete"].sum())
        )
    )
    print(
        "mean_nova_energy_mapping_coverage={:.6f}".format(
            float(participants["mean_nova_mapping_energy_coverage"].mean())
        )
    )
    print("NOVA_ANALYSIS_VARIABLE=nova4_classified_energy_pct")
    print("nova_excluded_events={:,}".format(int(daily["nova_excluded_event_count"].sum())))
    print("nova_excluded_missing_labels_events={:,}".format(int(daily["nova_unmapped_missing_labels_event_count"].sum())))
    print("nova_excluded_insufficient_evidence_events={:,}".format(int(daily["nova_review_required_event_count"].sum())))
    total_energy = float(daily["total_resolved_energy_kcal"].sum(min_count=1))
    excluded_energy = float(daily["nova_excluded_resolved_energy_kcal"].sum(min_count=1))
    print("nova_excluded_resolved_energy_pct={:.4f}".format(
        100.0 * excluded_energy / total_energy if total_energy > 0 else float("nan")
    ))
    print("nova_classified_valid_participants={:,}".format(int(participants["nova4_classified_energy_pct"].notna().sum())))
    return daily_path, participant_path, qc_path


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--mapping-csv", type=Path, default=DEFAULT_MAPPING_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=200000)
    parser.add_argument("--cohort", default=DEFAULT_COHORT)
    parser.add_argument("--research-stage", default=DEFAULT_RESEARCH_STAGE)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    for output_path in build_diet_indicator_intakes(
        input_csv=arguments.input_csv,
        mapping_csv=arguments.mapping_csv,
        output_dir=arguments.output_dir,
        chunk_size=arguments.chunk_size,
        cohort=arguments.cohort,
        research_stage=arguments.research_stage,
    ):
        print(output_path)
