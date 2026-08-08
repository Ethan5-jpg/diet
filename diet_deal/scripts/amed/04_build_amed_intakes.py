"""将饮食事件汇总为 AMED 日级和参与者级成分摄入量。"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR.parent
DEFAULT_INPUT_CSV = (
    DATA_DIR / "Transfer" / "diet_logging" / "diet_logging_events.csv"
)
DEFAULT_MAPPING_CSV = (
    PROJECT_DIR
    / "outputs"
    / "03_score_mapping"
    / "amed"
    / "amed_food_id_mapping.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "outputs" / "04_score_intakes" / "amed"
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
TEXT_COLUMNS = [
    "participant_id",
    "cohort",
    "research_stage",
    "food_id",
    "food_category",
]
NUMERIC_EVENT_COLUMNS = [
    "weight_g",
    "calories_kcal",
    "carbohydrate_g",
    "lipid_g",
    "protein_g",
    "alcohol_g",
]
REQUIRED_EVENT_COLUMNS = [
    *GROUP_COLUMNS,
    "food_id",
    "food_category",
    *NUMERIC_EVENT_COLUMNS,
]
REQUIRED_MAPPING_COLUMNS = [
    "food_id",
    "amed_component",
    "mapping_status",
]

AMED_COMPONENTS = [
    "fruit",
    "vegetables",
    "whole_grains",
    "nuts",
    "legumes",
    "fish",
    "red_processed_meat",
]
VALID_MAPPING_STATUSES = {
    "mapped",
    "not_applicable",
    "unmapped_missing_labels",
}
EVENT_STATUSES = [
    "mapped",
    "not_applicable",
    "unmapped_missing_labels",
    "unmatched_food_id",
    "missing_food_id",
]
ALCOHOLIC_DRINK_CATEGORY = "Alcoholic Drinks"


def normalize_text(series: pd.Series) -> pd.Series:
    """去除文本首尾空白，并将空字符串视为缺失。"""
    normalized = series.astype("string").str.strip()
    return normalized.mask(normalized.eq(""), pd.NA)


def validate_columns(
    columns: List[str], required: List[str], source_name: str
) -> None:
    """确认输入表包含所需字段。"""
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(
            f"{source_name}缺少字段：{', '.join(missing)}"
        )


def load_mapping(mapping_csv: Path) -> pd.DataFrame:
    """读取并验证已完成审核的 AMED food_id 映射。"""
    if not mapping_csv.is_file():
        raise FileNotFoundError(f"找不到 AMED 映射表：{mapping_csv}")

    header = pd.read_csv(mapping_csv, encoding="utf-8-sig", nrows=0)
    validate_columns(
        header.columns.tolist(), REQUIRED_MAPPING_COLUMNS, "AMED 映射表"
    )
    mapping = pd.read_csv(
        mapping_csv,
        encoding="utf-8-sig",
        usecols=REQUIRED_MAPPING_COLUMNS,
        dtype={"food_id": "string"},
        low_memory=False,
    )
    for column in REQUIRED_MAPPING_COLUMNS:
        mapping[column] = normalize_text(mapping[column])

    missing_food_ids = int(mapping["food_id"].isna().sum())
    if missing_food_ids:
        raise ValueError(
            f"AMED 映射表有 {missing_food_ids:,} 行缺少 food_id。"
        )

    duplicated = mapping["food_id"].duplicated(keep=False)
    if duplicated.any():
        duplicate_ids = sorted(
            mapping.loc[duplicated, "food_id"].astype(str).unique()
        )
        preview = ", ".join(duplicate_ids[:10])
        raise ValueError(f"AMED 映射表存在重复 food_id：{preview}")

    invalid_status = ~mapping["mapping_status"].isin(
        VALID_MAPPING_STATUSES
    )
    if invalid_status.any():
        statuses = sorted(
            mapping.loc[invalid_status, "mapping_status"]
            .fillna("<missing>")
            .astype(str)
            .unique()
        )
        raise ValueError(
            "AMED 映射尚未完成审核或含无效状态："
            + ", ".join(statuses)
        )

    mapped = mapping["mapping_status"].eq("mapped")
    invalid_component = mapped & ~mapping["amed_component"].isin(
        AMED_COMPONENTS
    )
    if invalid_component.any():
        components = sorted(
            mapping.loc[invalid_component, "amed_component"]
            .fillna("<missing>")
            .astype(str)
            .unique()
        )
        raise ValueError(
            "已映射 food_id 含无效 AMED 成分：" + ", ".join(components)
        )

    unexpected_component = ~mapped & mapping["amed_component"].notna()
    if unexpected_component.any():
        raise ValueError("非 mapped 记录不应包含 amed_component。")

    return mapping


def aggregate_chunk(
    chunk: pd.DataFrame,
    mapping: pd.DataFrame,
    cohort: str,
    research_stage: str,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """筛选一个数据块并生成参与者-日级局部汇总。"""
    chunk = chunk.loc[:, REQUIRED_EVENT_COLUMNS].copy()
    input_rows = len(chunk)
    for column in TEXT_COLUMNS:
        chunk[column] = normalize_text(chunk[column])

    selected = chunk["cohort"].eq(cohort) & chunk[
        "research_stage"
    ].eq(research_stage)
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
    collection_dates = pd.to_datetime(
        chunk["collection_date"], errors="coerce"
    )
    chunk["collection_date"] = collection_dates.dt.strftime("%Y-%m-%d")

    missing_key_rows = int(chunk[GROUP_COLUMNS].isna().any(axis=1).sum())
    if missing_key_rows:
        raise ValueError(
            f"筛选后发现 {missing_key_rows:,} 行缺少参与者-日期汇总键。"
        )

    for column in NUMERIC_EVENT_COLUMNS:
        original_nonmissing = chunk[column].notna()
        converted = pd.to_numeric(chunk[column], errors="coerce")
        metrics[f"invalid_numeric_{column}"] = int(
            (original_nonmissing & converted.isna()).sum()
        )
        chunk[column] = converted

    alcoholic_drink = (
        chunk["food_category"]
        .str.casefold()
        .eq(ALCOHOLIC_DRINK_CATEGORY.casefold())
        .fillna(False)
    )
    valid_weight = chunk["weight_g"].notna() & chunk["weight_g"].gt(0)
    observed_alcohol = chunk["alcohol_g"].notna()
    observed_alcohol_valid = (
        alcoholic_drink
        & valid_weight
        & observed_alcohol
        & chunk["alcohol_g"].ge(0)
        & chunk["alcohol_g"].le(chunk["weight_g"])
    )

    energy_fields_complete = chunk[
        ["calories_kcal", "carbohydrate_g", "lipid_g", "protein_g"]
    ].notna().all(axis=1)
    formula_alcohol = (
        chunk["calories_kcal"]
        - 4 * chunk["carbohydrate_g"]
        - 9 * chunk["lipid_g"]
        - 4 * chunk["protein_g"]
    ) / 7
    formula_alcohol_valid = (
        alcoholic_drink
        & valid_weight
        & ~observed_alcohol
        & energy_fields_complete
        & formula_alcohol.ge(0)
        & formula_alcohol.le(chunk["weight_g"])
    )
    alcohol_resolved = observed_alcohol_valid | formula_alcohol_valid
    alcohol_unresolved = alcoholic_drink & ~alcohol_resolved
    alcohol_invalid_observed = (
        alcoholic_drink & observed_alcohol & ~observed_alcohol_valid
    )

    chunk["__alcoholic_drink_event"] = alcoholic_drink.astype("int64")
    chunk["__alcohol_observed_event"] = observed_alcohol_valid.astype(
        "int64"
    )
    chunk["__alcohol_energy_formula_event"] = (
        formula_alcohol_valid.astype("int64")
    )
    chunk["__alcohol_unresolved_event"] = alcohol_unresolved.astype(
        "int64"
    )
    chunk["__alcohol_invalid_observed_event"] = (
        alcohol_invalid_observed.astype("int64")
    )
    chunk["__alcohol_g_resolved"] = chunk["alcohol_g"].where(
        observed_alcohol_valid
    )
    chunk.loc[
        formula_alcohol_valid, "__alcohol_g_resolved"
    ] = formula_alcohol.loc[formula_alcohol_valid]

    chunk = chunk.merge(
        mapping,
        on="food_id",
        how="left",
        validate="many_to_one",
    )
    chunk["event_mapping_status"] = chunk["mapping_status"]
    missing_food_id = chunk["food_id"].isna()
    unmatched_food_id = chunk["mapping_status"].isna() & ~missing_food_id
    chunk.loc[missing_food_id, "event_mapping_status"] = "missing_food_id"
    chunk.loc[
        unmatched_food_id, "event_mapping_status"
    ] = "unmatched_food_id"

    chunk["__food_id_nonmissing"] = chunk["food_id"].notna().astype("int64")
    for status in EVENT_STATUSES:
        mask = chunk["event_mapping_status"].eq(status)
        chunk[f"__{status}_event"] = mask.astype("int64")
        chunk[f"__{status}_weight_g"] = chunk["weight_g"].where(mask)

    for component in AMED_COMPONENTS:
        mask = (
            chunk["event_mapping_status"].eq("mapped")
            & chunk["amed_component"].eq(component)
        )
        chunk[f"__{component}_event"] = mask.astype("int64")
        chunk[f"__{component}_weight_g"] = chunk["weight_g"].where(mask)

    grouped = chunk.groupby(GROUP_COLUMNS, sort=False, dropna=False)
    daily = grouped.size().rename("event_count").to_frame()
    daily["food_id_nonmissing_events"] = grouped[
        "__food_id_nonmissing"
    ].sum()
    daily["weight_g_total_all_foods"] = grouped["weight_g"].sum(
        min_count=1
    )
    daily["weight_g_nonmissing_events"] = grouped["weight_g"].count()
    daily["alcoholic_drink_event_count"] = grouped[
        "__alcoholic_drink_event"
    ].sum()
    daily["alcohol_observed_event_count"] = grouped[
        "__alcohol_observed_event"
    ].sum()
    daily["alcohol_energy_formula_event_count"] = grouped[
        "__alcohol_energy_formula_event"
    ].sum()
    daily["alcohol_unresolved_event_count"] = grouped[
        "__alcohol_unresolved_event"
    ].sum()
    daily["alcohol_invalid_observed_event_count"] = grouped[
        "__alcohol_invalid_observed_event"
    ].sum()
    daily["alcohol_g_resolved_total"] = grouped[
        "__alcohol_g_resolved"
    ].sum(min_count=1)

    for status in EVENT_STATUSES:
        daily[f"{status}_event_count"] = grouped[
            f"__{status}_event"
        ].sum()
        daily[f"{status}_weight_g_total"] = grouped[
            f"__{status}_weight_g"
        ].sum(min_count=1)
        daily[f"{status}_weight_g_nonmissing_events"] = grouped[
            f"__{status}_weight_g"
        ].count()

    for component in AMED_COMPONENTS:
        daily[f"{component}_event_count"] = grouped[
            f"__{component}_event"
        ].sum()
        daily[f"{component}_g_total"] = grouped[
            f"__{component}_weight_g"
        ].sum(min_count=1)
        daily[f"{component}_weight_g_nonmissing_events"] = grouped[
            f"__{component}_weight_g"
        ].count()

    metrics["missing_food_id_rows"] = int(missing_food_id.sum())
    metrics["unmatched_food_id_rows"] = int(unmatched_food_id.sum())
    return daily.reset_index(), metrics


def combine_daily_partials(partials: List[pd.DataFrame]) -> pd.DataFrame:
    """合并跨 CSV 数据块的 AMED 日级局部汇总。"""
    combined = pd.concat(partials, ignore_index=True)
    grouped = combined.groupby(GROUP_COLUMNS, sort=True, dropna=False)

    count_columns = [
        "event_count",
        "food_id_nonmissing_events",
        "weight_g_nonmissing_events",
        "alcoholic_drink_event_count",
        "alcohol_observed_event_count",
        "alcohol_energy_formula_event_count",
        "alcohol_unresolved_event_count",
        "alcohol_invalid_observed_event_count",
        *[
            f"{status}_event_count"
            for status in EVENT_STATUSES
        ],
        *[
            f"{status}_weight_g_nonmissing_events"
            for status in EVENT_STATUSES
        ],
        *[
            f"{component}_event_count"
            for component in AMED_COMPONENTS
        ],
        *[
            f"{component}_weight_g_nonmissing_events"
            for component in AMED_COMPONENTS
        ],
    ]
    total_columns = [
        "weight_g_total_all_foods",
        "alcohol_g_resolved_total",
        *[
            f"{status}_weight_g_total"
            for status in EVENT_STATUSES
        ],
        *[
            f"{component}_g_total"
            for component in AMED_COMPONENTS
        ],
    ]

    daily = grouped[count_columns].sum()
    daily = daily.join(grouped[total_columns].sum(min_count=1)).reset_index()

    no_alcoholic_drinks = daily["alcoholic_drink_event_count"].eq(0)
    daily.loc[no_alcoholic_drinks, "alcohol_g_resolved_total"] = 0.0
    daily["alcohol_complete"] = daily[
        "alcohol_unresolved_event_count"
    ].eq(0)
    daily["alcohol_g_total"] = daily["alcohol_g_resolved_total"].where(
        daily["alcohol_complete"]
    )

    for status in EVENT_STATUSES:
        no_events = daily[f"{status}_event_count"].eq(0)
        daily.loc[no_events, f"{status}_weight_g_total"] = 0.0

    for component in AMED_COMPONENTS:
        no_events = daily[f"{component}_event_count"].eq(0)
        daily.loc[no_events, f"{component}_g_total"] = 0.0
        daily[f"{component}_weight_complete"] = daily[
            f"{component}_event_count"
        ].eq(daily[f"{component}_weight_g_nonmissing_events"])

    status_total = daily[
        [f"{status}_event_count" for status in EVENT_STATUSES]
    ].sum(axis=1)
    if not status_total.eq(daily["event_count"]).all():
        raise ValueError("饮食事件映射状态未完整分区。")

    daily["food_id_event_coverage"] = (
        daily["food_id_nonmissing_events"] / daily["event_count"]
    )
    daily["weight_g_event_coverage"] = (
        daily["weight_g_nonmissing_events"] / daily["event_count"]
    )
    resolved_alcohol_events = (
        daily["alcohol_observed_event_count"]
        + daily["alcohol_energy_formula_event_count"]
    )
    alcoholic_event_denominator = daily[
        "alcoholic_drink_event_count"
    ].replace(0, pd.NA)
    alcohol_coverage = resolved_alcohol_events / alcoholic_event_denominator
    daily["alcohol_g_event_coverage"] = alcohol_coverage.where(
        alcoholic_event_denominator.notna(), 1.0
    )
    daily["classified_food_event_share"] = (
        daily["mapped_event_count"]
        + daily["not_applicable_event_count"]
    ) / daily["event_count"]
    daily["unresolved_food_event_share"] = (
        daily["unmapped_missing_labels_event_count"]
        + daily["unmatched_food_id_event_count"]
        + daily["missing_food_id_event_count"]
    ) / daily["event_count"]

    return daily.sort_values(GROUP_COLUMNS).reset_index(drop=True)


def build_participant_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """计算每位参与者在实际记录日上的 AMED 平均摄入量。"""
    participant_keys = ["participant_id", "cohort", "research_stage"]
    daily = daily.copy()
    daily["collection_date_parsed"] = pd.to_datetime(
        daily["collection_date"], errors="raise"
    )
    grouped = daily.groupby(participant_keys, sort=True, dropna=False)

    summary = grouped.size().rename("observed_diet_days").to_frame()
    summary["unique_logging_days"] = grouped["logging_day"].nunique()
    summary["first_collection_date"] = grouped[
        "collection_date_parsed"
    ].min()
    summary["last_collection_date"] = grouped[
        "collection_date_parsed"
    ].max()
    summary["total_food_events"] = grouped["event_count"].sum()
    summary["mean_daily_all_food_weight_g"] = grouped[
        "weight_g_total_all_foods"
    ].mean()
    summary["mean_daily_alcohol_g_resolved_lower_bound"] = grouped[
        "alcohol_g_resolved_total"
    ].mean()
    summary["total_alcoholic_drink_events"] = grouped[
        "alcoholic_drink_event_count"
    ].sum()
    summary["total_alcohol_observed_events"] = grouped[
        "alcohol_observed_event_count"
    ].sum()
    summary["total_alcohol_energy_formula_events"] = grouped[
        "alcohol_energy_formula_event_count"
    ].sum()
    summary["total_alcohol_unresolved_events"] = grouped[
        "alcohol_unresolved_event_count"
    ].sum()
    summary["total_alcohol_invalid_observed_events"] = grouped[
        "alcohol_invalid_observed_event_count"
    ].sum()
    summary["alcohol_complete"] = summary[
        "total_alcohol_unresolved_events"
    ].eq(0)
    summary["mean_daily_alcohol_g"] = summary[
        "mean_daily_alcohol_g_resolved_lower_bound"
    ].where(summary["alcohol_complete"])
    summary["days_with_alcohol_values"] = grouped[
        "alcohol_complete"
    ].sum()
    resolved_alcohol_events = (
        summary["total_alcohol_observed_events"]
        + summary["total_alcohol_energy_formula_events"]
    )
    alcoholic_event_denominator = summary[
        "total_alcoholic_drink_events"
    ].replace(0, pd.NA)
    alcohol_coverage = resolved_alcohol_events / alcoholic_event_denominator
    summary["mean_alcohol_g_event_coverage"] = alcohol_coverage.where(
        alcoholic_event_denominator.notna(), 1.0
    )

    for status in EVENT_STATUSES:
        summary[f"total_{status}_events"] = grouped[
            f"{status}_event_count"
        ].sum()

    summary["classified_food_event_share"] = (
        summary["total_mapped_events"]
        + summary["total_not_applicable_events"]
    ) / summary["total_food_events"]
    summary["unresolved_food_event_share"] = (
        summary["total_unmapped_missing_labels_events"]
        + summary["total_unmatched_food_id_events"]
        + summary["total_missing_food_id_events"]
    ) / summary["total_food_events"]

    for component in AMED_COMPONENTS:
        total_column = f"{component}_g_total"
        event_column = f"{component}_event_count"
        nonmissing_column = f"{component}_weight_g_nonmissing_events"
        summary[f"mean_daily_{component}_g_proxy"] = grouped[
            total_column
        ].mean()
        summary[f"median_daily_{component}_g_proxy"] = grouped[
            total_column
        ].median()
        summary[f"total_{component}_g"] = grouped[total_column].sum(
            min_count=1
        )
        summary[f"total_{component}_events"] = grouped[event_column].sum()
        summary[f"days_with_{component}"] = grouped[event_column].apply(
            lambda values: int(values.gt(0).sum())
        )
        summary[f"days_with_complete_{component}_weight"] = grouped[
            f"{component}_weight_complete"
        ].sum()
        component_events = summary[f"total_{component}_events"]
        component_nonmissing = grouped[nonmissing_column].sum()
        summary[f"{component}_weight_event_coverage"] = (
            component_nonmissing / component_events.replace(0, pd.NA)
        )

    summary = summary.reset_index()
    summary["calendar_span_days"] = (
        summary["last_collection_date"]
        - summary["first_collection_date"]
    ).dt.days + 1
    summary["first_collection_date"] = summary[
        "first_collection_date"
    ].dt.strftime("%Y-%m-%d")
    summary["last_collection_date"] = summary[
        "last_collection_date"
    ].dt.strftime("%Y-%m-%d")
    return summary


def build_qc_table(
    daily: pd.DataFrame,
    participant_summary: pd.DataFrame,
    chunk_metrics: Dict[str, int],
    cohort: str,
    research_stage: str,
) -> pd.DataFrame:
    """生成 AMED 成分摄入量汇总的整体质量控制指标。"""
    selected_events = int(daily["event_count"].sum())
    metrics = {
        **chunk_metrics,
        "selected_cohort": cohort,
        "selected_research_stage": research_stage,
        "daily_rows": len(daily),
        "participant_rows": len(participant_summary),
        "unique_participants": participant_summary[
            "participant_id"
        ].nunique(),
        "minimum_collection_date": daily["collection_date"].min(),
        "maximum_collection_date": daily["collection_date"].max(),
        "classified_food_event_share": (
            daily["mapped_event_count"].sum()
            + daily["not_applicable_event_count"].sum()
        )
        / selected_events,
        "unresolved_food_event_share": (
            daily["unmapped_missing_labels_event_count"].sum()
            + daily["unmatched_food_id_event_count"].sum()
            + daily["missing_food_id_event_count"].sum()
        )
        / selected_events,
        "weight_g_event_coverage": daily[
            "weight_g_nonmissing_events"
        ].sum()
        / selected_events,
        "alcoholic_drink_event_count": int(
            daily["alcoholic_drink_event_count"].sum()
        ),
        "alcohol_observed_event_count": int(
            daily["alcohol_observed_event_count"].sum()
        ),
        "alcohol_energy_formula_event_count": int(
            daily["alcohol_energy_formula_event_count"].sum()
        ),
        "alcohol_unresolved_event_count": int(
            daily["alcohol_unresolved_event_count"].sum()
        ),
        "alcohol_invalid_observed_event_count": int(
            daily["alcohol_invalid_observed_event_count"].sum()
        ),
        "alcohol_g_event_coverage": (
            daily["alcohol_observed_event_count"].sum()
            + daily["alcohol_energy_formula_event_count"].sum()
        )
        / max(int(daily["alcoholic_drink_event_count"].sum()), 1),
        "participants_with_complete_alcohol": int(
            participant_summary["alcohol_complete"].sum()
        ),
        "participants_with_incomplete_alcohol": int(
            (~participant_summary["alcohol_complete"]).sum()
        ),
    }
    for status in EVENT_STATUSES:
        event_count = int(daily[f"{status}_event_count"].sum())
        metrics[f"{status}_event_count"] = event_count
        metrics[f"{status}_event_share"] = event_count / selected_events
    for component in AMED_COMPONENTS:
        metrics[f"{component}_event_count"] = int(
            daily[f"{component}_event_count"].sum()
        )
        metrics[f"{component}_weight_g_total"] = daily[
            f"{component}_g_total"
        ].sum(min_count=1)

    return pd.DataFrame(
        {"metric": list(metrics.keys()), "value": list(metrics.values())}
    )


def build_amed_intakes(
    input_csv: Path,
    mapping_csv: Path,
    output_dir: Path,
    chunk_size: int,
    cohort: str,
    research_stage: str,
) -> Tuple[Path, Path, Path]:
    """执行分块合并、日级汇总、参与者级汇总和结果保存。"""
    if not input_csv.is_file():
        raise FileNotFoundError(f"找不到饮食事件表：{input_csv}")

    mapping = load_mapping(mapping_csv)
    header = pd.read_csv(input_csv, encoding="utf-8-sig", nrows=0)
    validate_columns(
        header.columns.tolist(), REQUIRED_EVENT_COLUMNS, "饮食事件表"
    )

    partials = []
    accumulated_metrics: Dict[str, int] = {}
    reader = pd.read_csv(
        input_csv,
        encoding="utf-8-sig",
        usecols=REQUIRED_EVENT_COLUMNS,
        dtype={column: "string" for column in TEXT_COLUMNS},
        chunksize=chunk_size,
        low_memory=False,
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        partial, metrics = aggregate_chunk(
            chunk, mapping, cohort, research_stage
        )
        if not partial.empty:
            partials.append(partial)
        for key, value in metrics.items():
            accumulated_metrics[key] = (
                accumulated_metrics.get(key, 0) + value
            )
        print(
            f"已处理第 {chunk_number} 块，"
            f"累计读取 {accumulated_metrics['input_rows']:,} 行，"
            f"目标事件 {accumulated_metrics['selected_rows']:,} 行"
        )

    if not partials:
        raise ValueError(
            f"未找到 cohort={cohort}、research_stage={research_stage} 的记录。"
        )

    daily = combine_daily_partials(partials)
    unmatched_events = int(daily["unmatched_food_id_event_count"].sum())
    if unmatched_events:
        raise ValueError(
            f"发现 {unmatched_events:,} 条映射表外 food_id 事件；"
            "请先重新运行第02和第03步。"
        )

    participant_summary = build_participant_summary(daily)
    qc = build_qc_table(
        daily,
        participant_summary,
        accumulated_metrics,
        cohort,
        research_stage,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    daily_path = output_dir / "amed_daily_component_intakes.csv"
    participant_path = (
        output_dir / "amed_participant_component_intakes.csv"
    )
    qc_path = output_dir / "amed_intake_qc.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    participant_summary.to_csv(
        participant_path, index=False, encoding="utf-8-sig"
    )
    qc.to_csv(qc_path, index=False, encoding="utf-8-sig")

    print("-" * 72)
    print(f"目标饮食事件：{int(daily['event_count'].sum()):,} 行")
    print(f"日级记录：{len(daily):,} 行")
    print(f"参与者：{len(participant_summary):,} 人")
    print(
        "未解析事件占比："
        f"{daily['unresolved_food_event_share'].mul(daily['event_count']).sum() / daily['event_count'].sum():.3%}"
    )
    print(f"日级结果：{daily_path}")
    print(f"参与者级结果：{participant_path}")
    print(f"QC 结果：{qc_path}")
    return daily_path, participant_path, qc_path


def parse_args():
    parser = ArgumentParser(
        description="汇总 AMED 日级和参与者级成分摄入量"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"饮食事件 CSV（默认：{DEFAULT_INPUT_CSV}）",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_MAPPING_CSV,
        help=f"AMED food_id 映射（默认：{DEFAULT_MAPPING_CSV}）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录（默认：{DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200_000,
        help="每次读取的行数（默认：200000）",
    )
    parser.add_argument(
        "--cohort",
        default=DEFAULT_COHORT,
        help=f"目标队列（默认：{DEFAULT_COHORT}）",
    )
    parser.add_argument(
        "--research-stage",
        default=DEFAULT_RESEARCH_STAGE,
        help=f"目标研究阶段（默认：{DEFAULT_RESEARCH_STAGE}）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_amed_intakes(
        args.input,
        args.mapping,
        args.output_dir,
        args.chunk_size,
        args.cohort,
        args.research_stage,
    )
