"""将饮食事件汇总为 rEDIH 日级和参与者级食物组 g/day 摄入量。"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


SCRIPT_VERSION = "2026-08-14-redih-intakes-v3"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR.parent
DEFAULT_INPUT_CSV = DATA_DIR / "Transfer" / "diet_logging" / "diet_logging_events.csv"
DEFAULT_MAPPING_CSV = (
    PROJECT_DIR / "outputs" / "03_score_mapping" / "redih" / "redih_food_id_mapping.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "04_score_intakes" / "redih"
DEFAULT_WEIGHT_OVERRIDES_CSV = (
    DEFAULT_OUTPUT_DIR / "redih_missing_weight_review.csv"
)

DEFAULT_COHORT = "10k"
DEFAULT_RESEARCH_STAGE = "00_00_visit"
DEFAULT_CHUNK_SIZE = 200000
GROUP_COLUMNS = [
    "participant_id",
    "cohort",
    "research_stage",
    "logging_day",
    "collection_date",
]
EVENT_KEY_COLUMNS = [
    "source_event_row",
    "participant_id",
    "cohort",
    "research_stage",
    "collection_timestamp",
    "food_id",
]
TEXT_COLUMNS = [
    "participant_id",
    "cohort",
    "research_stage",
    "collection_timestamp",
    "food_id",
]
REQUIRED_EVENT_COLUMNS = [
    *GROUP_COLUMNS,
    "collection_timestamp",
    "food_id",
    "weight_g",
    "calories_kcal",
]
REQUIRED_MAPPING_COLUMNS = ["food_id", "redih_component", "mapping_status"]
REQUIRED_WEIGHT_OVERRIDE_COLUMNS = [
    *EVENT_KEY_COLUMNS,
    "resolution_status",
    "resolved_weight_g",
]

REDIH_COMPONENTS = [
    "red_meat",
    "low_energy_beverages",
    "cream_soups",
    "processed_meat",
    "poultry",
    "butter",
    "french_fries",
    "other_fish",
    "high_energy_drinks",
    "tomatoes",
    "low_fat_dairy",
    "eggs",
    "wine",
    "coffee",
    "whole_fruits",
    "high_fat_dairy",
    "green_leafy_vegetables",
]
CURRENT_SCORE_COMPONENTS = list(REDIH_COMPONENTS)
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


def normalize_text(series: pd.Series) -> pd.Series:
    """清理字符串并将空字符串转为缺失。"""
    normalized = series.astype("string").str.strip()
    return normalized.mask(normalized.eq(""), pd.NA)


def validate_columns(
    columns: List[str], required: List[str], source_name: str
) -> None:
    """确认输入表包含需要的字段。"""
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError("{}缺少字段：{}".format(source_name, ", ".join(missing)))


def validate_mapping_frame(
    mapping: pd.DataFrame, require_all_components: bool = False
) -> pd.DataFrame:
    """验证 rEDIH 映射已经冻结且每个 food_id 唯一。"""
    validate_columns(
        mapping.columns.tolist(),
        REQUIRED_MAPPING_COLUMNS,
        "rEDIH 映射表",
    )
    data = mapping.loc[:, REQUIRED_MAPPING_COLUMNS].copy()
    for column in REQUIRED_MAPPING_COLUMNS:
        data[column] = normalize_text(data[column])
    if data["food_id"].isna().any():
        raise ValueError("rEDIH 映射表存在缺失 food_id。")
    if data["food_id"].duplicated().any():
        duplicates = sorted(
            data.loc[data["food_id"].duplicated(keep=False), "food_id"]
            .astype(str)
            .unique()
        )
        raise ValueError("rEDIH 映射表存在重复 food_id：" + ", ".join(duplicates[:10]))

    invalid_status = ~data["mapping_status"].isin(VALID_MAPPING_STATUSES)
    if invalid_status.any():
        statuses = sorted(
            data.loc[invalid_status, "mapping_status"]
            .fillna("<missing>")
            .astype(str)
            .unique()
        )
        raise ValueError(
            "rEDIH 映射尚未冻结；请先清零 review_required 或无效状态："
            + ", ".join(statuses)
        )

    mapped = data["mapping_status"].eq("mapped")
    invalid_component = mapped & ~data["redih_component"].isin(REDIH_COMPONENTS)
    if invalid_component.any():
        components = sorted(
            data.loc[invalid_component, "redih_component"]
            .fillna("<missing>")
            .astype(str)
            .unique()
        )
        raise ValueError("已映射 food_id 含无效 rEDIH 组件：" + ", ".join(components))
    if data.loc[~mapped, "redih_component"].notna().any():
        raise ValueError("非 mapped 记录不应包含 redih_component。")

    if require_all_components:
        observed = set(data.loc[mapped, "redih_component"].dropna())
        missing_components = sorted(set(REDIH_COMPONENTS) - observed)
        if missing_components:
            raise ValueError(
                "rEDIH 映射缺少 HPP 可用组件：" + ", ".join(missing_components)
            )
    return data


def load_mapping(mapping_csv: Path) -> pd.DataFrame:
    """读取服务器上已冻结的 rEDIH 映射。"""
    if not mapping_csv.is_file():
        raise FileNotFoundError("找不到 rEDIH 映射表：{}".format(mapping_csv))
    header = pd.read_csv(mapping_csv, encoding="utf-8-sig", nrows=0)
    validate_columns(
        header.columns.tolist(), REQUIRED_MAPPING_COLUMNS, "rEDIH 映射表"
    )
    mapping = pd.read_csv(
        mapping_csv,
        encoding="utf-8-sig",
        usecols=REQUIRED_MAPPING_COLUMNS,
        dtype={"food_id": "string"},
        low_memory=False,
    )
    return validate_mapping_frame(mapping, require_all_components=True)


def load_weight_overrides(weight_overrides_csv: Path) -> pd.DataFrame:
    """读取已批准的事件级重量；pending/unresolved 行不会被应用。"""
    if not weight_overrides_csv.is_file():
        return pd.DataFrame(columns=[*EVENT_KEY_COLUMNS, "resolved_weight_g"])
    header = pd.read_csv(weight_overrides_csv, encoding="utf-8-sig", nrows=0)
    validate_columns(
        header.columns.tolist(),
        REQUIRED_WEIGHT_OVERRIDE_COLUMNS,
        "rEDIH 缺重量复核表",
    )
    overrides = pd.read_csv(
        weight_overrides_csv,
        encoding="utf-8-sig",
        usecols=REQUIRED_WEIGHT_OVERRIDE_COLUMNS,
        dtype={
            column: "string"
            for column in EVENT_KEY_COLUMNS
            if column != "source_event_row"
        },
        low_memory=False,
    )
    for column in EVENT_KEY_COLUMNS:
        if column == "source_event_row":
            overrides[column] = pd.to_numeric(
                overrides[column], errors="coerce"
            ).astype("Int64")
        else:
            overrides[column] = normalize_text(overrides[column])
    overrides["resolution_status"] = normalize_text(
        overrides["resolution_status"]
    ).str.casefold()
    valid_statuses = {"pending", "unresolved", "use_resolved_weight"}
    invalid_status = ~overrides["resolution_status"].isin(valid_statuses)
    if invalid_status.any():
        statuses = sorted(
            overrides.loc[invalid_status, "resolution_status"]
            .fillna("<missing>")
            .astype(str)
            .unique()
        )
        raise ValueError(
            "rEDIH 缺重量复核表含无效 resolution_status："
            + ", ".join(statuses)
        )
    if overrides.duplicated(EVENT_KEY_COLUMNS).any():
        raise ValueError("rEDIH 缺重量复核表存在重复事件键。")
    approved = overrides["resolution_status"].eq("use_resolved_weight")
    resolved = pd.to_numeric(overrides["resolved_weight_g"], errors="coerce")
    invalid_approved = approved & (resolved.isna() | resolved.le(0))
    if invalid_approved.any():
        raise ValueError(
            "resolution_status=use_resolved_weight 但 resolved_weight_g "
            "缺失或非正数。"
        )
    approved_overrides = overrides.loc[
        approved, [*EVENT_KEY_COLUMNS, "resolved_weight_g"]
    ].copy()
    approved_overrides["resolved_weight_g"] = resolved.loc[approved].astype(
        "float64"
    )
    return approved_overrides


def aggregate_chunk(
    chunk: pd.DataFrame,
    mapping: pd.DataFrame,
    cohort: str,
    research_stage: str,
    weight_overrides: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """筛选一个 CSV 数据块并生成参与者-日局部汇总。"""
    validate_columns(chunk.columns.tolist(), REQUIRED_EVENT_COLUMNS, "饮食事件表")
    chunk = chunk.loc[:, REQUIRED_EVENT_COLUMNS].copy()
    input_rows = len(chunk)
    chunk["source_event_row"] = chunk.index.to_series().astype("int64") + 2
    for column in TEXT_COLUMNS:
        chunk[column] = normalize_text(chunk[column])

    selected = chunk["cohort"].eq(cohort) & chunk["research_stage"].eq(
        research_stage
    )
    chunk = chunk.loc[selected].copy()
    metrics: Dict[str, int] = {
        "input_event_rows": input_rows,
        "selected_event_rows": len(chunk),
        "excluded_other_cohort_or_stage_rows": input_rows - len(chunk),
    }
    if chunk.empty:
        return pd.DataFrame(), metrics

    chunk["logging_day"] = pd.to_numeric(
        chunk["logging_day"], errors="coerce"
    ).astype("Int64")
    dates = pd.to_datetime(chunk["collection_date"], errors="coerce")
    chunk["collection_date"] = dates.dt.strftime("%Y-%m-%d")
    missing_group_key = chunk[GROUP_COLUMNS].isna().any(axis=1)
    if missing_group_key.any():
        raise ValueError(
            "目标记录中有 {:,} 行缺少日级汇总键。".format(
                int(missing_group_key.sum())
            )
        )

    original_weight = chunk["weight_g"]
    weight = pd.to_numeric(original_weight, errors="coerce")
    invalid_weight = original_weight.notna() & weight.isna()
    negative_weight = weight.lt(0).fillna(False)
    chunk["weight_g"] = weight.mask(negative_weight)
    metrics["invalid_numeric_weight_g_rows"] = int(invalid_weight.sum())
    metrics["negative_weight_g_rows"] = int(negative_weight.sum())

    if weight_overrides.empty:
        chunk["resolved_weight_g"] = pd.NA
    else:
        chunk = chunk.merge(
            weight_overrides,
            on=EVENT_KEY_COLUMNS,
            how="left",
            validate="one_to_one",
        )
    override_available = chunk["resolved_weight_g"].notna()
    override_applied = chunk["weight_g"].isna() & override_available
    if (chunk["weight_g"].notna() & override_available).any():
        raise ValueError("重量 override 匹配到原始有效重量事件；请重新生成复核表。")
    if override_applied.any():
        chunk.loc[override_applied, "weight_g"] = pd.to_numeric(
            chunk.loc[override_applied, "resolved_weight_g"], errors="raise"
        )
    chunk = chunk.drop(columns=["resolved_weight_g"])
    metrics["weight_override_applied_rows"] = int(override_applied.sum())

    original_calories = chunk["calories_kcal"]
    calories = pd.to_numeric(original_calories, errors="coerce")
    invalid_calories = original_calories.notna() & calories.isna()
    negative_calories = calories.lt(0).fillna(False)
    chunk["calories_kcal"] = calories.mask(negative_calories)
    metrics["invalid_numeric_calories_kcal_rows"] = int(invalid_calories.sum())
    metrics["negative_calories_kcal_rows"] = int(negative_calories.sum())

    chunk = chunk.merge(mapping, on="food_id", how="left", validate="many_to_one")
    chunk["event_mapping_status"] = chunk["mapping_status"]
    missing_food_id = chunk["food_id"].isna()
    unmatched_food_id = chunk["mapping_status"].isna() & ~missing_food_id
    chunk.loc[missing_food_id, "event_mapping_status"] = "missing_food_id"
    chunk.loc[unmatched_food_id, "event_mapping_status"] = "unmatched_food_id"

    # 三标签全缺失事件从计分输入删除，但保留其参与者-日汇总行。
    # 因此它们不进入组件、总重量、总能量和事件覆盖率分母。
    excluded_missing_labels = chunk["event_mapping_status"].eq(
        "unmapped_missing_labels"
    )
    retained_event = ~excluded_missing_labels
    chunk["__excluded_missing_labels_event"] = (
        excluded_missing_labels.astype("int64")
    )
    chunk["__excluded_missing_labels_weight_g"] = chunk["weight_g"].where(
        excluded_missing_labels
    )
    chunk["__retained_event"] = retained_event.astype("int64")
    chunk["__weight_imputed_event"] = (
        retained_event & override_applied
    ).astype("int64")
    chunk["__retained_weight_g"] = chunk["weight_g"].where(retained_event)
    chunk["__retained_calories_kcal"] = chunk["calories_kcal"].where(
        retained_event
    )
    chunk["__food_id_nonmissing"] = (
        retained_event & chunk["food_id"].notna()
    ).astype("int64")
    chunk["__weight_nonmissing"] = (
        retained_event & chunk["weight_g"].notna()
    ).astype("int64")
    chunk["__calories_nonmissing"] = (
        retained_event & chunk["calories_kcal"].notna()
    ).astype("int64")
    for status in EVENT_STATUSES:
        status_mask = retained_event & chunk["event_mapping_status"].eq(status)
        chunk["__{}_event".format(status)] = status_mask.astype("int64")

    for component in REDIH_COMPONENTS:
        mask = (
            retained_event
            & chunk["event_mapping_status"].eq("mapped")
            & chunk["redih_component"].eq(component)
        )
        chunk["__{}_event".format(component)] = mask.astype("int64")
        chunk["__{}_valid_weight_event".format(component)] = (
            mask & chunk["weight_g"].notna()
        ).astype("int64")
        chunk["__{}_weight_g".format(component)] = chunk["weight_g"].where(mask)

    grouped = chunk.groupby(GROUP_COLUMNS, sort=False, dropna=False)
    daily = grouped["__retained_event"].sum().rename("event_count").to_frame()
    daily["weight_imputed_event_count"] = grouped[
        "__weight_imputed_event"
    ].sum()
    daily["excluded_missing_labels_event_count"] = grouped[
        "__excluded_missing_labels_event"
    ].sum()
    daily["food_id_nonmissing_events"] = grouped["__food_id_nonmissing"].sum()
    daily["weight_g_nonmissing_events"] = grouped["__weight_nonmissing"].sum()
    daily["calories_kcal_nonmissing_events"] = grouped[
        "__calories_nonmissing"
    ].sum()
    daily["weight_g_total_all_foods"] = grouped["__retained_weight_g"].sum(
        min_count=1
    )
    daily["calories_kcal_total"] = grouped["__retained_calories_kcal"].sum(
        min_count=1
    )
    daily["excluded_missing_labels_weight_g_total"] = grouped[
        "__excluded_missing_labels_weight_g"
    ].sum(min_count=1)
    for status in EVENT_STATUSES:
        daily["{}_event_count".format(status)] = grouped[
            "__{}_event".format(status)
        ].sum()
    for component in REDIH_COMPONENTS:
        daily["{}_event_count".format(component)] = grouped[
            "__{}_event".format(component)
        ].sum()
        daily["{}_valid_weight_event_count".format(component)] = grouped[
            "__{}_valid_weight_event".format(component)
        ].sum()
        daily["{}_g_total".format(component)] = grouped[
            "__{}_weight_g".format(component)
        ].sum(min_count=1)

    metrics["missing_food_id_event_count"] = int(missing_food_id.sum())
    metrics["unmatched_food_id_event_count"] = int(unmatched_food_id.sum())
    metrics["excluded_missing_labels_rows"] = int(
        excluded_missing_labels.sum()
    )
    metrics["retained_selected_rows"] = int(retained_event.sum())
    return daily.reset_index(), metrics


def combine_daily_partials(partials: List[pd.DataFrame]) -> pd.DataFrame:
    """合并跨 CSV 数据块的日级局部汇总。"""
    if not partials:
        raise ValueError("目标 cohort 和 research_stage 没有饮食记录。")
    combined = pd.concat(partials, ignore_index=True)
    grouped = combined.groupby(GROUP_COLUMNS, sort=True, dropna=False)
    count_columns = [
        "event_count",
        "weight_imputed_event_count",
        "excluded_missing_labels_event_count",
        "food_id_nonmissing_events",
        "weight_g_nonmissing_events",
        "calories_kcal_nonmissing_events",
        *["{}_event_count".format(status) for status in EVENT_STATUSES],
        *["{}_event_count".format(component) for component in REDIH_COMPONENTS],
        *[
            "{}_valid_weight_event_count".format(component)
            for component in REDIH_COMPONENTS
        ],
    ]
    total_columns = [
        "weight_g_total_all_foods",
        "calories_kcal_total",
        "excluded_missing_labels_weight_g_total",
        *["{}_g_total".format(component) for component in REDIH_COMPONENTS],
    ]
    daily = grouped[count_columns].sum()
    daily = daily.join(grouped[total_columns].sum(min_count=1)).reset_index()

    no_retained_events = daily["event_count"].eq(0)
    daily.loc[no_retained_events, "weight_g_total_all_foods"] = 0.0
    daily.loc[no_retained_events, "calories_kcal_total"] = 0.0
    no_excluded_events = daily["excluded_missing_labels_event_count"].eq(0)
    daily.loc[
        no_excluded_events, "excluded_missing_labels_weight_g_total"
    ] = 0.0

    for component in REDIH_COMPONENTS:
        events = daily["{}_event_count".format(component)]
        valid_events = daily["{}_valid_weight_event_count".format(component)]
        daily["{}_missing_weight_event_count".format(component)] = (
            events - valid_events
        )
        daily["{}_g_total".format(component)] = pd.to_numeric(
            daily["{}_g_total".format(component)], errors="coerce"
        ).fillna(0.0)
        daily["{}_weight_complete".format(component)] = events.eq(valid_events)

    status_columns = ["{}_event_count".format(status) for status in EVENT_STATUSES]
    if not daily[status_columns].sum(axis=1).eq(daily["event_count"]).all():
        raise ValueError("饮食事件映射状态未完整分区。")
    retained_event_denominator = daily["event_count"].replace(0, pd.NA)
    daily["food_id_event_coverage"] = (
        daily["food_id_nonmissing_events"] / retained_event_denominator
    ).where(retained_event_denominator.notna(), 1.0)
    daily["weight_g_event_coverage"] = (
        daily["weight_g_nonmissing_events"] / retained_event_denominator
    ).where(retained_event_denominator.notna(), 1.0)
    daily["calories_kcal_event_coverage"] = (
        daily["calories_kcal_nonmissing_events"] / retained_event_denominator
    ).where(retained_event_denominator.notna(), 1.0)
    classified_share = (
        daily["mapped_event_count"] + daily["not_applicable_event_count"]
    ) / retained_event_denominator
    daily["classified_food_event_share"] = classified_share.where(
        retained_event_denominator.notna(), 1.0
    )
    unresolved_share = (
        daily["unmapped_missing_labels_event_count"]
        + daily["unmatched_food_id_event_count"]
        + daily["missing_food_id_event_count"]
    ) / retained_event_denominator
    daily["unresolved_food_event_share"] = unresolved_share.where(
        retained_event_denominator.notna(), 0.0
    )
    return daily.sort_values(GROUP_COLUMNS).reset_index(drop=True)


def build_participant_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """按参与者实际记录日计算17组平均每日克数。"""
    keys = ["participant_id", "cohort", "research_stage"]
    data = daily.copy()
    data["collection_date_parsed"] = pd.to_datetime(
        data["collection_date"], errors="raise"
    )
    grouped = data.groupby(keys, sort=True, dropna=False)
    summary = grouped.size().rename("observed_diet_days").to_frame()
    summary["unique_logging_days"] = grouped["logging_day"].nunique()
    summary["first_collection_date"] = grouped["collection_date_parsed"].min()
    summary["last_collection_date"] = grouped["collection_date_parsed"].max()
    summary["total_food_events"] = grouped["event_count"].sum()
    summary["total_weight_imputed_events"] = grouped[
        "weight_imputed_event_count"
    ].sum()
    summary["redih_any_weight_imputed"] = summary[
        "total_weight_imputed_events"
    ].gt(0)
    summary["total_excluded_missing_labels_events"] = grouped[
        "excluded_missing_labels_event_count"
    ].sum()
    summary["total_excluded_missing_labels_weight_g"] = grouped[
        "excluded_missing_labels_weight_g_total"
    ].sum(min_count=1)
    summary["mean_daily_all_food_weight_g"] = grouped[
        "weight_g_total_all_foods"
    ].mean()
    summary["mean_daily_calories_kcal"] = grouped["calories_kcal_total"].mean()
    summary["days_with_calories_kcal"] = grouped["calories_kcal_total"].count()
    summary["mean_event_coverage_calories_kcal"] = grouped[
        "calories_kcal_event_coverage"
    ].mean()
    for status in EVENT_STATUSES:
        summary["total_{}_events".format(status)] = grouped[
            "{}_event_count".format(status)
        ].sum()
    participant_event_denominator = summary["total_food_events"].replace(
        0, pd.NA
    )
    classified_share = (
        summary["total_mapped_events"] + summary["total_not_applicable_events"]
    ) / participant_event_denominator
    summary["classified_food_event_share"] = classified_share.where(
        participant_event_denominator.notna(), 1.0
    )
    unresolved_share = (
        summary["total_unmapped_missing_labels_events"]
        + summary["total_unmatched_food_id_events"]
        + summary["total_missing_food_id_events"]
    ) / participant_event_denominator
    summary["unresolved_food_event_share"] = unresolved_share.where(
        participant_event_denominator.notna(), 0.0
    )

    component_data: Dict[str, pd.Series] = {}
    for component in REDIH_COMPONENTS:
        event_column = "{}_event_count".format(component)
        total_column = "{}_g_total".format(component)
        missing_column = "{}_missing_weight_event_count".format(component)
        component_data["total_{}_events".format(component)] = grouped[
            event_column
        ].sum()
        component_data[
            "total_{}_missing_weight_events".format(component)
        ] = grouped[missing_column].sum()
        component_data["days_with_{}".format(component)] = grouped[event_column].apply(
            lambda values: int(values.gt(0).sum())
        )
        component_data["{}_weight_complete".format(component)] = grouped[
            missing_column
        ].sum().eq(0)
        component_data["mean_daily_{}_g".format(component)] = grouped[
            total_column
        ].mean()
        component_data["median_daily_{}_g".format(component)] = grouped[
            total_column
        ].median()

    summary = summary.join(pd.DataFrame(component_data))
    all_complete_columns = [
        "{}_weight_complete".format(component) for component in REDIH_COMPONENTS
    ]
    current_complete_columns = [
        "{}_weight_complete".format(component)
        for component in CURRENT_SCORE_COMPONENTS
    ]
    summary["redih_available_component_weight_complete_count"] = summary[
        all_complete_columns
    ].sum(axis=1)
    summary["redih_current_score_component_weight_complete_count"] = summary[
        current_complete_columns
    ].sum(axis=1)
    summary["redih_all_available_component_weights_complete"] = summary[
        "redih_available_component_weight_complete_count"
    ].eq(len(REDIH_COMPONENTS))
    summary["redih_all_current_score_component_weights_complete"] = summary[
        "redih_current_score_component_weight_complete_count"
    ].eq(len(CURRENT_SCORE_COMPONENTS))

    summary = summary.reset_index()
    summary["calendar_span_days"] = (
        summary["last_collection_date"] - summary["first_collection_date"]
    ).dt.days + 1
    summary["first_collection_date"] = summary["first_collection_date"].dt.strftime(
        "%Y-%m-%d"
    )
    summary["last_collection_date"] = summary["last_collection_date"].dt.strftime(
        "%Y-%m-%d"
    )
    return summary


def add_metric_totals(total: Dict[str, int], update: Dict[str, int]) -> None:
    """累加不同 CSV 数据块产生的整数 QC 指标。"""
    for key, value in update.items():
        total[key] = total.get(key, 0) + int(value)


def build_qc(
    daily: pd.DataFrame,
    participants: pd.DataFrame,
    chunk_metrics: Dict[str, int],
) -> pd.DataFrame:
    """验证日级和参与者级结果并生成长格式 QC。"""
    selected_rows = int(chunk_metrics.get("selected_event_rows", 0))
    retained_rows = int(daily["event_count"].sum())
    excluded_missing_labels_rows = int(
        daily["excluded_missing_labels_event_count"].sum()
    )
    if retained_rows + excluded_missing_labels_rows != selected_rows:
        raise ValueError("保留事件与删除三标签缺失事件之和不等于筛选事件数。")
    if daily.duplicated(GROUP_COLUMNS).any():
        raise ValueError("日级结果的参与者-日索引不唯一。")
    participant_keys = ["participant_id", "cohort", "research_stage"]
    if participants.duplicated(participant_keys).any():
        raise ValueError("参与者级结果索引不唯一。")

    unmatched = int(daily["unmatched_food_id_event_count"].sum())
    if unmatched:
        raise ValueError(
            "目标饮食事件中有 {:,} 条非缺失 food_id 未出现在冻结映射表。".format(
                unmatched
            )
        )
    intake_columns = [
        "mean_daily_{}_g".format(component) for component in REDIH_COMPONENTS
    ]
    if participants[intake_columns].isna().any().any():
        raise ValueError("参与者级 rEDIH 组件 g/day 存在缺失。")
    if participants[intake_columns].lt(0).any().any():
        raise ValueError("参与者级 rEDIH 组件 g/day 存在负值。")

    metrics: Dict[str, object] = dict(chunk_metrics)
    metrics.update(
        {
            "script_version": SCRIPT_VERSION,
            "current_intake_unit": "g/day",
            "hpp_available_component_count": len(REDIH_COMPONENTS),
            "current_scored_component_count": len(CURRENT_SCORE_COMPONENTS),
            "wine_in_current_score": True,
            "event_count_before_missing_label_exclusion": selected_rows,
            "excluded_missing_labels_event_count": excluded_missing_labels_rows,
            "excluded_missing_labels_event_share": (
                excluded_missing_labels_rows / max(selected_rows, 1)
            ),
            "retained_event_count": retained_rows,
            "missing_label_event_rule": (
                "exclude event; retain participant and logging day"
            ),
            "daily_rows": len(daily),
            "participant_rows": len(participants),
            "unique_participants": participants["participant_id"].nunique(),
            "observed_diet_days_min": int(participants["observed_diet_days"].min()),
            "observed_diet_days_median": float(
                participants["observed_diet_days"].median()
            ),
            "observed_diet_days_max": int(participants["observed_diet_days"].max()),
            "participants_with_positive_mean_energy": int(
                participants["mean_daily_calories_kcal"].gt(0).sum()
            ),
            "weight_imputed_event_count": int(
                daily["weight_imputed_event_count"].sum()
            ),
            "participants_with_any_weight_imputation": int(
                participants["redih_any_weight_imputed"].sum()
            ),
            "mapped_component_missing_weight_event_count": int(
                sum(
                    daily[
                        "{}_missing_weight_event_count".format(component)
                    ].sum()
                    for component in REDIH_COMPONENTS
                )
            ),
            "participants_all_available_component_weights_complete": int(
                participants[
                    "redih_all_available_component_weights_complete"
                ].sum()
            ),
            "participants_all_current_score_component_weights_complete": int(
                participants[
                    "redih_all_current_score_component_weights_complete"
                ].sum()
            ),
            "participants_with_incomplete_component_weights": int(
                (~participants[
                    "redih_all_current_score_component_weights_complete"
                ]).sum()
            ),
            "mean_classified_food_event_share": float(
                participants["classified_food_event_share"].mean()
            ),
            "mean_unresolved_food_event_share": float(
                participants["unresolved_food_event_share"].mean()
            ),
        }
    )
    for status in EVENT_STATUSES:
        metrics["{}_event_count".format(status)] = int(
            daily["{}_event_count".format(status)].sum()
        )
    for component in REDIH_COMPONENTS:
        metrics["{}_event_count".format(component)] = int(
            daily["{}_event_count".format(component)].sum()
        )
        metrics["{}_missing_weight_event_count".format(component)] = int(
            daily["{}_missing_weight_event_count".format(component)].sum()
        )
        metrics["{}_affected_participant_count".format(component)] = int(
            participants[
                "total_{}_missing_weight_events".format(component)
            ].gt(0).sum()
        )
    return pd.DataFrame(
        {"metric": list(metrics.keys()), "value": list(metrics.values())}
    )


def build_intakes(
    events: pd.DataFrame,
    mapping: pd.DataFrame,
    cohort: str,
    research_stage: str,
    weight_overrides: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """供测试和小数据使用的一次性 rEDIH 汇总入口。"""
    clean_mapping = validate_mapping_frame(mapping, require_all_components=False)
    if weight_overrides is None:
        weight_overrides = pd.DataFrame(
            columns=[*EVENT_KEY_COLUMNS, "resolved_weight_g"]
        )
    partial, metrics = aggregate_chunk(
        events, clean_mapping, cohort, research_stage, weight_overrides
    )
    partials = [] if partial.empty else [partial]
    daily = combine_daily_partials(partials)
    participants = build_participant_summary(daily)
    qc = build_qc(daily, participants, metrics)
    return daily, participants, qc


def run(
    input_csv: Path,
    mapping_csv: Path,
    weight_overrides_csv: Path,
    output_dir: Path,
    cohort: str,
    research_stage: str,
    chunk_size: int,
) -> Tuple[Path, Path, Path]:
    """分块读取全量饮食事件并保存 rEDIH 摄入结果。"""
    if not input_csv.is_file():
        raise FileNotFoundError("找不到饮食事件表：{}".format(input_csv))
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于0。")
    mapping = load_mapping(mapping_csv)
    weight_overrides = load_weight_overrides(weight_overrides_csv)
    header = pd.read_csv(input_csv, encoding="utf-8-sig", nrows=0)
    validate_columns(header.columns.tolist(), REQUIRED_EVENT_COLUMNS, "饮食事件表")

    partials: List[pd.DataFrame] = []
    total_metrics: Dict[str, int] = {}
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
            chunk, mapping, cohort, research_stage, weight_overrides
        )
        add_metric_totals(total_metrics, metrics)
        if not partial.empty:
            partials.append(partial)
        print(
            "处理数据块 {:,}：输入 {:,}，目标 {:,}".format(
                chunk_number,
                metrics["input_event_rows"],
                metrics["selected_event_rows"],
            )
        )

    daily = combine_daily_partials(partials)
    participants = build_participant_summary(daily)
    qc = build_qc(daily, participants, total_metrics)

    output_dir.mkdir(parents=True, exist_ok=True)
    daily_path = output_dir / "redih_daily_component_intakes.csv"
    participant_path = output_dir / "redih_participant_component_intakes.csv"
    qc_path = output_dir / "redih_intake_qc.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    participants.to_csv(participant_path, index=False, encoding="utf-8-sig")
    qc.to_csv(qc_path, index=False, encoding="utf-8-sig")

    print("=" * 88)
    print("rEDIH 17组 g/day 摄入汇总")
    print("SCRIPT_VERSION={}".format(SCRIPT_VERSION))
    print("筛选：cohort={}；research_stage={}".format(cohort, research_stage))
    print("当前计分使用全部17个 HPP 可用组件，包含 wine。")
    excluded_missing_labels_events = int(
        daily["excluded_missing_labels_event_count"].sum()
    )
    retained_events = int(daily["event_count"].sum())
    print(
        "删除三标签全缺失事件前：{:,}".format(
            retained_events + excluded_missing_labels_events
        )
    )
    print("删除三标签全缺失事件：{:,}".format(excluded_missing_labels_events))
    print("最终计分保留事件：{:,}".format(retained_events))
    print("参与者-日：{:,}".format(len(daily)))
    print("参与者：{:,}".format(len(participants)))
    print(
        "当前17组件重量严格完整：{:,}".format(
            int(
                participants[
                    "redih_all_current_score_component_weights_complete"
                ].sum()
            )
        )
    )
    print("各组件事件数与缺重量事件数：")
    for component in REDIH_COMPONENTS:
        print(
            "{:<32} events={:>9,} missing_weight={:>7,}".format(
                component,
                int(daily["{}_event_count".format(component)].sum()),
                int(
                    daily["{}_missing_weight_event_count".format(component)].sum()
                ),
            )
        )
    print("输出：{}".format(output_dir))
    print("=" * 88)
    return daily_path, participant_path, qc_path


def parse_args() -> object:
    """解析命令行参数。"""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--mapping-csv", type=Path, default=DEFAULT_MAPPING_CSV)
    parser.add_argument(
        "--weight-overrides-csv",
        type=Path,
        default=DEFAULT_WEIGHT_OVERRIDES_CSV,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cohort", default=DEFAULT_COHORT)
    parser.add_argument("--research-stage", default=DEFAULT_RESEARCH_STAGE)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""
    args = parse_args()
    run(
        args.input_csv,
        args.mapping_csv,
        args.weight_overrides_csv,
        args.output_dir,
        args.cohort,
        args.research_stage,
        args.chunk_size,
    )


if __name__ == "__main__":
    main()
