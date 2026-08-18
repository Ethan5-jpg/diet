"""将饮食事件汇总为 hPDI 日级和参与者级18组摄入量。"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


SCRIPT_VERSION = "2026-08-14-hpdi-intakes-v5"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR.parent
DEFAULT_INPUT_CSV = DATA_DIR / "Transfer" / "diet_logging" / "diet_logging_events.csv"
DEFAULT_MAPPING_CSV = (
    PROJECT_DIR / "outputs" / "03_score_mapping" / "hpdi" / "hpdi_food_id_mapping.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "04_score_intakes" / "hpdi"
DEFAULT_WEIGHT_OVERRIDES_CSV = (
    DEFAULT_OUTPUT_DIR / "hpdi_missing_weight_review.csv"
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
REQUIRED_MAPPING_COLUMNS = ["food_id", "hpdi_component", "mapping_status"]
REQUIRED_WEIGHT_OVERRIDE_COLUMNS = [
    *EVENT_KEY_COLUMNS,
    "resolution_status",
    "resolved_weight_g",
]
HPDI_COMPONENTS = [
    "whole_grains",
    "fruits",
    "vegetables",
    "nuts",
    "legumes",
    "vegetable_oils",
    "tea_coffee",
    "fruit_juice",
    "refined_grains",
    "potatoes",
    "sugar_sweetened_beverages",
    "sweets_desserts",
    "animal_fat",
    "dairy",
    "eggs",
    "fish_seafood",
    "meat",
    "miscellaneous_animal_foods",
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


def normalize_text(series: pd.Series) -> pd.Series:
    """去除文本首尾空白，并将空字符串视为缺失。"""
    normalized = series.astype("string").str.strip()
    return normalized.mask(normalized.eq(""), pd.NA)


def validate_columns(columns: List[str], required: List[str], source_name: str) -> None:
    """确认输入表包含所需字段。"""
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(f"{source_name}缺少字段：{', '.join(missing)}")


def load_mapping(mapping_csv: Path) -> pd.DataFrame:
    """读取并验证已冻结的 hPDI food_id 映射。"""
    if not mapping_csv.is_file():
        raise FileNotFoundError(f"找不到 hPDI 映射表：{mapping_csv}")

    header = pd.read_csv(mapping_csv, encoding="utf-8-sig", nrows=0)
    validate_columns(header.columns.tolist(), REQUIRED_MAPPING_COLUMNS, "hPDI 映射表")
    mapping = pd.read_csv(
        mapping_csv,
        encoding="utf-8-sig",
        usecols=REQUIRED_MAPPING_COLUMNS,
        dtype={"food_id": "string"},
        low_memory=False,
    )
    for column in REQUIRED_MAPPING_COLUMNS:
        mapping[column] = normalize_text(mapping[column])

    if mapping["food_id"].isna().any():
        raise ValueError("hPDI 映射表存在缺失 food_id。")
    if mapping["food_id"].duplicated().any():
        duplicate_ids = sorted(
            mapping.loc[mapping["food_id"].duplicated(keep=False), "food_id"]
            .astype(str)
            .unique()
        )
        raise ValueError(
            "hPDI 映射表存在重复 food_id：" + ", ".join(duplicate_ids[:10])
        )

    invalid_status = ~mapping["mapping_status"].isin(VALID_MAPPING_STATUSES)
    if invalid_status.any():
        statuses = sorted(
            mapping.loc[invalid_status, "mapping_status"]
            .fillna("<missing>")
            .astype(str)
            .unique()
        )
        raise ValueError(
            "hPDI 映射尚未冻结；请先处理 review_required 或无效状态："
            + ", ".join(statuses)
        )

    mapped = mapping["mapping_status"].eq("mapped")
    invalid_component = mapped & ~mapping["hpdi_component"].isin(HPDI_COMPONENTS)
    if invalid_component.any():
        components = sorted(
            mapping.loc[invalid_component, "hpdi_component"]
            .fillna("<missing>")
            .astype(str)
            .unique()
        )
        raise ValueError("已映射 food_id 含无效 hPDI 组件：" + ", ".join(components))
    if mapping.loc[~mapped, "hpdi_component"].notna().any():
        raise ValueError("非 mapped 记录不应包含 hpdi_component。")

    observed = set(mapping.loc[mapped, "hpdi_component"].dropna())
    missing_components = sorted(set(HPDI_COMPONENTS) - observed)
    if missing_components:
        raise ValueError("hPDI 映射缺少组件：" + ", ".join(missing_components))
    return mapping


def load_weight_overrides(weight_overrides_csv: Path) -> pd.DataFrame:
    """读取已批准的事件级重量；pending/unresolved 行不会被应用。"""
    if not weight_overrides_csv.is_file():
        return pd.DataFrame(columns=[*EVENT_KEY_COLUMNS, "resolved_weight_g"])

    header = pd.read_csv(weight_overrides_csv, encoding="utf-8-sig", nrows=0)
    validate_columns(
        header.columns.tolist(),
        REQUIRED_WEIGHT_OVERRIDE_COLUMNS,
        "hPDI 缺重量复核表",
    )
    overrides = pd.read_csv(
        weight_overrides_csv,
        encoding="utf-8-sig",
        usecols=REQUIRED_WEIGHT_OVERRIDE_COLUMNS,
        dtype={column: "string" for column in EVENT_KEY_COLUMNS if column != "source_event_row"},
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
            "hPDI 缺重量复核表含无效 resolution_status："
            + ", ".join(statuses)
        )

    approved = overrides["resolution_status"].eq("use_resolved_weight")
    resolved = pd.to_numeric(overrides["resolved_weight_g"], errors="coerce")
    invalid_resolved = approved & (resolved.isna() | resolved.le(0))
    if invalid_resolved.any():
        raise ValueError(
            "resolution_status=use_resolved_weight 但 resolved_weight_g "
            f"缺失或非正数：{int(invalid_resolved.sum()):,} 行。"
        )
    approved_overrides = overrides.loc[
        approved, EVENT_KEY_COLUMNS
    ].copy()
    approved_overrides["resolved_weight_g"] = resolved.loc[approved].astype(
        "float64"
    )
    if approved_overrides[EVENT_KEY_COLUMNS].isna().any(axis=1).any():
        raise ValueError("已确认的 hPDI 重量 override 存在缺失事件键。")
    if approved_overrides.duplicated(EVENT_KEY_COLUMNS).any():
        raise ValueError("hPDI 重量 override 存在重复事件键。")
    return approved_overrides.reset_index(drop=True)


def aggregate_chunk(
    chunk: pd.DataFrame,
    mapping: pd.DataFrame,
    cohort: str,
    research_stage: str,
    weight_overrides: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """筛选一个数据块并生成参与者-日级局部汇总。"""
    chunk = chunk.copy()
    if "source_event_row" not in chunk.columns:
        chunk["source_event_row"] = chunk.index.to_series().astype("int64") + 2
    chunk = chunk.loc[:, [*REQUIRED_EVENT_COLUMNS, "source_event_row"]].copy()
    input_rows = len(chunk)
    for column in TEXT_COLUMNS:
        chunk[column] = normalize_text(chunk[column])

    selected = chunk["cohort"].eq(cohort) & chunk["research_stage"].eq(research_stage)
    chunk = chunk.loc[selected].copy()
    metrics = {
        "input_rows": input_rows,
        "selected_rows": len(chunk),
        "excluded_other_cohort_or_stage_rows": input_rows - len(chunk),
    }
    if chunk.empty:
        return pd.DataFrame(), metrics

    chunk["logging_day"] = pd.to_numeric(chunk["logging_day"], errors="coerce").astype(
        "Int64"
    )
    dates = pd.to_datetime(chunk["collection_date"], errors="coerce")
    chunk["collection_date"] = dates.dt.strftime("%Y-%m-%d")
    if chunk[GROUP_COLUMNS].isna().any(axis=1).any():
        count = int(chunk[GROUP_COLUMNS].isna().any(axis=1).sum())
        raise ValueError(f"目标记录中有 {count:,} 行缺少日级汇总键。")

    original_weight = chunk["weight_g"]
    converted_weight = pd.to_numeric(original_weight, errors="coerce")
    invalid_numeric = original_weight.notna() & converted_weight.isna()
    negative_weight = converted_weight.lt(0).fillna(False)
    chunk["weight_g"] = converted_weight.mask(negative_weight)
    metrics["invalid_numeric_weight_g"] = int(invalid_numeric.sum())
    metrics["negative_weight_g_rows"] = int(negative_weight.sum())
    chunk["__missing_weight_before_override"] = chunk["weight_g"].isna()

    if weight_overrides is None:
        weight_overrides = pd.DataFrame(
            columns=[*EVENT_KEY_COLUMNS, "resolved_weight_g"]
        )
    if not weight_overrides.empty:
        chunk = chunk.merge(
            weight_overrides,
            on=EVENT_KEY_COLUMNS,
            how="left",
            validate="many_to_one",
        )
        missing_weight_before_override = chunk[
            "__missing_weight_before_override"
        ]
        override_available = chunk["resolved_weight_g"].notna()
        override_applied = missing_weight_before_override & override_available
        chunk.loc[override_applied, "weight_g"] = chunk.loc[
            override_applied, "resolved_weight_g"
        ]
        override_not_needed = (~missing_weight_before_override) & override_available
        chunk = chunk.drop(columns=["resolved_weight_g"])
    else:
        missing_weight_before_override = chunk[
            "__missing_weight_before_override"
        ]
        override_applied = pd.Series(False, index=chunk.index)
        override_not_needed = pd.Series(False, index=chunk.index)

    metrics["missing_weight_before_override_rows"] = int(
        missing_weight_before_override.sum()
    )
    metrics["weight_override_applied_rows"] = int(override_applied.sum())
    metrics["weight_override_not_needed_rows"] = int(
        override_not_needed.sum()
    )
    metrics["missing_weight_after_override_rows"] = int(
        chunk["weight_g"].isna().sum()
    )
    chunk = chunk.drop(columns=["__missing_weight_before_override"])

    original_calories = chunk["calories_kcal"]
    converted_calories = pd.to_numeric(original_calories, errors="coerce")
    invalid_calories = original_calories.notna() & converted_calories.isna()
    negative_calories = converted_calories.lt(0).fillna(False)
    chunk["calories_kcal"] = converted_calories.mask(negative_calories)
    metrics["invalid_numeric_calories_kcal"] = int(invalid_calories.sum())
    metrics["negative_calories_kcal_rows"] = int(negative_calories.sum())

    chunk = chunk.merge(mapping, on="food_id", how="left", validate="many_to_one")
    chunk["event_mapping_status"] = chunk["mapping_status"]
    missing_food_id = chunk["food_id"].isna()
    unmatched_food_id = chunk["mapping_status"].isna() & ~missing_food_id
    chunk.loc[missing_food_id, "event_mapping_status"] = "missing_food_id"
    chunk.loc[unmatched_food_id, "event_mapping_status"] = "unmatched_food_id"

    # 三个食品标签均缺失的 food_id 无法归入任一 hPDI 组。按已确认的
    # 主分析口径删除这些事件，但保留参与者及其原始记录日，以免改变
    # 两周日志的观察日分母。删除事件不进入食品重量、能量、组件摄入量
    # 或事件覆盖率，仅在独立 QC 字段中保留数量和可观测重量。
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
        mask = retained_event & chunk["event_mapping_status"].eq(status)
        chunk[f"__{status}_event"] = mask.astype("int64")

    for component in HPDI_COMPONENTS:
        mask = (
            retained_event
            & chunk["event_mapping_status"].eq("mapped")
            & chunk["hpdi_component"].eq(component)
        )
        chunk[f"__{component}_event"] = mask.astype("int64")
        chunk[f"__{component}_valid_weight_event"] = (
            mask & chunk["weight_g"].notna()
        ).astype("int64")
        chunk[f"__{component}_weight_g"] = chunk["weight_g"].where(mask)

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
        daily[f"{status}_event_count"] = grouped[f"__{status}_event"].sum()
    for component in HPDI_COMPONENTS:
        daily[f"{component}_event_count"] = grouped[f"__{component}_event"].sum()
        daily[f"{component}_valid_weight_event_count"] = grouped[
            f"__{component}_valid_weight_event"
        ].sum()
        daily[f"{component}_g_total"] = grouped[f"__{component}_weight_g"].sum(
            min_count=1
        )

    metrics["missing_food_id_rows"] = int(missing_food_id.sum())
    metrics["unmatched_food_id_rows"] = int(unmatched_food_id.sum())
    metrics["excluded_missing_labels_rows"] = int(
        excluded_missing_labels.sum()
    )
    metrics["retained_selected_rows"] = int(retained_event.sum())
    return daily.reset_index(), metrics


def combine_daily_partials(partials: List[pd.DataFrame]) -> pd.DataFrame:
    """合并跨 CSV 数据块的日级局部汇总。"""
    combined = pd.concat(partials, ignore_index=True)
    grouped = combined.groupby(GROUP_COLUMNS, sort=True, dropna=False)
    count_columns = [
        "event_count",
        "weight_imputed_event_count",
        "excluded_missing_labels_event_count",
        "food_id_nonmissing_events",
        "weight_g_nonmissing_events",
        "calories_kcal_nonmissing_events",
        *[f"{status}_event_count" for status in EVENT_STATUSES],
        *[f"{component}_event_count" for component in HPDI_COMPONENTS],
        *[f"{component}_valid_weight_event_count" for component in HPDI_COMPONENTS],
    ]
    total_columns = [
        "weight_g_total_all_foods",
        "calories_kcal_total",
        "excluded_missing_labels_weight_g_total",
        *[f"{component}_g_total" for component in HPDI_COMPONENTS],
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

    for component in HPDI_COMPONENTS:
        events = daily[f"{component}_event_count"]
        valid_events = daily[f"{component}_valid_weight_event_count"]
        missing_weight_events = events - valid_events
        daily[f"{component}_missing_weight_event_count"] = missing_weight_events
        # 已批准的替代重量在聚合前应用。仍未解决的缺重量事件不贡献克数，
        # 但不改变参与者的实际记录日分母，并由完整性标志排除出主评分。
        daily[f"{component}_g_total"] = pd.to_numeric(
            daily[f"{component}_g_total"], errors="coerce"
        ).fillna(0.0)
        daily[f"{component}_weight_complete"] = events.eq(valid_events)

    status_total = daily[[f"{status}_event_count" for status in EVENT_STATUSES]].sum(
        axis=1
    )
    if not status_total.eq(daily["event_count"]).all():
        raise ValueError("饮食事件映射状态未完整分区。")
    retained_event_denominator = daily["event_count"].replace(0, pd.NA)
    daily["food_id_event_coverage"] = (
        daily["food_id_nonmissing_events"] / retained_event_denominator
    ).where(retained_event_denominator.notna(), 1.0)
    daily["weight_g_event_coverage"] = (
        daily["weight_g_nonmissing_events"] / retained_event_denominator
    ).where(retained_event_denominator.notna(), 1.0)
    daily["calories_kcal_event_coverage"] = (
        daily["calories_kcal_nonmissing_events"]
        / retained_event_denominator
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
    """按参与者实际记录日计算18组平均每日摄入量。"""
    keys = ["participant_id", "cohort", "research_stage"]
    daily = daily.copy()
    daily["collection_date_parsed"] = pd.to_datetime(
        daily["collection_date"], errors="raise"
    )
    grouped = daily.groupby(keys, sort=True, dropna=False)
    summary = grouped.size().rename("observed_diet_days").to_frame()
    summary["unique_logging_days"] = grouped["logging_day"].nunique()
    summary["first_collection_date"] = grouped["collection_date_parsed"].min()
    summary["last_collection_date"] = grouped["collection_date_parsed"].max()
    summary["total_food_events"] = grouped["event_count"].sum()
    summary["total_weight_imputed_events"] = grouped[
        "weight_imputed_event_count"
    ].sum()
    summary["hpdi_any_weight_imputed"] = summary[
        "total_weight_imputed_events"
    ].gt(0)
    summary["total_excluded_missing_labels_events"] = grouped[
        "excluded_missing_labels_event_count"
    ].sum()
    summary["total_excluded_missing_labels_weight_g"] = grouped[
        "excluded_missing_labels_weight_g_total"
    ].sum(min_count=1)
    summary["mean_daily_all_food_weight_g"] = grouped["weight_g_total_all_foods"].mean()
    summary["mean_daily_calories_kcal_proxy"] = grouped[
        "calories_kcal_total"
    ].mean()
    summary["days_with_calories_kcal"] = grouped["calories_kcal_total"].count()
    summary["mean_event_coverage_calories_kcal"] = grouped[
        "calories_kcal_event_coverage"
    ].mean()
    for status in EVENT_STATUSES:
        summary[f"total_{status}_events"] = grouped[f"{status}_event_count"].sum()
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

    component_columns: Dict[str, pd.Series] = {}
    for component in HPDI_COMPONENTS:
        total_column = f"{component}_g_total"
        event_column = f"{component}_event_count"
        complete_column = f"{component}_weight_complete"
        total_events = grouped[event_column].sum()
        days_with_component = grouped[event_column].apply(
            lambda values: int(values.gt(0).sum())
        )
        complete_days = grouped[complete_column].sum()
        participant_complete = complete_days.eq(summary["observed_diet_days"])
        mean_intake = grouped[total_column].mean()
        median_intake = grouped[total_column].median()
        component_columns[f"total_{component}_events"] = total_events
        component_columns[f"total_{component}_missing_weight_events"] = grouped[
            f"{component}_missing_weight_event_count"
        ].sum()
        component_columns[f"days_with_{component}"] = days_with_component
        component_columns[f"days_with_complete_{component}_weight"] = complete_days
        component_columns[f"{component}_weight_complete"] = participant_complete
        component_columns[f"mean_daily_{component}_g_proxy"] = mean_intake
        component_columns[f"median_daily_{component}_g_proxy"] = median_intake

    summary = summary.join(pd.DataFrame(component_columns))
    weight_complete_columns = [
        f"{component}_weight_complete" for component in HPDI_COMPONENTS
    ]
    summary["hpdi_component_weight_complete_count"] = summary[
        weight_complete_columns
    ].sum(axis=1)
    summary["hpdi_all_18_component_weights_complete"] = summary[
        "hpdi_component_weight_complete_count"
    ].eq(len(HPDI_COMPONENTS))

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


def build_qc_table(
    daily: pd.DataFrame,
    participants: pd.DataFrame,
    chunk_metrics: Dict[str, int],
    cohort: str,
    research_stage: str,
) -> pd.DataFrame:
    """生成 hPDI 摄入量质量控制表。"""
    selected_events = int(daily["event_count"].sum())
    excluded_missing_labels_events = int(
        daily["excluded_missing_labels_event_count"].sum()
    )
    events_before_missing_label_exclusion = (
        selected_events + excluded_missing_labels_events
    )
    retained_denominator = max(selected_events, 1)
    metrics = {
        **chunk_metrics,
        "script_version": SCRIPT_VERSION,
        "selected_cohort": cohort,
        "selected_research_stage": research_stage,
        "daily_rows": len(daily),
        "participant_rows": len(participants),
        "unique_participants": participants["participant_id"].nunique(),
        "minimum_collection_date": daily["collection_date"].min(),
        "maximum_collection_date": daily["collection_date"].max(),
        "event_count_before_missing_label_exclusion": (
            events_before_missing_label_exclusion
        ),
        "excluded_missing_labels_event_count": (
            excluded_missing_labels_events
        ),
        "excluded_missing_labels_event_share": (
            excluded_missing_labels_events
            / max(events_before_missing_label_exclusion, 1)
        ),
        "retained_event_count": selected_events,
        "missing_label_event_rule": (
            "exclude event; retain participant and logging day"
        ),
        "classified_food_event_share": (
            daily["mapped_event_count"].sum()
            + daily["not_applicable_event_count"].sum()
        )
        / retained_denominator,
        "unresolved_food_event_share": (
            daily["unmapped_missing_labels_event_count"].sum()
            + daily["unmatched_food_id_event_count"].sum()
            + daily["missing_food_id_event_count"].sum()
        )
        / retained_denominator,
        "weight_g_event_coverage": daily["weight_g_nonmissing_events"].sum()
        / retained_denominator,
        "calories_kcal_event_coverage": daily[
            "calories_kcal_nonmissing_events"
        ].sum()
        / retained_denominator,
        "participants_with_positive_mean_daily_calories": int(
            participants["mean_daily_calories_kcal_proxy"].gt(0).sum()
        ),
        "weight_imputed_event_count": int(
            daily["weight_imputed_event_count"].sum()
        ),
        "participants_with_any_weight_imputation": int(
            participants["hpdi_any_weight_imputed"].sum()
        ),
        "mapped_component_missing_weight_event_count": int(
            sum(
                daily[f"{component}_missing_weight_event_count"].sum()
                for component in HPDI_COMPONENTS
            )
        ),
        "participants_with_all_18_component_weights_complete": int(
            participants["hpdi_all_18_component_weights_complete"].sum()
        ),
        "participants_with_incomplete_component_weights": int(
            (~participants["hpdi_all_18_component_weights_complete"]).sum()
        ),
    }
    for status in EVENT_STATUSES:
        count = int(daily[f"{status}_event_count"].sum())
        metrics[f"{status}_event_count"] = count
        metrics[f"{status}_event_share"] = count / retained_denominator
    for component in HPDI_COMPONENTS:
        metrics[f"{component}_event_count"] = int(
            daily[f"{component}_event_count"].sum()
        )
        metrics[f"participants_complete_{component}"] = int(
            participants[f"{component}_weight_complete"].sum()
        )
    return pd.DataFrame(
        {"metric": list(metrics.keys()), "value": list(metrics.values())}
    )


def build_hpdi_intakes(
    input_csv: Path,
    mapping_csv: Path,
    weight_overrides_csv: Path,
    output_dir: Path,
    chunk_size: int,
    cohort: str,
    research_stage: str,
) -> Tuple[Path, Path, Path]:
    """执行分块汇总并保存 hPDI 摄入量及 QC。"""
    if not input_csv.is_file():
        raise FileNotFoundError(f"找不到饮食事件表：{input_csv}")
    mapping = load_mapping(mapping_csv)
    weight_overrides = load_weight_overrides(weight_overrides_csv)
    header = pd.read_csv(input_csv, encoding="utf-8-sig", nrows=0)
    validate_columns(header.columns.tolist(), REQUIRED_EVENT_COLUMNS, "饮食事件表")

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
            chunk,
            mapping,
            cohort,
            research_stage,
            weight_overrides,
        )
        if not partial.empty:
            partials.append(partial)
        for key, value in metrics.items():
            accumulated_metrics[key] = accumulated_metrics.get(key, 0) + value
        print(
            f"已处理第 {chunk_number} 块，累计读取 "
            f"{accumulated_metrics['input_rows']:,} 行，目标事件 "
            f"{accumulated_metrics['selected_rows']:,} 行"
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
            "请先重新运行食物字典和 hPDI 映射。"
        )
    participants = build_participant_summary(daily)
    qc = build_qc_table(
        daily,
        participants,
        accumulated_metrics,
        cohort,
        research_stage,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    daily_path = output_dir / "hpdi_daily_component_intakes.csv"
    participant_path = output_dir / "hpdi_participant_component_intakes.csv"
    qc_path = output_dir / "hpdi_intake_qc.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    participants.to_csv(participant_path, index=False, encoding="utf-8-sig")
    qc.to_csv(qc_path, index=False, encoding="utf-8-sig")
    retained_events = int(daily["event_count"].sum())
    excluded_missing_labels_events = int(
        daily["excluded_missing_labels_event_count"].sum()
    )
    print("-" * 72)
    print(f"脚本版本：{SCRIPT_VERSION}")
    print(
        "删除三标签全缺失事件前："
        f"{retained_events + excluded_missing_labels_events:,} 行"
    )
    print(f"删除三标签全缺失事件：{excluded_missing_labels_events:,} 行")
    print(f"最终计分保留事件：{retained_events:,} 行")
    print(f"日级记录：{len(daily):,} 行")
    print(f"参与者：{len(participants):,} 人")
    print(f"日级结果：{daily_path}")
    print(f"参与者级结果：{participant_path}")
    print(f"QC 结果：{qc_path}")
    return daily_path, participant_path, qc_path


def parse_args():
    parser = ArgumentParser(description="汇总 hPDI 日级和参与者级成分摄入量")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_CSV)
    parser.add_argument(
        "--weight-overrides",
        type=Path,
        default=DEFAULT_WEIGHT_OVERRIDES_CSV,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--cohort", default=DEFAULT_COHORT)
    parser.add_argument("--research-stage", default=DEFAULT_RESEARCH_STAGE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_hpdi_intakes(
        args.input,
        args.mapping,
        args.weight_overrides,
        args.output_dir,
        args.chunk_size,
        args.cohort,
        args.research_stage,
    )
