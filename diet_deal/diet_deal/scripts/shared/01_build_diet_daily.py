"""将饮食事件表汇总为日级和参与者级分析底表。"""

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
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "01_daily_summary"

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
    "short_food_name",
    "product_name",
    "food_category",
]
LABEL_COLUMNS = ["short_food_name", "product_name", "food_category"]
NUTRIENT_COLUMNS = [
    "weight_g",
    "calories_kcal",
    "carbohydrate_g",
    "lipid_g",
    "protein_g",
    "sodium_mg",
    "alcohol_g",
    "dietary_fiber_g",
]
REQUIRED_COLUMNS = GROUP_COLUMNS + [
    "collection_timestamp",
    "food_id",
    "short_food_name",
    "product_name",
    "food_category",
    *NUTRIENT_COLUMNS,
    "eaten_in_restaurant",
]


def normalize_text(series: pd.Series) -> pd.Series:
    """去除文本首尾空白，并将空字符串视为缺失。"""
    normalized = series.astype("string").str.strip()
    return normalized.mask(normalized.eq(""), pd.NA)


def parse_restaurant_flag(series: pd.Series) -> pd.Series:
    """将常见布尔表示转换成 0/1，未知值保留为缺失。"""
    normalized = normalize_text(series).str.lower()
    mapping = {
        "true": 1,
        "1": 1,
        "yes": 1,
        "y": 1,
        "false": 0,
        "0": 0,
        "no": 0,
        "n": 0,
    }
    return normalized.map(mapping).astype("Int64")


def validate_columns(columns: List[str]) -> None:
    """确认输入表包含日级汇总所需字段。"""
    missing = sorted(set(REQUIRED_COLUMNS) - set(columns))
    if missing:
        raise ValueError(f"输入文件缺少字段：{', '.join(missing)}")


def aggregate_chunk(chunk: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """清理一个数据块并生成日级局部汇总。"""
    chunk = chunk.loc[:, REQUIRED_COLUMNS].copy()

    for column in TEXT_COLUMNS:
        chunk[column] = normalize_text(chunk[column])

    chunk["logging_day"] = pd.to_numeric(
        chunk["logging_day"], errors="coerce"
    ).astype("Int64")
    collection_dates = pd.to_datetime(
        chunk["collection_date"], errors="coerce"
    )
    chunk["collection_date"] = collection_dates.dt.strftime("%Y-%m-%d")
    chunk["collection_timestamp"] = pd.to_datetime(
        chunk["collection_timestamp"], errors="coerce", utc=True
    )

    missing_key_rows = int(chunk[GROUP_COLUMNS].isna().any(axis=1).sum())
    if missing_key_rows:
        raise ValueError(
            f"发现 {missing_key_rows:,} 行缺少日级汇总键，已停止处理。"
        )

    invalid_numeric_counts = {}
    for column in NUTRIENT_COLUMNS:
        original_nonmissing = chunk[column].notna()
        converted = pd.to_numeric(chunk[column], errors="coerce")
        invalid_numeric_counts[column] = int(
            (original_nonmissing & converted.isna()).sum()
        )
        chunk[column] = converted

    chunk["all_food_labels_missing"] = chunk[LABEL_COLUMNS].isna().all(axis=1)
    chunk["restaurant_flag"] = parse_restaurant_flag(
        chunk["eaten_in_restaurant"]
    )

    grouped = chunk.groupby(GROUP_COLUMNS, sort=False, dropna=False)
    daily = grouped.size().rename("event_count").to_frame()
    daily["food_id_nonmissing_events"] = grouped["food_id"].count()
    daily["all_food_labels_missing_count"] = grouped[
        "all_food_labels_missing"
    ].sum()
    daily["restaurant_event_count"] = grouped["restaurant_flag"].sum(
        min_count=1
    )
    daily["restaurant_flag_nonmissing_events"] = grouped[
        "restaurant_flag"
    ].count()
    daily["first_collection_timestamp"] = grouped[
        "collection_timestamp"
    ].min()
    daily["last_collection_timestamp"] = grouped[
        "collection_timestamp"
    ].max()

    nutrient_sums = grouped[NUTRIENT_COLUMNS].sum(min_count=1)
    nutrient_sums = nutrient_sums.rename(
        columns={column: f"{column}_total" for column in NUTRIENT_COLUMNS}
    )
    nutrient_counts = grouped[NUTRIENT_COLUMNS].count()
    nutrient_counts = nutrient_counts.rename(
        columns={
            column: f"{column}_nonmissing_events"
            for column in NUTRIENT_COLUMNS
        }
    )

    daily = daily.join(nutrient_sums).join(nutrient_counts).reset_index()
    metrics = {
        "input_rows": len(chunk),
        "invalid_collection_timestamp_rows": int(
            chunk["collection_timestamp"].isna().sum()
        ),
        "all_food_labels_missing_rows": int(
            chunk["all_food_labels_missing"].sum()
        ),
        "restaurant_flag_unrecognized_rows": int(
            chunk["restaurant_flag"].isna().sum()
            - chunk["eaten_in_restaurant"].isna().sum()
        ),
    }
    for column, count in invalid_numeric_counts.items():
        metrics[f"invalid_numeric_{column}"] = count

    return daily, metrics


def combine_daily_partials(partials: List[pd.DataFrame]) -> pd.DataFrame:
    """合并跨 CSV 分块的日级局部汇总。"""
    combined = pd.concat(partials, ignore_index=True)
    grouped = combined.groupby(GROUP_COLUMNS, sort=True, dropna=False)

    sum_columns = [
        "event_count",
        "food_id_nonmissing_events",
        "all_food_labels_missing_count",
        "restaurant_event_count",
        "restaurant_flag_nonmissing_events",
        *[f"{column}_total" for column in NUTRIENT_COLUMNS],
        *[f"{column}_nonmissing_events" for column in NUTRIENT_COLUMNS],
    ]
    daily = grouped[sum_columns].sum(min_count=1)
    daily["first_collection_timestamp"] = grouped[
        "first_collection_timestamp"
    ].min()
    daily["last_collection_timestamp"] = grouped[
        "last_collection_timestamp"
    ].max()
    daily = daily.reset_index()

    daily["all_food_labels_missing_rate"] = (
        daily["all_food_labels_missing_count"] / daily["event_count"]
    )
    daily["food_id_coverage"] = (
        daily["food_id_nonmissing_events"] / daily["event_count"]
    )
    daily["restaurant_event_rate"] = (
        daily["restaurant_event_count"]
        / daily["restaurant_flag_nonmissing_events"]
    )

    for column in NUTRIENT_COLUMNS:
        daily[f"{column}_coverage"] = (
            daily[f"{column}_nonmissing_events"] / daily["event_count"]
        )

    core_columns = [
        "calories_kcal",
        "carbohydrate_g",
        "lipid_g",
        "protein_g",
    ]
    daily["core_nutrient_totals_available"] = daily[
        [f"{column}_total" for column in core_columns]
    ].notna().all(axis=1)
    daily["core_nutrients_full_event_coverage"] = daily[
        [f"{column}_coverage" for column in core_columns]
    ].eq(1).all(axis=1)

    return daily.sort_values(GROUP_COLUMNS).reset_index(drop=True)


def build_participant_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """从日级表生成每个参与者-访视阶段一行的汇总。"""
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
    summary["mean_food_events_per_day"] = grouped["event_count"].mean()
    summary["days_with_all_food_labels_missing"] = grouped[
        "all_food_labels_missing_rate"
    ].apply(lambda values: int(values.eq(1).sum()))
    summary["days_with_core_nutrient_totals"] = grouped[
        "core_nutrient_totals_available"
    ].sum()
    summary["days_with_full_core_nutrient_coverage"] = grouped[
        "core_nutrients_full_event_coverage"
    ].sum()

    for column in NUTRIENT_COLUMNS:
        total_column = f"{column}_total"
        coverage_column = f"{column}_coverage"
        summary[f"mean_daily_{column}"] = grouped[total_column].mean()
        summary[f"median_daily_{column}"] = grouped[total_column].median()
        summary[f"days_with_{column}"] = grouped[total_column].count()
        summary[f"mean_event_coverage_{column}"] = grouped[
            coverage_column
        ].mean()

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
) -> pd.DataFrame:
    """生成便于人工审查的整体 QC 指标表。"""
    metrics = {
        **chunk_metrics,
        "daily_rows": len(daily),
        "participant_stage_rows": len(participant_summary),
        "unique_participants": participant_summary["participant_id"].nunique(),
        "cohorts": "|".join(
            sorted(participant_summary["cohort"].dropna().astype(str).unique())
        ),
        "research_stages": "|".join(
            sorted(
                participant_summary["research_stage"]
                .dropna()
                .astype(str)
                .unique()
            )
        ),
        "minimum_collection_date": daily["collection_date"].min(),
        "maximum_collection_date": daily["collection_date"].max(),
    }
    return pd.DataFrame(
        {"metric": list(metrics.keys()), "value": list(metrics.values())}
    )


def prepare_diet_data(
    input_csv: Path,
    output_dir: Path,
    chunk_size: int,
) -> Tuple[Path, Path, Path]:
    """执行分块读取、日级汇总、参与者级汇总和结果保存。"""
    if not input_csv.is_file():
        raise FileNotFoundError(f"找不到输入文件：{input_csv}")

    header = pd.read_csv(input_csv, encoding="utf-8-sig", nrows=0)
    validate_columns(header.columns.tolist())

    partials = []
    accumulated_metrics: Dict[str, int] = {}
    reader = pd.read_csv(
        input_csv,
        encoding="utf-8-sig",
        usecols=REQUIRED_COLUMNS,
        chunksize=chunk_size,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        partial, metrics = aggregate_chunk(chunk)
        partials.append(partial)
        for key, value in metrics.items():
            accumulated_metrics[key] = accumulated_metrics.get(key, 0) + value
        print(
            f"已处理第 {chunk_number} 块，"
            f"累计事件 {accumulated_metrics['input_rows']:,} 行"
        )

    if not partials:
        raise ValueError("输入文件只有表头，没有饮食事件记录。")

    daily = combine_daily_partials(partials)
    participant_summary = build_participant_summary(daily)
    qc_table = build_qc_table(
        daily, participant_summary, accumulated_metrics
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    daily_path = output_dir / "diet_daily_summary.csv"
    participant_path = output_dir / "diet_participant_summary.csv"
    qc_path = output_dir / "diet_preparation_qc.csv"

    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    participant_summary.to_csv(
        participant_path, index=False, encoding="utf-8-sig"
    )
    qc_table.to_csv(qc_path, index=False, encoding="utf-8-sig")

    print("-" * 72)
    print(f"输入事件：{accumulated_metrics['input_rows']:,} 行")
    print(f"日级数据：{len(daily):,} 行")
    print(f"参与者-阶段：{len(participant_summary):,} 行")
    print(f"日级结果：{daily_path}")
    print(f"参与者级结果：{participant_path}")
    print(f"QC 结果：{qc_path}")
    return daily_path, participant_path, qc_path


def parse_args():
    parser = ArgumentParser(
        description="将 diet_logging_events.csv 汇总为日级和参与者级数据"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"饮食事件 CSV（默认：{DEFAULT_INPUT_CSV}）",
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_diet_data(args.input, args.output_dir, args.chunk_size)
