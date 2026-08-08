"""从饮食事件表构建 food_id 标准名称字典和冲突清单。"""

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
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "02_food_dictionary"

KEY_COLUMN = "food_id"
LABEL_COLUMNS = ["short_food_name", "product_name", "food_category"]
REQUIRED_COLUMNS = [KEY_COLUMN, *LABEL_COLUMNS, "weight_g"]


def normalize_text(series: pd.Series) -> pd.Series:
    """去除文本首尾空白，并将空字符串视为缺失。"""
    normalized = series.astype("string").str.strip()
    return normalized.mask(normalized.eq(""), pd.NA)


def validate_columns(columns: List[str]) -> None:
    """确认输入表包含构建食物字典所需字段。"""
    missing = sorted(set(REQUIRED_COLUMNS) - set(columns))
    if missing:
        raise ValueError(f"输入文件缺少字段：{', '.join(missing)}")


def aggregate_chunk(chunk: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """按 food_id 和名称组合汇总一个 CSV 数据块。"""
    chunk = chunk.loc[:, REQUIRED_COLUMNS].copy()
    for column in [KEY_COLUMN, *LABEL_COLUMNS]:
        chunk[column] = normalize_text(chunk[column])

    original_weight_nonmissing = chunk["weight_g"].notna()
    chunk["weight_g"] = pd.to_numeric(chunk["weight_g"], errors="coerce")
    invalid_weight_rows = int(
        (original_weight_nonmissing & chunk["weight_g"].isna()).sum()
    )

    missing_food_id_rows = int(chunk[KEY_COLUMN].isna().sum())
    all_labels_missing_rows = int(chunk[LABEL_COLUMNS].isna().all(axis=1).sum())

    usable = chunk[chunk[KEY_COLUMN].notna()].copy()
    grouped = usable.groupby(
        [KEY_COLUMN, *LABEL_COLUMNS], sort=False, dropna=False
    )
    variants = grouped.size().rename("event_count").to_frame()
    variants["weight_g_total"] = grouped["weight_g"].sum(min_count=1)
    variants["weight_g_nonmissing_events"] = grouped["weight_g"].count()
    variants = variants.reset_index()

    metrics = {
        "input_rows": len(chunk),
        "missing_food_id_rows": missing_food_id_rows,
        "all_food_labels_missing_rows": all_labels_missing_rows,
        "invalid_numeric_weight_g": invalid_weight_rows,
    }
    return variants, metrics


def combine_variants(partials: List[pd.DataFrame]) -> pd.DataFrame:
    """合并跨 CSV 分块的 food_id 名称组合。"""
    combined = pd.concat(partials, ignore_index=True)
    group_columns = [KEY_COLUMN, *LABEL_COLUMNS]
    grouped = combined.groupby(group_columns, sort=True, dropna=False)

    variants = grouped[
        ["event_count", "weight_g_total", "weight_g_nonmissing_events"]
    ].sum(min_count=1)
    variants = variants.reset_index()
    variants["label_nonmissing_count"] = variants[LABEL_COLUMNS].notna().sum(
        axis=1
    )

    food_totals = variants.groupby(KEY_COLUMN)["event_count"].transform("sum")
    variants["food_id_event_share"] = variants["event_count"] / food_totals

    sort_columns = [
        KEY_COLUMN,
        "label_nonmissing_count",
        "event_count",
        "short_food_name",
        "product_name",
        "food_category",
    ]
    ascending = [True, False, False, True, True, True]
    variants = variants.sort_values(
        sort_columns, ascending=ascending, na_position="last"
    ).reset_index(drop=True)
    return variants


def build_dictionary(variants: pd.DataFrame) -> pd.DataFrame:
    """为每个 food_id 选择标准名称并统计名称冲突。"""
    grouped = variants.groupby(KEY_COLUMN, sort=True, dropna=False)
    dictionary = grouped.agg(
        food_id_event_count=("event_count", "sum"),
        food_id_weight_g_total=("weight_g_total", "sum"),
        observed_label_combinations=("event_count", "size"),
        short_food_name_variants=("short_food_name", "nunique"),
        product_name_variants=("product_name", "nunique"),
        food_category_variants=("food_category", "nunique"),
    )

    canonical = variants.drop_duplicates(KEY_COLUMN, keep="first").set_index(
        KEY_COLUMN
    )
    canonical = canonical[
        [
            "short_food_name",
            "product_name",
            "food_category",
            "event_count",
            "food_id_event_share",
            "label_nonmissing_count",
        ]
    ].rename(
        columns={
            "short_food_name": "canonical_short_food_name",
            "product_name": "canonical_product_name",
            "food_category": "canonical_food_category",
            "event_count": "canonical_combination_event_count",
            "food_id_event_share": "canonical_combination_event_share",
            "label_nonmissing_count": "canonical_label_nonmissing_count",
        }
    )

    dictionary = dictionary.join(canonical).reset_index()
    conflict_columns = [
        "short_food_name_variants",
        "product_name_variants",
        "food_category_variants",
    ]
    dictionary["has_label_conflict"] = dictionary[conflict_columns].gt(1).any(
        axis=1
    )
    dictionary["canonical_mapping_complete"] = dictionary[
        "canonical_label_nonmissing_count"
    ].eq(len(LABEL_COLUMNS))

    output_columns = [
        KEY_COLUMN,
        "canonical_short_food_name",
        "canonical_product_name",
        "canonical_food_category",
        "food_id_event_count",
        "food_id_weight_g_total",
        "canonical_combination_event_count",
        "canonical_combination_event_share",
        "observed_label_combinations",
        "short_food_name_variants",
        "product_name_variants",
        "food_category_variants",
        "canonical_mapping_complete",
        "has_label_conflict",
    ]
    return dictionary.loc[:, output_columns]


def build_category_summary(variants: pd.DataFrame) -> pd.DataFrame:
    """统计原始 food_category 的事件数、食物数和重量。"""
    grouped = variants.groupby("food_category", sort=True, dropna=False)
    summary = grouped.agg(
        event_count=("event_count", "sum"),
        unique_food_ids=(KEY_COLUMN, "nunique"),
        weight_g_total=("weight_g_total", "sum"),
        weight_g_nonmissing_events=("weight_g_nonmissing_events", "sum"),
    ).reset_index()
    summary["event_share"] = summary["event_count"] / summary[
        "event_count"
    ].sum()
    return summary.sort_values(
        ["event_count", "food_category"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)


def build_qc_table(
    variants: pd.DataFrame,
    dictionary: pd.DataFrame,
    accumulated_metrics: Dict[str, int],
) -> pd.DataFrame:
    """生成食物字典整体 QC 指标。"""
    metrics = {
        **accumulated_metrics,
        "unique_food_ids": dictionary[KEY_COLUMN].nunique(),
        "observed_label_combinations": len(variants),
        "food_ids_with_label_conflict": int(
            dictionary["has_label_conflict"].sum()
        ),
        "food_ids_with_complete_canonical_mapping": int(
            dictionary["canonical_mapping_complete"].sum()
        ),
        "food_ids_with_incomplete_canonical_mapping": int(
            (~dictionary["canonical_mapping_complete"]).sum()
        ),
    }
    return pd.DataFrame(
        {"metric": list(metrics.keys()), "value": list(metrics.values())}
    )


def build_food_dictionary(
    input_csv: Path,
    output_dir: Path,
    chunk_size: int,
) -> Tuple[Path, Path, Path, Path]:
    """执行分块汇总并保存食物字典、冲突和类别统计。"""
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
        dtype={
            KEY_COLUMN: "string",
            **{column: "string" for column in LABEL_COLUMNS},
        },
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

    variants = combine_variants(partials)
    dictionary = build_dictionary(variants)
    conflict_ids = dictionary.loc[
        dictionary["has_label_conflict"], [KEY_COLUMN]
    ]
    conflicts = variants.merge(conflict_ids, on=KEY_COLUMN, how="inner")
    conflicts = conflicts.sort_values(
        [KEY_COLUMN, "event_count"], ascending=[True, False]
    )
    category_summary = build_category_summary(variants)
    qc_table = build_qc_table(variants, dictionary, accumulated_metrics)

    output_dir.mkdir(parents=True, exist_ok=True)
    dictionary_path = output_dir / "food_id_dictionary.csv"
    conflicts_path = output_dir / "food_id_conflicts.csv"
    category_path = output_dir / "food_category_summary.csv"
    qc_path = output_dir / "food_dictionary_qc.csv"

    dictionary.to_csv(dictionary_path, index=False, encoding="utf-8-sig")
    conflicts.to_csv(conflicts_path, index=False, encoding="utf-8-sig")
    category_summary.to_csv(category_path, index=False, encoding="utf-8-sig")
    qc_table.to_csv(qc_path, index=False, encoding="utf-8-sig")

    print("-" * 72)
    print(f"输入事件：{accumulated_metrics['input_rows']:,} 行")
    print(f"唯一 food_id：{len(dictionary):,} 个")
    print(
        "存在名称或类别冲突："
        f"{int(dictionary['has_label_conflict'].sum()):,} 个 food_id"
    )
    print(f"食物字典：{dictionary_path}")
    print(f"冲突清单：{conflicts_path}")
    print(f"类别统计：{category_path}")
    print(f"QC 结果：{qc_path}")
    return dictionary_path, conflicts_path, category_path, qc_path


def parse_args():
    parser = ArgumentParser(
        description="从 diet_logging_events.csv 构建 food_id 标准名称字典"
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
    build_food_dictionary(args.input, args.output_dir, args.chunk_size)
