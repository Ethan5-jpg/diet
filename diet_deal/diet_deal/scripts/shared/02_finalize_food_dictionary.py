"""生成已人工确认的 food_category 覆盖表。"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DEFAULT_DICTIONARY_CSV = (
    PROJECT_DIR
    / "outputs"
    / "02_food_dictionary"
    / "food_id_dictionary.csv"
)
DEFAULT_OUTPUT_CSV = (
    PROJECT_DIR
    / "outputs"
    / "02_food_dictionary"
    / "food_category_overrides.csv"
)

NAME_COLUMNS = [
    "canonical_short_food_name",
    "canonical_product_name",
]
REQUIRED_COLUMNS = [
    "food_id",
    *NAME_COLUMNS,
    "canonical_food_category",
    "food_id_event_count",
]

REVIEWED_CATEGORIES: Dict[str, Tuple[str, str]] = {
    "1011465": (
        "milk, cream cheese and yogurts",
        "Existing coconut-milk drinks use this category.",
    ),
    "1009040": (
        "Oils and fats",
        "The item is pure sesame cooking oil.",
    ),
    "1012992": (
        "Industrialized vegetarian food ready to eat",
        "Generic protein powders predominantly use this category.",
    ),
    "1012247": (
        "sweet milk products",
        "The item is a vanilla-flavored milk drink.",
    ),
    "1013278": (
        "Industrialized vegetarian food ready to eat",
        "Generic protein powders predominantly use this category.",
    ),
    "1012144": (
        "Vegetables",
        "Existing plain vegetable soups use this category.",
    ),
    "1010729": (
        "fruit juices and soft drinks",
        "Wheatgrass is recorded as juice rather than whole vegetables.",
    ),
    "1008612": (
        "Fish and seafood",
        "Matched locust-fish and grouper dishes use this category.",
    ),
    "1010647": (
        "Drinks",
        "The product is coconut water rather than coconut cream.",
    ),
    "1012232": (
        "Industrialized vegetarian food ready to eat",
        "Generic protein powders predominantly use this category.",
    ),
    "1010477": (
        "Canned veg and fruits",
        "The product is explicitly canned artichoke with oil.",
    ),
    "1013656": (
        "Snacks",
        "The product is explicitly a roasted seaweed snack.",
    ),
}


def validate_columns(columns: List[str]) -> None:
    """确认食物字典包含生成覆盖表所需字段。"""
    missing = sorted(set(REQUIRED_COLUMNS) - set(columns))
    if missing:
        raise ValueError(f"食物字典缺少字段：{', '.join(missing)}")


def build_overrides(dictionary: pd.DataFrame) -> pd.DataFrame:
    """从完整食物字典提取12项并附加人工确认类别。"""
    dictionary = dictionary.loc[:, REQUIRED_COLUMNS].copy()
    dictionary["food_id"] = dictionary["food_id"].astype("string").str.strip()

    reviewed_ids = set(REVIEWED_CATEGORIES)
    overrides = dictionary[dictionary["food_id"].isin(reviewed_ids)].copy()
    observed_ids = set(overrides["food_id"].dropna().tolist())
    missing_ids = sorted(reviewed_ids - observed_ids)
    if missing_ids:
        raise ValueError(f"食物字典缺少预期 food_id：{', '.join(missing_ids)}")

    duplicated = overrides["food_id"].duplicated(keep=False)
    if duplicated.any():
        duplicated_ids = sorted(overrides.loc[duplicated, "food_id"].unique())
        raise ValueError(f"食物字典存在重复 food_id：{', '.join(duplicated_ids)}")

    category_map = {
        food_id: result[0]
        for food_id, result in REVIEWED_CATEGORIES.items()
    }
    basis_map = {
        food_id: result[1]
        for food_id, result in REVIEWED_CATEGORIES.items()
    }
    overrides["assigned_food_category"] = overrides["food_id"].map(
        category_map
    )
    overrides["review_status"] = "reviewed"
    overrides["review_basis"] = overrides["food_id"].map(basis_map)

    output_columns = [
        "food_id",
        *NAME_COLUMNS,
        "food_id_event_count",
        "assigned_food_category",
        "review_status",
        "review_basis",
    ]
    return overrides.loc[:, output_columns].sort_values(
        ["food_id_event_count", "food_id"],
        ascending=[False, True],
    ).reset_index(drop=True)


def finalize_food_dictionary(dictionary_csv: Path, output_csv: Path) -> Path:
    """验证食物字典并保存最终类别覆盖表。"""
    if not dictionary_csv.is_file():
        raise FileNotFoundError(f"找不到食物字典：{dictionary_csv}")

    header = pd.read_csv(dictionary_csv, encoding="utf-8-sig", nrows=0)
    validate_columns(header.columns.tolist())
    dictionary = pd.read_csv(
        dictionary_csv,
        encoding="utf-8-sig",
        dtype={"food_id": "string"},
        low_memory=False,
    )
    overrides = build_overrides(dictionary)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    overrides.to_csv(output_csv, index=False, encoding="utf-8-sig")

    covered_events = pd.to_numeric(
        overrides["food_id_event_count"], errors="coerce"
    ).sum()
    print("=" * 72)
    print(f"已审核 food_id：{len(overrides):,} 个")
    print(f"覆盖饮食事件：{covered_events:,.0f} 条")
    print(f"类别覆盖表：{output_csv}")
    print("-" * 72)
    print(
        overrides[
            [
                "food_id",
                "canonical_short_food_name",
                "assigned_food_category",
            ]
        ].to_string(index=False)
    )
    return output_csv


def parse_args():
    parser = ArgumentParser(description="生成食物类别人工审核覆盖表")
    parser.add_argument(
        "--dictionary",
        type=Path,
        default=DEFAULT_DICTIONARY_CSV,
        help=f"food_id 字典（默认：{DEFAULT_DICTIONARY_CSV}）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"类别覆盖表输出路径（默认：{DEFAULT_OUTPUT_CSV}）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    finalize_food_dictionary(args.dictionary, args.output)
