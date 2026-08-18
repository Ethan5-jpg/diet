"""生成 AMED 的 food_id 级候选映射和映射质量控制结果。"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DEFAULT_DICTIONARY_CSV = (
    PROJECT_DIR
    / "outputs"
    / "02_food_dictionary"
    / "food_id_dictionary.csv"
)
DEFAULT_OVERRIDES_CSV = (
    PROJECT_DIR
    / "outputs"
    / "02_food_dictionary"
    / "food_category_overrides.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "outputs" / "03_score_mapping" / "amed"
)

NAME_COLUMNS = [
    "canonical_short_food_name",
    "canonical_product_name",
]
REQUIRED_DICTIONARY_COLUMNS = [
    "food_id",
    *NAME_COLUMNS,
    "canonical_food_category",
    "food_id_event_count",
    "food_id_weight_g_total",
]
REQUIRED_OVERRIDE_COLUMNS = ["food_id", "assigned_food_category"]

DIRECT_CATEGORY_MAP: Dict[str, str] = {
    "fruits": "fruit",
    "vegetables": "vegetables",
    "bread wholewheat": "whole_grains",
    "pasta, grains and side dishes wholewheat": "whole_grains",
    "pulses and products": "legumes",
    "fish and seafood": "fish",
    "beef, veal, lamb, and other meat products": "red_processed_meat",
    "processed meat products": "red_processed_meat",
}

REVIEW_CATEGORIES = {
    "pasta, grains and side dishes",
    "bread",
    "nuts, seeds, and products",
    "soups and sauces",
    "canned veg and fruits",
    "fruit juices and soft drinks",
    "others",
    "cereals",
    "snacks",
    "baked goods",
    "fast foods",
    "industrialized vegetarian food ready to eat",
    "deep fried foods",
}

NOT_APPLICABLE_CATEGORIES = {
    "drinks",
    "sweets",
    "milk, cream cheese and yogurts",
    "med oil and fats",
    "poultry and its products",
    "hard cheese",
    "eggs and their products",
    "oils and fats",
    "alcoholic drinks",
    "spices and herbs",
    "sweet milk products",
    "low calories and diet drinks",
}

COMPONENT_KEYWORDS: List[Tuple[str, Tuple[str, ...]]] = [
    (
        "whole_grains",
        (
            "whole wheat",
            "wholewheat",
            "whole grain",
            "wholegrain",
            "brown rice",
            "oat",
            "bran",
            "rye",
            "bulgur",
            "quinoa",
            "barley",
            "popcorn",
        ),
    ),
    (
        "nuts",
        (
            "nut",
            "almond",
            "walnut",
            "cashew",
            "pistachio",
            "hazelnut",
            "pecan",
            "peanut butter",
        ),
    ),
    (
        "legumes",
        (
            "tofu",
            "bean",
            "lentil",
            "chickpea",
            "hummus",
            "green pea",
            "soy",
        ),
    ),
    (
        "fish",
        (
            "fish",
            "salmon",
            "tuna",
            "sardine",
            "shrimp",
            "prawn",
            "grouper",
            "tilapia",
            "cod",
            "seafood",
        ),
    ),
    (
        "red_processed_meat",
        (
            "beef",
            "veal",
            "lamb",
            "pork",
            "bacon",
            "sausage",
            "hot dog",
            "deli meat",
            "hamburger",
            "pastrami",
            "salami",
        ),
    ),
]

FRUIT_KEYWORDS = (
    "fruit",
    "apple",
    "pear",
    "peach",
    "apricot",
    "plum",
    "cherry",
    "grape",
    "berry",
    "strawberry",
    "blueberry",
    "raspberry",
    "blackberry",
    "cranberry",
    "orange",
    "grapefruit",
    "mandarin",
    "clementine",
    "banana",
    "mango",
    "pineapple",
    "melon",
    "date",
    "fig",
    "pomegranate",
    "kiwi",
)

VEGETABLE_KEYWORDS = (
    "vegetable",
    "artichoke",
    "asparagus",
    "beet",
    "broccoli",
    "cabbage",
    "carrot",
    "cauliflower",
    "celery",
    "cucumber",
    "eggplant",
    "garlic",
    "leek",
    "lettuce",
    "mushroom",
    "onion",
    "pepper",
    "pumpkin",
    "spinach",
    "squash",
    "tomato",
    "zucchini",
    "corn",
    "olive",
)

SOFT_DRINK_KEYWORDS = (
    "soft drink",
    "cola",
    "soda",
    "energy drink",
    "lemonade",
)

COMPOSITE_NOT_APPLICABLE_CATEGORIES = {
    "soups and sauces",
    "others",
    "snacks",
    "baked goods",
    "fast foods",
    "industrialized vegetarian food ready to eat",
    "deep fried foods",
}


def component_definitions() -> pd.DataFrame:
    """返回论文补充表9中的 AMED 组件和计分规则。"""
    rows = [
        (
            "fruit",
            "servings/day",
            "All fruit and juices",
            "intake > cohort median",
            "intake <= cohort median",
            True,
        ),
        (
            "vegetables",
            "servings/day",
            "All vegetables except potatoes",
            "intake > cohort median",
            "intake <= cohort median",
            True,
        ),
        (
            "whole_grains",
            "servings/day",
            "Whole-grain cereals, dark breads, brown rice and other whole grains",
            "intake > cohort median",
            "intake <= cohort median",
            True,
        ),
        (
            "nuts",
            "servings/day",
            "Nuts and peanut butter",
            "intake > cohort median",
            "intake <= cohort median",
            True,
        ),
        (
            "legumes",
            "servings/day",
            "Tofu, string beans, peas and beans",
            "intake > cohort median",
            "intake <= cohort median",
            True,
        ),
        (
            "red_processed_meat",
            "servings/day",
            "Red and processed meat",
            "intake <= cohort median",
            "intake > cohort median",
            True,
        ),
        (
            "fish",
            "servings/day",
            "Fish, shrimp and breaded fish",
            "intake > cohort median",
            "intake <= cohort median",
            True,
        ),
        (
            "alcohol",
            "g/day",
            "Alcohol from all logged items",
            "women: 5-15; men: 10-25",
            "outside the sex-specific range",
            True,
        ),
        (
            "mufa_sfa_ratio",
            "ratio",
            "Monounsaturated fat / saturated fat",
            "intake > cohort median",
            "intake <= cohort median",
            False,
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "amed_component",
            "paper_unit",
            "paper_food_items",
            "score_1_rule",
            "score_0_rule",
            "available_in_hpp",
        ],
    )


def validate_columns(
    columns: List[str],
    required: List[str],
    table_name: str,
) -> None:
    """确认输入表包含映射所需字段。"""
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(f"{table_name}缺少字段：{', '.join(missing)}")


def normalize_text(series: pd.Series) -> pd.Series:
    """清理文本并将空字符串转换为缺失。"""
    normalized = series.astype("string").str.strip()
    return normalized.mask(normalized.eq(""), pd.NA)


def normalize_category(series: pd.Series) -> pd.Series:
    """清理类别文本，并合并不一致的连续空格。"""
    normalized = normalize_text(series)
    return normalized.str.replace(r"\s+", " ", regex=True)


def category_key(series: pd.Series) -> pd.Series:
    """生成不受大小写、下划线和连续空格影响的类别匹配键。"""
    return (
        normalize_category(series)
        .str.replace("_", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.casefold()
    )


def suggest_component(combined_name: str) -> Optional[str]:
    """依据名称关键词为待审核食物提供候选组件，不直接确认映射。"""
    if not combined_name:
        return None
    if "potato" in combined_name:
        return None
    for component, keywords in COMPONENT_KEYWORDS:
        if any(keyword in combined_name for keyword in keywords):
            return component
    return None


def contains_any(series: pd.Series, keywords: Tuple[str, ...]) -> pd.Series:
    """判断每条标准化名称是否包含任一关键词。"""
    mask = pd.Series(False, index=series.index)
    for keyword in keywords:
        mask |= series.str.contains(keyword, regex=False, na=False)
    return mask


def assign_component(
    mapping: pd.DataFrame,
    mask: pd.Series,
    component: str,
    basis: str,
) -> None:
    """将满足保守规则的 food_id 确认为 AMED 组件。"""
    mapping.loc[mask, "amed_component"] = component
    mapping.loc[mask, "mapping_status"] = "mapped"
    mapping.loc[mask, "mapping_basis"] = basis


def exclude_from_components(
    mapping: pd.DataFrame,
    mask: pd.Series,
    basis: str,
) -> None:
    """将明确不属于 AMED 食物组的 food_id 标记为不适用。"""
    mapping.loc[mask, "amed_component"] = pd.NA
    mapping.loc[mask, "mapping_status"] = "not_applicable"
    mapping.loc[mask, "mapping_basis"] = basis


def resolve_categories(
    dictionary: pd.DataFrame,
    overrides: pd.DataFrame,
) -> pd.DataFrame:
    """使用人工覆盖表补齐已审核的12个食物类别。"""
    dictionary = dictionary.copy()
    overrides = overrides.loc[:, REQUIRED_OVERRIDE_COLUMNS].copy()
    overrides["food_id"] = normalize_text(overrides["food_id"])
    if overrides["food_id"].duplicated().any():
        raise ValueError("类别覆盖表存在重复 food_id。")

    override_map = overrides.set_index("food_id")["assigned_food_category"]
    dictionary["reviewed_food_category"] = dictionary["food_id"].map(
        override_map
    )
    dictionary["resolved_food_category"] = dictionary[
        "reviewed_food_category"
    ].fillna(dictionary["canonical_food_category"])
    dictionary["category_source"] = "original"
    dictionary.loc[
        dictionary["reviewed_food_category"].notna(), "category_source"
    ] = "manual_override"
    dictionary.loc[
        dictionary["resolved_food_category"].isna(), "category_source"
    ] = "missing"
    return dictionary


def build_amed_mapping(dictionary: pd.DataFrame) -> pd.DataFrame:
    """按照直接类别规则生成 AMED food_id 级候选映射。"""
    mapping = dictionary.copy()
    combined_names = (
        mapping[NAME_COLUMNS].fillna("").astype(str).agg(" ".join, axis=1)
    ).str.casefold()
    mapping["resolved_food_category_key"] = category_key(
        mapping["resolved_food_category"]
    )

    mapping["amed_component"] = mapping[
        "resolved_food_category_key"
    ].map(DIRECT_CATEGORY_MAP)
    mapping["suggested_amed_component"] = pd.NA
    mapping["mapping_status"] = "review_required"
    mapping["mapping_basis"] = "category_not_in_direct_rules"

    direct = mapping["amed_component"].notna()
    mapping.loc[direct, "mapping_status"] = "mapped"
    mapping.loc[direct, "mapping_basis"] = "direct_food_category"

    original_category_text = normalize_category(
        mapping["resolved_food_category"]
    ).fillna("").str.casefold()
    processed_meat_category = (
        original_category_text.str.contains(
            "processed", regex=False, na=False
        )
        & original_category_text.str.contains(
            "meat", regex=False, na=False
        )
        & original_category_text.str.contains(
            "product", regex=False, na=False
        )
    )
    assign_component(
        mapping,
        processed_meat_category,
        "red_processed_meat",
        "processed_meat_category_words",
    )

    missing_category = mapping["resolved_food_category"].isna()
    mapping.loc[missing_category, "mapping_status"] = "unmapped_missing_labels"
    mapping.loc[missing_category, "mapping_basis"] = "no_name_or_category"

    not_applicable = mapping["resolved_food_category_key"].isin(
        NOT_APPLICABLE_CATEGORIES
    )
    mapping.loc[not_applicable, "mapping_status"] = "not_applicable"
    mapping.loc[not_applicable, "mapping_basis"] = (
        "category_outside_amed_food_groups"
    )

    potatoes = combined_names.str.contains("potato", regex=False, na=False)
    vegetable_potatoes = potatoes & mapping["amed_component"].eq("vegetables")
    mapping.loc[vegetable_potatoes, "amed_component"] = pd.NA
    mapping.loc[vegetable_potatoes, "mapping_status"] = "not_applicable"
    mapping.loc[vegetable_potatoes, "mapping_basis"] = (
        "potatoes_excluded_from_amed_vegetables"
    )

    review = mapping["resolved_food_category_key"].isin(REVIEW_CATEGORIES)
    suggestions = combined_names.loc[review].map(suggest_component)
    mapping.loc[review, "suggested_amed_component"] = suggestions
    mapping.loc[review, "mapping_status"] = "review_required"
    mapping.loc[review, "mapping_basis"] = "mixed_or_ambiguous_food_category"

    category_keys = mapping["resolved_food_category_key"]

    nuts_category = category_keys.eq("nuts, seeds, and products")
    nut_items = nuts_category & mapping["suggested_amed_component"].eq("nuts")
    assign_component(
        mapping,
        nut_items,
        "nuts",
        "nut_name_within_nuts_and_seeds_category",
    )
    exclude_from_components(
        mapping,
        nuts_category & ~nut_items,
        "seed_or_other_product_not_in_amed_nuts_definition",
    )

    grain_review_categories = category_keys.isin(
        {
            "bread",
            "pasta, grains and side dishes",
            "cereals",
        }
    )
    explicit_whole_grains = grain_review_categories & mapping[
        "suggested_amed_component"
    ].eq("whole_grains")
    assign_component(
        mapping,
        explicit_whole_grains,
        "whole_grains",
        "explicit_whole_grain_name_in_grain_category",
    )
    exclude_from_components(
        mapping,
        grain_review_categories & ~explicit_whole_grains,
        "refined_or_composite_grain_without_whole_grain_evidence",
    )

    juice_category = category_keys.eq("fruit juices and soft drinks")
    explicit_juice = contains_any(combined_names, ("juice",))
    explicit_soft_drink = contains_any(
        combined_names, SOFT_DRINK_KEYWORDS
    )
    fruit_juice = juice_category & explicit_juice & ~explicit_soft_drink
    assign_component(
        mapping,
        fruit_juice,
        "fruit",
        "explicit_juice_name_in_juice_and_soft_drink_category",
    )
    exclude_from_components(
        mapping,
        juice_category & ~fruit_juice,
        "soft_drink_or_non_juice_beverage",
    )

    canned_category = category_keys.eq("canned veg and fruits")
    canned_potato = canned_category & contains_any(
        combined_names, ("potato",)
    )
    canned_fruit = (
        canned_category
        & contains_any(combined_names, FRUIT_KEYWORDS)
        & ~canned_potato
    )
    canned_vegetables = (
        canned_category
        & contains_any(combined_names, VEGETABLE_KEYWORDS)
        & ~canned_fruit
        & ~canned_potato
    )
    assign_component(
        mapping,
        canned_fruit,
        "fruit",
        "explicit_fruit_name_in_canned_category",
    )
    assign_component(
        mapping,
        canned_vegetables,
        "vegetables",
        "explicit_vegetable_name_in_canned_category",
    )
    exclude_from_components(
        mapping,
        canned_potato,
        "potatoes_excluded_from_amed_vegetables",
    )

    unresolved_canned = canned_category & mapping["mapping_status"].eq(
        "review_required"
    )
    exclude_from_components(
        mapping,
        unresolved_canned,
        "canned_item_without_reliable_fruit_or_vegetable_name",
    )

    composite_categories = category_keys.isin(
        COMPOSITE_NOT_APPLICABLE_CATEGORIES
    )
    exclude_from_components(
        mapping,
        composite_categories,
        "composite_food_weight_cannot_be_reliably_decomposed",
    )

    final_category_text = normalize_category(
        mapping["resolved_food_category"]
    ).fillna("").str.casefold()
    final_processed_meat = final_category_text.eq("processed meat products")
    assign_component(
        mapping,
        final_processed_meat,
        "red_processed_meat",
        "final_processed_meat_category_check",
    )

    output_columns = [
        "food_id",
        *NAME_COLUMNS,
        "canonical_food_category",
        "reviewed_food_category",
        "resolved_food_category",
        "resolved_food_category_key",
        "category_source",
        "food_id_event_count",
        "food_id_weight_g_total",
        "amed_component",
        "suggested_amed_component",
        "mapping_status",
        "mapping_basis",
    ]
    return mapping.loc[:, output_columns].sort_values(
        ["mapping_status", "resolved_food_category", "food_id_event_count"],
        ascending=[True, True, False],
        na_position="last",
    ).reset_index(drop=True)


def build_qc(mapping: pd.DataFrame) -> pd.DataFrame:
    """汇总映射状态、已映射组件和待审核类别。"""
    status = (
        mapping.groupby("mapping_status", dropna=False)
        .agg(
            food_id_count=("food_id", "nunique"),
            event_count=("food_id_event_count", "sum"),
        )
        .reset_index()
    )
    status.insert(0, "summary_type", "mapping_status")
    status["amed_component"] = ""
    status["food_category"] = ""

    components = (
        mapping[mapping["mapping_status"].eq("mapped")]
        .groupby("amed_component", dropna=False)
        .agg(
            food_id_count=("food_id", "nunique"),
            event_count=("food_id_event_count", "sum"),
        )
        .reset_index()
    )
    components.insert(0, "summary_type", "mapped_component")
    components["mapping_status"] = "mapped"
    components["food_category"] = ""

    review = (
        mapping[mapping["mapping_status"].eq("review_required")]
        .groupby("resolved_food_category", dropna=False)
        .agg(
            food_id_count=("food_id", "nunique"),
            event_count=("food_id_event_count", "sum"),
        )
        .reset_index()
        .rename(columns={"resolved_food_category": "food_category"})
    )
    review.insert(0, "summary_type", "review_category")
    review["mapping_status"] = "review_required"
    review["amed_component"] = ""

    suggestions = (
        mapping[mapping["mapping_status"].eq("review_required")]
        .groupby(
            ["resolved_food_category", "suggested_amed_component"],
            dropna=False,
        )
        .agg(
            food_id_count=("food_id", "nunique"),
            event_count=("food_id_event_count", "sum"),
        )
        .reset_index()
        .rename(
            columns={
                "resolved_food_category": "food_category",
                "suggested_amed_component": "amed_component",
            }
        )
    )
    suggestions.insert(0, "summary_type", "review_suggestion")
    suggestions["mapping_status"] = "review_required"

    output_columns = [
        "summary_type",
        "mapping_status",
        "amed_component",
        "food_category",
        "food_id_count",
        "event_count",
    ]
    return pd.concat(
        [status, components, review, suggestions], ignore_index=True
    ).loc[:, output_columns]


def prepare_amed_mapping(
    dictionary_csv: Path,
    overrides_csv: Path,
    output_dir: Path,
) -> Tuple[Path, Path, Path]:
    """读取食物词典并保存 AMED 定义、候选映射和 QC。"""
    if not dictionary_csv.is_file():
        raise FileNotFoundError(f"找不到食物字典：{dictionary_csv}")
    if not overrides_csv.is_file():
        raise FileNotFoundError(f"找不到类别覆盖表：{overrides_csv}")

    dictionary_header = pd.read_csv(
        dictionary_csv, encoding="utf-8-sig", nrows=0
    )
    overrides_header = pd.read_csv(
        overrides_csv, encoding="utf-8-sig", nrows=0
    )
    validate_columns(
        dictionary_header.columns.tolist(),
        REQUIRED_DICTIONARY_COLUMNS,
        "食物字典",
    )
    validate_columns(
        overrides_header.columns.tolist(),
        REQUIRED_OVERRIDE_COLUMNS,
        "类别覆盖表",
    )

    dictionary = pd.read_csv(
        dictionary_csv,
        encoding="utf-8-sig",
        dtype={"food_id": "string"},
        low_memory=False,
    )
    overrides = pd.read_csv(
        overrides_csv,
        encoding="utf-8-sig",
        dtype={"food_id": "string"},
        low_memory=False,
    )
    for column in ["food_id", *NAME_COLUMNS]:
        dictionary[column] = normalize_text(dictionary[column])
    dictionary["canonical_food_category"] = normalize_category(
        dictionary["canonical_food_category"]
    )
    overrides["assigned_food_category"] = normalize_category(
        overrides["assigned_food_category"]
    )
    dictionary["food_id_event_count"] = pd.to_numeric(
        dictionary["food_id_event_count"], errors="coerce"
    ).fillna(0)

    resolved = resolve_categories(dictionary, overrides)
    mapping = build_amed_mapping(resolved)
    definitions = component_definitions()
    qc = build_qc(mapping)

    output_dir.mkdir(parents=True, exist_ok=True)
    definitions_path = output_dir / "amed_component_definitions.csv"
    mapping_path = output_dir / "amed_food_id_mapping.csv"
    qc_path = output_dir / "amed_mapping_qc.csv"
    definitions.to_csv(definitions_path, index=False, encoding="utf-8-sig")
    mapping.to_csv(mapping_path, index=False, encoding="utf-8-sig")
    qc.to_csv(qc_path, index=False, encoding="utf-8-sig")

    status_summary = qc[qc["summary_type"].eq("mapping_status")].copy()
    total_events = status_summary["event_count"].sum()
    status_summary["event_share"] = (
        status_summary["event_count"] / total_events
    )
    print("=" * 72)
    print("AMED FOOD-ID MAPPING QC")
    print(status_summary.to_string(index=False))
    print("-" * 72)
    print("待审核类别：")
    review_summary = qc[qc["summary_type"].eq("review_category")].sort_values(
        "event_count", ascending=False
    )
    print(review_summary.to_string(index=False))
    print("-" * 72)
    print("待审核项的关键词候选（按事件数排序，前30项）：")
    suggestion_summary = qc[
        qc["summary_type"].eq("review_suggestion")
    ].sort_values("event_count", ascending=False)
    print(suggestion_summary.head(30).to_string(index=False))
    print("-" * 72)
    print(f"组件定义：{definitions_path}")
    print(f"food_id 映射：{mapping_path}")
    print(f"QC 结果：{qc_path}")
    return definitions_path, mapping_path, qc_path


def parse_args():
    parser = ArgumentParser(description="生成 AMED food_id 候选映射")
    parser.add_argument(
        "--dictionary",
        type=Path,
        default=DEFAULT_DICTIONARY_CSV,
        help=f"food_id 字典（默认：{DEFAULT_DICTIONARY_CSV}）",
    )
    parser.add_argument(
        "--overrides",
        type=Path,
        default=DEFAULT_OVERRIDES_CSV,
        help=f"人工类别覆盖表（默认：{DEFAULT_OVERRIDES_CSV}）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录（默认：{DEFAULT_OUTPUT_DIR}）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_amed_mapping(args.dictionary, args.overrides, args.output_dir)
