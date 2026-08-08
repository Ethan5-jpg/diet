"""生成 rEDIH 的 food_id 级候选映射、人工审核表和映射 QC。"""

import re
import unicodedata
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


SCRIPT_VERSION = "2026-08-08-redih-mapping-v2"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DEFAULT_DICTIONARY_CSV = (
    PROJECT_DIR / "outputs" / "02_food_dictionary" / "food_id_dictionary.csv"
)
DEFAULT_CATEGORY_OVERRIDES_CSV = (
    PROJECT_DIR / "outputs" / "02_food_dictionary" / "food_category_overrides.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "03_score_mapping" / "redih"
DEFAULT_COMPONENT_OVERRIDES_CSV = (
    DEFAULT_OUTPUT_DIR / "redih_component_overrides.csv"
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
REQUIRED_CATEGORY_OVERRIDE_COLUMNS = ["food_id", "assigned_food_category"]
REQUIRED_COMPONENT_OVERRIDE_COLUMNS = [
    "food_id",
    "mapping_status",
    "redih_component",
    "review_notes",
]

VALID_MAPPING_STATUSES = {
    "mapped",
    "not_applicable",
    "review_required",
    "unmapped_missing_labels",
}


def component_definitions() -> pd.DataFrame:
    """返回标准18组、HPP可用性和当前17组评分决定。"""
    rows = [
        ("red_meat", 0.250, "positive", True, True, ""),
        ("low_energy_beverages", 0.053, "positive", True, True, ""),
        ("cream_soups", 0.787, "positive", True, True, ""),
        ("processed_meat", 0.199, "positive", True, True, ""),
        (
            "margarine",
            0.054,
            "positive",
            False,
            False,
            "not measured in HPP",
        ),
        ("poultry", 0.183, "positive", True, True, ""),
        ("butter", 0.094, "positive", True, True, ""),
        ("french_fries", 0.581, "positive", True, True, ""),
        ("other_fish", 0.172, "positive", True, True, ""),
        ("high_energy_drinks", 0.104, "positive", True, True, ""),
        ("tomatoes", 0.095, "positive", True, True, ""),
        ("low_fat_dairy", 0.025, "positive", True, True, ""),
        ("eggs", 0.124, "positive", True, True, ""),
        (
            "wine",
            -0.165,
            "inverse",
            True,
            True,
            "",
        ),
        ("coffee", -0.035, "inverse", True, True, ""),
        ("whole_fruits", -0.029, "inverse", True, True, ""),
        ("high_fat_dairy", -0.046, "inverse", True, True, ""),
        ("green_leafy_vegetables", -0.055, "inverse", True, True, ""),
    ]
    definitions = pd.DataFrame(
        rows,
        columns=[
            "redih_component",
            "edih_weight_per_paper_table",
            "edih_association_direction",
            "available_in_hpp",
            "included_in_current_score",
            "current_exclusion_reason",
        ],
    )
    definitions["current_intake_unit"] = "g/day"
    definitions["current_score_contribution"] = (
        "mean_daily_component_g * edih_weight_per_paper_table"
    )
    definitions["redih_direction_transform"] = "rEDIH = -1 * EDIH"
    return definitions


COMPONENT_DEFINITIONS = component_definitions()
AVAILABLE_COMPONENTS: Set[str] = set(
    COMPONENT_DEFINITIONS.loc[
        COMPONENT_DEFINITIONS["available_in_hpp"], "redih_component"
    ]
)
CURRENT_SCORE_COMPONENTS: Set[str] = set(
    COMPONENT_DEFINITIONS.loc[
        COMPONENT_DEFINITIONS["included_in_current_score"], "redih_component"
    ]
)
COMPONENT_WEIGHTS: Dict[str, float] = dict(
    zip(
        COMPONENT_DEFINITIONS["redih_component"],
        COMPONENT_DEFINITIONS["edih_weight_per_paper_table"],
    )
)

DIRECT_CATEGORY_MAP: Dict[str, str] = {
    "beef, veal, lamb, and other meat products": "red_meat",
    "processed meat products": "processed_meat",
    "poultry and its products": "poultry",
    "eggs and their products": "eggs",
    "fruits": "whole_fruits",
    "low calories and diet drinks": "low_energy_beverages",
}

DAIRY_CATEGORIES = {
    "milk, cream cheese and yogurts",
    "hard cheese",
    "sweet milk products",
}
FISH_CATEGORIES = {"fish and seafood"}
BEVERAGE_CATEGORIES = {
    "drinks",
    "fruit juices and soft drinks",
    "alcoholic drinks",
    "low calories and diet drinks",
}
VEGETABLE_CATEGORIES = {
    "vegetables",
    "canned veg and fruits",
}
COMPOSITE_REVIEW_CATEGORIES = {
    "soups and sauces",
    "others",
    "fast foods",
    "deep fried foods",
    "industrialized vegetarian food ready to eat",
    "snacks",
}

LOW_ENERGY_PATTERN = re.compile(
    r"\b(?:diet|zero|zero sugar|sugar[ -]?free|low calorie|low[ -]?calorie|"
    r"no sugar|light)\b"
)
BEVERAGE_PATTERN = re.compile(
    r"\b(?:cola|coke|pepsi|soda|soft drink|carbonated|energy drink|"
    r"fruit punch|lemonade|tonic water|beverage)\b"
)
HIGH_ENERGY_PATTERN = re.compile(
    r"\b(?:cola|coke|pepsi|soda|soft drink|energy drink|fruit punch|"
    r"sweetened drink|sweetened beverage|lemonade|nectar)\b"
)
COFFEE_AUTO_PATTERN = re.compile(
    r"^(?:(?:black|filter|filtered|instant|turkish|decaffeinated|decaf) )?"
    r"coffee\b|^(?:espresso|americano)\b"
)
COFFEE_BROAD_PATTERN = re.compile(
    r"\b(?:coffee|espresso|americano|cappuccino|latte|macchiato|mocha)\b"
)
WINE_PATTERN = re.compile(
    r"\b(?:red wine|white wine|rose wine|rosé wine|dry wine|sweet wine|wine)\b"
)
CREAM_SOUP_PATTERN = re.compile(
    r"\b(?:chowder|cream of [a-z ]+ soup|cream soup|creamy [a-z ]*soup)\b"
)
FRENCH_FRIES_PATTERN = re.compile(r"^(?:french fries?|potato fries?)\b")
BUTTER_PATTERN = re.compile(r"^(?:salted |unsalted |clarified )?butter\b|^ghee\b")
MARGARINE_PATTERN = re.compile(r"\bmargarine\b")
TOMATO_AUTO_PATTERN = re.compile(
    r"^(?:fresh |raw |cooked |canned |cherry |grape )?tomato(?:es)?\b"
    r"|^tomato (?:juice|sauce|paste)\b"
)
TOMATO_BROAD_PATTERN = re.compile(r"\btomato(?:es)?\b")
GREEN_LEAFY_AUTO_PATTERN = re.compile(
    r"^(?:fresh |raw |cooked |frozen |baby )?"
    r"(?:spinach|lettuce|romaine|kale|swiss chard|chard|mustard greens?)\b"
)
GREEN_LEAFY_BROAD_PATTERN = re.compile(
    r"\b(?:spinach|lettuce|romaine|kale|swiss chard|chard|mustard greens?)\b"
)
LOW_FAT_DAIRY_PATTERN = re.compile(
    r"\b(?:skim(?:med)?|non[ -]?fat|fat[ -]?free|low[ -]?fat|reduced[ -]?fat|"
    r"0\s*%|1\s*%|1\.5\s*%|2\s*%)\b"
)
HIGH_FAT_DAIRY_PATTERN = re.compile(
    r"\b(?:whole milk|full[ -]?fat|high[ -]?fat|cream cheese|sour cream|"
    r"whipping cream|whipped cream|heavy cream|ice cream|hard cheese|"
    r"cheddar|parmesan|gouda|brie|camembert|mascarpone)\b"
)
DAIRY_BROAD_PATTERN = re.compile(
    r"\b(?:milk|yogurt|yoghurt|cheese|cream|sherbet|ice milk)\b"
)
OTHER_FISH_AUTO_PATTERN = re.compile(
    r"^(?:canned )?(?:tuna|cod|tilapia|grouper|hake|halibut|sole|flounder|"
    r"shrimp|prawn|lobster|scallop|crab|calamari|squid|octopus)\b"
)
FISH_BROAD_PATTERN = re.compile(
    r"\b(?:fish|seafood|tuna|salmon|sardine|mackerel|herring|trout|cod|"
    r"tilapia|grouper|shrimp|prawn|lobster|scallop|crab|calamari|squid)\b"
)
PROCESSED_MEAT_PATTERN = re.compile(
    r"\b(?:bacon|sausage|hot dog|salami|pastrami|ham|prosciutto|mortadella|"
    r"pepperoni|deli meat|processed meat|corned beef)\b"
)
POULTRY_PATTERN = re.compile(r"\b(?:chicken|turkey|duck|goose|poultry)\b")
RED_MEAT_PATTERN = re.compile(
    r"\b(?:beef|veal|lamb|mutton|pork|hamburger|steak|red meat)\b"
)
EGG_PATTERN = re.compile(r"\b(?:egg|eggs|omelette|omelet)\b")
FRUIT_BROAD_PATTERN = re.compile(
    r"\b(?:apple|pear|banana|orange|grapefruit|mandarin|clementine|grape|"
    r"berry|strawberry|blueberry|raspberry|blackberry|peach|apricot|plum|"
    r"cherry|mango|pineapple|melon|watermelon|cantaloupe|kiwi|fig|date|"
    r"pomegranate|avocado|fruit)\b"
)


def validate_columns(
    columns: List[str], required: List[str], table_name: str
) -> None:
    """确认输入表包含所有必要字段。"""
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError("{}缺少字段：{}".format(table_name, ", ".join(missing)))


def normalize_text(series: pd.Series) -> pd.Series:
    """清理字符串并将空字符串转为缺失。"""
    normalized = series.astype("string").str.strip()
    return normalized.mask(normalized.eq(""), pd.NA)


def normalize_category(series: pd.Series) -> pd.Series:
    """清理类别空白。"""
    return normalize_text(series).str.replace(r"\s+", " ", regex=True)


def fold_text(value: object) -> str:
    """生成适合英文规则匹配的大小写和重音不敏感文本。"""
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.casefold().replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def category_key(value: object) -> str:
    """将单个类别规范化为规则键。"""
    return fold_text(value)


def combine_names(row: pd.Series) -> str:
    """合并短名称和商品名，避免重复文本。"""
    values: List[str] = []
    for column in NAME_COLUMNS:
        value = fold_text(row.get(column))
        if value and value not in values:
            values.append(value)
    return " | ".join(values)


def prepare_dictionary(dictionary: pd.DataFrame) -> pd.DataFrame:
    """验证并清理公共 food_id 字典。"""
    validate_columns(
        dictionary.columns.tolist(),
        REQUIRED_DICTIONARY_COLUMNS,
        "公共 food_id 字典",
    )
    data = dictionary.loc[:, REQUIRED_DICTIONARY_COLUMNS].copy()
    data["food_id"] = normalize_text(data["food_id"])
    for column in NAME_COLUMNS:
        data[column] = normalize_text(data[column])
    data["canonical_food_category"] = normalize_category(
        data["canonical_food_category"]
    )

    if data["food_id"].isna().any():
        raise ValueError("公共 food_id 字典存在缺失 food_id。")
    if data["food_id"].duplicated().any():
        duplicates = sorted(
            data.loc[data["food_id"].duplicated(keep=False), "food_id"]
            .astype(str)
            .unique()
        )
        raise ValueError("公共字典存在重复 food_id：" + ", ".join(duplicates[:10]))

    for column in ["food_id_event_count", "food_id_weight_g_total"]:
        original_nonmissing = data[column].notna()
        numeric = pd.to_numeric(data[column], errors="coerce")
        invalid = original_nonmissing & numeric.isna()
        if invalid.any():
            raise ValueError(
                "{}有 {:,} 个无效数值。".format(column, int(invalid.sum()))
            )
        if numeric.lt(0).fillna(False).any():
            raise ValueError("{}不能包含负值。".format(column))
        data[column] = numeric.fillna(0)
    data["food_id_event_count"] = data["food_id_event_count"].astype("int64")
    return data


def apply_category_overrides(
    dictionary: pd.DataFrame, category_overrides: pd.DataFrame
) -> pd.DataFrame:
    """应用公共类别人工修订并生成最终类别。"""
    validate_columns(
        category_overrides.columns.tolist(),
        REQUIRED_CATEGORY_OVERRIDE_COLUMNS,
        "公共类别 override",
    )
    overrides = category_overrides.loc[:, REQUIRED_CATEGORY_OVERRIDE_COLUMNS].copy()
    overrides["food_id"] = normalize_text(overrides["food_id"])
    overrides["assigned_food_category"] = normalize_category(
        overrides["assigned_food_category"]
    )
    overrides = overrides.loc[overrides["food_id"].notna()].copy()
    if overrides["food_id"].duplicated().any():
        raise ValueError("公共类别 override 存在重复 food_id。")
    unknown_ids = sorted(set(overrides["food_id"]) - set(dictionary["food_id"]))
    if unknown_ids:
        raise ValueError(
            "公共类别 override 含字典外 food_id：" + ", ".join(unknown_ids[:10])
        )

    overrides = overrides.rename(
        columns={"assigned_food_category": "reviewed_food_category"}
    )
    data = dictionary.merge(overrides, on="food_id", how="left", validate="one_to_one")
    data["resolved_food_category"] = data["reviewed_food_category"].combine_first(
        data["canonical_food_category"]
    )
    data["category_source"] = "canonical_food_category"
    data.loc[
        data["reviewed_food_category"].notna(), "category_source"
    ] = "food_category_override"
    data["resolved_food_category_key"] = data["resolved_food_category"].map(
        category_key
    )
    data["combined_food_name"] = data.apply(combine_names, axis=1)
    return data


def dairy_suggestion(name: str) -> Optional[str]:
    """为需要人工确认的乳制品给出低脂或高脂候选。"""
    if LOW_FAT_DAIRY_PATTERN.search(name):
        return "low_fat_dairy"
    if HIGH_FAT_DAIRY_PATTERN.search(name):
        return "high_fat_dairy"
    if re.search(r"\b(?:yogurt|yoghurt|skim milk|ice milk|sherbet)\b", name):
        return "low_fat_dairy"
    if re.search(r"\b(?:cheese|cream|whole milk|ice cream)\b", name):
        return "high_fat_dairy"
    return None


def broad_suggestion(name: str, category: str) -> Optional[str]:
    """给宽泛类别或复合食物提供不自动确认的候选组件。"""
    if PROCESSED_MEAT_PATTERN.search(name):
        return "processed_meat"
    if POULTRY_PATTERN.search(name):
        return "poultry"
    if RED_MEAT_PATTERN.search(name):
        return "red_meat"
    if CREAM_SOUP_PATTERN.search(name):
        return "cream_soups"
    if FRENCH_FRIES_PATTERN.search(name):
        return "french_fries"
    if BUTTER_PATTERN.search(name):
        return "butter"
    if LOW_ENERGY_PATTERN.search(name) and BEVERAGE_PATTERN.search(name):
        return "low_energy_beverages"
    if HIGH_ENERGY_PATTERN.search(name) and not LOW_ENERGY_PATTERN.search(name):
        return "high_energy_drinks"
    if COFFEE_BROAD_PATTERN.search(name):
        return "coffee"
    if WINE_PATTERN.search(name):
        return "wine"
    dairy = dairy_suggestion(name)
    if dairy is not None:
        return dairy
    if FISH_BROAD_PATTERN.search(name):
        return "other_fish"
    if TOMATO_BROAD_PATTERN.search(name):
        return "tomatoes"
    if GREEN_LEAFY_BROAD_PATTERN.search(name):
        return "green_leafy_vegetables"
    if EGG_PATTERN.search(name):
        return "eggs"
    if FRUIT_BROAD_PATTERN.search(name):
        return "whole_fruits"
    if category in DAIRY_CATEGORIES:
        return dairy
    return None


def initial_decision(row: pd.Series) -> Tuple[str, Optional[str], Optional[str], str, str]:
    """对单个 food_id 生成保守的初始映射决定。"""
    name = row["combined_food_name"]
    category = row["resolved_food_category_key"]
    labels_missing = not name and not category
    if labels_missing:
        return (
            "unmapped_missing_labels",
            None,
            None,
            "all food labels missing",
            "missing_labels",
        )

    if MARGARINE_PATTERN.search(name):
        return (
            "not_applicable",
            None,
            "margarine",
            "margarine excluded because it was not measured in HPP",
            "paper_exclusion",
        )

    if category in DIRECT_CATEGORY_MAP:
        component = DIRECT_CATEGORY_MAP[category]
        return (
            "mapped",
            component,
            component,
            "high-confidence direct food category",
            "automatic_direct_category",
        )

    if category == "alcoholic drinks" and WINE_PATTERN.search(name):
        return (
            "mapped",
            "wine",
            "wine",
            "wine name within alcoholic drinks",
            "automatic_name_rule",
        )

    if category in BEVERAGE_CATEGORIES:
        if LOW_ENERGY_PATTERN.search(name) and BEVERAGE_PATTERN.search(name):
            return (
                "mapped",
                "low_energy_beverages",
                "low_energy_beverages",
                "explicit low-energy beverage name",
                "automatic_name_rule",
            )
        if COFFEE_AUTO_PATTERN.search(name):
            return (
                "mapped",
                "coffee",
                "coffee",
                "simple coffee beverage name",
                "automatic_name_rule",
            )
        if HIGH_ENERGY_PATTERN.search(name) and not LOW_ENERGY_PATTERN.search(name):
            return (
                "mapped",
                "high_energy_drinks",
                "high_energy_drinks",
                "explicit sugar-sweetened/high-energy drink name",
                "automatic_name_rule",
            )

    if category in DAIRY_CATEGORIES:
        suggestion = dairy_suggestion(name)
        return (
            "review_required",
            None,
            suggestion,
            "dairy fat level requires food-level review",
            "initial_review_rule",
        )

    if category in FISH_CATEGORIES:
        if OTHER_FISH_AUTO_PATTERN.search(name):
            return (
                "mapped",
                "other_fish",
                "other_fish",
                "simple other-fish or seafood name",
                "automatic_name_rule",
            )
        return (
            "review_required",
            None,
            "other_fish",
            "other-fish boundary or composite fish item requires review",
            "initial_review_rule",
        )

    if category in VEGETABLE_CATEGORIES:
        if TOMATO_AUTO_PATTERN.search(name):
            return (
                "mapped",
                "tomatoes",
                "tomatoes",
                "simple tomato item",
                "automatic_name_rule",
            )
        if GREEN_LEAFY_AUTO_PATTERN.search(name):
            return (
                "mapped",
                "green_leafy_vegetables",
                "green_leafy_vegetables",
                "simple green leafy vegetable item",
                "automatic_name_rule",
            )
        suggestion = broad_suggestion(name, category)
        if suggestion in {"tomatoes", "green_leafy_vegetables"}:
            return (
                "review_required",
                None,
                suggestion,
                "vegetable composite or boundary requires review",
                "initial_review_rule",
            )

    if category in COMPOSITE_REVIEW_CATEGORIES or category in {
        "oils and fats",
        "med oil and fats",
    }:
        if CREAM_SOUP_PATTERN.search(name):
            return (
                "mapped",
                "cream_soups",
                "cream_soups",
                "explicit cream soup or chowder name",
                "automatic_name_rule",
            )
        if FRENCH_FRIES_PATTERN.search(name):
            return (
                "mapped",
                "french_fries",
                "french_fries",
                "explicit French fries name",
                "automatic_name_rule",
            )
        if BUTTER_PATTERN.search(name):
            return (
                "mapped",
                "butter",
                "butter",
                "explicit butter name",
                "automatic_name_rule",
            )
        suggestion = broad_suggestion(name, category)
        if suggestion is not None:
            return (
                "review_required",
                None,
                suggestion,
                "broad or composite category requires food-level review",
                "initial_review_rule",
            )

    suggestion = broad_suggestion(name, category)
    if suggestion is not None:
        return (
            "review_required",
            None,
            suggestion,
            "name suggests rEDIH component but category is not high confidence",
            "initial_review_rule",
        )
    return (
        "not_applicable",
        None,
        None,
        "no rEDIH component rule",
        "automatic_not_applicable",
    )


def prepare_component_overrides(overrides: pd.DataFrame) -> pd.DataFrame:
    """清理并验证用户维护的 rEDIH override。"""
    validate_columns(
        overrides.columns.tolist(),
        REQUIRED_COMPONENT_OVERRIDE_COLUMNS,
        "rEDIH component override",
    )
    data = overrides.loc[:, REQUIRED_COMPONENT_OVERRIDE_COLUMNS].copy()
    for column in REQUIRED_COMPONENT_OVERRIDE_COLUMNS:
        data[column] = normalize_text(data[column])
    data = data.loc[data["food_id"].notna() & data["mapping_status"].notna()].copy()
    if data.empty:
        return data
    if data["food_id"].duplicated().any():
        duplicates = sorted(
            data.loc[data["food_id"].duplicated(keep=False), "food_id"]
            .astype(str)
            .unique()
        )
        raise ValueError("rEDIH override 存在重复 food_id：" + ", ".join(duplicates[:10]))
    invalid_status = ~data["mapping_status"].isin(VALID_MAPPING_STATUSES)
    if invalid_status.any():
        values = sorted(data.loc[invalid_status, "mapping_status"].astype(str).unique())
        raise ValueError("rEDIH override 含无效状态：" + ", ".join(values))

    mapped = data["mapping_status"].eq("mapped")
    invalid_component = mapped & ~data["redih_component"].isin(AVAILABLE_COMPONENTS)
    if invalid_component.any():
        values = sorted(
            data.loc[invalid_component, "redih_component"]
            .fillna("<missing>")
            .astype(str)
            .unique()
        )
        raise ValueError("rEDIH override 含无效 mapped 组件：" + ", ".join(values))
    if data.loc[~mapped, "redih_component"].notna().any():
        raise ValueError("非 mapped override 不应填写 redih_component。")
    return data


def build_mapping(
    dictionary: pd.DataFrame,
    category_overrides: pd.DataFrame,
    component_overrides: pd.DataFrame,
) -> pd.DataFrame:
    """生成并验证完整 rEDIH food_id 映射。"""
    prepared = prepare_dictionary(dictionary)
    mapping = apply_category_overrides(prepared, category_overrides)
    decisions = mapping.apply(initial_decision, axis=1, result_type="expand")
    decisions.columns = [
        "mapping_status",
        "redih_component",
        "suggested_redih_component",
        "mapping_rule",
        "mapping_source",
    ]
    mapping = pd.concat([mapping, decisions], axis=1)
    mapping["review_notes"] = pd.Series(pd.NA, index=mapping.index, dtype="string")

    overrides = prepare_component_overrides(component_overrides)
    unknown_ids = sorted(set(overrides["food_id"]) - set(mapping["food_id"]))
    if unknown_ids:
        raise ValueError("rEDIH override 含字典外 food_id：" + ", ".join(unknown_ids[:10]))
    if not overrides.empty:
        override_index = overrides.set_index("food_id")
        row_index = mapping["food_id"].map(
            dict(zip(mapping["food_id"], mapping.index))
        )
        for food_id, override in override_index.iterrows():
            index = int(row_index.loc[mapping["food_id"].eq(food_id)].iloc[0])
            mapping.at[index, "mapping_status"] = override["mapping_status"]
            mapping.at[index, "redih_component"] = override["redih_component"]
            mapping.at[index, "review_notes"] = override["review_notes"]
            mapping.at[index, "mapping_rule"] = "manual component override"
            mapping.at[index, "mapping_source"] = "manual_override"

    mapped = mapping["mapping_status"].eq("mapped")
    invalid_component = mapped & ~mapping["redih_component"].isin(AVAILABLE_COMPONENTS)
    if invalid_component.any():
        raise ValueError("最终 mapped 记录存在无效 rEDIH 组件。")
    if mapping.loc[~mapped, "redih_component"].notna().any():
        raise ValueError("最终非 mapped 记录不应包含 redih_component。")
    if ~mapping["mapping_status"].isin(VALID_MAPPING_STATUSES).all():
        raise ValueError("最终映射含无效 mapping_status。")
    if mapping["food_id"].duplicated().any():
        raise ValueError("最终 rEDIH 映射存在重复 food_id。")

    mapping["edih_weight_per_paper_table"] = mapping["redih_component"].map(
        COMPONENT_WEIGHTS
    )
    mapping["included_in_current_score"] = mapping["redih_component"].isin(
        CURRENT_SCORE_COMPONENTS
    )
    mapping.loc[~mapped, "included_in_current_score"] = False
    mapping["review_priority_event_count"] = mapping["food_id_event_count"]
    return mapping.sort_values(
        ["mapping_status", "review_priority_event_count", "food_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def mapping_summaries(
    mapping: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """生成状态、组件和长格式 QC 表。"""
    total_events = float(mapping["food_id_event_count"].sum())
    total_weight = float(mapping["food_id_weight_g_total"].sum())
    status = (
        mapping.groupby("mapping_status", dropna=False, sort=True)
        .agg(
            food_id_count=("food_id", "size"),
            event_count=("food_id_event_count", "sum"),
            weight_g_total=("food_id_weight_g_total", "sum"),
        )
        .reset_index()
    )
    status["food_id_share"] = status["food_id_count"] / len(mapping)
    status["event_share"] = status["event_count"] / total_events if total_events else 0
    status["weight_share"] = (
        status["weight_g_total"] / total_weight if total_weight else 0
    )

    component = (
        mapping.loc[mapping["mapping_status"].eq("mapped")]
        .groupby("redih_component", dropna=False, sort=True)
        .agg(
            food_id_count=("food_id", "size"),
            event_count=("food_id_event_count", "sum"),
            weight_g_total=("food_id_weight_g_total", "sum"),
        )
        .reset_index()
    )
    component["edih_weight_per_paper_table"] = component["redih_component"].map(
        COMPONENT_WEIGHTS
    )
    component["included_in_current_score"] = component["redih_component"].isin(
        CURRENT_SCORE_COMPONENTS
    )
    component = component.sort_values("event_count", ascending=False)

    metric_values: Dict[str, object] = {
        "script_version": SCRIPT_VERSION,
        "unique_food_ids": len(mapping),
        "dictionary_event_count": int(mapping["food_id_event_count"].sum()),
        "standard_component_count": len(COMPONENT_DEFINITIONS),
        "hpp_available_component_count": len(AVAILABLE_COMPONENTS),
        "current_scored_component_count": len(CURRENT_SCORE_COMPONENTS),
        "current_intake_unit": "g/day",
        "margarine_in_current_score": False,
        "wine_in_current_score": True,
        "mapped_food_id_count": int(mapping["mapping_status"].eq("mapped").sum()),
        "review_required_food_id_count": int(
            mapping["mapping_status"].eq("review_required").sum()
        ),
        "unmapped_missing_labels_food_id_count": int(
            mapping["mapping_status"].eq("unmapped_missing_labels").sum()
        ),
        "manual_override_food_id_count": int(
            mapping["mapping_source"].eq("manual_override").sum()
        ),
        "mapped_component_count_observed": int(
            mapping.loc[
                mapping["mapping_status"].eq("mapped"), "redih_component"
            ].nunique()
        ),
    }
    for row in status.itertuples(index=False):
        key = str(row.mapping_status)
        metric_values["{}_food_id_count".format(key)] = int(row.food_id_count)
        metric_values["{}_event_count".format(key)] = int(row.event_count)
        metric_values["{}_event_share".format(key)] = float(row.event_share)
    qc = pd.DataFrame(
        {
            "metric": list(metric_values.keys()),
            "value": list(metric_values.values()),
        }
    )
    return status, component, qc


def review_export(mapping: pd.DataFrame) -> pd.DataFrame:
    """提取按事件数降序排列的人工审核候选。"""
    columns = [
        "food_id",
        *NAME_COLUMNS,
        "canonical_food_category",
        "reviewed_food_category",
        "resolved_food_category",
        "food_id_event_count",
        "food_id_weight_g_total",
        "suggested_redih_component",
        "mapping_rule",
        "mapping_source",
        "review_notes",
    ]
    return (
        mapping.loc[mapping["mapping_status"].eq("review_required"), columns]
        .sort_values(["food_id_event_count", "food_id"], ascending=[False, True])
        .reset_index(drop=True)
    )


def override_template(review: pd.DataFrame) -> pd.DataFrame:
    """生成不会自动覆盖的人工 override 模板。"""
    template = review.copy()
    template["mapping_status"] = pd.Series(pd.NA, index=template.index, dtype="string")
    template["redih_component"] = pd.Series(pd.NA, index=template.index, dtype="string")
    template["review_notes"] = pd.Series(pd.NA, index=template.index, dtype="string")
    leading = REQUIRED_COMPONENT_OVERRIDE_COLUMNS
    context = [column for column in template.columns if column not in leading]
    return template.loc[:, [*leading, *context]]


def run(
    dictionary_csv: Path,
    category_overrides_csv: Path,
    component_overrides_csv: Path,
    output_dir: Path,
) -> Dict[str, Path]:
    """读取输入、生成 rEDIH 映射并保存全部审计输出。"""
    if not dictionary_csv.is_file():
        raise FileNotFoundError("找不到公共 food_id 字典：{}".format(dictionary_csv))
    if not category_overrides_csv.is_file():
        raise FileNotFoundError(
            "找不到公共类别 override：{}".format(category_overrides_csv)
        )

    dictionary = pd.read_csv(
        dictionary_csv,
        encoding="utf-8-sig",
        dtype={"food_id": "string"},
        low_memory=False,
    )
    category_overrides = pd.read_csv(
        category_overrides_csv,
        encoding="utf-8-sig",
        dtype={"food_id": "string"},
        low_memory=False,
    )
    if component_overrides_csv.is_file():
        component_overrides = pd.read_csv(
            component_overrides_csv,
            encoding="utf-8-sig",
            dtype={"food_id": "string"},
            low_memory=False,
        )
    else:
        component_overrides = pd.DataFrame(
            columns=REQUIRED_COMPONENT_OVERRIDE_COLUMNS
        )

    mapping = build_mapping(dictionary, category_overrides, component_overrides)
    status, component, qc = mapping_summaries(mapping)
    review = review_export(mapping)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "definitions": output_dir / "redih_component_definitions.csv",
        "mapping": output_dir / "redih_food_id_mapping.csv",
        "status": output_dir / "redih_mapping_status_summary.csv",
        "component": output_dir / "redih_component_summary.csv",
        "review": output_dir / "redih_review_candidates.csv",
        "qc": output_dir / "redih_mapping_qc.csv",
        "overrides": component_overrides_csv,
    }
    component_definitions().to_csv(
        paths["definitions"], index=False, encoding="utf-8-sig"
    )
    mapping.to_csv(paths["mapping"], index=False, encoding="utf-8-sig")
    status.to_csv(paths["status"], index=False, encoding="utf-8-sig")
    component.to_csv(paths["component"], index=False, encoding="utf-8-sig")
    review.to_csv(paths["review"], index=False, encoding="utf-8-sig")
    qc.to_csv(paths["qc"], index=False, encoding="utf-8-sig")
    if not component_overrides_csv.exists():
        override_template(review).to_csv(
            component_overrides_csv,
            index=False,
            encoding="utf-8-sig",
        )

    print("=" * 88)
    print("rEDIH food_id 映射")
    print("SCRIPT_VERSION={}".format(SCRIPT_VERSION))
    print("当前口径：g/day；margarine 因 HPP 未测量而排除；wine 进入17组件总分。")
    print("-" * 88)
    print(status.to_string(index=False))
    print("-" * 88)
    print("已映射组件：")
    if component.empty:
        print("<none>")
    else:
        print(component.to_string(index=False))
    print("-" * 88)
    print("待人工审核 food_id：{:,}".format(len(review)))
    if not review.empty:
        preview_columns = [
            "food_id",
            "canonical_short_food_name",
            "canonical_food_category",
            "food_id_event_count",
            "suggested_redih_component",
        ]
        print(review.loc[:, preview_columns].head(80).to_string(index=False))
    print("输出目录：{}".format(output_dir))
    print("人工 override：{}".format(component_overrides_csv))
    print("=" * 88)
    return paths


def parse_args() -> object:
    """解析命令行参数。"""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dictionary-csv",
        type=Path,
        default=DEFAULT_DICTIONARY_CSV,
    )
    parser.add_argument(
        "--category-overrides-csv",
        type=Path,
        default=DEFAULT_CATEGORY_OVERRIDES_CSV,
    )
    parser.add_argument(
        "--component-overrides-csv",
        type=Path,
        default=DEFAULT_COMPONENT_OVERRIDES_CSV,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""
    args = parse_args()
    run(
        args.dictionary_csv,
        args.category_overrides_csv,
        args.component_overrides_csv,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
