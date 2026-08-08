"""生成 hPDI 的 food_id 级候选映射和映射质量控制结果。"""

import re
import unicodedata
from argparse import ArgumentParser
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


SCRIPT_VERSION = "2026-08-07-hpdi-mapping-v12"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DEFAULT_DICTIONARY_CSV = (
    PROJECT_DIR / "outputs" / "02_food_dictionary" / "food_id_dictionary.csv"
)
DEFAULT_OVERRIDES_CSV = (
    PROJECT_DIR / "outputs" / "02_food_dictionary" / "food_category_overrides.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "03_score_mapping" / "hpdi"
DEFAULT_COMPONENT_OVERRIDES_CSV = DEFAULT_OUTPUT_DIR / "hpdi_component_overrides.csv"

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
REQUIRED_COMPONENT_OVERRIDE_COLUMNS = [
    "food_id",
    "mapping_status",
    "hpdi_component",
    "review_notes",
]
CATEGORY_MATCH_COLUMNS = (
    "canonical_food_category",
    "reviewed_food_category",
    "resolved_food_category",
)
PROCESSED_MEAT_ALIASES = (
    "processedmeatproduct",
    "processedmeatproducts",
)

HEALTHY_PLANT_COMPONENTS = {
    "whole_grains",
    "fruits",
    "vegetables",
    "nuts",
    "legumes",
    "vegetable_oils",
    "tea_coffee",
}
LESS_HEALTHY_PLANT_COMPONENTS = {
    "fruit_juice",
    "refined_grains",
    "potatoes",
    "sugar_sweetened_beverages",
    "sweets_desserts",
}
ANIMAL_COMPONENTS = {
    "animal_fat",
    "dairy",
    "eggs",
    "fish_seafood",
    "meat",
    "miscellaneous_animal_foods",
}

DIRECT_CATEGORY_MAP: Dict[str, str] = {
    "fruits": "fruits",
    "vegetables": "vegetables",
    "bread wholewheat": "whole_grains",
    "pasta, grains and side dishes wholewheat": "whole_grains",
    "nuts, seeds, and products": "nuts",
    "pulses and products": "legumes",
    "med oil and fats": "vegetable_oils",
    "sweets": "sweets_desserts",
    "baked goods": "sweets_desserts",
    "milk, cream cheese and yogurts": "dairy",
    "hard cheese": "dairy",
    "sweet milk products": "dairy",
    "eggs and their products": "eggs",
    "fish and seafood": "fish_seafood",
    "poultry and its products": "meat",
    "beef, veal, lamb, and other meat products": "meat",
    "processed meat products": "meat",
}

EXPECTED_COMPONENTS = (
    HEALTHY_PLANT_COMPONENTS | LESS_HEALTHY_PLANT_COMPONENTS | ANIMAL_COMPONENTS
)

GRAIN_CATEGORIES = {
    "bread",
    "pasta, grains and side dishes",
    "cereals",
}
NAME_RESOLUTION_CATEGORIES = {
    "drinks",
    "fruit juices and soft drinks",
    "canned veg and fruits",
    "oils and fats",
}
COMPOSITE_REVIEW_CATEGORIES = {
    "soups and sauces",
    "others",
    "snacks",
    "fast foods",
    "industrialized vegetarian food ready to eat",
    "deep fried foods",
}
NOT_APPLICABLE_CATEGORIES = {
    "alcoholic drinks",
    "spices and herbs",
    "low calories and diet drinks",
}

WHOLE_GRAIN_KEYWORDS = (
    "whole wheat",
    "wholewheat",
    "whole grain",
    "wholegrain",
    "brown rice",
    "wild rice",
    "oat",
    "bran",
    "rye",
    "bulgur",
    "quinoa",
    "barley",
    "popcorn",
)
POTATO_OR_CORN_CHIP_KEYWORDS = (
    "potato chip",
    "corn chip",
    "tortilla chip",
    "doritos",
    "cheetos",
    "apropo",
)
POTATO_KEYWORDS = (
    "potato",
    "french fries",
    "french fry",
) + POTATO_OR_CORN_CHIP_KEYWORDS
TEA_COFFEE_KEYWORDS = (
    "coffee",
    "espresso",
    "cappuccino",
    "americano",
    "latte",
    "tea",
)
FRUIT_JUICE_KEYWORDS = (
    "fruit juice",
    "apple juice",
    "orange juice",
    "grapefruit juice",
    "pomegranate juice",
    "grape juice",
    "pineapple juice",
    "prune juice",
)
SUGAR_SWEETENED_BEVERAGE_KEYWORDS = (
    "soft drink",
    "cola",
    "soda",
    "energy drink",
    "sweetened drink",
    "sweetened beverage",
    "lemonade",
    "fruit drink",
)
VEGETABLE_OIL_KEYWORDS = (
    "olive oil",
    "canola oil",
    "rapeseed oil",
    "sunflower oil",
    "soybean oil",
    "sesame oil",
    "corn oil",
    "vegetable oil",
    "avocado oil",
    "coconut oil",
    "tahini",
)
ANIMAL_FAT_KEYWORDS = (
    "butter",
    "lard",
    "tallow",
    "animal fat",
    "beef fat",
    "goose fat",
    "duck fat",
)
# Satija hPDI 将 pizza、chowder/cream soup、mayonnaise 或其他 creamy
# salad dressing 归为 miscellaneous animal-based foods。含 mayonnaise 的
# tuna/egg/potato salad 等复合菜不能整份归入该组，应保留其主要食物组。
STANDALONE_MAYO_DRESSING_PATTERN = (
    r"^(?:mayonnaise|mayonaise|mayo)(?:\b|[-_ ])"
    r"|^(?:creamy|ranch|caesar) (?:salad )?dressing(?:\b|[-_ ])"
    r"|^thousand island (?:salad )?dressing(?:\b|[-_ ])"
)
# 仅用于解决原规则保留下来的 review_required。规则锚定在标准短名称
# 开头，并且不覆盖已由可靠食物类别映射的记录。
EXPLICIT_REVIEW_COMPONENT_RULES: List[Tuple[str, str]] = [
    (
        "sweets_desserts",
        r"^(?:brown sugar|white sugar|granulated sugar|table sugar|"
        r"jam|jelly|preserves|honey|syrup|(?:fruit|raspberry) syrup|"
        r"jewish donut)\b",
    ),
    ("fruit_juice", r"^(?:(?:lemon|lime) juice|smoothies?)\b"),
    ("dairy", r"^(?:whipped cream|ice cream|salep)\b"),
    (
        "legumes",
        r"^(?:falafel|hummus|roasted soybeans|cooked canned green peas)\b",
    ),
    (
        "tea_coffee",
        r"^(?:coffee|espresso|cappuccino|americano|latte|tea|"
        r"green tea|black tea|herbal tea|chamomile tea|mint tea|chai tea)\b",
    ),
    (
        "vegetable_oils",
        r"^(?:pesto|vinaigrette|olive oil|chimichurri)\b",
    ),
    (
        "fruits",
        r"^(?:prune|dried cranberries|dried blueberries|cherries|lemon|"
        r"cooked raisins)\b",
    ),
    (
        "vegetables",
        r"^(?:olives|ketchup|tomato paste|tomato sauce|sauerkraut|"
        r"coleslaw|sweet potato fries|corn schnitzel|olive spread|"
        r"celery juice|carrot juice|beet juice|wheatgrass juice|"
        r"harissa salad|dried tomato spread|onion rings|matbucha|salsa)\b",
    ),
    ("nuts", r"^(?:coated peanuts|peanuts|sesame snack)\b"),
    ("dairy", r"^cooking cream\b"),
    ("meat", r"^hamburger\b"),
    (
        "potatoes",
        r"^(?:french fries|potato chips|mashed potatoes|apropo|doritos)\b",
    ),
    (
        "refined_grains",
        r"^(?:pretzels|bissli|crackers?|croutons|soup mandels)\b",
    ),
    ("whole_grains", r"^popcorn\b"),
]
EXPLICIT_REVIEW_NOT_APPLICABLE_PATTERN = (
    r"^(?:salt|sugar substitute|stevia(?: sweetener)?|soy sauce|mustard|"
    r"balsamic vinegar|diet coke|pepsi max|sucra light|cocoa powder|"
    r"schug|sechug|teriyaki sauce|protein powder|chili sauce|"
    r"sugar free gum|apple vinegar|apple cider vinegar|malt beverage|amba|"
    r"collagen powder|veggie burger|vegan hamburger|coconut water|"
    r"sweet(?: and| &) light sugar substitute|sweet n low|saccharin|"
    r"maca powder|spirulina|barbecue sauce|moringa powder|garlic sauce|"
    r"yeast|mct oil|curry paste|vinegar|energy bar|psyllium husks|tabasco|"
    r"flavou?red waters?|vitamin d3|veggie sausage|vegan protein powder|"
    r"seitan|seiten|veggie schnitzel|creamy sauce|vegetarian mincemeat|"
    r"sweet potato cream sauce|shawarma seitan|creatine|"
    r"tbevol vegetarian shawarma|fenugreek seeds|"
    r"thin schnitzel from (?:the )?(?:home )?plant|aloe vera|myocal|"
    r"vegetable sausages|"
    r"sodium-reduced soy sauce|plant-based hamburger|"
    r"reduced-fat plant-based schnitzel|asam powder chicken soup|"
    r"sehug adom)\b"
)

# 只在待审核记录中使用商品名处理短名称不足以表达主要成分的少数条目。
# 这些规则不作为通用商品名关键词，以免长商品描述造成误匹配。
EXPLICIT_REVIEW_PRODUCT_COMPONENT_RULES: List[Tuple[str, str]] = [
    (
        "vegetables",
        r"\b(?:palm hearts?|black olives|eggplant broccoli schnitzel|kimchi|"
        r"pickled ginger)\b",
    ),
    ("sweets_desserts", r"\bcandied ginger\b"),
    (
        "sugar_sweetened_beverages",
        r"\b(?:tonic water|cranberry juice cocktail|lemon nectar)\b",
    ),
    ("refined_grains", r"\bpretzels?\b"),
    (
        "potatoes",
        r"\b(?:potato chips?|corn chips?|tortilla chips?|cheetos|doritos|apropo)\b",
    ),
    ("vegetable_oils", r"\bvinaigrette\b"),
    ("dairy", r"\breduced cream\b"),
    ("legumes", r"\bpea wasabi\b"),
    ("meat", r"\bmcdoland nuggets\b"),
    (
        "miscellaneous_animal_foods",
        r"\breal organic mayonnaise\b",
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
    "sweet potato",
    "yam",
)

SUGGESTION_KEYWORDS: List[Tuple[str, Tuple[str, ...]]] = [
    ("whole_grains", WHOLE_GRAIN_KEYWORDS),
    ("potatoes", POTATO_KEYWORDS),
    ("tea_coffee", TEA_COFFEE_KEYWORDS),
    ("fruit_juice", FRUIT_JUICE_KEYWORDS),
    (
        "sugar_sweetened_beverages",
        SUGAR_SWEETENED_BEVERAGE_KEYWORDS,
    ),
    ("vegetable_oils", VEGETABLE_OIL_KEYWORDS),
    ("animal_fat", ANIMAL_FAT_KEYWORDS),
    (
        "dairy",
        (
            "milk",
            "yogurt",
            "yoghurt",
            "cheese",
            "cream",
        ),
    ),
    ("eggs", ("egg", "omelette", "omelet")),
    (
        "fish_seafood",
        (
            "fish",
            "salmon",
            "tuna",
            "sardine",
            "shrimp",
            "prawn",
            "seafood",
        ),
    ),
    (
        "meat",
        (
            "beef",
            "veal",
            "lamb",
            "pork",
            "chicken",
            "turkey",
            "bacon",
            "sausage",
            "hamburger",
            "pastrami",
            "salami",
        ),
    ),
    (
        "sweets_desserts",
        (
            "cake",
            "cookie",
            "biscuit",
            "chocolate",
            "candy",
            "ice cream",
            "dessert",
            "waffle",
            "pastry",
        ),
    ),
    ("fruits", FRUIT_KEYWORDS),
    ("vegetables", VEGETABLE_KEYWORDS),
]


def component_definitions() -> pd.DataFrame:
    """返回标准 hPDI 的18个食物组及五分位计分方向。"""
    rows = [
        ("whole_grains", "healthy_plant", "Whole grains", "positive"),
        ("fruits", "healthy_plant", "Fruits", "positive"),
        ("vegetables", "healthy_plant", "Vegetables", "positive"),
        ("nuts", "healthy_plant", "Nuts", "positive"),
        ("legumes", "healthy_plant", "Legumes", "positive"),
        ("vegetable_oils", "healthy_plant", "Vegetable oils", "positive"),
        ("tea_coffee", "healthy_plant", "Tea and coffee", "positive"),
        ("fruit_juice", "less_healthy_plant", "Fruit juice", "reverse"),
        (
            "refined_grains",
            "less_healthy_plant",
            "Refined grains",
            "reverse",
        ),
        ("potatoes", "less_healthy_plant", "Potatoes", "reverse"),
        (
            "sugar_sweetened_beverages",
            "less_healthy_plant",
            "Sugar-sweetened beverages",
            "reverse",
        ),
        (
            "sweets_desserts",
            "less_healthy_plant",
            "Sweets and desserts",
            "reverse",
        ),
        ("animal_fat", "animal", "Animal fat", "reverse"),
        ("dairy", "animal", "Dairy", "reverse"),
        ("eggs", "animal", "Eggs", "reverse"),
        ("fish_seafood", "animal", "Fish and seafood", "reverse"),
        ("meat", "animal", "Meat", "reverse"),
        (
            "miscellaneous_animal_foods",
            "animal",
            "Miscellaneous animal-based foods",
            "reverse",
        ),
    ]
    definitions = pd.DataFrame(
        rows,
        columns=[
            "hpdi_component",
            "hpdi_parent_group",
            "standard_food_group",
            "scoring_direction",
        ],
    )
    definitions["lowest_quintile_score"] = definitions["scoring_direction"].map(
        {"positive": 1, "reverse": 5}
    )
    definitions["highest_quintile_score"] = definitions["scoring_direction"].map(
        {"positive": 5, "reverse": 1}
    )
    definitions["planned_intake_basis"] = "mean daily grams proxy"
    return definitions


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
    """生成不受大小写、下划线和连续空格影响的匹配键。"""
    return (
        normalize_category(series)
        .str.replace("_", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.casefold()
    )


def compact_category_key(series: pd.Series) -> pd.Series:
    """生成忽略空格、标点和隐藏字符的类别匹配键。"""
    return (
        category_key(series)
        .str.normalize("NFKC")
        .str.replace(r"[^a-z0-9]+", "", regex=True)
    )


def category_matches_any_source(
    mapping: pd.DataFrame,
    compact_values: Tuple[str, ...],
) -> pd.Series:
    """在原始、人工审核和最终类别字段中匹配标准化类别值。"""
    mask = pd.Series(False, index=mapping.index, dtype=bool)
    for column in CATEGORY_MATCH_COLUMNS:
        if column in mapping.columns:
            mask |= compact_category_key(mapping[column]).isin(compact_values)
    return mask


def robust_compact_category_value(value: object) -> str:
    """生成可容忍隐藏字符和少量非 ASCII 混入的单个类别键。"""
    if pd.isna(value):
        return ""
    normalized = unicodedata.normalize("NFKD", str(value)).casefold()
    visible = "".join(
        character
        for character in normalized
        if not unicodedata.category(character).startswith("C")
    )
    return "".join(
        character
        for character in visible
        if character.isascii() and character.isalnum()
    )


def compact_alias_matches(
    value: object,
    aliases: Tuple[str, ...],
) -> bool:
    """严格匹配别名，或接受最多约两个字符的类别录入差异。"""
    compact = robust_compact_category_value(value)
    if compact in aliases:
        return True
    if not compact.startswith("pro") or "meat" not in compact:
        return False
    return any(
        abs(len(compact) - len(alias)) <= 2
        and SequenceMatcher(None, compact, alias).ratio() >= 0.94
        for alias in aliases
    )


def category_matches_robust_alias_any_source(
    mapping: pd.DataFrame,
    aliases: Tuple[str, ...],
) -> pd.Series:
    """在全部类别来源中执行隐藏字符安全的别名匹配。"""
    mask = pd.Series(False, index=mapping.index, dtype=bool)
    for column in CATEGORY_MATCH_COLUMNS:
        if column in mapping.columns:
            matches = mapping[column].map(
                lambda value: compact_alias_matches(value, aliases)
            )
            mask |= matches.fillna(False).astype(bool)
    return mask


def contains_any(series: pd.Series, keywords: Tuple[str, ...]) -> pd.Series:
    """判断每条标准化名称是否包含任一关键词。"""
    mask = pd.Series(False, index=series.index)
    for keyword in keywords:
        mask |= series.str.contains(keyword, regex=False, na=False)
    return mask


def component_parent_group(component: Optional[str]) -> Optional[str]:
    """返回 hPDI 组件所属的三大食物类型。"""
    if component in HEALTHY_PLANT_COMPONENTS:
        return "healthy_plant"
    if component in LESS_HEALTHY_PLANT_COMPONENTS:
        return "less_healthy_plant"
    if component in ANIMAL_COMPONENTS:
        return "animal"
    return None


def suggest_component(combined_name: str) -> Optional[str]:
    """为混合类别生成名称关键词候选，但不自动确认映射。"""
    if not combined_name:
        return None
    for component, keywords in SUGGESTION_KEYWORDS:
        if component == "potatoes" and re.search(
            r"\b(?:sweet potato(?:es)?|yams?)\b",
            combined_name,
        ):
            continue
        if component == "eggs":
            egg_pattern = r"\beggs?\b|\bomelettes?\b|\bomelets?\b"
            if re.search(egg_pattern, combined_name):
                return component
            continue
        if any(keyword in combined_name for keyword in keywords):
            return component
    return None


def assign_component(
    mapping: pd.DataFrame,
    mask: pd.Series,
    component: str,
    basis: str,
) -> None:
    """将满足保守规则的 food_id 确认为 hPDI 组件。"""
    mapping.loc[mask, "hpdi_component"] = component
    mapping.loc[mask, "mapping_status"] = "mapped"
    mapping.loc[mask, "mapping_basis"] = basis


def mark_not_applicable(
    mapping: pd.DataFrame,
    mask: pd.Series,
    basis: str,
) -> None:
    """标记明确不进入 hPDI 18个食物组的 food_id。"""
    mapping.loc[mask, "hpdi_component"] = pd.NA
    mapping.loc[mask, "mapping_status"] = "not_applicable"
    mapping.loc[mask, "mapping_basis"] = basis


def resolve_categories(
    dictionary: pd.DataFrame,
    overrides: pd.DataFrame,
) -> pd.DataFrame:
    """使用人工覆盖表补齐已审核的食物类别。"""
    dictionary = dictionary.copy()
    overrides = overrides.loc[:, REQUIRED_OVERRIDE_COLUMNS].copy()
    overrides["food_id"] = normalize_text(overrides["food_id"])
    if overrides["food_id"].duplicated().any():
        raise ValueError("类别覆盖表存在重复 food_id。")

    override_map = overrides.set_index("food_id")["assigned_food_category"]
    dictionary["reviewed_food_category"] = dictionary["food_id"].map(override_map)
    dictionary["resolved_food_category"] = dictionary["reviewed_food_category"].fillna(
        dictionary["canonical_food_category"]
    )
    dictionary["category_source"] = "original"
    dictionary.loc[dictionary["reviewed_food_category"].notna(), "category_source"] = (
        "manual_override"
    )
    dictionary.loc[dictionary["resolved_food_category"].isna(), "category_source"] = (
        "missing"
    )
    return dictionary


def apply_component_overrides(
    mapping: pd.DataFrame,
    overrides: pd.DataFrame,
) -> pd.DataFrame:
    """应用经人工审核的 food_id 级 hPDI 组件决定。"""
    if overrides.empty:
        return mapping

    overrides = overrides.loc[:, REQUIRED_COMPONENT_OVERRIDE_COLUMNS].copy()
    for column in REQUIRED_COMPONENT_OVERRIDE_COLUMNS:
        overrides[column] = normalize_text(overrides[column])

    missing_food_ids = int(overrides["food_id"].isna().sum())
    if missing_food_ids:
        raise ValueError(f"hPDI 组件覆盖表有 {missing_food_ids:,} 行缺少 food_id。")
    if overrides["food_id"].duplicated().any():
        raise ValueError("hPDI 组件覆盖表存在重复 food_id。")

    valid_statuses = {"mapped", "not_applicable"}
    invalid_status = ~overrides["mapping_status"].isin(valid_statuses)
    if invalid_status.any():
        values = sorted(
            overrides.loc[invalid_status, "mapping_status"]
            .fillna("<missing>")
            .astype(str)
            .unique()
        )
        raise ValueError(
            "hPDI 组件覆盖表 mapping_status 只能是 mapped 或 "
            "not_applicable：" + ", ".join(values)
        )

    mapped = overrides["mapping_status"].eq("mapped")
    invalid_component = mapped & ~overrides["hpdi_component"].isin(EXPECTED_COMPONENTS)
    if invalid_component.any():
        values = sorted(
            overrides.loc[invalid_component, "hpdi_component"]
            .fillna("<missing>")
            .astype(str)
            .unique()
        )
        raise ValueError("人工 mapped 记录含无效 hPDI 组件：" + ", ".join(values))
    if overrides.loc[~mapped, "hpdi_component"].notna().any():
        raise ValueError("人工 not_applicable 记录的 hpdi_component 必须留空。")

    unknown_ids = sorted(
        set(overrides["food_id"].dropna()) - set(mapping["food_id"].dropna())
    )
    if unknown_ids:
        raise ValueError(
            "hPDI 组件覆盖表含食物字典中不存在的 food_id："
            + ", ".join(unknown_ids[:10])
        )

    decisions = overrides.set_index("food_id")
    row_ids = mapping["food_id"]
    has_override = row_ids.isin(decisions.index)
    mapping = mapping.copy()
    mapping.loc[has_override, "mapping_status"] = row_ids.loc[has_override].map(
        decisions["mapping_status"]
    )
    mapping.loc[has_override, "hpdi_component"] = row_ids.loc[has_override].map(
        decisions["hpdi_component"]
    )
    mapping.loc[has_override, "mapping_basis"] = "manual_hpdi_component_override"
    mapping.loc[has_override, "review_notes"] = row_ids.loc[has_override].map(
        decisions["review_notes"]
    )
    return mapping


def build_hpdi_mapping(
    dictionary: pd.DataFrame,
    component_overrides: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """按照类别和保守名称规则生成 hPDI food_id 映射。"""
    mapping = dictionary.copy()
    short_names = (
        mapping["canonical_short_food_name"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )
    product_names = (
        mapping["canonical_product_name"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )
    # 具体短名称为空时才回退到商品名；短名称存在时不混入更长、更嘈杂的
    # 商品描述，避免扩大显式审核规则的误匹配范围。
    review_names = short_names.mask(short_names.eq(""), product_names)
    combined_names = (
        mapping[NAME_COLUMNS].fillna("").astype(str).agg(" ".join, axis=1)
    ).str.casefold()
    mapping["resolved_food_category_key"] = category_key(
        mapping["resolved_food_category"]
    )
    category_keys = mapping["resolved_food_category_key"]

    mapping["hpdi_component"] = category_keys.map(DIRECT_CATEGORY_MAP)
    mapping["suggested_hpdi_component"] = pd.NA
    mapping["mapping_status"] = "review_required"
    mapping["mapping_basis"] = "category_not_in_direct_rules"
    mapping["review_notes"] = pd.NA

    direct = mapping["hpdi_component"].notna()
    mapping.loc[direct, "mapping_status"] = "mapped"
    mapping.loc[direct, "mapping_basis"] = "direct_food_category"

    processed_meat_category = category_matches_any_source(
        mapping, PROCESSED_MEAT_ALIASES
    ) | category_matches_robust_alias_any_source(
        mapping,
        PROCESSED_MEAT_ALIASES,
    )
    assign_component(
        mapping,
        processed_meat_category,
        "meat",
        "processed_meat_food_category",
    )
    processed_meat_source_rows = processed_meat_category
    unresolved_processed_meat = processed_meat_source_rows & ~(
        mapping["mapping_status"].eq("mapped") & mapping["hpdi_component"].eq("meat")
    )
    if unresolved_processed_meat.any():
        raise AssertionError("加工肉类别未全部映射至 hPDI meat 组件。")

    missing_category = mapping["resolved_food_category"].isna()
    mapping.loc[missing_category, "mapping_status"] = "unmapped_missing_labels"
    mapping.loc[missing_category, "mapping_basis"] = "no_resolved_category"

    not_applicable = category_keys.isin(NOT_APPLICABLE_CATEGORIES)
    mark_not_applicable(
        mapping,
        not_applicable,
        "category_outside_hpdi_food_groups",
    )

    vegetable_category = category_keys.eq("vegetables")
    sweet_potatoes = combined_names.str.contains(
        r"\b(?:sweet potato(?:es)?|yams?)\b",
        regex=True,
        na=False,
    )
    potatoes = contains_any(combined_names, POTATO_KEYWORDS) & ~sweet_potatoes
    assign_component(
        mapping,
        vegetable_category & potatoes,
        "potatoes",
        "potato_name_within_vegetable_category",
    )

    grain_categories = category_keys.isin(GRAIN_CATEGORIES)
    whole_grains = contains_any(combined_names, WHOLE_GRAIN_KEYWORDS)
    assign_component(
        mapping,
        grain_categories & whole_grains,
        "whole_grains",
        "explicit_whole_grain_name_in_grain_category",
    )
    assign_component(
        mapping,
        grain_categories & ~whole_grains,
        "refined_grains",
        "grain_category_without_whole_grain_evidence",
    )

    beverage_category = category_keys.eq("drinks")
    tea_coffee = contains_any(combined_names, TEA_COFFEE_KEYWORDS)
    sugary_beverage = contains_any(
        combined_names,
        SUGAR_SWEETENED_BEVERAGE_KEYWORDS,
    )
    assign_component(
        mapping,
        beverage_category & tea_coffee & ~sugary_beverage,
        "tea_coffee",
        "tea_or_coffee_name_in_drinks_category",
    )
    assign_component(
        mapping,
        beverage_category & sugary_beverage,
        "sugar_sweetened_beverages",
        "sugary_beverage_name_in_drinks_category",
    )
    unresolved_drinks = beverage_category & mapping["mapping_status"].eq(
        "review_required"
    )
    mark_not_applicable(
        mapping,
        unresolved_drinks,
        "drink_without_tea_coffee_or_sugary_beverage_evidence",
    )

    juice_soft_drink_category = category_keys.eq("fruit juices and soft drinks")
    fruit_juice = contains_any(combined_names, FRUIT_JUICE_KEYWORDS)
    soft_drink = contains_any(
        combined_names,
        SUGAR_SWEETENED_BEVERAGE_KEYWORDS,
    )
    assign_component(
        mapping,
        juice_soft_drink_category & fruit_juice & ~soft_drink,
        "fruit_juice",
        "explicit_fruit_juice_name",
    )
    assign_component(
        mapping,
        juice_soft_drink_category & soft_drink,
        "sugar_sweetened_beverages",
        "explicit_soft_or_sweetened_drink_name",
    )

    canned_category = category_keys.eq("canned veg and fruits")
    canned_potatoes = canned_category & potatoes
    canned_fruits = (
        canned_category
        & contains_any(combined_names, FRUIT_KEYWORDS)
        & ~canned_potatoes
    )
    canned_vegetables = (
        canned_category
        & contains_any(combined_names, VEGETABLE_KEYWORDS)
        & ~canned_fruits
        & ~canned_potatoes
    )
    assign_component(
        mapping,
        canned_potatoes,
        "potatoes",
        "potato_name_in_canned_category",
    )
    assign_component(
        mapping,
        canned_fruits,
        "fruits",
        "fruit_name_in_canned_category",
    )
    assign_component(
        mapping,
        canned_vegetables,
        "vegetables",
        "vegetable_name_in_canned_category",
    )

    fats_category = category_keys.eq("oils and fats")
    vegetable_oils = contains_any(combined_names, VEGETABLE_OIL_KEYWORDS)
    animal_fats = contains_any(combined_names, ANIMAL_FAT_KEYWORDS)
    assign_component(
        mapping,
        fats_category & vegetable_oils & ~animal_fats,
        "vegetable_oils",
        "vegetable_oil_name_in_oils_and_fats_category",
    )
    assign_component(
        mapping,
        fats_category & animal_fats,
        "animal_fat",
        "animal_fat_name_in_oils_and_fats_category",
    )

    # 部分商品的标准短名称错误地写成 Coconut water，但商品名明确为
    # coconut cream。仅在 Oils and fats 类别中使用商品名纠正，避免将真正的
    # 椰子水归入植物油。
    coconut_cream = product_names.str.contains(
        r"\bcoconut cream\b",
        regex=True,
        na=False,
    )
    assign_component(
        mapping,
        fats_category & coconut_cream,
        "vegetable_oils",
        "explicit_coconut_cream_in_oils_and_fats_category",
    )

    # Supplementary Table 11 explicitly places potato or corn chips in the
    # potatoes component.  Apply this after broad source-category rules so a
    # corn/tortilla chip cannot remain refined_grains merely because the source
    # database labelled it Bread, Cereals, Snacks, or another mixed category.
    paper_potato_or_corn_chips = contains_any(
        combined_names,
        POTATO_OR_CORN_CHIP_KEYWORDS,
    )
    eligible_paper_chip_rows = (
        ~missing_category & ~not_applicable & paper_potato_or_corn_chips
    )
    assign_component(
        mapping,
        eligible_paper_chip_rows,
        "potatoes",
        "supplementary_table_11_potato_or_corn_chips",
    )
    unresolved_paper_chip_rows = eligible_paper_chip_rows & ~(
        mapping["mapping_status"].eq("mapped")
        & mapping["hpdi_component"].eq("potatoes")
    )
    if unresolved_paper_chip_rows.any():
        raise AssertionError(
            "Supplementary Table 11 的 potato/corn chips 未全部映射至 potatoes。"
        )

    for component, pattern in EXPLICIT_REVIEW_COMPONENT_RULES:
        unresolved = mapping["mapping_status"].eq("review_required")
        explicit_name = review_names.str.contains(
            pattern,
            regex=True,
            na=False,
        )
        assign_component(
            mapping,
            unresolved & explicit_name,
            component,
            "explicit_review_name_" + component,
        )

    for component, pattern in EXPLICIT_REVIEW_PRODUCT_COMPONENT_RULES:
        unresolved = mapping["mapping_status"].eq("review_required")
        explicit_product = product_names.str.contains(
            pattern,
            regex=True,
            na=False,
        )
        assign_component(
            mapping,
            unresolved & explicit_product,
            component,
            "explicit_review_product_name_" + component,
        )

    unresolved = mapping["mapping_status"].eq("review_required")
    explicit_not_applicable = review_names.str.contains(
        EXPLICIT_REVIEW_NOT_APPLICABLE_PATTERN,
        regex=True,
        na=False,
    )
    mark_not_applicable(
        mapping,
        unresolved & explicit_not_applicable,
        "explicit_review_name_outside_hpdi_groups",
    )

    pizza = combined_names.str.contains(
        r"\bpizza\b", regex=True, na=False
    ) & ~category_keys.isin(NOT_APPLICABLE_CATEGORIES)
    assign_component(
        mapping,
        pizza,
        "miscellaneous_animal_foods",
        "standard_miscellaneous_animal_pizza",
    )

    soup_category = category_keys.eq("soups and sauces")
    chowder = combined_names.str.contains(r"\bchowder\b", regex=True, na=False)
    cream_soup = combined_names.str.contains(
        r"\bcream(?:y)?\b", regex=True, na=False
    ) & combined_names.str.contains(r"\bsoup\b", regex=True, na=False)
    assign_component(
        mapping,
        soup_category & (chowder | cream_soup),
        "miscellaneous_animal_foods",
        "standard_miscellaneous_animal_cream_soup",
    )

    standalone_mayo_dressing = short_names.str.contains(
        STANDALONE_MAYO_DRESSING_PATTERN,
        regex=True,
        na=False,
    )
    missing_short_name = short_names.eq("")
    standalone_mayo_dressing |= missing_short_name & product_names.str.contains(
        STANDALONE_MAYO_DRESSING_PATTERN,
        regex=True,
        na=False,
    )
    assign_component(
        mapping,
        standalone_mayo_dressing,
        "miscellaneous_animal_foods",
        "standard_miscellaneous_animal_mayo_or_dressing",
    )

    if component_overrides is not None:
        mapping = apply_component_overrides(mapping, component_overrides)

    ambiguous_categories = category_keys.isin(
        NAME_RESOLUTION_CATEGORIES | COMPOSITE_REVIEW_CATEGORIES
    )
    unresolved_ambiguous = ambiguous_categories & mapping["mapping_status"].eq(
        "review_required"
    )
    mapping.loc[unresolved_ambiguous, "suggested_hpdi_component"] = combined_names.loc[
        unresolved_ambiguous
    ].map(suggest_component)
    mapping.loc[unresolved_ambiguous, "mapping_basis"] = (
        "mixed_or_ambiguous_food_category"
    )

    mapping["hpdi_parent_group"] = mapping["hpdi_component"].map(component_parent_group)
    mapping["suggested_hpdi_parent_group"] = mapping["suggested_hpdi_component"].map(
        component_parent_group
    )
    mapping["hpdi_parent_group"] = mapping["hpdi_component"].map(component_parent_group)

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
        "hpdi_component",
        "hpdi_parent_group",
        "suggested_hpdi_component",
        "suggested_hpdi_parent_group",
        "mapping_status",
        "mapping_basis",
        "review_notes",
    ]
    return (
        mapping.loc[:, output_columns]
        .sort_values(
            ["mapping_status", "resolved_food_category", "food_id_event_count"],
            ascending=[True, True, False],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def build_qc(mapping: pd.DataFrame) -> pd.DataFrame:
    """汇总映射状态、三大类型、18个组件及待审核类别。"""
    status = (
        mapping.groupby("mapping_status", dropna=False)
        .agg(
            food_id_count=("food_id", "nunique"),
            event_count=("food_id_event_count", "sum"),
        )
        .reset_index()
    )
    status.insert(0, "summary_type", "mapping_status")
    status["hpdi_parent_group"] = ""
    status["hpdi_component"] = ""
    status["food_category"] = ""

    parent_groups = (
        mapping[mapping["mapping_status"].eq("mapped")]
        .groupby("hpdi_parent_group", dropna=False)
        .agg(
            food_id_count=("food_id", "nunique"),
            event_count=("food_id_event_count", "sum"),
        )
        .reset_index()
    )
    parent_groups.insert(0, "summary_type", "mapped_parent_group")
    parent_groups["mapping_status"] = "mapped"
    parent_groups["hpdi_component"] = ""
    parent_groups["food_category"] = ""

    components = (
        mapping[mapping["mapping_status"].eq("mapped")]
        .groupby(["hpdi_parent_group", "hpdi_component"], dropna=False)
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
        .groupby(
            [
                "resolved_food_category",
                "suggested_hpdi_parent_group",
                "suggested_hpdi_component",
            ],
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
                "suggested_hpdi_parent_group": "hpdi_parent_group",
                "suggested_hpdi_component": "hpdi_component",
            }
        )
    )
    review.insert(0, "summary_type", "review_suggestion")
    review["mapping_status"] = "review_required"

    output_columns = [
        "summary_type",
        "mapping_status",
        "hpdi_parent_group",
        "hpdi_component",
        "food_category",
        "food_id_count",
        "event_count",
    ]
    return pd.concat(
        [status, parent_groups, components, review],
        ignore_index=True,
    ).loc[:, output_columns]


def prepare_hpdi_mapping(
    dictionary_csv: Path,
    overrides_csv: Path,
    component_overrides_csv: Optional[Path],
    output_dir: Path,
) -> Tuple[Path, Path, Path, Path, Path]:
    """读取食物字典并保存 hPDI 定义、候选映射和 QC。"""
    if not dictionary_csv.is_file():
        raise FileNotFoundError(f"找不到食物字典：{dictionary_csv}")
    if not overrides_csv.is_file():
        raise FileNotFoundError(f"找不到类别覆盖表：{overrides_csv}")

    dictionary_header = pd.read_csv(
        dictionary_csv,
        encoding="utf-8-sig",
        nrows=0,
    )
    overrides_header = pd.read_csv(
        overrides_csv,
        encoding="utf-8-sig",
        nrows=0,
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
    component_overrides = pd.DataFrame(columns=REQUIRED_COMPONENT_OVERRIDE_COLUMNS)
    if component_overrides_csv is not None and component_overrides_csv.is_file():
        component_overrides_header = pd.read_csv(
            component_overrides_csv,
            encoding="utf-8-sig",
            nrows=0,
        )
        validate_columns(
            component_overrides_header.columns.tolist(),
            REQUIRED_COMPONENT_OVERRIDE_COLUMNS,
            "hPDI 组件覆盖表",
        )
        component_overrides = pd.read_csv(
            component_overrides_csv,
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
        dictionary["food_id_event_count"],
        errors="coerce",
    ).fillna(0)

    resolved = resolve_categories(dictionary, overrides)
    mapping = build_hpdi_mapping(resolved, component_overrides)
    definitions = component_definitions()
    qc = build_qc(mapping)

    output_dir.mkdir(parents=True, exist_ok=True)
    definitions_path = output_dir / "hpdi_component_definitions.csv"
    mapping_path = output_dir / "hpdi_food_id_mapping.csv"
    qc_path = output_dir / "hpdi_mapping_qc.csv"
    review_path = output_dir / "hpdi_review_candidates.csv"
    review_template_path = output_dir / "hpdi_component_review_template.csv"
    definitions.to_csv(definitions_path, index=False, encoding="utf-8-sig")
    mapping.to_csv(mapping_path, index=False, encoding="utf-8-sig")
    qc.to_csv(qc_path, index=False, encoding="utf-8-sig")
    review = mapping[mapping["mapping_status"].eq("review_required")].copy()
    review = review.sort_values(
        "food_id_event_count", ascending=False, na_position="last"
    )
    review.to_csv(review_path, index=False, encoding="utf-8-sig")
    review_template = review.loc[
        :,
        [
            "food_id",
            "canonical_short_food_name",
            "canonical_product_name",
            "resolved_food_category",
            "food_id_event_count",
            "suggested_hpdi_component",
        ],
    ].copy()
    review_template["mapping_status"] = ""
    review_template["hpdi_component"] = ""
    review_template["review_notes"] = ""
    review_template.to_csv(
        review_template_path,
        index=False,
        encoding="utf-8-sig",
    )

    status_summary = qc[qc["summary_type"].eq("mapping_status")].copy()
    total_events = status_summary["event_count"].sum()
    status_summary["event_share"] = status_summary["event_count"] / total_events
    print("=" * 72)
    print(f"脚本版本：{SCRIPT_VERSION}")
    print("hPDI FOOD-ID MAPPING QC")
    print(status_summary.to_string(index=False))
    processed_meat_rows = mapping["mapping_basis"].eq("processed_meat_food_category")
    print(
        "加工肉强制映射："
        f"{int(processed_meat_rows.sum())} 个 food_id，"
        f"{int(mapping.loc[processed_meat_rows, 'food_id_event_count'].sum())} 条事件"
    )
    print("-" * 72)
    component_summary = qc[qc["summary_type"].eq("mapped_component")]
    mapped_components = set(component_summary["hpdi_component"].dropna().astype(str))
    missing_components = sorted(EXPECTED_COMPONENTS - mapped_components)
    print("已出现映射记录的组件：" f"{len(mapped_components)}/18 个")
    print(component_summary.to_string(index=False))
    if missing_components:
        print("尚无已确认 food_id 的组件：" + ", ".join(missing_components))
    print("-" * 72)
    print("待审核类别及名称候选（按事件数排序，前40项）：")
    review_summary = qc[qc["summary_type"].eq("review_suggestion")].sort_values(
        "event_count", ascending=False
    )
    print(review_summary.head(40).to_string(index=False))
    print("-" * 72)
    print(f"组件定义：{definitions_path}")
    print(f"food_id 映射：{mapping_path}")
    print(f"QC 结果：{qc_path}")
    print(f"待审核明细：{review_path}")
    print(f"人工审核模板：{review_template_path}")
    return (
        definitions_path,
        mapping_path,
        qc_path,
        review_path,
        review_template_path,
    )


def parse_args():
    parser = ArgumentParser(description="生成 hPDI food_id 候选映射")
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
        "--component-overrides",
        type=Path,
        default=DEFAULT_COMPONENT_OVERRIDES_CSV,
        help=(
            "已审核的 food_id 级 hPDI 决定；文件不存在时不应用"
            f"（默认：{DEFAULT_COMPONENT_OVERRIDES_CSV}）"
        ),
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
    prepare_hpdi_mapping(
        args.dictionary,
        args.overrides,
        args.component_overrides,
        args.output_dir,
    )
