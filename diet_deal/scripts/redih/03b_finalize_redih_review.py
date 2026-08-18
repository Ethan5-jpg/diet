"""按原始 EDIH 食物定义保守完成 rEDIH 待审核映射。"""

import re
import shutil
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


SCRIPT_VERSION = "2026-08-08-redih-review-finalize-v1"
AUTO_NOTE_PREFIX = "auto_redih_finalize_v1:"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR.parent
REDIH_DIR = PROJECT_DIR / "outputs" / "03_score_mapping" / "redih"
DEFAULT_REVIEW_CSV = REDIH_DIR / "redih_review_candidates.csv"
DEFAULT_EVENTS_CSV = DATA_DIR / "Transfer" / "diet_logging" / "diet_logging_events.csv"
DEFAULT_OVERRIDES_CSV = REDIH_DIR / "redih_component_overrides.csv"
DEFAULT_AUDIT_CSV = REDIH_DIR / "redih_review_resolution_audit.csv"
DEFAULT_NUTRIENT_PROFILES_CSV = REDIH_DIR / "redih_review_nutrient_profiles.csv"
DEFAULT_CHUNK_SIZE = 200000

OVERRIDE_COLUMNS = [
    "food_id",
    "mapping_status",
    "redih_component",
    "review_notes",
]
REVIEW_REQUIRED_COLUMNS = [
    "food_id",
    "canonical_short_food_name",
    "canonical_product_name",
    "resolved_food_category",
    "food_id_event_count",
]
NUTRIENT_COLUMNS = ["food_id", "weight_g", "lipid_g"]
VALID_COMPONENTS: Set[str] = {
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
}

LOW_FAT_MILK_MAX_G_PER_100G = 2.5
HIGH_FAT_MILK_MIN_G_PER_100G = 3.0


def text(value: object) -> str:
    """统一英文标签，缺失值转为空字符串。"""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().casefold())


def hit(name: str, pattern: str) -> bool:
    """执行大小写不敏感正则匹配。"""
    return bool(re.search(pattern, name, flags=re.IGNORECASE))


def primary_name(row: pd.Series) -> str:
    """优先使用短名称，短名称缺失时才回退到商品名。"""
    short_name = text(row.get("canonical_short_food_name"))
    if short_name:
        return short_name
    return text(row.get("canonical_product_name"))


def mapped(component: str, note: str) -> Tuple[str, str, str, str]:
    """生成高置信 mapped 决定。"""
    return "mapped", component, note, "high"


def excluded(
    note: str, confidence: str = "high"
) -> Tuple[str, Optional[str], str, str]:
    """生成保守不适用决定。"""
    return "not_applicable", None, note, confidence


def profile_number(profile: Optional[pd.Series], column: str) -> Optional[float]:
    """安全读取单个 food_id 的营养汇总数值。"""
    if profile is None or column not in profile.index:
        return None
    value = profile.get(column)
    if pd.isna(value):
        return None
    return float(value)


def classify(
    row: pd.Series,
    nutrient_profile: Optional[pd.Series],
) -> Tuple[str, Optional[str], str, str]:
    """按原始 EDIH 明确定义和保守复合食物策略审核一行。"""
    name = primary_name(row)
    category = text(row.get("resolved_food_category"))

    if not name:
        return excluded("label_missing_in_review_row", "medium")

    # HPP 明确未测量 margarine；植物奶、植物奶酪也不能归入乳制品组。
    if hit(name, r"\bmargarine\b"):
        return excluded("margarine_not_measured_in_hpp")
    if hit(
        name,
        r"\b(?:soy ?milk|soymilk|almond (?:milk|beverage)|rice (?:milk|drink)|"
        r"oat (?:milk|drink)|coconut milk|plant[- ]based milk|plant milk|"
        r"vegan cheese|plant[- ]based cheese|coffee creamer|coffee whitener)\b",
    ):
        return excluded("plant_dairy_analogue_not_in_edih_dairy_groups")

    # 先处理定义中允许的特例，避免随后被一般复合词规则排除。
    if hit(name, r"\b(?:diet|zero|sugar[- ]free|no sugar|low[- ]calorie)\b") and hit(
        name,
        r"\b(?:cola|coke|pepsi|soda|soft drink|carbonated|beverage|energy drink)\b",
    ):
        return mapped("low_energy_beverages", "explicit_low_energy_beverage")
    if not hit(name, r"\b(?:diet|zero|sugar[- ]free|no sugar|low[- ]calorie)\b") and hit(
        name,
        r"\b(?:cola|coke|pepsi|soda|soft drink|sweetened beverage|"
        r"energy drink|fruit punch)\b",
    ):
        return mapped("high_energy_drinks", "explicit_high_energy_beverage")
    if hit(name, r"\bwine\b") and not hit(
        name, r"\b(?:vinegar|sauce|cooking|reduction|gravy)\b"
    ):
        return mapped("wine", "explicit_red_or_white_wine")
    if hit(
        name,
        r"^(?:[a-z -]+ )?(?:chowder|cream(?:y)?(?: of [a-z -]+)? soup)$"
        r"|^(?:cream soup|soup [a-z -]*cream)$",
    ):
        return mapped("cream_soups", "explicit_chowder_or_cream_soup")
    if hit(name, r"^(?:french fries?|potato fries?)$"):
        return mapped("french_fries", "explicit_french_fries")
    if hit(name, r"^(?:(?:salted|unsalted|clarified) )?butter$|^ghee$"):
        return mapped("butter", "standalone_butter_or_ghee")
    if hit(
        name,
        r"^(?:(?:fresh|raw|cooked|canned|cherry|grape) )?tomato(?:es)?$"
        r"|^tomato (?:juice|sauce|paste)$",
    ):
        return mapped("tomatoes", "explicit_tomato_item_from_original_definition")
    if hit(name, r"^(?:[a-z -]+ )?ice cream$"):
        return mapped("high_fat_dairy", "ice_cream_in_original_high_fat_dairy")

    # 原定义不把果汁、深色鱼或复合菜的整份克数计入目标组。
    if category == "fruit juices and soft drinks" or hit(
        name, r"\b(?:juice|smoothie|nectar|cider)\b"
    ):
        return excluded("fruit_juice_is_not_whole_fruit_or_edih_beverage")
    if hit(
        name,
        r"\b(?:salmon|sardines?|mackerel|herring|bluefish|swordfish)\b",
    ):
        return excluded("dark_meat_fish_excluded_from_other_fish")
    if hit(
        name,
        r"\b(?:cookies?|biscuits?|cakes?|brownies?|doughnuts?|donuts?|pie|"
        r"pastry|burekas?|popsicle|ice pop|sandwich|pizza|sushi|pasta|"
        r"bolognese|noodles?|bread|roll|casserole|stew|stuffed|salad|"
        r"nuggets?|guacamole)\b",
    ):
        return excluded("composite_food_not_decomposable_to_one_edih_group")

    # 乳制品遵循原始 FFQ 食物组：所有 yogurt 归 low-fat；所有 cheese、
    # whole milk、cream、sour cream 归 high-fat。普通 milk 才使用脂肪密度。
    if category == "hard cheese" or hit(
        name,
        r"\b(?:cheese|whole milk|full[- ]fat milk|heavy cream|whipping cream|"
        r"whipped cream|sour cream|cream cheese|mascarpone|creme fraiche)\b",
    ):
        return mapped("high_fat_dairy", "original_high_fat_dairy_definition")
    if hit(name, r"\b(?:yogurt|yoghurt|sherbet|ice milk)\b"):
        return mapped("low_fat_dairy", "original_low_fat_dairy_definition")
    if hit(
        name,
        r"\b(?:skim(?:med)?|non[- ]fat|fat[- ]free|low[- ]fat|reduced[- ]fat|"
        r"0\s*%|1\s*%|1\.5\s*%|2\s*%)\b[^|]{0,30}\bmilk\b"
        r"|\bmilk\b[^|]{0,30}\b(?:skim(?:med)?|non[- ]fat|fat[- ]free|"
        r"low[- ]fat|reduced[- ]fat|0\s*%|1\s*%|1\.5\s*%|2\s*%)\b",
    ):
        return mapped("low_fat_dairy", "explicit_low_fat_milk")
    if hit(name, r"\bmilk\b") and category in {
        "milk, cream cheese and yogurts",
        "sweet milk products",
    }:
        lipid_density = profile_number(nutrient_profile, "lipid_g_per_100g")
        valid_events = profile_number(
            nutrient_profile, "valid_nutrient_event_count"
        )
        if valid_events is not None and valid_events >= 1 and lipid_density is not None:
            if lipid_density <= LOW_FAT_MILK_MAX_G_PER_100G:
                return mapped(
                    "low_fat_dairy",
                    "generic_milk_lipid_density_le_2_5_g_per_100g",
                )
            if lipid_density >= HIGH_FAT_MILK_MIN_G_PER_100G:
                return mapped(
                    "high_fat_dairy",
                    "generic_milk_lipid_density_ge_3_0_g_per_100g",
                )
        return excluded(
            "generic_milk_missing_or_borderline_lipid_density",
            "medium",
        )

    # 加工肉类别优先于禽肉词；原始 processed-meat 组包括常见腌制肉。
    if category == "processed meat products" or hit(
        name,
        r"\b(?:bacon|sausage|hot dog|salami|pastrami|ham|prosciutto|"
        r"mortadella|pepperoni|cold cuts?|deli meat|processed meat|corned beef)\b",
    ):
        if hit(name, r"^(?:beef )?(?:hamburger|burger) patty$"):
            return mapped("red_meat", "explicit_hamburger_patty")
        return mapped("processed_meat", "explicit_or_category_processed_meat")

    if hit(
        name,
        r"^(?:(?:grilled|roasted|cooked|raw|fresh|lean) )?"
        r"(?:beef|veal|lamb|mutton|pork|steak|hamburger|beef patty)$",
    ):
        return mapped("red_meat", "simple_red_meat_item")
    if hit(
        name,
        r"^(?:(?:grilled|roasted|cooked|raw|fresh) )?"
        r"(?:chicken|turkey)(?: breast| thigh| leg| wing)?$",
    ):
        return mapped("poultry", "simple_chicken_or_turkey_item")
    if hit(name, r"^(?:(?:boiled|fried|poached|scrambled) )?eggs?$|^omelett?e$"):
        return mapped("eggs", "simple_egg_item")

    if category == "fish and seafood" and hit(
        name,
        r"\b(?:fish|seafood|tuna|cod|tilapia|grouper|hake|halibut|sole|"
        r"flounder|haddock|shrimp|prawn|lobster|scallop|crab|calamari|"
        r"squid|octopus)\b",
    ):
        return mapped("other_fish", "simple_non_dark_fish_or_seafood")

    if hit(
        name,
        r"^(?:(?:black|filter|filtered|instant|turkish|decaffeinated|decaf) )?"
        r"coffee$|^(?:espresso|americano|cappuccino|latte|macchiato|mocha)$",
    ):
        return mapped("coffee", "explicit_coffee_beverage")

    if hit(
        name,
        r"\b(?:apple|pear|banana|orange|grapefruit|mandarin|clementine|grape|"
        r"raisins?|berry|strawberry|blueberry|raspberry|blackberry|peach|"
        r"apricot|plum|cherry|mango|pineapple|melon|watermelon|cantaloupe|"
        r"kiwi|figs?|dates?|pomegranate|avocado|goji)\b",
    ):
        return mapped("whole_fruits", "explicit_whole_or_preserved_fruit")

    # 原始 EDIH 的 green-leafy 组只列 spinach 和各类 lettuce；kale/chard
    # 属于另一个未被 HPP EDIH 保留的 cruciferous candidate group。
    if hit(
        name,
        r"^(?:(?:fresh|raw|cooked|frozen|baby) )?"
        r"(?:spinach|iceberg lettuce|head lettuce|romaine|romaine lettuce|"
        r"leaf lettuce|lettuce)$",
    ):
        return mapped("green_leafy_vegetables", "original_green_leafy_definition")

    return excluded("conservative_no_unique_edih_group", "medium")


def validate_columns(
    columns: List[str], required: List[str], source_name: str
) -> None:
    """确认表中存在全部必要列。"""
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError("{}缺少字段：{}".format(source_name, ", ".join(missing)))


def build_nutrient_profiles(
    events: pd.DataFrame,
    target_food_ids: Set[str],
) -> pd.DataFrame:
    """汇总目标 food_id 的有效重量和脂肪，计算加权脂肪密度。"""
    validate_columns(events.columns.tolist(), NUTRIENT_COLUMNS, "饮食事件表")
    data = events.loc[:, NUTRIENT_COLUMNS].copy()
    data["food_id"] = data["food_id"].astype("string").str.strip()
    data["weight_g"] = pd.to_numeric(data["weight_g"], errors="coerce")
    data["lipid_g"] = pd.to_numeric(data["lipid_g"], errors="coerce")
    valid = (
        data["food_id"].isin(target_food_ids)
        & data["weight_g"].gt(0)
        & data["lipid_g"].ge(0)
    )
    data = data.loc[valid].copy()
    if data.empty:
        return pd.DataFrame(
            columns=[
                "food_id",
                "valid_nutrient_event_count",
                "valid_weight_g_total",
                "valid_lipid_g_total",
                "lipid_g_per_100g",
            ]
        )
    profile = (
        data.groupby("food_id", sort=False)
        .agg(
            valid_nutrient_event_count=("food_id", "size"),
            valid_weight_g_total=("weight_g", "sum"),
            valid_lipid_g_total=("lipid_g", "sum"),
        )
        .reset_index()
    )
    profile["lipid_g_per_100g"] = (
        100.0
        * profile["valid_lipid_g_total"]
        / profile["valid_weight_g_total"]
    )
    return profile


def aggregate_nutrients_from_csv(
    events_csv: Path,
    target_food_ids: Set[str],
    chunk_size: int,
) -> pd.DataFrame:
    """分块读取服务器事件表并合并目标 food_id 的营养汇总。"""
    if not events_csv.is_file():
        raise FileNotFoundError("找不到饮食事件表：{}".format(events_csv))
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于0。")
    header = pd.read_csv(events_csv, encoding="utf-8-sig", nrows=0)
    validate_columns(header.columns.tolist(), NUTRIENT_COLUMNS, "饮食事件表")
    partials: List[pd.DataFrame] = []
    reader = pd.read_csv(
        events_csv,
        encoding="utf-8-sig",
        usecols=NUTRIENT_COLUMNS,
        dtype={"food_id": "string"},
        chunksize=chunk_size,
        low_memory=False,
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        partial = build_nutrient_profiles(chunk, target_food_ids)
        if not partial.empty:
            partials.append(partial)
        print("营养数据块 {:,}：命中 {:,} 个 food_id".format(
            chunk_number,
            partial["food_id"].nunique() if not partial.empty else 0,
        ))
    if not partials:
        return build_nutrient_profiles(
            pd.DataFrame(columns=NUTRIENT_COLUMNS),
            target_food_ids,
        )
    combined = pd.concat(partials, ignore_index=True)
    totals = (
        combined.groupby("food_id", sort=False)[
            [
                "valid_nutrient_event_count",
                "valid_weight_g_total",
                "valid_lipid_g_total",
            ]
        ]
        .sum()
        .reset_index()
    )
    totals["lipid_g_per_100g"] = (
        100.0 * totals["valid_lipid_g_total"] / totals["valid_weight_g_total"]
    )
    return totals


def classify_review(
    review: pd.DataFrame,
    nutrient_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """将候选表与营养汇总结合，生成完整逐行审核表。"""
    validate_columns(review.columns.tolist(), REVIEW_REQUIRED_COLUMNS, "待审核表")
    data = review.copy().reset_index(drop=True)
    data["food_id"] = data["food_id"].astype("string").str.strip()
    if data["food_id"].isna().any() or data["food_id"].duplicated().any():
        raise ValueError("待审核表存在缺失或重复 food_id。")
    profile_index: Dict[str, pd.Series] = {}
    if not nutrient_profiles.empty:
        profile_index = {
            str(row["food_id"]): row
            for _, row in nutrient_profiles.iterrows()
        }

    decisions: List[Tuple[str, Optional[str], str, str]] = []
    for _, row in data.iterrows():
        profile = profile_index.get(str(row["food_id"]))
        decisions.append(classify(row, profile))
    decision_frame = pd.DataFrame(
        decisions,
        columns=[
            "mapping_status",
            "redih_component",
            "review_notes",
            "confidence",
        ],
    )
    for column in decision_frame.columns:
        data[column] = decision_frame[column].to_numpy()
    data["review_notes"] = AUTO_NOTE_PREFIX + data["review_notes"].astype("string")
    data["decision_source"] = "automatic_finalize"
    if nutrient_profiles.empty:
        data["valid_nutrient_event_count"] = pd.NA
        data["valid_weight_g_total"] = pd.NA
        data["valid_lipid_g_total"] = pd.NA
        data["lipid_g_per_100g"] = pd.NA
    else:
        data = data.merge(
            nutrient_profiles,
            on="food_id",
            how="left",
            validate="one_to_one",
        )
    return data


def preserve_manual_decisions(
    audit: pd.DataFrame,
    existing_overrides: pd.DataFrame,
) -> pd.DataFrame:
    """已有非自动、非空的人工决定优先于本脚本自动规则。"""
    if existing_overrides.empty:
        return audit.copy()
    validate_columns(
        existing_overrides.columns.tolist(),
        OVERRIDE_COLUMNS,
        "现有 rEDIH override",
    )
    existing = existing_overrides.loc[:, OVERRIDE_COLUMNS].copy()
    for column in OVERRIDE_COLUMNS:
        existing[column] = existing[column].astype("string").str.strip()
    has_status = existing["mapping_status"].notna() & existing["mapping_status"].ne("")
    automatic = existing["review_notes"].fillna("").str.startswith(
        "auto_redih_finalize_"
    )
    manual = existing.loc[has_status & ~automatic].copy()
    if manual["food_id"].duplicated().any():
        raise ValueError("现有人工 override 存在重复 food_id。")
    result = audit.copy()
    manual_index = manual.set_index("food_id")
    for index, food_id in result["food_id"].items():
        if food_id not in manual_index.index:
            continue
        decision = manual_index.loc[food_id]
        result.at[index, "mapping_status"] = decision["mapping_status"]
        result.at[index, "redih_component"] = decision["redih_component"]
        result.at[index, "review_notes"] = decision["review_notes"]
        result.at[index, "confidence"] = "manual"
        result.at[index, "decision_source"] = "existing_manual_override"
    return result


def validate_audit(audit: pd.DataFrame) -> None:
    """验证审核结果已全部落到 mapped 或 not_applicable。"""
    valid_statuses = {"mapped", "not_applicable"}
    if not audit["mapping_status"].isin(valid_statuses).all():
        raise ValueError("审核结果仍含未解决或无效状态。")
    mapped_rows = audit["mapping_status"].eq("mapped")
    invalid_component = mapped_rows & ~audit["redih_component"].isin(
        VALID_COMPONENTS
    )
    if invalid_component.any():
        values = sorted(
            audit.loc[invalid_component, "redih_component"]
            .fillna("<missing>")
            .astype(str)
            .unique()
        )
        raise ValueError("审核结果含无效组件：{}".format(", ".join(values)))
    if audit.loc[~mapped_rows, "redih_component"].notna().any():
        raise ValueError("not_applicable 审核行不应填写组件。")


def build_override_output(
    audit: pd.DataFrame,
    existing_overrides: pd.DataFrame,
) -> pd.DataFrame:
    """合并当前审核结果与不在当前候选表中的既有非空决定。"""
    current = audit.loc[:, OVERRIDE_COLUMNS].copy()
    if existing_overrides.empty:
        return current
    existing = existing_overrides.loc[:, OVERRIDE_COLUMNS].copy()
    existing["food_id"] = existing["food_id"].astype("string").str.strip()
    existing["mapping_status"] = existing["mapping_status"].astype("string").str.strip()
    existing = existing.loc[
        existing["food_id"].notna()
        & existing["mapping_status"].notna()
        & existing["mapping_status"].ne("")
        & ~existing["food_id"].isin(current["food_id"])
    ].copy()
    if existing.empty:
        return current
    output = pd.concat([existing, current], ignore_index=True, sort=False)
    if output["food_id"].duplicated().any():
        raise ValueError("合并后的 rEDIH override 存在重复 food_id。")
    return output.loc[:, OVERRIDE_COLUMNS]


def run(
    review_csv: Path,
    events_csv: Path,
    overrides_csv: Path,
    audit_csv: Path,
    nutrient_profiles_csv: Path,
    chunk_size: int,
) -> Dict[str, Path]:
    """读取服务器候选和事件，保存自动审核、营养审计与 override。"""
    if not review_csv.is_file():
        raise FileNotFoundError("找不到待审核文件：{}".format(review_csv))
    review = pd.read_csv(
        review_csv,
        encoding="utf-8-sig",
        dtype={"food_id": "string"},
        low_memory=False,
    )
    validate_columns(review.columns.tolist(), REVIEW_REQUIRED_COLUMNS, "待审核表")
    target_food_ids = set(
        review["food_id"].astype("string").str.strip().dropna().tolist()
    )
    nutrient_profiles = aggregate_nutrients_from_csv(
        events_csv,
        target_food_ids,
        chunk_size,
    )
    if overrides_csv.is_file():
        existing = pd.read_csv(
            overrides_csv,
            encoding="utf-8-sig",
            dtype={"food_id": "string"},
            low_memory=False,
        )
    else:
        existing = pd.DataFrame(columns=OVERRIDE_COLUMNS)

    audit = classify_review(review, nutrient_profiles)
    audit = preserve_manual_decisions(audit, existing)
    validate_audit(audit)
    output = build_override_output(audit, existing)

    overrides_csv.parent.mkdir(parents=True, exist_ok=True)
    if overrides_csv.is_file():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = overrides_csv.with_name(
            "redih_component_overrides_before_{}.csv".format(stamp)
        )
        shutil.copy2(str(overrides_csv), str(backup))
        print("已备份原 override：{}".format(backup))
    output.to_csv(overrides_csv, index=False, encoding="utf-8-sig")
    audit.to_csv(audit_csv, index=False, encoding="utf-8-sig")
    nutrient_profiles.to_csv(
        nutrient_profiles_csv,
        index=False,
        encoding="utf-8-sig",
    )

    event_counts = pd.to_numeric(audit["food_id_event_count"], errors="coerce").fillna(0)
    report = audit.assign(__event_count=event_counts)
    status = (
        report.groupby("mapping_status")
        .agg(
            food_id_count=("food_id", "nunique"),
            event_count=("__event_count", "sum"),
        )
        .sort_values("event_count", ascending=False)
    )
    components = (
        report.loc[report["mapping_status"].eq("mapped")]
        .groupby("redih_component")
        .agg(
            food_id_count=("food_id", "nunique"),
            event_count=("__event_count", "sum"),
        )
        .sort_values("event_count", ascending=False)
    )
    conservative = report.loc[report["confidence"].eq("medium")].sort_values(
        "__event_count", ascending=False
    )

    print("=" * 88)
    print("rEDIH 待审核自动结案")
    print("SCRIPT_VERSION={}".format(SCRIPT_VERSION))
    print("输入待审核 food_id：{:,}".format(len(audit)))
    print("\n== 决定状态 ==")
    print(status.to_string())
    print("\n== 新增 mapped 组件 ==")
    print(components.to_string())
    print("\n保守排除（medium）：{:,} 个 food_id，{:,} 条事件".format(
        conservative["food_id"].nunique(),
        int(conservative["__event_count"].sum()),
    ))
    print("保留既有人工决定：{:,}".format(
        int(report["decision_source"].eq("existing_manual_override").sum())
    ))
    print("override：{}".format(overrides_csv))
    print("审计表：{}".format(audit_csv))
    print("营养表：{}".format(nutrient_profiles_csv))
    print("下一步：python3 scripts/redih/03_prepare_redih_mapping.py")
    print("=" * 88)
    return {
        "overrides": overrides_csv,
        "audit": audit_csv,
        "nutrient_profiles": nutrient_profiles_csv,
    }


def parse_args() -> object:
    """解析命令行参数。"""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--events-csv", type=Path, default=DEFAULT_EVENTS_CSV)
    parser.add_argument("--overrides-csv", type=Path, default=DEFAULT_OVERRIDES_CSV)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT_CSV)
    parser.add_argument(
        "--nutrient-profiles-csv",
        type=Path,
        default=DEFAULT_NUTRIENT_PROFILES_CSV,
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""
    args = parse_args()
    run(
        args.review_csv,
        args.events_csv,
        args.overrides_csv,
        args.audit_csv,
        args.nutrient_profiles_csv,
        args.chunk_size,
    )


if __name__ == "__main__":
    main()
