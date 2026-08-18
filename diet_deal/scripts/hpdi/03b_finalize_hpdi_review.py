"""Resolve all remaining labelled hPDI review candidates conservatively.

The script reads the exact server-side candidate table, applies ordered rules
anchored to Supplementary Table 11, and writes the component override file
consumed by 03_prepare_hpdi_mapping.py.  It also writes a wider audit table so
every decision remains inspectable.
"""

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


SCRIPT_VERSION = "2026-08-08-hpdi-review-finalize-v2"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
HPDI_DIR = PROJECT_DIR / "outputs" / "03_score_mapping" / "hpdi"
INPUT_CSV = HPDI_DIR / "hpdi_review_candidates.csv"
OUTPUT_CSV = HPDI_DIR / "hpdi_component_overrides.csv"
AUDIT_CSV = HPDI_DIR / "hpdi_review_resolution_audit.csv"

VALID_COMPONENTS = {
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
}


def text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().casefold())


def hit(name: str, pattern: str) -> bool:
    return bool(re.search(pattern, name, flags=re.IGNORECASE))


def mapped(component: str, note: str, confidence: str = "high") -> Tuple[str, str, str, str]:
    return "mapped", component, note, confidence


def excluded(note: str, confidence: str = "high") -> Tuple[str, Optional[str], str, str]:
    return "not_applicable", None, note, confidence


def classify(row: pd.Series) -> Tuple[str, Optional[str], str, str]:
    short_name = text(row.get("canonical_short_food_name"))
    product_name = text(row.get("canonical_product_name"))
    category = text(row.get("resolved_food_category"))
    name = " ".join(part for part in (short_name, product_name) if part)

    # Terms indicating that animal-food words describe an analogue rather than
    # meat/dairy itself.  Without an ingredient breakdown, processed analogues
    # cannot be assigned to a unique healthy plant group.
    plant_analogue = hit(
        name,
        r"\b(?:(?:vegan|vegetarian|plant[- ]based|from the plant)[^|]{0,60}"
        r"(?:burger|hamburger|sausage|schnitzel|meatballs?|meat|shawarma|"
        r"pastrami|kebab|bacon|breast|mincemeat|nuggets?)|"
        r"(?:burger|hamburger|sausage|schnitzel|meatballs?|shawarma|pastrami|"
        r"kebab|bacon|breast|mincemeat)[^|]{0,60}(?:vegan|vegetarian|plant)|"
        r"meat substitute|seitan|seiten|tivol|tivolov|tevol|tevolov|"
        r"te[g]?ul|teba deli|teva deli|zoglo|zoglavac|zoglovac|zoglubak|"
        r"veggie (?:burger|sausage|schnitzel)|soy chips like|"
        r"soybean vegetable meat|corn schnitzel|vegetable sausages?)\b",
    )
    plant_analogue |= category == "industrialized vegetarian food ready to eat" and hit(
        name,
        r"\b(?:burger|hamburger|sausage|schnitzel|meatballs?|shawarma|"
        r"pastrami|kebab|bacon|breast|fillet|omelett?e)\b",
    )
    plant_analogue |= hit(
        name, r"\bsoy[^|]{0,35}(?:chicken|beef|meat|fillet)\b"
    )

    # Supplementary Table 11 miscellaneous animal-food group.
    if hit(name, r"\bpizza\b"):
        return mapped("miscellaneous_animal_foods", "table11_pizza")
    mayo_or_creamy_dressing = hit(
        name,
        r"\b(?:mayonn?aise|mayonaise|mayo|caesar dressing|ranch dressing|"
        r"thousand island(?: sauce| dressing)?|creamy salad dressing|"
        r"yogurt dressing|blue cheese salad dressing|russian salad dressing|"
        r"hellmans?[^|]{0,30}(?:sauce|dressing)|"
        r"vegenaise|veganaise|vegenaise)\b",
    )
    if mayo_or_creamy_dressing and hit(
        name, r"\b(?:vegan|vegetarian|plant[- ]based|vegenaise|veganaise)\b"
    ):
        return excluded("plant_mayonnaise_no_unique_table11_group")
    if mayo_or_creamy_dressing:
        return mapped("miscellaneous_animal_foods", "table11_mayonnaise_or_creamy_dressing")
    if hit(name, r"\b(?:chowder|cream(?:y)?(?: of [a-z -]+)? soup|soup .*\bcream\b)\b"):
        return mapped("miscellaneous_animal_foods", "table11_chowder_or_cream_soup")

    if hit(name, r"\b(?:sparkling|flavou?red) (?:mineral )?water\b") and not hit(
        name, r"\b(?:sweetened|sugar)\b"
    ):
        return excluded("unspecified_or_unsweetened_flavoured_water")

    # Condiments whose fruit/meat words describe flavour or intended use, not
    # the food's hPDI component.
    if hit(name, r"\b(?:seasoning for chicken|mango curry sauce|plum sauce|"
                 r"mustard and honey sauce|honey sauce|apple balsamic vinegar|"
                 r"pomegranate vinegar|rice vinegar|wine vinegar|vinegar)\b"):
        return excluded("outside_18_groups_condiment")

    # Some beverage labels mention added vitamins but are still ordinary
    # flavoured drinks rather than supplements.
    if category == "fruit juices and soft drinks" and hit(
        name, r"\b(?:flavou?rs?|flavou?red|drink)\b"
    ) and not hit(name, r"\b(?:diet|sugar[- ]free|without sugar|juice|nectar|smoothie|tea|coffee)\b"):
        return mapped("sugar_sweetened_beverages", "fruit_drink_category_flavoured_drink")
    if category == "fruit juices and soft drinks" and hit(
        name, r"\bjuice\b"
    ) and hit(name, r"\b(?:added sugar|sweetened|honey)\b"):
        return mapped("sugar_sweetened_beverages", "sweetened_juice_drink")
    if hit(name, r"\b(?:orange|apple|pineapple)[^|]{0,35}(?:carrot|beet|beetroot)[^|]{0,20}juice\b"):
        return mapped("fruit_juice", "mixed_fruit_and_vegetable_juice")

    # Foods outside the paper's 18 groups.  Apply before broad food keywords so
    # whey protein, collagen, supplements, and plant analogues cannot be
    # mistaken for dairy/meat/healthy whole foods.
    if hit(
        name,
        r"\b(?:protein powder|protein isolate|protein bar|pea protein|"
        r"powder[^|]{0,30}protein|protein[^|]{0,30}powder|whey\b|collagen|creatine|"
        r"amino acids?|vitamin(?:s)?(?:\s+[a-z0-9]+)?|magnesium|calcium lactate|"
        r"mineral supplement|preworkout|supplement|probiotics?|omega\s*3|"
        r"psyllium|maca powder|spirulina|moringa|bee pollen|chlorophyll|"
        r"lecithin|guar gum|pectin|gelatin|baking soda|baking powder|"
        r"cream of tartar|artificial vanilla extract|pure vanilla extract|"
        r"compressed fresh yeast|dry yeast|carnivore protein|mct powder|energy gel)\b",
    ):
        return excluded("outside_18_groups_supplement_or_additive")
    if plant_analogue:
        return excluded("processed_plant_analogue_no_unique_table11_group")

    # Non-dairy analogues do not have a dedicated hPDI group.  Plant creams
    # and margarine are retained as the available vegetable-oil proxy, whereas
    # plant milks/cheeses are conservatively excluded.
    if hit(name, r"\b(?:soy cream|rice cream|vegetable cream|plant cream|"
                 r"coconut cream|vegan sweet cream|margarine)\b"):
        return mapped("vegetable_oils", "plant_cream_or_margarine_oil_proxy")
    if hit(name, r"\b(?:almond coconut drink|almond drink|coconut drink|"
                 r"soy milk|almond milk|coconut milk|plant milk|"
                 r"vegan cheese|plant[- ]based cheese|creamy vegan cheese|"
                 r"(?:vegan|plant[- ]based)[^|]{0,30}cream|"
                 r"coffee whitener|coffee creamer|cremora)\b"):
        return excluded("plant_dairy_analogue_no_table11_group")

    # Sugar-free/diet drinks and gums do not meet the SSB definition.
    if hit(name, r"\b(?:diet|sugar[- ]free|without sugar|low calorie sugar substitute|"
                 r"sweet\s*(?:'n|and)\s*low|saccharin|stevia|maltitol)\b"):
        return excluded("not_sugar_sweetened_or_nonfood_sweetener")

    # Strong principal-food phrases take precedence over flavourings/fillings.
    if hit(name, r"\bcaramelized popcorn\b"):
        return mapped("sweets_desserts", "caramelized_popcorn_sweet")
    if hit(name, r"\b(?:cookies?|biscuits?|cakes?|brownies?|doughnuts?|donuts?|"
                 r"pudding|cand(?:y|ies)|pastry|strudel)\b"):
        return mapped("sweets_desserts", "principal_sweet_or_dessert")
    if hit(name, r"\bpopcorn\b"):
        return mapped("whole_grains", "table11_popcorn")
    if hit(name, r"\b(?:peanut|almond|rice bran|palm|coconut|olive|canola|"
                 r"sunflower|soybean|sesame|corn|poppy|avocado) oil\b") and not hit(
        name, r"\b(?:chicken|turkey|beef|veal|lamb|pork|liver|salami|meat)\b"
    ):
        return mapped("vegetable_oils", "explicit_vegetable_oil")
    if hit(name, r"\bsweet potato\b"):
        return mapped("vegetables", "table11_sweet_potato_is_vegetable")
    if hit(name, r"\b(?:corn (?:tortilla )?(?:snack|chips?|crisps?)|"
                 r"cornmeal snack|corn crispy|nachos|"
                 r"potato (?:snack|chips?|crisps?))\b"):
        return mapped("potatoes", "table11_principal_potato_or_corn_chip")
    if hit(name, r"\b(?:oat(?:meal)?|whole ?grain)[^|]{0,35}snack\b"):
        return mapped("whole_grains", "principal_whole_grain_snack")
    if hit(name, r"\bcorny snack\b"):
        return excluded("ambiguous_cereal_snack_no_unique_table11_group", "medium")

    # Animal groups.  These precede soup/category fallbacks and require explicit
    # food evidence.
    if hit(name, r"\b(?:sheep fat|beef fat|chicken fat|animal fat|lard|tallow|"
                 r"butter\b|ghee\b)\b"):
        if not hit(name, r"\b(?:peanut butter|popcorn)\b"):
            return mapped("animal_fat", "explicit_animal_fat")
    if hit(name, r"\b(?:fish sauce|anchov(?:y|ies)|tuna|salmon|sardine|"
                 r"fish\b|seafood|shrimp|prawn|lobster|crab)\b"):
        return mapped("fish_seafood", "explicit_fish_or_seafood")
    if hit(name, r"\b(?:egg|eggs|omelett?e)\b"):
        return mapped("eggs", "explicit_egg")
    if hit(name, r"\b(?:milk|yogurt|yoghurt|cheese|sour cream|creme fraiche|"
                 r"cr[eè]me fresh|whipping cream|whipped cream|cooking cream|"
                 r"cream sauce|ice cream|salep|orchid .*milk[- ]based)\b"):
        return mapped("dairy", "explicit_dairy")
    if hit(name, r"\b(?:chicken|turkey|beef|veal|lamb|pork|bacon|salami|"
                 r"pastrami|liver|hamburger|mcdonald|mcdoland|mcroyal|"
                 r"nuggets?|meat soup|meat stock|chicken stock)\b"):
        return mapped("meat", "explicit_meat")
    if hit(name, r"\b(?:pea|bean|lentil) soup\b"):
        return mapped("legumes", "principal_legume_soup")
    if hit(name, r"\b(?:tomato|vegetable|mushroom|onion|broccoli|cauliflower|"
                 r"corn|minestrone)[^|]{0,30}soup\b"):
        return mapped("vegetables", "principal_vegetable_soup")

    # Whole or preserved fruits remain fruit even if the packing liquid contains
    # sugar; standalone jam/syrup/candy is handled as sweets below.
    if hit(name, r"\b(?:dried|frozen|cooked|canned|preserved|raisins?|prunes?)\b") and hit(
        name,
        r"\b(?:apple|pear|peach|apricot|plum|prune|cherr(?:y|ies)|grape|"
        r"raisin|cranberr(?:y|ies)|blueberr(?:y|ies)|papaya|lychee|"
        r"pineapple|rhubarb|quince|nectarine|tangerine|black currants?)\b",
    ) and not hit(name, r"\b(?:jam|jelly|candied)\b"):
        return mapped("fruits", "whole_or_preserved_fruit")

    # A drink made from diluted syrup/concentrate is an SSB, whereas an
    # explicitly 100% fruit concentrate is treated as fruit juice.
    if hit(name, r"\b(?:100%[^|]{0,30}concentrate|concentrate 100%)\b"):
        return mapped("fruit_juice", "explicit_100_percent_fruit_concentrate")
    if hit(name, r"\b(?:cranberry|blueberry|apple|orange|pomegranate|grape|"
                 r"pineapple|lemon|lime) concentrate\b"):
        return mapped("fruit_juice", "named_fruit_juice_concentrate")
    if hit(name, r"\b(?:diluted[^|]{0,30}syrup|syrup[^|]{0,30}(?:diluted|drink)|"
                 r"fruit concentrate|chocolate almond drink)\b"):
        return mapped("sugar_sweetened_beverages", "sweetened_syrup_or_concentrate_drink")

    # Sweets and desserts, including the paper's syrup/honey/jam examples.
    if hit(
        name,
        r"\b(?:chocolate|cocoa mass|cookies?|biscuits?|cakes?|brownie|"
        r"doughnuts?|donuts?|pastry|strudel|pudding|cand(?:y|ies)|sour candies|"
        r"jam\b|jelly|preserves?|honey|syrup(?! for making)|"
        r"candied|sweet snack|caramelized popcorn|sweetened cranberries|"
        r"sugar\b|carob cream)\b",
    ):
        return mapped("sweets_desserts", "table11_sweet_or_syrup")

    # Tea and coffee are a distinct healthy plant group.  Creamers and powders
    # were already handled above.
    if hit(name, r"\b(?:coffee|cappuccino|espresso|americano|nescafe|tea\b|chai\b)\b"):
        return mapped("tea_coffee", "explicit_tea_or_coffee")

    # Drinks: vegetable juice stays with vegetables; genuine fruit juice,
    # nectar, cider, and smoothies go to fruit_juice.  Sweetened/flavoured drink
    # products go to SSB.
    if hit(name, r"\b(?:tomato|carrot|celery|beet|beetroot|wheat ?grass) juice\b"):
        return mapped("vegetables", "table11_vegetable_juice")
    if hit(
        name,
        r"\b(?:sweetened carbonated|carbonated sweetened|carbonated[^|]{0,30}drink|soft drink|"
        r"fruit[- ]flavou?red drink|flavou?red drink|fruit drink|"
        r"energy drink|lemonade|powdered (?:fruit|strawberry) juice|"
        r"drink prepared from sweetened powder|syrup for making|"
        r"concentrate lemonade|bitter lemon|tonic water|fizzy drink)\b",
    ):
        return mapped("sugar_sweetened_beverages", "table11_sugar_sweetened_drink")
    if category == "fruit juices and soft drinks" and hit(
        name, r"\b(?:flavou?rs?|flavou?red|drink)\b"
    ) and not hit(name, r"\b(?:juice|nectar|smoothie|tea|coffee)\b"):
        return mapped("sugar_sweetened_beverages", "fruit_drink_category_flavoured_drink")
    if hit(name, r"\b(?:juice|nectar|smoothie|non[- ]alcoholic cider)\b"):
        return mapped("fruit_juice", "table11_fruit_juice_or_nectar")
    if category == "fruit juices and soft drinks" and hit(
        name,
        r"\b(?:apple|pear|peach|plum|prune|cherr(?:y|ies)|grape|cranberry|"
        r"blueberry|strawberry|orange|grapefruit|tangerine|banana|mango|"
        r"pineapple|watermelon|melon|pomegranate|kiwi|guava|pomelo)\b",
    ):
        return mapped("fruit_juice", "fruit_name_in_juice_category")

    # Paper-specific potatoes rule.  Sweet potato/yam is explicitly a
    # vegetable in Table 11; ordinary potato and potato/corn chips are reverse
    # scored as potatoes.
    if hit(name, r"\b(?:sweet potato|yam)\b"):
        return mapped("vegetables", "table11_sweet_potato_is_vegetable")
    if hit(name, r"\b(?:potato|french fries?|corn chips?|corn crisps?|"
                 r"corn snack|tortilla chips?|doritos|cheetos|bissli|apropo)\b"):
        return mapped("potatoes", "table11_potato_or_corn_chips")

    # Whole grains must have explicit whole-grain evidence.  Refined grain
    # products are reverse-scored.
    if hit(name, r"\b(?:whole ?grain|wholemeal|whole wheat|brown rice|black rice|"
                 r"quinoa|buckwheat|oat(?:meal)?|barley|bran|spelt|rye|"
                 r"popcorn|wheat germ)\b"):
        return mapped("whole_grains", "table11_explicit_whole_grain")
    if hit(name, r"\b(?:pretzels?|crackers?|croutons?|pita|grissini|"
                 r"noodles?|pancakes?|white rice|rice (?:snack|munchies)|"
                 r"multigrain snack|flour mix|bread snack|"
                 r"soup mandels?|cornflakes|onion rings|snack rings)\b"):
        return mapped("refined_grains", "table11_refined_grain_food")

    # Nuts/seeds and legumes.  The source dictionary's combined nuts/seeds
    # category is retained as the study's nuts proxy.
    if hit(name, r"\b(?:peanut|almond|walnut|cashew|pistachio|hazelnut|"
                 r"nuts?\b|seeds?\b|sesame snack)\b"):
        return mapped("nuts", "explicit_nut_or_seed")
    if hit(name, r"\b(?:tofu|soybeans?|beans?\b|lentils?|peas?\b|hummus|"
                 r"falafel|miso\b|chickpeas?)\b"):
        return mapped("legumes", "table11_legume_or_soy_food")

    # Vegetable oils and oil-based dressings.  Plant creams/margarines are
    # used as the available vegetable-oil proxy; pure MCT oil remains excluded
    # because its source is not identifiable in these labels.
    if hit(name, r"\b(?:olive oil|canola oil|sunflower oil|soybean oil|"
                 r"sesame oil|corn oil|rice bran oil|rice oil|palm oil|coconut oil|"
                 r"peanut oil|avocado oil|poppy oil|seed oil|vegetable oil|margarine|"
                 r"vegetable hydrogenated fat|vinaigrette|pesto|"
                 r"oil[- ]based dressing|plant cream|soy cream|rice cream|tahini)\b"):
        return mapped("vegetable_oils", "table11_vegetable_oil_or_oil_dressing")
    if hit(name, r"\bmct oil\b"):
        return excluded("mct_source_not_identifiable")

    # Fruits and vegetables after juice/sweets rules so a fruit-flavoured drink
    # or syrup cannot be counted as whole fruit.
    if hit(name, r"\b(?:apple|pear|peach|apricot|plum|prune|cherr(?:y|ies)|"
                 r"grape|raisin|cranberr(?:y|ies)|blueberr(?:y|ies)|"
                 r"strawberr(?:y|ies)|raspberr(?:y|ies)|black currants?|"
                 r"orange|grapefruit|tangerine|mandarin|banana|mango|"
                 r"pineapple|watermelon|melon|papaya|lychee|rhubarb|"
                 r"pomegranate|kiwi|dates?\b|figs?\b|quince|nectarine)\b"):
        return mapped("fruits", "explicit_whole_or_preserved_fruit")
    if hit(
        name,
        r"\b(?:tomato|broccoli|cauliflower|cabbage|carrot|beetroot|beet|"
        r"spinach|kale|lettuce|celery|mushroom|eggplant|zucchini|onion|"
        r"garlic|pepper|cucumber|corn\b|olives?\b|turnip|sauerkraut|"
        r"kimchi|okra|artichoke|palm hearts?|vegetable soup|pea soup|"
        r"minestrone|coleslaw|truffle|seaweed|taro|salsa|ketchup)\b",
    ):
        return mapped("vegetables", "explicit_vegetable")

    # Ordinary condiments, flavourings, extracts, and ingredients have no
    # unique Table 11 group.  Chicken/fish/tomato/miso products were caught
    # earlier.
    if hit(
        name,
        r"\b(?:vinegar|soy sauce|mustard|ketchup|chili sauce|hot sauce|"
        r"barbecue sauce|garlic sauce|curry sauce|curry paste|schug|sechug|"
        r"wasabi|horseradish|tabasco|seasoning|stock powder|dashi|"
        r"salt\b|sweetener|extract|cocoa powder|carob powder|maltitol|"
        r"gum\b|water\b|orchid .*water[- ]based|spread with olives)\b",
    ):
        return excluded("outside_18_groups_condiment_additive_or_water")

    # Category-assisted fallbacks used only when the product label itself does
    # not give a unique Table 11 group.
    if category == "low calories and diet drinks":
        return excluded("diet_drink_not_ssb")
    if category == "alcoholic drinks":
        return excluded("alcohol_not_in_hpdi_18_groups")
    if category in {
        "others",
        "soups and sauces",
        "snacks",
        "fast foods",
        "deep fried foods",
        "industrialized vegetarian food ready to eat",
        "oils and fats",
        "fruit juices and soft drinks",
        "canned veg and fruits",
    }:
        return excluded("conservative_no_unique_table11_group", "medium")

    return excluded("conservative_unclassified_labelled_food", "medium")


def main() -> None:
    if not INPUT_CSV.is_file():
        raise FileNotFoundError("找不到待审核文件：{}".format(INPUT_CSV))

    review = pd.read_csv(INPUT_CSV, encoding="utf-8-sig", dtype={"food_id": "string"})
    required = {
        "food_id",
        "canonical_short_food_name",
        "canonical_product_name",
        "resolved_food_category",
        "food_id_event_count",
    }
    missing = sorted(required - set(review.columns))
    if missing:
        raise ValueError("待审核文件缺少字段：{}".format(", ".join(missing)))
    if review["food_id"].isna().any() or review["food_id"].duplicated().any():
        raise ValueError("待审核文件存在缺失或重复 food_id。")

    decisions = review.apply(classify, axis=1, result_type="expand")
    decisions.columns = ["mapping_status", "hpdi_component", "review_notes", "confidence"]
    # The candidate table already contains preliminary columns with these
    # names.  Overwrite them instead of concatenating duplicate column names;
    # otherwise audit["hpdi_component"] returns a DataFrame rather than a
    # Series and the validation below becomes ambiguous.
    audit = review.reset_index(drop=True).copy()
    for column in decisions.columns:
        audit[column] = decisions[column].to_numpy()
    if audit.columns.duplicated().any():
        duplicates = sorted(set(audit.columns[audit.columns.duplicated()].tolist()))
        raise AssertionError("审计表存在重复列名：{}".format(", ".join(duplicates)))

    invalid = audit["mapping_status"].eq("mapped") & ~audit["hpdi_component"].isin(VALID_COMPONENTS)
    if invalid.any():
        raise AssertionError("存在无效的 mapped 组件。")
    if audit.loc[audit["mapping_status"].eq("not_applicable"), "hpdi_component"].notna().any():
        raise AssertionError("not_applicable 行不应含 hpdi_component。")

    new_overrides = audit.loc[:, ["food_id", "mapping_status", "hpdi_component", "review_notes"]]
    if OUTPUT_CSV.is_file():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = OUTPUT_CSV.with_name("hpdi_component_overrides_before_{}.csv".format(stamp))
        shutil.copy2(str(OUTPUT_CSV), str(backup))
        existing = pd.read_csv(OUTPUT_CSV, encoding="utf-8-sig", dtype={"food_id": "string"})
        existing = existing[~existing["food_id"].isin(new_overrides["food_id"])]
        new_overrides = pd.concat([existing, new_overrides], ignore_index=True, sort=False)

    new_overrides.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    audit.to_csv(AUDIT_CSV, index=False, encoding="utf-8-sig")

    audit["food_id_event_count"] = pd.to_numeric(audit["food_id_event_count"], errors="coerce").fillna(0)
    status = audit.groupby("mapping_status").agg(
        food_id_count=("food_id", "nunique"),
        event_count=("food_id_event_count", "sum"),
    )
    components = audit[audit["mapping_status"].eq("mapped")].groupby("hpdi_component").agg(
        food_id_count=("food_id", "nunique"),
        event_count=("food_id_event_count", "sum"),
    ).sort_values("event_count", ascending=False)
    conservative = audit[audit["confidence"].eq("medium")].sort_values(
        "food_id_event_count", ascending=False
    )

    print("=" * 78)
    print("脚本版本：{}".format(SCRIPT_VERSION))
    print("输入待审核 food_id：{:,}".format(review["food_id"].nunique()))
    print("\n== 决定状态 ==")
    print(status.to_string())
    print("\n== mapped 组件 ==")
    print(components.to_string())
    print("\n保守排除（medium confidence）：{:,} 个 food_id，{:,} 条事件".format(
        conservative["food_id"].nunique(), int(conservative["food_id_event_count"].sum())
    ))
    if not conservative.empty:
        cols = ["food_id", "food_id_event_count", "resolved_food_category", "canonical_product_name", "review_notes"]
        print(conservative.loc[:, cols].head(15).to_string(index=False))
    print("\noverride：{}".format(OUTPUT_CSV))
    print("审计表：{}".format(AUDIT_CSV))
    print("下一步：python3 scripts/hpdi/03_prepare_hpdi_mapping.py")


if __name__ == "__main__":
    main()
