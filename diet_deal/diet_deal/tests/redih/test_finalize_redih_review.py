import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_DIR / "scripts" / "redih" / "03b_finalize_redih_review.py"
SPEC = importlib.util.spec_from_file_location("redih_review_finalizer", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("无法加载测试模块：{}".format(SCRIPT_PATH))
FINALIZER = importlib.util.module_from_spec(SPEC)
sys.modules["redih_review_finalizer"] = FINALIZER
SPEC.loader.exec_module(FINALIZER)


def review_row(short_name, category, product_name=""):
    return pd.Series(
        {
            "food_id": short_name.lower().replace(" ", "_"),
            "canonical_short_food_name": short_name,
            "canonical_product_name": product_name,
            "resolved_food_category": category,
            "food_id_event_count": 1,
        }
    )


class RedihReviewClassificationTests(unittest.TestCase):
    def assert_decision(self, short_name, category, status, component=None):
        decision = FINALIZER.classify(review_row(short_name, category), None)
        self.assertEqual(decision[0], status)
        self.assertEqual(decision[1], component)

    def test_original_edih_examples_are_mapped(self):
        examples = [
            ("Avocado", "Med Oil and fats", "whole_fruits"),
            ("Natural Yogurt", "milk, cream cheese and yogurts", "low_fat_dairy"),
            ("Yellow Cheese", "Hard cheese", "high_fat_dairy"),
            ("Light Cream Cheese", "milk, cream cheese and yogurts", "high_fat_dairy"),
            ("Ice cream", "sweets", "high_fat_dairy"),
            ("Cappuccino", "Drinks", "coffee"),
            ("Canned tuna", "Fish and seafood", "other_fish"),
            ("Pastrami", "Processed meat products", "processed_meat"),
            ("French fries", "Pasta, Grains and Side dishes", "french_fries"),
            ("Cream of mushroom soup", "Soups and sauces", "cream_soups"),
            ("Tomato paste", "Soups and sauces", "tomatoes"),
            ("Red wine", "Alcoholic drinks", "wine"),
            ("Spinach", "Vegetables", "green_leafy_vegetables"),
            ("Boiled egg", "Pasta, Grains and Side dishes", "eggs"),
        ]
        for short_name, category, component in examples:
            with self.subTest(short_name=short_name):
                self.assert_decision(short_name, category, "mapped", component)

    def test_false_name_suggestions_and_composites_are_excluded(self):
        examples = [
            ("Orange juice", "fruit juices and soft drinks"),
            ("Butter Cookies", "sweets"),
            ("Apple Cake", "sweets"),
            ("Popsicle", "sweets"),
            ("Smoked Salmon", "Fish and seafood"),
            ("Sushi", "Pasta, Grains and Side dishes"),
            ("Pasta Bolognese", "Pasta, Grains and Side dishes"),
            ("Chicken pasta", "Pasta, Grains and Side dishes"),
            ("Egg sandwich", "Bread"),
            ("Guacamole", "Med Oil and fats"),
            ("Soymilk", "milk, cream cheese and yogurts"),
            ("Kale", "Vegetables"),
        ]
        for short_name, category in examples:
            with self.subTest(short_name=short_name):
                self.assert_decision(short_name, category, "not_applicable")

    def test_generic_milk_uses_fat_density_with_conservative_gap(self):
        row = review_row("Milk", "milk, cream cheese and yogurts")
        low = pd.Series({"lipid_g_per_100g": 1.5, "valid_nutrient_event_count": 20})
        high = pd.Series({"lipid_g_per_100g": 3.4, "valid_nutrient_event_count": 20})
        gap = pd.Series({"lipid_g_per_100g": 2.7, "valid_nutrient_event_count": 20})
        self.assertEqual(FINALIZER.classify(row, low)[1], "low_fat_dairy")
        self.assertEqual(FINALIZER.classify(row, high)[1], "high_fat_dairy")
        self.assertEqual(FINALIZER.classify(row, gap)[0], "not_applicable")
        self.assertEqual(FINALIZER.classify(row, None)[0], "not_applicable")

    def test_nutrient_profile_is_weighted_by_valid_event_weight(self):
        events = pd.DataFrame(
            [
                ("milk", 100.0, 1.0),
                ("milk", 200.0, 4.0),
                ("milk", 0.0, 10.0),
                ("milk", 100.0, -1.0),
                ("other", 100.0, 50.0),
            ],
            columns=["food_id", "weight_g", "lipid_g"],
        )
        profile = FINALIZER.build_nutrient_profiles(events, {"milk"}).set_index("food_id")
        self.assertEqual(int(profile.loc["milk", "valid_nutrient_event_count"]), 2)
        self.assertAlmostEqual(float(profile.loc["milk", "lipid_g_per_100g"]), 5.0 / 3.0)

    def test_existing_manual_decision_is_preserved(self):
        audit = pd.DataFrame(
            [
                {
                    "food_id": "manual",
                    "mapping_status": "not_applicable",
                    "redih_component": pd.NA,
                    "review_notes": "auto decision",
                    "confidence": "medium",
                    "decision_source": "automatic_finalize",
                }
            ]
        )
        existing = pd.DataFrame(
            [
                {
                    "food_id": "manual",
                    "mapping_status": "mapped",
                    "redih_component": "whole_fruits",
                    "review_notes": "reviewed by investigator",
                }
            ]
        )
        resolved = FINALIZER.preserve_manual_decisions(audit, existing)
        self.assertEqual(resolved.loc[0, "mapping_status"], "mapped")
        self.assertEqual(resolved.loc[0, "redih_component"], "whole_fruits")
        self.assertEqual(resolved.loc[0, "decision_source"], "existing_manual_override")


if __name__ == "__main__":
    unittest.main()
