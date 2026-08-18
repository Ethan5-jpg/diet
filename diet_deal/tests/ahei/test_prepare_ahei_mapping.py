import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_DIR / "scripts" / "ahei" / "03_prepare_ahei_mapping.py"
SPEC = importlib.util.spec_from_file_location("ahei_mapping_builder", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("无法加载测试模块：{}".format(SCRIPT_PATH))
MAPPER = importlib.util.module_from_spec(SPEC)
sys.modules["ahei_mapping_builder"] = MAPPER
SPEC.loader.exec_module(MAPPER)


class AheiMappingTests(unittest.TestCase):
    def test_reviewed_hpdi_and_amed_mappings_translate_without_juice_leakage(self):
        food_ids = [
            "fruit",
            "juice",
            "ssb",
            "veg",
            "wg",
            "nuts",
            "legumes",
            "red",
            "chicken",
            "missing",
        ]
        dictionary = pd.DataFrame(
            {
                "food_id": food_ids,
                "canonical_short_food_name": [
                    "Apple", "Apple juice", "Cola", "Carrot", "Oatmeal",
                    "Walnut", "Beans", "Beef", "Chicken", pd.NA,
                ],
                "canonical_product_name": [
                    "Apple", "Apple juice", "Cola", "Carrot", "Oatmeal",
                    "Walnut", "Beans", "Beef", "Chicken", pd.NA,
                ],
                "canonical_food_category": [
                    "Fruits", "Fruit juices and soft drinks",
                    "Fruit juices and soft drinks", "Vegetables", "Cereals",
                    "Nuts, seeds, and products", "Pulses and products",
                    "Beef, veal, lamb, and other meat products",
                    "Poultry and its products", pd.NA,
                ],
                "food_id_event_count": [10] * len(food_ids),
                "food_id_weight_g_total": [1000] * len(food_ids),
            }
        )
        hpdi_components = [
            "fruits",
            "fruit_juice",
            "sugar_sweetened_beverages",
            "vegetables",
            "whole_grains",
            "nuts",
            "legumes",
            "meat",
            "meat",
            pd.NA,
        ]
        hpdi = pd.DataFrame(
            {
                "food_id": food_ids,
                "hpdi_component": hpdi_components,
                "mapping_status": ["mapped"] * 9 + ["unmapped_missing_labels"],
            }
        )
        amed = pd.DataFrame(
            {
                "food_id": food_ids,
                "amed_component": [
                    "fruit", "fruit", pd.NA, "vegetables", "whole_grains",
                    "nuts", "legumes", "red_processed_meat", pd.NA, pd.NA,
                ],
                "mapping_status": [
                    "mapped", "mapped", "not_applicable", "mapped", "mapped",
                    "mapped", "mapped", "mapped", "not_applicable",
                    "unmapped_missing_labels",
                ],
            }
        )

        mapping = MAPPER.build_ahei_mapping(dictionary, hpdi, amed).set_index(
            "food_id"
        )

        expected = {
            "fruit": "fruit",
            "juice": "ssb_plus_fruit_juice",
            "ssb": "ssb_plus_fruit_juice",
            "veg": "vegetables",
            "wg": "whole_grains",
            "nuts": "nuts_plus_legumes",
            "legumes": "nuts_plus_legumes",
            "red": "red_plus_processed_meat",
        }
        for food_id, component in expected.items():
            self.assertEqual(mapping.loc[food_id, "ahei_component"], component)
            self.assertEqual(mapping.loc[food_id, "mapping_status"], "mapped")
        self.assertEqual(mapping.loc["chicken", "mapping_status"], "not_applicable")
        self.assertTrue(pd.isna(mapping.loc["chicken", "ahei_component"]))
        self.assertEqual(
            mapping.loc["missing", "mapping_status"],
            "unmapped_missing_labels",
        )
        self.assertNotEqual(mapping.loc["juice", "ahei_component"], "fruit")

    def test_component_definitions_freeze_seven_included_and_three_excluded(self):
        definitions = MAPPER.component_definitions().set_index("ahei_component")
        self.assertEqual(len(definitions), 10)
        self.assertEqual(int(definitions["included_in_final_score"].sum()), 7)
        for component in ["trans_fat", "pufa", "sodium"]:
            self.assertFalse(bool(definitions.loc[component, "included_in_final_score"]))
            self.assertEqual(
                definitions.loc[component, "availability_status"],
                "excluded_by_user_protocol",
            )


if __name__ == "__main__":
    unittest.main()
