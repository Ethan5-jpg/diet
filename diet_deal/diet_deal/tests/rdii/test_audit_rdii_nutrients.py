import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    PROJECT_DIR / "scripts" / "rdii" / "00_audit_rdii_nutrients.py"
)
SPEC = importlib.util.spec_from_file_location(
    "rdii_nutrient_auditor", SCRIPT_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("无法加载测试模块：{}".format(SCRIPT_PATH))
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)


DIRECT_EVENT_COLUMNS = [
    "participant_id",
    "food_id",
    "alcohol_g",
    "carbohydrate_g",
    "calories_kcal",
    "lipid_g",
    "dietary_fiber_g",
    "protein_g",
]

LOOKUP_COLUMNS = {
    "food_id": ["f1"],
    "vitamin_b12_ug": [1.0],
    "vitamin_b6_mg": [1.0],
    "cholesterol_mg": [1.0],
    "iron_mg": [1.0],
    "magnesium_mg": [1.0],
    "mufa_g": [1.0],
    "niacin_mg": [1.0],
    "pufa_g": [1.0],
    "riboflavin_mg": [1.0],
    "saturated_fat_g": [1.0],
    "thiamin_mg": [1.0],
    "vitamin_a_re": [1.0],
    "vitamin_c_mg": [1.0],
    "vitamin_e_mg": [1.0],
    "zinc_mg": [1.0],
}


class RdiiNutrientAuditTests(unittest.TestCase):
    def test_definition_has_exactly_21_unique_components(self):
        components = [
            item["component"] for item in AUDITOR.NUTRIENT_DEFINITIONS
        ]
        self.assertEqual(len(components), 21)
        self.assertEqual(len(set(components)), 21)

    def test_main_event_columns_match_only_the_six_direct_components(self):
        matches = AUDITOR.match_nutrient_columns(DIRECT_EVENT_COLUMNS)
        present = {key for key, columns in matches.items() if columns}
        self.assertEqual(
            present,
            {
                "alcohol",
                "carbohydrate",
                "energy",
                "total_fat",
                "fibre",
                "protein",
            },
        )

    def test_alias_boundaries_do_not_confuse_related_nutrients(self):
        matches = AUDITOR.match_nutrient_columns(
            [
                "vitamin_b12_ug",
                "vitamin_b1_mg",
                "saturated_fat_g",
                "polyunsaturated_fat_g",
                "caffeine_mg",
                "k__Bacteria|f__FAT_WI_3",
            ]
        )
        self.assertEqual(matches["vitamin_b12"], ["vitamin_b12_ug"])
        self.assertEqual(matches["thiamin"], ["vitamin_b1_mg"])
        self.assertEqual(matches["saturated_fat"], ["saturated_fat_g"])
        self.assertEqual(matches["pufa"], ["polyunsaturated_fat_g"])
        self.assertEqual(matches["total_fat"], [])
        self.assertEqual(matches["iron"], [])

    def test_total_fat_accepts_nutrient_fields_but_not_microbiome_taxa(self):
        matches = AUDITOR.match_nutrient_columns(
            [
                "fat_g",
                "fat_grams",
                "fat_content_g",
                "k__Bacteria|f__FAT_WI_3",
                "s__bacterium_RF_744_FAT_WI_3",
            ]
        )
        self.assertEqual(
            matches["total_fat"],
            ["fat_content_g", "fat_g", "fat_grams"],
        )

    def test_zero_file_limit_scans_all_supported_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for number in range(3):
                pd.DataFrame({"food_id": [number]}).to_csv(
                    root / "table_{}.csv".format(number), index=False
                )

            limited, limited_reached = AUDITOR.list_table_paths(
                root, set(), 2
            )
            unlimited, unlimited_reached = AUDITOR.list_table_paths(
                root, set(), 0
            )
            self.assertEqual(len(limited), 2)
            self.assertTrue(limited_reached)
            self.assertEqual(len(unlimited), 3)
            self.assertFalse(unlimited_reached)

    def test_scan_detects_wide_and_long_format_food_lookups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wide = root / "food_composition.csv"
            long_table = root / "nutrient_values.tsv"
            pd.DataFrame(LOOKUP_COLUMNS).to_csv(wide, index=False)
            pd.DataFrame(
                {
                    "food_id": ["f1"],
                    "nutrient_name": ["zinc"],
                    "value": [1.0],
                    "unit": ["mg"],
                }
            ).to_csv(long_table, sep="\t", index=False)

            candidates, scanned, limit_reached = (
                AUDITOR.scan_transfer_headers(root, set(), 0, 0)
            )
            self.assertEqual(scanned, 2)
            self.assertFalse(limit_reached)
            indexed = candidates.set_index("path")
            self.assertEqual(
                int(indexed.loc[str(wide), "matched_rdii_component_count"]),
                15,
            )
            self.assertTrue(
                bool(
                    indexed.loc[
                        str(long_table), "long_format_nutrient_candidate"
                    ]
                )
            )

    def test_small_end_to_end_audit_reports_all_21_as_direct_or_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transfer = root / "Transfer"
            logging = transfer / "diet_logging"
            logging.mkdir(parents=True)
            events = logging / "diet_logging_events.csv"
            raw_events = logging / "raw_diet_logging_events.csv"
            lookup = transfer / "food_composition.csv"
            output = root / "output"

            pd.DataFrame(columns=DIRECT_EVENT_COLUMNS).to_csv(
                events, index=False
            )
            pd.DataFrame(columns=DIRECT_EVENT_COLUMNS).to_csv(
                raw_events, index=False
            )
            pd.DataFrame(LOOKUP_COLUMNS).to_csv(lookup, index=False)

            paths = AUDITOR.audit_rdii_nutrients(
                events_csv=events,
                raw_events_csv=raw_events,
                transfer_dir=transfer,
                output_dir=output,
                file_limit=0,
                progress_every=0,
            )
            self.assertEqual(len(paths), 4)
            self.assertTrue(all(path.is_file() for path in paths))

            availability = pd.read_csv(paths[0]).set_index("rdii_component")
            counts = availability["availability_status"].value_counts()
            self.assertEqual(int(counts["direct_event_column"]), 6)
            self.assertEqual(
                int(counts["food_lookup_candidate_requires_validation"]),
                15,
            )
            self.assertNotIn("unavailable", counts.index)

            qc = pd.read_csv(paths[3]).set_index("metric")["value"]
            self.assertEqual(qc.loc["complete_unlimited_scan"], "True")
            self.assertEqual(qc.loc["all_21_direct_or_candidate"], "True")


if __name__ == "__main__":
    unittest.main()
