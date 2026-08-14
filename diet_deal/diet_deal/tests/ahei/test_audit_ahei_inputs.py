import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_DIR / "scripts" / "ahei" / "00_audit_ahei_inputs.py"
SPEC = importlib.util.spec_from_file_location("ahei_input_auditor", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("无法加载测试模块：{}".format(SCRIPT_PATH))
AUDITOR = importlib.util.module_from_spec(SPEC)
sys.modules["ahei_input_auditor"] = AUDITOR
SPEC.loader.exec_module(AUDITOR)

LOOKUP_SCRIPT_PATH = (
    PROJECT_DIR / "scripts" / "ahei" / "00b_scan_ahei_lookup_sources.py"
)
LOOKUP_SPEC = importlib.util.spec_from_file_location(
    "ahei_lookup_scanner", LOOKUP_SCRIPT_PATH
)
if LOOKUP_SPEC is None or LOOKUP_SPEC.loader is None:
    raise RuntimeError("无法加载测试模块：{}".format(LOOKUP_SCRIPT_PATH))
LOOKUP_SCANNER = importlib.util.module_from_spec(LOOKUP_SPEC)
sys.modules["ahei_lookup_scanner"] = LOOKUP_SCANNER
LOOKUP_SPEC.loader.exec_module(LOOKUP_SCANNER)


class AheiInputAuditTests(unittest.TestCase):
    def test_keyword_detection_covers_required_field_families(self):
        columns = [
            "portion_size_g",
            "trans_fat_g",
            "polyunsaturated_fat_g",
            "whole_grain_g",
            "nutrition_source",
        ]
        hits = AUDITOR.find_keyword_hits(columns)
        groups = {hit["keyword_group"] for hit in hits}
        self.assertEqual(
            groups,
            {
                "serving_or_portion",
                "trans_fat",
                "pufa",
                "whole_grain",
                "nutrient",
            },
        )

    def test_boolean_parser_preserves_missing_and_rejects_unknown(self):
        values = pd.Series(["true", "0", "YES", "", None])
        parsed = AUDITOR.parse_boolean(values, "flag")
        self.assertTrue(bool(parsed.iloc[0]))
        self.assertFalse(bool(parsed.iloc[1]))
        self.assertTrue(bool(parsed.iloc[2]))
        self.assertTrue(pd.isna(parsed.iloc[3]))
        self.assertTrue(pd.isna(parsed.iloc[4]))
        with self.assertRaises(ValueError):
            AUDITOR.parse_boolean(pd.Series(["unknown"]), "flag")

    def test_event_coverage_uses_strict_all_event_completeness(self):
        chunk = pd.DataFrame(
            [
                ("p1", "10k", "00_00_visit", 100, 500, 100),
                ("p1", "10k", "00_00_visit", None, 400, 0),
                ("p2", "10k", "00_00_visit", 0, 300, None),
                ("p3", "other", "00_00_visit", 100, 500, 100),
            ],
            columns=[
                "participant_id",
                "cohort",
                "research_stage",
                "weight_g",
                "calories_kcal",
                "sodium_mg",
            ],
        )
        fields = ["weight_g", "calories_kcal", "sodium_mg"]
        partial, metrics = AUDITOR.aggregate_event_chunk(
            chunk, "10k", "00_00_visit", fields
        )
        coverage = AUDITOR.build_event_coverage(partial, fields).set_index("field")
        self.assertEqual(metrics["selected_rows"], 3)
        self.assertEqual(int(coverage.loc["weight_g", "valid_event_count"]), 1)
        self.assertEqual(
            int(coverage.loc["weight_g", "strictly_complete_participant_count"]),
            0,
        )
        self.assertEqual(
            int(coverage.loc["calories_kcal", "strictly_complete_participant_count"]),
            2,
        )
        self.assertEqual(int(coverage.loc["sodium_mg", "valid_event_count"]), 2)
        self.assertEqual(
            int(coverage.loc["sodium_mg", "strictly_complete_participant_count"]),
            1,
        )

    def test_default_component_classification_does_not_invent_fatty_acids(self):
        main_columns = [
            "participant_id",
            "food_id",
            "weight_g",
            "calories_kcal",
            "sodium_mg",
        ]
        availability = AUDITOR.build_component_availability(
            main_columns,
            list(main_columns),
            pd.DataFrame(columns=["has_food_id", "matched_keyword_groups"]),
            {
                "target_participant_count": 2,
                "current_official_ahei_eligible_count": 1,
            },
            {"valid_sex_matched_count": 2},
        ).set_index("ahei_component")
        self.assertEqual(
            availability.loc["vegetables", "availability_status"],
            "proxy_only",
        )
        self.assertEqual(
            availability.loc["whole_grains", "availability_status"],
            "proxy_only",
        )
        self.assertEqual(
            availability.loc["trans_fat", "availability_status"],
            "unavailable",
        )
        self.assertEqual(
            availability.loc["pufa", "availability_status"],
            "unavailable",
        )
        self.assertEqual(
            availability.loc["sodium", "availability_status"],
            "strictly_available",
        )
        self.assertEqual(
            availability.loc["alcohol", "availability_status"],
            "strictly_available",
        )

    def test_candidate_scan_finds_food_id_nutrient_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "food_nutrition_lookup.csv"
            pd.DataFrame(
                {
                    "food_id": ["1"],
                    "trans_fat_g": [0.1],
                    "pufa_g": [1.0],
                }
            ).to_csv(candidate, index=False)
            pd.DataFrame({"participant_id": ["p1"]}).to_csv(
                root / "ordinary.csv", index=False
            )

            candidates, scanned, limit_reached = AUDITOR.scan_candidate_sources(
                root, set(), 100
            )
            self.assertEqual(scanned, 2)
            self.assertFalse(limit_reached)
            likely = candidates.loc[candidates["likely_ahei_lookup"].eq(True)]
            self.assertEqual(len(likely), 1)
            self.assertEqual(Path(likely.iloc[0]["path"]), candidate)

    def test_candidate_scan_zero_limit_means_unlimited(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for number in range(3):
                pd.DataFrame({"value": [number]}).to_csv(
                    root / "table_{}.csv".format(number), index=False
                )

            _, limited_count, limited_reached = AUDITOR.scan_candidate_sources(
                root, set(), 2
            )
            _, unlimited_count, unlimited_reached = AUDITOR.scan_candidate_sources(
                root, set(), 0
            )
            self.assertEqual(limited_count, 2)
            self.assertTrue(limited_reached)
            self.assertEqual(unlimited_count, 3)
            self.assertFalse(unlimited_reached)

    def test_lookup_only_scanner_writes_complete_scan_qc(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            pd.DataFrame(
                {
                    "food_id": ["f1"],
                    "whole_grain_g": [10.0],
                    "pufa_g": [1.0],
                }
            ).to_csv(root / "food_composition.csv", index=False)

            candidate_path, qc_path = LOOKUP_SCANNER.scan_lookup_sources(
                transfer_dir=root,
                output_dir=output,
                file_limit=0,
                progress_every=0,
            )
            self.assertTrue(candidate_path.is_file())
            self.assertTrue(qc_path.is_file())
            candidates = pd.read_csv(candidate_path)
            qc = pd.read_csv(qc_path).set_index("metric")["value"]
            self.assertEqual(int(candidates["likely_ahei_lookup"].sum()), 1)
            self.assertEqual(qc.loc["scan_complete_without_limit"], "True")

    def test_small_end_to_end_audit_writes_all_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = root / "diet_logging_events.csv"
            raw_events = root / "raw_diet_logging_events.csv"
            population = root / "population.csv"
            alcohol = root / "amed_participant_component_intakes.csv"
            output = root / "output"

            event_data = pd.DataFrame(
                [
                    ("p1", "10k", "00_00_visit", 100, 500, 100),
                    ("p2", "10k", "00_00_visit", 200, 600, 200),
                    ("p3", "10k", "00_00_visit", 300, 700, 300),
                ],
                columns=[
                    "participant_id",
                    "cohort",
                    "research_stage",
                    "weight_g",
                    "calories_kcal",
                    "sodium_mg",
                ],
            )
            event_data.to_csv(events, index=False)
            event_data.assign(portion_size_g=100).to_csv(raw_events, index=False)
            pd.DataFrame(
                {
                    "participant_id": ["p1", "p2", "p3"],
                    "cohort": ["10k", "10k", "10k"],
                    "sex": [0, 1, 0],
                }
            ).to_csv(population, index=False)
            pd.DataFrame(
                {
                    "participant_id": ["p1", "p2", "p3"],
                    "cohort": ["10k", "10k", "10k"],
                    "research_stage": ["00_00_visit"] * 3,
                    "alcohol_complete": [True, True, False],
                    "mean_daily_alcohol_g": [0.0, 14.0, None],
                }
            ).to_csv(alcohol, index=False)
            pd.DataFrame(
                {
                    "food_id": ["f1"],
                    "trans_fat_g": [0.1],
                    "pufa_g": [1.0],
                }
            ).to_csv(root / "food_nutrient_lookup.csv", index=False)

            paths = AUDITOR.audit_ahei_inputs(
                events_csv=events,
                raw_events_csv=raw_events,
                population_csv=population,
                alcohol_csv=alcohol,
                transfer_dir=root,
                output_dir=output,
                chunk_size=2,
                candidate_file_limit=100,
                cohort="10k",
                research_stage="00_00_visit",
            )
            for path in paths:
                self.assertTrue(path.is_file())
            availability = pd.read_csv(paths[0])
            self.assertEqual(len(availability), 10)
            alcohol_row = availability.loc[
                availability["ahei_component"].eq("alcohol")
            ].iloc[0]
            self.assertEqual(
                int(alcohol_row["current_eligible_participant_count"]), 2
            )


if __name__ == "__main__":
    unittest.main()
