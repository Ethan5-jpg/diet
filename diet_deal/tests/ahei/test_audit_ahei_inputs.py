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
    def test_hpp_ahei_protocol_constants_are_frozen(self):
        self.assertEqual(
            AUDITOR.DEFAULT_POPULATION_CSV,
            AUDITOR.DATA_DIR / "Transfer" / "population" / "population.csv",
        )
        self.assertEqual(
            AUDITOR.SCORE_PROTOCOL_NAME,
            "modified_AHEI_2010_7_component",
        )
        self.assertEqual(AUDITOR.SCORE_COMPONENT_COUNT, 7)
        self.assertEqual(AUDITOR.SCORE_RAW_MIN, 0.0)
        self.assertEqual(AUDITOR.SCORE_RAW_MAX, 70.0)
        self.assertEqual(
            AUDITOR.EXCLUDED_SCORE_COMPONENTS,
            ("trans_fat", "pufa", "sodium"),
        )
        self.assertEqual(
            AUDITOR.HPP_EXPECTED_COUNTS[
                "excluded_missing_label_food_id_count"
            ],
            333,
        )
        self.assertEqual(
            AUDITOR.FULL_MAPPING_MISSING_LABEL_FOOD_ID_REFERENCE_COUNT,
            368,
        )
        self.assertEqual(AUDITOR.STANDARD_DRINK_G, 14.0)
        self.assertEqual(AUDITOR.ALCOHOL_NONDRINKER_SCORE, 2.5)
        self.assertEqual(
            AUDITOR.HPP_KCAL_PER_SERVING,
            {
                "vegetables": 30.0,
                "fruit": 70.0,
                "ssb_plus_fruit_juice": 70.0,
                "red_plus_processed_meat": 220.0,
            },
        )
        self.assertEqual(
            AUDITOR.WHOLE_GRAIN_PROXY_VARIABLE,
            "whole_grain_proxy_g",
        )

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
                ("p1", "10k", "00_00_visit", "2026-01-01", "f1", "Food", "Product", "Category", 100, 500, 100),
                ("p1", "10k", "00_00_visit", "2026-01-02", "f2", "Food", "Product", "Category", None, 400, 0),
                ("p2", "10k", "00_00_visit", "2026-01-01", "f3", "Food", "Product", "Category", 0, 300, None),
                ("p3", "other", "00_00_visit", "2026-01-01", "f4", "Food", "Product", "Category", 100, 500, 100),
            ],
            columns=[
                "participant_id",
                "cohort",
                "research_stage",
                "collection_date",
                "food_id",
                "short_food_name",
                "product_name",
                "food_category",
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
                "alcohol_g_source_complete_count": 2,
                "energy_source_complete_count": 2,
            },
            {"valid_sex_matched_count": 2},
        ).set_index("ahei_component")
        self.assertEqual(
            availability.loc["vegetables", "availability_status"],
            "protocol_defined_mapping_pending",
        )
        self.assertEqual(
            availability.loc["whole_grains", "availability_status"],
            "proxy_protocol_approved_mapping_pending",
        )
        self.assertIn(
            "whole_grain_proxy_g",
            availability.loc["whole_grains", "reason"],
        )
        self.assertIn(
            "明确全谷物",
            availability.loc["whole_grains", "reason"],
        )
        self.assertEqual(
            availability.loc["trans_fat", "availability_status"],
            "excluded_by_user_protocol",
        )
        self.assertEqual(
            availability.loc["pufa", "availability_status"],
            "excluded_by_user_protocol",
        )
        self.assertFalse(
            bool(availability.loc["trans_fat", "included_in_final_score"])
        )
        self.assertFalse(
            bool(availability.loc["pufa", "included_in_final_score"])
        )
        self.assertFalse(
            bool(availability.loc["sodium", "included_in_final_score"])
        )
        self.assertEqual(
            int(availability["included_in_final_score"].sum()),
            7,
        )
        self.assertEqual(
            availability.loc["sodium", "availability_status"],
            "excluded_by_user_protocol",
        )
        self.assertIn("不填补", availability.loc["sodium", "reason"])
        self.assertEqual(
            availability.loc["alcohol", "availability_status"],
            "strictly_available",
        )
        self.assertIn(
            "14",
            availability.loc["alcohol", "reason"],
        )
        self.assertIn(
            "drinks/day",
            availability.loc["alcohol", "reason"],
        )

    def test_missing_label_events_are_excluded_but_index_is_preserved(self):
        chunk = pd.DataFrame(
            [
                ("p1", "10k", "00_00_visit", "2026-01-01", "food_a", "Apple", "Apple", "Fruit", 100, 50, 1),
                ("p2", "10k", "00_00_visit", "2026-01-01", "food_missing", None, "", None, 200, 100, 2),
                ("p2", "10k", "00_00_visit", "2026-01-02", "food_b", "Beans", "Beans", "Legumes", 150, 120, 3),
                ("p3", "10k", "00_00_visit", "2026-01-01", "food_missing", None, None, None, 300, 200, 4),
            ],
            columns=[
                "participant_id",
                "cohort",
                "research_stage",
                "collection_date",
                "food_id",
                "short_food_name",
                "product_name",
                "food_category",
                "weight_g",
                "calories_kcal",
                "sodium_mg",
            ],
        )
        partial, metrics = AUDITOR.aggregate_event_chunk(
            chunk,
            "10k",
            "00_00_visit",
            ["weight_g", "calories_kcal", "sodium_mg"],
        )
        summary = partial.set_index("participant_id")
        self.assertEqual(metrics["excluded_missing_labels_rows"], 2)
        self.assertEqual(metrics["excluded_missing_label_food_ids"], {"food_missing"})
        self.assertEqual(metrics["excluded_missing_label_participant_ids"], {"p2", "p3"})
        self.assertEqual(len(metrics["participant_day_keys"]), 4)
        self.assertEqual(len(metrics["retained_participant_day_keys"]), 2)
        self.assertEqual(int(summary.loc["p3", "event_count"]), 1)
        self.assertEqual(int(summary.loc["p3", "retained_event_count"]), 0)

    def test_generic_long_nutrient_table_is_kept_as_unverified_candidate(self):
        columns = ["food_id", "nutrient_name", "amount", "unit"]
        self.assertTrue(AUDITOR.has_long_format_nutrient_schema(columns))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(
                {
                    "food_id": ["f1"],
                    "nutrient_name": ["example"],
                    "amount": [1.0],
                    "unit": ["g"],
                }
            ).to_csv(root / "long_nutrients.csv", index=False)
            candidates, _, _ = AUDITOR.scan_candidate_sources(
                root, set(), 0
            )
            self.assertEqual(int(candidates["likely_ahei_lookup"].sum()), 1)
            for requirement in ["whole_grain", "trans_fat", "pufa"]:
                self.assertEqual(
                    AUDITOR.candidate_count_for_group(
                        candidates, requirement
                    ),
                    1,
                )

    def test_amed_shared_audit_requires_zero_unresolved_and_complete_energy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "amed.csv"
            pd.DataFrame(
                {
                    "participant_id": ["p1", "p2"],
                    "cohort": ["10k", "10k"],
                    "research_stage": ["00_00_visit", "00_00_visit"],
                    "alcohol_complete": [True, True],
                    "mean_daily_alcohol_g": [0.0, 7.0],
                    "total_alcohol_unresolved_events": [0, 1],
                    "energy_complete": [True, False],
                    "mean_daily_energy_kcal": [1800.0, 1600.0],
                }
            ).to_csv(path, index=False)
            metrics = AUDITOR.audit_amed_shared_inputs(
                path, {"p1", "p2"}, "10k", "00_00_visit"
            )
            self.assertEqual(metrics["alcohol_complete_participant_count"], 2)
            self.assertEqual(metrics["alcohol_g_source_complete_count"], 1)
            self.assertEqual(metrics["energy_complete_participant_count"], 1)
            self.assertEqual(metrics["amed_shared_input_complete_count"], 1)
            self.assertEqual(metrics["standard_drink_g"], 14.0)
            self.assertEqual(
                metrics["standard_drink_g_parameter_status"],
                "verified_and_frozen_hpp",
            )
            self.assertEqual(
                metrics["drinks_per_day_formula"],
                "mean_daily_alcohol_g / 14.0",
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

            candidate_path, qc_path, summary_path = LOOKUP_SCANNER.scan_lookup_sources(
                transfer_dir=root,
                output_dir=output,
                file_limit=0,
                progress_every=0,
            )
            self.assertTrue(candidate_path.is_file())
            self.assertTrue(qc_path.is_file())
            candidates = pd.read_csv(candidate_path)
            qc = pd.read_csv(qc_path).set_index("metric")["value"]
            summary = pd.read_csv(summary_path).set_index("requirement")
            self.assertEqual(int(candidates["likely_ahei_lookup"].sum()), 1)
            self.assertEqual(qc.loc["scan_complete_without_limit"], "True")
            self.assertEqual(
                int(summary.loc["whole_grain", "candidate_table_count"]), 1
            )
            self.assertEqual(
                summary.loc["whole_grain", "formal_ahei_status"],
                "optional_strict_source_proxy_protocol_approved",
            )
            for requirement in ["trans_fat", "pufa"]:
                self.assertEqual(
                    summary.loc[requirement, "formal_ahei_status"],
                    "not_required_excluded_by_user_protocol",
                )

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
                    ("p1", "10k", "00_00_visit", "2026-01-01", "f1", "Apple", "Apple", "Fruit", 100, 500, 100),
                    ("p2", "10k", "00_00_visit", "2026-01-01", "f_missing", None, None, None, 200, 600, 200),
                    ("p2", "10k", "00_00_visit", "2026-01-02", "f2", "Beans", "Beans", "Legumes", 250, 650, 250),
                    ("p3", "10k", "00_00_visit", "2026-01-01", "f_missing", None, "", None, 300, 700, 300),
                ],
                columns=[
                    "participant_id",
                    "cohort",
                    "research_stage",
                    "collection_date",
                    "food_id",
                    "short_food_name",
                    "product_name",
                    "food_category",
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
                    "alcohol_complete": [True, True, True],
                    "mean_daily_alcohol_g": [0.0, 14.0, 7.0],
                    "total_alcohol_unresolved_events": [0, 0, 0],
                    "energy_complete": [True, True, True],
                    "mean_daily_energy_kcal": [1800.0, 2000.0, 1600.0],
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
                enforce_expected_counts=False,
            )
            for path in paths:
                self.assertTrue(path.is_file())
            availability = pd.read_csv(paths[0])
            self.assertEqual(len(availability), 10)
            included = availability["included_in_final_score"].astype(bool)
            self.assertEqual(int(included.sum()), 7)
            sodium_row = availability.loc[
                availability["ahei_component"].eq("sodium")
            ].iloc[0]
            self.assertEqual(
                sodium_row["availability_status"],
                "excluded_by_user_protocol",
            )
            self.assertFalse(bool(sodium_row["included_in_final_score"]))
            alcohol_row = availability.loc[
                availability["ahei_component"].eq("alcohol")
            ].iloc[0]
            self.assertEqual(
                int(alcohol_row["current_eligible_participant_count"]), 3
            )
            self.assertEqual(
                alcohol_row["availability_status"], "strictly_available"
            )
            qc = pd.read_csv(paths[4])
            qc_values = qc.set_index(["section", "metric"])["value"]
            self.assertEqual(
                qc_values.loc[("protocol", "whole_grain_proxy_variable")],
                "whole_grain_proxy_g",
            )
            self.assertEqual(
                float(qc_values.loc[("protocol", "vegetables_kcal_per_serving")]),
                30.0,
            )
            self.assertEqual(
                qc_values.loc[("protocol", "score_protocol_name")],
                "modified_AHEI_2010_7_component",
            )
            self.assertEqual(
                float(qc_values.loc[("protocol", "score_raw_max")]),
                70.0,
            )
            self.assertEqual(
                int(
                    qc_values.loc[
                        (
                            "protocol",
                            "full_mapping_missing_label_food_id_reference_count",
                        )
                    ]
                ),
                368,
            )
            self.assertEqual(
                int(qc_values.loc[("events", "excluded_missing_label_food_id_count")]),
                1,
            )
            self.assertEqual(
                int(qc_values.loc[("events", "excluded_missing_labels_rows")]),
                2,
            )
            self.assertEqual(
                int(qc_values.loc[("events", "retained_selected_rows")]),
                2,
            )
            self.assertEqual(
                int(qc_values.loc[("events", "participant_day_count_preserved")]),
                4,
            )
            self.assertEqual(
                int(qc_values.loc[("events", "participant_days_with_only_excluded_events")]),
                2,
            )
            self.assertEqual(
                int(qc_values.loc[("amed_shared", "energy_complete_participant_count")]),
                3,
            )


if __name__ == "__main__":
    unittest.main()
