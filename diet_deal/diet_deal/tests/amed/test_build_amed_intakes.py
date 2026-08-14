import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_DIR / "scripts" / "amed" / "04_build_amed_intakes.py"
SPEC = importlib.util.spec_from_file_location("amed_intake_builder", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("无法加载测试模块：{}".format(SCRIPT_PATH))
BUILDER = importlib.util.module_from_spec(SPEC)
sys.modules["amed_intake_builder"] = BUILDER
SPEC.loader.exec_module(BUILDER)


class AmedAlcoholRecoveryTests(unittest.TestCase):
    def build_events(self):
        rows = [
            (
                "observed", "10k", "00_00_visit", 0, "2026-01-01",
                "1013942", "Alcoholic Drinks", 100, None, None, None,
                None, 5,
            ),
            (
                "formula", "10k", "00_00_visit", 0, "2026-01-01",
                "formula_food", "Alcoholic Drinks", 100, 70, 0, 0, 0,
                None,
            ),
            (
                "standard", "10k", "00_00_visit", 0, "2026-01-01",
                "1014735", "Alcoholic Drinks", 330, None, None, None,
                None, None,
            ),
            (
                "zero_unknown", "10k", "00_00_visit", 0, "2026-01-01",
                "unknown_food", "Alcoholic Drinks", 100, None, None, None,
                None, None,
            ),
            (
                "zero_weight", "10k", "00_00_visit", 0, "2026-01-01",
                "1012615", "Alcoholic Drinks", -62.5, None, None, None,
                None, 0,
            ),
            (
                "nonalcohol", "10k", "00_00_visit", 0, "2026-01-01",
                "soft_drink", "Drinks", 100, 40, 10, 0, 0, None,
            ),
            (
                "energy_formula", "10k", "00_00_visit", 0, "2026-01-01",
                "formula_energy_food", "Foods", 100, None, 10, 2, 3,
                None,
            ),
        ]
        return pd.DataFrame(rows, columns=BUILDER.REQUIRED_EVENT_COLUMNS)

    def build_mapping(self):
        food_ids = self.build_events()["food_id"].tolist()
        return pd.DataFrame(
            {
                "food_id": pd.Series(food_ids, dtype="string"),
                "amed_component": pd.Series([pd.NA] * len(food_ids), dtype="string"),
                "mapping_status": ["not_applicable"] * len(food_ids),
            }
        )

    def test_reviewed_abv_map_contains_all_32_food_ids(self):
        self.assertEqual(len(BUILDER.STANDARD_ABV_PERCENT_BY_FOOD_ID), 32)
        self.assertEqual(
            BUILDER.STANDARD_ABV_PERCENT_BY_FOOD_ID["1014735"], 4.5
        )
        self.assertEqual(
            BUILDER.STANDARD_ABV_PERCENT_BY_FOOD_ID["1014436"], 40.0
        )

    def test_resolution_order_and_zero_fallback(self):
        partial, _ = BUILDER.aggregate_chunk(
            self.build_events(),
            self.build_mapping(),
            "10k",
            "00_00_visit",
        )
        daily = BUILDER.combine_daily_partials([partial]).set_index(
            "participant_id"
        )

        self.assertAlmostEqual(float(daily.loc["observed", "alcohol_g_total"]), 5.0)
        self.assertAlmostEqual(float(daily.loc["formula", "alcohol_g_total"]), 10.0)
        expected_corona = 330 * 4.5 / 100 * 0.789
        self.assertAlmostEqual(
            float(daily.loc["standard", "alcohol_g_total"]),
            expected_corona,
        )
        self.assertAlmostEqual(
            float(daily.loc["zero_unknown", "alcohol_g_total"]), 0.0
        )
        self.assertAlmostEqual(
            float(daily.loc["zero_weight", "alcohol_g_total"]), 0.0
        )
        self.assertAlmostEqual(
            float(daily.loc["nonalcohol", "alcohol_g_total"]), 0.0
        )
        self.assertAlmostEqual(
            float(
                daily.loc[
                    "energy_formula", "calories_kcal_total_all_foods"
                ]
            ),
            70.0,
        )
        self.assertAlmostEqual(
            float(
                daily.loc["observed", "calories_kcal_total_all_foods"]
            ),
            35.0,
        )
        self.assertEqual(int(daily["energy_observed_event_count"].sum()), 2)
        self.assertEqual(
            int(daily["energy_complete_macro_formula_event_count"].sum()),
            1,
        )
        self.assertEqual(
            int(daily["energy_partial_macro_formula_event_count"].sum()),
            4,
        )
        self.assertTrue(bool(daily["energy_complete"].all()))

        self.assertEqual(int(daily["alcoholic_drink_event_count"].sum()), 5)
        self.assertEqual(int(daily["alcohol_observed_event_count"].sum()), 1)
        self.assertEqual(
            int(daily["alcohol_energy_formula_event_count"].sum()), 1
        )
        self.assertEqual(
            int(daily["alcohol_pre_recovery_unresolved_event_count"].sum()), 3
        )
        self.assertEqual(
            int(daily["alcohol_standard_abv_event_count"].sum()), 1
        )
        self.assertEqual(
            int(daily["alcohol_zero_fallback_event_count"].sum()), 2
        )
        self.assertEqual(int(daily["alcohol_unresolved_event_count"].sum()), 0)
        self.assertTrue(bool(daily["alcohol_complete"].all()))
        self.assertTrue(bool(daily["alcohol_g_event_coverage"].eq(1).all()))

    def test_participant_and_qc_outputs_record_recovery_sources(self):
        partial, metrics = BUILDER.aggregate_chunk(
            self.build_events(),
            self.build_mapping(),
            "10k",
            "00_00_visit",
        )
        daily = BUILDER.combine_daily_partials([partial])
        participants = BUILDER.build_participant_summary(daily)
        qc = BUILDER.build_qc_table(
            daily,
            participants,
            metrics,
            "10k",
            "00_00_visit",
        ).set_index("metric")["value"]

        self.assertEqual(int(participants["alcohol_complete"].sum()), 7)
        self.assertEqual(
            int(participants["total_alcohol_standard_abv_events"].sum()), 1
        )
        self.assertEqual(
            int(participants["total_alcohol_zero_fallback_events"].sum()), 2
        )
        self.assertEqual(int(qc["alcohol_pre_recovery_unresolved_event_count"]), 3)
        self.assertEqual(int(qc["alcohol_standard_abv_event_count"]), 1)
        self.assertEqual(int(qc["alcohol_zero_fallback_event_count"]), 2)
        self.assertEqual(int(qc["alcohol_unresolved_event_count"]), 0)
        self.assertEqual(int(qc["participants_with_incomplete_alcohol"]), 0)
        self.assertEqual(qc["unrecoverable_alcohol_rule"], "set alcohol_g to 0")

    def test_missing_label_events_are_deleted_but_logging_days_are_retained(self):
        events = pd.DataFrame(
            [
                (
                    "mixed", "10k", "00_00_visit", 0, "2026-01-01",
                    "retained_food", "Drinks", 100, 40, 10, 0, 0, None,
                ),
                (
                    "mixed", "10k", "00_00_visit", 0, "2026-01-01",
                    "deleted_food", pd.NA, 50, None, None, None, None, None,
                ),
                (
                    "only_deleted", "10k", "00_00_visit", 0, "2026-01-01",
                    "deleted_food", pd.NA, 75, None, None, None, None, None,
                ),
            ],
            columns=BUILDER.REQUIRED_EVENT_COLUMNS,
        )
        mapping = pd.DataFrame(
            {
                "food_id": pd.Series(
                    ["retained_food", "deleted_food"], dtype="string"
                ),
                "amed_component": pd.Series([pd.NA, pd.NA], dtype="string"),
                "mapping_status": [
                    "not_applicable",
                    "unmapped_missing_labels",
                ],
            }
        )

        partial, metrics = BUILDER.aggregate_chunk(
            events, mapping, "10k", "00_00_visit"
        )
        daily = BUILDER.combine_daily_partials([partial]).set_index(
            "participant_id"
        )

        self.assertEqual(int(daily.loc["mixed", "event_count"]), 1)
        self.assertEqual(
            int(daily.loc["mixed", "excluded_missing_labels_event_count"]), 1
        )
        self.assertAlmostEqual(
            float(daily.loc["mixed", "weight_g_total_all_foods"]), 100.0
        )
        self.assertAlmostEqual(
            float(daily.loc["mixed", "calories_kcal_total_all_foods"]),
            40.0,
        )
        self.assertTrue(bool(daily.loc["mixed", "energy_complete"]))
        self.assertEqual(
            int(daily.loc["mixed", "unmapped_missing_labels_event_count"]), 0
        )
        self.assertEqual(int(daily.loc["only_deleted", "event_count"]), 0)
        self.assertEqual(
            int(
                daily.loc[
                    "only_deleted", "excluded_missing_labels_event_count"
                ]
            ),
            1,
        )
        self.assertAlmostEqual(
            float(daily.loc["only_deleted", "weight_g_total_all_foods"]), 0.0
        )
        self.assertAlmostEqual(
            float(
                daily.loc[
                    "only_deleted", "calories_kcal_total_all_foods"
                ]
            ),
            0.0,
        )
        self.assertTrue(bool(daily.loc["only_deleted", "energy_complete"]))
        self.assertEqual(metrics["excluded_missing_labels_rows"], 2)
        self.assertEqual(metrics["retained_selected_rows"], 1)

        participants = BUILDER.build_participant_summary(
            daily.reset_index()
        ).set_index("participant_id")
        self.assertEqual(
            int(participants.loc["only_deleted", "observed_diet_days"]), 1
        )
        self.assertEqual(
            int(participants.loc["only_deleted", "total_food_events"]), 0
        )
        self.assertAlmostEqual(
            float(
                participants.loc[
                    "mixed", "mean_daily_energy_kcal"
                ]
            ),
            40.0,
        )
        self.assertAlmostEqual(
            float(
                participants.loc[
                    "only_deleted", "mean_daily_energy_kcal"
                ]
            ),
            0.0,
        )
        self.assertEqual(
            int(
                participants.loc[
                    "only_deleted", "total_excluded_missing_labels_events"
                ]
            ),
            1,
        )

        qc = BUILDER.build_qc_table(
            daily.reset_index(),
            participants.reset_index(),
            metrics,
            "10k",
            "00_00_visit",
        ).set_index("metric")["value"]
        self.assertEqual(int(qc["event_count_before_missing_label_exclusion"]), 3)
        self.assertEqual(int(qc["excluded_missing_labels_event_count"]), 2)
        self.assertEqual(int(qc["retained_event_count"]), 1)
        self.assertEqual(
            qc["missing_label_event_rule"],
            "exclude event; retain participant and logging day",
        )


if __name__ == "__main__":
    unittest.main()
