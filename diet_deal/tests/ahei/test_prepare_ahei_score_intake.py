import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    PROJECT_DIR / "scripts" / "ahei" / "04_prepare_ahei_score_intake.py"
)
SPEC = importlib.util.spec_from_file_location("ahei_intake_builder", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("无法加载测试模块：{}".format(SCRIPT_PATH))
BUILDER = importlib.util.module_from_spec(SPEC)
sys.modules["ahei_intake_builder"] = BUILDER
SPEC.loader.exec_module(BUILDER)


class AheiIntakeTests(unittest.TestCase):
    def build_mapping(self):
        return pd.DataFrame(
            {
                "food_id": ["fruit", "veg", "wg", "nuts", "red", "ssb", "missing"],
                "ahei_component": [
                    "fruit", "vegetables", "whole_grains", "nuts_plus_legumes",
                    "red_plus_processed_meat", "ssb_plus_fruit_juice", pd.NA,
                ],
                "mapping_status": ["mapped"] * 6 + ["unmapped_missing_labels"],
            }
        )

    def build_events(self):
        rows = [
            (2, "p1", "10k", "00_00_visit", 0, "2026-01-01", "2026-01-01 08:00", "fruit", 100, 140, 0, 0, 0),
            (3, "p1", "10k", "00_00_visit", 0, "2026-01-01", "2026-01-01 09:00", "veg", 100, 60, 0, 0, 0),
            (4, "p1", "10k", "00_00_visit", 0, "2026-01-01", "2026-01-01 10:00", "wg", 30, 100, 0, 0, 0),
            (5, "p1", "10k", "00_00_visit", 0, "2026-01-01", "2026-01-01 11:00", "nuts", 14.175, 100, 0, 0, 0),
            (6, "p1", "10k", "00_00_visit", 0, "2026-01-01", "2026-01-01 12:00", "red", 100, 220, 0, 0, 0),
            (7, "p1", "10k", "00_00_visit", 0, "2026-01-01", "2026-01-01 13:00", "ssb", 100, 70, 0, 0, 0),
            (8, "p1", "10k", "00_00_visit", 1, "2026-01-02", "2026-01-02 08:00", "missing", 50, None, None, None, None),
            (9, "p2", "10k", "00_00_visit", 0, "2026-01-01", "2026-01-01 08:00", "wg", None, None, 10, 2, 3),
        ]
        return pd.DataFrame(rows, columns=BUILDER.REQUIRED_EVENT_COLUMNS)

    def test_preserves_original_days_and_calculates_zero_intake_days(self):
        partial, metrics = BUILDER.aggregate_chunk(
            self.build_events(), self.build_mapping(), "10k", "00_00_visit"
        )
        daily = BUILDER.combine_daily_partials([partial])
        amed = pd.DataFrame(
            {
                "participant_id": ["p1", "p2"],
                "cohort": ["10k", "10k"],
                "research_stage": ["00_00_visit", "00_00_visit"],
                "alcohol_complete": [True, True],
                "mean_daily_alcohol_g": [7.0, 0.0],
                "total_alcohol_unresolved_events": [0, 0],
                "energy_complete": [True, True],
                "mean_daily_energy_kcal": [1800.0, 1600.0],
            }
        )
        participants = BUILDER.build_participant_summary(daily, amed).set_index(
            "participant_id"
        )

        self.assertEqual(metrics["excluded_missing_labels_rows"], 1)
        self.assertEqual(int(participants.loc["p1", "observed_diet_days"]), 2)
        self.assertAlmostEqual(
            float(participants.loc["p1", "mean_daily_fruit_servings"]), 1.0
        )
        self.assertAlmostEqual(
            float(participants.loc["p1", "mean_daily_vegetables_servings"]), 1.0
        )
        self.assertAlmostEqual(
            float(participants.loc["p1", "mean_daily_whole_grain_proxy_g"]), 15.0
        )
        self.assertAlmostEqual(
            float(participants.loc["p1", "mean_daily_nuts_plus_legumes_servings"]),
            0.25,
        )
        self.assertAlmostEqual(
            float(participants.loc["p1", "mean_daily_red_plus_processed_meat_servings"]),
            0.5,
        )
        self.assertAlmostEqual(
            float(participants.loc["p1", "mean_daily_ssb_plus_fruit_juice_servings"]),
            0.5,
        )
        self.assertAlmostEqual(
            float(participants.loc["p1", "mean_daily_alcohol_servings"]), 0.5
        )
        self.assertFalse(bool(participants.loc["p2", "whole_grains_complete"]))

    def test_approved_weight_override_restores_mapped_weight_completeness(self):
        events = self.build_events().loc[lambda frame: frame["participant_id"].eq("p2")]
        overrides = pd.DataFrame(
            {
                "source_event_row": [9],
                "participant_id": ["p2"],
                "cohort": ["10k"],
                "research_stage": ["00_00_visit"],
                "collection_timestamp": ["2026-01-01 08:00"],
                "food_id": ["wg"],
                "resolved_weight_g": [45.0],
            }
        )
        partial, metrics = BUILDER.aggregate_chunk(
            events,
            self.build_mapping(),
            "10k",
            "00_00_visit",
            weight_overrides=overrides,
        )
        daily = BUILDER.combine_daily_partials([partial])
        self.assertEqual(metrics["weight_override_applied_rows"], 1)
        self.assertTrue(bool(daily.loc[0, "whole_grains_complete"]))
        self.assertAlmostEqual(float(daily.loc[0, "whole_grains_g_total"]), 45.0)
        self.assertAlmostEqual(float(daily.loc[0, "resolved_energy_kcal_total"]), 70.0)


if __name__ == "__main__":
    unittest.main()
