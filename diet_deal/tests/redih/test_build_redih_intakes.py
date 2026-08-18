import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_DIR / "scripts" / "redih" / "04_build_redih_intakes.py"
SPEC = importlib.util.spec_from_file_location("redih_intakes", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("无法加载测试模块：{}".format(SCRIPT_PATH))
INTAKES = importlib.util.module_from_spec(SPEC)
sys.modules["redih_intakes"] = INTAKES
SPEC.loader.exec_module(INTAKES)


def event(participant_id, food_id, weight_g, calories_kcal, timestamp):
    return {
        "participant_id": participant_id,
        "cohort": "10k",
        "research_stage": "00_00_visit",
        "logging_day": 0,
        "collection_date": "2026-01-01",
        "collection_timestamp": timestamp,
        "food_id": food_id,
        "weight_g": weight_g,
        "calories_kcal": calories_kcal,
    }


class RedihIntakeMissingDataTests(unittest.TestCase):
    def setUp(self):
        self.mapping = pd.DataFrame(
            [
                {
                    "food_id": "wine",
                    "redih_component": "wine",
                    "mapping_status": "mapped",
                },
                {
                    "food_id": "missing_labels",
                    "redih_component": pd.NA,
                    "mapping_status": "unmapped_missing_labels",
                },
            ]
        )

    def test_missing_label_events_are_deleted_but_participant_day_is_retained(self):
        events = pd.DataFrame(
            [
                event("mixed", "wine", 100.0, 200.0, "2026-01-01 08:00:00+00:00"),
                event(
                    "mixed",
                    "missing_labels",
                    50.0,
                    100.0,
                    "2026-01-01 09:00:00+00:00",
                ),
                event(
                    "only_deleted",
                    "missing_labels",
                    80.0,
                    120.0,
                    "2026-01-01 10:00:00+00:00",
                ),
            ]
        )

        daily, participants, qc = INTAKES.build_intakes(
            events, self.mapping, "10k", "00_00_visit"
        )
        daily = daily.set_index("participant_id")
        participants = participants.set_index("participant_id")
        qc_values = dict(zip(qc["metric"], qc["value"]))

        self.assertEqual(int(daily.loc["mixed", "event_count"]), 1)
        self.assertEqual(
            int(daily.loc["mixed", "excluded_missing_labels_event_count"]), 1
        )
        self.assertEqual(float(daily.loc["mixed", "weight_g_total_all_foods"]), 100.0)
        self.assertEqual(float(daily.loc["mixed", "calories_kcal_total"]), 200.0)
        self.assertEqual(int(daily.loc["only_deleted", "event_count"]), 0)
        self.assertEqual(
            int(daily.loc["only_deleted", "excluded_missing_labels_event_count"]), 1
        )
        self.assertEqual(float(daily.loc["only_deleted", "wine_g_total"]), 0.0)
        self.assertEqual(int(participants.loc["only_deleted", "observed_diet_days"]), 1)
        self.assertEqual(int(qc_values["excluded_missing_labels_event_count"]), 2)
        self.assertEqual(int(qc_values["retained_event_count"]), 1)

    def test_approved_weight_is_applied_before_component_aggregation(self):
        events = pd.DataFrame(
            [
                event(
                    "target",
                    "wine",
                    pd.NA,
                    180.0,
                    "2026-01-01 08:00:00+00:00",
                )
            ]
        )
        overrides = pd.DataFrame(
            [
                {
                    "source_event_row": 2,
                    "participant_id": "target",
                    "cohort": "10k",
                    "research_stage": "00_00_visit",
                    "collection_timestamp": "2026-01-01 08:00:00+00:00",
                    "food_id": "wine",
                    "resolved_weight_g": 120.0,
                }
            ]
        )

        daily, participants, qc = INTAKES.build_intakes(
            events,
            self.mapping,
            "10k",
            "00_00_visit",
            weight_overrides=overrides,
        )

        self.assertEqual(float(daily.loc[0, "wine_g_total"]), 120.0)
        self.assertEqual(int(daily.loc[0, "weight_imputed_event_count"]), 1)
        self.assertEqual(int(daily.loc[0, "wine_missing_weight_event_count"]), 0)
        self.assertEqual(int(participants.loc[0, "total_weight_imputed_events"]), 1)
        self.assertTrue(bool(participants.loc[0, "redih_any_weight_imputed"]))
        self.assertTrue(
            bool(participants.loc[0, "redih_all_current_score_component_weights_complete"])
        )


if __name__ == "__main__":
    unittest.main()
