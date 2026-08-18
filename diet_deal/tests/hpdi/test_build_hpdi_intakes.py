import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_DIR / "scripts" / "hpdi" / "04_build_hpdi_intakes.py"
SPEC = importlib.util.spec_from_file_location("hpdi_intake_builder", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("无法加载测试模块：{}".format(SCRIPT_PATH))
BUILDER = importlib.util.module_from_spec(SPEC)
sys.modules["hpdi_intake_builder"] = BUILDER
SPEC.loader.exec_module(BUILDER)


class HpdiMissingLabelExclusionTests(unittest.TestCase):
    def test_weight_override_loader_applies_only_approved_rows(self):
        overrides = pd.DataFrame(
            {
                "source_event_row": [2, 3, 4],
                "participant_id": ["approved", "pending", "unresolved"],
                "cohort": ["10k"] * 3,
                "research_stage": ["00_00_visit"] * 3,
                "collection_timestamp": [
                    "2026-01-01 08:00:00+00:00",
                    "2026-01-01 09:00:00+00:00",
                    "2026-01-01 10:00:00+00:00",
                ],
                "food_id": ["food_a", "food_b", "food_c"],
                "resolution_status": [
                    "use_resolved_weight",
                    "pending",
                    "unresolved",
                ],
                "resolved_weight_g": [125.0, pd.NA, pd.NA],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "review.csv"
            overrides.to_csv(path, index=False, encoding="utf-8-sig")
            loaded = BUILDER.load_weight_overrides(path)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded.loc[0, "participant_id"], "approved")
        self.assertAlmostEqual(float(loaded.loc[0, "resolved_weight_g"]), 125.0)

    def test_missing_label_events_are_deleted_but_logging_days_are_retained(self):
        events = pd.DataFrame(
            [
                (
                    "mixed",
                    "10k",
                    "00_00_visit",
                    0,
                    "2026-01-01",
                    "2026-01-01 08:00:00+00:00",
                    "retained_food",
                    100,
                    40,
                ),
                (
                    "mixed",
                    "10k",
                    "00_00_visit",
                    0,
                    "2026-01-01",
                    "2026-01-01 09:00:00+00:00",
                    "deleted_food",
                    50,
                    25,
                ),
                (
                    "only_deleted",
                    "10k",
                    "00_00_visit",
                    0,
                    "2026-01-01",
                    "2026-01-01 10:00:00+00:00",
                    "deleted_food",
                    75,
                    30,
                ),
            ],
            columns=BUILDER.REQUIRED_EVENT_COLUMNS,
        )
        mapping = pd.DataFrame(
            {
                "food_id": pd.Series(
                    ["retained_food", "deleted_food"], dtype="string"
                ),
                "hpdi_component": pd.Series(
                    ["fruits", pd.NA], dtype="string"
                ),
                "mapping_status": ["mapped", "unmapped_missing_labels"],
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
            int(daily.loc["mixed", "excluded_missing_labels_event_count"]),
            1,
        )
        self.assertAlmostEqual(
            float(daily.loc["mixed", "weight_g_total_all_foods"]), 100.0
        )
        self.assertAlmostEqual(
            float(daily.loc["mixed", "calories_kcal_total"]), 40.0
        )
        self.assertAlmostEqual(
            float(daily.loc["mixed", "fruits_g_total"]), 100.0
        )
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
            float(daily.loc["only_deleted", "weight_g_total_all_foods"]),
            0.0,
        )
        self.assertAlmostEqual(
            float(daily.loc["only_deleted", "calories_kcal_total"]), 0.0
        )
        self.assertAlmostEqual(
            float(daily.loc["only_deleted", "fruits_g_total"]), 0.0
        )
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
        self.assertEqual(
            int(
                participants.loc[
                    "only_deleted", "total_excluded_missing_labels_events"
                ]
            ),
            1,
        )
        self.assertAlmostEqual(
            float(
                participants.loc[
                    "only_deleted", "mean_daily_calories_kcal_proxy"
                ]
            ),
            0.0,
        )
        self.assertAlmostEqual(
            float(
                participants.loc[
                    "only_deleted", "mean_daily_fruits_g_proxy"
                ]
            ),
            0.0,
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
        self.assertEqual(int(qc["unmapped_missing_labels_event_count"]), 0)
        self.assertEqual(
            qc["missing_label_event_rule"],
            "exclude event; retain participant and logging day",
        )

    def test_approved_weight_override_replaces_missing_weight(self):
        events = pd.DataFrame(
            [
                (
                    "participant",
                    "10k",
                    "00_00_visit",
                    0,
                    "2026-01-01",
                    "2026-01-01 08:00:00+00:00",
                    "fruit_food",
                    pd.NA,
                    80,
                )
            ],
            columns=BUILDER.REQUIRED_EVENT_COLUMNS,
        )
        mapping = pd.DataFrame(
            {
                "food_id": pd.Series(["fruit_food"], dtype="string"),
                "hpdi_component": pd.Series(["fruits"], dtype="string"),
                "mapping_status": ["mapped"],
            }
        )
        weight_overrides = pd.DataFrame(
            {
                "source_event_row": [2],
                "participant_id": pd.Series(["participant"], dtype="string"),
                "cohort": pd.Series(["10k"], dtype="string"),
                "research_stage": pd.Series(["00_00_visit"], dtype="string"),
                "collection_timestamp": pd.Series(
                    ["2026-01-01 08:00:00+00:00"], dtype="string"
                ),
                "food_id": pd.Series(["fruit_food"], dtype="string"),
                "resolved_weight_g": [120.0],
            }
        )

        partial, metrics = BUILDER.aggregate_chunk(
            events,
            mapping,
            "10k",
            "00_00_visit",
            weight_overrides,
        )
        daily = BUILDER.combine_daily_partials([partial])

        self.assertAlmostEqual(float(daily.loc[0, "fruits_g_total"]), 120.0)
        self.assertEqual(int(daily.loc[0, "weight_imputed_event_count"]), 1)
        self.assertEqual(int(daily.loc[0, "fruits_missing_weight_event_count"]), 0)
        self.assertTrue(bool(daily.loc[0, "fruits_weight_complete"]))
        self.assertEqual(metrics["missing_weight_before_override_rows"], 1)
        self.assertEqual(metrics["weight_override_applied_rows"], 1)
        self.assertEqual(metrics["missing_weight_after_override_rows"], 0)

        participant = BUILDER.build_participant_summary(daily).iloc[0]
        self.assertEqual(int(participant["total_weight_imputed_events"]), 1)
        self.assertTrue(bool(participant["hpdi_any_weight_imputed"]))


if __name__ == "__main__":
    unittest.main()
