import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    PROJECT_DIR / "scripts" / "hpdi" / "04a_prepare_hpdi_weight_review.py"
)
SPEC = importlib.util.spec_from_file_location("hpdi_weight_review", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("无法加载测试模块：{}".format(SCRIPT_PATH))
REVIEW = importlib.util.module_from_spec(SPEC)
sys.modules["hpdi_weight_review"] = REVIEW
SPEC.loader.exec_module(REVIEW)


class HpdiWeightSuggestionTests(unittest.TestCase):
    def test_regeneration_preserves_existing_manual_resolution(self):
        event_key = {
            "source_event_row": 2,
            "participant_id": "participant",
            "cohort": "10k",
            "research_stage": "00_00_visit",
            "collection_timestamp": "2026-01-01 08:00:00+00:00",
            "food_id": "food_a",
        }
        review = pd.DataFrame(
            [
                {
                    **event_key,
                    "suggested_weight_g": 120.0,
                    "resolution_status": "pending",
                    "resolved_weight_g": pd.NA,
                    "review_notes": "",
                }
            ]
        )
        existing = pd.DataFrame(
            [
                {
                    **event_key,
                    "resolution_status": "use_resolved_weight",
                    "resolved_weight_g": 115.0,
                    "review_notes": "confirmed from source record",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "review.csv"
            existing.to_csv(path, index=False, encoding="utf-8-sig")
            regenerated = REVIEW.preserve_existing_resolutions(review, path)

        self.assertEqual(
            regenerated.loc[0, "resolution_status"], "use_resolved_weight"
        )
        self.assertAlmostEqual(
            float(regenerated.loc[0, "resolved_weight_g"]), 115.0
        )
        self.assertEqual(
            regenerated.loc[0, "review_notes"], "confirmed from source record"
        )

    def test_other_participant_median_excludes_target_participant(self):
        donors = pd.DataFrame(
            [
                {"participant_id": "target", "food_id": "food_a", "weight_g": 999.0},
                {"participant_id": "donor_1", "food_id": "food_a", "weight_g": 120.0},
                {"participant_id": "donor_2", "food_id": "food_a", "weight_g": 140.0},
            ]
        )
        missing = pd.DataFrame(
            [{"participant_id": "target", "food_id": "food_a"}]
        )

        donor_summary = REVIEW.build_participant_food_donors(donors)
        review = REVIEW.add_other_participant_median_imputations(
            missing, donor_summary
        )

        self.assertAlmostEqual(float(review.loc[0, "suggested_weight_g"]), 130.0)
        self.assertEqual(int(review.loc[0, "other_participant_donor_count"]), 2)
        self.assertEqual(int(review.loc[0, "other_event_donor_count"]), 2)
        self.assertEqual(
            review.loc[0, "resolution_status"], "use_resolved_weight"
        )
        self.assertAlmostEqual(float(review.loc[0, "resolved_weight_g"]), 130.0)

    def test_each_other_participant_contributes_one_median(self):
        donors = pd.DataFrame(
            [
                *[
                    {"participant_id": "frequent", "food_id": "food_a", "weight_g": 50.0}
                    for _ in range(5)
                ],
                {"participant_id": "occasional", "food_id": "food_a", "weight_g": 150.0},
            ]
        )
        missing = pd.DataFrame(
            [{"participant_id": "target", "food_id": "food_a"}]
        )

        donor_summary = REVIEW.build_participant_food_donors(donors)
        review = REVIEW.add_other_participant_median_imputations(
            missing, donor_summary
        )

        self.assertAlmostEqual(float(review.loc[0, "suggested_weight_g"]), 100.0)
        self.assertEqual(int(review.loc[0, "other_participant_donor_count"]), 2)
        self.assertEqual(int(review.loc[0, "other_event_donor_count"]), 6)

    def test_no_other_participant_donor_remains_unresolved(self):
        donors = pd.DataFrame(
            [
                {"participant_id": "target", "food_id": "food_a", "weight_g": 80.0},
                {"participant_id": "other", "food_id": "food_b", "weight_g": 120.0},
            ]
        )
        missing = pd.DataFrame(
            [{"participant_id": "target", "food_id": "food_a"}]
        )

        donor_summary = REVIEW.build_participant_food_donors(donors)
        review = REVIEW.add_other_participant_median_imputations(
            missing, donor_summary
        )

        self.assertTrue(pd.isna(review.loc[0, "suggested_weight_g"]))
        self.assertEqual(review.loc[0, "resolution_status"], "unresolved")
        self.assertTrue(pd.isna(review.loc[0, "resolved_weight_g"]))
        self.assertEqual(
            review.loc[0, "imputation_method"], "no_other_participant_donor"
        )

    def test_old_unresolved_status_does_not_block_selected_imputation_rule(self):
        event_key = {
            "source_event_row": 2,
            "participant_id": "participant",
            "cohort": "10k",
            "research_stage": "00_00_visit",
            "collection_timestamp": "2026-01-01 08:00:00+00:00",
            "food_id": "food_a",
        }
        review = pd.DataFrame(
            [
                {
                    **event_key,
                    "resolution_status": "use_resolved_weight",
                    "resolved_weight_g": 130.0,
                    "review_notes": "automatic other-participant median",
                }
            ]
        )
        existing = pd.DataFrame(
            [
                {
                    **event_key,
                    "resolution_status": "unresolved",
                    "resolved_weight_g": pd.NA,
                    "review_notes": "old nutrient-density review",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "review.csv"
            existing.to_csv(path, index=False, encoding="utf-8-sig")
            regenerated = REVIEW.preserve_existing_resolutions(review, path)

        self.assertEqual(
            regenerated.loc[0, "resolution_status"], "use_resolved_weight"
        )
        self.assertAlmostEqual(
            float(regenerated.loc[0, "resolved_weight_g"]), 130.0
        )


if __name__ == "__main__":
    unittest.main()
