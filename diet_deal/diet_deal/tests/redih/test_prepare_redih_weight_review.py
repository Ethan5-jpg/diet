import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    PROJECT_DIR / "scripts" / "redih" / "04a_prepare_redih_weight_review.py"
)
SPEC = importlib.util.spec_from_file_location("redih_weight_review", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("无法加载测试模块：{}".format(SCRIPT_PATH))
REVIEW = importlib.util.module_from_spec(SPEC)
sys.modules["redih_weight_review"] = REVIEW
SPEC.loader.exec_module(REVIEW)


class RedihWeightReviewTests(unittest.TestCase):
    def test_target_participant_is_excluded_and_donors_are_equally_weighted(self):
        donors = pd.DataFrame(
            [
                {"participant_id": "target", "food_id": "wine", "weight_g": 1000.0},
                *[
                    {"participant_id": "frequent", "food_id": "wine", "weight_g": 50.0}
                    for _ in range(5)
                ],
                {"participant_id": "occasional", "food_id": "wine", "weight_g": 150.0},
            ]
        )
        missing = pd.DataFrame(
            [{"participant_id": "target", "food_id": "wine"}]
        )

        donor_summary = REVIEW.build_participant_food_donors(donors)
        review = REVIEW.add_other_participant_median_imputations(
            missing, donor_summary
        )

        self.assertAlmostEqual(float(review.loc[0, "resolved_weight_g"]), 100.0)
        self.assertEqual(int(review.loc[0, "other_participant_donor_count"]), 2)
        self.assertEqual(int(review.loc[0, "other_event_donor_count"]), 6)
        self.assertEqual(
            review.loc[0, "resolution_status"], "use_resolved_weight"
        )

    def test_no_other_participant_donor_remains_unresolved(self):
        donors = pd.DataFrame(
            [{"participant_id": "target", "food_id": "wine", "weight_g": 80.0}]
        )
        missing = pd.DataFrame(
            [{"participant_id": "target", "food_id": "wine"}]
        )

        donor_summary = REVIEW.build_participant_food_donors(donors)
        review = REVIEW.add_other_participant_median_imputations(
            missing, donor_summary
        )

        self.assertEqual(review.loc[0, "resolution_status"], "unresolved")
        self.assertTrue(pd.isna(review.loc[0, "resolved_weight_g"]))


if __name__ == "__main__":
    unittest.main()
