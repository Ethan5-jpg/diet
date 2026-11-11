"""Synthetic tests for classification, linkage and overlap."""

import unittest

import pandas as pd

from processing import coverage, nearby_pair_count, reconstruct, repeated_support


class ProcessingTest(unittest.TestCase):
    def test_unknown_label_or_unmapped_id_stays_unknown(self):
        meals = pd.DataFrame({
            "input_row": [1, 2], "start": [100., 100.], "date": ["d1", "d1"],
            "energy": [250., 80.], "carbs": [50., 20.], "fat": [5., 0.],
            "protein": [5., 0.], "fiber": [3., 0.],
        })
        food = pd.DataFrame({
            "input_row": [1, 1, 1, 1, 2, 1], "t": [100., 110., 130., 1900., 100., 1901.],
            "date": ["d1"] * 6, "food_id": [1, 2, 3, 999, 1, 999],
            "food_category": ["bread", "bread_wholewheat", None, None, "bread", None],
            "clock_agrees": True, "qc_primary_eligible": True,
            "qc_same_food_time_n": 1, "qc_exact_signature_n": 1,
            "e": [80., 120., 40., 10., 80., 99.], "c": [20., 20., 10., 0., 20., 90.],
            "f": [0., 5., 0., 0., 0., 0.], "p": [0., 5., 0., 0., 0., 0.], "b": [0., 3., 0., 0., 0., 0.],
        })
        crosswalk = pd.DataFrame({"food_id": [1, 2, 3], "NOVA": pd.array([4, 1, None], dtype="Int64")})
        nutrients = list("ecfpb")
        result = reconstruct(meals, food, crosswalk, nutrients, 1800)
        self.assertEqual(result.upf_carbs.tolist(), [20., 20.])
        self.assertEqual(result.non_upf_carbs.tolist(), [20., 0.])
        self.assertEqual(result.unknown_carbs.tolist(), [10., 0.])
        self.assertEqual(result.unknown_kcal.tolist(), [50., 0.])
        self.assertEqual(result.unknown_events.tolist(), [2, 0])
        self.assertEqual(result.wholewheat_carbs.tolist(), [20., 0.])
        self.assertEqual(coverage(result)["high_coverage"].tolist(), [True, True])
        self.assertEqual(coverage(result)["classified_carb"].tolist(), [False, True])
        bad = meals.assign(carbs=[49., 20.])
        with self.assertRaisesRegex(ValueError, "nutrient totals"):
            reconstruct(bad, food, crosswalk, nutrients, 1800)

    def test_repeated_support_uses_distinct_days_and_overlap_wraps_midnight(self):
        meal = pd.DataFrame({
            "input_row": [1, 1, 1, 1, 2, 2, 2, 2], "date": ["a", "b", "a", "b", "a", "a", "b", "b"],
            "upf_carbs": [30, 30, 0, 0, 30, 30, 0, 0], "non_upf_carbs": [0, 0, 30, 30, 0, 0, 30, 30],
            "carbs": 30, "energy": 300, "protein": 10, "local_hour": [23, 23, 1, 1, 23, 23, 1, 1],
        })
        sources = ("upf_carbs", "non_upf_carbs")
        self.assertEqual(repeated_support(meal, sources).tolist(), [True] * 4 + [False] * 4)
        self.assertEqual(nearby_pair_count(meal.loc[meal.input_row.eq(1)], *sources), 1)


if __name__ == "__main__":
    unittest.main()
