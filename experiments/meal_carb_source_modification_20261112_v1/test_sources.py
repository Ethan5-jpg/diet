import unittest

import numpy as np
import pandas as pd

from analyze_sources import exposures, summarize
from exposure import TOTALS, reconstruct, supported


class ExposureTests(unittest.TestCase):
    def test_reconstruction_preserves_carbohydrate_and_exact_category(self):
        names = ("k", "c", "f", "p", "fi")
        meal = pd.DataFrame(dict(input_row=[0], start=[1.], date=["day"], energy=[100.], carbs=[20.],
                                 fat=[1.], protein=[2.], fiber=[1.]))
        food = pd.DataFrame(dict(input_row=[0, 0, 0, 1], t=[1., 301., 2000., 1.], date="day",
                                 food_category=[" Bread ", "bread_wholewheat", "fruits", "fruits"],
                                 clock_agrees=True, qc_primary_eligible=True, qc_same_food_time_n=1,
                                 qc_exact_signature_n=1))
        food[list(names)] = [[60, 12, 1, 1, 1], [40, 8, 0, 1, 0], [50, 20, 0, 0, 0], [50, 20, 0, 0, 0]]
        joined = reconstruct(meal, food, names, 1800)
        self.assertEqual(joined.loc[0, "bread"], 12)
        self.assertEqual(joined.loc[0, "bread_wholewheat"], 8)
        self.assertEqual(joined.loc[0, "fruits"], 0)
        np.testing.assert_equal(joined[list(TOTALS)].to_numpy(), meal[list(TOTALS)].to_numpy())
        with self.assertRaisesRegex(ValueError, "nutrients or dates"):
            reconstruct(meal.assign(carbs=21), food, names, 1800)

    def test_support_counts_distinct_days_per_source(self):
        meal = pd.DataFrame(dict(input_row=[0, 0, 1, 1], date=["a", "b", "a", "a"],
                                 bread=[10, 10, 10, 10], fruits=[11, 11, 11, 11]))
        np.testing.assert_equal(supported(meal).to_numpy(), [True, True, False, False])

    def test_swap_changes_only_the_intended_dose_coordinates(self):
        meal = pd.DataFrame(dict(bread=[20, 10], fruits=[0, 10], carbs=[45, 45],
                                 age_calendar_years=49, sex_code=0, bread_wholewheat=4, unlabelled=0))
        exchange, pair, total, _ = exposures(meal)
        self.assertEqual(exchange[1] - exchange[0], 1)
        self.assertEqual(pair[0], pair[1])
        self.assertEqual(total[0], total[1])

    def test_prediction_uses_matched_losses(self):
        parts = dict(none=np.array([10., 20.]), exchange=np.array([9., 19.]), tss=np.array([100., 150.]))
        row = summarize(parts, np.array([[0, 1], [1, 1], [0, 0]]))
        self.assertEqual(row["delta_r2"], 2 / 250)
        self.assertGreater(row["ci95"][0], 0)


if __name__ == "__main__":
    unittest.main()
