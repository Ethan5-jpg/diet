import unittest

import numpy as np
import pandas as pd

from time_transfer import assign_periods
from paired import family_totals


class SplitTests(unittest.TestCase):
    def test_periods_keep_dates_together_and_drop_unsupported_people(self):
        counts = ([2, 2, 1, 1, 2], [3, 2, 3], [1, 1, 2, 2])
        rows = [(person, f"2026-01-{day + 1:02d}") for person, days in enumerate(counts)
                for day, n_meals in enumerate(days) for _ in range(n_meals)]
        meal = pd.DataFrame(rows, columns=["person_index", "date"]).sample(frac=1, random_state=2)
        keep, late = assign_periods(meal)
        self.assertEqual(set(meal.loc[keep, "person_index"]), {0})
        self.assertEqual(set(meal.loc[keep & late, "date"]), {"2026-01-03", "2026-01-04", "2026-01-05"})
        self.assertEqual(set(meal.loc[keep & ~late, "date"]), {"2026-01-01", "2026-01-02"})

    def test_prediction_loss_ignores_other_periods(self):
        class Model:
            @staticmethod
            def fit_background(x, y, w):
                return [np.linalg.lstsq(x, y, rcond=None)[0]]

        x = np.ones((6, 1))
        y = np.array([2., 2., 3., 4., 100., 100.])
        train = np.array([True, True, False, False, False, False])
        valid = np.array([False, False, True, True, False, False])
        family = np.arange(6) // 2
        total = family_totals(y, x, np.ones(6), {}, train, family, 3, Model(), valid=valid)
        np.testing.assert_allclose(total["none"], [0., 5., 0.])
        np.testing.assert_allclose(total["tss"], [0., 25., 0.])


if __name__ == "__main__":
    unittest.main()
