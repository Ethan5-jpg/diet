import unittest

import numpy as np
import pandas as pd

from context import interaction_design, select_contexts, split_background
from evaluate_context import summarize


class ContextTests(unittest.TestCase):
    def test_support_requires_repeated_distinct_dates_and_dose_span_in_each_context(self):
        rows = [
            (1, "a", 8., 500., 20.), (1, "b", 8., 600., 30.),
            (1, "a", 13., 180., 23.), (1, "b", 13., 240., 36.),
            (1, "a", 10., 600., 30.), (1, "c", 14., 240., 65.),
            (2, "a", 8., 500., 20.), (2, "b", 8., 500., 24.),
            (2, "a", 12., 180., 25.), (2, "b", 12., 180., 39.),
        ]
        frame = pd.DataFrame(rows, columns=["input_row", "date", "local_hour", "prior_gap", "carbs"])
        observed = select_contexts(frame)
        self.assertEqual(observed.index.tolist(), [0, 1, 2, 3])
        self.assertEqual(observed.morning.tolist(), [True, True, False, False])

    def test_context_design_keeps_lower_order_terms(self):
        morning = np.array([False, True])
        marker = np.array([[3., 2.], [3., 2.]])
        dose = np.ones(2)
        common, diff, host = interaction_design(dose, marker, morning, np.array([[4.], [4.]]))
        np.testing.assert_equal(common, marker)
        np.testing.assert_equal(diff[:, 0], [-1.5, 1.5])
        np.testing.assert_equal(host, [[4., 0.], [0., 4.]])
        np.testing.assert_equal(split_background(marker[:, :1], morning), [[3., 0.], [0., 3.]])

    def test_increment_uses_shared_comparator(self):
        values = {k: np.array(v) for k, v in
                  dict(none=[8., 12.], shared=[6., 10.], context=[4., 9.], tss=[40., 50.]).items()}
        stat = summarize(values, np.array([[0, 1], [0, 0], [1, 1]]))
        self.assertEqual(stat["shared"]["delta_r2"], 4 / 90)
        self.assertEqual(stat["context_beyond_shared"]["delta_r2"], 3 / 90)


if __name__ == "__main__":
    unittest.main()
