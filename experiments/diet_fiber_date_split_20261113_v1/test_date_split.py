import unittest

import numpy as np

from date_split import design, gap, period_means


class CalendarTest(unittest.TestCase):
    def test_interleaved_rows_unequal_periods_and_support(self):
        rows = np.repeat([2, 7, 9], [5, 4, 3])
        day = np.array([1, 2, 5, 8, 10, 2, 4, 6, 9, 1, 2, 3])
        values = np.column_stack((day, 2 * day))
        dates = np.datetime64("2026-01-01") + day
        order = np.random.default_rng(9).permutation(len(day))
        people, means, early, late = period_means(rows[order], dates[order], values[order])
        np.testing.assert_array_equal(people, [2, 7])
        np.testing.assert_array_equal(early, [2, 2])
        np.testing.assert_array_equal(late, [3, 2])
        np.testing.assert_allclose(means["early"], [[1.5, 3], [3, 6]])
        np.testing.assert_allclose(means["late"], [[23 / 3, 46 / 3], [7.5, 15]])
        np.testing.assert_allclose(means["all"][:, 0], [5.2, 5.25])

    def test_duplicate_date_fails(self):
        with self.assertRaisesRegex(ValueError, "repeated person-date"):
            period_means(np.zeros(4), np.repeat(np.datetime64("2026-01-01"), 4), np.ones((4, 2)))

    def test_cross_date_design_uses_only_designated_exposures_and_outcome(self):
        x = np.array([[90, 10, 40, 1, 2000, 50, 20], [95, 15, 60, 0, 2500, 40, 16]], float)
        later = x + 1
        y, initial, dose = design(x, later, lambda d: ((d - 10) / 5)[:, None])
        later[:, 1:] *= 100
        x[:, 0] += 99
        other = design(x, later, lambda d: ((d - 10) / 5)[:, None])
        np.testing.assert_array_equal(y, [91, 96])
        np.testing.assert_array_equal(dose[:, 0], [0, 1])
        for a, b in zip((y, initial, dose), other):
            np.testing.assert_array_equal(a, b)

    def test_paired_gap_uses_correlated_households(self):
        scores = np.arange(5.)[:, None]
        result = gap((np.array([2.]), scores), (np.array([3.]), scores), 1.25, 2)
        self.assertEqual(result, {"mg_dl": -1.25, "ci95": [-1.25, -1.25]})


if __name__ == "__main__":
    unittest.main()
