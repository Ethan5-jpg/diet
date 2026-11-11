import unittest

import numpy as np

from evaluate import period_stats
from slopes import family_prediction, family_repeatability, fold_models, read_repeatability, sums


class RepeatedSlopeTests(unittest.TestCase):
    def test_slope_and_corrected_meal_loss_reproduce_row_calculation(self):
        dose = np.array([-2., -1., 1., 2.])
        residual = 1.5 * dose + np.array([1., -1., -1., 1.])
        people = np.zeros(4, int)
        data = sums(people, dose, residual, residual, np.full(4, .25), 1)
        pred = {"marker": np.array([1.5])}
        loss = family_prediction(np.array([0]), data, pred, np.array([True]), 1)
        self.assertEqual(data[0, 1] / data[0, 0], 1.5)
        self.assertAlmostEqual(loss["meal_marker"][0], np.mean((residual - 1.5 * dose) ** 2))

    def test_repeatability_matches_person_level_moments(self):
        a = np.array([1., 2., 3., 4.])
        b = np.array([2., 1., 4., 3.])
        info = family_repeatability(np.array([0, 0, 1, 2]), a, b, np.ones(4, bool), 3)
        stat = read_repeatability(info, np.array([[0, 1, 2], [0, 0, 2]]))
        self.assertAlmostEqual(stat["pearson_r"]["value"], np.corrcoef(a, b)[0, 1])
        self.assertAlmostEqual(stat["covariance"]["value"], np.cov(a, b)[0, 1])

    def test_outer_held_out_target_does_not_change_prediction(self):
        source = np.array([2., 1., 4., 3., 8.])
        target = source + .2
        info = np.full((len(source), 2), 10.)
        train = np.arange(len(source)) < 4
        score = source[:, None]
        original = fold_models(score, source, target, info, train, 0)
        target[-1] = -500
        modified = fold_models(score, source, target, info, train, 0)
        for name in original:
            np.testing.assert_equal(original[name], modified[name])

    def test_held_out_household_does_not_enter_background_fits(self):
        class Model:
            @staticmethod
            def fit_background(x, y, w):
                return [np.linalg.lstsq(x, y, rcond=None)[0]]

        person = np.repeat(np.arange(5), 8)
        late = np.tile(np.repeat([False, True], 4), 5)
        dose = np.tile([-2., -1., 1., 2.], 10)
        x = dose[:, None]
        weights = np.full(len(person), .25)
        y = 2 * dose
        a = period_stats((y, x, weights), dose, person, late, np.arange(5), 0, Model())
        y[person == 0] += 100 * dose[person == 0]
        b = period_stats((y, x, weights), dose, person, late, np.arange(5), 0, Model())
        for first, second in zip(a, b):
            np.testing.assert_equal(first[1:], second[1:])
            self.assertGreater(second[0, 1], first[0, 1])


if __name__ == "__main__":
    unittest.main()
