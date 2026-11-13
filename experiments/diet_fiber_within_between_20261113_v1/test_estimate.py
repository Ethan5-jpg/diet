import unittest

import numpy as np

from estimate import contrasts, fold_losses, group_mean, infer, summarize


class FixedContrastTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.count = 80
        self.family = np.repeat(np.arange(self.count), 3)
        self.x = np.column_stack((np.ones(len(self.family)), rng.normal(size=len(self.family))))
        self.z = rng.normal(size=(len(self.family), 2))
        self.weight = rng.uniform(.25, 1., size=len(self.family))
        self.noise = rng.normal(scale=.15, size=len(self.family))
        self.base = self.x @ np.array([100, 2])

    def test_paired_contrast_uses_correlated_household_scores(self):
        def fit(beta):
            y = self.base + self.z @ beta + self.noise
            return infer(y, self.x, self.z, self.weight, self.family, self.count, 0)

        a, b = fit([1., 2.]), fit([-1., 1.])
        result = contrasts(a, b, 1.5, 1.96)
        self.assertAlmostEqual(result["full"]["within_minus_between"]["mg_dl"], 3.75)
        self.assertAlmostEqual(result["above_minus_below"]["within_minus_between"]["mg_dl"], .75)
        self.assertLess(result["full"]["within_minus_between"]["se"], 1e-10)

    def test_inference_agrees_with_weighted_full_design(self):
        y = self.base + self.z @ np.array([1., 2.]) + self.noise
        beta, scores = infer(y, self.x, self.z, self.weight, self.family, self.count, 0)
        design = np.column_stack((self.x, self.z))
        root = np.sqrt(self.weight)
        design = design * root[:, None]
        expected = np.linalg.lstsq(design, root * y, rcond=None)[0]
        influence = design @ np.linalg.inv(design.T @ design)[:, -2:]
        cluster = np.zeros_like(scores)
        np.add.at(cluster, self.family, influence * (root * y - design @ expected)[:, None])
        adjustment = self.count / (self.count - 1) * (len(y) - 1) / (len(y) - 4)
        np.testing.assert_allclose(beta, expected[-2:], atol=1e-12)
        np.testing.assert_allclose(scores, cluster * np.sqrt(adjustment), atol=1e-12)

    def test_prediction_scoring_and_fixed_nuisance(self):
        held = self.family % 5 == 0
        y = self.base + self.z @ np.array([1., 2.]) + self.noise
        total = fold_losses(y, self.x, self.z, self.weight, self.family, self.count, ~held, held)
        result = summarize(total, np.ones((3, self.count)))
        self.assertGreater(result["hinge_vs_linear"]["delta_r2"], 0.)
        self.assertGreater(result["linear_vs_baseline"]["delta_r2"], 0.)
        self.assertTrue(np.all(total[1] == 0))

    def test_person_means(self):
        np.testing.assert_allclose(group_mean(np.array([[1, 3], [5, 2], [3, 1]]),
                                             np.array([0, 1, 0]), 2), [[2, 2], [5, 2]])


if __name__ == "__main__":
    unittest.main()
