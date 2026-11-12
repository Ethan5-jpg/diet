import unittest

import numpy as np

from community import PENALTIES, centered_kernel, clr_kernels, covariance_moments, predict, summarize


class CommunityTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(13)
        self.profile = rng.normal(size=(120, 3))
        self.training = np.arange(len(self.profile)) < 90
        self.info = rng.uniform(.5, 2, len(self.profile))
        self.y = np.column_stack((self.profile[:, 0] ** 2, self.profile[:, 1]))

    def test_weighted_centering_and_unit_trace(self):
        x = self.profile
        keep = self.training
        weights = self.info[keep] / self.info[keep].sum()
        kernel = centered_kernel(x @ x[keep].T, (x ** 2).sum(axis=1), keep, weights)
        np.testing.assert_allclose(kernel @ weights, 0, atol=1e-12)
        self.assertAlmostEqual(np.diag(kernel[keep]) @ weights, 1)

    def test_nonlinear_recovery_and_no_held_outcome_use(self):
        keep = self.training
        x = self.profile
        geometry = dict(pc=[(x @ x[keep].T, (x ** 2).sum(axis=1))], community=clr_kernels(x, keep))
        prediction = predict(geometry, self.info, self.y, keep)
        changed = self.y.copy()
        changed[~keep] += 100
        refit = predict(geometry, self.info, changed, keep)
        for name in prediction:
            np.testing.assert_allclose(prediction[name], refit[name])
        error = {key: np.mean((self.y[~keep, 0] - prediction[key][~keep, 0]) ** 2) for key in prediction}
        self.assertLess(error["community_weak"], error["pc"] * .5)

    def test_training_kernel_ignores_held_profiles(self):
        keep = self.training
        original = clr_kernels(self.profile, keep)
        edited = self.profile.copy()
        edited[~keep] *= 20
        for before, after in zip(original, clr_kernels(edited, keep)):
            np.testing.assert_array_equal(before[0][keep], after[0][keep])

    def test_paired_covariance_and_loss_scale(self):
        x = np.arange(5.)
        people = np.arange(5)
        raw = covariance_moments(x, 2 * x, people, len(x))
        adjusted = covariance_moments(x, x, people, len(x))
        stats = np.column_stack((raw, raw, raw, adjusted))
        loss = np.tile([10, 10, 8, 5, 6, 7], (len(x), 1))
        self.assertEqual(loss.shape[1], len(PENALTIES) + 2)
        result = summarize({"net__covariance": stats, "net__meal": loss}, np.ones((3, 5)))["net"]
        self.assertAlmostEqual(result["covariance"]["community_vs_pc"]["value"], 2.5)
        self.assertAlmostEqual(result["meal"]["community_vs_pc"]["value"], .3)


if __name__ == "__main__":
    unittest.main()
