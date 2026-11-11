import sys
import unittest

import numpy as np

from curve_models import MODES, clustered_test, household_losses, summarize
from prepare_curve import MEAL

sys.path.insert(0, str(MEAL.parent / "diet_daily_modification_20261112_v1"))
try:
    import models
finally:
    sys.path.pop(0)


class CurveModelTests(unittest.TestCase):
    def test_orthonormal_modes_recover_the_two_hour_mean(self):
        phase = np.array([[0., 4., 6., 2.], [1., 9., 2., 4.]])
        np.testing.assert_allclose(MODES.T @ MODES, np.eye(4), atol=1e-12)
        np.testing.assert_allclose((phase @ MODES)[:, 0] / 2, phase.mean(axis=1))

    def test_level_projection_equals_original_matched_fit(self):
        rng = np.random.default_rng(5)
        x, z, y = rng.normal(size=(100, 3)), rng.normal(size=(100, 2)), rng.normal(size=(100, 4))
        family = np.arange(100) // 2
        w = rng.uniform(.3, 1., size=100)
        tr = family < 40
        fitted = models.fit_background(x[tr], y[tr, 0], w[tr])
        _, sse_null, sse = models.fit_interaction(x, z, y[:, 0], w, tr, ~tr, fitted, .03)
        result = household_losses(y, x, z, w, tr, ~tr, family, 50, models)
        self.assertAlmostEqual(sse_null, result["additive"][:, 0].sum())
        self.assertAlmostEqual(sse, result["level"][:, 0].sum())
        np.testing.assert_equal(result["additive"][:, 1:], result["level"][:, 1:])

    def test_scalar_cluster_inference_matches_archived_test(self):
        rng = np.random.default_rng(6)
        x, z, y = rng.normal(size=(100, 2)), rng.normal(size=(100, 3)), rng.normal(size=100)
        family = np.arange(100) // 2
        w = rng.uniform(.3, 1., size=100)
        fit = models.fit_background(x, y, w)
        ref, _, expected = models.household_test(z, fit, family, 50)
        p, b = clustered_test(y[:, None], x, z, w, family, 50, models)
        self.assertAlmostEqual(p, expected)
        np.testing.assert_allclose(b[:, 0], ref)

    def test_level_and_shape_denominators_use_distinct_components(self):
        parts = {k: np.ones((2, 4)) for k in ("tss","additive","level","smooth","unrestricted")}
        parts["smooth"][:, 2] *= .5
        result = summarize(parts, np.array([[0, 1], [1, 1], [0, 0]]))
        self.assertEqual(result["level"]["smooth_beyond_level"]["delta_r2"], 0)
        self.assertAlmostEqual(result["shape"]["smooth_beyond_level"]["delta_r2"], 1 / 6)


if __name__ == "__main__":
    unittest.main()
