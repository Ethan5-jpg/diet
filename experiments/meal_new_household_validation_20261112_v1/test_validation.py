"""Synthetic regression checks for original QC and frozen prediction."""

import importlib.util
import unittest

import numpy as np
import pandas as pd

from contracts import ENDPOINTS, ROOT
from frozen import PENALTY, fit_frozen, summarize_losses
from prepare_holdout import eligible_windows


class ValidationTest(unittest.TestCase):
    def test_frozen_prediction_matches_original_estimator(self):
        path = ROOT.parent / "diet_daily_modification_20261112_v1/models.py"
        spec = importlib.util.spec_from_file_location("original_models", path)
        model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model)
        rng = np.random.default_rng(52)
        x = rng.normal(size=(60, 4))
        z = rng.normal(size=(60, 3))
        y = x[:, 0] + z[:, 1] + rng.normal(size=60)
        w = rng.uniform(.1, .4, size=60)
        train = np.arange(60) < 40
        valid = ~train
        baseline, projection, beta = fit_frozen(y[train], x[train], w[train], z[train], model)
        null = x[valid] @ baseline
        alternative = null + (z[valid] - x[valid] @ projection) @ beta
        fit = model.fit_background(x[train], y[train], w[train])
        expected_beta, expected_null, expected_alt = model.fit_interaction(x, z, y, w, train, valid, fit, PENALTY)
        np.testing.assert_allclose(beta, expected_beta, atol=1e-12, rtol=0)
        np.testing.assert_allclose([np.sum(w[valid] * (y[valid] - v) ** 2) for v in (null, alternative)],
                                   [expected_null, expected_alt], atol=1e-12, rtol=0)

    def test_original_minimums_apply_after_full_offset_qc(self):
        meal = pd.DataFrame({
            "input_row": [0] * 5 + [1] * 5, "date": ["a", "a", "b", "c", "c"] * 2,
            "full_offsets": [True] * 9 + [False],
        })
        retained = eligible_windows(meal, {"minimum_meals": 5, "minimum_days": 3})
        self.assertEqual(retained.input_row.tolist(), [0] * 5)

    def test_paired_improvement_uses_fixed_household_denominator(self):
        data = np.array([[2., 4.], [1., 2.], [4., 8.]])
        out = summarize_losses({endpoint: data for endpoint in ENDPOINTS})
        self.assertEqual(out["net_p0"]["delta_r2"], .25)
        self.assertEqual(out["net_p0"]["interval"], [.25, .25])


if __name__ == "__main__":
    unittest.main()
