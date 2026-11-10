import pickle
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import check_grad

from engine import blend_weights
from estimators import Factorization, MixtureExperts, fm_value_gradient


class NumericalChecks(unittest.TestCase):
    def test_fm_gradient_and_explicit_pair_prediction(self):
        rng = np.random.default_rng(17)
        x, y, rank = rng.normal(size=(23, 5)), rng.normal(size=23), 3
        theta = rng.normal(size=1 + 5 + 5 * rank) * 0.2
        error = check_grad(
            lambda t: fm_value_gradient(t, x, y, rank, 0.1)[0],
            lambda t: fm_value_gradient(t, x, y, rank, 0.1)[1],
            theta,
        )
        self.assertLess(error, 1e-6)
        fitted = Factorization(rank, 0.1, 100, 4).fit(x, y)
        z = fitted.scaler.transform(x)
        v = fitted.theta[6:].reshape(5, rank)
        explicit = fitted.theta[0] + z @ fitted.theta[1:6]
        for i in range(5):
            for j in range(i):
                explicit += z[:, i] * z[:, j] * (v[i] @ v[j])
        assert_allclose(fitted.predict(x), explicit, atol=1e-12)

    def test_blend_simplex_known_optimum_and_roundtrip(self):
        x = np.column_stack([np.arange(20), np.arange(20)[::-1]])
        y = x @ np.array([0.3, 0.7])
        weight = blend_weights(x, y)
        assert_allclose(weight, [0.3, 0.7], atol=1e-6)
        self.assertTrue((weight >= 0).all())
        self.assertAlmostEqual(weight.sum(), 1.0)

    def test_mixture_gate_is_train_fitted_and_normalized(self):
        rng = np.random.default_rng(29)
        x, g = rng.normal(size=(90, 6)), rng.normal(size=(90, 3))
        y = x[:, 0] + rng.normal(size=90)
        model = MixtureExperts(2, 0.1, 3).fit(x, g, y)
        old = model.gate.means_.copy()
        w = model.gate.predict_proba(g[:7] * 100)
        assert_allclose(w.sum(1), 1.0)
        model.predict(x[:7], g[:7] * 100)
        assert_allclose(model.gate.means_, old)
        assert_allclose(pickle.loads(pickle.dumps(model)).predict(x, g), model.predict(x, g))
