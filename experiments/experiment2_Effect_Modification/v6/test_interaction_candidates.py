"""Synthetic checks of held-out replay and pure-product candidate semantics."""

import inspect
import json
import unittest

import numpy as np

from interaction_candidates import Candidate, KINDS, fit_correction, options, prepare_modifier


class CandidateChecks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20261118)
        self.z = rng.normal(size=(600, 10))
        self.zv = rng.normal(size=(100, 10))
        self.d = rng.normal(size=(600, 2))
        self.dv = rng.normal(size=(100, 2))
        self.delta = np.array([1.0, -0.25])
        centered = self.z - self.z.mean(axis=0)
        self.y = self.d[:, 0] * (centered @ np.linspace(-1.0, 1.0, 10))

    def fit(self, config, z_valid=None, dose_valid=None, delta=None, y=None):
        return fit_correction(self.z, self.zv if z_valid is None else z_valid,
                              self.d, self.dv if dose_valid is None else dose_valid,
                              self.delta if delta is None else delta,
                              self.y if y is None else y, config)

    def test_fixed_grid_and_serialization(self):
        grid = options()
        self.assertEqual(len(grid), 66)
        self.assertEqual(sum(map(len, grid.values())), 198)
        self.assertEqual(len({c.key() for choices in grid.values() for c in choices}), 198)
        self.assertEqual(sum(choices[0].view != "pc" for choices in grid.values()), 18)
        self.assertIn("gut_ridge_pc10_dose1", grid)
        self.assertIn("both_elasticnet_two_part16_dose1", grid)
        self.assertIn("oral_tensor_pc10_dose2", grid)
        self.assertIn("gut_vc_boost_pc20_dose1", grid)
        for name, choices in grid.items():
            self.assertTrue(all(c.route == name for c in choices))
            json.dumps([c.to_dict() for c in choices], allow_nan=False)
        self.assertNotIn("y_valid", inspect.signature(fit_correction).parameters)

    def test_ridge_matches_direct_product_oracle(self):
        config = Candidate("gut", 10, "ridge", 0.1, n_dose=2)
        prediction, contrast, _ = self.fit(config)
        z = self.z - self.z.mean(axis=0)
        zv = self.zv - self.z.mean(axis=0)
        a = (self.d[:, :, None] * z[:, None, :]).reshape(len(z), -1)
        b = (self.dv[:, :, None] * zv[:, None, :]).reshape(len(zv), -1)
        c = (self.delta[None, :, None] * zv[:, None, :]).reshape(len(zv), -1)
        # Ridge on RMS-scaled columns is an unscaled product regression with
        # a diagonal penalty; this independent normal-equation form is the oracle.
        penalty = len(a) * config.strength * np.diag(np.mean(a ** 2, axis=0))
        beta = np.linalg.solve(a.T @ a + penalty, a.T @ self.y)
        np.testing.assert_allclose(prediction, b @ beta, atol=1e-11)
        np.testing.assert_allclose(contrast, c @ beta, atol=1e-11)

    def test_all_families_are_zero_at_zero_dose_and_replay(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                config = Candidate("gut", 10, kind, 0.01, n_dose=1 if kind == "vc_boost" else 2)
                prediction, contrast, metadata = self.fit(config, dose_valid=np.zeros_like(self.dv))
                np.testing.assert_array_equal(prediction, np.zeros(len(self.dv)))
                self.assertTrue(np.isfinite(contrast).all())
                json.dumps(metadata, allow_nan=False)
                expected = self.fit(config)
                repeated = self.fit(config)
                np.testing.assert_array_equal(expected[0], repeated[0])
                np.testing.assert_array_equal(expected[1], repeated[1])
                # Same fitted coefficients: difference of two dose predictions
                # must equal the separately computed 10-to-15 contrast.
                low = self.fit(config, dose_valid=np.zeros_like(self.dv))[0]
                high = self.fit(config, dose_valid=np.broadcast_to(self.delta, self.dv.shape))[0]
                np.testing.assert_allclose(high - low, expected[1], atol=1e-12)

    def test_linear_signal_recovered_without_validation_targets(self):
        zv = self.zv - self.z.mean(axis=0)
        truth = self.dv[:, 0] * (zv @ np.linspace(-1.0, 1.0, 10))
        for kind in ("ridge", "pls", "elasticnet"):
            with self.subTest(kind=kind):
                config = Candidate("gut", 10, kind, 0.01)
                prediction, _, metadata = self.fit(config)
                self.assertLess(np.mean((prediction - truth) ** 2), 0.01 * np.mean(truth ** 2))
                self.assertTrue(metadata["converged"])

    def test_validation_rows_never_fit_representation(self):
        for kind in ("rbf", "tensor", "vc_boost"):
            with self.subTest(kind=kind):
                config = Candidate("both", 10, kind, 0.1)
                first = self.fit(config)
                changed_z = self.zv.copy()
                changed_z[-1] += 1000
                second = self.fit(config, z_valid=changed_z)
                np.testing.assert_allclose(first[0][:-1], second[0][:-1], atol=1e-12)
                np.testing.assert_allclose(first[1][:-1], second[1][:-1], atol=1e-12)

    def test_prepared_modifier_reuses_identical_baseline_columns(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                config = Candidate("gut", 10, kind, 0.1)
                prepared = prepare_modifier(self.z, self.zv, config)
                changed = prepare_modifier(self.z, self.zv, Candidate("gut", 10, kind, .01,
                                           n_dose=1 if kind == "vc_boost" else 2))
                np.testing.assert_array_equal(prepared[0], changed[0])
                np.testing.assert_array_equal(prepared[1], changed[1])
                original = self.fit(config)
                replay = fit_correction(self.z, self.zv, self.d, self.dv,
                                        self.delta, self.y, config,
                                        prepared_modifier=prepared)
                np.testing.assert_array_equal(original[0], replay[0])
                np.testing.assert_array_equal(original[1], replay[1])

    def test_zero_residual_yields_zero_correction(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                prediction, contrast, _ = self.fit(Candidate("gut", 10, kind, 0.1), y=np.zeros(len(self.z)))
                np.testing.assert_array_equal(prediction, np.zeros(len(self.zv)))
                np.testing.assert_array_equal(contrast, np.zeros(len(self.zv)))

    def test_tensor_matches_independent_hinge_product_oracle(self):
        config = Candidate("gut", 10, "tensor", .1, n_dose=2)
        predicted, contrast, _ = self.fit(config)
        center = self.z.mean(axis=0)
        train, valid = self.z - center, self.zv - center
        knots = np.quantile(train, (.25, .5, .75), axis=0)
        a = np.column_stack([train, *(np.maximum(train-knot, 0) for knot in knots)])
        b = np.column_stack([valid, *(np.maximum(valid-knot, 0) for knot in knots)])
        a, b = a-a.mean(axis=0), b-a.mean(axis=0)
        product = (self.d[:, :, None]*a[:, None, :]).reshape(len(a), -1)
        valid_product = (self.dv[:, :, None]*b[:, None, :]).reshape(len(b), -1)
        contrast_product = (self.delta[None, :, None]*b[:, None, :]).reshape(len(b), -1)
        penalty = len(a)*config.strength*np.diag(np.mean(product**2, axis=0))
        beta = np.linalg.solve(product.T@product+penalty, product.T@self.y)
        np.testing.assert_allclose(predicted, valid_product@beta, atol=1e-10)
        np.testing.assert_allclose(contrast, contrast_product@beta, atol=1e-10)
        self.assertEqual(a.shape[1], 40)

    def test_varying_coefficient_step_signal_has_closed_form(self):
        # Balanced +/-1 Z and +/-1 dose make each shallow tree recover the
        # exact step direction. The entire 40-stage shrinkage has a closed form.
        z = np.repeat([[-1.], [1.]], 64, axis=0)
        d = np.tile([-1., 1.], 64)[:, None]
        truth = 2*z[:, 0]
        y = d[:, 0]*truth
        zv = np.array([[-1.], [1.], [-1.], [1.]])
        dv = np.array([[-2.], [-2.], [3.], [3.]])
        strength = .1
        config = Candidate("gut", 1, "vc_boost", strength)
        pred, contrast, metadata = fit_correction(z, zv, d, dv, [5.], y, config)
        g = 2*zv[:, 0]*(1-(1-.15/(1+strength))**40)
        np.testing.assert_allclose(pred, dv[:, 0]*g, rtol=0, atol=1e-12)
        np.testing.assert_allclose(contrast, 5*g, rtol=0, atol=1e-12)
        self.assertEqual(metadata["vc_stages"], 40)
        self.assertAlmostEqual(metadata["vc_training_modifier_mean"], 0., places=14)

    def test_varying_coefficient_recovers_nonlinear_and_linear_modifier(self):
        rng = np.random.default_rng(2026)
        z, zv = rng.uniform(-1, 1, (600, 1)), rng.uniform(-1, 1, (120, 1))
        d, dv = rng.normal(size=(600, 1)), rng.normal(size=(120, 1))
        for function in (lambda x: 2*x, lambda x: 3*(x > .2).astype(float)-1):
            train_g = function(z[:, 0])
            center = train_g.mean()
            truth = dv[:, 0]*(function(zv[:, 0])-center)
            predicted, _, metadata = fit_correction(z, zv, d, dv, [1.],
                d[:, 0]*(train_g-center), Candidate("gut", 1, "vc_boost", .01))
            self.assertLess(np.mean((predicted-truth)**2), .03*np.mean(truth**2))
            self.assertAlmostEqual(metadata["vc_training_modifier_mean"], 0., places=14)

    def test_varying_coefficient_accepts_negative_centered_descent_direction(self):
        from interaction_candidates import _varying_coefficient_boost
        # Both dose means are zero, but their variances differ by microbial
        # group. The weighted tree fits -.5+.2*Z exactly. Removing its mean
        # leaves h=.2*Z, whose residual covariance is negative. Rejecting that
        # sign would incorrectly return zero although a signed step lowers SSE.
        z = np.repeat([-1., 1.], 50)[:, None]
        dose = np.r_[np.tile([-1., 1.], 25), np.tile([-3., 3.], 25)]
        residual = dose * (-.5 + .2*z[:, 0])
        h = .2*z[:, 0]
        direction = dose*h
        numerator = float(direction @ residual)
        denominator = float(direction @ direction + .01*(h @ h))
        coefficient = .15*numerator/denominator
        self.assertLess(numerator, 0.)
        fitted, replay, metadata = _varying_coefficient_boost(
            z, z, dose, residual, .01, 20261118, steps=1)
        np.testing.assert_allclose(fitted, coefficient*h, rtol=0, atol=1e-13)
        np.testing.assert_array_equal(fitted, replay)
        self.assertEqual(metadata["vc_stages"], 1)
        self.assertLess(np.sum((residual-dose*fitted)**2), np.sum(residual**2))
        self.assertAlmostEqual(fitted.mean(), 0., places=14)

    def test_varying_coefficient_handles_zero_and_tiny_doses(self):
        d = self.d.copy()
        d[:25, 0] = 0
        d[25:50, 0] = 1e-20
        config = Candidate("gut", 10, "vc_boost", .1)
        result = fit_correction(self.z, self.zv, d, self.dv, self.delta, self.y, config)
        self.assertTrue(np.isfinite(result[0]).all())
        self.assertEqual(result[2]["vc_training_dose_nonzero_rows"], len(d)-50)
        zeros = fit_correction(self.z, self.zv, np.zeros_like(d), self.dv,
                               self.delta, self.y, config)
        np.testing.assert_array_equal(zeros[0], np.zeros(len(self.zv)))
        np.testing.assert_array_equal(zeros[1], np.zeros(len(self.zv)))
        with self.assertRaisesRegex(ValueError, "LINEAR_DOSE"):
            Candidate("gut", 10, "vc_boost", .1, n_dose=2)
        with self.assertRaisesRegex(ValueError, "COMPOSITIONAL"):
            Candidate("gut", 16, "tensor", .1, view="balance")

    def test_both_site_columns_are_not_truncated(self):
        rng = np.random.default_rng(8)
        z, zv = rng.normal(size=(600, 20)), rng.normal(size=(100, 20))
        d = self.d[:, :1]
        target = d[:, 0] * (z[:, 18] - z[:, 18].mean())
        prediction, _, metadata = fit_correction(z, zv, d, self.dv[:, :1], [1], target,
                                                Candidate("both", 10, "ridge", 0.001))
        truth = self.dv[:, 0] * (zv[:, 18] - z[:, 18].mean())
        self.assertEqual(metadata["supplied_microbiome_columns"], 20)
        self.assertLess(np.mean((prediction - truth) ** 2), 0.001 * np.mean(truth ** 2))

    def test_degenerate_and_invalid_inputs(self):
        zeros = np.zeros_like(self.z)
        prediction, contrast, metadata = fit_correction(zeros, np.zeros_like(self.zv), self.d,
                                                       self.dv, self.delta, self.y,
                                                       Candidate("gut", 10, "ridge", .1))
        self.assertEqual(metadata["active_columns"], 0)
        self.assertTrue(np.all(prediction == 0) and np.all(contrast == 0))
        with self.assertRaises(ValueError):
            self.fit(Candidate("gut", 10, "ridge", .1), y=self.y[:-1])
        with self.assertRaises(ValueError):
            self.fit(Candidate("gut", 10, "ridge", .1), delta=[np.nan])
        with self.assertRaises(ValueError):
            Candidate("gut", 10, "ridge", 0)


if __name__ == "__main__":
    unittest.main()
