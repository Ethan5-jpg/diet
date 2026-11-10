"""Independent numerical references and actual pinned estimator APIs."""

import pickle
import unittest

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import nnls
from sklearn.metrics import r2_score

from estimators import GraphRidge, PairedHinges, SparseGP
from fusion import blend_weights, linear_fit
from models import BinRegressor, QuantileMean, atomic
from registry import specifications
from support import fixture
from test_features import role_for


class ModelChecks(unittest.TestCase):
    def test_every_default_case_fits_predicts_and_serializes_wide_sparse_profiles(self):
        data, cfg = fixture(wide=True)
        role, done = role_for(data, cfg), 0
        for spec in specifications(cfg):
            for case in spec["grid"]:
                fitted, _ = atomic(role, spec, case, cfg, 1)
                prediction = fitted.predict(role)
                if case["count"] == 256:
                    self.assertTrue(all(len(m["indices"]) == 256 for m in fitted.state["microbes"]))
                self.assertEqual(prediction.shape, role.split.yv.shape)
                self.assertTrue(np.isfinite(prediction).all())
                np.testing.assert_array_equal(prediction, pickle.loads(pickle.dumps(fitted)).predict(role))
                design = role.replay(fitted.state, True)[0]
                np.testing.assert_allclose(
                    fitted.predict(role, True),
                    fitted.mean + fitted.scale * fitted.model.predict(design),
                    atol=1e-12,
                )
                done += 1
                if done % 20 == 0:
                    print("CHECK_CANDIDATES %d/84" % done, flush=True)
        self.assertEqual(done, 84)

    def test_paired_hinges_learn_piecewise_interaction_and_allow_intercept(self):
        rng = np.random.default_rng(1303)
        x = rng.uniform(-2, 2, (900, 3))
        y = 2 * np.maximum(x[:, 0], 0) + 3 * np.maximum(x[:, 0], 0) * np.maximum(x[:, 1], 0)
        case = dict(max_terms=25, degree=2, penalty=0.0001)
        fit = PairedHinges(case).fit(x[:600], y[:600])
        self.assertGreater(r2_score(y[600:], fit.predict(x[600:])), 0.94)
        self.assertTrue(any(parent > 0 for parent, _, _, _ in fit.terms))
        baseline = PairedHinges(case).fit(np.ones((80, 3)), np.arange(80, dtype=float))
        np.testing.assert_allclose(baseline.predict(np.ones((9, 3))), 39.5)
        self.assertEqual(baseline.active, [0])

    def test_fitc_matches_independent_corrected_prior_and_exact_gp_limit(self):
        rng = np.random.default_rng(1304)
        x, t = rng.normal(size=(56, 5)), rng.normal(size=(13, 5))
        y = np.sin(x[:, 0]) + 0.1 * rng.normal(size=56)
        for inducing in (11, 56):
            case = dict(inducing=inducing, length=0.9, noise=0.15)
            fit = SparseGP(case, [2, 3], 19).fit(x, y)
            z = fit.scaler.transform(x) * fit.weights
            v = fit.scaler.transform(t) * fit.weights
            kuu = fit.kernel(fit.centers, fit.centers) + np.eye(inducing) * 1e-8
            kuf = fit.kernel(fit.centers, z)
            qff = kuf.T @ np.linalg.solve(kuu, kuf)
            covariance = qff + np.diag(0.15 + np.maximum(1 - np.diag(qff), 0))
            cross = fit.kernel(v, fit.centers) @ np.linalg.solve(kuu, kuf)
            expected = y.mean() + cross @ np.linalg.solve(covariance, y - y.mean())
            np.testing.assert_allclose(fit.predict(t), expected, atol=2e-7)
            if inducing == len(x):
                exact = y.mean() + fit.kernel(v, z) @ np.linalg.solve(
                    fit.kernel(z, z) + 0.15 * np.eye(len(x)), y - y.mean()
                )
                np.testing.assert_allclose(fit.predict(t), exact, atol=2e-7)
        duplicates = np.ones((30, 5))
        fit = SparseGP(dict(inducing=30, length=1.0, noise=0.1), [5], 1).fit(duplicates, np.arange(30.0))
        self.assertTrue(np.isfinite(fit.predict(duplicates)).all())

    def test_graph_laplacian_and_coefficients_match_independent_normal_equations(self):
        rng = np.random.default_rng(1305)
        x, y = rng.normal(size=(130, 12)), rng.normal(size=130)
        x[:, 5:8] += x[:, 4, None]
        for gamma in (0.0, 2.0):
            fit = GraphRidge(
                dict(graph_weight=gamma, penalty=0.02), [3, 0, 5, 4], ["host", "context", "oral", "gut"]
            ).fit(x, y)
            z = fit.scaler.transform(x)
            priors, start = [], 0
            for label, w in zip(fit.labels, fit.widths):
                laplacian = fit.laplacian(z[:, start : start + w])
                self.assertGreater(np.linalg.eigvalsh(laplacian).min(initial=0), -1e-10)
                priors.append(
                    max(w, 1) * (np.eye(w) + (gamma * laplacian if label in ("oral", "gut") else 0))
                )
                start += w
            penalty = block_diag(*priors)
            beta = np.linalg.solve(z.T @ z + len(y) * 0.02 * penalty, z.T @ (y - y.mean()))
            np.testing.assert_allclose(fit.predict(x), y.mean() + z @ beta, atol=1e-10)

    def test_bins_preserve_tied_classes_probability_mean_and_constant_target(self):
        rng = np.random.default_rng(1306)
        x = rng.normal(size=(120, 4))
        y = np.repeat([0.0, 2.0, 5.0], [40, 40, 40])
        cases = [
            dict(bins=8, iterations=5, depth=3, l2=10),
            dict(bins=8, iterations=5, leaves=7, leaf=10, l2=10),
        ]
        for family, case in zip(("bin_cat", "bin_hgb"), cases):
            fit = BinRegressor(case, family, 1, 1).fit(x, y)
            p = fit.predict(x)
            self.assertTrue(np.all((p >= 0) & (p <= 5)))
            np.testing.assert_allclose(p, fit.model.predict_proba(x) @ fit.means)
            constant = BinRegressor(case, family, 1, 1).fit(x, np.full(120, 9.0))
            np.testing.assert_array_equal(constant.predict(x[:7]), np.full(7, 9.0))

    def test_atomic_constant_targets_use_exact_native_unit_fallback(self):
        data, cfg = fixture()
        role = role_for(data, cfg)
        role.split.yt[:] = 92.5
        seen = set()
        for spec in specifications(cfg):
            if spec["learner"] in seen:
                continue
            seen.add(spec["learner"])
            fitted, _ = atomic(role, spec, spec["grid"][0], cfg, 1)
            np.testing.assert_array_equal(fitted.predict(role), np.full(len(role.split.yv), 92.5))
        self.assertEqual(len(seen), 8)

    def test_multi_quantile_pinned_output_is_midpoint_mean(self):
        rng = np.random.default_rng(1307)
        x, y = rng.normal(size=(100, 4)), rng.normal(size=100)
        case = dict(iterations=5, depth=3, l2=10, quantiles=5)
        fit = QuantileMean(case, 2, 1).fit(x, y)
        p = np.asarray(fit.model.predict(x[:3]))
        self.assertEqual(p.shape, (3, 5))
        np.testing.assert_array_equal(fit.predict(x[:3]), p.mean(1))

    def test_positive_fusion_agrees_with_augmented_nonnegative_least_squares(self):
        rng = np.random.default_rng(1308)
        x = rng.normal(size=(120, 7))
        y = x[:, 0] + 0.8 * x[:, 1] - 0.3 * x[:, 2] + 20
        fit = linear_fit(x, y, "positive", 0.03)
        z = fit.scaler.transform(x)
        augmented = np.vstack([z, np.sqrt(len(y) * 0.03) * np.eye(z.shape[1])])
        target = np.r_[(y - fit.mean) / fit.scale, np.zeros(z.shape[1])]
        beta, _ = nnls(augmented, target)
        np.testing.assert_allclose(fit.predict(x), fit.mean + fit.scale * z @ beta, atol=2e-4)
        a = np.column_stack([y - 1, y + 2])
        w = blend_weights(a, y)
        np.testing.assert_allclose(w, [2 / 3, 1 / 3], atol=1e-6)
        self.assertAlmostEqual(float(w.sum()), 1.0)
