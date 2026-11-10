"""Independent numerical and learner invariants; no HPP measurements."""

import unittest

import numpy as np
from scipy import linalg

import learners
from numerics import (
    KernelProfile,
    interaction_contrast,
    interaction_kernel,
    site_kernel,
)


class NumericalChecks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(76)
        self.ht = np.column_stack([np.ones(45), rng.normal(size=(45, 2))])
        self.hv = np.column_stack([np.ones(12), rng.normal(size=(12, 2))])
        self.zt, self.zv = rng.normal(size=(45, 7)), rng.normal(size=(12, 7))
        self.y = (
            self.ht @ np.array([90.0, 2.0, 3.0])
            + self.zt[:, 0] * 4
            + rng.normal(size=45)
        )

    def test_dual_kernel_agrees_with_independent_primal_ridge(self):
        kt, kv = self.zt @ self.zt.T, self.zv @ self.zt.T
        profile = KernelProfile(self.ht, self.hv, kt, kv, self.y)
        design, valid = (
            np.column_stack([self.ht, self.zt]),
            np.column_stack([self.hv, self.zv]),
        )
        for penalty, fit in zip([0.1, 0.001], profile.path([0.1, 0.001])):
            ridge = np.diag([0.0] * 3 + [len(self.y) * penalty] * 7)
            expected = valid @ linalg.solve(
                design.T @ design + ridge, design.T @ self.y, assume_a="pos"
            )
            np.testing.assert_allclose(fit["prediction"], expected, atol=1e-8, rtol=0)
            np.testing.assert_allclose(
                profile.fit(penalty)["prediction"], expected, atol=1e-8, rtol=0
            )

    def test_kernel_training_quantities_ignore_validation_features(self):
        for kind in ("linear", "rbf"):
            kt, _, state = site_kernel(self.zt, self.zv, kind)
            changed, _, other = site_kernel(self.zt, self.zv * 100, kind)
            np.testing.assert_array_equal(kt, changed)
            self.assertEqual(state["bandwidth"], other["bandwidth"])
            np.testing.assert_allclose(kt.mean(0), 0, atol=1e-13)
            self.assertGreaterEqual(float(np.linalg.eigvalsh(kt).min()), -1e-10)

    def test_interaction_psd_and_contrast_finite_difference(self):
        rng = np.random.default_rng(61)
        kg, kv, _ = site_kernel(self.zt, self.zv, "rbf")
        dt, dv = rng.normal(size=(45, 2)), rng.normal(size=(12, 2))
        alpha, low, high = (
            rng.normal(size=45),
            np.array([-0.5, -0.2]),
            np.array([0.5, 0.4]),
        )
        for form in ("linear", "spline"):
            ki, _, state = interaction_kernel(kg, kv, dt, dv, form)
            self.assertGreaterEqual(float(np.linalg.eigvalsh(ki).min()), -1e-10)
            count = len(state["scales"])

            def predicted(value):
                basis = (value[:count] - state["means"]) / state["scales"]
                return (
                    0.3
                    * (
                        kv
                        * (state["training_bases"] @ basis)[None, :]
                        / state["trace_scale"]
                    )
                    @ alpha
                )

            np.testing.assert_allclose(
                interaction_contrast(kv, state, high - low, alpha, 0.3),
                predicted(high) - predicted(low),
                atol=1e-12,
            )

    def test_screening_and_all_tree_backends_roundtrip(self):
        self.assertIn(0, learners.screen(self.ht, self.zt, self.y, 2))
        np.testing.assert_array_equal(
            learners.screen(self.ht, self.zt, self.y, 0), np.arange(7)
        )
        grids = dict(
            catboost=dict(depth=2, iterations=8, l2_leaf_reg=20.0),
            xgboost=dict(max_depth=2, n_estimators=8, min_child_weight=5),
            hgb=dict(max_leaf_nodes=7, max_iter=8, min_samples_leaf=10),
        )
        for algorithm, settings in grids.items():
            with self.subTest(algorithm=algorithm):
                fit = learners.tree_fit(
                    algorithm,
                    self.ht,
                    self.hv,
                    self.zt,
                    self.zv,
                    self.y,
                    dict(top_k=3, settings=settings),
                    19,
                    1,
                    True,
                )
                matrix = np.column_stack([self.hv, self.zv[:, fit["columns"]]])
                expected = (
                    fit["model"].predict(matrix) * fit["target_scale"]
                    + fit["target_mean"]
                )
                np.testing.assert_allclose(fit["prediction"], expected, atol=1e-12)

    def test_supervised_interaction_basis_learns_known_synthetic_direction(self):
        rng = np.random.default_rng(419)
        z, u = rng.normal(size=(500, 8)), rng.normal(size=500)
        residual = u * z[:, 0] * 3 + rng.normal(size=500) * 0.1
        state = learners.supervised_basis(z, np.column_stack([u, u**2]), residual, 2)
        scores = learners.apply_basis(z, state)
        self.assertEqual(scores.shape, (500, 2))
        self.assertGreater(abs(np.corrcoef(scores[:, 0], z[:, 0])[0, 1]), 0.9)
        zeros = learners.supervised_basis(
            z, np.column_stack([u, u**2]), np.zeros(500), 2
        )
        np.testing.assert_array_equal(
            learners.apply_basis(z, zeros), np.zeros((500, 2))
        )

    def test_fusion_is_convex_and_uses_prediction_information(self):
        y = np.linspace(-2, 2, 100)
        x = np.column_stack([y, -y, y + 1])
        weight = learners.blend_weights(x, y)
        self.assertTrue((weight >= 0).all())
        self.assertAlmostEqual(float(weight.sum()), 1.0)
        self.assertGreater(weight[0], 0.9)
        weight = learners.blend_weights(np.column_stack([y] * 4), y)
        np.testing.assert_allclose(weight, np.full(4, 0.25), atol=1e-8)
