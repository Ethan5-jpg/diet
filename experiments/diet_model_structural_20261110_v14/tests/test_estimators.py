"""Independent equations for local, kernel and graph models; serialization contracts."""

import pickle
import unittest
import warnings

import numpy as np

from estimators import ForestLocal, GraphSmooth, LocalRidge, MetricScale, NeuralKernel, SparseBasis, relu_ntk


class EstimatorChecks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1411)
        self.x = rng.normal(size=(60, 5))
        self.y = np.sin(self.x[:, 0]) + self.x[:, 1] + 0.1 * rng.normal(size=60)

    def test_local_ridge_agrees_with_independent_unpenalized_intercept_equations(self):
        model = LocalRidge(dict(neighbors=60, penalty=0.07)).fit(self.x, self.y)
        query = self.x[[0]] + 0.1
        z = model.metric.transform(query)[0]
        distance = np.linalg.norm(model.x - z, axis=1)
        w = np.exp(-0.5 * (distance / distance.max()) ** 2)
        w /= w.sum()
        design = np.column_stack([np.ones(len(self.x)), model.x - z])
        penalty = np.diag([0.0] + [0.07] * 5)
        solution = np.linalg.solve(design.T @ (w[:, None] * design) + penalty, design.T @ (w * self.y))
        self.assertAlmostEqual(model.predict(query)[0], solution[0], places=10)

    def test_training_inactive_columns_are_zero_for_training_and_unseen_queries(self):
        rng = np.random.default_rng(1412)
        x = np.column_stack([self.x, np.ones(len(self.x)), 1 + 1e-8 * rng.normal(size=len(self.x))])
        metric = MetricScale([5, 2]).fit(x)
        query = x[:4].copy()
        query[:, -2:] = 100
        np.testing.assert_array_equal(metric.transform(x)[:, -2:], 0)
        np.testing.assert_array_equal(metric.transform(query)[:, -2:], 0)

    def test_all_metric_heads_ignore_query_changes_in_training_inactive_columns(self):
        from models import regressor

        rng = np.random.default_rng(1413)
        x = np.column_stack([self.x, np.ones(len(self.x)), 1 + 1e-8 * rng.normal(size=len(self.x))])
        query = x[:6].copy()
        query[:, -2:] = -100
        cases = dict(
            local=dict(neighbors=30, penalty=0.1),
            forest_local=dict(trees=6, depth=3, leaf=4, penalty=0.1),
            ntk=dict(inducing=16, layers=2, penalty=0.1),
            diffusion=dict(neighbors=8, smoothing=1.0, trend_penalty=0.1),
            rbf_ard=dict(inducing=12, length=1.0, max_iter=100),
            svr=dict(C=1.0, epsilon=0.1, gamma_scale=1.0),
        )
        for name, case in cases.items():
            model = regressor(name, dict(case, widths=[5, 2]), 14, 1).fit(x, self.y)
            np.testing.assert_array_equal(model.predict(x[:6]), model.predict(query), err_msg=name)

    def test_ntk_is_psd_normalized_and_full_center_fit_matches_exact_centered_krr(self):
        model = NeuralKernel(dict(inducing=60, layers=2, penalty=0.02), 14).fit(self.x, self.y)
        z = model.metric.transform(self.x)
        gram = relu_ntk(z, z, 2)
        self.assertGreater(np.linalg.eigvalsh(gram).min(), -1e-10)
        np.testing.assert_allclose(np.diag(gram), 1, atol=1e-8)
        center = np.eye(len(z)) - np.ones((len(z), len(z))) / len(z)
        alpha = np.linalg.solve(
            center @ gram @ center + len(z) * 0.02 * np.eye(len(z)), self.y - self.y.mean()
        )
        query = model.metric.transform(self.x[:7] + 0.15)
        cross = relu_ntk(query, z, 2)
        expected = self.y.mean() + (cross - cross.mean(1, keepdims=True) - gram.mean(0) + gram.mean()) @ alpha
        np.testing.assert_allclose(model.predict(self.x[:7] + 0.15), expected, atol=1e-8)

    def test_graph_smoother_matches_dense_equation_and_is_inductive(self):
        model = GraphSmooth(dict(neighbors=7, smoothing=1.3, trend_penalty=0.1)).fit(self.x, self.y)
        residual = self.y - model.trend.predict(model.x)
        exact = np.linalg.solve(np.eye(60) + 1.3 * model.laplacian.toarray(), residual)
        np.testing.assert_allclose(model.smooth, exact, atol=1e-10)
        self.assertAlmostEqual(np.max(abs(model.laplacian @ np.ones(60))), 0, places=10)
        before = model.smooth.copy()
        model.predict(self.x[:4] + 100)
        np.testing.assert_array_equal(before, model.smooth)

    def test_forest_partitions_separate_households_and_leaf_kernel_is_normalized(self):
        groups = np.repeat(np.arange(30), 2)
        c = dict(trees=8, depth=3, leaf=3, penalty=0.1)
        model = ForestLocal(c, 14).fit(self.x, self.y, groups)
        for structure, estimate in model.partitions:
            self.assertFalse(set(groups[structure]) & set(groups[estimate]))
            self.assertEqual(len(structure) + len(estimate), len(groups))
        weights = model.kernel_weights(model.x[:7])
        self.assertTrue((weights >= 0).all())
        np.testing.assert_allclose(weights.sum(1), 1, atol=1e-12)
        first = model.forests[0][0]
        changed = self.y.copy()
        changed[model.partitions[0][1]] += 100
        again = ForestLocal(c, 14).fit(self.x, changed, groups)
        np.testing.assert_array_equal(first.apply(model.x), again.forests[0][0].apply(again.x))

    def test_all_new_regressors_preserve_constant_response_and_pickle(self):
        cases = [
            LocalRidge(dict(neighbors=20, penalty=0.1)),
            ForestLocal(dict(trees=6, depth=3, leaf=5, penalty=0.1)),
            NeuralKernel(dict(inducing=16, layers=2, penalty=0.1)),
            GraphSmooth(dict(neighbors=8, smoothing=1, trend_penalty=0.1)),
            SparseBasis(dict(inducing=12, length=1.0, max_iter=100)),
        ]
        for model in cases:
            model.fit(self.x, np.full(60, 2.5))
            np.testing.assert_allclose(model.predict(self.x[:5]), 2.5, atol=1e-8)
            copied = pickle.loads(pickle.dumps(model))
            np.testing.assert_array_equal(model.predict(self.x[:5]), copied.predict(self.x[:5]))

    def test_sparse_basis_iteration_limit_emits_a_capturable_fit_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = SparseBasis(dict(inducing=16, length=1.0, max_iter=1)).fit(self.x, self.y)
        self.assertEqual(model.model.n_iter_, 1)
        self.assertTrue(any("iteration limit" in str(w.message) for w in caught))
