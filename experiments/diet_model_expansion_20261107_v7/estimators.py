"""Small structured predictors; all objectives are fit on the supplied train role."""

import numpy as np
from scipy.optimize import minimize
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def fm_value_gradient(theta, x, y, rank, penalty):
    p = x.shape[1]
    intercept, linear = theta[0], theta[1 : p + 1]
    factors = theta[p + 1 :].reshape(p, rank)
    xv, squared = x @ factors, x * x
    pred = intercept + x @ linear + 0.5 * ((xv * xv).sum(1) - squared @ (factors * factors).sum(1))
    error = pred - y
    value = 0.5 * np.mean(error * error) + 0.5 * penalty * (
        linear @ linear + np.sum(factors * factors)
    )
    e = error / len(x)
    grad_v = x.T @ (e[:, None] * xv) - (squared.T @ e)[:, None] * factors + penalty * factors
    grad = np.r_[e.sum(), x.T @ e + penalty * linear, grad_v.ravel()]
    return float(value), grad


class Factorization:
    def __init__(self, rank, penalty, iterations, seed):
        self.rank, self.penalty, self.iterations, self.seed = rank, penalty, iterations, seed

    def fit(self, x, y):
        self.scaler = StandardScaler().fit(x)
        z = self.scaler.transform(x)
        ridge = Ridge(alpha=len(z) * self.penalty).fit(z, y)
        factors = np.random.default_rng(self.seed).normal(
            0, 0.1 / np.sqrt(z.shape[1]), (z.shape[1], self.rank)
        )
        start = np.r_[ridge.intercept_, ridge.coef_, factors.ravel()]
        fitted = minimize(
            fm_value_gradient,
            start,
            args=(z, y, self.rank, self.penalty),
            jac=True,
            method="L-BFGS-B",
            options=dict(maxiter=self.iterations, ftol=1e-9),
        )
        if not np.isfinite(fitted.x).all():
            raise ValueError("Nonfinite factorization fit")
        self.theta, self.converged = fitted.x, bool(fitted.success)
        self.message = str(fitted.message)
        return self

    def predict(self, x):
        x = self.scaler.transform(x)
        p = x.shape[1]
        v = self.theta[p + 1 :].reshape(p, self.rank)
        return (
            self.theta[0]
            + x @ self.theta[1 : p + 1]
            + 0.5 * ((x @ v) ** 2 - (x * x) @ (v * v)).sum(1)
        )


class AdditiveBoost:
    """Backfitting across H/D, oral and gut blocks excludes D-by-microbe terms."""

    def __init__(self, cycles, iterations, leaves, seed):
        self.cycles, self.iterations, self.leaves, self.seed = cycles, iterations, leaves, seed

    def fit(self, blocks, y):
        self.intercept = float(np.mean(y))
        parts = [np.zeros(len(y)) for _ in blocks]
        self.models, self.centers = [None] * len(blocks), [0.0] * len(blocks)
        for _ in range(self.cycles):
            for j, x in enumerate(blocks):
                residual = y - self.intercept - sum(parts) + parts[j]
                model = HistGradientBoostingRegressor(
                    max_iter=self.iterations,
                    max_leaf_nodes=self.leaves,
                    min_samples_leaf=30,
                    l2_regularization=30.0,
                    learning_rate=0.04,
                    early_stopping=False,
                    random_state=self.seed,
                ).fit(x, residual)
                values = model.predict(x)
                self.models[j], self.centers[j] = model, float(values.mean())
                parts[j] = values - self.centers[j]
        return self

    def predict(self, blocks):
        return self.intercept + sum(
            m.predict(x) - c for m, x, c in zip(self.models, blocks, self.centers)
        )


class MixtureExperts:
    """Gut-gated, ridge-shrunk deviations around a global regression."""

    def __init__(self, clusters, penalty, seed):
        self.clusters, self.penalty, self.seed = clusters, penalty, seed

    def fit(self, x, g, y):
        self.scaler = StandardScaler().fit(x)
        z = self.scaler.transform(x)
        self.gate = GaussianMixture(
            n_components=self.clusters,
            covariance_type="diag",
            reg_covar=0.01,
            n_init=1,
            max_iter=100,
            random_state=self.seed,
        ).fit(g)
        weights = self.gate.predict_proba(g)
        self.base = Ridge(alpha=len(x) * self.penalty).fit(z, y)
        residual = y - self.base.predict(z)
        self.local = [
            Ridge(alpha=max(float(w.sum()), 1.0) * self.penalty).fit(z, residual, sample_weight=w)
            for w in weights.T
        ]
        return self

    def predict(self, x, g):
        z = self.scaler.transform(x)
        weights = self.gate.predict_proba(g)
        return self.base.predict(z) + sum(w * m.predict(z) for w, m in zip(weights.T, self.local))


class ResidualTrees:
    def __init__(self, host_width, leaf, trees, threads, seed):
        self.host_width, self.leaf, self.trees, self.threads, self.seed = (
            host_width,
            leaf,
            trees,
            threads,
            seed,
        )

    def fit(self, x, y):
        h = x[:, : self.host_width]
        self.beta = np.linalg.lstsq(h, y, rcond=None)[0]
        self.tree = ExtraTreesRegressor(
            n_estimators=self.trees,
            min_samples_leaf=self.leaf,
            max_features=0.6,
            n_jobs=self.threads,
            random_state=self.seed,
        ).fit(x[:, self.host_width :], y - h @ self.beta)
        return self

    def predict(self, x):
        return x[:, : self.host_width] @ self.beta + self.tree.predict(x[:, self.host_width :])


class PLSSVR:
    def __init__(self, host_width, components, c, seed):
        self.host_width, self.components, self.c, self.seed = host_width, components, c, seed

    def fit(self, x, y):
        h, z = x[:, : self.host_width], x[:, self.host_width :]
        residual = y - h @ np.linalg.lstsq(h, y, rcond=None)[0]
        self.pls = PLSRegression(
            n_components=min(self.components, z.shape[1], len(z) - 1),
            scale=True,
            max_iter=500,
            tol=1e-7,
        ).fit(z, residual)
        a = np.column_stack([h, self.pls.transform(z)])
        self.scaler = StandardScaler().fit(a)
        self.svr = SVR(C=self.c, epsilon=0.1, gamma="scale").fit(self.scaler.transform(a), y)
        return self

    def predict(self, x):
        a = np.column_stack([x[:, : self.host_width], self.pls.transform(x[:, self.host_width :])])
        return self.svr.predict(self.scaler.transform(a))
