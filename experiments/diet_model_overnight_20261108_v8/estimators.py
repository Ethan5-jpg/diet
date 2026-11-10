"""Bounded CPU estimators with explicit objectives and convergence diagnostics."""

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def fm_objective(theta, x, y, rank, penalty):
    p = x.shape[1]
    linear, factors = theta[1 : p + 1], theta[p + 1 :].reshape(p, rank)
    xv, squared = x @ factors, x * x
    error = theta[0] + x @ linear + 0.5 * ((xv * xv).sum(1) - squared @ (factors * factors).sum(1)) - y
    e = error / len(x)
    value = 0.5 * np.mean(error**2) + 0.5 * penalty * (linear @ linear + np.sum(factors**2))
    grad_v = x.T @ (e[:, None] * xv) - (squared.T @ e)[:, None] * factors + penalty * factors
    return float(value), np.r_[e.sum(), x.T @ e + penalty * linear, grad_v.ravel()]


class GuardedFM:
    def __init__(self, rank, penalty, iterations, seed):
        self.rank, self.penalty, self.iterations, self.seed = rank, penalty, iterations, seed

    def fit(self, x, y):
        self.scaler = StandardScaler().fit(x)
        z = self.scaler.transform(x)
        ridge = Ridge(alpha=len(z) * self.penalty).fit(z, y)
        factors = np.random.default_rng(self.seed).normal(
            0, 0.03 / np.sqrt(z.shape[1]), (z.shape[1], self.rank)
        )
        start = np.r_[ridge.intercept_, ridge.coef_, factors.ravel()]
        fit = minimize(
            fm_objective,
            start,
            args=(z, y, self.rank, self.penalty),
            jac=True,
            method="L-BFGS-B",
            options=dict(maxiter=self.iterations, ftol=1e-11, gtol=1e-6, maxls=40),
        )
        if not np.isfinite(fit.x).all():
            raise ValueError("Nonfinite FM optimization")
        self.theta = fit.x
        self.diagnostic = dict(
            converged=bool(fit.success),
            iterations=int(fit.nit),
            gradient_inf=float(np.max(abs(fit.jac))),
            objective=float(fit.fun),
            optimizer_message=str(fit.message),
        )
        return self

    def predict(self, x):
        z = self.scaler.transform(x)
        p = z.shape[1]
        v = self.theta[p + 1 :].reshape(p, self.rank)
        return self.theta[0] + z @ self.theta[1 : p + 1] + 0.5 * ((z @ v) ** 2 - (z * z) @ (v * v)).sum(1)


class GroupLasso:
    """Disjoint groups: squared loss + lambda sum ||b_g||_2 + ridge/2 ||b||^2."""

    def __init__(self, fraction, ridge=0.01, iterations=2500, tolerance=1e-6):
        self.fraction, self.ridge = fraction, ridge
        self.iterations, self.tolerance = iterations, tolerance

    @staticmethod
    def prox(value, threshold):
        norm = np.linalg.norm(value, axis=1, keepdims=True)
        return value * np.maximum(0.0, 1.0 - threshold / np.maximum(norm, 1e-30))

    def fit(self, x, y, width=2):
        if x.shape[1] % width:
            raise ValueError("Invalid group width")
        gram, rhs = x.T @ x / len(x), x.T @ y / len(x)
        gram += self.ridge * np.eye(gram.shape[0])
        lipschitz = max(float(np.linalg.eigvalsh(gram)[-1]), 1e-12)
        self.penalty = self.fraction * np.linalg.norm(rhs.reshape(-1, width), axis=1).max(initial=0)
        beta, momentum, t = np.zeros_like(rhs), np.zeros_like(rhs), 1.0
        for iteration in range(self.iterations):
            next_beta = self.prox(
                (momentum - (gram @ momentum - rhs) / lipschitz).reshape(-1, width), self.penalty / lipschitz
            ).ravel()
            next_t = (1 + np.sqrt(1 + 4 * t * t)) / 2
            next_momentum = next_beta + (t - 1) / next_t * (next_beta - beta)
            if np.dot(momentum - next_beta, next_beta - beta) > 0:
                next_t, next_momentum = 1.0, next_beta.copy()
            beta, momentum, t = next_beta, next_momentum, next_t
            if iteration % 10 == 0:
                stationary = (
                    beta
                    - self.prox(
                        (beta - (gram @ beta - rhs) / lipschitz).reshape(-1, width), self.penalty / lipschitz
                    ).ravel()
                )
                if lipschitz * np.max(abs(stationary), initial=0) <= self.tolerance:
                    break
        self.coef_ = beta
        stationary = (
            beta
            - self.prox(
                (beta - (gram @ beta - rhs) / lipschitz).reshape(-1, width), self.penalty / lipschitz
            ).ravel()
        )
        residual = float(lipschitz * np.max(abs(stationary), initial=0))
        self.diagnostic = dict(
            converged=residual <= self.tolerance,
            iterations=iteration + 1,
            proximal_residual=residual,
            active_groups=int(np.sum(np.linalg.norm(beta.reshape(-1, width), axis=1) > 1e-10)),
        )
        return self

    def predict(self, x):
        return x @ self.coef_


class LeafRules:
    """Supervised leaf indicators plus linear features; elastic-net aggregation."""

    def __init__(self, case, threads, seed):
        self.case, self.threads, self.seed = case, threads, seed

    def fit(self, x, y):
        c = self.case
        self.forest = ExtraTreesRegressor(
            n_estimators=c["trees"],
            max_depth=c["depth"],
            min_samples_leaf=c["leaf"],
            max_features=0.7,
            n_jobs=self.threads,
            random_state=self.seed,
        ).fit(x, y)
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True).fit(self.forest.apply(x))
        self.model = make_pipeline(
            StandardScaler(with_mean=False),
            ElasticNet(alpha=c["penalty"], l1_ratio=0.8, max_iter=4000, tol=1e-5, selection="cyclic"),
        )
        self.model.fit(self.design(x), y)
        return self

    def design(self, x):
        leaves = self.encoder.transform(self.forest.apply(x))
        return sparse.hstack([sparse.csr_matrix(x), leaves], format="csr")

    def predict(self, x):
        return self.model.predict(self.design(x))


class RotationBoost:
    def __init__(self, host_width, case, seed):
        self.host_width, self.case, self.seed = host_width, case, seed

    def fit(self, x, y):
        self.scaler = StandardScaler().fit(x)
        z = self.scaler.transform(x)
        self.models = []
        width = x.shape[1] - self.host_width
        for j in range(self.case["rotations"]):
            rng = np.random.default_rng(self.seed + j)
            rotation = np.linalg.qr(rng.normal(size=(width, width)))[0]
            a = np.column_stack([z[:, : self.host_width], z[:, self.host_width :] @ rotation])
            model = hgb(self.case, self.seed + j).fit(a, y)
            self.models.append((rotation, model))
        return self

    def predict(self, x):
        z = self.scaler.transform(x)
        return np.mean(
            [
                m.predict(np.column_stack([z[:, : self.host_width], z[:, self.host_width :] @ r]))
                for r, m in self.models
            ],
            axis=0,
        )


class ResidualNeighbors:
    def __init__(self, host_width, case, threads):
        self.host_width, self.case, self.threads = host_width, case, threads

    def fit(self, x, y):
        h, m = x[:, : self.host_width], x[:, self.host_width :]
        self.beta = np.linalg.lstsq(h, y, rcond=None)[0]
        self.scaler = StandardScaler().fit(m)
        self.model = KNeighborsRegressor(
            n_neighbors=min(self.case["neighbors"], len(y)),
            weights="distance",
            algorithm="brute",
            n_jobs=self.threads,
        )
        self.model.fit(self.scaler.transform(m), y - h @ self.beta)
        return self

    def predict(self, x):
        h, m = x[:, : self.host_width], x[:, self.host_width :]
        return h @ self.beta + self.case["shrinkage"] * self.model.predict(self.scaler.transform(m))


def hgb(case, seed):
    return HistGradientBoostingRegressor(
        max_iter=case["iterations"],
        max_leaf_nodes=case["leaves"],
        min_samples_leaf=case.get("leaf", 30),
        l2_regularization=case.get("l2", 30.0),
        learning_rate=0.04,
        early_stopping=False,
        random_state=seed,
    )


class AdditiveBoost:
    """H/D and site-specific blocks; diet is excluded from microbial blocks."""

    def __init__(self, case, seed):
        self.case, self.seed = case, seed

    def fit(self, blocks, y):
        self.intercept = float(y.mean())
        parts = [np.zeros(len(y)) for _ in blocks]
        self.models, self.centers = [None] * len(blocks), [0.0] * len(blocks)
        for _ in range(self.case["cycles"]):
            for j, x in enumerate(blocks):
                residual = y - self.intercept - sum(parts) + parts[j]
                model = hgb(self.case, self.seed).fit(x, residual)
                values = model.predict(x)
                self.models[j], self.centers[j] = model, float(values.mean())
                parts[j] = values - self.centers[j]
        return self

    def predict(self, blocks):
        return self.intercept + sum(m.predict(x) - c for m, x, c in zip(self.models, blocks, self.centers))
