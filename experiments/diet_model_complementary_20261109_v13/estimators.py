"""Paired hinges, diagonal-corrected sparse GP and graph coefficient smoothing."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class PairedHinges:
    """Bounded quantile-knot MARS-style search, not a full MARS package."""

    def __init__(self, case):
        self.case = case

    def basis(self, x):
        z = self.input.transform(x)
        columns = [np.ones(len(x))]
        for parent, j, k, sign in self.terms:
            columns.append(columns[parent] * np.maximum(sign * (z[:, j] - k), 0))
        return np.column_stack(columns)

    def fit(self, x, y):
        self.input = StandardScaler().fit(x)
        z = self.input.transform(x)
        self.terms = []
        basis = np.ones((len(x), 1))
        degrees = [0]
        used = [set()]
        knots = np.quantile(z, [0.2, 0.5, 0.8], axis=0).T
        while basis.shape[1] + 2 <= self.case["max_terms"]:
            u, s, _ = np.linalg.svd(basis, full_matrices=False)
            q = u[:, s > max(float(s.max()), 1.0) * 1e-10]
            residual = y - q @ (q.T @ y)
            best = None
            gain = 1e-8 * max(float(np.sum((y - y.mean()) ** 2)), 1.0)
            for parent, degree in enumerate(degrees):
                if degree >= self.case["degree"]:
                    continue
                eligible = [
                    j for j in range(z.shape[1]) if j not in used[parent] and np.unique(z[:, j]).size > 1
                ]
                if not eligible:
                    continue
                features = np.repeat(eligible, 3)
                cuts = knots[eligible].ravel()
                delta = z[:, features] - cuts
                support = ((delta > 0).sum(0) >= max(5, len(x) // 30)) & (
                    (delta < 0).sum(0) >= max(5, len(x) // 30)
                )
                if not support.any():
                    continue
                features, cuts, delta = features[support], cuts[support], delta[:, support]
                plus = basis[:, parent, None] * np.maximum(delta, 0)
                minus = basis[:, parent, None] * np.maximum(-delta, 0)
                a, b = plus - q @ (q.T @ plus), minus - q @ (q.T @ minus)
                aa, bb, ab = (a * a).sum(0), (b * b).sum(0), (a * b).sum(0)
                ta, tb = a.T @ residual, b.T @ residual
                jitter = 1e-10 * np.maximum(aa + bb, 1.0)
                aa, bb = aa + jitter, bb + jitter
                improvement = (bb * ta**2 + aa * tb**2 - 2 * ab * ta * tb) / np.maximum(
                    aa * bb - ab**2, 1e-30
                )
                j = int(np.argmax(improvement))
                if improvement[j] > gain:
                    gain = float(improvement[j])
                    best = (parent, int(features[j]), float(cuts[j]), plus[:, j], minus[:, j])
            if best is None:
                break
            parent, j, k, plus, minus = best
            self.terms.extend([(parent, j, k, 1.0), (parent, j, k, -1.0)])
            degrees.extend([degrees[parent] + 1] * 2)
            used.extend([used[parent] | {j}, used[parent] | {j}])
            basis = np.column_stack([basis, plus, minus])
        scaler = StandardScaler().fit(basis)
        expanded = scaler.transform(basis)

        def fit(columns):
            model = Ridge(alpha=len(y) * self.case["penalty"]).fit(expanded[:, columns], y)
            error = np.mean((y - model.predict(expanded[:, columns])) ** 2)
            effective = 1 + len(columns) + 1.5 * len(columns)
            gcv = error / max(1 - effective / len(y), 0.05) ** 2
            return model, float(gcv)

        active = list(range(1, basis.shape[1]))
        # Keep one constant, zero-valued design column as an intercept-only fallback.
        if not active:
            active = [0]
        self.model, best_score = fit([0])
        self.active = [0]
        model, score = fit(active)
        if score < best_score:
            self.active, self.model, best_score = active[:], model, score
        while len(active) > 1:
            choices = []
            for col in active:
                subset = [j for j in active if j != col]
                candidate, score = fit(subset)
                choices.append((score, col, subset, candidate))
            score, _, active, candidate = min(choices, key=lambda v: (v[0], v[1]))
            if score < best_score:
                self.active, self.model, best_score = active[:], candidate, score
        self.scaler, self.gcv = scaler, best_score
        return self

    def predict(self, x):
        return self.model.predict(self.scaler.transform(self.basis(x))[:, self.active])


class SparseGP:
    """Matern-3/2 FITC posterior mean; no calibrated uncertainty claim."""

    def __init__(self, case, widths, seed):
        self.case, self.widths, self.seed = case, widths, seed

    def kernel(self, x, y):
        r = np.sqrt(3) * cdist(x, y) / self.case["length"]
        return (1 + r) * np.exp(-r)

    def fit(self, x, y):
        self.scaler = StandardScaler().fit(x)
        self.weights = np.concatenate([np.full(w, 1 / np.sqrt(max(w, 1))) for w in self.widths])
        z = self.scaler.transform(x) * self.weights
        m = min(self.case["inducing"], len(x))
        self.centers = (
            z.copy()
            if m == len(x)
            else MiniBatchKMeans(
                n_clusters=m,
                n_init=3,
                max_iter=100,
                batch_size=512,
                reassignment_ratio=0,
                random_state=self.seed,
            )
            .fit(z)
            .cluster_centers_
        )
        kuu = self.kernel(self.centers, self.centers) + np.eye(m) * 1e-8
        kuf = self.kernel(self.centers, z)
        chol = cho_factor(kuu, lower=True, check_finite=False)
        projected = solve_triangular(chol[0], kuf, lower=True, check_finite=False)
        self.diagonal = self.case["noise"] + np.maximum(1 - (projected * projected).sum(0), 0)
        h = kuu + (kuf / self.diagonal) @ kuf.T
        self.mean = float(y.mean())
        self.alpha = cho_solve(
            cho_factor(h, lower=True, check_finite=False),
            kuf @ ((y - self.mean) / self.diagonal),
            check_finite=False,
        )
        return self

    def predict(self, x):
        z = self.scaler.transform(x) * self.weights
        return self.mean + self.kernel(z, self.centers) @ self.alpha


class GraphRidge:
    def __init__(self, case, widths, labels):
        self.case, self.widths, self.labels = case, widths, labels

    @staticmethod
    def laplacian(x):
        p = x.shape[1]
        corr = (x.T @ x / len(x)).clip(0)
        np.fill_diagonal(corr, 0)
        a = np.zeros((p, p))
        for i in range(p):
            keep = np.argsort(-corr[i], kind="stable")[: min(8, max(p - 1, 0))]
            a[i, keep] = corr[i, keep]
        a = np.maximum(a, a.T)
        degree = a.sum(1)
        inv = np.divide(1, np.sqrt(degree), out=np.zeros(p), where=degree > 1e-12)
        return np.diag((degree > 1e-12).astype(float)) - inv[:, None] * a * inv[None, :]

    def fit(self, x, y):
        self.scaler = StandardScaler().fit(x)
        z = self.scaler.transform(x)
        self.transforms = []
        start = 0
        for label, w in zip(self.labels, self.widths):
            prior = np.eye(w)
            if label in ("oral", "gut") and w:
                prior += self.case["graph_weight"] * self.laplacian(z[:, start : start + w])
            values, vectors = np.linalg.eigh(prior)
            transform = (vectors / np.sqrt(values.clip(1e-10))) @ vectors.T / np.sqrt(max(w, 1))
            self.transforms.append(transform)
            start += w
        self.model = Ridge(alpha=len(y) * self.case["penalty"]).fit(self.design(x), y)
        return self

    def design(self, x):
        z = self.scaler.transform(x)
        parts = []
        start = 0
        for width, transform in zip(self.widths, self.transforms):
            parts.append(z[:, start : start + width] @ transform)
            start += width
        return np.column_stack(parts)

    def predict(self, x):
        return self.model.predict(self.design(x))
