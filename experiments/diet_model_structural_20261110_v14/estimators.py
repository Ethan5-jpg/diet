"""Finite-budget local, neural-kernel, graph and sparse-Bayesian regression."""

import warnings

import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import spsolve
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ARDRegression, Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class MetricScale:
    def __init__(self, widths=None):
        self.widths = widths

    def fit(self, x):
        self.scaler = StandardScaler().fit(x)
        widths = self.widths or [x.shape[1]]
        if sum(widths) != x.shape[1]:
            raise ValueError("Metric block widths differ")
        active = self.scaler.var_ > 1e-12
        self.active = active
        weights, offset = [], 0
        for width in widths:
            weights.extend([1 / np.sqrt(max(int(active[offset : offset + width].sum()), 1))] * width)
            offset += width
        self.weights, self.blocks = np.array(weights), len(widths)
        return self

    def transform(self, x):
        scaled = self.scaler.transform(x) * self.weights
        scaled[:, ~self.active] = 0.0
        return scaled


def local_value(x, y, query, weights, penalty):
    weights = weights / max(float(weights.sum()), 1e-15)
    mean_x, mean_y = weights @ x, float(weights @ y)
    centered = x - mean_x
    gram = centered.T @ (weights[:, None] * centered) + penalty * np.eye(x.shape[1])
    beta = np.linalg.solve(gram, centered.T @ (weights * (y - mean_y)))
    return float(mean_y + (query - mean_x) @ beta)


class LocalRidge:
    def __init__(self, case, seed=0, threads=1):
        self.case, self.seed, self.threads = case, seed, threads

    def fit(self, x, y):
        self.metric = MetricScale(self.case.get("widths")).fit(x)
        self.x, self.y = self.metric.transform(x), np.asarray(y).copy()
        self.neighbors = NearestNeighbors(n_neighbors=min(self.case["neighbors"], len(x)), n_jobs=1).fit(
            self.x
        )
        return self

    def predict(self, x):
        query = self.metric.transform(x)
        distance, index = self.neighbors.kneighbors(query)
        predictions = []
        for q, d, ix in zip(query, distance, index):
            weight = np.exp(-0.5 * (d / max(float(d[-1]), 1e-12)) ** 2)
            predictions.append(local_value(self.x[ix], self.y[ix], q, weight, self.case["penalty"]))
        return np.array(predictions)


class ForestLocal:
    """Partition/leaf labels are group-split; upstream supervised projections are not honest."""

    def __init__(self, case, seed=0, threads=1):
        self.case, self.seed, self.threads = case, seed, threads

    def fit(self, x, y, groups=None):
        self.metric = MetricScale(self.case.get("widths")).fit(x)
        self.x, self.y = self.metric.transform(x), np.asarray(y).copy()
        groups = np.arange(len(x)) if groups is None else np.asarray(groups)
        unique = np.unique(groups)
        if len(unique) < 2:
            raise ValueError("Forest partition and estimation need distinct households")
        unique = np.random.default_rng(self.seed).permutation(unique)
        mask = np.isin(groups, unique[: max(1, len(unique) // 2)])
        self.forests, self.partitions = [], []
        for j, structure in enumerate((mask, ~mask)):
            train, estimate = np.flatnonzero(structure), np.flatnonzero(~structure)
            forest = RandomForestRegressor(
                n_estimators=self.case["trees"],
                max_depth=self.case["depth"],
                min_samples_leaf=self.case["leaf"],
                max_features=0.8,
                random_state=self.seed + j,
                n_jobs=self.threads,
            )
            forest.fit(self.x[train], self.y[train])
            self.forests.append((forest, estimate, forest.apply(self.x[estimate])))
            self.partitions.append((train, estimate))
        return self

    def kernel_weights(self, query):
        weights = np.zeros((len(query), len(self.x)))
        for forest, estimate, leaves in self.forests:
            at = forest.apply(query)
            for tree in range(at.shape[1]):
                for leaf in np.unique(at[:, tree]):
                    qi = np.flatnonzero(at[:, tree] == leaf)
                    members = estimate[leaves[:, tree] == leaf]
                    if not len(members):
                        members = estimate
                    weights[np.ix_(qi, members)] += 1 / (2 * at.shape[1] * len(members))
        return weights

    def predict(self, x):
        out = []
        for start in range(0, len(x), 128):
            query = self.metric.transform(x[start : start + 128])
            for q, w in zip(query, self.kernel_weights(query)):
                keep = w > 0
                out.append(local_value(self.x[keep], self.y[keep], q, w[keep], self.case["penalty"]))
        return np.array(out)


def relu_ntk(x, z, layers):
    """Normalized fully connected ReLU NTK; constant input coordinate supplies bias."""
    a, b = np.column_stack([x, np.ones(len(x))]), np.column_stack([z, np.ones(len(z))])
    norm = np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :]
    covariance = np.clip(a @ b.T / norm, -1, 1)
    tangent = covariance.copy()
    for _ in range(layers):
        angle = np.arccos(covariance)
        derivative = (np.pi - angle) / np.pi
        covariance = (np.sin(angle) + (np.pi - angle) * covariance) / np.pi
        tangent = tangent * derivative + covariance
        covariance = np.clip(covariance, -1, 1)
    return tangent / (layers + 1)


class NeuralKernel:
    def __init__(self, case, seed=0, threads=1):
        self.case, self.seed, self.threads = case, seed, threads

    def fit(self, x, y):
        self.metric = MetricScale(self.case.get("widths")).fit(x)
        z = self.metric.transform(x)
        count = min(self.case["inducing"], len(x))
        ix = (
            np.arange(len(x))
            if count == len(x)
            else np.random.default_rng(self.seed).choice(len(x), count, replace=False)
        )
        self.centers = z[ix].copy()
        gram = relu_ntk(self.centers, self.centers, self.case["layers"])
        values, vectors = np.linalg.eigh((gram + gram.T) / 2)
        keep = values > max(float(values.max()), 1.0) * 1e-10
        self.mapping = vectors[:, keep] / np.sqrt(values[keep])
        self.model = Ridge(alpha=len(x) * self.case["penalty"]).fit(self.basis(z), y)
        return self

    def basis(self, z):
        return relu_ntk(z, self.centers, self.case["layers"]) @ self.mapping

    def predict(self, x):
        return self.model.predict(self.basis(self.metric.transform(x)))


class GraphSmooth:
    def __init__(self, case, seed=0, threads=1):
        self.case, self.seed, self.threads = case, seed, threads

    def fit(self, x, y):
        self.metric = MetricScale(self.case.get("widths")).fit(x)
        self.x = self.metric.transform(x)
        self.trend = Ridge(alpha=len(x) * self.case["trend_penalty"]).fit(self.x, y)
        k = min(self.case["neighbors"], max(len(x) - 1, 1))
        self.neighbors = NearestNeighbors(n_neighbors=min(k + 1, len(x)), n_jobs=1).fit(self.x)
        distance, index = self.neighbors.kneighbors(self.x)
        row, column, values = [], [], []
        positive = distance[distance > 1e-12]
        self.bandwidth = max(float(np.median(positive)) if len(positive) else 1.0, 1e-12)
        for i, (d, ix) in enumerate(zip(distance, index)):
            keep = ix != i
            row.extend([i] * int(keep.sum()))
            column.extend(ix[keep])
            values.extend(np.exp(-0.5 * (d[keep] / self.bandwidth) ** 2))
        graph = sparse.csr_matrix((values, (row, column)), shape=(len(x), len(x)))
        graph = (graph + graph.T) * 0.5
        degree = np.asarray(graph.sum(1)).ravel()
        self.laplacian = (sparse.diags(degree) - graph) / max(float(degree.mean()), 1e-12)
        residual = y - self.trend.predict(self.x)
        self.smooth = spsolve(
            (sparse.eye(len(x)) + self.case["smoothing"] * self.laplacian).tocsc(), residual
        )
        if not np.isfinite(self.smooth).all():
            raise ValueError("Graph residual solve is nonfinite")
        return self

    def predict(self, x):
        z = self.metric.transform(x)
        distance, index = self.neighbors.kneighbors(z, n_neighbors=min(self.case["neighbors"], len(self.x)))
        exponent = -0.5 * (distance / self.bandwidth) ** 2
        weights = np.exp(exponent - exponent.max(1, keepdims=True))
        weights /= weights.sum(1, keepdims=True)
        return self.trend.predict(z) + np.sum(weights * self.smooth[index], axis=1)


class SparseBasis:
    def __init__(self, case, seed=0, threads=1):
        self.case, self.seed, self.threads = case, seed, threads

    def fit(self, x, y):
        self.metric = MetricScale(self.case.get("widths")).fit(x)
        z = self.metric.transform(x)
        ix = np.random.default_rng(self.seed).choice(
            len(x), min(len(x), self.case["inducing"]), replace=False
        )
        self.centers = z[ix].copy()
        self.scaler = StandardScaler().fit(self.basis(z))
        self.model = ARDRegression(max_iter=self.case["max_iter"], tol=1e-3, threshold_lambda=10000)
        self.model.fit(self.scaler.transform(self.basis(z)), y)
        if self.model.n_iter_ >= self.case["max_iter"]:
            warnings.warn("Sparse Bayesian basis reached iteration limit", ConvergenceWarning)
        return self

    def basis(self, z):
        distance = cdist(z, self.centers, "sqeuclidean")
        return np.column_stack([z, np.exp(-distance / (2 * self.metric.blocks * self.case["length"] ** 2))])

    def predict(self, x):
        return self.model.predict(self.scaler.transform(self.basis(self.metric.transform(x))))
