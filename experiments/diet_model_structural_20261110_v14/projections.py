"""Training-only conditional nonlinear directions and regularized paired views."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from bridge import SIRDirections


class StablePCA(PCA):
    """Retain output width while zeroing numerically unsupported directions."""

    def fit(self, x, y=None):
        super().fit(x, y)
        self.active_ = self.explained_variance_ > max(float(self.explained_variance_.max()), 1.0) * 1e-12
        return self

    def transform(self, x):
        scores = super().transform(x)
        scores[:, ~self.active_] = 0
        return scores

    def fit_transform(self, x, y=None):
        return self.fit(x, y).transform(x)


def inverse_root(covariance, penalty):
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    return (vectors / np.sqrt(values.clip(0) + penalty)) @ vectors.T


class ConditionalPC:
    def __init__(self, components, seed):
        self.components, self.seed = components, seed

    def fit(self, x, nuisance):
        self.nuisance = np.linalg.lstsq(nuisance, x, rcond=None)[0]
        r = x - nuisance @ self.nuisance
        self.pca = StablePCA(
            n_components=max(1, min(self.components, len(x) - 1, x.shape[1])),
            svd_solver="full",
            random_state=self.seed,
        ).fit(r)
        self.scaler = StandardScaler().fit(self.pca.transform(r))
        return self

    def transform(self, x, nuisance):
        return self.scaler.transform(self.pca.transform(x - nuisance @ self.nuisance))


class FourierMap:
    def __init__(self, features, bandwidth, seed):
        self.features, self.bandwidth, self.seed = features, bandwidth, seed

    def fit(self, x):
        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(size=(x.shape[1], self.features)) / (np.sqrt(x.shape[1]) * self.bandwidth)
        self.offset = rng.uniform(0, 2 * np.pi, self.features)
        return self

    def transform(self, x):
        return np.sqrt(2 / self.features) * np.cos(x @ self.weights + self.offset)

    def gradient(self, x, coefficient):
        return (
            -np.sqrt(2 / self.features)
            * (np.sin(x @ self.weights + self.offset) * coefficient)
            @ self.weights.T
        )


class NonlinearDirections:
    def __init__(self, method, case, seed):
        self.method, self.case, self.seed = method, case, seed

    def fit(self, x, nuisance, y):
        c = self.case
        self.pc = ConditionalPC(c["pre_components"], self.seed).fit(x, nuisance)
        z = self.pc.transform(x, nuisance)
        self.fourier = FourierMap(c["rff_features"], c["bandwidth"], self.seed).fit(z)
        if self.method == "rff_sir":
            self.direction = SIRDirections("sir", c["components"], c["slices"]).fit(
                self.fourier.transform(z), nuisance, y
            )
            return self
        if self.method != "pilot_gradient":
            raise ValueError("Unsupported nonlinear projection")
        target = y - nuisance @ np.linalg.lstsq(nuisance, y, rcond=None)[0]
        self.pilot = Ridge(alpha=len(y) * c["pilot_penalty"]).fit(self.fourier.transform(z), target)
        gradient = self.fourier.gradient(z, self.pilot.coef_)
        values, vectors = np.linalg.eigh(gradient.T @ gradient / len(z))
        k = min(c["components"], z.shape[1])
        order = np.argsort(values)[::-1][:k]
        self.projection = vectors[:, order]
        self.projection[:, values[order] < max(float(values.max()), 1.0) * 1e-12] = 0
        self.scaler = StandardScaler().fit(z @ self.projection)
        return self

    def transform(self, x, nuisance):
        z = self.pc.transform(x, nuisance)
        if self.method == "rff_sir":
            return self.direction.transform(self.fourier.transform(z), nuisance)
        return self.scaler.transform(z @ self.projection)


class PairedViews:
    """Equal-width PC control or regularized CCA shared/difference/private coordinates."""

    def __init__(self, method, case, seed):
        self.method, self.case, self.seed = method, case, seed

    def fit(self, blocks, nuisance):
        c = self.case
        self.pcs = [
            ConditionalPC(c["pre_components"], self.seed + j).fit(x, nuisance) for j, x in enumerate(blocks)
        ]
        a, b = [pc.transform(x, nuisance) for pc, x in zip(self.pcs, blocks)]
        self.shared = min(c["components"], a.shape[1] // 2, b.shape[1] // 2)
        self.private = self.shared
        k = self.shared
        if k < 1:
            raise ValueError("Paired views need at least two variable input columns per site")
        if self.method == "cca":
            wa = inverse_root(a.T @ a / len(a), c["cca_penalty"])
            wb = inverse_root(b.T @ b / len(b), c["cca_penalty"])
            u, self.correlations, v = np.linalg.svd(wa @ (a.T @ b / len(a)) @ wb, full_matrices=False)
            self.axes = (wa @ u[:, :k], wb @ v.T[:, :k])
            self.private_axes = []
            for x, axis in zip((a, b), self.axes):
                scores = x @ axis
                removal = np.linalg.lstsq(scores, x, rcond=None)[0]
                residual = x - scores @ removal
                pca = StablePCA(n_components=k, svd_solver="full").fit(residual)
                self.private_axes.append((removal, pca))
        elif self.method != "paired_pc":
            raise ValueError("Unsupported paired projection")
        self.scaler = StandardScaler().fit(self.raw(a, b))
        return self

    def raw(self, a, b):
        k = self.shared
        if self.method == "paired_pc":
            return np.column_stack([a[:, : 2 * k], b[:, : 2 * k]])
        ca, cb = a @ self.axes[0], b @ self.axes[1]
        private = [
            pca.transform(x - score @ removal)
            for x, score, (removal, pca) in zip((a, b), (ca, cb), self.private_axes)
        ]
        return np.column_stack([(ca + cb) / np.sqrt(2), (ca - cb) / np.sqrt(2)] + private)

    def transform(self, blocks, nuisance):
        a, b = [pc.transform(x, nuisance) for pc, x in zip(self.pcs, blocks)]
        return self.scaler.transform(self.raw(a, b))
