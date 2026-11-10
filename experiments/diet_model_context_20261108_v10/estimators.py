"""Block-scaled ridge, additive spline ridge and additive kernel regression."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import SplineTransformer, StandardScaler


def slices(widths):
    ends = np.cumsum([0] + list(widths))
    return [slice(a, b) for a, b in zip(ends[:-1], ends[1:])]


class BlockRidge:
    def __init__(self, widths, labels, case, spline=False):
        self.widths, self.labels, self.case, self.spline = widths, labels, case, spline

    def fit(self, x, y):
        self.scaler = StandardScaler().fit(x)
        self.transforms, self.weights = [], []
        z = self.scaler.transform(x)
        parts = []
        for label, section in zip(self.labels, slices(self.widths)):
            a, transform = z[:, section], None
            if a.shape[1] and self.spline:
                # Constants/binary columns stay linear; uniform knots use training ranges.
                columns = np.array([len(np.unique(c)) > 4 for c in a.T])
                if columns.any():
                    transform = (
                        columns,
                        SplineTransformer(
                            n_knots=4, degree=2, knots="uniform", extrapolation="linear", include_bias=False
                        ).fit(a[:, columns]),
                    )
            a = self.expand(a, transform)
            weight = self.case.get("micro_weight", 1.0) if label in ("oral", "gut") else 1.0
            if label == "gut":
                weight *= self.case.get("gut_weight", 1.0)
            weight = np.sqrt(weight / max(a.shape[1], 1))
            self.transforms.append(transform)
            self.weights.append(weight)
            parts.append(a * weight)
        self.model = Ridge(alpha=len(y) * self.case["penalty"]).fit(np.column_stack(parts), y)
        return self

    @staticmethod
    def expand(a, transform):
        if transform is None:
            return a
        columns, spline = transform
        return np.column_stack([a[:, ~columns], spline.transform(a[:, columns])])

    def predict(self, x):
        z = self.scaler.transform(x)
        a = [
            self.expand(z[:, s], t) * w for s, t, w in zip(slices(self.widths), self.transforms, self.weights)
        ]
        return self.model.predict(np.column_stack(a))


class AdditiveKernel:
    """RBF(host/context) plus separately scaled linear microbial kernels."""

    def __init__(self, widths, case):
        self.widths, self.case = widths, case

    def kernel(self, a, b):
        width = sum(self.widths[:2])
        value = rbf_kernel(a[:, :width], b[:, :width], gamma=self.case.get("bandwidth", 1) / max(width, 1))
        for section in slices(self.widths)[2:]:
            count = section.stop - section.start
            value += self.case.get("micro_weight", 1) * (a[:, section] @ b[:, section].T) / max(count, 1)
        return value

    def fit(self, x, y):
        self.scaler = StandardScaler().fit(x)
        self.train = self.scaler.transform(x)
        gram = self.kernel(self.train, self.train)
        gram.flat[:: len(gram) + 1] += len(y) * self.case["penalty"]
        self.alpha = cho_solve(cho_factor(gram, lower=True, check_finite=False), y, check_finite=False)
        return self

    def predict(self, x):
        return self.kernel(self.scaler.transform(x), self.train) @ self.alpha
