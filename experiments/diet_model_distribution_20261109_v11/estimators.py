"""Block-scaled ridge and training-only coefficient-reweighted ElasticNet."""

import numpy as np
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler


def slices(widths):
    ends = np.cumsum([0] + list(widths))
    return [slice(a, b) for a, b in zip(ends[:-1], ends[1:])]


class BlockRidge:
    def __init__(self, widths, labels, case):
        self.widths, self.labels, self.case = widths, labels, case

    def fit(self, x, y):
        self.scaler = StandardScaler().fit(x)
        self.weights = []
        for label, width in zip(self.labels, self.widths):
            weight = self.case.get("micro_weight", 1.0) if label in ("oral", "gut") else 1.0
            if label == "gut":
                weight *= self.case.get("gut_weight", 1.0)
            self.weights.append(np.sqrt(weight / max(width, 1)))
        self.model = Ridge(alpha=len(y) * self.case["penalty"]).fit(self.design(x), y)
        return self

    def design(self, x):
        z = self.scaler.transform(x)
        return np.column_stack([z[:, s] * w for s, w in zip(slices(self.widths), self.weights)])

    def predict(self, x):
        return self.model.predict(self.design(x))


class AdaptiveENet:
    """Training-only coefficient reweighting; no claim of causal variable selection."""

    def __init__(self, case):
        self.case = case

    def fit(self, x, y):
        self.scaler = StandardScaler().fit(x)
        z = self.scaler.transform(x)
        initial = Ridge(alpha=len(y) * 0.1).fit(z, y).coef_
        weights = (abs(initial) + 0.02) ** (-self.case["gamma"])
        self.weights = np.clip(weights / np.median(weights), 0.1, 10)
        self.model = ElasticNet(alpha=self.case["penalty"], l1_ratio=0.3, max_iter=10000, tol=1e-5).fit(
            z / self.weights, y
        )
        return self

    def predict(self, x):
        return self.model.predict(self.scaler.transform(x) / self.weights)
