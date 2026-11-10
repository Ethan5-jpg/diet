"""Minimal training-role encoding of the already validated joint context."""

import numpy as np
from sklearn.preprocessing import StandardScaler


class DietEncoder:
    def __init__(self, name, parameters=None):
        if name != "joint":
            raise ValueError("V13 uses the fixed joint nutrition context")

    def raw(self, extra, index, rows):
        x, cat = extra["blocks"]["joint"][rows], extra["alcohol"][rows]
        onehot = (
            np.column_stack([cat == c for c in self.categories])
            if self.categories
            else np.empty((len(rows), 0))
        )
        return np.column_stack(
            [
                np.where(np.isfinite(x), x, self.medians),
                ~np.isfinite(x),
                onehot,
                [v is None for v in cat],
                [v is not None and v not in self.categories for v in cat],
            ]
        ).astype(float)

    def fit(self, extra, index, rows):
        x = extra["blocks"]["joint"][rows]
        self.medians = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.0 for c in x.T])
        self.categories = sorted({v for v in extra["alcohol"][rows] if v is not None})
        self.scaler = StandardScaler().fit(self.raw(extra, index, rows))
        return self

    def transform(self, extra, index, rows):
        return self.scaler.transform(self.raw(extra, index, rows))
