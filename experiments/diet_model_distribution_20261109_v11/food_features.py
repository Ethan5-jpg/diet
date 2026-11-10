"""Train-role food shrinkage and person-weighted distribution representations."""

import numpy as np
from scipy.special import softmax
from sklearn.cluster import MiniBatchKMeans
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler

DEFAULTS = dict(
    seed=20261111,
    fourier_width=64,
    prototypes=12,
    directions=16,
    quantiles=[0.1, 0.3, 0.5, 0.7, 0.9],
    gamma=0.25,
)
UPF = ["UPF_lower_equalday_fraction", "UPF_upper_equalday_fraction"]


class DietEncoder:
    def __init__(self, name, parameters=None):
        self.name, self.parameters = name, dict(DEFAULTS, **(parameters or {}))

    def fit(self, extra, index, rows):
        self.index = None
        base = "full" if self.name == "full" else "joint"
        self.base = base
        if self.name.startswith("shrink"):
            x, n = extra["blocks"]["joint"][rows, :3], extra["joint_count"][rows]
            v = extra["joint_variance"][rows]
            self.prior_mean, self.tau, self.noise = [], [], []
            for j in range(3):
                valid = np.isfinite(x[:, j]) & np.isfinite(v[:, j]) & (n >= 3)
                if valid.sum() < 2:
                    mean, tau, noise = 0.0, 1e-8, 0.0
                else:
                    mean = x[valid, j].mean()
                    noise = np.average(v[valid, j], weights=n[valid] - 1)
                    tau = max(float(x[valid, j].var(ddof=1) - np.mean(v[valid, j] / n[valid])), 1e-8)
                self.prior_mean.append(mean)
                self.tau.append(tau)
                self.noise.append(noise)
            self.prior_mean, self.tau, self.noise = map(np.array, (self.prior_mean, self.tau, self.noise))
        if self.name in ("rff", "prototype", "sliced"):
            owners = extra["owners"]
            use = np.isin(owners, rows) & (extra["joint_count"][owners] >= 3)
            days, who = extra["days"][use], owners[use]
            if len(days) < 3:
                self.empty_basis = True
            else:
                self.empty_basis = False
                weights = 1 / extra["joint_count"][who]
                self.day_scaler = StandardScaler().fit(days, sample_weight=weights)
                z = self.day_scaler.transform(days)
                p = self.parameters
                if self.name == "rff":
                    self.basis = RBFSampler(
                        gamma=p["gamma"], n_components=p["fourier_width"], random_state=p["seed"]
                    ).fit(z)
                elif self.name == "prototype":
                    self.basis = MiniBatchKMeans(
                        n_clusters=min(p["prototypes"], len(z)),
                        batch_size=512,
                        max_iter=100,
                        n_init=3,
                        reassignment_ratio=0,
                        random_state=p["seed"],
                    ).fit(z, sample_weight=weights)
                else:
                    directions = np.random.default_rng(p["seed"]).normal(size=(4, p["directions"]))
                    self.basis = directions / np.linalg.norm(directions, axis=0)
        a = self.numerical(extra, index, rows)
        self.medians = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0 for c in a.T])
        self.categories = sorted({v for v in extra["alcohol"][rows] if v is not None})
        self.scaler = StandardScaler().fit(self.raw(extra, index, rows))
        return self

    def bag(self, extra, rows):
        p, name = self.parameters, self.name
        width = (
            p["fourier_width"]
            if name == "rff"
            else (p["prototypes"] if name == "prototype" else p["directions"] * len(p["quantiles"]))
        )
        out = np.full((len(rows), width), np.nan)
        if self.empty_basis:
            return out
        who = extra["owners"]
        use = np.isin(who, rows) & (extra["joint_count"][who] >= 3)
        if not use.any():
            return out
        owners, z = who[use], self.day_scaler.transform(extra["days"][use])
        if name == "sliced":
            projected = z @ self.basis
            for i, row in enumerate(rows):
                x = projected[owners == row]
                if len(x) >= 3:
                    out[i] = np.quantile(x, p["quantiles"], axis=0).ravel()
        else:
            values = self.basis.transform(z)
            if name == "prototype":
                values = softmax(-p["gamma"] * values**2, axis=1)
            valid = extra["joint_count"][rows] >= 3
            for j in range(values.shape[1]):
                sums = np.bincount(owners, weights=values[:, j], minlength=len(extra["joint_count"]))
                out[valid, j] = sums[rows[valid]] / extra["joint_count"][rows[valid]]
            if name == "prototype" and values.shape[1] < width:
                out[valid, values.shape[1] :] = 0
        return out

    def numerical(self, extra, index, rows):
        x = extra["blocks"][self.base][rows].copy()
        if self.name.startswith("shrink"):
            n = np.maximum(extra["joint_count"][rows], 1)[:, None]
            noise = np.broadcast_to(self.noise, (len(rows), 3))
            if self.name in ("shrink_local", "shrink_uncertainty"):
                noise = 0.5 * noise + 0.5 * extra["joint_variance"][rows]
            error = noise / n
            weight = self.tau / (self.tau + error)
            x[:, :3] = self.prior_mean + weight * (x[:, :3] - self.prior_mean)
            if self.name == "shrink_uncertainty":
                x = np.column_stack([x, np.sqrt(self.tau * error / (self.tau + error)), np.log1p(n[:, 0])])
        elif self.name == "quality":
            x = np.column_stack(
                [
                    x,
                    np.sqrt(extra["joint_variance"][rows] / np.maximum(extra["joint_count"][rows, None], 1)),
                    np.log1p(extra["joint_count"][rows]),
                ]
            )
        elif self.name == "joint_upf":
            a = index.iloc[rows][UPF].to_numpy(float, na_value=np.nan)
            finite = np.isfinite(a)
            if ((a[finite] < 0) | (a[finite] > 1)).any() or np.any(
                finite.all(1) & (a[:, 0] > a[:, 1] + 1e-10)
            ):
                raise ValueError("UPF bounds must be ordered fractions")
            x = np.column_stack([x, a])
        elif self.name == "covariance":
            x = np.column_stack([x, extra["covariance"][rows]])
        elif self.name in ("rff", "prototype", "sliced"):
            x = np.column_stack([x, self.bag(extra, rows)])
        return x

    def raw(self, extra, index, rows):
        x, cat = self.numerical(extra, index, rows), extra["alcohol"][rows]
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

    def transform(self, extra, index, rows):
        return self.scaler.transform(self.raw(extra, index, rows))
