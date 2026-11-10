"""Kernel algebra shared by every matched main-effect and interaction model."""

import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist


def finite_matrix(value):
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("Expected a finite matrix")
    return result


def metric(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    if (
        y.ndim != 1
        or prediction.shape != y.shape
        or not np.isfinite(y + prediction).all()
    ):
        raise ValueError("Invalid outcome/prediction vectors")
    variance = float(np.var(y))
    if variance <= 0:
        raise ValueError("Constant evaluation target")
    mse = float(np.mean((y - prediction) ** 2))
    return dict(n=len(y), R2=1.0 - mse / variance, MSE=mse, RMSE=float(np.sqrt(mse)))


def site_kernel(train, valid, kind, seed=20261107):
    """Centre/trace-normalise a kernel using training rows only.

    RBF bandwidth is the median positive squared distance among at most 256
    training rows selected independently of outcomes. Inputs are fold-local
    training-standardised CLR, not unscaled Aitchison coordinates.
    """
    train, valid = finite_matrix(train), finite_matrix(valid)
    if train.shape[1] != valid.shape[1] or not train.shape[1] or len(train) < 3:
        raise ValueError("Invalid microbial kernel shapes")
    bandwidth = None
    if kind == "linear":
        kt, kv = train @ train.T, valid @ train.T
    elif kind == "rbf":
        choose = np.random.default_rng(seed).choice(
            len(train), min(256, len(train)), replace=False
        )
        distances = cdist(train[choose], train[choose], "sqeuclidean")
        distances = distances[np.triu_indices(len(choose), 1)]
        positive = distances[distances > 1e-12]
        bandwidth = float(np.median(positive)) if len(positive) else 1.0
        kt = np.exp(-cdist(train, train, "sqeuclidean") / bandwidth)
        kv = np.exp(-cdist(valid, train, "sqeuclidean") / bandwidth)
    else:
        raise ValueError("Unknown kernel kind")
    mean = kt.mean(axis=0)
    grand = float(mean.mean())
    kt = kt - mean[:, None] - mean[None, :] + grand
    kv = kv - kv.mean(axis=1)[:, None] - mean[None, :] + grand
    scale = float(np.trace(kt) / len(kt))
    if scale < 1e-12:
        scale = 1.0
    return (
        kt / scale,
        kv / scale,
        dict(
            kind=kind,
            bandwidth=bandwidth,
            training_column_mean=mean,
            grand_mean=grand,
            trace_scale=scale,
        ),
    )


def interaction_kernel(kg_train, kg_valid, diet_train, diet_valid, form):
    """Product of the same gut kernel and centred training-scaled diet bases."""
    diet_train, diet_valid = finite_matrix(diet_train), finite_matrix(diet_valid)
    if (
        form not in ("linear", "spline")
        or diet_train.shape[1] != 2
        or diet_valid.shape[1] != 2
    ):
        raise ValueError("Expected linear/RCS diet columns")
    count = 1 if form == "linear" else 2
    means, scales = diet_train[:, :count].mean(0), diet_train[:, :count].std(0)
    scales[scales < 1e-12] = 1.0
    dt = (diet_train[:, :count] - means) / scales
    dv = (diet_valid[:, :count] - means) / scales
    kt = kg_train * (dt @ dt.T)
    kv = kg_valid * (dv @ dt.T)
    scale = float(np.trace(kt) / len(kt))
    if scale < 1e-12:
        scale = 1.0
    return (
        kt / scale,
        kv / scale,
        dict(
            form=form, means=means, scales=scales, training_bases=dt, trace_scale=scale
        ),
    )


def interaction_contrast(kg, state, delta_bases, alpha, weight):
    """Interaction component of a fixed diet contrast; not an intervention effect."""
    count = len(state["scales"])
    delta = np.asarray(delta_bases, float)[:count] / state["scales"]
    weights = state["training_bases"] @ delta
    return float(weight) * (kg @ (weights * alpha)) / state["trace_scale"]


class KernelProfile:
    """Unpenalised host basis plus an RKHS penalty at mean-loss scale.

    Objective: 0.5/n ||y-H beta-f||^2 + 0.5*lambda ||f||_K^2.
    The shared host projection is fitted exclusively on training rows.
    """

    def __init__(self, host_train, host_valid, kernel_train, kernel_valid, y):
        self.ht, self.hv = finite_matrix(host_train), finite_matrix(host_valid)
        self.kt, self.kv = finite_matrix(kernel_train), finite_matrix(kernel_valid)
        self.y = np.asarray(y, float)
        n = len(self.ht)
        if (
            self.kt.shape != (n, n)
            or self.kv.shape != (len(self.hv), n)
            or self.y.shape != (n,)
            or not np.isfinite(self.y).all()
        ):
            raise ValueError("Invalid kernel profile shapes")
        q, r, piv = linalg.qr(self.ht, mode="economic", pivoting=True)
        diagonal = abs(np.diag(r))
        rank = int(np.sum(diagonal > max(float(diagonal.max(initial=0)), 1.0) * 1e-10))
        self.keep = np.sort(piv[:rank])
        self.q = q[:, :rank]
        qk = self.q.T @ self.kt
        self.residual_kernel = (
            self.kt - self.q @ qk - qk.T @ self.q.T + self.q @ (qk @ self.q) @ self.q.T
        )
        self.residual_kernel = (self.residual_kernel + self.residual_kernel.T) * 0.5
        self.residual_y = self.y - self.q @ (self.q.T @ self.y)
        self.n = n

    def recover(self, alpha):
        alpha = np.asarray(alpha, float)
        alpha = alpha - self.q @ (self.q.T @ alpha)
        coefficient = np.zeros(self.ht.shape[1])
        coefficient[self.keep] = np.linalg.lstsq(
            self.ht[:, self.keep], self.y - self.kt @ alpha, rcond=None
        )[0]
        pred = self.hv @ coefficient + self.kv @ alpha
        if not np.isfinite(pred).all():
            raise ValueError("Nonfinite kernel prediction")
        return dict(prediction=pred, alpha=alpha, host_coef=coefficient)

    def path(self, penalties):
        if any(p <= 0 for p in penalties):
            raise ValueError("Positive main penalties required")
        values, vectors = linalg.eigh(self.residual_kernel, check_finite=False)
        if float(values.min(initial=0)) < -1e-6 * max(
            float(values.max(initial=0)), 1.0
        ):
            raise ValueError("Kernel is not positive semidefinite")
        values = np.maximum(values, 0)
        rhs = vectors.T @ self.residual_y
        return [
            self.recover(vectors @ (rhs / (values + self.n * p))) for p in penalties
        ]

    def fit(self, penalty):
        if penalty <= 0:
            raise ValueError("Positive main penalty required")
        matrix = self.residual_kernel.copy()
        matrix.flat[:: len(matrix) + 1] += self.n * penalty
        alpha = linalg.cho_solve(
            linalg.cho_factor(matrix, lower=True, check_finite=False),
            self.residual_y,
            check_finite=False,
        )
        return self.recover(alpha)
