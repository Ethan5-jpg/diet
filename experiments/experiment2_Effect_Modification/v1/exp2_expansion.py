"""Feature maps and solvers for the six prespecified expansion families."""

import numpy as np
from scipy import linalg
from sklearn.cluster import KMeans


NEW_FAMILIES = ("adaptive_sparse", "kernel_mix_curve", "rank_curve", "component_curve",
                "soft_cluster", "detection_slope")
NEW_CURVES = ("kernel_mix_curve", "rank_curve", "component_curve")


class RankConvergenceError(RuntimeError):
    """The nuclear-norm fit did not satisfy the proximal convergence criterion."""


def detection_features(train, valid, limit):
    """Training-ranked detection indicators; a zero read is not biological absence."""
    a, b = train > 0, valid > 0
    frequency = a.mean(axis=0)
    eligible = np.flatnonzero((frequency >= 0.05) & (frequency <= 0.95))
    order = np.lexsort((eligible, np.abs(frequency[eligible] - 0.5)))
    use = eligible[order[:limit]]
    return a[:, use].astype(float), b[:, use].astype(float)


def soft_profiles(train, valid, seed):
    """Six unsupervised centers; convert distances to continuous memberships."""
    n = min(6, len(np.unique(train, axis=0)))
    if n <= 1 or train.shape[1] == 0:
        return np.empty((len(train), 0)), np.empty((len(valid), 0))
    model = KMeans(n_clusters=n, n_init=5, algorithm="lloyd", random_state=seed).fit(train)
    dtrain, dvalid = (model.transform(x) ** 2 for x in (train, valid))
    nearest = dtrain.min(axis=1)
    positive = nearest[nearest > 1e-12]
    if not len(positive):
        # Distinct point masses at the centers have no within-group variance.
        between = linalg.norm(model.cluster_centers_[:, None] - model.cluster_centers_, axis=2) ** 2
        positive = between[between > 1e-12]
    scale = max(float(np.median(positive)) if len(positive) else 1.0, 1e-8)

    def membership(dist):
        strength = np.exp(-(dist - dist.min(axis=1, keepdims=True)) / scale)
        return strength / strength.sum(axis=1, keepdims=True)

    return membership(dtrain), membership(dvalid)


def adaptive_weights(pilot):
    """Inverse pilot magnitude with a training floor and bounded penalty ratios."""
    magnitude = np.abs(pilot)
    scale = float(np.sqrt(np.mean(magnitude ** 2)))
    if scale <= 1e-10:
        return np.ones_like(pilot)
    return np.clip(scale / np.maximum(magnitude, 0.10 * scale), 0.25, 4.0)


def component_path(x, y, steps):
    """Squared-loss componentwise boosting with a fixed 0.10 shrinkage."""
    gram = (x.T @ x) / len(x)
    gradient = (x.T @ y) / len(x)
    diagonal = gram.diagonal()
    eligible = diagonal > 1e-12
    beta, stored = np.zeros(x.shape[1]), {}
    wanted = set(steps)
    for j in range(1, max(wanted) + 1):
        gain = np.divide(gradient ** 2, diagonal, out=np.zeros_like(diagonal), where=eligible)
        best = int(np.argmax(gain))
        if gain[best] <= 1e-14:
            for step in wanted.difference(stored):
                stored[step] = beta.copy()
            break
        update = 0.10 * gradient[best] / diagonal[best]
        beta[best] += update
        gradient -= gram[:, best] * update
        if j in wanted:
            stored[j] = beta.copy()
    return [stored[step] for step in steps]


def rank_curve_fit(x, y, fraction):
    """Nuclear norm plus a small L2 penalty on a 2-by-PC coefficient matrix."""
    width = x.shape[1] // 2
    if not width or x.shape[1] != 2 * width:
        raise ValueError("Low-rank curve requires two complete dose blocks")
    gram = x.T @ x / len(x)
    gram.flat[::len(gram) + 1] += 0.005
    cross = (x.T @ y) / len(x)
    max_penalty = linalg.norm(cross.reshape(2, width), ord=2)
    if max_penalty <= 1e-12:
        return np.zeros(x.shape[1])
    lipschitz = linalg.eigvalsh(gram, subset_by_index=[len(gram) - 1, len(gram) - 1])[0]
    cutoff = fraction * max_penalty / lipschitz

    def prox(value):
        u, s, v = linalg.svd(value.reshape(2, width), full_matrices=False, check_finite=False)
        return ((u * np.maximum(s - cutoff, 0.0)) @ v).ravel()

    b = accelerated = np.zeros(x.shape[1])
    momentum = 1.0
    for _ in range(800):
        candidate = prox(accelerated + (cross - gram @ accelerated) / lipschitz)
        fixed_point = prox(candidate + (cross - gram @ candidate) / lipschitz)
        if linalg.norm(candidate - fixed_point) * lipschitz <= 1e-6 * max(1.0, max_penalty):
            return candidate
        next_momentum = (1 + np.sqrt(1 + 4 * momentum ** 2)) / 2
        accelerated = candidate + (momentum - 1) / next_momentum * (candidate - b)
        b, momentum = candidate, next_momentum
    raise RankConvergenceError("failed_rank_curve_convergence")
