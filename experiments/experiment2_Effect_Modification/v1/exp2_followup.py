"""Training-only representations and estimators for the follow-up routes."""

import numpy as np
from scipy import linalg
from scipy.cluster.hierarchy import linkage
from sklearn.linear_model import ARDRegression


FOLLOWUP_FAMILIES = ("balance_curve", "two_part_curve", "interaction_pcr",
                     "dose_robust_curve", "local_dose_curve", "ard_curve")


def log_profiles(train, valid, indices):
    """Positive log proportions and presence, using the full profile as denominator."""
    blocks = []
    for raw in (train, valid):
        selected = raw[:, indices]
        positive = selected > 0
        logged = np.zeros(selected.shape)
        np.log(selected, out=logged, where=positive)
        logged = np.where(positive, logged - np.log(raw.sum(1, keepdims=True)), 0.0)
        blocks.append((logged, positive))
    return blocks


def balance_profiles(train, valid, limit=24):
    """Learn a binary partition of prevalent taxa and replay its log contrasts."""
    frequency = (train > 0).mean(0)
    candidates = np.flatnonzero(frequency >= 0.20)
    indices = candidates[np.argsort(-frequency[candidates], kind="stable")[:limit]]
    width = len(indices)
    if width < 2:
        return np.empty((len(train), 0)), np.empty((len(valid), 0))
    profiles = log_profiles(train, valid, indices)
    floor = np.quantile(profiles[0][0][profiles[0][1]], 0.01) - np.log(2)
    logs = [np.where(present, values, floor) for values, present in profiles]
    training_clr = logs[0] - logs[0].mean(1, keepdims=True)
    normalized = training_clr - training_clr.mean(0)
    normalized /= np.maximum(normalized.std(0), 1e-8)
    leaves = [[j] for j in range(width)]
    contrasts = np.zeros((width, width - 1))
    for j, (a, b, _, _) in enumerate(linkage(normalized.T, method="ward")):
        left, right = leaves[int(a)], leaves[int(b)]
        scale = np.sqrt(len(left) * len(right) / (len(left) + len(right)))
        contrasts[left, j] = scale / len(left)
        contrasts[right, j] = -scale / len(right)
        leaves.append(left + right)
    return tuple(values @ contrasts for values in logs)


def two_part_profiles(train, valid, limit=20):
    """Represent detection separately from centered positive abundance."""
    frequency = (train > 0).mean(0)
    candidates = np.flatnonzero(frequency >= 0.10)
    order = np.argsort(np.abs(frequency[candidates] - 0.5), kind="stable")
    profiles = log_profiles(train, valid, candidates[order[:limit]])
    values, present = profiles[0]
    center = values.sum(0) / present.sum(0)
    return tuple(np.column_stack([p, np.where(p, a - center, 0.0)])
                 for a, p in profiles)


def local_dose_features(diets):
    """Linear interpolation weights centered at the training median dose."""
    knots = np.unique(np.quantile(diets[0][:, 0], [0.10, 0.35, 0.65, 0.90]))
    weights = np.eye(len(knots))
    center = np.array([np.interp(0.0, knots, row) for row in weights])
    return tuple(np.column_stack([np.interp(d[:, 0], knots, row) for row in weights])
                 - center for d in diets)


def pcr_path(x, y, widths):
    """Use well-supported interaction directions with a fixed small ridge penalty."""
    u, singular, vt = linalg.svd(x, full_matrices=False, check_finite=False)
    usable = int(np.count_nonzero(singular > 1e-3 * singular[0]))
    if not usable:
        return [np.zeros(x.shape[1]) for _ in widths]
    coefficients = ((u[:, :usable].T @ y) * singular[:usable]
                    / (singular[:usable] ** 2 + 0.03 * len(x)))
    result = []
    for width in widths:
        use = min(width, usable)
        result.append(vt[:use].T @ coefficients[:use])
    return result


def dose_weights(dose):
    """Bound the leverage of unusually large training exposure deviations."""
    absolute = np.abs(dose - np.median(dose))
    cutoff = max(float(np.quantile(absolute, 0.85)), 1e-6)
    weights = (cutoff / np.maximum(cutoff, absolute)) ** 2
    return weights / weights.mean()


def ard_fit(x, y, threshold):
    """Empirical Bayes shrinkage on standardized interaction coefficients."""
    scale = float(np.sqrt(np.mean(y ** 2)))
    if scale <= 1e-10:
        return np.zeros(x.shape[1]), "fit"
    model = ARDRegression(max_iter=160, tol=5e-4, threshold_lambda=threshold,
                          fit_intercept=False)
    try:
        model.fit(x, y / scale)
    except linalg.LinAlgError:
        return np.zeros(x.shape[1]), "failed_ard_linear_algebra"
    if not np.isfinite(model.coef_).all():
        return np.zeros(x.shape[1]), "failed_ard_nonfinite"
    status = "ard_iteration_cap" if model.n_iter_ == model.max_iter else "fit"
    return model.coef_ * scale, status
