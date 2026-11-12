"""Weighted community-kernel predictors and household scoring."""

import numpy as np


PENALTIES = {"pc": .03, "community": .03, "community_weak": .003, "community_strong": .3}
COMPARISONS = {
    "pc_vs_constant": ("pc", "constant"),
    "community_vs_constant": ("community", "constant"),
    "community_vs_pc": ("community", "pc"),
    "community_weak_vs_pc": ("community_weak", "pc"),
    "community_strong_vs_pc": ("community_strong", "pc"),
}


def clr_kernels(values, training):
    reference = values[training]
    centered = values - reference.mean(axis=0)
    linear = centered @ centered[training].T
    norm = (centered ** 2).sum(axis=1)
    distance = np.maximum(norm[:, None] + norm[training] - 2 * linear, 0)
    pairs = distance[training][np.triu_indices(len(reference), 1)]
    bandwidth = float(np.median(pairs))
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Nonpositive training CLR distance")
    return (linear, norm), (np.exp(-distance / bandwidth), np.ones(len(values)))


def centered_kernel(gram, diag, training, weights):
    mean = gram @ weights
    grand = weights @ mean[training]
    scale = weights @ (diag[training] - 2 * mean[training] + grand)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Nonpositive weighted kernel trace")
    return (gram - mean[:, None] - mean[training] + grand) / scale


def predict(geometry, info, response, training):
    source_info = info[training]
    weights = np.minimum(source_info, np.quantile(source_info, .95))
    weights /= weights.sum()
    mean = weights @ response[training]
    root = np.sqrt(weights)
    targets = root[:, None] * (response[training] - mean)
    predicted = {"constant": np.broadcast_to(mean, response.shape)}
    for name, components in geometry.items():
        kernel = sum(centered_kernel(g, d, training, weights) for g, d in components) / len(components)
        weighted = (root[:, None] * kernel[training]) * root
        if name == "community":
            eig, axes = np.linalg.eigh((weighted + weighted.T) / 2)
            eig = np.maximum(eig, 0)
            rhs = axes.T @ targets
            projection = (kernel * root) @ axes
            for candidate in ("community", "community_weak", "community_strong"):
                predicted[candidate] = mean + projection @ (rhs / (eig[:, None] + PENALTIES[candidate]))
        else:
            solution = np.linalg.solve(weighted + PENALTIES[name] * np.eye(len(weights)), targets)
            predicted[name] = mean + (kernel * root) @ solution
    return predicted


def household_totals(values, family, families):
    return np.column_stack([np.bincount(family, weights=column, minlength=families)
                            for column in np.column_stack(values).T])


def covariance_moments(early, late, family, families):
    return household_totals((np.ones(len(early)), early, late, early * late), family, families)


def covariance(moments):
    n, early, late, cross = moments.T
    return (cross - early * late / n) / (n - 1)


def estimate(values):
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite score or bootstrap interval")
    return dict(value=float(values[0]), interval=np.quantile(values[1:], [.025, .975]).tolist())


def summarize(totals, draws):
    result = {}
    for name in sorted(totals):
        target = draws @ totals[name]
        prefix, key = name.rsplit("__", 1)
        section = result.setdefault(prefix, {})
        if key == "covariance":
            observed = covariance(target[:, :4])
            reference = covariance(target[:, 4:8])
            pc = covariance(target[:, 8:12])
            community = covariance(target[:, 12:])
            section[key] = dict(observed=estimate(observed), pc_vs_constant=estimate(reference - pc),
                                community_vs_constant=estimate(reference - community),
                                community_vs_pc=estimate(pc - community))
        else:
            denominator, *loss = target.T
            if np.any(denominator <= 0):
                raise ValueError("Nonpositive prediction denominator")
            losses = dict(zip(("constant", *PENALTIES), loss))
            section[key] = {name: estimate((losses[ref] - losses[model]) / denominator)
                            for name, (model, ref) in COMPARISONS.items()}
    return result
