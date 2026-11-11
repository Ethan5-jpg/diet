"""Matched curve prediction and household-clustered interaction tests."""

import numpy as np
from scipy.stats import f


MODES = np.column_stack((
    np.array([1, 1, 1, 1]) / 2,
    np.array([-3, -1, 1, 3]) / np.sqrt(20),
    np.array([1, -1, -1, 1]) / 2,
    np.array([-1, 3, -3, 1]) / np.sqrt(20),
))
CONTRASTS = {
    "level": ("additive", "level"),
    "smooth_beyond_level": ("level", "smooth"),
    "unrestricted_beyond_level": ("level", "unrestricted"),
}
METRICS = {"curve": slice(None), "level": slice(0, 1), "shape": slice(1, None)}


def project(x, y, z, weights, model):
    _, u, inverse, sw, _ = model.fit_background(x, y[:, 0], weights)
    wz, wy = z * sw[:, None], y * sw[:, None]
    z_coordinates, y_coordinates = u.T @ wz, u.T @ wy
    return (inverse @ y_coordinates, inverse @ z_coordinates,
            wz - u @ z_coordinates, wy - u @ y_coordinates, u.shape[1])


def household_losses(y, x, z, weights, train, valid, family, groups, model):
    background, projection, zr, yr, _ = project(x[train], y[train], z[train], weights[train], model)
    gram = zr.T @ zr + .03 * weights[train].sum() * np.eye(zr.shape[1])
    beta = np.linalg.solve(gram, zr.T @ yr)
    residual = y[valid] - x[valid] @ background
    correction = (z[valid] - x[valid] @ projection) @ beta
    score = {"tss": y[valid] ** 2, "additive": residual ** 2}
    for key, width in (("level", 1), ("smooth", 3), ("unrestricted", 4)):
        after = residual.copy()
        after[:, :width] -= correction[:, :width]
        score[key] = after ** 2
    return {key: np.column_stack([
        np.bincount(family[valid], weights=weights[valid] * phase, minlength=groups) for phase in loss.T
    ]) for key, loss in score.items()}


def clustered_test(y, x, z, weights, family, n_people, model):
    _, _, zr, yr, rank_x = project(x, y, z, weights, model)
    beta, _, rank, _ = np.linalg.lstsq(zr, yr, rcond=1e-10)
    if rank != z.shape[1]:
        raise ValueError("Rank-deficient microbial interaction")
    error = yr - zr @ beta
    scores = np.einsum("ni,nj->nij", zr, error).reshape(len(z), -1)
    groups, inverse = np.unique(family, return_inverse=True)
    totals = np.zeros((len(groups), scores.shape[1]))
    np.add.at(totals, inverse, scores)
    bread = np.kron(np.linalg.inv(zr.T @ zr), np.eye(y.shape[1]))
    df = len(y) - n_people - rank_x - rank
    if df <= 0 or len(groups) <= beta.size:
        raise ValueError("Insufficient household or residual degrees of freedom")
    adjustment = len(groups) / (len(groups) - 1) * (len(y) - 1) / df
    covariance = adjustment * bread @ (totals.T @ totals) @ bread
    b = beta.ravel()
    stat = float(b @ np.linalg.solve(covariance, b) / len(b))
    return float(f.sf(stat, len(b), len(groups) - 1)), beta


def summarize(parts, draws):
    output = {}
    for metric, span in METRICS.items():
        denominator = parts["tss"][:, span].sum(axis=1)
        boot_denom = denominator[draws].sum(axis=1)
        if denominator.sum() <= 0 or not (boot_denom > 0).all():
            raise ValueError("Invalid curve prediction denominator")
        output[metric] = {}
        for contrast, (null, alt) in CONTRASTS.items():
            gain = (parts[null] - parts[alt])[:, span].sum(axis=1)
            output[metric][contrast] = dict(
                delta_r2=float(gain.sum() / denominator.sum()),
                ci95=np.quantile(gain[draws].sum(axis=1) / boot_denom, [.025, .975]).tolist())
    return output
