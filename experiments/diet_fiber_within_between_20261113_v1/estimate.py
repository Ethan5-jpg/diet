"""Matched prediction and household inference for two fiber contrasts."""

import numpy as np


PENALTY = .03
ROUTES = ("baseline", "linear", "hinge")
CONTRASTS = {"full": [1., .5], "below": [.5, 0.], "above": [.5, .5], "above_minus_below": [0., .5]}


def group_mean(values, people, count):
    counts = np.bincount(people, minlength=count)
    return np.column_stack([np.bincount(people, weights=column, minlength=count) / counts for column in values.T])


def household_totals(matrix, household, count):
    return np.column_stack([np.bincount(household, weights=column, minlength=count) for column in matrix.T])


def projections(matrix, tol=1e-10):
    u, singular, axes = np.linalg.svd(matrix, full_matrices=False)
    keep = singular > tol * singular[0]
    return u[:, keep], axes[keep].T / singular[keep]


def infer(y, x, z, weight, family, count, absorbed):
    root = np.sqrt(weight)
    y = y * root
    z = z * root[:, None]
    nuisance, _ = projections(x * root[:, None])
    target = z - nuisance @ (nuisance.T @ z)
    outcome = y - nuisance @ (nuisance.T @ y)
    u, inverse = projections(target)
    if inverse.shape[1] != z.shape[1] or len(y) <= absorbed + nuisance.shape[1] + z.shape[1] or count < 2:
        raise ValueError("Insufficient rank or residual degrees of freedom for the fixed contrast")
    influence = u @ inverse.T
    beta = influence.T @ outcome
    residual = outcome - target @ beta
    cr1 = count / (count - 1) * (len(y) - 1) / (len(y) - absorbed - nuisance.shape[1] - z.shape[1])
    scores = household_totals(influence * residual[:, None], family, count) * np.sqrt(cr1)
    return beta, scores


def fold_losses(y, x, raw_interaction, weight, household, count, train, valid, absorbed=False):
    """The same weighted nuisance fit supplies every extension's background."""
    root = np.sqrt(weight[train])
    u, inv = projections(x[train] * root[:, None])
    fit_y = y[train] * root
    beta = inv @ (u.T @ fit_y)
    baseline = y[valid] - x[valid] @ beta
    center = np.average(y[train], weights=weight[train]) if not absorbed else 0.
    losses = [(y[valid] - center) ** 2, baseline ** 2]
    for width in (1, 2):
        z = raw_interaction[:, :width]
        target = z[train] * root[:, None]
        rotation = u.T @ target
        residual = target - u @ rotation
        gram = residual.T @ residual + PENALTY * weight[train].sum() * np.eye(width)
        effect = np.linalg.solve(gram, residual.T @ fit_y)
        correction = (z[valid] - x[valid] @ inv @ rotation) @ effect
        losses.append((baseline - correction) ** 2)
    return household_totals(np.column_stack(losses) * weight[valid, None], household[valid], count)


def interval(values):
    return dict(delta_r2=float(values[0]), ci95=np.quantile(values[1:], [.025, .975]).tolist())


def summarize(totals, weights):
    den, base, linear, hinge = (weights @ totals).T
    if not np.isfinite(den).all() or (den <= 0).any():
        raise ValueError("Degenerate household loss denominator")
    return dict(r2={name: float(1 - loss[0] / den[0]) for name, loss in zip(ROUTES, (base, linear, hinge))},
                linear_vs_baseline=interval((base - linear) / den),
                hinge_vs_baseline=interval((base - hinge) / den),
                hinge_vs_linear=interval((linear - hinge) / den))


def contrasts(within, between, iqr, critical):
    """Use aligned household scores for the within-minus-between covariance."""
    beta = dict(within=within[0], between=between[0], within_minus_between=within[0] - between[0])
    scores = dict(within=within[1], between=between[1], within_minus_between=within[1] - between[1])
    result = {}
    planned = CONTRASTS.items() if len(within[0]) == 2 else (("full", [1.]),)
    for name, vector in planned:
        vector = np.asarray(vector)[:len(within[0])] * iqr
        result[name] = {}
        for level in beta:
            effect = float(beta[level] @ vector)
            se = float(np.linalg.norm(scores[level] @ vector))
            result[name][level] = dict(mg_dl=effect, se=se, ci95=[effect - critical * se, effect + critical * se])
    return result
