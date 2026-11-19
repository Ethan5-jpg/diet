"""Paired no-diet arm and training-only structured interaction fits."""

from dataclasses import dataclass

import numpy as np

import protocol as p
import v4_design as d
from models import create_design, dose_basis, products, standardize
from source import check, clean_outcome
from v3_models import Forecast, forecasts as previous_forecasts, views_for_split


@dataclass(frozen=True)
class Forecasts:
    old: Forecast
    correction: np.ndarray
    selectable: np.ndarray
    no_diet: np.ndarray | None


def no_diet_main(study, rows, n, main, dose):
    # v1 orders the design as intercept, three host terms, smoking,
    # dose basis, dose x age/sex, ecology, PCs and marker main effects.
    levels = set(study.smoking[rows[:n]]) | {"__UNKNOWN__"}
    start = 4 + len(levels) - 1
    diet = np.column_stack((dose, products(dose, main[:, 1:3])))
    end = start + diet.shape[1]
    check(np.allclose(main[:, start:end], diet, rtol=0., atol=1e-12), "V4_DIET_BLOCK_MISALIGNED")
    return np.column_stack((main[:, :start], main[:, end:]))


def robust_weights(residual):
    """One fixed Huber-style weight from the training-only main residual."""
    med = np.median(residual)
    scale = 1.4826 * np.median(abs(residual-med))
    if scale <= p.RANK_TOL:
        scale = float(residual.std())
    if scale <= p.RANK_TOL:
        return np.ones(len(residual))
    distance = abs(residual-med)
    cutoff = 1.345 * scale
    return cutoff / np.maximum(cutoff, distance)


def ridge_forecast(z, y, test, penalty, weights=None):
    if weights is not None:
        root = np.sqrt(weights)
        y, z = y*root, z*root[:, None]
        scale = float(weights.sum())
    else:
        scale = len(y)
    beta = np.linalg.solve(z.T @ z / scale + np.diag(penalty), z.T @ y / scale)
    return test @ beta


def forecasts(study, exposure, outcomes, fold, *, with_no_diet=False):
    old = previous_forecasts(study, exposure, outcomes, fold)
    raw, views = views_for_split(study, exposure, fold)
    role = study.backgrounds[fold].role
    results = {}
    for endpoint, original in old.items():
        outcome = outcomes[endpoint]
        y = clean_outcome(outcome.y, role)
        tr = np.flatnonzero(role & study.eligible(exposure, endpoint, fold, y))
        rows = np.r_[tr, original.rows]
        n = len(tr)
        check(n == original.counts["training_people"] and np.array_equal(y[original.rows], original.y),
              "V4_FORECAST_ROWS_DIFFER")
        main = create_design(study, exposure, fold, rows, n).main
        inv = np.linalg.pinv(main[:n], rcond=p.RANK_TOL)
        resid = y[tr] - main[:n] @ (inv @ y[tr])
        dose = dose_basis(raw[rows], n)[0]
        nodiet = None
        if with_no_diet:
            plain = no_diet_main(study, rows, n, main, dose)
            nodiet = plain[n:] @ (np.linalg.pinv(plain[:n], rcond=p.RANK_TOL) @ y[tr])
        dose = standardize(dose, n)[0]
        bases = {}
        for name in d.VIEWS:
            marker = standardize(views[name][rows], n)[0]
            z = products(dose, marker)
            bases[name] = z - main @ (inv @ z[:n]), marker.shape[1]
        w = robust_weights(resid)
        extra = []
        for view, penalty, ratio in d.CURVES:
            z, k = bases[view]
            charges = np.r_[np.full(k, penalty), np.full(k, penalty*ratio)]
            extra.append(ridge_forecast(z[:n], resid, z[n:], charges))
        for view, shape, penalty in d.ROBUST:
            z, k = bases[view]
            if shape == "linear":
                z = z[:, :k]
            extra.append(ridge_forecast(z[:n], resid, z[n:], np.full(z.shape[1], penalty), w))
        matrix = np.column_stack((original.correction, *extra))
        selectable = np.r_[original.selectable, np.ones(len(d.NEW), dtype=bool)]
        check(np.isfinite(matrix).all() and (nodiet is None or np.isfinite(nodiet).all()),
              "V4_NONFINITE_PREDICTIONS")
        results[endpoint] = Forecasts(original, matrix, selectable, nodiet)
    return results
