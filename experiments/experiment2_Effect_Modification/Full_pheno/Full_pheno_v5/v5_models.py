"""Train nonlinear interaction corrections while retaining each v4 main arm."""

from dataclasses import dataclass

import numpy as np

import protocol as p
import v4_design as v4
import v5_design as d
from models import create_design, dose_basis, products, standardize
from source import check, clean_outcome
from v3_models import views_for_split
from v4_models import Forecasts, forecasts as previous_forecasts, robust_weights


@dataclass(frozen=True)
class Forecast:
    previous: Forecasts
    correction: np.ndarray
    available: np.ndarray


def bends(values, n, quantiles):
    cutoffs = np.quantile(values[:n], quantiles, axis=0)
    expanded = np.column_stack([np.maximum(values-cut, 0.) for cut in cutoffs])
    return standardize(expanded, n)[0]


def orthogonalize(main, product, target, n, weights):
    scale = np.sqrt(weights)
    training = main[:n]
    inverse = np.linalg.pinv(training*scale[:, None], rcond=p.RANK_TOL)
    z = product - main @ (inverse @ (product[:n]*scale[:, None]))
    r = target - training @ (inverse @ (target*scale))
    fitted = z[:n]*scale[:, None]
    total = weights.sum()
    return z[n:], fitted.T @ fitted / total, fitted.T @ (r*scale) / total


def pc5_half_training(main, inv, residual, raw, marker, saved, n):
    z = products(raw[:, :1], marker)
    z -= main @ (inv @ z[:n])
    beta = np.linalg.solve(z[:n].T @ z[:n]/n + np.eye(z.shape[1]), z[:n].T @ residual/n)
    pc5 = z @ beta
    check(np.allclose(pc5[n:], saved, atol=2e-8, rtol=1e-10), "V5_PC5_TRAINING_REPLAY_DIFF")
    return .5*pc5


def interaction_masks(dose, marker, d_base, m_base):
    di = np.arange(dose.shape[1]) >= d_base
    mi = np.arange(marker.shape[1]) >= m_base
    local_dose = np.repeat(di, marker.shape[1])
    local_marker = np.tile(mi, dose.shape[1])
    penalties = d.EXTRA_PENALTY ** (local_dose.astype(int) + local_marker.astype(int))
    return {"dose": ~local_marker, "marker": ~local_dose, "joint": np.ones(len(penalties), bool)}, penalties


def forecasts(study, exposure, outcomes, fold, *, with_no_diet=False):
    previous = previous_forecasts(study, exposure, outcomes, fold, with_no_diet=with_no_diet)
    raw, views = views_for_split(study, exposure, fold)
    role = study.backgrounds[fold].role
    result = {}
    for endpoint, fitted in previous.items():
        old = fitted.old
        y = clean_outcome(outcomes[endpoint].y, role)
        train = np.flatnonzero(role & study.eligible(exposure, endpoint, fold, y))
        rows = np.r_[train, old.rows]
        n = len(train)
        check(n == old.counts["training_people"] and np.array_equal(y[old.rows], old.y), "V5_TARGET_ROWS_DIFF")
        main = create_design(study, exposure, fold, rows, n).main
        inv = np.linalg.pinv(main[:n], rcond=p.RANK_TOL)
        residual = y[train]-main[:n] @ (inv @ y[train])
        raw_dose = raw[rows]
        linear = standardize(dose_basis(raw_dose, n)[0], n)[0]
        dose = np.column_stack((linear, bends(linear[:, :1], n, (.25, .75))))
        half = pc5_half_training(main, inv, residual, raw_dose, views["gut_pc5"][rows],
                                 fitted.correction[:, v4.INDEX[d.PC5]], n)
        centers = {"off": np.zeros(len(rows)), "half": half}
        weights = {c: robust_weights(residual-value[:n]) for c, value in centers.items()}
        extra = {}
        for view in d.VIEWS:
            raw_marker = views[view][rows]
            if view == "panel":
                raw_marker = raw_marker[:, :1]
            ordinary = standardize(raw_marker, n)[0]
            marker = np.column_stack((ordinary, bends(ordinary, n, (.50,))))
            nuisance = np.column_stack((main, dose, marker))
            block = products(dose, marker)
            masks, charge = interaction_masks(dose, marker, linear.shape[1], ordinary.shape[1])
            for center, base in centers.items():
                target = y[train]-base[:n]
                plans = {
                    "ridge": orthogonalize(nuisance, block, target, n, np.ones(n)),
                    "robust": orthogonalize(nuisance, block, target, n, weights[center]),
                }
                for fit, penalty in d.SETTINGS:
                    test, gram, rhs = plans[fit]
                    for shape, keep in masks.items():
                        g = gram[np.ix_(keep, keep)] + np.diag(charge[keep]*penalty)
                        correction = test[:, keep] @ np.linalg.solve(g, rhs[keep])
                        extra[d.name(view, shape, center, fit, penalty)] = base[n:] + correction
        matrix = np.column_stack((fitted.correction, *(extra[key] for key in d.NEW)))
        available = np.r_[fitted.selectable, np.ones(len(d.NEW), bool)]
        check(np.isfinite(matrix).all(), "V5_NONFINITE_FORECASTS")
        result[endpoint] = Forecast(fitted, matrix, available)
    return result
