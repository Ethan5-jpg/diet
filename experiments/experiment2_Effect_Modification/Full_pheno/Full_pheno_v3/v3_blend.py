"""Learn choices and blend weights from inner held-household loss only."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

import v3_design as d
from design import domain
from source import check


@dataclass(frozen=True)
class InnerLoss:
    gram: np.ndarray
    rhs: np.ndarray
    fold_gram: np.ndarray
    fold_rhs: np.ndarray
    selectable: np.ndarray

    @property
    def fold_diag(self):
        return np.diagonal(self.fold_gram, axis1=1, axis2=2)

    @property
    def gains(self):
        return 2.*self.rhs - self.gram.diagonal()

    def scaled(self):
        diag = self.gram.diagonal()
        factor = np.divide(self.rhs, diag, out=np.zeros_like(diag), where=diag > 0.)
        factor = np.clip(factor, 0., 1.)
        gains = factor * (2.*self.rhs - factor * diag)
        wins = (factor * (2.*self.fold_rhs - factor * self.fold_diag) > 0.).sum(axis=0)
        return factor, gains, wins

    def fold_gains(self, weights):
        return 2.*(self.fold_rhs @ weights) - np.einsum("i,fij,j->f", weights, self.fold_gram, weights)


def inner_loss(forecasts):
    rows = np.concatenate([f.rows for f in forecasts])
    check(len(np.unique(rows)) == len(rows), "V3_DUPLICATE_INNER_HELD_ROWS")
    y = np.concatenate([f.y for f in forecasts])
    sst = float(np.sum((y - y.mean())**2))
    check(np.isfinite(sst) and sst > 0., "V3_INNER_DEGENERATE_OUTCOME")
    base = forecasts[0].selectable
    check(all(np.array_equal(f.selectable, base) for f in forecasts), "V3_INNER_LIBRARY_DIFFERS")
    fold_gram, fold_rhs = [], []
    for f in forecasts:
        c = f.correction
        fold_gram.append((c.T @ c) / sst)
        fold_rhs.append((c.T @ (f.y - f.off)) / sst)
    fold_gram, rhs = np.asarray(fold_gram), np.asarray(fold_rhs)
    gram = fold_gram.sum(axis=0)
    check(np.isfinite(gram).all() and np.isfinite(rhs).all(), "V3_INNER_NONFINITE_MOMENTS")
    return InnerLoss(gram, rhs.sum(axis=0), fold_gram, rhs, base)


def diverse(scores, allowed, k):
    order = np.argsort(-scores, kind="stable")
    seen, chosen = set(), []
    for j in order:
        if not allowed[j]:
            continue
        family = d.representation(d.LIBRARY[j])
        if family not in seen:
            seen.add(family)
            chosen.append(j)
            if len(chosen) == k:
                break
    return np.asarray(chosen, dtype=np.int64)


def mean_weights(indices):
    w = np.zeros(len(d.LIBRARY))
    if len(indices):
        w[indices] = 1./len(indices)
    return w


def blend_weights(loss, indices):
    """Nonnegative correction mixture with sum <= 1; spare weight is the off arm."""
    w = np.zeros(len(d.LIBRARY))
    if not len(indices):
        return w
    gram = loss.gram[np.ix_(indices, indices)]
    b = loss.rhs[indices]
    ridge = d.BLEND_PENALTY * float(np.median(gram.diagonal()))
    gram += ridge * np.eye(len(indices))
    scale = max(float(np.max(gram.diagonal())), float(np.max(abs(b))), 1e-12)
    gram, b = gram/scale, b/scale

    def objective(w):
        v = gram @ w
        return float(w @ (0.5*v - b)), v - b

    fit = minimize(objective, np.zeros(len(b)), method="SLSQP", jac=True,
                   bounds=[(0., 1.)] * len(b),
                   constraints=[{"type": "ineq", "fun": lambda w: 1. - w.sum(),
                                 "jac": lambda w: -np.ones_like(w)}],
                   options={"ftol": 1e-12, "maxiter": 200})
    check(fit.success and np.isfinite(fit.x).all() and fit.x.min() >= -1e-7
          and fit.x.sum() <= 1. + 1e-7, "V3_BLEND_WEIGHT_SOLVE")
    value = np.where(fit.x > 1e-9, fit.x, 0.)
    value /= max(1., value.sum())
    if objective(value)[0] < 0.:
        w[indices] = value
    return w


def guarded_blend(loss, scaled):
    factor, gains, wins = scaled
    selected = diverse(gains, loss.selectable & (gains >= d.GUARD_GAIN) & (wins >= d.GUARD_WIN_FOLDS), d.TOP_K)
    weights = blend_weights(loss, selected)
    if len(selected):
        fold_gains = loss.fold_gains(weights)
        if fold_gains.sum() < d.GUARD_GAIN or (fold_gains > 0.).sum() < d.GUARD_WIN_FOLDS:
            # Vetted components can lose different folds. Revert to the best
            # eligible single correction at its vetted shrinkage.
            best = selected[0]
            weights[:] = 0.
            weights[best] = factor[best]
    return weights


def choose(loss):
    scaled = loss.scaled()
    _, scale_gain, _ = scaled
    gains = loss.gains
    fold_wins = (2.*loss.fold_rhs > loss.fold_diag).sum(axis=0)
    return {
        "fixed_pc5": mean_weights([d.LIBRARY_INDEX[d.PC5_REFERENCE]]),
        "select": mean_weights(diverse(gains, loss.selectable & (gains > 1e-12), 1)),
        "select_guarded": mean_weights(diverse(
            gains, loss.selectable & (gains >= d.GUARD_GAIN) & (fold_wins >= d.GUARD_WIN_FOLDS), 1)),
        "mean_top3": mean_weights(diverse(
            gains, loss.selectable & (gains > 1e-12) & (fold_wins >= d.GUARD_WIN_FOLDS), 3)),
        "blend": blend_weights(loss, diverse(scale_gain, loss.selectable & (scale_gain > 1e-12), d.TOP_K)),
        "blend_guarded": guarded_blend(loss, scaled),
    }


def by_domain(losses):
    weights = {}
    for label in ("glucose", "adiposity", "liver"):
        group = [key for key in losses if domain(key) == label]
        if not group:
            continue
        local = [losses[key] for key in group]
        pooled = InnerLoss(
            np.mean([r.gram for r in local], axis=0), np.mean([r.rhs for r in local], axis=0),
            np.mean([r.fold_gram for r in local], axis=0), np.mean([r.fold_rhs for r in local], axis=0),
            np.logical_and.reduce([r.selectable for r in local]),
        )
        shared = guarded_blend(pooled, pooled.scaled())
        weights.update({key: shared for key in group})
    return weights


def plan(losses):
    domain_weights = by_domain(losses)
    result = {}
    for endpoint, loss in losses.items():
        w = choose(loss)
        w["domain_blend"] = domain_weights[endpoint]
        w["domain_partial"] = d.DOMAIN_WEIGHT*w["domain_blend"] + (1. - d.DOMAIN_WEIGHT)*w["blend_guarded"]
        result[endpoint] = w
    return result
