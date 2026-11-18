"""Shared-baseline panel surface and family-nested penalty selection."""
import numpy as np


FIXED = (.01, .03)
GRID = tuple((linear, curved) for linear in (.003, .01, .1, 1.)
             for curved in (.003, .03, .3, 3.))
ROUTES = ("baseline", "shannon", "fixed", "tuned", "joint")


def marker_basis(score, train):
    mean, sd = score[train].mean(), score[train].std()
    if sd <= 1e-10:
        raise ValueError("Degenerate training marker")
    linear = (score - mean) / sd
    hinge = np.maximum(linear - np.median(linear[train]), 0.)
    return np.column_stack((linear, hinge - hinge[train].mean()))


def dose_basis(dose, train):
    line = np.column_stack((np.ones(len(dose)), dose[:, 0]))
    coef = np.linalg.lstsq(line[train], dose[train, 1], rcond=1e-10)[0]
    return np.column_stack((dose[:, 0] - dose[train, 0].mean(), dose[:, 1] - line @ coef))


def products(dose, markers):
    return (dose[:, :, None] * markers[:, None, :]).reshape(len(dose), -1)


def design(role, ecology, pcs, scores, train):
    y, nuisance, dose = role
    panel, shannon = (marker_basis(scores[:, i], train) for i in range(2))
    background = np.column_stack((nuisance, ecology, ecology**2, pcs, panel, shannon))
    basis = dose_basis(dose, train)
    return y, background, products(basis, panel), products(basis, shannon)


def candidates(y, x, panel, shannon, train, valid):
    # Only training outcomes and the training feature projection enter this solve.
    inverse = np.linalg.pinv(x[train], rcond=1e-10)
    xvalid = x[valid]
    beta = inverse @ y[train]
    baseline = xvalid @ beta
    residual = y[train] - x[train] @ beta
    result = {("baseline", None): baseline}
    for name, raw in (("shannon", shannon[:, :1]), ("panel", panel),
                      ("joint", np.column_stack((panel, shannon)))):
        rotation = inverse @ raw[train]
        fit = raw[train] - x[train] @ rotation
        held = raw[valid] - xvalid @ rotation
        gram, rhs = fit.T @ fit, fit.T @ residual
        opts = ((.03, .03),) if name == "shannon" else tuple(dict.fromkeys((FIXED, *GRID)))
        for linear, curved in opts:
            weights = np.resize([linear, linear, curved, curved], raw.shape[1])
            coef = np.linalg.solve(gram + np.diag(weights * train.sum()), rhs)
            result[name, (linear, curved)] = baseline + held @ coef
    return result


def inner_roles(family, outer_train, seed):
    groups = np.unique(family[outer_train])
    parts = np.array_split(np.random.default_rng(seed).permutation(groups), 3)
    for subset in parts:
        valid = outer_train & np.isin(family, subset)
        yield outer_train & ~valid, valid


def choose(losses, name):
    options = [("baseline", None), (name, FIXED), *((name, pair) for pair in GRID)]
    return min(options, key=lambda key: round(losses[key], 8))
