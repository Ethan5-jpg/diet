"""Measure the dietary main effect and microbiome correction on matched families."""

from dataclasses import dataclass

import numpy as np

import protocol as p
from breadth import basis, fit_profile
from models import create_design, dose_basis, predict, products, standardize
from source import check, clean_outcome
from v4_models import no_diet_main


RECIPES = ("legacy", "linear", "linear_ridge", "curve_ridge", "host_ridge", "legacy_fusion")
METRICS = ("diet_main_pp", "interaction_pp", "diet_total_pp", "vs_pc5_half_pp")


@dataclass(frozen=True)
class Fit:
    rows: np.ndarray
    outcome: np.ndarray
    no_diet: np.ndarray
    off: np.ndarray
    on: np.ndarray


def projection(columns, inverse, nuisance, n):
    return columns - nuisance @ (inverse @ columns[:n])


def ridge_diet(main, inverse, dietary, y, penalty):
    n = len(y)
    base = main @ (inverse @ y)
    residual = y - base[:n]
    adapted = projection(standardize(dietary, n)[0], inverse, main, n)
    train = adapted[:n]
    gram = train.T @ train / n
    weights = np.linalg.solve(gram + np.diag(penalty), train.T @ residual / n)
    return base + adapted @ weights, adapted


def fit_fold(study, exposure, endpoint, outcome, fold, diet_basis=None):
    bg = study.backgrounds[fold]
    train = bg.role
    clean = clean_outcome(outcome.y, train)
    eligible = study.eligible(exposure, endpoint, fold, clean)
    tr, held = np.flatnonzero(train & eligible), np.flatnonzero(~train & eligible)
    check(len(tr) > 20 and len(held) > 2, "V6_TOO_FEW_ROWS")
    rows, n, y = np.r_[tr, held], len(tr), clean[tr]
    design = create_design(study, exposure, fold, rows, n)
    raw, views = diet_basis if diet_basis is not None else basis(study, exposure, fold)
    dose = dose_basis(raw[rows], n)[0]
    host_terms = products(dose, design.main[:, 1:3])
    base = no_diet_main(study, rows, n, design.main, dose)
    inverse = np.linalg.pinv(base[:n], rcond=p.RANK_TOL)
    no_diet = base @ (inverse @ y)
    legacy_inverse = np.linalg.pinv(design.main[:n], rcond=p.RANK_TOL)
    legacy = design.main @ (legacy_inverse @ y)
    linear_dose = standardize(dose[:, :1], n)[0]
    interaction = products(raw[rows, :1], views["gut_pc5"][rows])
    corr = fit_profile(interaction, design.main, legacy_inverse, y - legacy[:n]).fit(1.)

    linear_main = np.column_stack((base, linear_dose))
    linear = linear_main @ (np.linalg.pinv(linear_main[:n], rcond=p.RANK_TOL) @ y)
    specs = (
        (dose[:, :1], (0.1,)),
        (dose, (0.1, 0.5)),
        (np.column_stack((dose, host_terms)), (0.1, 0.5, 1.0, 1.0, 1.0, 1.0)),
    )
    pred_off, pred_on = [legacy[n:]], [legacy[n:] + 0.5 * corr]
    for prediction, dietary in ((linear, linear_dose), *(ridge_diet(base, inverse, x, y, a) for x, a in specs)):
        # Project away host and dose terms before learning a microbiome correction.
        main = np.column_stack((base, dietary))
        inv = np.linalg.pinv(main[:n], rcond=p.RANK_TOL)
        update = fit_profile(interaction, main, inv, y - prediction[:n]).fit(1.)
        pred_off.append(prediction[n:])
        pred_on.append(prediction[n:] + 0.5 * update)
    pred_off.append(legacy[n:])
    pred_on.append(predict(design, y)[:, 2])
    return Fit(held, clean[held], no_diet[n:], np.column_stack(pred_off), np.column_stack(pred_on))


def household_moments(parts, household, count):
    rows = np.concatenate([part.rows for part in parts])
    check(len(np.unique(rows)) == len(rows), "V6_DUPLICATE_HELD_ROWS")
    groups = household[rows]
    y = np.concatenate([part.outcome for part in parts])
    centered = y - y.mean()
    base = np.concatenate([part.no_diet for part in parts])
    off = np.concatenate([part.off for part in parts])
    on = np.concatenate([part.on for part in parts])
    columns = np.column_stack((np.ones(len(y)), centered, centered**2, (y - base)**2,
                               (y[:, None] - off)**2, (y[:, None] - on)**2))
    check(np.isfinite(columns).all(), "V6_NONFINITE_MOMENTS")
    return np.column_stack([np.bincount(groups, weights=x, minlength=count) for x in columns.T])


def scores(moments):
    count, a, b, base_error = (moments[..., i] for i in range(4))
    sst = b - a**2 / count
    check(np.all(np.isfinite(sst) & (sst > 0.)), "V6_DEGENERATE_OUTCOME")
    off_error, on_error = np.split(moments[..., 4:], 2, axis=-1)
    sst, base_error = sst[..., None], base_error[..., None]
    r2 = 1. - np.stack((np.broadcast_to(base_error, off_error.shape), off_error, on_error), axis=-1) / sst[..., None]
    contrasts = np.stack((base_error - off_error, off_error - on_error, base_error - on_error,
                          on_error[..., :1] - on_error), axis=-1)
    return r2, 100. * contrasts / sst[..., None]


def simultaneous(moments, draws):
    r2, points = scores(moments.sum(axis=1))
    use = np.any(moments[:, :, 0] > 0, axis=0)
    packed = np.concatenate(moments, axis=1)[use]
    count = len(packed)
    rng = np.random.default_rng(p.BOOTSTRAP_SEED)
    prob = np.full(count, 1. / count)
    samples = []
    for start in range(0, draws, 64):
        weights = rng.multinomial(count, prob, size=min(64, draws - start))
        trial = (weights @ packed).reshape(len(weights), len(moments), -1)
        samples.append(scores(trial)[1])
    errors = np.concatenate(samples) - points
    scale = errors.std(axis=0, ddof=1)
    active = scale > 1e-12
    statistic = np.abs(errors[:, active] / scale[active]).max(axis=1)
    critical = np.quantile(statistic, 0.95, method="higher")
    radius = np.where(active, critical * scale, np.abs(errors).max(axis=0))
    return r2, points, radius
