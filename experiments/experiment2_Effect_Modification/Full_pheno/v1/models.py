"""Fixed fits for matched off, three-species, and three-species plus Shannon."""

from dataclasses import dataclass

import numpy as np

import protocol as p
from source import check, clean_outcome


def standardize(values, n_train):
    train = values[:n_train]
    sd = train.std(axis=0)
    scale = np.where(sd > p.RANK_TOL, sd, 1.0)
    return (values - train.mean(axis=0)) / scale, scale


def fiber_basis(fiber):
    lo, hi = p.FIBER_DOSE
    return np.column_stack(((fiber - lo) / (hi - lo), np.maximum(fiber - p.FIBER_KNOT, 0) / (hi - lo)))


def dose_basis(raw, n_train):
    u = raw[:, 0]
    x = np.column_stack((np.ones(len(u)), u))
    shift = np.linalg.lstsq(x[:n_train], raw[:n_train, 1], rcond=p.RANK_TOL)[0]
    return np.column_stack((u - u[:n_train].mean(), raw[:, 1] - x @ shift)), float(shift[1])


def products(basis, markers):
    return (basis[:, :, None] * markers[:, None, :]).reshape(len(basis), -1)


@dataclass(frozen=True)
class Design:
    main: np.ndarray
    panel: np.ndarray
    shannon: np.ndarray
    marker_scales: np.ndarray
    marker_medians: np.ndarray
    dose_difference: np.ndarray
    exposure_values: np.ndarray
    n_train: int


def create_design(study, exposure, fold, rows, n_train, *, association=False):
    bg = study.backgrounds[fold]
    energy_name = p.EXPOSURES[exposure][1]
    host = np.column_stack([study.numeric[k][rows] for k in ("age_calendar_years", "sex_code", energy_name)])
    host, _ = standardize(host, n_train)
    smoke = study.smoking[rows]
    levels = sorted(set(smoke[:n_train]) | {"__UNKNOWN__"})
    smoke = np.where(np.isin(smoke, levels), smoke, "__UNKNOWN__")
    indicators = (smoke[:, None] == np.asarray(levels[1:], dtype=str)[None, :]).astype(float)

    if exposure == "fiber_density":
        amounts = study.numeric[exposure][rows]
        raw = fiber_basis(amounts)
        contrast = np.diff(fiber_basis(np.array(p.FIBER_DOSE)), axis=0)[0]
    else:
        raw = bg.archived[exposure][rows]
        amounts = raw[:, 0] * bg.upf["scale"] + bg.upf["center"]
        # The averaged nonlinear basis can't be recovered by plugging a single
        # expected fraction into the spline. Use a separate linear contrast.
        contrast = np.array([(p.UPF_DOSE[1] - p.UPF_DOSE[0]) / bg.upf["scale"], 0.])
    dose, slope = dose_basis(raw, n_train)
    contrast[1] -= slope * contrast[0]

    markers = study.markers[rows]
    linear, scales = standardize(markers, n_train)
    medians = np.median(markers[:n_train], axis=0)
    hinges = np.maximum(markers - medians, 0) / scales
    hinges -= hinges[:n_train].mean(axis=0)
    panel = np.column_stack((linear[:, 0], hinges[:, 0]))
    shannon = linear[:, 1:2]

    ecology, _ = standardize(study.ecology[rows], n_train)
    pcs, _ = standardize(bg.pcs[rows], n_train)
    # Shannon's linear main effect is already one column of ecology.
    main = np.column_stack((np.ones(len(rows)), host, indicators, dose,
                            products(dose, host[:, :2]), ecology, ecology**2,
                            pcs, panel, hinges[:, 1]))
    active = dose[:, :1] if association and exposure == "UPF" else dose
    delta = contrast[:active.shape[1]]
    z, w = products(active, panel), products(active, shannon)
    check(all(np.isfinite(a).all() for a in (main, z, w, delta)), "NONFINITE_MODEL_DESIGN")
    return Design(main, z, w, scales, medians, delta, amounts, n_train)


def correction(columns, nuisance, y_train, penalty):
    n = len(y_train)
    # Residualize the target and proposed interaction columns on the same train-only span.
    inv = np.linalg.pinv(nuisance[:n], rcond=p.RANK_TOL)
    partial = columns - nuisance @ (inv @ columns[:n])
    resid = y_train - nuisance[:n] @ (inv @ y_train)
    z = partial[:n]
    fitted = np.linalg.solve(z.T @ z + n * np.diag(penalty), z.T @ resid)
    return partial[n:] @ fitted


def predict(design, y_train):
    n = design.n_train
    check(y_train.shape == (n,) and np.isfinite(y_train).all(), "INVALID_TRAIN_OUTCOME")
    x, z, w = design.main, design.panel, design.shannon
    base = x[n:] @ (np.linalg.pinv(x[:n], rcond=p.RANK_TOL) @ y_train)
    panel = base + correction(z, x, y_train, p.PANEL_PENALTY)
    fusion = panel + correction(w, np.column_stack((x, z)), y_train, p.SHANNON_PENALTY)
    result = np.column_stack((base, panel, fusion))
    check(np.isfinite(result).all(), "NONFINITE_HELD_PREDICTION")
    return result


@dataclass(frozen=True)
class Fold:
    fold: int
    rows: np.ndarray
    y: np.ndarray
    predictions: np.ndarray
    training_people: int
    training_households: int
    held_households: int
    clipped: int
    outcome_excluded: int

    def counts(self):
        return dict(fold=self.fold, training_people=self.training_people, held_people=len(self.rows),
                    training_households=self.training_households, held_households=self.held_households,
                    held_outcome_winsorized=self.clipped, held_outcome_8sd_excluded=self.outcome_excluded)


def fit_fold(study, exposure, endpoint, outcome, fold):
    train = study.backgrounds[fold].role
    y = clean_outcome(outcome.y, train)
    eligible = study.eligible(exposure, endpoint, fold, y)
    fit, held = np.flatnonzero(eligible & train), np.flatnonzero(eligible & ~train)
    check(len(fit) >= 3 and len(held) >= 2, f"EMPTY_OR_SMALL_OUTER_FOLD {endpoint} {exposure} {fold}")
    hfit, hheld = np.unique(study.family[fit]), np.unique(study.family[held])
    check(len(hfit) >= 2 and len(hheld) >= 1 and not np.intersect1d(hfit, hheld).size,
          "OUTER_HOUSEHOLD_ISOLATION")
    design = create_design(study, exposure, fold, np.r_[fit, held], len(fit))
    q = predict(design, y[fit])
    excluded = (~train) & np.isfinite(outcome.y) & ~np.isfinite(y)
    return Fold(fold, held, y[held], q, len(fit), len(hfit), len(hheld),
                int((y[held] != outcome.y[held]).sum()), int(excluded.sum()))
