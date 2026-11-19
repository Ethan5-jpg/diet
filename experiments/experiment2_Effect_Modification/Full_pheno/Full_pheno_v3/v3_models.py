"""Candidate interaction corrections on the common v1 main-effect prediction."""

from dataclasses import dataclass

import numpy as np

import protocol as p
import v3_design as d
from breadth import basis, evaluate_split, fit_profile
from design import domain
from models import create_design, products
from source import check, clean_outcome


@dataclass(frozen=True)
class Forecast:
    rows: np.ndarray
    y: np.ndarray
    off: np.ndarray
    correction: np.ndarray
    counts: dict
    selectable: np.ndarray


def views_for_split(study, exposure, fold):
    raw, views = basis(study, exposure, fold)
    oral, gut = views["oral_pc5"][:, :2], views["gut_pc5"][:, :2]
    views.update(oral_pc2=oral, gut_pc2=gut, both_pc2=np.column_stack((oral, gut)),
                 oral_shannon=views["ecology"][:, :1], gut_ecology=views["ecology"][:, 2:],
                 gut_shannon_pc2=np.column_stack((views["shannon"], gut)))
    return raw, views


def common_coefficients(profiles, penalty):
    gram = np.mean([prof.gram for prof in profiles], axis=0)
    rhs = np.mean([prof.rhs for prof in profiles], axis=0)
    return np.linalg.solve(gram + penalty * np.eye(len(rhs)), rhs)


def forecasts(study, exposure, outcomes, fold):
    """Replay v2 and add narrower, stronger-ridge and domain-specific candidates."""
    original = evaluate_split(study, exposure, outcomes, fold)
    raw, views = views_for_split(study, exposure, fold)
    training = study.backgrounds[fold].role
    library, donor_profiles = {}, {}

    for endpoint, outcome in outcomes.items():
        old = original[endpoint]
        y = clean_outcome(outcome.y, training)
        tr = np.flatnonzero(training & study.eligible(exposure, endpoint, fold, y))
        rows = np.r_[tr, old.rows]
        n = len(tr)
        check(n == old.counts["training_people"] and np.array_equal(y[old.rows], old.y),
              "V3_LIBRARY_TARGET_ALIGNMENT")
        main = create_design(study, exposure, fold, rows, n).main
        inverse = np.linalg.pinv(main[:n], rcond=p.RANK_TOL)
        residual = y[tr] - main[:n] @ (inverse @ y[tr])
        off = old.preds[d.BASE]
        corrections = {name: old.preds[name] - off for name in d.V2_NAMES}
        profiles = {}
        for name, values in views.items():
            z = products(raw[rows], values[rows])
            for dose in d.SHARED_MODES:
                k = values.shape[1] if dose == "linear" else z.shape[1]
                profiles[name, dose] = fit_profile(z[:, :k], main, inverse, residual)
        for view, dose, penalty in d.RIDGES:
            corrections[d.ridge_name(view, dose, penalty)] = profiles[view, dose].fit(penalty)
        donor_profiles[endpoint] = {key: prof for key, prof in profiles.items() if key[0] in d.DOMAIN_VIEWS}
        library[endpoint] = corrections

    for label in ("adiposity", "liver"):
        donors = [endpoint for endpoint in outcomes if domain(endpoint) == label]
        if len(donors) < 2:
            continue
        for view in d.DOMAIN_VIEWS:
            for dose in d.SHARED_MODES:
                profiles = [donor_profiles[e][view, dose] for e in donors]
                for penalty in d.DOMAIN_RIDGE:
                    prior = common_coefficients(profiles, penalty)
                    for endpoint, prof in zip(donors, profiles, strict=True):
                        corrections = library[endpoint]
                        for adapt in d.DOMAIN_ADAPT:
                            value = (prof.z_valid @ prior) * prof.scale if adapt is None else prof.fit(adapt, prior)
                            corrections[d.domain_name(view, dose, penalty, adapt)] = value

    result = {}
    for endpoint, old in original.items():
        corrections = library[endpoint]
        allowed = np.array([name in corrections for name in d.LIBRARY], dtype=bool)
        zero = np.zeros(len(old.rows))
        matrix = np.column_stack([corrections.get(name, zero) for name in d.LIBRARY])
        check(np.isfinite(matrix).all(), "V3_LIBRARY_NONFINITE_PREDICTION")
        result[endpoint] = Forecast(old.rows, old.y, old.preds[d.BASE], matrix, old.counts, allowed)
    return result
