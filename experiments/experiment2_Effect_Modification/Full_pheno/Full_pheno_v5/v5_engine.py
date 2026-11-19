"""Nested training on archived discovery household folds."""

from dataclasses import dataclass
import time

import numpy as np

import protocol as p
import v4_design as v4
import v5_design as d
from source import Outcome, check
from v3_inputs import InnerContexts, read_sites
from v4_select import losses as v4_losses, single as v4_single
from v5_models import forecasts
from v5_select import losses, plans


@dataclass(frozen=True)
class Held:
    rows: np.ndarray
    y: np.ndarray
    no_diet: np.ndarray
    off: np.ndarray
    correction: np.ndarray
    library_gain: np.ndarray
    available: np.ndarray
    weights: dict


def fit_outer(study, contexts, outcomes, exposure, fold):
    train = study.fold != fold
    inner_y = {key: Outcome(np.where(train, y.y, np.nan), y.unit) for key, y in outcomes.items()}
    inner = []
    for validation in p.FOLDS:
        if validation == fold:
            continue
        inner_fit = forecasts(contexts.split(fold, validation), exposure, inner_y, validation)
        check(all(np.all(study.fold[f.previous.old.rows] == validation)
                  for f in inner_fit.values()), "V5_INNER_ROLE_DIFF")
        inner.append(inner_fit)
    separate, maximin = {}, {}
    for e in outcomes:
        if (exposure, e) == d.ANCHOR:
            continue
        splits = [item[e] for item in inner]
        separate[e] = losses(splits)
        maximin[e] = v4_single(v4_losses([f.previous for f in splits]), v4.NEW, maximin=True)
    weights = plans(separate, maximin)
    del inner, separate
    if exposure == d.ANCHOR[0]:
        weights[d.ANCHOR[1]] = {method: {d.ANCHOR_RECIPE: 1.} for method in d.WORKFLOWS}
    held = {}
    for e, f in forecasts(study, exposure, outcomes, fold, with_no_diet=True).items():
        base = f.previous
        check(np.all(study.fold[base.old.rows] == fold) and base.no_diet is not None, "V5_OUTER_ROLE_DIFF")
        array = np.zeros((len(d.LIBRARY), len(d.WORKFLOWS)))
        for j, method in enumerate(d.WORKFLOWS):
            for recipe, w in weights[e][method].items():
                array[d.INDEX[recipe], j] = w
        check(np.isfinite(array).all() and np.all(array >= 0.) and np.all(array.sum(axis=0) <= 1.+1e-12)
              and not array[~f.available].any(), "V5_INVALID_WEIGHTS")
        c = f.correction @ array
        y, off = base.old.y, base.old.off
        library = np.sum(f.correction*(2.*(y-off)[:, None]-f.correction), axis=0)
        check(np.isfinite(c).all() and np.isfinite(library).all(), "V5_NONFINITE_OUTER_GAIN")
        held[e] = Held(base.old.rows, y, base.no_diet, off, c, library, f.available, weights[e])
    return held


def fit_all(study, progress=None):
    outcomes = {e: study.outcome(e) for e in p.ENDPOINTS}
    contexts = InnerContexts(study, read_sites(study))
    parts = {key: [] for key in p.MODES["all"]}
    for fold in p.FOLDS:
        for exposure in p.EXPOSURES:
            start = time.monotonic()
            for e, fit in fit_outer(study, contexts, outcomes, exposure, fold).items():
                parts[exposure, e].append(fit)
            if progress:
                progress("FOLD_COMPLETE", dict(exposure=exposure, fold=fold, endpoints=len(outcomes),
                                              seconds=round(time.monotonic()-start, 2)))
    return parts
