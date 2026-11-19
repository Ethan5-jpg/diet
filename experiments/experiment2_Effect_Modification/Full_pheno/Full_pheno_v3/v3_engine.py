"""Nested fitting with one unchanged fiber–glucose branch."""

from dataclasses import dataclass
import time

import numpy as np

import protocol as p
import v3_design as d
from source import Outcome, check
from v3_blend import inner_loss, mean_weights, plan
from v3_inputs import InnerContexts, read_sites
from v3_models import Forecast, forecasts


@dataclass(frozen=True)
class Held:
    candidate: Forecast
    correction: np.ndarray
    weights: dict


def fit_outer(study, contexts, outcomes, exposure, outer_fold):
    """Determine all workflow weights before constructing outer predictions."""
    train = study.fold != outer_fold
    training_outcomes = {endpoint: Outcome(np.where(train, raw.y, np.nan), raw.unit)
                         for endpoint, raw in outcomes.items()}
    inner = []
    for fold in p.FOLDS:
        if fold == outer_fold:
            continue
        view = contexts.split(outer_fold, fold)
        prediction = forecasts(view, exposure, training_outcomes, fold)
        check(all(np.all(study.fold[f.rows] == fold) and np.all(train[f.rows])
                  for f in prediction.values()), "V3_INNER_PREDICTION_ROLE")
        inner.append(prediction)
    losses = {e: inner_loss([fold[e] for fold in inner])
              for e in outcomes if (exposure, e) != d.ANCHOR}
    selected = plan(losses)
    anchor = mean_weights([d.LIBRARY_INDEX[d.ANCHOR_RECIPE]])
    if exposure == d.ANCHOR[0]:
        selected[d.ANCHOR[1]] = dict.fromkeys(d.WORKFLOWS, anchor)
    scored = forecasts(study, exposure, outcomes, outer_fold)
    results = {}
    for endpoint, data in scored.items():
        check(np.all(study.fold[data.rows] == outer_fold), "V3_OUTER_PREDICTION_ROLE")
        weights = selected[endpoint]
        w = np.column_stack([weights[name] for name in d.WORKFLOWS])
        check(np.isfinite(w).all() and np.all(w >= 0.) and np.all(w.sum(axis=0) <= 1. + 1e-12)
              and not np.any(w[~data.selectable]), "V3_INVALID_WORKFLOW_WEIGHTS")
        if (exposure, endpoint) == d.ANCHOR:
            c = np.repeat(data.correction[:, [d.LIBRARY_INDEX[d.ANCHOR_RECIPE]]], len(d.WORKFLOWS), axis=1)
        else:
            c = data.correction @ w
        check(np.isfinite(c).all(), "V3_NONFINITE_WORKFLOW")
        summaries = {name: {d.LIBRARY[j]: float(weight[j]) for j in np.flatnonzero(weight)}
                     for name, weight in weights.items()}
        results[endpoint] = Held(data, c, summaries)
    return results


def fit_all(study, progress=None):
    outcomes = {e: study.outcome(e) for e in p.ENDPOINTS}
    contexts = InnerContexts(study, read_sites(study))
    collected = {pair: [] for pair in p.MODES["all"]}
    for fold in p.FOLDS:
        for exposure in p.EXPOSURES:
            start = time.monotonic()
            parts = fit_outer(study, contexts, outcomes, exposure, fold)
            for endpoint, held in parts.items():
                collected[exposure, endpoint].append(held)
            if progress:
                progress("FOLD_COMPLETE", dict(fold=fold, exposure=exposure, endpoints=len(parts),
                                              seconds=round(time.monotonic()-start, 2)))
    return collected
