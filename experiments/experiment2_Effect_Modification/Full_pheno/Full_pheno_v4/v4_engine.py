"""Nested family fitting and paired held predictions for all 18 tasks."""

from dataclasses import dataclass
import time

import numpy as np

import protocol as p
import v3_design as v3
import v4_design as d
from source import Outcome, check
from v3_blend import guarded_blend, inner_loss
from v3_inputs import InnerContexts, read_sites
from v4_models import forecasts
from v4_select import losses, select


@dataclass(frozen=True)
class Held:
    rows: np.ndarray
    y: np.ndarray
    no_diet: np.ndarray
    off: np.ndarray
    corrections: np.ndarray
    available: np.ndarray
    library_gain: np.ndarray
    weights: dict


def fit_outer(study, contexts, outcomes, exposure, fold):
    outer_train = study.fold != fold
    inner_outcomes = {e: Outcome(np.where(outer_train, y.y, np.nan), y.unit) for e, y in outcomes.items()}
    inner = []
    for valid in p.FOLDS:
        if valid == fold:
            continue
        split = contexts.split(fold, valid)
        f = forecasts(split, exposure, inner_outcomes, valid)
        check(all(np.all(outer_train[t.old.rows]) and np.all(study.fold[t.old.rows] == valid)
                  for t in f.values()), "V4_INNER_ROLE_MISMATCH")
        inner.append(f)
    nested = {}
    for e in outcomes:
        if (exposure, e) != d.ANCHOR:
            nested[e] = losses([s[e] for s in inner])
    w = {}
    for e, loss in nested.items():
        inherited = inner_loss([s[e].old for s in inner])
        old_guard = guarded_blend(inherited, inherited.scaled())
        saved_rule = {v3.LIBRARY[i]: float(old_guard[i]) for i in np.flatnonzero(old_guard)}
        w[e] = select(e, loss, nested, saved_rule)
    del inner, nested
    if exposure == d.ANCHOR[0]:
        w[d.ANCHOR[1]] = {method: {d.ANCHOR_RECIPE: 1.} for method in d.WORKFLOWS}
        w[d.ANCHOR[1]].update(uniform_pc5={d.PC5: 1.}, uniform_oral={d.ORAL: 1.})

    output = {}
    for e, f in forecasts(study, exposure, outcomes, fold, with_no_diet=True).items():
        methods = w[e]
        weight = np.zeros((len(d.LIBRARY), len(d.WORKFLOWS)))
        for j, method in enumerate(d.WORKFLOWS):
            for recipe, scale in methods[method].items():
                weight[d.INDEX[recipe], j] = scale
        check(np.isfinite(weight).all() and np.all(weight >= 0.)
              and np.all(weight.sum(axis=0) <= 1.+1e-12) and not weight[~f.selectable].any(),
              "V4_INVALID_WORKFLOW_WEIGHTS")
        check(np.all(study.fold[f.old.rows] == fold) and f.no_diet is not None, "V4_OUTER_ROLE_MISMATCH")
        c = f.correction @ weight
        y, off = f.old.y, f.old.off
        gain = np.sum(f.correction*(2.*(y-off)[:, None] - f.correction), axis=0)
        check(np.isfinite(c).all() and np.isfinite(gain).all(), "V4_NONFINITE_OUTER_MOMENTS")
        output[e] = Held(f.old.rows, y, f.no_diet, off, c, f.selectable, gain, methods)
    return output


def fit_all(study, progress=None):
    outcomes = {e: study.outcome(e) for e in p.ENDPOINTS}
    contexts = InnerContexts(study, read_sites(study))
    all_parts = {key: [] for key in p.MODES["all"]}
    for fold in p.FOLDS:
        for exposure in p.EXPOSURES:
            start = time.monotonic()
            parts = fit_outer(study, contexts, outcomes, exposure, fold)
            for e, f in parts.items():
                all_parts[exposure, e].append(f)
            if progress:
                progress("FOLD_COMPLETE", dict(fold=fold, exposure=exposure, endpoints=len(parts),
                                              seconds=round(time.monotonic()-start, 2)))
    return all_parts
