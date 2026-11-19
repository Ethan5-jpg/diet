"""Matched household loss, aggregate candidate screen and simultaneous intervals."""

from dataclasses import dataclass

import numpy as np

import protocol as p
import v3_design as d
from source import check


@dataclass(frozen=True)
class Task:
    pair: tuple
    people: int
    households: int
    aggregate: np.ndarray
    fold_delta: np.ndarray
    active: np.ndarray
    alignment: np.ndarray
    penalty: np.ndarray
    library: tuple
    v2_replay_error: float


def collect(pair, parts, study, codes, households, ref_v1, ref_v2):
    forecast = [part.candidate for part in parts]
    check([f.counts for f in forecast] == ref_v1["folds"], "V3_V1_FOLD_COUNTS_DIFF " + "/".join(pair))
    rows = np.concatenate([f.rows for f in forecast])
    g = codes[rows]
    n, hh = len(rows), len(np.unique(g))
    check(n == len(np.unique(rows)) == ref_v1["people"] and hh == ref_v1["households"]
          and sum(f.counts["held_households"] for f in forecast) == hh
          and all(np.all(study.fold[f.rows] == j) for j, f in enumerate(forecast)), "V3_COHORT_DIFF")
    y = np.concatenate([f.y for f in forecast])
    off = np.concatenate([f.off for f in forecast])
    tail = y - y.mean()
    sst = tail @ tail
    check(np.isfinite(sst) and sst > 0., "V3_OUTER_DEGENERATE_OUTCOME")
    resid = y - off
    off_r2 = 1. - (resid @ resid)/sst
    c = np.concatenate([part.correction for part in parts])
    fit_gain = c*(2.*resid[:, None] - c)
    check(np.isfinite(fit_gain).all(), "V3_NONFINITE_WORKFLOW_ERRORS")
    cols = np.column_stack((np.ones(n), tail, tail**2, resid**2, fit_gain))
    aggregate = np.column_stack([np.bincount(g, weights=x, minlength=households) for x in cols.T])
    fold_delta = []
    active = []
    points = np.zeros(len(d.LIBRARY))
    available = forecast[0].selectable
    check(all(np.array_equal(f.selectable, available) for f in forecast), "V3_OUTER_LIBRARY_DIFFERS")
    for part in parts:
        f = part.candidate
        r = f.y - f.off
        adjustment = f.correction
        points += (adjustment*(2.*r[:, None] - adjustment)).sum(axis=0)
        z = part.correction
        fold_delta.append((z*(2.*r[:, None] - z)).sum(axis=0))
        active.append(np.any(abs(z) > 1e-12, axis=0))
    delta = points/sst
    discrepancies = []
    for name in d.V2_NAMES:
        r = ref_v2[name]
        check(r["people"] == n and r["households"] == hh, "V3_V2_COHORT_DIFF")
        inc = delta[d.LIBRARY_INDEX[name]]
        discrepancies.extend((abs(r["r2_off"]-off_r2), abs(r["r2_on"]-(off_r2+inc)),
                              abs(r["delta_r2_pp"]/100.-inc)))
    err = max(discrepancies)
    check(np.isfinite(err) and err <= 2e-8, "V3_V2_R2_REPLAY_DIFF " + "/".join(pair))
    if pair == d.ANCHOR:
        exact = np.concatenate([f.correction[:, d.LIBRARY_INDEX[d.ANCHOR_RECIPE]] for f in forecast])
        check(np.array_equal(c, np.broadcast_to(exact[:, None], c.shape)), "V3_ANCHOR_WAS_CHANGED")
    fixed = tuple(dict(exposure=pair[0], endpoint=pair[1], recipe=name,
                       source="v2" if name in d.V2_NAMES else "v3",
                       r2_off=float(off_r2), r2_on=float(off_r2+inc), delta_r2_pp=float(100.*inc))
                  for name, inc, ok in zip(d.LIBRARY, delta, available, strict=True) if ok)
    return Task(pair, n, hh, aggregate, np.asarray(fold_delta), np.asarray(active),
                2.*(resid @ c)/sst, np.sum(c**2, axis=0)/sst, fixed, err)


def diet_overlap(parts, study):
    """Diagnose exposure differences on the same held people, using fixed PC5."""
    j = d.LIBRARY_INDEX[d.PC5_REFERENCE]
    output = []
    for endpoint in p.ENDPOINTS:
        rows, values = [], []
        for left, right in zip(parts["fiber_density", endpoint], parts["UPF", endpoint], strict=True):
            a, b = left.candidate, right.candidate
            overlap, ia, ib = np.intersect1d(a.rows, b.rows, assume_unique=True, return_indices=True)
            y = a.y[ia]
            check(np.array_equal(y, b.y[ib]), "V3_DIET_OVERLAP_OUTCOME_DIFF")
            rows.append(overlap)
            values.append(np.column_stack((y, a.off[ia], b.off[ib], a.correction[ia, j], b.correction[ib, j])))
        members = np.concatenate(rows)
        y, fa, fb, ca, cb = np.concatenate(values).T
        sst = np.sum((y-y.mean())**2) if len(y) else 0.
        a_gain = float(100.*(ca @ (2.*(y-fa)-ca))/sst) if sst > 0. else None
        b_gain = float(100.*(cb @ (2.*(y-fb)-cb))/sst) if sst > 0. else None
        output.append(dict(endpoint=endpoint, people=len(members),
                           households=len(np.unique(study.family[members])),
                           fiber_pc5_delta_pp=a_gain, upf_pc5_delta_pp=b_gain))
    return output


def scores(sums):
    n, a, b = (sums[..., i] for i in range(3))
    check(np.all(n > 0.), "V3_BOOTSTRAP_EMPTY_ENDPOINT")
    sst = b - a**2/n
    check(np.all(np.isfinite(sst) & (sst > 0.)), "V3_BOOTSTRAP_DEGENERATE_ENDPOINT")
    off = 1. - sums[..., 3]/sst
    delta = sums[..., 4:]/sst[..., None]
    check(np.isfinite(off).all() and np.isfinite(delta).all(), "V3_NONFINITE_SCORE")
    return off, delta


def contrasts(delta):
    primary, selected = (d.WORKFLOWS.index(name) for name in (d.PRIMARY, "select"))
    return np.concatenate((delta, delta[..., 1:] - delta[..., :1],
                           (delta[..., primary] - delta[..., selected])[..., None]), axis=-1)


def paired_intervals(tasks, draws, seed=p.BOOTSTRAP_SEED):
    packed = np.concatenate([t.aggregate for t in tasks], axis=1)
    active = np.any(np.column_stack([t.aggregate[:, 0] > 0. for t in tasks]), axis=1)
    packed = packed[active]
    ng = len(packed)
    check(ng >= 2 and draws >= 20, "V3_BOOTSTRAP_SUPPORT")
    off, delta = scores(np.stack([t.aggregate.sum(axis=0) for t in tasks]))
    theta = contrasts(delta)
    boot = np.empty((draws, *theta.shape))
    stable = (off[:, None] > p.R2_EPS) & (off[:, None] + delta > p.R2_EPS)
    rng = np.random.default_rng(seed)
    prob = np.full(ng, 1./ng)
    for start in range(0, draws, 96):
        count = min(96, draws-start)
        weights = rng.multinomial(ng, prob, size=count)
        totals = (weights @ packed).reshape(count, len(tasks), -1)
        b, inc = scores(totals)
        boot[start:start+count] = contrasts(inc)
        stable &= np.all((b[..., None] > p.R2_EPS) & (b[..., None]+inc > p.R2_EPS), axis=0)
    sd = boot.std(axis=0, ddof=1)
    flat = sd <= 1e-15
    diff = abs(boot-theta)
    flat_radius = diff[:, flat].max(axis=0)
    np.divide(diff, sd, out=diff, where=~flat)
    diff[:, flat] = 0.
    critical = float(np.quantile(diff.max(axis=(1, 2)), 1.-p.ALPHA, method="higher"))
    radius = critical*sd
    radius[flat] = flat_radius
    return off, delta, theta, theta-radius, theta+radius, stable, critical


def inference(tasks, draws=p.BOOTSTRAP_DRAWS):
    base, delta, theta, low, high, valid_share, critical = paired_intervals(tasks, draws)
    results, comparisons = [], []
    for i, task in enumerate(tasks):
        x, e = task.pair
        for j, method in enumerate(d.WORKFLOWS):
            gain = float(delta[i, j])
            off = float(base[i])
            full = off + gain
            results.append(dict(exposure=x, endpoint=e, workflow=method, people=task.people,
                                households=task.households, r2_off=off, r2_on=full,
                                delta_pp=100.*gain, low_pp=float(100.*low[i, j]), high_pp=float(100.*high[i, j]),
                                share_pct=100.*gain/full if valid_share[i, j] else None,
                                active_folds=int(task.active[:, j].sum()),
                                positive_folds=int((task.fold_delta[:, j] > 0.).sum()),
                                aligned_pp=float(100.*task.alignment[j]),
                                correction_cost_pp=float(100.*task.penalty[j])))
        pairings = [(m, "fixed_pc5") for m in d.WORKFLOWS[1:]]
        pairings.append((d.PRIMARY, "select"))
        for j, (on, ref) in enumerate(pairings, start=len(d.WORKFLOWS)):
            comparisons.append(dict(exposure=x, endpoint=e, workflow=on, reference=ref,
                                    difference_pp=float(100.*theta[i, j]),
                                    low_pp=float(100.*low[i, j]), high_pp=float(100.*high[i, j])))
    return dict(draws=draws, simultaneous_cells=int(theta.size), critical=critical, results=results,
                comparisons=comparisons)
