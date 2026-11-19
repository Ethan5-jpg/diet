"""Three-arm scoring and paired household resampling."""

from dataclasses import dataclass

import numpy as np

import protocol as p
import v4_design as d
from source import check


@dataclass(frozen=True)
class Task:
    key: tuple
    people: int
    households: int
    aggregate: np.ndarray
    fold_gains: np.ndarray
    fixed: list
    replay_error: float


def scores(moments):
    n, a, b = (moments[..., i] for i in range(3))
    check(np.all(n > 0.), "V4_EMPTY_ENDPOINT_RESAMPLE")
    sst = b-a**2/n
    check(np.all(np.isfinite(sst) & (sst > 0.)), "V4_DEGENERATE_OUTCOME")
    off = 1. - moments[..., 3]/sst
    diet_main = moments[..., 4]/sst
    delta = moments[..., 5:]/sst[..., None]
    check(all(np.isfinite(x).all() for x in (off, diet_main, delta)), "V4_NONFINITE_SCORE")
    return off, diet_main, delta


def collect(key, folds, study, codes, households, previous):
    rows = np.concatenate([f.rows for f in folds])
    g = codes[rows]
    n, h = len(rows), len(np.unique(g))
    check(len(np.unique(rows)) == n and len(folds) == len(p.FOLDS) and
          all(np.all(study.fold[f.rows] == k) for k, f in zip(p.FOLDS, folds, strict=True)),
          "V4_HELD_ROWS_DIFFER")
    y = np.concatenate([f.y for f in folds])
    off = np.concatenate([f.off for f in folds])
    no_diet = np.concatenate([f.no_diet for f in folds])
    c = np.concatenate([f.corrections for f in folds])
    err, change = y-off, off-no_diet
    residual = y-y.mean()
    diet = change*(2.*err+change)
    gains = c*(2.*err[:, None]-c)
    columns = np.column_stack((np.ones(n), residual, residual**2, err**2, diet, gains))
    check(np.isfinite(columns).all(), "V4_NONFINITE_HELD_MOMENTS")
    aggregate = np.column_stack([np.bincount(g, weights=col, minlength=households) for col in columns.T])
    base, diet_main, inc = scores(aggregate.sum(axis=0))
    err_replay = []
    for name, ref in ((d.REFERENCE, "fixed_pc5"), ("v3_guarded", "blend_guarded")):
        old = previous[key, ref]
        j = d.WORKFLOWS.index(name)
        check(old["people"] == n and old["households"] == h, "V4_V3_COHORT_MISMATCH")
        err_replay.extend((abs(base-old["r2_off"]), abs(base+inc[j]-old["r2_on"]),
                           abs(inc[j]-old["delta_pp"]/100.)))
    err_replay = float(max(err_replay))
    check(np.isfinite(err_replay) and err_replay <= 2e-8, "V4_V3_REPLAY_DIFF " + "/".join(key))
    if key == d.ANCHOR:
        for j, method in enumerate(d.WORKFLOWS):
            if method not in d.UNIFORM:
                check(np.array_equal(c[:, j], c[:, 0]), "V4_ANCHOR_WAS_CHANGED")
    gains_by_fold = np.stack([np.sum(f.corrections*(2.*(f.y-f.off)[:, None]-f.corrections), axis=0)
                              for f in folds])
    mask = folds[0].available
    check(all(np.array_equal(f.available, mask) for f in folds), "V4_FIXED_LIBRARY_SUPPORT_DIFFERS")
    fixed_delta = np.sum([f.library_gain for f in folds], axis=0) / (residual @ residual)
    library = [dict(exposure=key[0], endpoint=key[1], recipe=name, r2_no_diet=float(base-diet_main),
                    r2_off=float(base), r2_on=float(base+x), delta_pp=float(100.*x),
                    total_diet_pp=float(100.*(diet_main+x)))
               for name, x, ok in zip(d.LIBRARY, fixed_delta, mask, strict=True) if ok and key != d.ANCHOR]
    return Task(key, n, h, aggregate, gains_by_fold, library, err_replay)


def contrasts(main, inc):
    return np.concatenate((inc, inc[..., 1:]-inc[..., :1], main[..., None]+inc), axis=-1)


def bootstrap(tasks, draws, seed=p.BOOTSTRAP_SEED):
    union = np.any(np.column_stack([t.aggregate[:, 0] > 0. for t in tasks]), axis=1)
    packed = np.concatenate([t.aggregate for t in tasks], axis=1)[union]
    n = len(packed)
    check(n >= 2 and draws >= 20, "V4_INSUFFICIENT_BOOTSTRAP_SUPPORT")
    off, main, inc = scores(np.stack([t.aggregate.sum(axis=0) for t in tasks]))
    center = contrasts(main, inc)
    all_draws = np.empty((draws, *center.shape))
    full_ok = (off[:, None] > p.R2_EPS) & (off[:, None]+inc > p.R2_EPS)
    diet_ok = main[:, None]+inc >= d.RATIO_DIET_FLOOR
    rng = np.random.default_rng(seed)
    prob = np.full(n, 1./n)
    for start in range(0, draws, 96):
        k = min(96, draws-start)
        w = rng.multinomial(n, prob, size=k)
        b, m, f = scores((w @ packed).reshape(k, len(tasks), -1))
        all_draws[start:start+k] = contrasts(m, f)
        full_ok &= np.all((b[..., None] > p.R2_EPS) & (b[..., None]+f > p.R2_EPS), axis=0)
        diet_ok &= np.all(m[..., None]+f >= d.RATIO_DIET_FLOOR, axis=0)
    sd = all_draws.std(axis=0, ddof=1)
    flat = sd <= 1e-15
    errors = abs(all_draws-center)
    flat_radius = errors[:, flat].max(axis=0)
    np.divide(errors, sd, out=errors, where=~flat)
    errors[:, flat] = 0.
    critical = float(np.quantile(errors.max(axis=(1, 2)), 1.-p.ALPHA, method="higher"))
    radius = critical*sd
    radius[flat] = flat_radius
    return off, main, inc, center, center-radius, center+radius, full_ok, diet_ok, critical


def inference(tasks, draws=p.BOOTSTRAP_DRAWS):
    off, main, inc, center, low, high, full_ok, diet_ok, critical = bootstrap(tasks, draws)
    result, paired = [], []
    m = len(d.WORKFLOWS)
    for i, task in enumerate(tasks):
        x, e = task.key
        diet_main = float(main[i])
        base = float(off[i])
        for j, method in enumerate(d.WORKFLOWS):
            gain = float(inc[i, j])
            full = base+gain
            total = diet_main+gain
            full_pct = 100.*gain/full if base > p.R2_EPS and full > p.R2_EPS else None
            diet_pct = 100.*gain/total if total > p.R2_EPS else None
            slot = 2*m-1+j
            result.append(dict(
                exposure=x, endpoint=e, workflow=method, people=task.people, households=task.households,
                r2_no_diet=base-diet_main, r2_off=base, r2_on=full,
                delta_pp=100.*gain, low_pp=float(100.*low[i, j]), high_pp=float(100.*high[i, j]),
                main_diet_pp=100.*diet_main, total_diet_pp=100.*total,
                diet_low_pp=float(100.*low[i, slot]), diet_high_pp=float(100.*high[i, slot]),
                full_share_point_pct=full_pct, full_share_pct=full_pct if full_ok[i, j] else None,
                diet_share_point_pct=diet_pct, diet_share_pct=diet_pct if diet_ok[i, j] else None,
                folds_positive=int((task.fold_gains[:, j] > 0.).sum()),
            ))
        for j, method in enumerate(d.WORKFLOWS[1:], start=m):
            paired.append(dict(exposure=x, endpoint=e, workflow=method, reference=d.REFERENCE,
                               difference_pp=float(100.*center[i, j]), low_pp=float(100.*low[i, j]),
                               high_pp=float(100.*high[i, j])))
    return dict(draws=draws, simultaneous_cells=int(center.size), critical=critical, rows=result, comparisons=paired)
