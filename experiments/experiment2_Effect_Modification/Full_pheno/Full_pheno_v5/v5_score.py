"""Aggregate held predictions and reuse v4's shared household bootstrap."""

import numpy as np

import protocol as p
import v4_design as v4
import v5_design as d
from source import check
from v4_score import Task, bootstrap, scores


def collect(key, folds, study, codes, n_groups, refs):
    rows = np.concatenate([f.rows for f in folds])
    n = len(rows)
    g = codes[rows]
    h = len(np.unique(g))
    check(len(np.unique(rows)) == n and len(folds) == len(p.FOLDS)
          and all(np.all(study.fold[f.rows] == k) for k, f in zip(p.FOLDS, folds, strict=True)),
          "V5_HELD_ROWS_DIFF")
    y = np.concatenate([f.y for f in folds])
    off = np.concatenate([f.off for f in folds])
    no_diet = np.concatenate([f.no_diet for f in folds])
    corr = np.concatenate([f.correction for f in folds])
    centered = y-y.mean()
    r, step = y-off, off-no_diet
    cols = np.column_stack((np.ones(n), centered, centered**2, r**2, step*(2.*r+step),
                            corr*(2.*r[:, None]-corr)))
    check(np.isfinite(cols).all(), "V5_NONFINITE_HELD_MOMENTS")
    aggregate = np.column_stack([np.bincount(g, weights=col, minlength=n_groups) for col in cols.T])
    b, main, inc = scores(aggregate.sum(axis=0))
    discrepancies = []
    for method, reference in d.REPLAY.items():
        old = refs[key, reference]
        j = d.WORKFLOWS.index(method)
        check(old["people"] == n and old["households"] == h, "V5_V4_COHORT_DIFF")
        discrepancies.extend((abs(b-old["r2_off"]), abs(b-main-old["r2_no_diet"]),
                              abs(b+inc[j]-old["r2_on"]), abs(inc[j]-old["delta_pp"]/100.)))
    err = float(max(discrepancies))
    check(np.isfinite(err) and err <= 2e-8, "V5_V4_REPLAY_DIFF " + "/".join(key))
    if key == d.ANCHOR:
        check(np.array_equal(corr, np.broadcast_to(corr[:, :1], corr.shape)), "V5_ANCHOR_CHANGED")
    fold_gains = np.stack([np.sum(f.correction*(2.*(f.y-f.off)[:, None]-f.correction), axis=0) for f in folds])
    available = folds[0].available
    check(all(np.array_equal(f.available, available) for f in folds), "V5_LIBRARY_SUPPORT_DIFF")
    library_delta = np.sum([f.library_gain for f in folds], axis=0)/(centered @ centered)
    library = [dict(exposure=key[0], endpoint=key[1], recipe=name, delta_pp=float(100.*x),
                    r2_no_diet=float(b-main), r2_off=float(b), r2_on=float(b+x),
                    total_diet_pp=float(100.*(main+x)))
               for name, x, ok in zip(d.LIBRARY, library_delta, available, strict=True)
               if ok and key != d.ANCHOR]
    return Task(key, n, h, aggregate, fold_gains, library, err)


def inference(tasks, draws=p.BOOTSTRAP_DRAWS):
    off, main, inc, points, lo, hi, full_ok, diet_ok, critical = bootstrap(tasks, draws)
    rows, compared = [], []
    m = len(d.WORKFLOWS)
    for i, task in enumerate(tasks):
        exposure, endpoint = task.key
        b, a = float(off[i]), float(main[i])
        for j, method in enumerate(d.WORKFLOWS):
            gain = float(inc[i, j])
            full, total = b+gain, a+gain
            full_share = 100.*gain/full if b > p.R2_EPS and full > p.R2_EPS else None
            diet_share = 100.*gain/total if total > p.R2_EPS else None
            slot = 2*m-1+j
            rows.append(dict(
                exposure=exposure, endpoint=endpoint, workflow=method, people=task.people,
                households=task.households, r2_no_diet=b-a, r2_off=b, r2_on=full,
                delta_pp=100.*gain, low_pp=float(100.*lo[i, j]), high_pp=float(100.*hi[i, j]),
                main_diet_pp=100.*a, total_diet_pp=100.*total,
                diet_low_pp=float(100.*lo[i, slot]), diet_high_pp=float(100.*hi[i, slot]),
                full_share_point_pct=full_share, full_share_pct=full_share if full_ok[i, j] else None,
                diet_share_point_pct=diet_share, diet_share_pct=diet_share if diet_ok[i, j] else None,
                folds_positive=int(np.count_nonzero(task.fold_gains[:, j] > 0.)),
            ))
        for j, method in enumerate(d.WORKFLOWS[1:], start=m):
            compared.append(dict(exposure=exposure, endpoint=endpoint, workflow=method,
                                 reference=d.REFERENCE, difference_pp=float(100.*points[i, j]),
                                 low_pp=float(100.*lo[i, j]), high_pp=float(100.*hi[i, j])))
    return dict(draws=draws, simultaneous_cells=int(points.size), critical=critical,
                diet_floor_pp=100.*v4.RATIO_DIET_FLOOR, rows=rows, comparisons=compared)
