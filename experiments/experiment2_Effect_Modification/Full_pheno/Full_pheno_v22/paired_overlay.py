"""Fit the reference interaction under both V21 diet scales on identical samples."""

import numpy as np


ARMS = ("half", "off")


def predict(host, gut, diet, y, scales, n):
    from category_fit import left_inverse, residualize

    left = left_inverse(host[:n])
    a = host @ (left @ y[:n])
    z = residualize(diet, host, left, n)
    train = z[:n]
    d = z @ np.linalg.solve(train.T @ train + .10 * n * np.eye(train.shape[1]),
                            train.T @ (y[:n] - a[:n]))
    b = [a + d * alpha for alpha in scales]
    raw = (diet[:, :, None] * gut[:, None, :]).reshape(len(host), -1)
    h = residualize(residualize(raw, host, left, n), z, left_inverse(train), n)
    x = h[:n]
    y_residual = np.column_stack([y[:n] - p[:n] for p in b])
    coef = np.linalg.solve(x.T @ x + n * np.eye(x.shape[1]), x.T @ y_residual)
    effects = np.split(.25 * (h[n:] @ coef), len(b), axis=1)
    scale = np.maximum(1., np.abs(effects[0]).max(axis=0))
    gap = max(float((abs(c - effects[0]) / scale).max()) for c in effects)
    return a[n:], [p[n:] for p in b], effects, gap


def collect(work, fold, diets, values, masks, scales):
    import protocol as p
    from batch_fit import clip_outcomes, context_design, fold_moments, match_rows
    from category_fit import panels
    from run_v2 import require

    bg = work.backgrounds[fold]
    support, plain = context_design(work, fold, "UPF")
    panel, _ = panels(diets, work, fold, support)
    outcome = clip_outcomes(values, bg.role)
    family = np.unique(work.family, return_inverse=True)[1]
    metrics = np.zeros((len(scales),len(ARMS),values.shape[1],10))
    max_gap = 0.
    for observed, cols in masks:
        matched = match_rows(observed & support, bg.role, family)
        if matched is None:
            continue
        rows, n, codes, sizes = matched
        require(np.all(work.fold[rows[n:]] == fold) and np.all(work.fold[rows[:n]] != fold),
                "V22_HELD_HOUSEHOLD_CHANGED")
        y = outcome[np.ix_(rows, cols)]
        a, baseline, effects, gap = predict(plain[rows], bg.pcs[rows, p.N_PC:p.N_PC+2],
                                            panel["whole_diet"][rows], y, scales[:, cols], n)
        max_gap = max(max_gap, gap)
        for i, (b, c) in enumerate(zip(baseline, effects, strict=True)):
            for j, target in enumerate((b + c, b)):
                metrics[i,j,cols] = np.column_stack(fold_moments(y[n:], (a,b,target), codes, sizes))
    require(np.isfinite(metrics).all()
            and np.all(metrics[:,:,:,:5] == metrics[:1,:1,:,:5])
            and np.all(metrics[:,:,:,5] == metrics[:,:1,:,5]), "V22_COMMON_ARMS_CHANGED")
    return metrics, max_gap


def score(frames, endpoints, main_names):
    from adapt_interactions import score_partition
    from run_expanded_grid import holm

    by_method = np.stack(frames, axis=3)
    rows = []
    for m, values in zip(main_names, by_method, strict=True):
        r = score_partition(values, endpoints, ARMS)
        for e in r:
            e.pop("vs_fixed_pp")
            e["diet"] = f"{m}:{e['diet']}"
        rows.extend(r)
    holm(rows)
    return rows
