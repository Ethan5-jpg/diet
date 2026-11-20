"""Pair each expanded diet main with the same interaction design and strengths."""

import numpy as np


MAINS = ("reference", "fixed_half_main", "ridge_100", "domain_calibrated")


def predict(plain, gut, diet, y, scales, n, mains, interactions):
    from category_fit import left_inverse, residualize

    left = left_inverse(plain[:n])
    a = plain @ (left @ y[:n])
    z = residualize(diet, plain, left, n)
    train = z[:n]
    covariance = train.T @ train
    cross = train.T @ (y[:n] - a[:n])
    regularizer = n * np.eye(train.shape[1])
    contribution = z @ np.linalg.solve(covariance + .10 * regularizer, cross)
    b = {"reference": a + contribution}
    if "fixed_half_main" in mains:
        b["fixed_half_main"] = a + .50 * contribution
    if "ridge_100" in mains:
        b["ridge_100"] = a + z @ np.linalg.solve(covariance + regularizer, cross)
    if "domain_calibrated" in mains:
        b["domain_calibrated"] = a + contribution * scales
    product = (diet[:, :, None] * gut[:, None, :]).reshape(len(plain), -1)
    g = residualize(residualize(product, plain, left, n), z, left_inverse(train), n)
    x = g[:n]
    residuals = np.column_stack([y[:n] - b[m][:n] for m in mains])
    beta = np.linalg.solve(x.T @ x + n * np.eye(x.shape[1]), x.T @ residuals)
    effect = np.split(g[n:] @ beta, len(mains), axis=1)
    scale = np.maximum(1., np.abs(effect[0]).max(axis=0))
    gap = max(float((np.abs(change - effect[0]) / scale).max()) for change in effect)
    fits = [(a[n:], b[main][n:], b[main][n:] + factor * change)
            for main, change in zip(mains, effect, strict=True) for factor in interactions.values()]
    return fits, gap


def collect(work, fold, diets, values, masks, scales, mains=MAINS):
    import protocol as p
    from batch_fit import clip_outcomes, context_design, fold_moments, match_rows
    from category_fit import panels
    from main_fit import INTERACTIONS
    from run_v2 import require

    bg = work.backgrounds[fold]
    support, plain = context_design(work, fold, "UPF")
    diet, _ = panels(diets, work, fold, support)
    y = clip_outcomes(values, bg.role)
    family = np.unique(work.family, return_inverse=True)[1]
    metrics = np.zeros((len(mains) * len(INTERACTIONS), values.shape[1], 10))
    max_gap = 0.
    for observed, cols in masks:
        matched = match_rows(observed & support, bg.role, family)
        if matched is None:
            continue
        rows, n, codes, counts = matched
        require(np.all(work.fold[rows[n:]] == fold) and np.all(work.fold[rows[:n]] != fold),
                "V19_WRONG_HOUSEHOLD_MEMBERSHIP")
        target = y[np.ix_(rows, cols)]
        fits, gap = predict(plain[rows], bg.pcs[rows, p.N_PC:p.N_PC + 2], diet["whole_diet"][rows],
                            target, scales[cols], n, mains, INTERACTIONS)
        max_gap = max(max_gap, gap)
        for j, bundle in enumerate(fits):
            metrics[j, cols] = np.column_stack(fold_moments(target[n:], bundle, codes, counts))
    require(np.isfinite(metrics).all() and np.all(metrics[:, :, :5] == metrics[:1, :, :5]),
            "V19_COMMON_NO_DIET_MODEL_CHANGED")
    return metrics, max_gap
