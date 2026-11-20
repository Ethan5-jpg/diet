"""Compare calibrated conditionals and keep signed loss-decomposition terms."""

import numpy as np

from taxonomy import GROUPS


WHOLE = "whole_half"
OFF = "interaction_off"
METHODS = (WHOLE, *GROUPS, OFF)
BASELINES = ("reference", "domain_calibrated")


def predict(host, gut, diet, y, alpha, n):
    from category_fit import left_inverse, residualize

    full = diet["whole_diet"]
    left = left_inverse(host[:n])
    a = host @ (left @ y[:n])
    z = residualize(full, host, left, n)
    train = z[:n]
    contribution = z @ np.linalg.solve(
        train.T @ train + .10 * n * np.eye(train.shape[1]), train.T @ (y[:n] - a[:n]))
    baseline = (a + contribution, a + alpha * contribution)
    response = np.column_stack([y[:n] - b[:n] for b in baseline])
    gut = gut[:, None, :]
    main_left = left_inverse(train)
    arms = [[], []]
    for kind in ("whole_diet", *GROUPS):
        raw = (diet[kind][:, :, None] * gut).reshape(len(host), -1)
        interactions = residualize(residualize(raw, host, left, n), z, main_left, n)
        h = interactions[:n]
        coef = np.linalg.solve(h.T @ h + n * np.eye(h.shape[1]), h.T @ response)
        correction = .25 * (interactions[n:] @ coef)
        for j, u in enumerate(np.split(correction, len(baseline), axis=1)):
            arms[j].append(u)
    return a[n:], [b[n:] for b in baseline], arms


def collect(work, fold, diets, values, masks, scales):
    import protocol as p
    from batch_fit import clip_outcomes, context_design, fold_moments, match_rows
    from category_fit import panels
    from run_v2 import require

    bg = work.backgrounds[fold]
    supported, plain = context_design(work, fold, "UPF")
    panels_by_class, _ = panels(diets, work, fold, supported)
    target = clip_outcomes(values, bg.role)
    family = np.unique(work.family, return_inverse=True)[1]
    matrix = np.zeros((len(BASELINES), len(METHODS), values.shape[1], 12))
    for valid, cols in masks:
        matched = match_rows(supported & valid, bg.role, family)
        if matched is None:
            continue
        rows, n, codes, counts = matched
        require(np.all(work.fold[rows[n:]] == fold) and np.all(work.fold[rows[:n]] != fold),
                "V20_HELD_HOUSEHOLD_MISMATCH")
        y = target[np.ix_(rows, cols)]
        a, baseline, forecasts = predict(plain[rows], bg.pcs[rows, p.N_PC:p.N_PC+2],
                                         {k: panels_by_class[k][rows] for k in ("whole_diet", *GROUPS)},
                                         y, scales[cols], n)
        held = y[n:]
        for j, (b, updates) in enumerate(zip(baseline, forecasts, strict=True)):
            main_dot = 2. * ((held - a) * (b - a)).sum(axis=0)
            for i, delta in enumerate((*updates, np.zeros_like(b))):
                c = b + delta
                matrix[j, i, cols, :10] = np.column_stack(fold_moments(held, (a, b, c), codes, counts))
                matrix[j, i, cols, 10] = main_dot
                matrix[j, i, cols, 11] = 2. * ((held - b) * delta).sum(axis=0)
    require(np.isfinite(matrix).all()
            and np.all(matrix[:, :, :, :5] == matrix[:1, :1, :, :5])
            and np.all(matrix[:, :, :, 5] == matrix[:, :1, :, 5]), "V20_BASELINE_CHANGED")
    return matrix


def score(folds, endpoints):
    from adapt_interactions import score_partition
    from run_expanded_grid import holm

    joined = np.stack(folds, axis=3)
    n, _, y, square, *_ = joined[0, 0].sum(axis=1).T
    variance = square - np.divide(y*y, n, out=np.zeros_like(n), where=n>0.)
    norm = np.divide(100., variance, out=np.zeros_like(variance), where=variance>0.)
    lookup = {e: j for j, e in enumerate(endpoints)}
    method = {m: j for j, m in enumerate(METHODS)}
    rows = []
    for i, baseline in enumerate(BASELINES):
        outcome = score_partition(joined[i, :, :, :, :10], endpoints, METHODS)
        for r in outcome:
            j, k = method[r["diet"]], lookup[r["endpoint"]]
            r["main_alignment_pp"], r["interaction_alignment_pp"] = (
                float(x * norm[k]) for x in joined[i, j, k, :, 10:].sum(axis=0))
            r["diet"] = f"{baseline}:{r['diet']}"
        rows.extend(outcome)
    holm(rows)
    return rows
