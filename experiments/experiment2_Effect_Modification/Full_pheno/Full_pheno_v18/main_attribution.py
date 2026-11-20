"""Decompose held-out diet main gains and estimate bounded domain amplitudes."""

from dataclasses import replace

import numpy as np


ARMS = ("reference", "fixed_half_main", "ridge_100", "domain_calibrated")
MIN_GAIN = .01
MARGIN = .10


def forecasts(work, fold, diets, values, masks, *, strong=False):
    from batch_fit import clip_outcomes, context_design, match_rows
    from category_fit import left_inverse, panels, residualize
    from run_v2 import require

    bg = work.backgrounds[fold]
    support, plain = context_design(work, fold, "UPF")
    active = getattr(work, "active", np.ones(len(support), dtype=bool))
    panel, _ = panels(diets, work, fold, support)
    target = clip_outcomes(values, bg.role)
    family = np.unique(work.family, return_inverse=True)[1]
    for observed, cols in masks:
        matched = match_rows(observed & support & active, bg.role, family)
        if matched is None:
            continue
        rows, n, _, counts = matched
        require(np.all(active[rows]) and not bg.role[rows[n:]].any(), "V18_WRONG_HOLDOUT_ROWS")
        z = plain[rows]
        left = left_inverse(z[:n])
        y = target[np.ix_(rows, cols)]
        a = z @ (left @ y[:n])
        design = residualize(panel["whole_diet"][rows], z, left, n)
        train = design[:n]
        cov = train.T @ train
        term = train.T @ (y[:n] - a[:n])
        identity = n * np.eye(cov.shape[0])
        ref = design[n:] @ np.linalg.solve(cov + .10 * identity, term)
        alt = design[n:] @ np.linalg.solve(cov + identity, term) if strong else None
        yield cols, y[n:], a[n:], ref, alt, len(counts)


def normalized_terms(work, fold, diets, values, masks):
    p = np.zeros((values.shape[1], 2))
    accepted = np.zeros(values.shape[1], dtype=bool)
    for cols, y, a, ref, _, _ in forecasts(work, fold, diets, values, masks):
        sst = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
        use = sst > 1e-10 * np.maximum(np.finfo(float).tiny, (y * y).sum(axis=0))
        i = cols[use]
        p[i, 0] = 100. * ((y - a) * ref).sum(axis=0)[use] / sst[use]
        p[i, 1] = 100. * (ref * ref).sum(axis=0)[use] / sst[use]
        accepted[i] = True
    return p, accepted


def amplitude(term):
    dot, square = term
    penalty = max(.25 * square, 1e-8)
    return float(np.clip((dot + penalty) / (square + penalty), .25, 1.))


def choose_amplitudes(terms, valid, domain, source):
    scales = np.ones(len(domain))
    record = {}
    for k in sorted(set(domain)):
        members = domain == k
        eligible = members & valid
        item = dict(n=int(eligible.sum()), amplitude=1.)
        if eligible.sum() >= 5:
            by_source = [terms[:, eligible & (source == s)].mean(axis=1) for s in sorted(set(source[eligible]))]
            fold = np.mean(by_source, axis=0)
            gains = []
            for j, (dot, sq) in enumerate(fold):
                a = amplitude(np.delete(fold, j, axis=0).mean(axis=0))
                gains.append(float(2. * (a - 1.) * dot + (1. - a * a) * sq))
            item["crossvalidated_gain_pp"] = gains
            if sum(x > 0. for x in gains) >= 3 and np.mean(gains) >= MIN_GAIN:
                scales[members] = amplitude(fold.mean(axis=0))
                item["amplitude"] = float(scales[members][0])
        record[k] = item
    return scales, record


def inner_scales(work, contexts, calibration, outer, diets, values, masks, domains, sources, seed, emit):
    import protocol as p
    from run_v2 import require

    profiles, covered = [], []
    for inner in p.FOLDS:
        if inner == outer:
            continue
        ctx = contexts.split(outer, inner)
        ctx, _ = calibration.inner(ctx, inner, 100 * seed + 10 * outer + inner)
        bg = ctx.backgrounds[inner]
        ctx.backgrounds[inner] = replace(bg, supported=bg.supported & ctx.active)
        require(np.array_equal(ctx.active & ~bg.role, work.fold == inner)
                and not np.any(bg.role & (work.fold == outer)), "V18_INNER_TRAIN_CONFLICT")
        term, use = normalized_terms(ctx, inner, diets, values, masks)
        profiles.append(term)
        covered.append(use)
        emit("V18_INNER", dict(seed=seed, fold=outer, inner=inner, scored=int(use.sum())))
    return choose_amplitudes(np.stack(profiles), np.logical_and.reduce(covered), domains, sources)


def fold_moments(work, fold, diets, values, masks, scales=None):
    from run_v2 import require

    use_all = scales is not None
    out = np.zeros((len(ARMS) if use_all else 1, values.shape[1], 7))
    for cols, y, a, ref, ridge, families in forecasts(work, fold, diets, values, masks, strong=use_all):
        error = y - a
        delta = (ref, .5 * ref, ridge, ref * scales[cols]) if use_all else (ref,)
        common = (len(y), families, y.sum(axis=0), (y * y).sum(axis=0), (error * error).sum(axis=0))
        for j, d in enumerate(delta):
            dot, sq = (error * d).sum(axis=0), (d * d).sum(axis=0)
            out[j, cols] = np.column_stack(np.broadcast_arrays(*common, dot, sq))
    require(np.isfinite(out).all() and np.all(out[:, :, :5] == out[:1, :, :5]), "V18_COMMON_BASELINE_CHANGE")
    return out


def score(folds, endpoints, methods):
    x = np.stack(folds, axis=2)
    people, family, sy, sy2, _, dot, sq = x.sum(axis=2).transpose(2, 0, 1)
    n = people[0]
    sst = sy2[0] - np.divide(sy[0] ** 2, n, out=np.zeros_like(n), where=n > 0.)
    enough = ((x[0, :, :, 0] > 0.).all(axis=1)
              & (sst > 1e-10 * np.maximum(np.finfo(float).tiny, abs(sy2[0]))))
    gain = 2. * dot - sq
    signs = ((2. * x[:, :, :, 5] - x[:, :, :, 6]) > 0.).sum(axis=2)
    rows = []
    for arm, method in enumerate(methods):
        for e in np.flatnonzero(enough):
            scale = 100. / sst[e]
            rows.append(dict(arm=method, endpoint=endpoints[e], people=int(n[e]), households=int(family[arm, e]),
                             main_pp=float(scale * gain[arm, e]), alignment_pp=float(2. * scale * dot[arm, e]),
                             size_cost_pp=float(scale * sq[arm, e]), positive_folds=int(signs[arm, e]),
                             versus_ref_pp=float(scale * (gain[arm, e] - gain[0, e]))))
    return rows
