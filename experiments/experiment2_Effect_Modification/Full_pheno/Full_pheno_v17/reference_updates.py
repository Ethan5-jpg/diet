"""Fit bounded category adjustments around one shared whole-diet interaction."""

from dataclasses import replace

import numpy as np

from taxonomy import GROUPS


REFERENCE_HALF = "whole_half"
OFF = "interaction_off"
ADJUST = "anchored_categories"
DIARY = "diary_person"
DAYS = "diary_days"
MEAN = "diary_mean"
METHODS = (REFERENCE_HALF, OFF, ADJUST, DIARY, DAYS, MEAN, *GROUPS)
GAIN = .005
LIMIT = .25


def shared_fit(plain, pcs, groups, y, n, *, singles=False):
    from category_fit import left_inverse, residualize

    diet = np.column_stack(groups)
    left = left_inverse(plain[:n])
    a = plain @ (left @ y[:n])
    z = residualize(diet, plain, left, n)
    train = z[:n]
    b = a + z @ np.linalg.solve(train.T @ train + .10 * n * np.eye(z.shape[1]), train.T @ (y[:n] - a[:n]))
    raw = (diet[:, :, None] * pcs[:, None, :]).reshape(len(plain), -1)
    h = residualize(residualize(raw, plain, left, n), z, left_inverse(train), n)
    r = y[:n] - b[:n]

    def coefficients(design):
        train = design[:n]
        return np.linalg.solve(train.T @ train + n * np.eye(train.shape[1]), train.T @ r)

    c = coefficients(h)
    ref = b[n:] + .25 * (h[n:] @ c)
    slices = np.split(np.arange(h.shape[1]), np.cumsum([u.shape[1] * pcs.shape[1] for u in groups])[:-1])
    pieces = np.stack([.25 * (h[n:, columns] @ c[columns]) for columns in slices], axis=2)
    single = []
    if singles:
        for columns in slices:
            design = h[:, columns]
            single.append(b[n:] + .25 * (design[n:] @ coefficients(design)))
    return a[n:], b[n:], ref, pieces, single


def match_forecasts(work, fold, diets, values, masks, *, singles=False, diary=None):
    import protocol as p
    from batch_fit import clip_outcomes, context_design, match_rows
    from category_fit import panels
    from diary_reliability import reliability_weights
    from run_v2 import require

    bg = work.backgrounds[fold]
    support, plain = context_design(work, fold, "UPF")
    active = getattr(work, "active", np.ones(len(support), dtype=bool))
    scores, _ = panels(diets, work, fold, support)
    day_model = None if diary is None else reliability_weights(
        diary, work.numeric["n_energy_days"], support & bg.role)
    y = clip_outcomes(values, bg.role)
    family = np.unique(work.family, return_inverse=True)[1]
    for obs, cols in masks:
        matched = match_rows(obs & support & active, bg.role, family)
        if matched is None:
            continue
        rows, n, codes, counts = matched
        require(np.all(active[rows]) and not np.any(bg.role[rows[n:]]), "V17_WRONG_HELD_HOUSEHOLDS")
        z = y[np.ix_(rows, cols)]
        a, b, ref, terms, single = shared_fit(plain[rows], bg.pcs[rows, p.N_PC:p.N_PC + 2],
                                             [scores[k][rows] for k in GROUPS], z, n, singles=singles)
        require(np.allclose(terms.sum(axis=2), ref - b, rtol=1e-8, atol=1e-9),
                "V17_REFERENCE_CATEGORY_DECOMPOSITION")
        yield cols, z[n:], a, b, ref, terms, single, rows[n:], codes, counts, day_model


def inner_profile(work, fold, diets, values, masks):
    width = len(GROUPS)
    result = np.zeros((values.shape[1], width + 1, width))
    valid = np.zeros(values.shape[1], dtype=bool)
    for cols, y, _, _, ref, terms, *_ in match_forecasts(work, fold, diets, values, masks):
        sst = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
        usable = sst > 1e-10 * np.maximum(np.finfo(float).tiny, (y * y).sum(axis=0))
        i = cols[usable]
        x = terms[:, usable]
        factor = 100. / sst[usable]
        result[i, 0] = np.einsum("nkg,nk->kg", x, (y - ref)[:, usable]) * factor[:, None]
        result[i, 1:] = np.einsum("nkg,nkj->kgj", x, x) * factor[:, None, None]
        valid[i] = True
    return result, valid


def bounded_fit(profile):
    from scipy.optimize import minimize
    from run_v2 import require

    cross, gram = profile[0], profile[1:]
    penalty = max(float(np.trace(gram) / len(GROUPS)), 1e-6)
    objective = gram + penalty * np.eye(len(GROUPS))

    def loss(delta):
        return float(delta @ objective @ delta - 2. * cross @ delta), 2. * (objective @ delta - cross)

    fit = minimize(loss, np.zeros(len(GROUPS)), jac=True, method="L-BFGS-B",
                   bounds=[(-LIMIT, LIMIT)] * len(GROUPS), options={"gtol": 1e-9, "ftol": 1e-12})
    require(fit.success and np.isfinite(fit.x).all(), "V17_CATEGORY_WEIGHT_SOLVER")
    return fit.x


def choose_adjustments(profiles, valid, domains, sources):
    profiles = np.asarray(profiles)
    result = np.zeros((len(domains), len(GROUPS)))
    audit = {}
    for domain in sorted(set(domains)):
        member = domains == domain
        eligible = member & valid
        record = dict(common=int(eligible.sum()), active=False)
        if eligible.sum() >= 5:
            by_source = [profiles[:, eligible & (sources == s)].mean(axis=1) for s in sorted(set(sources[eligible]))]
            fold = np.mean(by_source, axis=0)
            held_gains = []
            for j in range(len(fold)):
                delta = bounded_fit(np.delete(fold, j, axis=0).mean(axis=0))
                cross, gram = fold[j, 0], fold[j, 1:]
                held_gains.append(float(2. * cross @ delta - delta @ gram @ delta))
            record["crossfit_gain_pp"] = held_gains
            if sum(x > 0. for x in held_gains) >= 3 and np.mean(held_gains) >= GAIN:
                delta = bounded_fit(fold.mean(axis=0))
                result[member] = delta
                record.update(active=True, weights=dict(zip(GROUPS, (1. + delta).tolist(), strict=True)))
        audit[domain] = record
    return result, audit


def train_adjustments(work, contexts, calibration, fold, diets, values, masks, domains, sources, seed, emit):
    import protocol as p
    from run_v2 import require

    parts, accepted = [], []
    for inner in p.FOLDS:
        if inner == fold:
            continue
        view = contexts.split(fold, inner)
        view, _ = calibration.inner(view, inner, 100 * seed + 10 * fold + inner)
        bg = view.backgrounds[inner]
        view.backgrounds[inner] = replace(bg, supported=bg.supported & view.active)
        require(np.array_equal(view.active & ~bg.role, work.fold == inner)
                and not np.any(bg.role & (work.fold == fold)), "V17_CALIBRATION_LEAKAGE")
        x, ok = inner_profile(view, inner, diets, values, masks)
        parts.append(x)
        accepted.append(ok)
        emit("V17_INNER", dict(seed=seed, fold=fold, inner=inner, scorable=int(ok.sum())))
    return choose_adjustments(parts, np.logical_and.reduce(accepted), domains, sources)


def fold_scores(work, fold, diets, values, masks, adjustment=None, diary=None):
    from batch_fit import fold_moments
    from run_v2 import require

    full = adjustment is not None
    moments = np.zeros((len(METHODS) if full else 1, values.shape[1], 10))
    audit = None
    for cols, y, a, b, ref, terms, singles, rows, codes, counts, reliability in match_forecasts(
        work, fold, diets, values, masks, singles=full, diary=diary,
    ):
        forecasts = (ref,)
        if full:
            weight, average, coverage, diagnostics = reliability
            if audit is None:
                audit = dict(average_by_category=dict(zip(GROUPS, average.tolist(), strict=True)),
                             features=diagnostics)
            forecasts = (
                ref, b,
                ref + np.einsum("nkg,kg->nk", terms, adjustment[cols]),
                ref + np.einsum("nkg,ng->nk", terms, weight[rows] - 1.),
                b + coverage[rows, None] * (ref - b),
                ref + np.einsum("nkg,g->nk", terms, average - 1.),
                *singles,
            )
        for j, c in enumerate(forecasts):
            moments[j, cols] = np.column_stack(fold_moments(y, (a, b, c), codes, counts))
    require(np.isfinite(moments).all() and np.all(moments[:, :, :6] == moments[:1, :, :6]),
            "V17_MAIN_BASELINE_CHANGED")
    return moments, audit
