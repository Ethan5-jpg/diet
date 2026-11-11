"""Fit a common microbial interaction across fixed alternative meal times."""

import numpy as np

ENDPOINTS = ("net_m15", "net_p0", "net_p15")
PENALTY = .03
SEED = 202611132


def fit_shared(matrices, term, training, model):
    """Average the offset-specific residual objectives with the original penalty."""
    terms = term[training]
    w = matrices[ENDPOINTS[0], "carbs"][2]
    penalty = PENALTY * w[training].sum() * np.eye(term.shape[1])
    fits, grams, targets = {}, [], []
    for endpoint in ENDPOINTS:
        y, x, weights = matrices[endpoint, "carbs"]
        if not np.array_equal(w, weights):
            raise ValueError("All three offsets must use the same meal weights")
        fit = model.fit_background(x[training], y[training], w[training])
        residual, target, projection = model.orthogonal_block(terms, fit)
        gram, rhs = residual.T @ residual, residual.T @ target
        fits[endpoint] = (fit[0], projection, np.linalg.solve(gram + penalty, rhs))
        grams.append(gram)
        targets.append(rhs)
    shared = np.linalg.solve(np.mean(grams, axis=0) + penalty, np.mean(targets, axis=0))
    return fits, shared


def household_losses(matrices, term, fits, shared, valid, family, groups):
    result = {}
    for endpoint, (base, projection, separate) in fits.items():
        y, x, w = matrices[endpoint, "carbs"]
        y, x, w = y[valid], x[valid], w[valid]
        residual = y - x @ base
        features = term[valid] - x @ projection
        parts = (residual ** 2, (residual - features @ separate) ** 2,
                 (residual - features @ shared) ** 2, y ** 2)
        result[endpoint] = np.stack([np.bincount(family[valid], weights=w * value, minlength=groups)
                                    for value in parts])
    return result


def summarize(arrays, seed):
    result = {}
    for endpoint in ENDPOINTS:
        none, separate, shared, tss = arrays[endpoint]
        support = np.flatnonzero(np.any(arrays[endpoint] != 0, axis=0))
        if len(support) < 2 or not np.isfinite(arrays[endpoint]).all():
            raise ValueError("Insufficient valid household losses")
        draws = np.random.default_rng(seed).choice(support, (1999, len(support)), replace=True)
        denom = tss[draws].sum(axis=1)
        if not (tss.sum() > 0 and (denom > 0).all()):
            raise ValueError("Nonpositive held-out loss denominator")
        row = {"baseline_r2": float(1 - none.sum() / tss.sum())}
        for name, gain in (
            ("separate_vs_baseline", none - separate),
            ("shared_vs_baseline", none - shared),
            ("shared_vs_separate", separate - shared),
        ):
            row[name] = dict(delta_r2=float(gain.sum() / tss.sum()),
                             ci95=np.quantile(gain[draws].sum(axis=1) / denom, [.025, .975]).tolist())
        result[endpoint] = row
    return result
