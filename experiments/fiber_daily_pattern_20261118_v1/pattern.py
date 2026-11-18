"""Prespecified split-date fiber variability extensions. All exposure values are in 5 g/1000 kcal units."""
import numpy as np


ROUTES = ("original_off", "original_fixed", "matched_off", "matched_fixed", "daily_hinge", "spread", "combined")
NEW = ROUTES[4:]


def daily_moments(fiber):
    centered = (np.asarray(fiber) - 12.5) / 5.
    return np.column_stack((np.maximum(centered, 0), centered ** 2))


def pattern(mean_fiber, moments, counts):
    """Calculate mean positive part and sample SD using exposure-period dates only."""
    if np.any(counts < 2):
        raise ValueError("Each exposure period requires two days")
    centered = (mean_fiber - 12.5) / 5.
    gap = moments[:, 0] - np.maximum(centered, 0)
    var = moments[:, 1] - centered ** 2
    if np.any(gap < -1e-7) or np.any(var < -1e-6):
        raise ValueError("Inconsistent fiber moment inputs")
    spread = np.sqrt(np.maximum(var, 0) * counts / (counts - 1))
    return np.column_stack((np.maximum(gap, 0), spread))


def standardize(values, train):
    scale = values[train].std(axis=0)
    return (values-values[train].mean(axis=0)) / np.where(scale > 1e-8, scale, 1.)


def surface_prediction(y, x, z, train, held, extra=None):
    """Residualize on the common adjustment. Keep optional spread corrections orthogonal to the fitted surface."""
    inv = np.linalg.pinv(x[train], rcond=1e-10)
    beta = inv @ y[train]
    residual = y[train] - x[train] @ beta
    z = z - x @ (inv @ z[train])
    penalty = np.asarray([.01, .01, .03, .03]) * train.sum()
    coef = np.linalg.solve(z[train].T @ z[train] + np.diag(penalty), z[train].T @ residual)
    off = x[held] @ beta
    pred = off + z[held] @ coef
    if extra is None:
        return off, pred, pred
    # A two-column, fixed stronger ridge for deviations beyond the four original products.
    adjust = np.column_stack((x, z))
    extra = extra - adjust @ (np.linalg.pinv(adjust[train], rcond=1e-10) @ extra[train])
    target = residual - z[train] @ coef
    coef2 = np.linalg.solve(extra[train].T @ extra[train] + .30 * train.sum() * np.eye(2),
                            extra[train].T @ target)
    return off, pred, pred + extra[held] @ coef2


def predictions(surface, role, eco, pcs, scores, context, train, held):
    y, x, original, _ = surface.design(role, eco, pcs, scores, train)
    marker = surface.marker_basis(scores[:, 0], train)
    scaled = standardize(context, train)
    common = np.column_stack((x, scaled))
    daily_dose = role[2].copy()
    daily_dose[:, 1] += context[:, 0]
    integrated = surface.products(surface.dose_basis(daily_dose, train), marker)
    spread = scaled[:, 1, None] * marker
    old_off, old_fixed, _ = surface_prediction(y, x, original, train, held)
    off, fixed, extra = surface_prediction(y, common, original, train, held, spread)
    _, daily, both = surface_prediction(y, common, integrated, train, held, spread)
    return np.column_stack((old_off, old_fixed, off, fixed, daily, extra, both))


def loss_totals(y, preds, train, held, families, n):
    err = y[held, None] - preds
    cols = np.column_stack((np.ones(held.sum()), (y[held] - y[train].mean())**2, err**2, abs(err)))
    return np.column_stack([np.bincount(families[held], weights=c, minlength=n) for c in cols.T])


def metrics(totals, weights):
    data = weights @ totals
    count, var = data[:, :2].T
    if (count <= 0).any() or (var <= 0).any() or not np.isfinite(data).all():
        raise ValueError("Invalid household totals")
    n = len(ROUTES)
    r2 = 1-data[:, 2:2+n]/var[:, None]
    mae = data[:, 2+n:]/count[:, None]
    return r2, mae


def summarize(totals, seed, draws):
    """Share household draws across repeats and directions, averaging paired repeat differences."""
    households = next(iter(totals[0].values())).shape[0]
    rng = np.random.default_rng(seed)
    weights = np.ones((draws+1, households))
    for row in weights[1:]:
        row[:] = np.bincount(rng.integers(households, size=households), minlength=households)
    per_repeat, paired = {}, {}
    for direction in totals[0]:
        repeat_diffs = []
        for r, group in sorted(totals.items()):
            r2, mae = metrics(group[direction], weights)
            vals = {}
            for j, name in enumerate(ROUTES):
                off = 0 if j < 2 else 2
                base = r2[0, off]
                vals[name] = dict(r2=float(r2[0,j]), mae=float(mae[0,j]),
                                  relative_pct=float(100*(r2[0,j]-base)/base) if base>0 else None,
                                  vs_original=float(r2[0,j]-r2[0,1]), vs_matched=float(r2[0,j]-r2[0,3]))
            per_repeat.setdefault(str(r), {})[direction] = vals
            repeat_diffs.append((r2[:,4:]-r2[:,3,None], mae[:,3,None]-mae[:,4:],
                                 r2[:,4:]-r2[:,1,None], r2[:,1]-r2[:,0]))
        delta, err_gain, orig, fixed_gain = (np.mean(x, axis=0) for x in zip(*repeat_diffs))
        paired[direction] = {}
        for k,name in enumerate(NEW):
            q = .05/(2*len(NEW)*len(totals[0]))
            paired[direction][name] = dict(delta_r2=float(delta[0,k]), ci95=np.quantile(delta[1:,k], [.025,.975]).tolist(),
                                           simultaneous=np.quantile(delta[1:,k], [q,1-q]).tolist(),
                                           vs_original=float(orig[0,k]), mae_gain=float(err_gain[0,k]),
                                           mae_gain_ci95=np.quantile(err_gain[1:,k], [.025,.975]).tolist())
        paired[direction]["original_gain"] = dict(delta_r2=float(fixed_gain[0]),
                                                  ci95=np.quantile(fixed_gain[1:], [.025,.975]).tolist())
    return dict(repeats=per_repeat, paired=paired)
