"""Prespecified interaction-only changes to the split-date three-species predictor."""
import numpy as np


SHARE = .25
ROUTES = ("off", "fixed", "spread25", "transfer25", "huber", "transfer_spread", "combined")


def huber_ridge(z, residual, penalty, start):
    """Fixed-scale Huber ridge; the main predictor and factor projections are frozen."""
    err = residual - z @ start
    scale = 1.4826 * np.median(np.abs(err - np.median(err)))
    if scale <= 1e-10:
        return start.copy()
    cutoff = 1.345 * scale
    coef = start.copy()
    for _ in range(150):
        err = residual - z @ coef
        weights = np.minimum(1., cutoff / np.maximum(np.abs(err), 1e-12))
        new = np.linalg.solve(z.T @ (weights[:, None]*z) + penalty, z.T @ (weights*residual))
        if np.max(abs(new-coef)) <= 1e-8 * (1+np.max(abs(coef))):
            return new
        coef = new
    raise ValueError("Huber ridge failed to converge")


def correction(surface, fit_role, target_role, eco, pcs, scores, train, held):
    """Take adjustment and scale from fit-role training rows; evaluate only target-role held features."""
    role = (np.r_[fit_role[0][train], np.zeros(held.sum())],
            np.vstack((fit_role[1][train], target_role[1][held])),
            np.vstack((fit_role[2][train], target_role[2][held])))
    features = [np.vstack((a[train],a[held])) for a in (eco,pcs,scores)]
    n = int(train.sum())
    mask = np.arange(len(role[0])) < n
    y, x, z, _ = surface.design(role,*features,mask)
    inv = np.linalg.pinv(x[:n],rcond=1e-10)
    residual = y[:n] - x[:n] @ (inv @ y[:n])
    z = z - x @ (inv @ z[:n])
    penalty = np.diag(np.array([.01,.01,.03,.03])*n)
    coef = np.linalg.solve(z[:n].T @ z[:n] + penalty, z[:n].T @ residual)
    robust = huber_ridge(z[:n],residual,penalty,coef)
    return z[n:] @ coef, z[n:] @ robust


def predictions(surface, pattern, target, auxiliary, eco, pcs, scores, context, train, held):
    reference = pattern.predictions(surface,target,eco,pcs,scores,context,train,held)
    off, fixed = reference[:,0], reference[:,1]
    # Only the extra spread correction enters; both of its source fits have identical nuisance adjustment.
    spread = SHARE*(reference[:,5]-reference[:,3])
    own, robust = correction(surface,target,target,eco,pcs,scores,train,held)
    other, robust_other = correction(surface,auxiliary,target,eco,pcs,scores,train,held)
    np.testing.assert_allclose(fixed-off,own,rtol=1e-7,atol=2e-8)
    transfer = (1-SHARE)*own + SHARE*other
    combined = off + (1-SHARE)*robust + SHARE*robust_other + spread
    preds = np.column_stack((off,fixed,fixed+spread,off+transfer,off+robust,off+transfer+spread,combined))
    return preds, reference


def summarize(pattern, totals, seed, draws):
    families = next(iter(totals[0].values())).shape[0]
    rng = np.random.default_rng(seed)
    w = np.ones((draws+1,families))
    for row in w[1:]:
        row[:] = np.bincount(rng.integers(families,size=families), minlength=families)
    repeats, paired = {}, {}
    for direction in totals[0]:
        comparisons = []
        for r, group in sorted(totals.items()):
            # Archived and new schemas both have seven routes: count, denominator, SSEs, absolute errors.
            if group[direction].shape[1] != 2+2*len(ROUTES) or len(ROUTES)!=len(pattern.ROUTES):
                raise ValueError("Unexpected loss layout")
            r2, mae = pattern.metrics(group[direction],w)
            result = {}
            for j,name in enumerate(ROUTES):
                base = r2[0,0]
                result[name] = dict(r2=float(r2[0,j]), relative_pct=float(100*(r2[0,j]-base)/base) if base>0 else None,
                                    vs_fixed=float(r2[0,j]-r2[0,1]), mae=float(mae[0,j]))
            repeats.setdefault(str(r),{})[direction] = result
            comparisons.append((r2[:,2:]-r2[:,1,None],mae[:,1,None]-mae[:,2:]))
        d, m = (np.mean(a,axis=0) for a in zip(*comparisons))
        adj = .05/(2*(len(ROUTES)-2)*len(totals[0]))
        paired[direction] = {}
        for j,name in enumerate(ROUTES[2:]):
            paired[direction][name] = dict(delta_r2=float(d[0,j]), ci95=np.quantile(d[1:,j],[.025,.975]).tolist(),
                                           simultaneous=np.quantile(d[1:,j],[adj,1-adj]).tolist(),
                                           mae_gain=float(m[0,j]), mae_ci95=np.quantile(m[1:,j],[.025,.975]).tolist())
    return dict(repeats=repeats,paired=paired)
