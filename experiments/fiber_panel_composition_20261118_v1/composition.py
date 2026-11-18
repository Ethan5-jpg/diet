"""Conditional two-species interaction on top of the frozen split-date predictors."""
import numpy as np


NAMES = ("off", "retained", "shared_half", "own_half", "own_full",
         "matched_off", "matched_retained", "matched_shared_half")
TESTED = ("shared_half", "own_half", "own_full")


def scores_without_bromii(panel, matches, mass_unit):
    names = list(matches)
    keep = [names.index(name) for name in ("Bifidobacterium adolescentis", "Eubacterium rectale")]
    if len(names) != 3 or "Ruminococcus bromii" not in names:
        raise ValueError("Unexpected species mapping")
    return np.log1p(panel[:, keep].sum(axis=1) / mass_unit)


def residualize(matrix, adjustment, n):
    beta = np.linalg.lstsq(adjustment[:n], matrix[:n], rcond=1e-10)[0]
    return matrix - adjustment @ beta


def correction(surface, fit, target, ecology, pcs, scores, marker, train, held):
    """All fitting uses training households. Only target exposure features enter held rows."""
    def stack(a, b):
        return np.concatenate((a[train], b[held]))

    n = int(train.sum())
    y = fit[0][train]
    role = (np.r_[y, np.zeros(held.sum())], stack(fit[1], target[1]), stack(fit[2], target[2]))
    features = [stack(v, v) for v in (ecology, pcs, scores)]
    mask = np.arange(len(role[0])) < n
    _, x, panel_terms, _ = surface.design(role, *features, mask)
    omit = stack(marker, marker)
    if omit[:n].std() <= 1e-10:
        return np.zeros(held.sum()), np.zeros(held.sum())
    omit_basis = surface.marker_basis(omit, mask)[:, :1]
    # Keep the historical background fixed for the primary endpoint. The sensitivity
    # pair includes the linear extra marker main effect on both sides of its comparison.
    main = residualize(omit_basis, x, n)
    main_resid = y - x[:n] @ np.linalg.lstsq(x[:n], y, rcond=1e-10)[0]
    main_coef = np.linalg.solve(main[:n].T @ main[:n] + .01*n*np.eye(1), main[:n].T @ main_resid)
    design = np.column_stack((x, panel_terms, omit_basis))
    products = surface.products(surface.dose_basis(role[2], mask), omit_basis)
    z = residualize(products, design, n)
    residual = y - design[:n] @ np.linalg.lstsq(design[:n], y, rcond=1e-10)[0]
    # v10's linear and bent dose products with the omitted-species score have weights 4 and 8.
    penalty = n*np.diag([.04, .08])
    coef = np.linalg.solve(z[:n].T @ z[:n] + penalty, z[:n].T @ residual)
    return main[n:] @ main_coef, z[n:] @ coef


def assemble(off, retained, main, own, other):
    shared = .5*(.75*own + .25*other)
    return np.column_stack((off, retained, retained+shared, retained+.5*own, retained+own,
                            off+main, retained+main, retained+main+shared))


def summarize(repeats, weights):
    reps = []
    for totals in repeats:
        a = weights @ totals
        n = len(NAMES)
        if not np.isfinite(a).all() or np.any(a[:, :2] <= 0):
            raise ValueError("Invalid household bootstrap aggregation")
        reps.append((1-a[:, 2:2+n]/a[:, 1, None], a[:, 2+n:]/a[:, 0, None]))
    r2, mae = (np.stack(v) for v in zip(*reps))
    rows = {}
    for j, name in enumerate(NAMES):
        matched = name.startswith("matched_")
        off = NAMES.index("matched_off") if matched else 0
        ref = NAMES.index("matched_retained") if matched else 1
        gains = r2[:, :, j]-r2[:, :, ref]
        delta = gains.mean(axis=0)
        err_gain = (mae[:, :, ref]-mae[:, :, j]).mean(axis=0)
        pct = None
        if np.all(r2[:, 0, off] > 0):
            pct = float(np.mean(100*(r2[:, 0, j]-r2[:, 0, off])/r2[:, 0, off]))
        q = .025/(2*len(TESTED))
        rows[name] = dict(mean_r2=float(r2[:, 0, j].mean()), relative_pct=pct,
                          delta=float(delta[0]), wins=int(np.count_nonzero(gains[:, 0] > 0)),
                          ci95=np.quantile(delta[1:], [.025, .975]).tolist(),
                          simultaneous=np.quantile(delta[1:], [q, 1-q]).tolist(),
                          mae_gain=float(err_gain[0]), mean_mae=float(mae[:, 0, j].mean()),
                          per_repeat_gain=gains[:, 0].tolist())
    return rows
