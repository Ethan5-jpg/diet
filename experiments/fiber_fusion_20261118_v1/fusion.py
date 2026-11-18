"""Fixed discovery recipes combining ecological and dietary signals with frozen date-split anchors."""
import numpy as np


RECIPES = {
    "balances": ("balances",),
    "shannon": ("shannon",),
    "shannon_balances": ("shannon", "balances"),
    "lower_dose": ("lower",),
    "lower_balances": ("lower", "balances"),
    "gut_pc4": ("pcs",),
    "ecology_pc4": ("shannon", "balances", "pcs"),
}
METHODS = (*RECIPES, "fixed_blend")
NAMES = ("off", "anchor", *METHODS, "matched_off", "matched_anchor", *("matched_"+m for m in METHODS))
CONTRASTS = 2 * 2 * len(METHODS)


def panel_views(panel, matches, unit):
    labels = list(matches)
    cols = [labels.index(s) for s in ("Ruminococcus bromii", "Bifidobacterium adolescentis", "Eubacterium rectale")]
    r, b, e = np.log1p(panel[:, cols]/unit).T
    return np.column_stack(((r-b)/np.sqrt(2), (r+b-2*e)/np.sqrt(6)))


def scale(arr, train):
    sd = arr[train].std(axis=0)
    return (arr-arr[train].mean(axis=0))/np.where(sd > 1e-10, sd, 1.)


def project(features, adjustment, train):
    coef = np.linalg.lstsq(adjustment[train], features[train], rcond=1e-10)[0]
    return features-adjustment@coef


def ridge(x, residual, train, held, strength):
    fit = x[train]
    weights = np.broadcast_to(strength, fit.shape[1])
    coef = np.linalg.solve(fit.T@fit + train.sum()*np.diag(weights), fit.T@residual)
    return x[held]@coef


def additions(surface, role, eco, pcs, scores, balances, omit, context, train, held, reverse):
    y, x, total_int, _ = surface.design(role, eco, pcs, scores, train)
    if pcs.shape[1] != 10:
        raise ValueError("Expected five fold-trained oral and five gut PCs")
    dose = surface.dose_basis(role[2], train)
    marker = surface.marker_basis(scores[:, 0], train)
    contrasts = scale(balances, train)
    shannon = scale(scores[:, 1, None], train)
    # A training-only 35th-percentile bend, as in v10's crossed dose/composition blocks.
    low = np.maximum(role[2][:, 0]-np.quantile(role[2][train, 0], .35), 0.)[:, None]
    low = project(low, np.column_stack((np.ones(len(low)), dose[:, 0])), train)
    extra_mains = [contrasts, low]
    existing = [total_int]
    if reverse:
        om = scale(omit[:, None], train)
        extra_mains.extend((om, context))
        existing.extend((surface.products(dose, om), scale(context, train)[:, 1, None]*marker))
    extra_mains = np.column_stack(extra_mains)
    shared = np.column_stack((x, extra_mains))
    fit_y = y[train]
    def residual(adjustment):
        a = adjustment[train]
        return fit_y-a@np.linalg.lstsq(a, fit_y, rcond=1e-10)[0]
    # A paired sensitivity includes the same new additive prediction in both arms.
    main = ridge(project(extra_mains, x, train), residual(x), train, held, .03)
    background = np.column_stack((shared, *existing))
    y_res = residual(background)
    columns, penalties = {}, {}
    for name, feat, weight in (("balances", contrasts, .18), ("shannon", shannon, .18), ("pcs", pcs[:, 5:9], .18)):
        columns[name] = project(surface.products(dose, feat), background, train)
        penalties[name] = np.repeat([weight, 3*weight], feat.shape[1])
    columns["lower"] = project(low*marker, background, train)
    penalties["lower"] = np.full(marker.shape[1], .03)
    preds = {}
    for name, blocks in RECIPES.items():
        design = np.column_stack([columns[b] for b in blocks])
        penalty = np.concatenate([penalties[b] for b in blocks])
        preds[name] = ridge(design, y_res, train, held, penalty)
    preds["fixed_blend"] = np.mean([preds[m] for m in ("shannon_balances", "lower_dose", "gut_pc4")], axis=0)
    return main, np.column_stack([preds[m] for m in METHODS])


def assemble(off, anchor, additive, delta):
    on = anchor[:, None]+delta
    base = np.column_stack((off, anchor, on))
    return np.column_stack((base, base+additive[:, None]))


def scores(data):
    n = len(NAMES)
    if not np.isfinite(data).all() or np.any(data[:, :2] <= 0):
        raise ValueError("Invalid household loss aggregation")
    return 1-data[:, 2:2+n]/data[:, 1, None], data[:, 2+n:]/data[:, 0, None]


def summarize(repeats, weights):
    r2, mae = (np.stack(group) for group in zip(*(scores(weights@a) for a in repeats)))
    out = {}
    for j, name in enumerate(NAMES):
        offset = NAMES.index("matched_off") if name.startswith("matched_") else 0
        ref = offset+1
        point, off = r2[:, 0, j], r2[:, 0, offset]
        if (point > 0).all() and (off > 0).all():
            on_fraction = 100*(point-off)/point
            off_gain = 100*(point-off)/off
        else:
            on_fraction = off_gain = None
        d = r2[:, :, j]-r2[:, :, ref]
        paired = d.mean(axis=0)
        q = .025/CONTRASTS
        out[name] = dict(r2=float(point.mean()), off_r2=float(off.mean()), delta_off=float((point-off).mean()),
                         fraction_pct=None if on_fraction is None else float(on_fraction.mean()),
                         worst_fraction=None if on_fraction is None else float(on_fraction.min()),
                         gain_pct=None if off_gain is None else float(off_gain.mean()),
                         delta=float(paired[0]), wins=int(np.count_nonzero(d[:, 0] > 0)),
                         interval=np.quantile(paired[1:], [q, 1-q]).tolist(),
                         delta_historical=float((point-r2[:, 0, 1]).mean()),
                         mae_gain=float((mae[:, 0, ref]-mae[:, 0, j]).mean()),
                         r2_repeats=point.tolist())
    return out
