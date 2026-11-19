"""Low-complexity nested calibration, restricted fusion, and peer transfer."""

from dataclasses import dataclass

import numpy as np

import v4_design as d
from design import domain
from source import check


@dataclass(frozen=True)
class Loss:
    align: np.ndarray
    norm: np.ndarray
    cross_pc5: np.ndarray
    available: np.ndarray
    fold_sst_fraction: np.ndarray


def losses(parts):
    rows = np.concatenate([f.old.rows for f in parts])
    y = np.concatenate([f.old.y for f in parts])
    sst = np.sum((y-y.mean())**2)
    check(len(np.unique(rows)) == len(rows) and np.isfinite(sst) and sst > 0., "V4_INVALID_INNER_LOSS")
    available = parts[0].selectable
    check(all(np.array_equal(f.selectable, available) for f in parts), "V4_INNER_LIBRARY_DIFFERS")
    core = d.INDEX[d.PC5]
    a, b, c, fold_sst = [], [], [], []
    for f in parts:
        z, r = f.correction, f.old.y-f.old.off
        a.append(z.T @ r / sst)
        b.append(np.sum(z**2, axis=0) / sst)
        c.append(z.T @ z[:, core] / sst)
        y_fold = f.old.y
        fold_sst.append(np.sum((y_fold-y_fold.mean())**2) / sst)
    align, norm, cross = map(np.asarray, (a, b, c))
    fraction = np.asarray(fold_sst)
    check(all(np.isfinite(t).all() for t in (align, norm, cross, fraction)), "V4_NONFINITE_INNER_LOSS")
    return Loss(align, norm, cross, available, fraction)


def fit_scale(align, norm, cap=1.):
    """Calibrate one nonnegative coefficient with a small ridge penalty."""
    denom = (1. + d.SHRINK_PENALTY)*float(np.sum(norm))
    value = float(np.sum(align))/denom if denom > 0. else 0.
    w = float(np.clip(value, 0., cap))
    fold_gain = w*(2.*align-w*norm)
    return w, fold_gain


def checked(gain, wins=d.INNER_WINS):
    return gain.sum() >= d.MIN_INNER_GAIN and np.count_nonzero(gain > 0.) >= wins


def single(loss, names, *, maximin=False):
    if maximin and not np.all(loss.fold_sst_fraction > 0.):
        return {}
    settings = []
    for name in names:
        j = d.INDEX[name]
        if not loss.available[j]:
            continue
        w, gain = fit_scale(loss.align[:, j], loss.norm[:, j])
        if checked(gain, wins=4 if maximin else d.INNER_WINS):
            total = float(gain.sum())
            if maximin:
                priority = (float((gain/loss.fold_sst_fraction).min()), total)
            else:
                priority = (total, float(gain.min()))
            settings.append((priority, name, w))
    if not settings:
        return {}
    _, name, w = max(settings, key=lambda r: r[0])
    return {name: w}


def capped_mix(loss):
    core = d.INDEX[d.PC5]
    a, b = loss.align[:, core], loss.norm[:, core]
    scale, gain = fit_scale(a, b)
    best_core = {d.PC5: scale} if checked(gain) else {}
    core_gain = float(gain.sum()) if best_core else 0.
    options = []
    for name in d.MIX_CORE:
        j = d.INDEX[name]
        if j == core or not loss.available[j]:
            continue
        for frac in d.MIX_FRACTIONS[1:]:
            pc = 1.-frac
            a_mix = pc*a + frac*loss.align[:, j]
            v_mix = pc**2*b + frac**2*loss.norm[:, j] + 2.*pc*frac*loss.cross_pc5[:, j]
            w, inc = fit_scale(a_mix, np.maximum(v_mix, 0.))
            if checked(inc) and inc.sum() >= core_gain + .0002:
                options.append((float(inc.sum()), name, w, frac))
    if not options:
        return best_core
    _, name, w, frac = max(options, key=lambda r: r[0])
    return {d.PC5: w*(1.-frac), name: w*frac}


def peer_choices(endpoint, loss, related, own):
    donors = [peer for name, peer in related.items() if name != endpoint and domain(name) == domain(endpoint)]
    if not donors:
        return own, own
    peer = Loss(np.mean([r.align for r in donors], axis=0),
                np.mean([r.norm for r in donors], axis=0),
                np.mean([r.cross_pc5 for r in donors], axis=0),
                np.logical_and.reduce([loss.available, *(r.available for r in donors)]),
                np.mean([r.fold_sst_fraction for r in donors], axis=0))
    shared = single(peer, d.CORE)
    if not shared:
        return {}, {}
    name, cap = next(iter(shared.items()))
    j = d.INDEX[name]
    w, gain = fit_scale(loss.align[:, j], loss.norm[:, j], cap=cap)
    return shared, {name: w} if checked(gain) else {}


def select(endpoint, loss, related, old_weights):
    core = single(loss, d.CORE)
    donor, checked_donor = peer_choices(endpoint, loss, related, core)
    return {
        d.REFERENCE: {d.PC5: 1.}, "v3_guarded": old_weights,
        "uniform_pc5": {d.PC5: 1.}, "uniform_oral": {d.ORAL: 1.},
        "pc5_half": {d.PC5: .5}, "pc5_quarter": {d.PC5: .25},
        "fixed_two_site": {d.PC5: .50, d.ORAL: .15},
        "core_calibrated": core, "new_block_guarded": single(loss, d.NEW),
        "new_block_maximin": single(loss, d.NEW, maximin=True),
        d.PRIMARY: capped_mix(loss), "peer_direction": donor, "peer_checked": checked_donor,
    }
