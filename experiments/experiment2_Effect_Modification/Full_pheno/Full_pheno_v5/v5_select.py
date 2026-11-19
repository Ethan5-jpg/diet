"""Select bounded supplements relative to PC5-half inside outer training."""

from dataclasses import dataclass

import numpy as np

import v5_design as d
from design import domain
from source import check


@dataclass(frozen=True)
class Loss:
    dot: np.ndarray
    norm: np.ndarray
    base: np.ndarray
    fold_sst: np.ndarray
    available: np.ndarray


def losses(parts):
    y = np.concatenate([f.previous.old.y for f in parts])
    rows = np.concatenate([f.previous.old.rows for f in parts])
    sst = np.sum((y-y.mean())**2)
    support = parts[0].available
    check(len(np.unique(rows)) == len(rows) and sst > 0. and np.isfinite(sst)
          and all(np.array_equal(f.available, support) for f in parts), "V5_INNER_LOSS_DIFF")
    dot, norm, base, fold_sst = [], [], [], []
    for f in parts:
        earlier = f.previous.old
        pc = .5*f.correction[:, d.INDEX[d.PC5]]
        err = earlier.y-earlier.off
        z = f.correction-pc[:, None]
        dot.append(z.T @ (err-pc)/sst)
        norm.append(np.sum(z**2, axis=0)/sst)
        base.append(float(pc @ (2.*err-pc)/sst))
        fold_sst.append(np.sum((earlier.y-earlier.y.mean())**2)/sst)
    values = tuple(np.asarray(x) for x in (dot, norm, base, fold_sst))
    check(all(np.isfinite(x).all() for x in values), "V5_INNER_LOSS_NONFINITE")
    return Loss(*values, support)


def pooled(parts):
    combined = [np.mean([getattr(p, key) for p in parts], axis=0)
                for key in ("dot", "norm", "base", "fold_sst")]
    return Loss(*combined, np.logical_and.reduce([f.available for f in parts]))


def calibrate(loss, j, cap=d.CAP):
    dot, norm = loss.dot[:, j], loss.norm[:, j]
    charge = 1.10*float(norm.sum())
    scale = float(np.clip(float(dot.sum())/charge, 0., cap)) if charge > 0. else 0.
    return scale, scale*(2.*dot-scale*norm)


def allowed(loss, gain, strict=False):
    return (gain.sum() >= d.MIN_RELATIVE and (loss.base+gain).sum() > 0.
            and np.count_nonzero(gain > 0.) >= (4 if strict else 3)
            and np.count_nonzero(loss.base+gain > 0.) >= np.count_nonzero(loss.base > 0.))


def choose(loss, names, *, strict=False):
    if strict and not np.all(loss.fold_sst > 0.):
        return None
    picked = None
    best = (-np.inf, -np.inf)
    for recipe in names:
        j = d.INDEX[recipe]
        if not loss.available[j]:
            continue
        w, gain = calibrate(loss, j)
        if allowed(loss, gain, strict):
            total = float(gain.sum())
            value = (float((gain/loss.fold_sst).min()), total) if strict else (total, float(gain.min()))
            if value > best:
                picked, best = (recipe, w), value
    return picked


def anchored(pick):
    if pick is None:
        return {d.PC5: .5}
    recipe, scale = pick
    weight = .5*(1.-scale)
    if recipe == d.PC5:
        return {d.PC5: weight+scale}
    return {d.PC5: weight, recipe: scale}


def plans(losses_by_outcome, maximin):
    domains = {}
    for name, loss in losses_by_outcome.items():
        domains.setdefault(domain(name), []).append(loss)
    by_domain = {key: pooled(rows) for key, rows in domains.items()}
    diet_pick = choose(pooled(list(by_domain.values())), d.CORE)
    groups = {key: choose(loss, d.CORE) for key, loss in by_domain.items()}
    output = {}
    for key, loss in losses_by_outcome.items():
        shared = groups[domain(key)]
        checked = None
        if shared is not None:
            name, cap = shared
            w, gain = calibrate(loss, d.INDEX[name], cap=cap)
            if allowed(loss, gain):
                checked = (name, w)
        output[key] = {
            d.REFERENCE: {d.PC5: .5}, "v4_fixed_pc5": {d.PC5: 1.}, "v4_maximin": maximin[key],
            "pc5_035": {d.PC5: .35}, "pc5_065": {d.PC5: .65}, "pc5_080": {d.PC5: .80},
            "fixed_oral_mix": anchored((d.ORAL_CURVE, .25)), "fixed_tensor_mix": anchored((d.FIXED_TENSOR, .25)),
            d.PRIMARY: anchored(choose(loss, d.CORE)), "add_tensor": anchored(choose(loss, d.NEW)),
            "add_all_four": anchored(choose(loss, (*d.CORE, *d.NEW), strict=True)),
            "diet_shared": anchored(diet_pick), "domain_shared": anchored(shared), "domain_veto": anchored(checked),
        }
    return output
