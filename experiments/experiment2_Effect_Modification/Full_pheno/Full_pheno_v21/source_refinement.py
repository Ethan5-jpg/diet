"""Tune bounded main-effect scales and measure exact held-outcome coverage."""

from dataclasses import replace

import numpy as np


FOCUS = ("cardiovascular", "sleep")
BASE = "domain_calibrated"
REFINED = "source_refined"
METHODS = (BASE, REFINED)
MIN_GAIN = .01


def secondary_scale(dot, square, alpha):
    q = alpha * alpha * square
    penalty = max(.25 * q, 1e-8)
    return float(np.clip((alpha * dot + penalty) / (q + penalty), max(.75, .25 / alpha), 1.))


def refine(terms, covers, domains, sources):
    from main_attribution import choose_amplitudes

    valid = covers.all(axis=0)
    baseline, _ = choose_amplitudes(terms, valid, domains, sources)
    kept = np.arange(len(terms))
    inner_parents = []
    for held in kept:
        tr = kept[kept != held]
        parent, _ = choose_amplitudes(terms[tr], covers[tr].all(axis=0), domains, sources)
        inner_parents.append(parent)
    adjusted = baseline.copy()
    audit = {}
    for domain in FOCUS:
        domain_mask = domains == domain
        for source in sorted(set(sources[domain_mask])):
            members = domain_mask & (sources == source)
            valid_in_source = members & valid
            n = int(valid_in_source.sum())
            e = int(np.flatnonzero(members)[0])
            current = baseline[e]
            note = dict(common=n, parent=float(current), multiplier=1.)
            if n >= 5:
                estimates = terms[:, valid_in_source, :].mean(axis=1)
                held_gain = []
                for j, (dot, sq) in enumerate(estimates):
                    q = inner_parents[j][e]
                    direction = estimates[kept != j].mean(axis=0)
                    w = secondary_scale(*direction, q)
                    held_gain.append(float(2*q*(w-1.)*dot + q*q*(1.-w*w)*sq))
                note["positive_alignment_folds"] = int((estimates[:, 0] > 0.).sum())
                note["held_gain_pp"] = held_gain
                if (note["positive_alignment_folds"] >= 3 and sum(x > 0. for x in held_gain) >= 3
                        and np.mean(held_gain) >= MIN_GAIN):
                    w = secondary_scale(*estimates.mean(axis=0), current)
                    adjusted[members] = current * w
                    note["multiplier"] = w
            audit[f"{domain}:{source}"] = note
    return np.stack((baseline, adjusted)), audit


def inner_terms(work, context, calibration, outer, diets, y, masks, domains, sources, seed, emit):
    import protocol as p
    from main_attribution import normalized_terms
    from run_v2 import require

    terms, masks_covered = [], []
    for inner in p.FOLDS:
        if inner == outer:
            continue
        v = context.split(outer, inner)
        v, _ = calibration.inner(v, inner, 100 * seed + 10 * outer + inner)
        b = v.backgrounds[inner]
        v.backgrounds[inner] = replace(b, supported=b.supported & v.active)
        require(np.array_equal(v.active & ~b.role, work.fold == inner)
                and not np.any(b.role & (work.fold == outer)), "V21_INNER_OUTER_LEAKAGE")
        q, a = normalized_terms(v, inner, diets, y, masks)
        terms.append(q)
        masks_covered.append(a)
        emit("V21_INNER", dict(seed=seed, outer=outer, inner=inner, outcomes=int(a.sum())))
    return refine(np.stack(terms), np.stack(masks_covered), domains, sources)


def collect(work, fold, diets, values, masks, scales):
    from batch_fit import context_design
    from main_attribution import forecasts
    from run_v2 import require

    eligible, _ = context_design(work, fold, "UPF")
    held_x = eligible & ~work.backgrounds[fold].role
    base_count = int(held_x.sum())
    diary = work.numeric["n_energy_days"]
    data = np.zeros((len(scales), values.shape[1], 7))
    technical = np.zeros((values.shape[1], 3))
    for cols, y, a, ref, _, families in forecasts(work, fold, diets, values, masks):
        rows = held_x & np.isfinite(values[:, cols[0]])
        require(rows.sum() == len(y) and np.all(work.fold[rows] == fold), "V21_COVERAGE_MEMBERSHIP_CHANGED")
        err = y - a
        common = (len(y), families, y.sum(axis=0), (y*y).sum(axis=0), (err*err).sum(axis=0))
        for i, w in enumerate(scales):
            delta = ref * w[cols]
            dot, sq = (err*delta).sum(axis=0), (delta*delta).sum(axis=0)
            data[i, cols] = np.column_stack(np.broadcast_arrays(*common, dot, sq))
        n_days = diary[rows]
        technical[cols] = [base_count, n_days.sum(), (n_days < 10.).sum()]
    require(np.isfinite(data).all() and np.isfinite(technical).all()
            and np.all(data[:, :, :5] == data[:1, :, :5]), "V21_MATCHED_MAIN_DIFF")
    return data, technical
