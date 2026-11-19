"""Matched alternative dietary features and nested interaction shrinkage."""

import numpy as np

import protocol as p
from breadth import basis, fit_profile
from design import domain
from diet_audit import Fit, fit_fold, ridge_diet
from models import create_design, dose_basis, predict, products, standardize
from source import check, clean_outcome
from v3_inputs import InnerContexts, read_sites
from v4_models import no_diet_main


ANCHOR = ("fiber_density", "D_Glu")
ALTERNATIVES = {
    "fiber_density": ("fiber_cumulative", "fiber_sources", "fiber_total_and_sources", "nutrient_density", "sodium_density"),
    "UPF": ("upf_known_and_unknown", "upf_known_subtypes", "nutrient_density", "sodium_density"),
}
FEATURE_METHODS = ("matched_reference", "new_pc5_half", "new_pc5_full", "new_ecology_half")
NESTED_METHODS = ("pc5_half", "fixed_diet_gated", "diet_gated", "diet_partial_pool",
                  "strict_upf_diet_gated", "strict_upf_partial_pool")


def inputs(study, exposure, endpoint, outcome, fold, exposure_values=None):
    training = study.backgrounds[fold].role
    y = clean_outcome(outcome.y, training)
    valid = study.eligible(exposure, endpoint, fold, y)
    if exposure_values is not None:
        valid &= np.isfinite(exposure_values).all(axis=1)
    tr, held = np.flatnonzero(valid & training), np.flatnonzero(valid & ~training)
    return y, tr, held


def with_feature(study, fold, exposure, endpoint, outcome, values, shared):
    y, tr, held = inputs(study, exposure, endpoint, outcome, fold, values)
    rows, n = np.r_[tr, held], len(tr)
    check(n >= 30 and len(held) >= 10, "V6_ALTERNATIVE_FOLD_SUPPORT")
    design = create_design(study, exposure, fold, rows, n)
    raw, views = shared
    dose = dose_basis(raw[rows], n)[0]
    base = no_diet_main(study, rows, n, design.main, dose)
    base_inv = np.linalg.pinv(base[:n], rcond=p.RANK_TOL)
    no_diet = base[n:] @ (base_inv @ y[tr])
    legacy_inv = np.linalg.pinv(design.main[:n], rcond=p.RANK_TOL)
    legacy = design.main @ (legacy_inv @ y[tr])
    old_product = products(raw[rows, :1], views["gut_pc5"][rows])
    matched = fit_profile(old_product, design.main, legacy_inv, y[tr] - legacy[:n]).fit(1.)

    dietary = values[rows]
    new, adjusted = ridge_diet(base, base_inv, dietary, y[tr], np.full(dietary.shape[1], .1))
    nuisance = np.column_stack((base, adjusted))
    inverse = np.linalg.pinv(nuisance[:n], rcond=p.RANK_TOL)
    z = standardize(dietary, n)[0]
    ecology = views["ecology"][rows][:, (0, 2)]
    residuals = y[tr] - new[:n]
    gut = fit_profile(products(z, views["gut_pc5"][rows]), nuisance, inverse, residuals).fit(1.)
    eco = fit_profile(products(z, ecology), nuisance, inverse, residuals).fit(1.)
    new_off, old_off = new[n:], legacy[n:]
    reference = predict(design, y[tr])[:, 2] if (exposure, endpoint) == ANCHOR else old_off + .5*matched
    return Fit(held, y[held], no_diet, np.column_stack((old_off, new_off, new_off, new_off)),
               np.column_stack((reference, new_off + .5*gut, new_off + gut, new_off + .5*eco)))


def feature_fits(study, endpoints, values, progress):
    raw = {(exposure, fold): basis(study, exposure, fold) for exposure in p.EXPOSURES for fold in p.FOLDS}
    results, skipped, coverage = {}, [], {}
    for exposure, recipes in ALTERNATIVES.items():
        for feature in recipes:
            x = values[feature]
            for endpoint, outcome in endpoints.items():
                key = exposure, endpoint, feature
                old, new = [], []
                for f in p.FOLDS:
                    _, _, held = inputs(study, exposure, endpoint, outcome, f)
                    _, tr, common = inputs(study, exposure, endpoint, outcome, f, x)
                    old.append(len(held))
                    new.append((len(tr), len(common)))
                if min(a for a, _ in new) < 30 or min(b for _, b in new) < 10:
                    skipped.append(dict(exposure=exposure, endpoint=endpoint, feature=feature,
                                        available=sum(b for _, b in new), baseline=sum(old)))
                    continue
                coverage[key] = sum(old)
                results[key] = [with_feature(study, f, exposure, endpoint, outcome, x, raw[exposure, f]) for f in p.FOLDS]
            progress("FEATURE_DONE", {"exposure": exposure, "feature": feature,
                                      "completed_pairs": sum(k[0] == exposure and k[2] == feature for k in results)})
    return results, skipped, coverage


def choose_diet(parts, sst):
    gains = np.stack([np.sum((p.outcome-p.no_diet)[:, None]**2 - (p.outcome[:, None]-p.off[:, :5])**2,
                             axis=0) for p in parts]) / sst
    valid = (gains.sum(axis=0) >= .0002) & ((gains > 0.).sum(axis=0) >= 3)
    if not valid.any():
        return -1
    return int(np.argmax(np.where(valid, gains.sum(axis=0), -np.inf)))


def profile(parts, recipe, sst):
    if recipe < 0:
        return np.zeros(4), np.zeros(4)
    dot, norm = [], []
    for held in parts:
        diff = held.on[:, recipe] - held.off[:, recipe]
        dot.append(float(diff @ (held.outcome - held.off[:, recipe])) / sst)
        norm.append(float(diff @ diff) / sst)
    return np.asarray(dot), np.asarray(norm)


def slope(dot, norm):
    return float(np.clip(dot.sum() / (1.25*norm.sum()), 0., 1.6)) if norm.sum() > 0. else 0.


def guard(dot, norm, weight):
    change = weight*(2.*dot - weight*norm)
    return weight if change.sum() >= .0002 and (change > 0.).sum() >= 3 else 0.


def plans(folds, exposure):
    choices, profiles = {}, {}
    for endpoint, parts in folds.items():
        if (exposure, endpoint) == ANCHOR:
            continue
        y = np.concatenate([part.outcome for part in parts])
        sst = float(np.sum((y-y.mean())**2))
        check(sst > 0. and np.isfinite(sst), "V6_SMALL_INNER_VARIANCE")
        recipe = choose_diet(parts, sst)
        info = profile(parts, recipe, sst)
        native = profile(parts, 0, sst)
        gate = guard(*info, slope(*info))
        choices[endpoint] = (recipe, gate, guard(*native, slope(*native)))
        profiles[endpoint] = info
    pool = {}
    for label in {domain(e) for e in choices}:
        arrays = [profiles[e] for e, (recipe, _, _) in choices.items() if domain(e) == label and recipe >= 0]
        if arrays:
            dot, norm = np.mean(arrays, axis=0)
            pool[label] = slope(dot, norm)
    return {e: ((0, native), (recipe, gate),
                (recipe, guard(*profiles[e], .5 * (slope(*profiles[e]) + pool.get(domain(e), 0.)))))
            for e, (recipe, gate, native) in choices.items()}


def apply_plans(held, ordinary, strict, anchor):
    if anchor:
        off = np.repeat(held.off[:, -1:], len(NESTED_METHODS), axis=1)
        on = np.repeat(held.on[:, -1:], len(NESTED_METHODS), axis=1)
    else:
        specs = [(0, 1.), *ordinary, *strict[1:]]
        off = np.column_stack([held.off[:, j] if j >= 0 else held.no_diet for j, _ in specs])
        on = off + np.column_stack([w * (held.on[:, j] - held.off[:, j]) if j >= 0 else np.zeros(len(held.rows))
                                     for j, w in specs])
    return Fit(held.rows, held.outcome, held.no_diet, off, on)


def nested_fits(study, outcomes, calibration, progress):
    contexts = InnerContexts(study, read_sites(study))
    outer = {(x, e, "registered_exposure"): [] for x, e in p.MODES["all"]}
    audits, selections = [], []
    for fold in p.FOLDS:
        collected = {x: {e: [] for e in outcomes} for x in p.EXPOSURES}
        strict = {e: [] for e in outcomes}
        for inner in p.FOLDS:
            if inner == fold:
                continue
            context = contexts.split(fold, inner)
            for exposure in p.EXPOSURES:
                shared = basis(context, exposure, inner)
                for endpoint, outcome in outcomes.items():
                    held = fit_fold(context, exposure, endpoint, outcome, inner, shared)
                    collected[exposure][endpoint].append(held)
            context, n_foods = calibration.inner(context, inner, 2026112000 + 10*fold + inner)
            shared = basis(context, "UPF", inner)
            for endpoint, outcome in outcomes.items():
                strict[endpoint].append(fit_fold(context, "UPF", endpoint, outcome, inner, shared))
            audits.append(dict(outer=fold, inner=inner, training_catalogue_foods=n_foods))
        for exposure in p.EXPOSURES:
            default_plan = plans(collected[exposure], exposure)
            strict_plan = plans(strict, exposure) if exposure == "UPF" else default_plan
            shared = basis(study, exposure, fold)
            for endpoint, outcome in outcomes.items():
                held = fit_fold(study, exposure, endpoint, outcome, fold, shared)
                anchor = (exposure, endpoint) == ANCHOR
                primary, sensitivity = default_plan.get(endpoint), strict_plan.get(endpoint)
                selections.append(dict(outer=fold, exposure=exposure, endpoint=endpoint,
                                       ordinary=primary, strictly_refit=sensitivity))
                outer[exposure, endpoint, "registered_exposure"].append(apply_plans(held, primary, sensitivity, anchor))
        progress("NESTED_OUTER_DONE", {"fold": fold, "strict_upf_inner_fits": 4})
    return outer, audits, selections
