"""Fit diet and microbiome terms on training households and score held households."""

from dataclasses import dataclass

import numpy as np

import protocol as p
from breadth import Profile, basis, fit_profile
from design import domain
from diet_audit import Fit, ridge_diet
from exposure_models import ANCHOR, inputs
from interaction_rules import RULES, choose_upgrade, domain_prior, prediction
from models import create_design, dose_basis, predict, products
from source import Background, Outcome, check
from v3_inputs import InnerContexts, InnerStudy, fit_site, read_sites
from v4_models import no_diet_main, robust_weights


SETTINGS = {"linear": (.1,), "curve": (.1, .5), "host": (.1, .5, 1., 1., 1., 1.)}


@dataclass(frozen=True)
class EndpointFit:
    rows: np.ndarray
    y: np.ndarray
    no_diet: np.ndarray
    off: dict
    profiles: dict
    anchor: np.ndarray | None


def weighted_profile(columns, main, residual):
    """Downweight training residuals before fitting the interaction correction."""
    n = len(residual)
    weight = robust_weights(residual)
    root = np.sqrt(weight)
    weighted_main = root[:, None] * main[:n]
    inverse = np.linalg.pinv(weighted_main, rcond=p.RANK_TOL)
    z = columns - main @ (inverse @ (root[:, None] * columns[:n]))
    a = z[:n] * root[:, None]
    scale = float(np.sqrt(np.average(residual**2, weights=weight)))
    scale = max(scale, p.RANK_TOL)
    denom = weight.sum()
    return Profile(z[n:], (a.T @ a)/denom, (a.T @ (root*residual/scale))/denom, scale)


def fit_endpoint(study, fold, exposure, endpoint, outcome, shared):
    y, tr, held = inputs(study, exposure, endpoint, outcome, fold)
    n, rows = len(tr), np.r_[tr, held]
    check(n >= 30 and len(held) >= 10, "V7_ENDPOINT_FOLD_SUPPORT")
    check(not np.intersect1d(study.family[tr], study.family[held]).size, "V7_SPLIT_FAMILY_OVERLAP")
    raw, views = shared
    design = create_design(study, exposure, fold, rows, n)
    dose = dose_basis(raw[rows], n)[0]
    base = no_diet_main(study, rows, n, design.main, dose)
    base_inv = np.linalg.pinv(base[:n], rcond=p.RANK_TOL)
    no_diet = base[n:] @ (base_inv @ y[tr])
    legacy = design.main @ (np.linalg.pinv(design.main[:n], rcond=p.RANK_TOL) @ y[tr])
    if (exposure, endpoint) == ANCHOR:
        existing = predict(design, y[tr])
        return EndpointFit(held, y[held], no_diet, {"legacy": legacy[n:]}, {}, existing[:, 2])

    diet_data = {"linear": dose[:, :1], "curve": dose,
                 "host": np.column_stack((dose, products(dose, design.main[:, 1:3])))}
    designs = {"legacy": (legacy, design.main)}
    for name, penalty in SETTINGS.items():
        off, residualized = ridge_diet(base, base_inv, diet_data[name], y[tr], penalty)
        designs[name] = off, np.column_stack((base, residualized))
    gut = products(raw[rows, :1], views["gut_pc5"][rows])
    oral = products(raw[rows], views["ecology"][rows, :1])
    blocks = {"pc5": gut, "pc2": gut[:, :2], "oral": oral}
    profiles, offsets = {}, {}
    for name, (off, main) in designs.items():
        inv = np.linalg.pinv(main[:n], rcond=p.RANK_TOL)
        residual = y[tr]-off[:n]
        for view, cols in blocks.items():
            profiles[name, view, False] = fit_profile(cols, main, inv, residual)
        if name == "host":
            for view in ("pc5", "oral"):
                profiles[name, view, True] = weighted_profile(blocks[view], main, residual)
        offsets[name] = off[n:]
    return EndpointFit(held, y[held], no_diet, offsets, profiles, None)


def fit_split(study, fold, exposure, outcomes):
    shared = basis(study, exposure, fold)
    work = {e: fit_endpoint(study, fold, exposure, e, y, shared) for e, y in outcomes.items()}
    results = {}
    for e, target in work.items():
        if target.anchor is not None:
            off = np.repeat(target.off["legacy"][:, None], len(RULES), axis=1)
            on = np.repeat(target.anchor[:, None], len(RULES), axis=1)
        else:
            donors = [v.profiles for key, v in work.items() if key != e and domain(key) == domain(e)]
            priors = {}
            off, on = [], []
            for rule in RULES:
                offset = target.off[rule.diet]
                off.append(offset)
                correction = np.zeros(len(offset))
                for view, scale, penalty in (("pc5", rule.pc5, 1.), ("pc2", rule.pc2, 1.), ("oral", rule.oral, .12)):
                    if scale == 0.:
                        continue
                    key = rule.diet, view, rule.robust
                    if rule.pooling and key not in priors:
                        priors[key] = domain_prior([d[key] for d in donors])
                    correction += scale * prediction(target.profiles[key], penalty, rule.pooling, priors.get(key))
                on.append(offset + correction)
            off, on = np.column_stack(off), np.column_stack(on)
        check(np.isfinite(off).all() and np.isfinite(on).all(), "V7_NONFINITE_PREDICTION")
        results[e] = Fit(target.rows, target.y, target.no_diet, off, on)
    return results


def append_upgrade(fit, upgrade):
    j, w = upgrade.index, upgrade.weight
    off = fit.off[:, 0] + w*(fit.off[:, j]-fit.off[:, 0])
    on = fit.on[:, 0] + w*(fit.on[:, j]-fit.on[:, 0])
    return Fit(fit.rows, fit.outcome, fit.no_diet, np.column_stack((fit.off, off)), np.column_stack((fit.on, on)))


class Repartition:
    """Refit every fold-dependent transform for a new grouping of the same families."""

    def __init__(self, study, sites, calibration, seed, progress):
        self.study = study
        self.family, self.keys = study.family, study.keys
        self.numeric, self.markers, self.ecology, self.smoking = study.numeric, study.markers, study.ecology, study.smoking
        _, codes = np.unique(study.family, return_inverse=True)
        permutation = np.random.default_rng(seed).permutation(int(codes.max())+1)
        self.fold = permutation[codes] % len(p.FOLDS)
        self.backgrounds = {}
        for fold in p.FOLDS:
            train = self.fold != fold
            oral, gut = (fit_site(s, sites[s], study.keys[0], train) for s in ("oral", "gut"))
            values, meta, _ = calibration.refit(train, 10*seed + fold)
            archived = {**study.backgrounds[0].archived, "UPF": values}
            self.backgrounds[fold] = Background(train, oral[1] & gut[1], np.column_stack((oral[0], gut[0])), archived, meta)
            progress("REPARTITION_FOLD_READY", dict(fold=fold, seed=seed))

    def eligible(self, exposure, endpoint, fold, cleaned):
        active = np.ones_like(self.fold, dtype=bool)
        context = InnerStudy(self.study, fold, self.backgrounds[fold], active)
        return context.eligible(exposure, endpoint, fold, cleaned)


def fit_all(study, outcomes, calibration, partition_seed, progress):
    sites = read_sites(study)
    working = study if partition_seed is None else Repartition(study, sites, calibration, partition_seed, progress)
    inner = InnerContexts(working, sites)
    fits = {k: [] for k in p.MODES["all"]}
    decisions = []
    for outer_fold in p.FOLDS:
        candidates = {x: {key: [] for key in outcomes} for x in p.EXPOSURES}
        inner_outcomes = {e: Outcome(np.where(working.fold != outer_fold, y.y, np.nan), y.unit)
                          for e, y in outcomes.items()}
        for held_fold in p.FOLDS:
            if held_fold == outer_fold:
                continue
            ctx = inner.split(outer_fold, held_fold)
            for exposure in p.EXPOSURES:
                if exposure == "UPF":
                    ctx, _ = calibration.inner(ctx, held_fold, 100*(partition_seed or p.BOOTSTRAP_SEED)+10*outer_fold+held_fold)
                for endpoint, fit in fit_split(ctx, held_fold, exposure, inner_outcomes).items():
                    candidates[exposure][endpoint].append(fit)
        for exposure in p.EXPOSURES:
            for endpoint, fit in fit_split(working, outer_fold, exposure, outcomes).items():
                choice = choose_upgrade(candidates[exposure][endpoint])
                check(np.all(working.fold[fit.rows] == outer_fold), "V7_WRONG_HELD_FOLD")
                fits[exposure, endpoint].append(append_upgrade(fit, choice))
                if choice.weight:
                    decisions.append(dict(outer=outer_fold, exposure=exposure, endpoint=endpoint,
                                          rule=RULES[choice.index].name, weight=choice.weight))
        progress("OUTER_DONE", dict(fold=outer_fold, pairs=len(fits), accepted_upgrades=len(decisions)))
    return fits, decisions
