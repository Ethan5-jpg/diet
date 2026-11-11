"""Correction estimators: model the fiber slope, not the entire outcome."""

import warnings

import numpy as np
from scipy import linalg, optimize
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeRegressor

from exp2_design import Projection
from exp2_expansion import (NEW_CURVES, NEW_FAMILIES, RankConvergenceError,
                            adaptive_weights, component_path, rank_curve_fit)
from exp2_followup import FOLLOWUP_FAMILIES, ard_fit, dose_weights, pcr_path


CORE_OTHER = ("elastic", "fourier", "hellinger", "forest", "boost", "neural")
RIDGE = ("ridge_pc10", "ridge_pc30", "ridge_spline", "ridge_ecology")
SPARSE = ("elastic", "taxa_sparse")
V3_ONLY = ("spline_modifier", "huber_curve", "local_curve", "oral_gut_plus")
V2_ADDED = ("taxa_screen", "group_curve", "boost_curve")
PARAMETRIC = RIDGE + SPARSE + (
    "fourier", "hellinger", "pls", "fourier_curve", "dose_hinge", "oral_gut_tensor",
    "spline_modifier", "huber_curve", "oral_gut_plus", "taxa_screen", "group_curve",
)
CURVED = ("ridge_spline", "fourier_curve", "huber_curve", "group_curve")


class FitFailure(RuntimeError):
    """Known solver failure in a single candidate setting."""


def specifications(cfg):
    result = []
    for family, settings in cfg["grids"].items():
        for site in cfg["sites"]:
            if family in ("oral_gut_tensor", "oral_gut_plus") and site != "both":
                continue
            core = family in CORE_OTHER or family in ("ridge_pc10", "ridge_pc30") or (
                family == "ridge_spline" and site != "oral"
            )
            if family in FOLLOWUP_FAMILIES:
                edition = 6
            elif family in NEW_FAMILIES:
                edition = 5
            else:
                edition = 1 if core else 4 if family in V2_ADDED else 3 if family in V3_ONLY else 2
            result.append(dict(name=family + "_" + site, family=family, site=site,
                               core=core, edition=edition, settings=[None] + settings))
    return result


def ridge(xt, y, penalty, free_intercept=False):
    gram = xt.T @ xt
    diagonal = np.arange(0, gram.size, len(gram) + 1)
    gram.flat[diagonal[int(free_intercept):]] += len(y) * penalty
    return linalg.cho_solve(linalg.cho_factor(gram, lower=True, check_finite=False),
                            xt.T @ y, check_finite=False)


def products(diet, microbial):
    return np.column_stack([diet[:, j:j + 1] * microbial for j in range(diet.shape[1])])


class ProductMap:
    """Observed and replayed designs in the same training-fitted coordinates."""

    def __init__(self, role, microbial, family, contrast, projection=None, contrast_main=None):
        projection = role.projection if projection is None else projection
        if family == "dose_hinge":
            dt, dv, lo, hi = role.dose_hinge
        elif family == "local_dose_curve":
            dt, dv, lo, hi = role.dose_bins
        else:
            width = 2 if family in CURVED + NEW_CURVES + FOLLOWUP_FAMILIES else 1
            dt, dv, lo, hi = (d[:, :width] for d in (*role.diet, role.lo, role.hi))
        raw = products(dt, microbial[0])
        coefficient = projection.coefficients(raw)
        z = projection.residual(raw)
        scale = z.std(0)
        keep = np.flatnonzero(scale > 1e-7 * np.maximum(1, raw.std(0)))
        sd = scale[keep]
        self.keep = keep
        self.train = z[:, keep] / sd
        self.val = projection.transform(products(dv, microbial[1])[:, keep],
                                        coefficient[:, keep]) / sd
        self.delta = None
        if contrast:
            raw_lo, raw_hi = (products(d, microbial[1])[:, keep] for d in (lo, hi))
            if contrast_main is None:
                self.delta = role.contrast(raw_lo, raw_hi, coefficient[:, keep]) / sd
            else:
                self.delta = projection.transform(raw_hi - raw_lo, coefficient[:, keep], x=contrast_main) / sd


def huber_ridge(x, y, penalty):
    """Huber training loss with a fixed training MAD threshold and L2 penalty."""
    if np.max(np.abs(y)) <= 1e-10:
        return np.zeros(x.shape[1])
    mad = np.median(np.abs(y - np.median(y)))
    threshold = max(1e-4, 1.345 * 1.4826 * mad)

    def objective(beta):
        error = x @ beta - y
        clipped = np.clip(error, -threshold, threshold)
        loss = 0.5 * clipped ** 2 + threshold * (np.abs(error) - np.abs(clipped))
        value = loss.mean() + 0.5 * penalty * (beta @ beta)
        gradient = (x.T @ clipped) / len(x) + penalty * beta
        return value, gradient

    fit = optimize.minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                            options={"maxiter": 150, "ftol": 1e-10, "gtol": 1e-6})
    if not fit.success or not np.isfinite(fit.x).all() or not np.isfinite(fit.fun):
        raise FitFailure("failed_huber_curve_status_%s" % fit.status)
    return fit.x


def fit_linear_setting(x, y, family, value):
    """Return coefficients in the already projected, scaled product basis."""
    if family == "huber_curve":
        return huber_ridge(x, y, value)
    if family in SPARSE:
        alpha_max = np.max(np.abs(x.T @ y)) / (len(y) * 0.65)
        if alpha_max <= 1e-12:
            return np.zeros(x.shape[1])
        model = ElasticNet(alpha=alpha_max * value, l1_ratio=0.65, fit_intercept=False,
                           max_iter=3500, tol=1e-5)
    elif family == "pls":
        if np.max(np.abs(x.T @ y)) / len(y) <= 1e-12:
            return np.zeros(x.shape[1])
        model = PLSRegression(n_components=min(value, x.shape[1], len(y) - 1),
                              scale=False, max_iter=500, tol=1e-6)
    else:
        return ridge(x, y, value)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            model.fit(x, y)
        except ConvergenceWarning as exc:
            raise FitFailure("failed_" + family + "_convergence") from exc
    coefficient = model.coef_.ravel()
    if not np.isfinite(coefficient).all():
        raise FitFailure("failed_" + family + "_nonfinite")
    # PLS centers internally. The projected products and residual target
    # already have mean zero, so only its linear coefficient is replayed.
    return coefficient


def screened_taxa(role, microbial, residual):
    """Rank eligible taxa by the training product after removing their own main term."""
    train, valid = microbial
    marker = role.projection.residual(train)
    interaction = role.projection.residual(train * role.diet[0][:, :1])
    marker_ss = np.sum(marker ** 2, axis=0)
    adjust = np.divide(np.sum(marker * interaction, axis=0), marker_ss,
                       out=np.zeros_like(marker_ss), where=marker_ss > 1e-9)
    interaction -= marker * adjust
    norm = np.linalg.norm(interaction, axis=0)
    score = np.divide(np.abs(interaction.T @ residual), norm,
                      out=np.zeros_like(norm), where=norm > 1e-7)
    chosen = np.argsort(-score, kind="stable")[:role.screen_keep]
    return train[:, chosen], valid[:, chosen]


def group_curve_fit(x, y, group, fraction):
    """Fit a squared-loss group elastic net with one group per microbial PC."""
    _, groups = np.unique(group, return_inverse=True)
    n = len(x)
    gram = x.T @ x / n
    gram.flat[::len(gram) + 1] += 0.005
    xy = x.T @ y / n
    max_group = np.sqrt(np.bincount(groups, weights=xy ** 2)).max()
    if max_group <= 1e-12:
        return np.zeros(x.shape[1])
    penalty = fraction * max_group
    lipschitz = linalg.eigvalsh(gram, subset_by_index=[len(gram) - 1, len(gram) - 1])[0]

    def prox(value):
        norms = np.sqrt(np.bincount(groups, weights=value ** 2))
        shrink = np.maximum(0.0, 1.0 - penalty / (lipschitz * np.maximum(norms, 1e-30)))
        return value * shrink[groups]

    coefficients = accelerated = np.zeros(x.shape[1])
    momentum = 1.0
    for _ in range(1000):
        candidate = prox(accelerated - (gram @ accelerated - xy) / lipschitz)
        stationary = prox(candidate - (gram @ candidate - xy) / lipschitz)
        if np.linalg.norm(candidate - stationary) * lipschitz <= 1e-6 * max(1.0, max_group):
            return candidate
        next_momentum = (1 + np.sqrt(1 + 4 * momentum ** 2)) / 2
        accelerated = candidate + (momentum - 1) / next_momentum * (candidate - coefficients)
        coefficients, momentum = candidate, next_momentum
    raise FitFailure("failed_group_curve_convergence")


def parametric(role, spec, settings, residual, contrast):
    family, site = spec["family"], spec["site"]
    microbial = role.features[(family, site)]
    if family == "taxa_screen" and microbial[0].shape[1]:
        if np.max(np.abs(residual)) <= 1e-10:
            zero = np.zeros(len(role.split.rows))
            return [(zero, zero.copy() if contrast else None, "fit") for _ in settings]
        microbial = screened_taxa(role, microbial, residual)
        # Give selected taxa their own nuisance main effects without changing B.
        x = tuple(np.column_stack([base, taxon]) for base, taxon in zip(role.x, microbial))
        projection = Projection.fit(*x, role.project_rcond)
        contrast_main = (np.column_stack([role.x_hi - role.x_lo, np.zeros_like(microbial[1])])
                         if contrast else None)
        mapped = ProductMap(role, microbial, family, contrast, projection, contrast_main)
        residual = projection.target_residual(role.split.yt)
    else:
        mapped = ProductMap(role, microbial, family, contrast)
    out = []
    for value in settings:
        status = "fit"
        try:
            if mapped.train.shape[1] == 0:
                coefficient = np.empty(0)
                status = "empty_product"
            elif family == "group_curve":
                groups = mapped.keep % microbial[0].shape[1]
                coefficient = group_curve_fit(mapped.train, residual, groups, value)
            else:
                # Retain the oral and gut slope blocks and shrink cross-site terms harder.
                penalty = (value * np.where(mapped.keep < role.joint_linear_width, 1.0, 4.0)
                           if family == "oral_gut_plus" else value)
                coefficient = fit_linear_setting(mapped.train, residual, family, penalty)
        except FitFailure as exc:
            coefficient = np.zeros(mapped.train.shape[1])
            status = str(exc)
        out.append((mapped.val @ coefficient,
                    mapped.delta @ coefficient if contrast else None, status))
    return out


def mixed_kernel_products(role, site, family, contrast):
    """Equal total scale for nonempty CLR and Hellinger kernel product blocks."""
    maps = [ProductMap(role, role.features[(key, site)], family, contrast)
            for key in ("fourier", "hellinger")]
    available = [m for m in maps if m.train.shape[1]]
    if not available:
        return maps[0].train, maps[0].val, maps[0].delta
    scale = [np.sqrt(64 / (len(available) * m.train.shape[1])) for m in available]
    return tuple(np.column_stack([getattr(m, key) * s for m, s in zip(available, scale)])
                 if key != "delta" or contrast else None for key in ("train", "val", "delta"))


def expanded_corrections(role, spec, settings, residual, contrast):
    """All six families use the original B. New microbial-only bases are projected out."""
    family, site = spec["family"], spec["site"]
    if family == "kernel_mix_curve":
        x, xv, dx = mixed_kernel_products(role, site, family, contrast)
    else:
        source = {"adaptive_sparse": "elastic", "rank_curve": "ridge_pc10",
                  "component_curve": "taxa_sparse"}.get(family, family)
        microbes = role.features[(source, site)]
        if family in ("soft_cluster", "detection_slope") and microbes[0].shape[1]:
            nuisance = tuple(np.column_stack([a, b]) for a, b in zip(role.x, microbes))
            projection = Projection.fit(*nuisance, role.project_rcond)
            contrast_main = (np.column_stack([role.x_hi - role.x_lo, np.zeros_like(microbes[1])])
                             if contrast else None)
            m = ProductMap(role, microbes, family, contrast, projection, contrast_main)
            residual = projection.target_residual(role.split.yt)
        else:
            m = ProductMap(role, microbes, family, contrast)
        x, xv, dx = m.train, m.val, m.delta
        if family == "rank_curve":
            width = microbes[0].shape[1]
            linear, curve = m.keep[m.keep < width], m.keep[m.keep >= width] - width
            shared = np.intersect1d(linear, curve, assume_unique=True)
            pos = np.searchsorted(m.keep, np.r_[shared, shared + width])
            x, xv, dx = x[:, pos], xv[:, pos], dx[:, pos] if contrast else None
    if not x.shape[1]:
        zero = np.zeros(len(xv))
        return [(zero, zero.copy() if contrast else None, "empty_product") for _ in settings]
    if family == "component_curve":
        betas = component_path(x, residual, settings)
        return [(xv @ b, dx @ b if contrast else None, "fit") for b in betas]
    if family == "adaptive_sparse":
        weights = adaptive_weights(ridge(x, residual, 0.10))
        x, xv, dx = x / weights, xv / weights, dx / weights if contrast else None
    result = []
    for setting in settings:
        status = "fit"
        try:
            if family == "adaptive_sparse":
                beta = fit_linear_setting(x, residual, "elastic", setting)
            elif family == "rank_curve":
                beta = rank_curve_fit(x, residual, setting)
            else:
                beta = ridge(x, residual, setting)
        except (FitFailure, RankConvergenceError) as exc:
            beta = np.zeros(x.shape[1])
            status = str(exc).replace("failed_elastic_", "failed_adaptive_sparse_", 1)
        result.append((xv @ beta, dx @ beta if contrast else None, status))
    return result


def followup_corrections(role, spec, settings, residual, contrast):
    family, site = spec["family"], spec["site"]
    sources = {"interaction_pcr": "ridge_pc30", "dose_robust_curve": "ridge_pc10",
               "local_dose_curve": "ridge_pc10", "ard_curve": "ridge_pc10"}
    profiles = role.features[(sources.get(family, family), site)]
    if not profiles[0].shape[1]:
        zeros = np.zeros(len(role.split.rows))
        return [(zeros, zeros.copy() if contrast else None, "empty_product") for _ in settings]
    if family in ("balance_curve", "two_part_curve", "local_dose_curve"):
        nuisance = role.dose_bins if family == "local_dose_curve" else (*profiles, profiles[1], profiles[1])
        train, val = (np.column_stack([a, b]) for a, b in zip(role.x, nuisance))
        projection = Projection.fit(train, val, role.project_rcond)
        contrast_main = (np.column_stack([role.x_hi - role.x_lo, nuisance[3] - nuisance[2]])
                         if contrast else None)
        mapped = ProductMap(role, profiles, family, contrast, projection, contrast_main)
        residual = projection.target_residual(role.split.yt)
    else:
        mapped = ProductMap(role, profiles, family, contrast)
    if not mapped.train.shape[1]:
        zeros = np.zeros(len(mapped.val))
        return [(zeros, zeros.copy() if contrast else None, "empty_product") for _ in settings]

    x, y = mapped.train, residual
    if family == "interaction_pcr":
        coefficients = [(beta, "fit") for beta in pcr_path(x, y, settings)]
    elif family == "ard_curve":
        coefficients = [ard_fit(x, y, threshold) for threshold in settings]
    else:
        if family == "dose_robust_curve":
            weight = np.sqrt(dose_weights(role.diet[0][:, 0]))
            x, y = x * weight[:, None], y * weight
        coefficients = [(ridge(x, y, penalty), "fit") for penalty in settings]
    return [(mapped.val @ beta, mapped.delta @ beta if contrast else None, status)
            for beta, status in coefficients]


def neighbor_weights(x, xv, neighbors):
    largest = min(max(neighbors), len(x))
    search = NearestNeighbors(n_neighbors=largest, algorithm="brute", n_jobs=1).fit(x)
    neighborhoods = [search.kneighbors(a) for a in (x, xv)]
    for size in neighbors:
        k = min(size, largest)
        rows = []
        for distance, index in neighborhoods:
            distance, index = distance[:, :k], index[:, :k]
            scale = np.maximum(distance[:, -1:], 1e-12)
            rows.append((np.exp(-2 * (distance / scale) ** 2), index))
        yield rows


def local_slopes(x, xv, d, residual, neighbors):
    """A shrunken weighted slope on nearby microbial profiles."""
    for rows in neighbor_weights(x, xv, neighbors):
        slopes = []
        for weight, index in rows:
            dose = d[index]
            numerator = np.sum(weight * dose * residual[index], axis=1)
            denominator = np.sum(weight * dose ** 2, axis=1) + 12.0
            slopes.append(numerator / denominator)
        yield slopes


def local_curve_corrections(role, x, xv, residual, neighbors, contrast):
    """Fit both dose coefficients in the same local weighted regression."""
    dt, dv, lo, hi = role.local_curve_diet
    for rows in neighbor_weights(x, xv, neighbors):
        coefficients = []
        for weight, index in rows:
            dose = dt[index]
            gram = np.einsum("nk,nki,nkj->nij", weight, dose, dose)
            gram += 12.0 * np.eye(2)
            cross = np.einsum("nk,nki,nk->ni", weight, dose, residual[index])
            coefficients.append(np.linalg.solve(gram, cross[..., None])[..., 0])
        train_coef, valid_coef = coefficients
        replay = role.projection.coefficients(np.sum(dt * train_coef, axis=1))
        correction = role.projection.transform(np.sum(dv * valid_coef, axis=1), replay)
        delta = None
        if contrast:
            delta = role.contrast(np.sum(lo * valid_coef, axis=1), np.sum(hi * valid_coef, axis=1), replay)
        yield correction, delta, "fit"


def fit_slope_tree(x, d, r, rows, depth, leaf, seed):
    """Weighted regression on r/d minimizes sum (r - d*g(x))^2."""
    rows = rows[np.abs(d[rows]) > 1e-7]
    if len(rows) < 2 * leaf:
        return None
    tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf,
                                 min_weight_fraction_leaf=0.01, random_state=seed)
    tree.fit(x[rows], r[rows] / d[rows], sample_weight=d[rows] ** 2)
    return tree


def split_forest(x, xv, d, r, households, cfg, seed):
    """Split and leaf fits use disjoint households on common role residuals."""
    rng = np.random.default_rng(seed)
    unique, membership = np.unique(households, return_inverse=True)
    slopes = [np.zeros(len(a)) for a in (x, xv)]
    leaf = min(cfg["leaf"], max(8, len(x) // 10))
    used = 0
    for j in range(40):
        shuffled = rng.permutation(len(unique))[:max(2, int(0.84 * len(unique)))]
        halves = np.array_split(shuffled, 2)
        build, estimate = (np.flatnonzero(np.isin(membership, half)) for half in halves)
        cols = rng.choice(x.shape[1], min(x.shape[1], max(3, int(np.sqrt(x.shape[1]) * 2))),
                          replace=False)
        tree = fit_slope_tree(x[:, cols], d, r, build, cfg["depth"], leaf, seed + j)
        if tree is None or not len(estimate):
            continue
        lid = tree.apply(x[estimate][:, cols])
        n_nodes = tree.tree_.node_count
        count = np.bincount(lid, minlength=n_nodes)
        support = np.bincount(lid, weights=d[estimate] ** 2, minlength=n_nodes)
        numerator = np.bincount(lid, weights=d[estimate] * r[estimate], minlength=n_nodes)
        values = np.where(count >= max(5, leaf // 4), numerator / (support + 12.0), 0.0)
        for i, a in enumerate((x, xv)):
            slopes[i] += values[tree.apply(a[:, cols])]
        used += 1
    if not used:
        return slopes, "no_supported_trees"
    return [a / used for a in slopes], "fit"


def projected_boost(role, x, xv, d, target, cfg, seed):
    """Stagewise trees with projected, shrunken updates and bounded depth."""
    prediction = np.zeros(len(x))
    slopes = [np.zeros(len(a)) for a in (x, xv)]
    rows = np.arange(len(x))
    leaf = max(16, int(0.055 * len(x)))
    for j in range(cfg["steps"]):
        error = target - prediction
        tree = fit_slope_tree(x, d, error, rows, cfg["depth"], leaf, seed + j)
        if tree is None:
            break
        step = tree.predict(x)
        h = role.projection.residual(d * step)
        denom = h @ h
        if denom <= 1e-12:
            break
        weight = 0.10 * np.clip((error @ h) / denom, 0, 1.5)
        prediction += weight * h
        slopes[0] += weight * step
        slopes[1] += weight * tree.predict(xv)
    return slopes, "fit"


def boosted_curve(role, x, xv, target, cfg, seed, contrast):
    """Choose a tree update for either anchored dose component at each step."""
    dt, dv, lo, hi = role.local_curve_diet
    train, valid = np.zeros_like(dt), np.zeros_like(dv)
    prediction = np.zeros(len(x))
    rows = np.arange(len(x))
    leaf = max(16, int(0.055 * len(x)))
    for step in range(cfg["steps"]):
        error = target - prediction
        best_gain = 1e-12
        best = None
        for j in range(dt.shape[1]):
            dose = dt[:, j]
            tree = fit_slope_tree(x, dose, error, rows, cfg["depth"], leaf, seed + 2 * step + j)
            if tree is None:
                continue
            update = tree.predict(x)
            direction = role.projection.residual(dose * update)
            norm = direction @ direction
            if norm <= 1e-12:
                continue
            cross = error @ direction
            weight = 0.15 * np.clip(cross / norm, 0.0, 1.5)
            gain = weight * (2 * cross - weight * norm)
            if gain > best_gain:
                best_gain, best = gain, (j, tree, update, direction, weight)
        if best is None:
            break
        j, tree, update, direction, weight = best
        prediction += weight * direction
        train[:, j] += weight * update
        valid[:, j] += weight * tree.predict(xv)
    coefficient = role.projection.coefficients(np.sum(dt * train, axis=1))
    correction = role.projection.transform(np.sum(dv * valid, axis=1), coefficient)
    delta = None
    if contrast:
        delta = role.contrast(np.sum(lo * valid, axis=1), np.sum(hi * valid, axis=1), coefficient)
    return correction, delta, "fit"


def neural_objective(p, x, d, target, projection, width, penalty):
    """Differentiable objective for the projected one-hidden-layer slope."""
    w = p[:x.shape[1] * width].reshape(x.shape[1], width)
    v = p[x.shape[1] * width:]
    t = np.tanh(x @ w)
    error = projection.residual(d * (t @ v)) - target
    back = d * projection.residual(error) / len(x)
    value = 0.5 * np.mean(error ** 2) + 0.5 * penalty * (p @ p)
    grad_w = x.T @ ((1 - t ** 2) * (back[:, None] * v))
    grad_v = t.T @ back
    return value, np.r_[grad_w.ravel(), grad_v] + penalty * p


def neural(role, x, xv, d, target, cfg, seed):
    x, xv = tuple(np.column_stack([a, np.ones(len(a))]) for a in (x, xv))
    width = cfg["width"]
    scale = max(1e-6, np.sqrt(np.mean(target ** 2)))
    rng = np.random.default_rng(seed)
    p = np.r_[rng.normal(0, 0.3 / np.sqrt(x.shape[1]), x.shape[1] * width),
              rng.normal(0, 0.04, width)]
    fit = optimize.minimize(neural_objective, p, jac=True, method="L-BFGS-B",
                            args=(x, d, target / scale, role.projection, width, cfg["lambda"]),
                            options={"maxiter": 130, "ftol": 1e-8, "gtol": 1e-5})
    if not np.isfinite(fit.fun) or not np.isfinite(fit.x).all() or (not fit.success and fit.status != 1):
        raise FitFailure("failed_neural_status_%s" % fit.status)
    w = fit.x[:x.shape[1] * width].reshape(x.shape[1], width)
    v = fit.x[x.shape[1] * width:]
    status = "fit" if fit.success else "iteration_cap"
    return [scale * (np.tanh(a @ w) @ v) for a in (x, xv)], status


def fit_corrections(role, spec, selected, residual, contrast=False):
    """Return correction, replayed contrast, and status for ordered setting IDs."""
    result = {}
    if 0 in selected:
        zeros = np.zeros(len(role.split.rows))
        result[0] = (zeros, zeros.copy() if contrast else None, "null")
    active = [j for j in selected if j != 0]
    if not active:
        return result
    settings = [spec["settings"][j] for j in active]
    family, site = spec["family"], spec["site"]
    if family in FOLLOWUP_FAMILIES:
        result.update(zip(active, followup_corrections(role, spec, settings, residual, contrast)))
    elif family in NEW_FAMILIES:
        result.update(zip(active, expanded_corrections(role, spec, settings, residual, contrast)))
    elif family in PARAMETRIC:
        result.update(zip(active, parametric(role, spec, settings, residual, contrast)))
    elif family == "local_curve":
        x, xv = role.features[(family, site)]
        result.update(zip(active, local_curve_corrections(role, x, xv, residual, settings, contrast)))
    else:
        x, xv = role.features[(family, site)]
        d = role.diet[0][:, 0]
        nearby = local_slopes(x, xv, d, residual, settings) if family == "local_slope" else None
        for j, value in zip(active, settings):
            # Keep random feature/subsample seeds fixed within a family and site across tuning.
            seed = role.seed + {"oral": 4001, "gut": 5001, "both": 6001}[site]
            if family == "forest":
                (a, b), status = split_forest(x, xv, d, residual, role.groups, value, seed)
            elif family == "boost":
                (a, b), status = projected_boost(role, x, xv, d, residual, value, seed)
            elif family == "neural":
                try:
                    (a, b), status = neural(role, x, xv, d, residual, value, seed)
                except FitFailure as exc:
                    a, b, status = np.zeros(len(x)), np.zeros(len(xv)), str(exc)
            elif family == "boost_curve":
                result[j] = boosted_curve(role, x, xv, residual, value, seed, contrast)
                continue
            elif family == "local_slope":
                a, b = next(nearby)
                status = "fit"
            else:
                raise ValueError("Unknown family: " + family)
            coefficient = role.projection.coefficients(d * a)
            correction = role.projection.transform(role.diet[1][:, 0] * b, coefficient)
            delta = role.contrast(role.lo[:, 0] * b, role.hi[:, 0] * b, coefficient) if contrast else None
            result[j] = (correction, delta, status)
    if not all(np.isfinite(val).all() and (delta is None or np.isfinite(delta).all())
               for val, delta, _ in result.values()):
        raise ValueError("Nonfinite correction: " + spec["name"])
    return result
