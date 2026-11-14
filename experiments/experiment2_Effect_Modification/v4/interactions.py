"""Explicit fiber-by-microbiome correction candidates.

All predictions add centered dose products to the same fitted baseline.
A 10-to-15 contrast replaces the held dose basis with its 10-to-15
difference while keeping each held participant's marker vector fixed.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from scipy.linalg import lstsq, solve
from sklearn.linear_model import ElasticNet


ZERO = "zero"
ECOLOGY = ("oral_shannon", "oral_dominance", "gut_shannon", "gut_dominance")
RIDGE_PENALTIES = (0.1, 1.0, 10.0)


@dataclass(frozen=True)
class Family:
    markers: tuple[str, ...]
    n_dose: int
    method: str = "ridge"
    strengths: tuple[float, ...] = RIDGE_PENALTIES


FAMILIES = {
    "gut_shannon": Family(("gut_shannon",), 1, strengths=(0.1, 1.0)),
    "oral_gut_ecology": Family(ECOLOGY, 1),
    "oral_gut_ecology_spline": Family(ECOLOGY, 2),
    "gut_linear_ridge": Family(("learned_gut",), 1),
    "gut_spline_ridge": Family(("learned_gut",), 2),
    "oral_spline_ridge": Family(("learned_oral",), 2),
    "both_spline_ridge": Family(("learned_oral", "learned_gut"), 2),
    "both_smooth_ridge": Family(("learned_oral", "learned_gut"), 2, "smooth"),
    "both_spline_elastic": Family(("learned_oral", "learned_gut"), 2, "elastic", (0.15, 0.5)),
    "both_spline_spectral": Family(("learned_oral", "learned_gut"), 2, "spectral"),
    "both_site_shrink": Family(("learned_oral", "learned_gut"), 2, "site_shrink", (0.1, 10.0)),
    "both_pc5_spline": Family(("learned_oral", "learned_gut"), 2, "pc5"),
    "both_projected_ridge": Family(("learned_oral", "learned_gut"), 2, "projected"),
    "ecology_hinge_curve": Family(ECOLOGY, 2, "hinge"),
    "oral_gut_tensor_curve": Family(("learned_oral", "learned_gut"), 2, "tensor"),
    "both_huber_curve": Family(("learned_oral", "learned_gut"), 2, "huber", (0.1, 1.0)),
    "both_adaptive_group_curve": Family(("learned_oral", "learned_gut"), 2, "adaptive_group", (0.1, 1.0)),
    "fermentation": Family(("fermentation",), 1, strengths=(0.1, 1.0)),
    "shannon_fermentation_joint": Family(("gut_shannon", "fermentation"), 1),
    "shannon_fermentation_curve": Family(("gut_shannon", "fermentation"), 2),
    "fermentation_smooth_curve": Family(("fermentation",), 2, "smooth"),
}

SETTINGS = {
    f"{name}@{strength:g}": (name, strength)
    for name, spec in FAMILIES.items()
    for strength in spec.strengths
}
ROUTES = {
    name: (ZERO, *(f"{name}@{strength:g}" for strength in spec.strengths))
    for name, spec in FAMILIES.items()
}


def available_routes(role):
    """Keep all core routes and add score routes when its column is supplied."""
    return {name: keys for name, keys in ROUTES.items()
            if "fermentation" not in FAMILIES[name].markers or "fermentation" in role.markers}


def _center(train, valid):
    mean = train.mean(axis=0)
    return train - mean, valid - mean


def _product(dose, markers):
    return (dose[:, :, None] * markers[:, None, :]).reshape(len(dose), dose.shape[1] * markers.shape[1])


def _smooth_markers(train, valid):
    # A restricted cubic spline adds one smooth term per retained marker.
    # Knots, scaling and centering all use the training marker distribution.
    lo, mid, hi = np.quantile(train, (0.1, 0.5, 0.9), axis=0)
    use = (mid - lo > 1e-8) & (hi - mid > 1e-8)
    lo, mid, hi = lo[use], mid[use], hi[use]
    width = hi - lo

    def transform(values):
        u = (values[:, use] - lo) / width
        knot = (mid - lo) / width
        bounded = np.clip(u, 0, 1)
        curve = bounded ** 3 - np.maximum(bounded - knot, 0) ** 3 / (1 - knot)
        return curve + 3 * knot * np.maximum(u - 1, 0)

    fit, held = _center(transform(train), transform(valid))
    return np.column_stack((train, fit)), np.column_stack((valid, held))


def _hinge_markers(train, valid):
    """Allow different slopes beyond the training ecology quartiles."""
    lower, upper = np.quantile(train, (0.25, 0.75), axis=0)
    return tuple(np.column_stack((part, np.maximum(lower - part, 0), np.maximum(part - upper, 0)))
                 for part in (train, valid))


def _site_tensor(blocks):
    """Use at most three PCs per site in the extra oral-by-gut products."""
    oral, gut = blocks
    n_oral = min(3, max(0, oral[0].shape[1] - 2))
    n_gut = min(3, max(0, gut[0].shape[1] - 2))
    fit, held = (_product(a[:, :n_oral], b[:, :n_gut]) for a, b in zip(oral, gut, strict=True))
    return _center(fit, held)


def _marker_blocks(role, spec, lengths):
    blocks = []
    for name in spec.markers:
        if name not in role.markers:
            raise ValueError(f"INTERACTION_MARKER_MISSING {name}")
        train, valid = (np.asarray(part, dtype=float) for part in role.markers[name])
        if (train.ndim != 2 or train.shape[0] != lengths[0] or valid.shape != (lengths[1], train.shape[1])
                or not np.isfinite(train).all() or not np.isfinite(valid).all()):
            raise ValueError(f"INVALID_INTERACTION_MARKER {name}")
        if spec.method == "pc5":
            # The two last columns of each site's learned block are ecology markers.
            count = min(5, max(0, train.shape[1] - 2))
            train, valid = train[:, :count], valid[:, :count]
        if spec.method == "smooth":
            train, valid = _smooth_markers(train, valid)
        elif spec.method == "hinge":
            train, valid = _hinge_markers(train, valid)
        blocks.append(_center(train, valid))
    if spec.method == "tensor":
        blocks.append(_site_tensor(blocks))
    return blocks


def _residualize_training(role, products, residual):
    """Fit the interaction on residualized training data only."""
    background = np.asarray(role.design[0], dtype=float)
    if background.ndim != 2 or len(background) != len(residual) or np.isinf(background).any():
        raise ValueError("INVALID_INTERACTION_PROJECTION_BACKGROUND")
    missing = np.isnan(background)
    medians = np.array([np.median(col[~miss]) if (~miss).any() else 0.0
                        for col, miss in zip(background.T, missing.T, strict=True)])
    filled = np.where(missing, medians, background)
    nuisance = np.column_stack((filled, missing))
    nuisance -= nuisance.mean(axis=0)
    scale = nuisance.std(axis=0)
    use = scale > 1e-10
    nuisance = np.column_stack((np.ones(len(products)), nuisance[:, use] / scale[use]))
    target = np.column_stack((products, residual))
    coef = lstsq(nuisance, target, cond=1e-8, lapack_driver="gelsy")[0]
    projected = target - nuisance @ coef
    return projected[:, :-1], projected[:, -1]


def _huber_ridge(train, target, penalty, initial):
    """Iteratively reweight using a fixed training MAD cutoff."""
    mad = np.median(np.abs(target - np.median(target)))
    cutoff = max(1.345 * 1.4826 * mad, 1e-8)
    beta = initial
    for _ in range(30):
        error = np.abs(target - train @ beta)
        weights = cutoff / np.maximum(error, cutoff) / len(target)
        weighted_train = train * weights[:, None]
        updated = solve(train.T @ weighted_train + penalty, weighted_train.T @ target, assume_a="pos")
        change = np.linalg.norm(updated - beta)
        beta = updated
        if change <= 1e-7 * (1 + np.linalg.norm(beta)):
            break
    return beta


def _adaptive_group_penalty(gram, cross, identity, marker_ids, width):
    """Share a bounded adaptive penalty across the same marker's dose terms."""
    pilot = solve(gram + identity, cross, assume_a="pos")
    size = np.sqrt(np.bincount(marker_ids, weights=pilot ** 2, minlength=width))[marker_ids]
    supported = size > 1e-10
    reference = np.median(size[supported]) if supported.any() else 1e-10
    weights = np.clip(reference / np.maximum(size, 1e-10), 0.25, 4.0)
    return np.diag(weights)


def _fit_family(role, residual, dose, delta, spec, settings):
    n_train, n_valid = map(len, dose)
    blocks = _marker_blocks(role, spec, (n_train, n_valid))
    fit_m, held_m = (np.column_stack(parts) for parts in zip(*blocks, strict=True))
    width = fit_m.shape[1]
    d0, d1 = (part[:, :spec.n_dose] for part in dose)
    train = _product(d0, fit_m)
    held = _product(d1, held_m)
    contrast = (delta[None, :spec.n_dose, None] * held_m[:, None, :]).reshape(len(held), -1)
    scale = np.sqrt(np.mean(train ** 2, axis=0))
    use = scale > 1e-10
    if not use.any():
        zeros = np.zeros(n_valid)
        return {key: (zeros, zeros) for key, _ in settings}
    train, held, contrast = (part[:, use] / scale[use] for part in (train, held, contrast))

    target = residual
    if spec.method == "projected":
        train, target = _residualize_training(role, train, target)

    gram = train.T @ train / n_train
    cross = train.T @ target / n_train
    identity = np.eye(len(cross))
    if spec.method == "spectral":
        eigen, vec = np.linalg.eigh(gram)
        # Penalize nearly collinear directions more than the top eigenmode.
        spectral_penalty = np.maximum(eigen[-1], 1e-12) / np.maximum(eigen, 1e-12)
        coordinates = vec.T @ cross
    if spec.method == "site_shrink":
        oral_width = blocks[0][0].shape[1]
        oral = np.tile(np.arange(width) < oral_width, spec.n_dose)[use]
    if spec.method == "elastic":
        alpha_max = np.max(np.abs(cross)) / 0.5
        train = np.asfortranarray(train)
    if spec.method == "adaptive_group":
        marker_ids = np.tile(np.arange(width), spec.n_dose)[use]
        adaptive = _adaptive_group_penalty(gram, cross, identity, marker_ids, width)

    result = {}
    for key, strength in settings:
        if spec.method == "spectral":
            beta = vec @ (coordinates / (np.maximum(eigen, 0) + strength * spectral_penalty))
        elif spec.method == "site_shrink":
            penalty = np.diag(np.where(oral, strength, 1.0 / strength))
            beta = solve(gram + penalty, cross, assume_a="pos")
        elif spec.method == "elastic":
            beta = np.zeros(len(cross))
            if alpha_max > 1e-10:
                model = ElasticNet(alpha=strength * alpha_max, l1_ratio=0.5, fit_intercept=False,
                                   tol=1e-6, max_iter=4000)
                beta = model.fit(train, target).coef_
        elif spec.method == "huber":
            penalty = strength * identity
            beta = _huber_ridge(train, target, penalty, solve(gram + penalty, cross, assume_a="pos"))
        elif spec.method == "adaptive_group":
            beta = solve(gram + strength * adaptive, cross, assume_a="pos")
        else:
            beta = solve(gram + strength * identity, cross, assume_a="pos")
        # Even for the projected fit, replay only the original explicit products.
        result[key] = (held @ beta, contrast @ beta)
    return result


def fit_corrections(role, residual, candidates=None):
    """Return {candidate: (held correction, 10-to-15 correction contrast)}.

    Select penalties using inner household splits outside this function.
    `residual` must come from the current training role alone. A zero-only
    call does not require training residuals.
    """
    routes = available_routes(role)
    allowed = dict.fromkeys(key for keys in routes.values() for key in keys)
    if candidates is None:
        requested = tuple(allowed)
    elif isinstance(candidates, str):
        requested = (candidates,)
    else:
        requested = tuple(dict.fromkeys(candidates))
    if any(key not in allowed for key in requested):
        raise ValueError("UNKNOWN_INTERACTION_CANDIDATE")
    result = {}
    if ZERO in requested:
        result[ZERO] = (np.zeros(len(role.rows)), np.zeros(len(role.rows)))
    active = [key for key in requested if key != ZERO]
    if not active:
        return result

    train_dose, valid_dose = (np.asarray(part, dtype=float) for part in role.dose)
    residual = np.asarray(residual, dtype=float)
    delta = np.asarray(role.delta, dtype=float)
    if (train_dose.ndim != 2 or not len(train_dose) or train_dose.shape[1] < 2
            or valid_dose.shape != (len(role.rows), train_dose.shape[1])
            or delta.shape != (train_dose.shape[1],) or residual.shape != (len(train_dose),)
            or not all(np.isfinite(part).all() for part in (train_dose, valid_dose, delta, residual))):
        raise ValueError("INVALID_INTERACTION_TRAINING_INPUT")
    dose = _center(train_dose, valid_dose)
    by_family = {}
    for key in active:
        name, strength = SETTINGS[key]
        by_family.setdefault(name, []).append((key, strength))
    for name, settings in by_family.items():
        result.update(_fit_family(role, residual, dose, delta, FAMILIES[name], settings))
    if not all(np.isfinite(part).all() for parts in result.values() for part in parts):
        raise ValueError("NONFINITE_INTERACTION_CORRECTION")
    return {key: result[key] for key in requested}


def interaction_selftest():
    """Check explicit product and contrast replay on generated role data."""
    rng = np.random.default_rng(625)
    n, m = 260, 60
    train_dose, held_dose = rng.normal(size=(n, 2)), rng.normal(size=(m, 2))
    marker_widths = dict.fromkeys((*ECOLOGY, "fermentation"), 1)
    marker_widths.update(learned_oral=14, learned_gut=14)
    markers = {
        name: (rng.normal(size=(n, width)), rng.normal(size=(m, width)))
        for name, width in marker_widths.items()
    }
    background = np.column_stack((train_dose, *(values[0] for values in markers.values())))
    background[:4, -1] = np.nan
    delta = np.array([1.0, -0.15])
    role = SimpleNamespace(rows=np.arange(m), dose=(train_dose, held_dose), delta=delta,
                           markers=markers, design=(background,))

    d0, d1 = _center(train_dose[:, :1], held_dose[:, :1])
    z0, z1 = _center(*markers["gut_shannon"])
    residual = 4 * (d0 * z0)[:, 0]
    candidates = fit_corrections(role, residual)
    shifted = SimpleNamespace(**{**vars(role), "dose": (train_dose, held_dose + delta)})
    shifted_candidates = fit_corrections(shifted, residual)
    assert set(candidates) == {key for keys in ROUTES.values() for key in keys}
    for key, (prediction, contrast) in candidates.items():
        np.testing.assert_allclose(shifted_candidates[key][0] - prediction, contrast, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(candidates["gut_shannon@0.1"][0], 4 * (d1 * z1)[:, 0] / 1.1)

    # Changing held marker rows cannot alter the correction on earlier held rows.
    changed_markers = {name: (fit, np.vstack((held[:m//2], held[m//2:] + 20)))
                       for name, (fit, held) in markers.items()}
    held_changed = SimpleNamespace(**{**vars(role), "markers": changed_markers})
    changed_candidates = fit_corrections(held_changed, residual)
    for key, pair in candidates.items():
        for unchanged, changed in zip(pair, changed_candidates[key], strict=True):
            np.testing.assert_allclose(unchanged[:m//2], changed[:m//2], rtol=1e-11, atol=1e-11)

    no_score = SimpleNamespace(**{**vars(role), "markers": {name: value for name, value in markers.items()
                                                            if name != "fermentation"}})
    assert set(ROUTES) - set(available_routes(no_score)) == {
        "fermentation", "shannon_fermentation_joint", "shannon_fermentation_curve", "fermentation_smooth_curve",
    }
    zero = fit_corrections(no_score, None, ZERO)[ZERO]
    assert all(np.array_equal(values, np.zeros(m)) for values in zero)
    np.testing.assert_allclose(fit_corrections(role, residual, "both_projected_ridge@0.1")["both_projected_ridge@0.1"],
                               candidates["both_projected_ridge@0.1"])

    _, smooth = _smooth_markers(np.arange(100.)[:, None], np.array([[1e9], [2e9], [3e9]]))
    np.testing.assert_allclose(smooth[2, 1] - smooth[1, 1], smooth[1, 1] - smooth[0, 1])

    # Recover a planted cross-site modifier with no per-site linear signal.
    oral_fit, oral_held = _center(*markers["learned_oral"])
    gut_fit, gut_held = _center(*markers["learned_gut"])
    t_fit, t_held = _center(oral_fit[:, 0] * gut_fit[:, 0], oral_held[:, 0] * gut_held[:, 0])
    truth = 3 * d1[:, 0] * t_held
    tensor = fit_corrections(role, 3 * d0[:, 0] * t_fit, "oral_gut_tensor_curve@0.1")[
        "oral_gut_tensor_curve@0.1"
    ][0]
    assert np.corrcoef(truth, tensor)[0, 1] > 0.95

    # A high-leverage contaminated residual should affect Huber less than ridge.
    clean = 3 * d0[:, 0] * gut_fit[:, 0]
    truth = 3 * d1[:, 0] * gut_held[:, 0]
    contaminated = clean.copy()
    idx = np.argmax(np.abs(clean))
    contaminated[idx] += 1000 * np.sign(clean[idx])
    robust, ridge = fit_corrections(role, contaminated, ("both_huber_curve@0.1", "both_spline_ridge@0.1")).values()
    assert np.linalg.norm(robust[0] - truth) < np.linalg.norm(ridge[0] - truth) / 2
