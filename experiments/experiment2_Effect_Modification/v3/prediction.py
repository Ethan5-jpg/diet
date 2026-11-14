"""Same-base, grouped out-of-fold fiber-by-marker prediction."""

from dataclasses import replace

import numpy as np

from moderation_models import BASES, SEED, ZERO, correction, fit_base, prepare, residuals, scores


ROUTES = ("gut_shannon", "fermentation")
OPTIONS = {name: (ZERO, *((name, 1, 1, strength) for strength in (0.1, 1.0))) for name in ROUTES}
BASE = BASES[0]


def _with_score(source, split, score, *, association=False):
    values = (score[split.train_rows], score[split.rows])
    if not all(np.isfinite(part).all() for part in values):
        raise ValueError("NONFINITE_FERMENTATION_SCORE_IN_ROLE")

    center, spread = values[0].mean(), values[0].std()
    # A constant training score carries no information about a held-out correction.
    standardized = (tuple(np.zeros((len(part), 1)) for part in values) if spread <= 1e-10
                    else tuple(((part - center) / spread)[:, None] for part in values))
    if not all(np.isfinite(part).all() for part in standardized):
        raise ValueError("INVALID_FERMENTATION_SCORE_SCALE")

    role = prepare(source, split, association=association, shared_host=True)
    design = tuple(np.column_stack((base, column)) for base, column in zip(role.design, standardized, strict=True))
    background = role.association_main
    if association:
        held = standardized[1]
        background = np.column_stack((background, held, held ** 2))
        if not np.isfinite(background).all():
            raise ValueError("INVALID_FERMENTATION_ASSOCIATION_BACKGROUND")
    return replace(role, design=design, markers={**role.markers, "fermentation": standardized},
                   association_main=background)


def _predict_base(role):
    model = fit_base(role.design[0], role.y[0], role.width, BASE, shared_columns=role.shared_columns)
    return model.predict(role.design[1])


def run_fold(source, repeat, fold, score):
    """Fit one shared baseline, then choose each linear interaction inside training."""
    score = np.asarray(score, dtype=float)
    if score.shape != (len(source.groups),):
        raise ValueError("FERMENTATION_SCORE_ROW_ALIGNMENT")
    inner, outer = source.roles(repeat, fold)
    held_groups = source.groups[outer.rows]
    if any(np.intersect1d(source.groups[np.r_[split.train_rows, split.rows]], held_groups).size
           for split in inner):
        raise ValueError("INNER_OUTER_HOUSEHOLD_OVERLAP")

    losses = {key: 0.0 for options in OPTIONS.values() for key in options}
    for split in inner:
        role = _with_score(source, split, score)
        base_error = role.y[1] - _predict_base(role)
        losses[ZERO] += base_error @ base_error
        training_residual = residuals(role, BASE)
        for options in OPTIONS.values():
            for key in options[1:]:
                added, _ = correction(role, training_residual, *key)
                error = base_error - added
                losses[key] += error @ error
    choices = {name: min(options, key=lambda key: round(losses[key], 9)) for name, options in OPTIONS.items()}

    role = _with_score(source, outer, score, association=True)
    baseline = _predict_base(role)
    training_residual = residuals(role, BASE) if any(key != ZERO for key in choices.values()) else None
    added, contrasts = zip(*(correction(role, training_residual, *key) for key in choices.values()), strict=True)
    full = baseline[:, None] + np.column_stack(added)
    contrast = np.column_stack(contrasts)
    if not np.isfinite(full).all() or not np.isfinite(contrast).all():
        raise ValueError("INVALID_HELD_OUT_PREDICTION")
    return role, baseline, full, contrast, choices, BASE[0]


def summarize_prediction(y, baseline, full, groups, *, seed=SEED, draws=1999):
    """Report pooled R² and paired fixed-prediction household intervals."""
    y, baseline, full = (np.asarray(part, dtype=float) for part in (y, baseline, full))
    groups = np.asarray(groups)
    if (y.ndim != 1 or baseline.shape != y.shape or groups.shape != y.shape
            or full.shape != (len(y), len(ROUTES)) or not isinstance(draws, int) or draws < 1
            or not all(np.isfinite(part).all() for part in (y, baseline, full))):
        raise ValueError("INVALID_PAIRED_PREDICTION_INPUT")
    base_r2, full_r2, delta, ci = scores(y, baseline, full, groups, seed=seed, draws=draws)
    # Use the same households and seed for the direct paired comparison. Its
    # interval uses the paired SSE difference, not differences of interval ends.
    _, _, direct, direct_ci = scores(y, full[:, 0], full[:, 1:], groups, seed=seed, draws=draws)
    return {
        "baseline_r2": float(base_r2),
        "routes": {
            name: {"r2": float(full_r2[j]), "delta_r2": float(delta[j]), "delta_ci": ci[j].tolist()}
            for j, name in enumerate(ROUTES)
        },
        "fermentation_minus_shannon": {"delta_r2": float(direct[0]), "delta_ci": direct_ci[0].tolist()},
    }


def prediction_selftest():
    # At the summary boundary, a duplicated route has an exact paired zero.
    y = np.array([1., 2., 3., 2., 4., 6.])
    groups = np.repeat(np.arange(3), 2)
    baseline = np.full_like(y, 3.)
    shannon = (y + baseline) / 2
    full = np.column_stack((shannon, y))
    result = summarize_prediction(y, baseline, full, groups, draws=63)
    routes = result["routes"]
    direct = result["fermentation_minus_shannon"]
    expected = routes["fermentation"]["r2"] - routes["gut_shannon"]["r2"]
    np.testing.assert_allclose(direct["delta_r2"], expected)
    assert direct["delta_ci"][0] > 0

    same = summarize_prediction(y, baseline, np.column_stack((shannon, shannon)), groups, draws=63)
    assert same["fermentation_minus_shannon"] == {"delta_r2": 0.0, "delta_ci": [0.0, 0.0]}
