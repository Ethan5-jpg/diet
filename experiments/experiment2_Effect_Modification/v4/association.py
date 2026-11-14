"""Joint fixed linear fiber moderation with fold-specific nuisance coefficients."""

from dataclasses import dataclass

import numpy as np
from scipy import stats


LOW_DOSE, HIGH_DOSE = 10.0, 15.0
MARKERS = ("gut_shannon", "fermentation")
RANK_TOL = 1e-10


@dataclass(frozen=True)
class JointFold:
    rows: np.ndarray
    partial_y: np.ndarray
    partial_products: np.ndarray
    product_norms: np.ndarray
    fiber: np.ndarray
    markers: np.ndarray
    nuisance_rank: int


def joint_fold(rows, y, background, fiber, markers):
    """Project the outcome and both interaction columns onto the same complement."""
    rows = np.asarray(rows)
    y, background, fiber, markers = (np.asarray(a, dtype=float) for a in (y, background, fiber, markers))
    n = rows.size
    if (
        n == 0
        or rows.shape != (n,)
        or not np.issubdtype(rows.dtype, np.integer)
        or np.any(rows < 0)
        or np.unique(rows).size != n
        or y.shape != (n,)
        or fiber.shape != (n,)
        or markers.shape != (n, len(MARKERS))
        or background.ndim != 2
        or background.shape[0] != n
        or background.shape[1] == 0
        or not all(np.isfinite(a).all() for a in (y, background, fiber, markers))
    ):
        raise ValueError("INVALID_JOINT_ASSOCIATION_FOLD")

    dose = (fiber - LOW_DOSE) / (HIGH_DOSE - LOW_DOSE)
    centered_markers = markers - markers.mean(axis=0)
    # The shared main effects absorb the fold-specific marker centering.
    # Retain the same marker main-effect basis as the v3 separate fits.
    design = np.column_stack([background, np.ones(n), dose, centered_markers, centered_markers ** 2])
    scale = np.max(np.abs(design), axis=0)
    scale[scale == 0] = 1
    u, singular, _ = np.linalg.svd(design / scale, full_matrices=False)
    rank = int(np.count_nonzero(singular > RANK_TOL * singular[0]))
    basis = u[:, :rank]
    products = dose[:, None] * centered_markers
    values = np.column_stack([y, products])
    partial = values - basis @ (basis.T @ values)
    return JointFold(rows.copy(), partial[:, 0], partial[:, 1:], np.linalg.norm(products, axis=0),
                     fiber.copy(), markers.copy(), rank)


def _households(parts, groups):
    groups = np.asarray(groups)
    rows = np.concatenate([part.rows for part in parts])
    if groups.ndim != 1 or np.any(rows < 0) or np.any(rows >= len(groups)) or np.unique(rows).size != len(rows):
        raise ValueError("JOINT_ASSOCIATION_ROW_ALIGNMENT")
    labels = groups[rows]
    if any(label is None or not str(label).strip() or
           (isinstance(label, (float, np.floating)) and not np.isfinite(label)) for label in labels):
        raise ValueError("MISSING_JOINT_ASSOCIATION_HOUSEHOLD")
    unique, group = np.unique(labels.astype(str), return_inverse=True)
    seen = set()
    offset = 0
    for part in parts:
        end = offset + len(part.rows)
        members = set(group[offset:end])
        if seen & members:
            raise ValueError("JOINT_ASSOCIATION_HOUSEHOLD_CROSSES_FOLDS")
        seen.update(members)
        offset = end
    return group, len(unique)


def _correlation(two_columns):
    centered = two_columns - two_columns.mean(axis=0)
    norms = np.linalg.norm(centered, axis=0)
    if np.any(norms == 0):
        return None
    unit = centered / norms
    return float(np.clip(unit[:, 0] @ unit[:, 1], -1, 1))


def _tail_support(fiber, mask):
    values = fiber[mask]
    q05, q95 = np.quantile(values, (0.05, 0.95))
    return {
        "n": len(values),
        "q05": float(q05),
        "q95": float(q95),
        "covers_10_to_15": bool(q05 <= LOW_DOSE and HIGH_DOSE <= q95),
        "near_10": int(np.count_nonzero(np.abs(values - LOW_DOSE) <= 1)),
        "near_15": int(np.count_nonzero(np.abs(values - HIGH_DOSE) <= 1)),
    }


def _contrast_support(fiber, values):
    q25, q75 = np.quantile(values, (0.25, 0.75))
    low = _tail_support(fiber, values <= q25)
    high = _tail_support(fiber, values >= q75)
    return {
        "q25": float(q25),
        "q75": float(q75),
        "support": {
            "low": low,
            "high": high,
            "both_tails_cover_10_to_15": low["covers_10_to_15"] and high["covers_10_to_15"],
        },
    }


def _interval(estimate, se, critical):
    radius = se * critical
    return [float(estimate - radius), float(estimate + radius)]


def summarize_joint(parts, groups):
    """Report conditional marker contrasts from one joint, household-clustered fit.

    The empirical marker quartiles and fitted fold feature bases are fixed for
    inference. Tail summaries describe marginal dose support for each marker.
    """
    parts = tuple(parts)
    if not parts:
        raise ValueError("EMPTY_JOINT_ASSOCIATION_FOLDS")
    group, g = _households(parts, groups)
    y = np.concatenate([part.partial_y for part in parts])
    products = np.concatenate([part.partial_products for part in parts])
    fiber = np.concatenate([part.fiber for part in parts])
    markers = np.concatenate([part.markers for part in parts])
    n = len(y)
    nuisance_rank = sum(part.nuisance_rank for part in parts)
    # Scale by the unprojected interaction norms so products almost entirely
    # absorbed by nuisance columns do not gain rank by normalizing residuals.
    norms = np.sqrt(np.sum([part.product_norms ** 2 for part in parts], axis=0))
    scale = np.where(norms > 0, norms, 1.0)
    u, singular, vh = np.linalg.svd(products / scale, full_matrices=False)
    rank = int(np.count_nonzero(singular > RANK_TOL))
    df_residual = n - nuisance_rank - rank
    output = {name: _contrast_support(fiber, markers[:, j]) for j, name in enumerate(MARKERS)}
    result = {
        "status": "unavailable",
        "n": n,
        "households": g,
        "nuisance_rank": nuisance_rank,
        "interaction_rank": rank,
        "df_residual": df_residual,
        "contrast_g_per_1000_kcal": [LOW_DOSE, HIGH_DOSE],
        "marker_correlation": _correlation(markers),
        "partial_product_correlation": _correlation(products),
        "scaled_interaction_condition_number": float(singular[0] / singular[-1]) if rank == len(MARKERS) else None,
        "markers": output,
    }
    if rank != len(MARKERS) or g < 2 or df_residual <= 0:
        reason = ("Joint interaction columns are rank deficient" if rank != len(MARKERS)
                  else "Insufficient household or residual degrees of freedom")
        for entry in output.values():
            entry.update(status="unavailable", reason=reason)
        return dict(result, reason=reason)

    weights = ((u / singular) @ vh) / scale
    coef = weights.T @ y
    residual = y - products @ coef
    household_influence = np.zeros((g, len(MARKERS)))
    np.add.at(household_influence, group, weights * residual[:, None])
    cr1 = (g / (g - 1)) * ((n - 1) / df_residual)
    critical = float(stats.t.ppf(0.975, g - 1))
    family_critical = float(stats.t.ppf(1 - 0.05 / (2 * len(MARKERS)), g - 1))
    for j, name in enumerate(MARKERS):
        entry = output[name]
        iqr = entry["q75"] - entry["q25"]
        estimate = coef[j] * iqr
        se = np.sqrt(cr1) * np.linalg.norm(household_influence[:, j]) * iqr
        if iqr <= 0:
            entry.update(status="unavailable", reason="Zero marker IQR")
        elif not np.isfinite(estimate) or not np.isfinite(se) or se <= 0:
            entry.update(status="unavailable", reason="Degenerate household score variance")
        else:
            entry.update(
                status="ok",
                estimate_mg_dl=float(estimate),
                se_mg_dl=float(se),
                ci95_mg_dl=_interval(estimate, se, critical),
                family95_mg_dl=_interval(estimate, se, family_critical),
            )
    if any(entry["status"] == "ok" for entry in output.values()):
        result["status"] = "ok"
    else:
        result["reason"] = "No conditional marker contrast is estimable"
    return result


def joint_selftest():
    rng = np.random.default_rng(2405)
    households = 96
    group = np.repeat(np.arange(households), 2)
    folds = group % 3
    n = len(group)
    fiber = rng.uniform(7, 19, n)
    dose = (fiber - LOW_DOSE) / (HIGH_DOSE - LOW_DOSE)
    markers = rng.normal(size=(n, len(MARKERS)))
    markers[:, 1] += 0.7 * markers[:, 0]
    background = np.column_stack([np.ones(n), dose ** 2, rng.normal(size=n)])
    nuisance = np.column_stack([background, dose, markers, markers ** 2])
    supplied = np.column_stack([background, dose, np.zeros(n)])
    products = dose[:, None] * markers
    planted = np.array([0.4, -0.9])
    noise = rng.normal(0, 0.02, households)[group] + rng.normal(0, 0.015, n)
    y = 89 + folds + nuisance[:, 1:].sum(1) + products @ planted + noise
    parts = [
        joint_fold(rows, y[rows], supplied[rows], fiber[rows], markers[rows])
        for rows in (np.flatnonzero(folds == f)[::-1] for f in range(3))
    ]
    got = summarize_joint(parts, group)
    assert got["status"] == "ok" and got["interaction_rank"] == 2
    assert got["nuisance_rank"] == 3 * nuisance.shape[1]

    # Check the partial model against a direct, full-rank stacked regression.
    full = np.column_stack([nuisance * (folds == f)[:, None] for f in range(3)] + [products])
    coef = np.linalg.lstsq(full, y, rcond=None)[0]
    bread = np.linalg.inv(full.T @ full)
    scores = np.zeros((households, full.shape[1]))
    np.add.at(scores, group, full * (y - full @ coef)[:, None])
    cr1 = households / (households - 1) * (n - 1) / (n - full.shape[1])
    critical = stats.t.ppf(0.975, households - 1)
    for j, name in enumerate(MARKERS):
        iqr = np.diff(np.quantile(markers[:, j], (0.25, 0.75)))[0]
        contrast = np.eye(full.shape[1])[full.shape[1] - len(MARKERS) + j] * iqr
        effect = float(contrast @ coef)
        se = np.sqrt(cr1) * np.linalg.norm(scores @ bread @ contrast)
        observed = got["markers"][name]
        np.testing.assert_allclose(observed["estimate_mg_dl"], effect, atol=1e-10)
        np.testing.assert_allclose(observed["estimate_mg_dl"], planted[j] * iqr, atol=0.035)
        np.testing.assert_allclose(observed["ci95_mg_dl"], _interval(effect, se, critical), atol=1e-10)

    duplicate_markers = np.column_stack([markers[:, 0], markers[:, 0] * 2 + 3])
    collinear = joint_fold(np.arange(n), y, supplied, fiber, duplicate_markers)
    assert summarize_joint([collinear], group)["interaction_rank"] == 1
    one_fold = joint_fold(np.arange(n), y, supplied, fiber, markers)
    assert summarize_joint([one_fold], np.zeros(n))["status"] == "unavailable"
    crossed = group.copy()
    crossed[parts[0].rows[0]] = group[parts[1].rows[0]]
    try:
        summarize_joint(parts, crossed)
    except ValueError as exc:
        assert str(exc) == "JOINT_ASSOCIATION_HOUSEHOLD_CROSSES_FOLDS"
    else:
        raise AssertionError("A household that crosses folds must be rejected")
