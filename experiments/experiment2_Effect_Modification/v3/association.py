"""Paired, fixed-marker fiber moderation contrasts in the discovery cohort."""

from dataclasses import dataclass

import numpy as np
from scipy import stats


LOW_DOSE, HIGH_DOSE = 10.0, 15.0
MARKERS = ("gut_shannon", "fermentation")
RANK_TOL = 1e-10


@dataclass(frozen=True)
class FoldEffect:
    rows: np.ndarray
    partial_y: np.ndarray
    partial_products: np.ndarray
    raw_product_norms: np.ndarray
    fiber: np.ndarray
    markers: np.ndarray
    nuisance_rank: int


def fold_effect(rows, y, background, fiber, markers):
    """Project two fixed interactions off a shared background with both marker main effects."""
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
        raise ValueError("INVALID_ASSOCIATION_FOLD")

    dose = (fiber - LOW_DOSE) / (HIGH_DOSE - LOW_DOSE)
    # Add the shared linear dose and both marker main effects. SVD removes
    # exact dependencies when the caller already supplies these columns.
    centered_markers = markers - markers.mean(axis=0)
    design = np.column_stack([background, np.ones(n), dose, centered_markers, centered_markers ** 2])
    scale = np.max(np.abs(design), axis=0)
    scale[scale == 0] = 1
    u, singular, _ = np.linalg.svd(design / scale, full_matrices=False)
    rank = int(np.count_nonzero(singular > RANK_TOL * singular[0]))
    basis = u[:, :rank]
    products = dose[:, None] * markers
    values = np.column_stack([y, products])
    partial = values - basis @ (basis.T @ values)

    return FoldEffect(rows.copy(), partial[:, 0], partial[:, 1:], np.linalg.norm(products, axis=0),
                      fiber.copy(), markers.copy(), rank)


def _support(fiber, selected):
    values = fiber[selected]
    q05, q95 = np.quantile(values, (0.05, 0.95))
    return {
        "n": len(values),
        "q05": float(q05),
        "q95": float(q95),
        "near_10": int(np.count_nonzero(np.abs(values - LOW_DOSE) <= 1)),
        "near_15": int(np.count_nonzero(np.abs(values - HIGH_DOSE) <= 1)),
        "covers_10_to_15": bool(q05 <= LOW_DOSE and HIGH_DOSE <= q95),
    }


def _households(parts, groups):
    groups = np.asarray(groups)
    rows = np.concatenate([part.rows for part in parts])
    if groups.ndim != 1 or np.any(rows < 0) or np.any(rows >= len(groups)) or np.unique(rows).size != len(rows):
        raise ValueError("ASSOCIATION_ROW_ALIGNMENT")
    labels = groups[rows]
    if any(
        label is None
        or not str(label).strip()
        or isinstance(label, (float, np.floating)) and not np.isfinite(label)
        for label in labels
    ):
        raise ValueError("MISSING_ASSOCIATION_HOUSEHOLD")
    unique, household = np.unique(labels.astype(str), return_inverse=True)
    seen = set()
    offset = 0
    for part in parts:
        end = offset + len(part.rows)
        members = set(household[offset:end])
        if seen & members:
            raise ValueError("ASSOCIATION_HOUSEHOLD_CROSSES_FOLDS")
        seen.update(members)
        offset = end
    return household, len(unique)


def _interval(estimate, influence, critical):
    se = float(np.linalg.norm(influence))
    if not np.isfinite(estimate) or not np.isfinite(se) or se <= 0:
        return {"status": "unavailable", "reason": "Degenerate household score variance"}
    half_width = critical * se
    return {
        "status": "ok",
        "estimate_mg_dl": float(estimate),
        "se_mg_dl": se,
        "ci95_mg_dl": [float(estimate - half_width), float(estimate + half_width)],
    }


def _absolute_bounds(interval):
    a, b = interval
    lower = 0.0 if a <= 0 <= b else min(abs(a), abs(b))
    return lower, max(abs(a), abs(b))


def summarize_effects(parts, groups):
    """Compare two separate regressions using paired household CR1 influence.

    The common interaction coefficient in each model is pooled across folds.
    Nuisance coefficients are fold-specific. Intervals condition on the held
    nuisance design and pooled marker quartiles. The signed difference uses
    aligned household influence. A separate magnitude difference maps
    simultaneous Bonferroni intervals through the absolute-value contrast.
    """
    parts = tuple(parts)
    if not parts:
        raise ValueError("EMPTY_ASSOCIATION_FOLDS")
    household, g = _households(parts, groups)
    y = np.concatenate([part.partial_y for part in parts])
    products = np.concatenate([part.partial_products for part in parts])
    fiber = np.concatenate([part.fiber for part in parts])
    markers = np.concatenate([part.markers for part in parts])
    raw_norms = np.sqrt(np.sum([part.raw_product_norms ** 2 for part in parts], axis=0))
    n = len(y)
    nuisance_rank = sum(part.nuisance_rank for part in parts)
    df_residual = n - nuisance_rank - 1
    enough = g > 1 and df_residual > 0
    cr1_root = np.sqrt(g / (g - 1) * (n - 1) / df_residual) if enough else np.nan
    critical = float(stats.t.ppf(0.975, g - 1)) if enough else np.nan

    family_critical = float(stats.t.ppf(1 - 0.05 / (2 * len(MARKERS)), g - 1)) if enough else np.nan

    output, influences = {}, {}
    for j, name in enumerate(MARKERS):
        marker, product = markers[:, j], products[:, j]
        q25, q75 = np.quantile(marker, (0.25, 0.75))
        iqr = q75 - q25
        low, high = _support(fiber, marker <= q25), _support(fiber, marker >= q75)
        result = {
            "q25": float(q25),
            "q75": float(q75),
            "support": {
                "low": low,
                "high": high,
                "both_tails_cover_10_to_15": low["covers_10_to_15"] and high["covers_10_to_15"],
            },
        }
        if not enough:
            result.update(status="unavailable", reason="Insufficient household or residual degrees of freedom")
        elif iqr <= 0:
            result.update(status="unavailable", reason="Zero marker IQR")
        elif np.linalg.norm(product) <= 1e-8 * max(raw_norms[j], np.finfo(float).tiny):
            result.update(status="unavailable", reason="Interaction absorbed by nuisance basis")
        else:
            weight = product / np.dot(product, product)
            beta = float(weight @ y)
            residual = y - product * beta
            influence = cr1_root * np.bincount(household, weights=weight * residual)
            influences[name] = influence * iqr
            result.update(_interval(beta * iqr, influences[name], critical))
            if result["status"] == "ok":
                center = result["estimate_mg_dl"]
                radius = family_critical * result["se_mg_dl"]
                result["family95_mg_dl"] = [center - radius, center + radius]
        output[name] = result

    shannon, fermentation = (output[name] for name in MARKERS)
    difference = absolute_difference = {"status": "unavailable", "reason": "Both marker contrasts must be estimable"}
    if all(item["status"] == "ok" for item in (shannon, fermentation)):
        difference = _interval(
            fermentation["estimate_mg_dl"] - shannon["estimate_mg_dl"],
            influences["fermentation"] - influences["gut_shannon"],
            critical,
        )
        s_low, s_high = _absolute_bounds(shannon["family95_mg_dl"])
        f_low, f_high = _absolute_bounds(fermentation["family95_mg_dl"])
        absolute_difference = {
            "status": "ok",
            "estimate_mg_dl": abs(fermentation["estimate_mg_dl"]) - abs(shannon["estimate_mg_dl"]),
            "ci95_mg_dl": [f_low - s_high, f_high - s_low],
        }
    return {
        "n": n,
        "households": g,
        "nuisance_rank": nuisance_rank,
        "df_residual": df_residual,
        "contrast_g_per_1000_kcal": [LOW_DOSE, HIGH_DOSE],
        "markers": output,
        "difference": difference,
        "absolute_difference": absolute_difference,
    }


def association_selftest():
    rng = np.random.default_rng(131)
    g = 90
    household = np.repeat(np.arange(g), 2)
    folds = household % 3
    n = len(folds)
    dose = rng.uniform(-0.6, 1.6, n)
    markers = rng.normal(size=(n, len(MARKERS)))
    markers[:, 1] += 0.4 * markers[:, 0]
    fiber = LOW_DOSE + (HIGH_DOSE - LOW_DOSE) * dose
    nuisance = np.column_stack([np.ones(n), dose, dose ** 2, markers, markers ** 2])
    background = np.column_stack([dose, dose ** 2, np.ones(n), np.zeros(n)])
    y = 90 + folds + nuisance[:, 1:].sum(axis=1) + dose * (1.2 * markers[:, 0] + 2.4 * markers[:, 1])
    y += rng.normal(0, 0.04, g)[household] + rng.normal(0, 0.02, n)
    parts = [
        fold_effect(rows, y[rows], background[rows], fiber[rows], markers[rows])
        for rows in (np.flatnonzero(folds == fold)[::-1] for fold in range(3))
    ]
    got = summarize_effects(parts, household)
    assert got["nuisance_rank"] == 3 * nuisance.shape[1]
    critical = stats.t.ppf(0.975, g - 1)
    influence = []
    for j, name in enumerate(MARKERS):
        stacked = np.column_stack([nuisance * (folds == k)[:, None] for k in range(3)]
                                  + [dose * markers[:, j]])
        coef = np.linalg.lstsq(stacked, y, rcond=None)[0]
        bread = np.linalg.inv(stacked.T @ stacked)
        scores = np.zeros((g, stacked.shape[1]))
        np.add.at(scores, household, stacked * (y - stacked @ coef)[:, None])
        correction = np.sqrt(g / (g - 1) * (n - 1) / (n - stacked.shape[1]))
        q25, q75 = np.quantile(markers[:, j], (0.25, 0.75))
        contrast = np.r_[np.zeros(stacked.shape[1] - 1), q75 - q25]
        route = got["markers"][name]
        se = np.linalg.norm(correction * scores @ bread @ contrast)
        point = contrast @ coef
        np.testing.assert_allclose(route["estimate_mg_dl"], point, atol=1e-10)
        np.testing.assert_allclose(route["ci95_mg_dl"], point + critical * se * np.array([-1, 1]), atol=1e-10)
        family_critical = stats.t.ppf(0.9875, g - 1)
        np.testing.assert_allclose(route["family95_mg_dl"],
                                   point + family_critical * se * np.array([-1, 1]), atol=1e-10)
        influence.append(correction * scores @ bread @ contrast)

    shannon, fermentation = (got["markers"][key]["estimate_mg_dl"] for key in MARKERS)
    assert fermentation > shannon
    np.testing.assert_allclose(got["difference"]["estimate_mg_dl"], fermentation - shannon, atol=1e-10)
    np.testing.assert_allclose(got["difference"]["se_mg_dl"], np.linalg.norm(influence[1] - influence[0]), atol=1e-10)
    absolute = got["absolute_difference"]
    np.testing.assert_allclose(absolute["estimate_mg_dl"], abs(fermentation) - abs(shannon), atol=1e-10)
    assert absolute["ci95_mg_dl"][0] < absolute["estimate_mg_dl"] < absolute["ci95_mg_dl"][1]
    assert _absolute_bounds([-3, 2]) == (0, 3) and _absolute_bounds([-3, -2]) == (2, 3)
    reversed_marker = markers * [1, -1]
    reversed_parts = [
        fold_effect(rows, y[rows], background[rows], fiber[rows], reversed_marker[rows])
        for rows in (np.flatnonzero(folds == fold) for fold in range(3))
    ]
    reversed_result = summarize_effects(reversed_parts, household)
    np.testing.assert_allclose(reversed_result["absolute_difference"]["ci95_mg_dl"],
                               absolute["ci95_mg_dl"], atol=1e-10)
    assert reversed_result["difference"]["estimate_mg_dl"] < 0
    collapsed = np.column_stack([markers[:, 0], np.ones(n)])
    single = fold_effect(np.arange(n), y, background, fiber, collapsed)
    assert summarize_effects([single], household)["markers"]["fermentation"]["reason"] == "Zero marker IQR"
    crossed = household.copy()
    crossed[parts[0].rows[0]] = household[parts[1].rows[0]]
    try:
        summarize_effects(parts, crossed)
    except ValueError as exc:
        assert str(exc) == "ASSOCIATION_HOUSEHOLD_CROSSES_FOLDS"
    else:
        raise AssertionError("A household that crosses folds must be rejected")
