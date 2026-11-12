"""Descriptive gut-Shannon association sensitivity at a fixed fiber hinge."""

from dataclasses import dataclass

import numpy as np
from scipy import stats

from marker_effects import HIGH_DOSE, LOW_DOSE, _tail_support, nuisance_basis


HINGE = 12.5
DOSE_STEP = HIGH_DOSE - LOW_DOSE


@dataclass(frozen=True)
class DoseFold:
    rows: np.ndarray
    partial_y: np.ndarray
    partial_targets: np.ndarray
    target_ss: np.ndarray
    marker: np.ndarray
    fiber: np.ndarray
    nuisance_rank: int


def _dose_terms(fiber):
    return np.column_stack(((fiber - LOW_DOSE) / DOSE_STEP,
                            np.maximum(fiber - HINGE, 0) / DOSE_STEP))


def dose_fold(rows, y, background, fiber, gut_shannon):
    """Use the held-fold covariates to define the fixed nuisance projection."""
    rows = np.asarray(rows)
    y = np.asarray(y, float)
    background = np.asarray(background, float)
    fiber = np.asarray(fiber, float)
    marker = np.asarray(gut_shannon, float)
    n = len(y) if y.ndim == 1 else 0
    if (
        n == 0
        or rows.shape != (n,)
        or not np.issubdtype(rows.dtype, np.integer)
        or np.any(rows < 0)
        or np.unique(rows).size != n
        or fiber.shape != (n,)
        or marker.shape != (n,)
        or background.ndim != 2
        or background.shape[0] != n
        or background.shape[1] == 0
        or not all(np.isfinite(a).all() for a in (y, fiber, marker, background))
    ):
        raise ValueError("INVALID_DOSE_SENSITIVITY_FOLD")

    # Keep the matched main effects even when the supplied background
    # already includes a curved dose and microbiome main effects.
    dose = _dose_terms(fiber)
    basis = nuisance_basis(np.column_stack([background, dose, marker]))
    targets = dose * marker[:, None]
    values = np.column_stack([y, targets])
    partial = values - basis @ (basis.T @ values)
    return DoseFold(rows.copy(), partial[:, 0], partial[:, 1:],
                    np.sum(targets ** 2, axis=0), marker.copy(), fiber.copy(), basis.shape[1])


def summarize_dose(parts, groups):
    """Pool fixed interaction coefficients and form pointwise household CR1 intervals."""
    parts = tuple(parts)
    groups = np.asarray(groups)
    if not parts or groups.ndim != 1:
        raise ValueError("INVALID_DOSE_SENSITIVITY_GROUPS")
    rows = np.concatenate([part.rows for part in parts])
    n = len(rows)
    if np.any(rows < 0) or np.any(rows >= len(groups)) or np.unique(rows).size != n:
        raise ValueError("DOSE_SENSITIVITY_ROW_ALIGNMENT")

    labels = groups[rows]
    if any(
        label is None
        or not str(label).strip()
        or isinstance(label, (float, np.floating)) and not np.isfinite(label)
        for label in labels
    ):
        raise ValueError("MISSING_DOSE_SENSITIVITY_HOUSEHOLD")
    unique, group = np.unique(labels.astype(str), return_inverse=True)
    households = len(unique)
    seen = set()
    offset = 0
    for part in parts:
        end = offset + len(part.rows)
        members = set(group[offset:end])
        if members & seen:
            raise ValueError("DOSE_SENSITIVITY_HOUSEHOLD_CROSSES_FOLDS")
        seen.update(members)
        offset = end

    x = np.concatenate([part.partial_targets for part in parts])
    y = np.concatenate([part.partial_y for part in parts])
    marker = np.concatenate([part.marker for part in parts])
    fiber = np.concatenate([part.fiber for part in parts])
    q25, q75 = np.quantile(marker, (0.25, 0.75))
    iqr = q75 - q25
    support = {}
    for label, mask in (("low", marker <= q25), ("high", marker >= q75)):
        support[label] = dict(
            _tail_support(fiber, mask),
            near_hinge=int(np.count_nonzero(np.abs(fiber[mask] - HINGE) <= 1)),
        )
    support["both_tails_cover_10_to_15"] = all(
        support[label]["covers_10_to_15"] for label in ("low", "high")
    )

    # Scale by the raw target norms. Nearly absorbed columns should not
    # acquire apparent rank merely because their residual norms are tiny.
    scale = np.maximum(np.sqrt(np.sum([part.target_ss for part in parts], axis=0)), 1)
    u, singular, vh = np.linalg.svd(x / scale, full_matrices=False)
    rank = int(np.sum(singular > 1e-10))
    nuisance_rank = sum(part.nuisance_rank for part in parts)
    df_residual = n - nuisance_rank - rank
    result = {
        "status": "unavailable",
        "n": n,
        "households": households,
        "nuisance_rank": nuisance_rank,
        "interaction_rank": rank,
        "df_residual": df_residual,
        "hinge_g_per_1000_kcal": HINGE,
        "q25": float(q25),
        "q75": float(q75),
        "support": support,
    }
    if iqr <= 0:
        return dict(result, reason="Zero marker IQR")
    if rank != 2:
        return dict(result, reason="Insufficient interaction rank")
    if households < 2 or df_residual <= 0:
        return dict(result, reason="Insufficient household or residual degrees of freedom")

    # The transpose of the pseudoinverse gives each row's influence on
    # the two unscaled coefficients. Aggregate residual scores by household.
    influence = ((u / singular) @ vh) / scale
    coef = influence.T @ y
    residual = y - x @ coef
    cluster_scores = np.zeros((households, 2))
    np.add.at(cluster_scores, group, influence * residual[:, None])
    cr1 = (households / (households - 1)) * ((n - 1) / df_residual)
    critical = float(stats.t.ppf(0.975, households - 1))
    intervals = (
        ("full", LOW_DOSE, HIGH_DOSE),
        ("below", LOW_DOSE, HINGE),
        ("above", HINGE, HIGH_DOSE),
    )
    terms = {name: np.diff(_dose_terms(np.array([low, high])), axis=0)[0]
             for name, low, high in intervals}
    terms["above_minus_below"] = terms["above"] - terms["below"]
    contrasts = {}
    for name, term in terms.items():
        contrast = iqr * term
        estimate = float(contrast @ coef)
        se = float(np.sqrt(cr1) * np.linalg.norm(cluster_scores @ contrast))
        if not np.isfinite(estimate) or not np.isfinite(se) or se <= 0:
            return dict(result, reason="Degenerate household score variance")
        contrasts[name] = {
            "estimate_mg_dl": estimate,
            "se_mg_dl": se,
            "ci95_mg_dl": [estimate - critical * se, estimate + critical * se],
        }
    for name, low, high in intervals:
        contrasts[name]["dose_g_per_1000_kcal"] = [low, high]
    return dict(result, status="ok",
                coefficients={"linear_per_5": float(coef[0]), "hinge_per_5": float(coef[1])},
                contrasts=contrasts)


def dose_selftest():
    rng = np.random.default_rng(623)
    households = 96
    group = np.repeat(np.arange(households), 2)
    n = len(group)
    folds = group % 3
    fiber = rng.uniform(7, 19, n)
    marker = rng.uniform(1.2, 3.9, n)
    dose = _dose_terms(fiber)
    background = np.column_stack([np.ones(n), rng.normal(size=n), dose[:, 0] ** 2])
    nuisance = np.column_stack([background, dose, marker])
    supplied = np.column_stack([background, dose[:, 0], np.zeros(n)])
    products = dose * marker[:, None]
    planted = np.array([2.2, -3.0])
    noise = rng.normal(0, 0.025, households)[group] + rng.normal(0, 0.015, n)
    y = 89 + folds + nuisance[:, 1:].sum(1) + products @ planted + noise
    parts = [
        dose_fold(rows, y[rows], supplied[rows], fiber[rows], marker[rows])
        for rows in (np.flatnonzero(folds == f)[::-1] for f in range(3))
    ]
    got = summarize_dose(parts, group)
    assert got["status"] == "ok"
    assert got["nuisance_rank"] == 3 * nuisance.shape[1]
    assert got["interaction_rank"] == 2
    assert got["support"]["low"]["near_hinge"] == np.count_nonzero(
        (marker <= np.quantile(marker, 0.25)) & (np.abs(fiber - HINGE) <= 1)
    )
    np.testing.assert_allclose(list(got["coefficients"].values()), planted, atol=0.07)

    # Independently check against full stacked least squares and sandwich scores.
    full = np.column_stack([nuisance * (folds == f)[:, None] for f in range(3)] + [products])
    coef = np.linalg.lstsq(full, y, rcond=None)[0]
    bread = np.linalg.inv(full.T @ full)
    score = np.zeros((households, full.shape[1]))
    np.add.at(score, group, full * (y - full @ coef)[:, None])
    cr1 = households / (households - 1) * (n - 1) / (n - full.shape[1])
    iqr = np.diff(np.quantile(marker, (0.25, 0.75)))[0]
    for name, multiplier in (("full", [1, 0.5]), ("below", [0.5, 0]), ("above", [0.5, 0.5]),
                             ("above_minus_below", [0, 0.5])):
        contrast = np.r_[np.zeros(full.shape[1] - 2), iqr * np.asarray(multiplier)]
        estimate = float(contrast @ coef)
        se = float(np.sqrt(cr1) * np.linalg.norm(score @ bread @ contrast))
        interval = estimate + stats.t.ppf([0.025, 0.975], households - 1) * se
        np.testing.assert_allclose(got["contrasts"][name]["estimate_mg_dl"], estimate, atol=1e-10)
        np.testing.assert_allclose(got["contrasts"][name]["ci95_mg_dl"], interval, atol=1e-10)

    rank_one = dose_fold(np.arange(n), y, supplied, np.minimum(fiber, HINGE - 0.25), marker)
    assert summarize_dose([rank_one], group)["reason"] == "Insufficient interaction rank"
    one_fold = dose_fold(np.arange(n), y, supplied, fiber, marker)
    assert summarize_dose([one_fold], np.zeros(n))["reason"] == "Insufficient household or residual degrees of freedom"
