"""Fixed-marker person-level association contrasts, separate from prediction.

Each marker gets its own partial regression with fold-specific nuisance
coefficients and a common interaction coefficient in raw marker units.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


MARKERS = ("oral_shannon", "oral_dominance", "gut_shannon", "gut_dominance")
LOW_DOSE = 10.0
HIGH_DOSE = 15.0


@dataclass(frozen=True)
class FoldEffect:
    rows: np.ndarray
    partial_y: np.ndarray
    partial_targets: np.ndarray
    fiber: np.ndarray
    markers: np.ndarray
    nuisance_rank: int
    partial_raw_y: np.ndarray | None = None


def nuisance_basis(background):
    """Return an orthonormal basis for a finite design with an intercept."""
    background = np.asarray(background, dtype=float)
    if background.ndim != 2 or not all(background.shape) or not np.isfinite(background).all():
        raise ValueError("INVALID_NUISANCE_BACKGROUND")
    # Scale columns before measuring rank so a change in units preserves the basis.
    scale = np.max(np.abs(background), axis=0)
    keep = scale > 0
    basis_input = background[:, keep] / scale[keep]
    u, singular, _ = np.linalg.svd(basis_input, full_matrices=False)
    rank = int(np.sum(singular > 1e-10 * singular[0])) if singular.size else 0
    basis = u[:, :rank]
    intercept = np.ones(len(background))
    if np.linalg.norm(intercept - basis @ (basis.T @ intercept)) > 1e-8 * np.sqrt(len(background)):
        raise ValueError("FIXED_MARKER_BACKGROUND_NEEDS_INTERCEPT")
    return basis


def partial_fold(rows, y, background, fiber, markers, raw_y=None):
    """Project outcomes and all fixed targets off the same held-role basis."""
    rows = np.asarray(rows)
    y = np.asarray(y, float)
    background = np.asarray(background, float)
    fiber = np.asarray(fiber, float)
    markers = np.asarray(markers, float)
    n = rows.size
    if (
        rows.shape != (n,)
        or n == 0
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
        raise ValueError("INVALID_FIXED_MARKER_FOLD")

    basis = nuisance_basis(background)

    dose = (fiber - LOW_DOSE) / (HIGH_DOSE - LOW_DOSE)
    targets = dose[:, None] * markers
    columns = [y, targets]
    if raw_y is not None:
        raw_y = np.asarray(raw_y, dtype=float)
        if raw_y.shape != (n,) or not np.isfinite(raw_y).all():
            raise ValueError("INVALID_RAW_OUTCOME_SENSITIVITY")
        columns.append(raw_y)
    values = np.column_stack(columns)
    residuals = values - basis @ (basis.T @ values)
    partial_raw_y = residuals[:, -1] if raw_y is not None else None
    return FoldEffect(rows.copy(), residuals[:, 0], residuals[:, 1:1 + len(MARKERS)],
                      fiber.copy(), markers.copy(), basis.shape[1], partial_raw_y)


def _tail_support(fiber, mask):
    q05, q95 = np.quantile(fiber[mask], (0.05, 0.95))
    return {
        "n": int(mask.sum()),
        "q05": float(q05),
        "q95": float(q95),
        "covers_10_to_15": bool(q05 <= LOW_DOSE and HIGH_DOSE <= q95),
        "near_10": int(np.count_nonzero(np.abs(fiber[mask] - LOW_DOSE) <= 1)),
        "near_15": int(np.count_nonzero(np.abs(fiber[mask] - HIGH_DOSE) <= 1)),
    }


def _estimate(y, x, group, households, df_resid, iqr, critical, family_critical):
    denominator = np.dot(x, x)
    coefficient = np.dot(x, y) / denominator
    scores = np.bincount(group, weights=x * (y - x * coefficient))
    correction = (households / (households - 1)) * ((len(y) - 1) / df_resid)
    variance = correction * np.dot(scores, scores) / denominator ** 2
    if not np.isfinite(variance) or variance <= 0 or not np.isfinite(coefficient):
        return {"status": "unavailable", "reason": "Degenerate household score variance"}

    estimate = float(coefficient * iqr)
    se = float(np.sqrt(variance) * iqr)
    return {
        "status": "ok",
        "estimate_mg_dl": estimate,
        "se_mg_dl": se,
        "ci95_mg_dl": [estimate - critical * se, estimate + critical * se],
        "family95_mg_dl": [estimate - family_critical * se, estimate + family_critical * se],
    }


def summarize_effects(parts, groups, *, raw=False):
    """Pool separate fixed-marker regressions and form household CR1 intervals.

    Percentiles use the pooled raw marker values in the evaluated people. The
    interval conditions on those percentiles and the fixed nuisance bases.
    """
    parts = tuple(parts)
    groups = np.asarray(groups)
    if not parts or groups.ndim != 1:
        raise ValueError("INVALID_FIXED_MARKER_GROUPS")
    rows = np.concatenate([part.rows for part in parts])
    n = len(rows)
    if np.any(rows < 0) or np.any(rows >= len(groups)) or np.unique(rows).size != n:
        raise ValueError("FIXED_MARKER_ROW_ALIGNMENT")

    labels = groups[rows]
    if any(
        label is None
        or not str(label).strip()
        or isinstance(label, (float, np.floating)) and not np.isfinite(label)
        for label in labels
    ):
        raise ValueError("MISSING_FIXED_MARKER_HOUSEHOLD")
    _, group = np.unique(labels.astype(str), return_inverse=True)
    households = int(group.max() + 1)
    seen = set()
    offset = 0
    for part in parts:
        member = set(group[offset:offset + len(part.rows)])
        if seen & member:
            raise ValueError("FIXED_MARKER_HOUSEHOLD_CROSSES_FOLDS")
        seen.update(member)
        offset += len(part.rows)

    outcome = [part.partial_raw_y if raw else part.partial_y for part in parts]
    if any(value is None for value in outcome):
        raise ValueError("RAW_OUTCOME_SENSITIVITY_MISSING")
    y = np.concatenate(outcome)
    x = np.concatenate([part.partial_targets for part in parts])
    fiber = np.concatenate([part.fiber for part in parts])
    markers = np.concatenate([part.markers for part in parts])
    nuisance_rank = sum(part.nuisance_rank for part in parts)
    df_resid = n - nuisance_rank - 1
    critical = float(stats.t.ppf(0.975, households - 1))
    family_critical = float(stats.t.ppf(1 - 0.05 / (2 * len(MARKERS)), households - 1))
    targets = ((fiber - LOW_DOSE) / (HIGH_DOSE - LOW_DOSE))[:, None] * markers
    output = {}
    for j, name in enumerate(MARKERS):
        marker = markers[:, j]
        q25, q75 = np.quantile(marker, (0.25, 0.75))
        iqr = q75 - q25
        low = _tail_support(fiber, marker <= q25)
        high = _tail_support(fiber, marker >= q75)
        result = {
            "q25": float(q25),
            "q75": float(q75),
            "support": {
                "low": low,
                "high": high,
                "both_tails_cover_10_to_15": low["covers_10_to_15"] and high["covers_10_to_15"],
            },
        }
        if iqr <= 0:
            result.update(status="unavailable", reason="Zero marker IQR")
        elif households < 2 or df_resid <= 0:
            result.update(status="unavailable", reason="Insufficient household or residual degrees of freedom")
        elif np.linalg.norm(x[:, j]) <= 1e-8 * max(np.linalg.norm(targets[:, j]), 1):
            result.update(status="unavailable", reason="Interaction absorbed by nuisance basis")
        else:
            result.update(_estimate(y, x[:, j], group, households, df_resid, iqr,
                                    critical, family_critical))
        output[name] = result
    return {
        "n": n,
        "households": households,
        "nuisance_rank": nuisance_rank,
        "df_residual": df_resid,
        "contrast_g_per_1000_kcal": [LOW_DOSE, HIGH_DOSE],
        "markers": output,
    }


def effect_selftest():
    rng = np.random.default_rng(615)
    households = 96
    group = np.repeat(np.arange(households), 2)
    n = len(group)
    fiber = rng.uniform(7.0, 18.0, n)
    dose = (fiber - LOW_DOSE) / (HIGH_DOSE - LOW_DOSE)
    markers = rng.uniform(0.3, 0.9, (n, len(MARKERS)))
    covariate = rng.normal(size=n)
    folds = group % 3
    nuisance = np.column_stack([np.ones(n), dose, dose ** 2, markers, covariate])
    # A duplicate and a zero column exercise rank accounting.
    redundant = np.column_stack([nuisance, dose, np.zeros(n)])
    residual = rng.normal(0, 0.03, households)[group] + rng.normal(0, 0.03, n)
    background_y = 88 + folds + 1.2 * dose ** 2 + markers.sum(1) + covariate
    for j, beta in enumerate((2.0, -2.0, 3.0, -3.0)):
        y = background_y + beta * dose * markers[:, j] + residual
        parts = tuple(
            partial_fold(rows, y[rows], redundant[rows], fiber[rows], markers[rows],
                         raw_y=(y - beta * dose * markers[:, j])[rows])
            for rows in (np.flatnonzero(folds == f)[::-1] for f in range(3))
        )
        result = summarize_effects(parts, group)
        got = result["markers"][MARKERS[j]]
        raw_check = summarize_effects(parts, group, raw=True)["markers"][MARKERS[j]]
        assert raw_check["status"] == "ok" and abs(raw_check["estimate_mg_dl"]) < .04
        q25, q75 = np.quantile(markers[:, j], (0.25, 0.75))
        iqr = q75 - q25
        assert got["status"] == "ok" and np.isclose(got["estimate_mg_dl"], beta * iqr, atol=0.04)
        assert result["nuisance_rank"] == 3 * nuisance.shape[1]
        assert result["households"] == households
        assert got["family95_mg_dl"][0] < got["ci95_mg_dl"][0] < got["ci95_mg_dl"][1] < got["family95_mg_dl"][1]
        assert np.isclose(got["support"]["low"]["q05"], np.quantile(fiber[markers[:, j] <= q25], 0.05))
        assert np.isclose(got["support"]["high"]["q95"], np.quantile(fiber[markers[:, j] >= q75], 0.95))

        # Check the rank-aware sandwich directly in the full stacked model.
        full = np.column_stack([nuisance * (folds == f)[:, None] for f in range(3)]
                               + [dose * markers[:, j]])
        coef = np.linalg.lstsq(full, y, rcond=None)[0]
        influence = full @ np.linalg.inv(full.T @ full)[:, -1]
        by_household = np.bincount(group, weights=influence * (y - full @ coef))
        cr1 = households / (households - 1) * (n - 1) / (n - full.shape[1])
        np.testing.assert_allclose(got["estimate_mg_dl"], coef[-1] * iqr, atol=1e-10)
        np.testing.assert_allclose(got["se_mg_dl"], np.sqrt(cr1 * np.dot(by_household, by_household)) * iqr,
                                   atol=1e-10)

    collapsed = markers.copy()
    collapsed[:, 0] = 0.5
    saturated = np.column_stack([redundant, dose * markers[:, 1]])
    part = partial_fold(np.arange(n), y, saturated, fiber, collapsed)
    rejected = summarize_effects([part], group)["markers"]
    assert rejected[MARKERS[0]]["reason"] == "Zero marker IQR"
    assert rejected[MARKERS[1]]["reason"] == "Interaction absorbed by nuisance basis"
    crossed = group.copy()
    crossed[parts[0].rows[0]] = group[parts[1].rows[0]]
    try:
        summarize_effects(parts, crossed)
    except ValueError as exc:
        assert str(exc) == "FIXED_MARKER_HOUSEHOLD_CROSSES_FOLDS"
    else:
        raise AssertionError("A cross-fold household should be rejected")
