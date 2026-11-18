"""Paired interaction scores on a fixed cohort and paired household bootstrap.

The ratios in this module are arithmetic ratios of predictive R2 quantities.
They are not a causal partition of a dietary effect. Bootstrap intervals fix
the fitted predictions and do not cover historical search or model refitting.
"""

from numbers import Integral

import numpy as np


DEFAULT_SEED = 61118
DENOMINATOR_FLOOR = 1e-8
TIE_RTOL = 1e-10
TIE_ATOL = 1e-12


def _prediction_arrays(y, baseline, full):
    y, baseline, full = (np.asarray(value, dtype=float) for value in (y, baseline, full))
    if y.ndim != 1 or len(y) < 2 or baseline.shape != y.shape:
        raise ValueError("INVALID_PAIRED_OUTCOME_OR_BASELINE")
    if full.ndim == 1:
        full = full[:, None]
    if full.ndim != 2 or full.shape[0] != len(y):
        raise ValueError("INVALID_PAIRED_FULL_PREDICTIONS")
    if not all(np.isfinite(value).all() for value in (y, baseline, full)):
        raise ValueError("NONFINITE_PAIRED_INPUT")
    return y, baseline, full


def _group_index(groups, n):
    # Object conversion preserves mixed IDs so integer 1 and string "1" are
    # not silently collapsed by NumPy's automatic conversion to Unicode.
    groups = np.asarray(groups, dtype=object)
    if groups.ndim != 1 or len(groups) != n:
        raise ValueError("INVALID_HOUSEHOLD_SHAPE")
    # Do not turn missing values into apparently valid string labels.
    for value in groups:
        if value is None or isinstance(value, (list, tuple, dict, set)):
            raise ValueError("INVALID_HOUSEHOLD_LABEL")
        if isinstance(value, (str, bytes, np.str_, np.bytes_)) and not value.strip():
            raise ValueError("MISSING_HOUSEHOLD_LABEL")
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            raise ValueError("MISSING_HOUSEHOLD_LABEL")
        if not isinstance(value, (str, bytes, np.str_, np.bytes_, Integral, np.integer,
                                  float, np.floating)):
            raise ValueError("UNSUPPORTED_HOUSEHOLD_LABEL")
    try:
        unique, index = np.unique(groups, return_inverse=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("INCOMPARABLE_HOUSEHOLD_LABELS") from exc
    if len(unique) < 2:
        raise ValueError("INSUFFICIENT_HOUSEHOLDS")
    return index, len(unique)


def _names(names, count):
    if names is None:
        return tuple(f"candidate_{j}" for j in range(count))
    names = tuple(names)
    if len(names) != count or any(
        not isinstance(name, str) or not name.strip() for name in names
    ):
        raise ValueError("INVALID_ROUTE_NAMES")
    if len(set(names)) != count:
        raise ValueError("INVALID_ROUTE_NAMES")
    return names


def _percentile(values):
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def _ratio(point_delta, point_denominator, boot_delta, boot_denominator,
           denominator_ci, denominator_floor, bootstrap_reliable):
    """Keep signed gains; do not promote a negative/negative ratio as a benefit.

    A percentile ratio interval is withheld if the denominator's paired 95%
    interval reaches zero or if any otherwise-valid draw is numerically zero.
    Negative bootstrap denominators are never silently truncated or resampled.
    """
    arithmetic = (float(100 * point_delta / point_denominator)
                  if abs(point_denominator) > denominator_floor else None)
    if point_denominator <= denominator_floor:
        return dict(value=None, arithmetic=arithmetic, ci95=None,
                    status="nonpositive_or_near_zero_point_denominator")
    point = float(100 * point_delta / point_denominator)
    if not bootstrap_reliable:
        return dict(value=point, arithmetic=arithmetic, ci95=None,
                    status="insufficient_valid_bootstrap_draws")
    if denominator_ci[0] <= denominator_floor:
        return dict(value=point, arithmetic=arithmetic, ci95=None,
                    status="denominator_ci_reaches_zero")
    if np.any(np.abs(boot_denominator) <= denominator_floor):
        return dict(value=point, arithmetic=arithmetic, ci95=None,
                    status="bootstrap_denominator_near_zero")
    values = 100 * boot_delta / boot_denominator
    if not np.isfinite(values).all():
        return dict(value=point, arithmetic=arithmetic, ci95=None,
                    status="nonfinite_bootstrap_ratio")
    return dict(value=point, arithmetic=arithmetic, ci95=_percentile(values),
                status="paired_percentile_denominator_positive_ci")


def summarize_paired(y, baseline, full, groups, *, names=None, draws=1999,
                     seed=DEFAULT_SEED, denominator_floor=DENOMINATOR_FLOOR,
                     chunk_size=64):
    """Return JSON-ready scores and uncertainty for paired held-out predictions.

    ``full`` is n people by k candidate routes (a vector is one route). Every
    route shares ``y``, ``baseline`` and the identical person rows. Household
    bootstrap draws sample H households with replacement and retain every
    person in a sampled household with that household's multiplicity. R2 is
    pooled across all persons, not an average of fold or household R2 values.

    The outcome total sum of squares is recomputed in every draw, using a
    globally centered outcome for numerical stability. The same household draw
    changes the R2 numerator, full-model denominator and baseline denominator.
    Undefined constant-outcome draws are counted. Intervals are withheld if
    more than 5% of draws have undefined outcome variance. Ratio intervals have
    the additional denominator checks documented in ``_ratio``.
    """
    y, baseline, full = _prediction_arrays(y, baseline, full)
    if full.shape[1] == 0:
        raise ValueError("NO_PREDICTION_ROUTES")
    names = _names(names, full.shape[1])
    group, households = _group_index(groups, len(y))
    if isinstance(draws, bool) or not isinstance(draws, Integral) or draws < 99:
        raise ValueError("BOOTSTRAP_DRAWS_MUST_BE_AT_LEAST_99")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral) or chunk_size < 1:
        raise ValueError("INVALID_BOOTSTRAP_CHUNK_SIZE")
    if not np.isfinite(denominator_floor) or denominator_floor <= 0:
        raise ValueError("INVALID_RATIO_DENOMINATOR_FLOOR")

    centered = y - y.mean()
    sst = float(centered @ centered)
    if not np.isfinite(sst) or sst <= 0:
        raise ValueError("INVALID_POOLED_OUTCOME_VARIANCE")
    with np.errstate(over="ignore", invalid="ignore"):
        baseline_error = (y - baseline) ** 2
        full_error = (y[:, None] - full) ** 2
        row_statistics = np.column_stack((np.ones(len(y)), centered, centered ** 2,
                                          baseline_error, full_error))
    if not np.isfinite(row_statistics).all():
        raise ValueError("PAIRED_SCORE_NUMERIC_OVERFLOW")
    statistics = np.column_stack([
        np.bincount(group, weights=column, minlength=households)
        for column in row_statistics.T
    ])
    baseline_sse = float(baseline_error.sum())
    full_sse = full_error.sum(axis=0)
    if not np.isfinite(statistics).all() or not np.isfinite(baseline_sse) or not np.isfinite(full_sse).all():
        raise ValueError("PAIRED_SCORE_NUMERIC_OVERFLOW")
    base_r2 = float(1 - baseline_sse / sst)
    full_r2 = 1 - full_sse / sst
    # Compute a paired difference directly, avoiding cancellation of close R2.
    delta = (baseline_sse - full_sse) / sst
    if not np.isfinite(base_r2) or not np.isfinite(full_r2).all() or not np.isfinite(delta).all():
        raise ValueError("PAIRED_SCORE_NUMERIC_OVERFLOW")

    boot_base = np.full(draws, np.nan)
    boot_full = np.full((draws, len(names)), np.nan)
    boot_delta = np.full((draws, len(names)), np.nan)
    rng = np.random.default_rng(seed)
    for start in range(0, draws, chunk_size):
        stop = min(start + chunk_size, draws)
        pick = rng.integers(0, households, size=(stop - start, households))
        counts = np.vstack([np.bincount(row, minlength=households) for row in pick])
        totals = counts @ statistics
        total_sst = totals[:, 2] - totals[:, 1] ** 2 / totals[:, 0]
        tolerance = 64 * np.finfo(float).eps * np.abs(totals[:, 2])
        valid = np.isfinite(total_sst) & (total_sst > tolerance)
        positions = np.arange(start, stop)[valid]
        valid_totals, denominator = totals[valid], total_sst[valid]
        boot_base[positions] = 1 - valid_totals[:, 3] / denominator
        boot_full[positions] = 1 - valid_totals[:, 4:] / denominator[:, None]
        boot_delta[positions] = (valid_totals[:, [3]] - valid_totals[:, 4:]) / denominator[:, None]
    valid = np.isfinite(boot_base) & np.isfinite(boot_full).all(axis=1) & np.isfinite(boot_delta).all(axis=1)
    valid_draws = int(valid.sum())
    reliable = valid_draws >= int(np.ceil(0.95 * draws))
    base_boot, full_boot, delta_boot = boot_base[valid], boot_full[valid], boot_delta[valid]
    baseline_ci = _percentile(base_boot) if reliable else None
    nonpositive_base = float(np.mean(base_boot <= 0)) if valid_draws else None
    result = dict(
        people=len(y), households=households, baseline_r2=base_r2,
        baseline_r2_ci95=baseline_ci, bootstrap_draws=int(draws),
        bootstrap_valid_draws=valid_draws,
        bootstrap_undefined_outcome_fraction=float(1 - valid_draws / draws),
        bootstrap_nonpositive_baseline_fraction=nonpositive_base,
        ratio_denominator_floor=float(denominator_floor),
        interval_method="paired household percentile bootstrap; predictions fixed",
        interval_scope="conditional on fitted predictions; excludes refitting, historical search and selection of best reported route",
        routes={},
    )
    for j, name in enumerate(names):
        full_ci = _percentile(full_boot[:, j]) if reliable else None
        numerator_ci = _percentile(delta_boot[:, j]) if reliable else None
        full_ratio = _ratio(delta[j], full_r2[j], delta_boot[:, j], full_boot[:, j],
                            full_ci, denominator_floor, reliable)
        base_ratio = _ratio(delta[j], base_r2, delta_boot[:, j], base_boot,
                            baseline_ci, denominator_floor, reliable)
        positive_denominators = bool(base_r2 > denominator_floor and full_r2[j] > denominator_floor)
        result["routes"][name] = dict(
            full_r2=float(full_r2[j]), full_r2_ci95=full_ci,
            delta_r2=float(delta[j]), delta_r2_ci95=numerator_ci,
            delta_r2_percentage_points=float(100 * delta[j]),
            ratio_over_full_pct=full_ratio["value"],
            ratio_over_full_ci95_pct=full_ratio["ci95"],
            ratio_over_full_status=full_ratio["status"],
            arithmetic_ratio_over_full_pct=full_ratio["arithmetic"],
            relative_r2_change_over_base_pct=base_ratio["value"],
            relative_r2_change_over_base_ci95_pct=base_ratio["ci95"],
            ratio_over_base_status=base_ratio["status"],
            arithmetic_ratio_over_base_pct=base_ratio["arithmetic"],
            positive_r2_denominators=positive_denominators,
            rankable_positive_gain=bool(positive_denominators and delta[j] > 0),
            bootstrap_nonpositive_full_fraction=(float(np.mean(full_boot[:, j] <= 0))
                                                 if valid_draws else None),
            bootstrap_near_zero_full_fraction=(float(np.mean(np.abs(full_boot[:, j]) <= denominator_floor))
                                               if valid_draws else None),
        )
    return result


def select_correction(y, baseline, corrections, *, names=None):
    """Choose the minimum pooled inner-validation SSE, always including zero.

    Each correction column must be an inner-held-out correction to the same
    baseline, never fitted values on the tuning outcomes. Names should arrive
    in the caller's predeclared tie order (for example, stronger penalties
    first). Zero has first priority in a numerical tie. This objective also
    maximizes a positive delta/full ratio when baseline R2 is fixed and positive;
    it cannot win by changing or weakening the shared main-effect baseline.
    """
    y, baseline, corrections = _prediction_arrays(y, baseline, corrections)
    names = _names(names, corrections.shape[1])
    if "zero" in names:
        raise ValueError("ZERO_NAME_RESERVED_FOR_NULL_CANDIDATE")
    with np.errstate(over="ignore", invalid="ignore"):
        prediction = baseline[:, None] + corrections
        zero_sse = float(np.sum((y - baseline) ** 2))
        candidate_sse = np.sum((y[:, None] - prediction) ** 2, axis=0)
    if not np.isfinite(zero_sse) or not np.isfinite(candidate_sse).all():
        raise ValueError("CORRECTION_SCORE_NUMERIC_OVERFLOW")
    errors = np.r_[zero_sse, candidate_sse]
    minimum = float(errors.min())
    tolerance = TIE_ATOL + TIE_RTOL * max(abs(zero_sse), abs(minimum))
    chosen = int(np.flatnonzero(errors <= minimum + tolerance)[0])
    labels = ("zero",) + names
    return dict(name=labels[chosen], index=None if chosen == 0 else chosen - 1,
                is_zero=chosen == 0, sse=float(errors[chosen]), zero_sse=zero_sse,
                sse_gain=float(zero_sse - errors[chosen]),
                scores={name: float(error) for name, error in zip(labels, errors)},
                objective="minimum pooled inner-held-out SSE with fixed baseline",
                tie_rule="zero first, then predeclared candidate order")
