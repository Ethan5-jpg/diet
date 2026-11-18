"""Nested paired ratio selection and paired household inference for V7.

All choices use the current outer training set's inner held-out predictions.
Model setting keys describe both arms. A disabled setting reuses the exact off
prediction in both arms. All negatives remain in the outer results.
"""

from collections import Counter

import numpy as np

from v7_reference import HISTORICAL_RATIO as BENCHMARK

PRIMARY = "ratio_selector"
GUARDED = "ratio_selector_guarded"
ORIGINAL = "ratio_selector_original35"
# Copy of the original upload's model order and penalties. Numerical ties
# use that original order, including when new model definitions are inserted.
ORIGINAL_FAMILIES = (
    "gut_pc10_ridge",
    "gut_pc4_ridge",
    "gut_pc6_ridge",
    "gut_pc8_ridge",
    "gut_pc12_ridge",
    "gut_pc16_ridge",
    "gut_pc20_ridge",
    "gut_pc32_ridge",
    "gut_pc6_dose_spline",
    "gut_pc10_dose_spline",
    "gut_pc16_dose_spline",
    "gut_pc24_dose_spline",
    "gut_pc10_spectral_taper",
    "gut_pc16_spectral_taper",
    "gut_pc24_spectral_taper",
    "gut_pc32_spectral_taper",
    "gut_pc10_dose_spline_shrink",
    "gut_pc16_dose_spline_taper",
    "gut_pc4_quadratic",
    "gut_pc4_hinge",
    "gut_pc10_quadratic",
    "gut_pc10_hinge",
    "gut_pc10_community_pairs",
    "gut_pc10_oral_pc2",
    "gut_pc10_oral_pc2_shrink",
    "gut_pc10_oral_pc5",
    "gut_pc10_oral_pc5_shrink",
    "gut_pc10_oral_pc10",
    "gut_pc10_oral_pc10_shrink",
    "gut_pc10_shannon",
    "gut_pc10_fermentation",
    "gut_pc10_ecology_both",
    "blend_gut_rank",
    "blend_gut_shape",
    "blend_gut_ecology",
)
ORIGINAL_PENALTIES = ("100", "10", "1", "0.1", "0.01")
REFERENCE = "reference_pc10_p01"
FLOOR = 1e-8
TIE_PCT = 1e-10
# Treat a numerically reproduced benchmark as a tie. Units are percentage points of the ratio.
BENCHMARK_TOL_PCT = 1e-8
BOOT_SEED = 20261118


def route_names():
    from v7_models import families

    return [*families(), REFERENCE, PRIMARY, GUARDED, ORIGINAL]


def _original_settings(current):
    keys = []
    for name in ORIGINAL_FAMILIES:
        expected = [f"{name}@{penalty}" for penalty in ORIGINAL_PENALTIES]
        if name not in current or not set(expected).issubset(current[name]):
            raise ValueError("ORIGINAL35_COMPARISON_SETTING_MISSING " + name)
        keys.extend(expected)
    return keys


def _matrices(y, groups, off, on):
    y, off, on = (np.asarray(a, dtype=float) for a in (y, off, on))
    groups = np.asarray(groups)
    if (
        y.ndim != 1
        or len(y) < 2
        or groups.shape != y.shape
        or off.ndim != 2
        or off.shape[0] != len(y)
        or off.shape != on.shape
        or off.shape[1] == 0
        or not all(np.isfinite(a).all() for a in (y, off, on))
    ):
        raise ValueError("INVALID_PAIRED_PREDICTIONS")
    if groups.dtype.kind in "fc" and not np.isfinite(groups).all():
        raise ValueError("INVALID_HOUSEHOLD")
    if any(v is None or str(v).strip() == "" for v in groups):
        raise ValueError("INVALID_HOUSEHOLD")
    _, g = np.unique(groups, return_inverse=True)
    h = int(g.max()) + 1
    if h < 2:
        raise ValueError("INSUFFICIENT_HOUSEHOLDS")
    yc = y - y.mean()
    sst = yc @ yc
    if not np.isfinite(sst) or sst <= 0:
        raise ValueError("INVALID_OUTCOME_VARIANCE")
    offerr, onerr = (y[:, None] - off) ** 2, (y[:, None] - on) ** 2
    stats = np.column_stack([np.ones(len(y)), yc, yc**2, offerr, onerr])
    household = np.column_stack([np.bincount(g, weights=v, minlength=h) for v in stats.T])
    if not np.isfinite(household).all():
        raise ValueError("NONFINITE_SCORING_STATISTICS")
    return household


def _score_components(household, count):
    totals = household.sum(axis=0)
    sst = totals[2] - totals[1] ** 2 / totals[0]
    eoff, eon = totals[3 : 3 + count], totals[3 + count :]
    return sst, 1 - eoff / sst, 1 - eon / sst, (eoff - eon) / sst


def _select_inner(y, groups, off, on, keys, families, anchor, reference_key, original_keys):
    """Choose settings by their paired inner-held-out ratio.

    First-order cluster influences of 1 - SSE/SST yield an inner selection
    guard, not final confidence limits. The spread of (B - anchor_B) uses
    paired household influences. Family null predictions must be identical
    across all interaction settings, by construction.
    """
    stats = _matrices(y, groups, off, on)
    cols = {k: i for i, k in enumerate(keys)}
    if len(cols) != len(keys) or reference_key not in families[anchor]:
        raise ValueError("INVALID_REFERENCE_OR_DUPLICATE_SETTINGS")
    k, h = len(keys), len(stats)
    sst, b, f, delta = _score_components(stats, k)
    ref = cols[reference_key]
    off_if = (stats[:, 2, None] * (1 - b) - stats[:, 3 : 3 + k]) / sst
    on_if = (stats[:, 2, None] * (1 - f) - stats[:, 3 + k :]) / sst

    def se(influences):
        return np.sqrt((h / (h - 1)) * np.sum(influences**2, axis=0))

    ratio = np.divide(100 * delta, f, out=np.full(k, np.nan), where=f > FLOOR)
    eligible = (b > FLOOR) & (f > FLOOR) & (ratio > TIE_PCT)
    protected = (
        eligible
        & (b - 1.96 * se(off_if) > FLOOR)
        & (f - 1.96 * se(on_if) > FLOOR)
        & (b >= b[ref] - se(off_if - off_if[:, [ref]]) - 1e-12)
    )

    def choice(indices, admissible, fallback):
        winner = fallback
        best = 0.0
        enabled = False
        for i in indices:
            if admissible[i] and ratio[i] > best + TIE_PCT:
                winner, best, enabled = i, float(ratio[i]), True
        return keys[winner], enabled

    choices = {}
    for name, settings in families.items():
        idx = [cols[key] for key in settings]
        if not np.array_equal(off[:, idx], np.broadcast_to(off[:, [idx[0]]], (len(off), len(idx)))):
            raise ValueError("FAMILY_NULL_CHANGED_WITH_INTERACTION_PENALTY " + name)
        choices[name] = choice(idx, eligible, idx[0])
    choices[REFERENCE] = (reference_key, True)
    choices[PRIMARY] = choice(range(k), eligible, ref)
    choices[ORIGINAL] = choice([cols[key] for key in original_keys], eligible, ref)
    choices[GUARDED] = choice(range(k), protected, ref)
    return choices, dict(
        positive_settings=int(eligible.sum()),
        protected_settings=int(protected.sum()),
        inner_people=len(y),
        inner_households=h,
        anchor_off_r2=float(b[ref]),
    )


def _pack_predictions(pairs, keys, rows):
    if set(pairs) != set(keys):
        raise ValueError("MODEL_GRID_MISMATCH")
    result = []
    for arm in (0, 1):
        cols = [np.asarray(pairs[key][arm], dtype=float) for key in keys]
        if any(a.shape != (rows,) or not np.isfinite(a).all() for a in cols):
            raise ValueError("NONFINITE_OR_MISALIGNED_PREDICTION")
        result.append(np.column_stack(cols))
    return result


def _row_mask(rows, size):
    rows = np.asarray(rows)
    if (rows.ndim != 1 or rows.dtype.kind not in "iu" or not rows.size
            or rows.min() < 0 or rows.max() >= size):
        raise ValueError("INVALID_ROLE_ROW_INDICES")
    mask = np.zeros(size, dtype=bool)
    mask[rows] = True
    if mask.sum() != rows.size:
        raise ValueError("DUPLICATE_ROLE_ROWS")
    return mask


def _check_roles(groups, assignment, fold, inner, outer):
    """Check the complete archived household partition before evaluating any model."""
    if assignment.shape != groups.shape or len(inner) != 3:
        raise ValueError("INVALID_HOUSEHOLD_PARTITION")
    held = assignment == fold
    if (not held.any() or held.all() or set(groups[held]).intersection(groups[~held])):
        raise ValueError("OUTER_HOUSEHOLD_OVERLAP")
    inner_validation_groups = set()
    pca_held_count = np.zeros(len(groups), dtype=int)
    for split in (outer, *inner):
        train = _row_mask(split.train_rows, len(groups))
        valid = _row_mask(split.rows, len(groups))
        if np.any(held & train) or (np.any(valid & ~held) if split is outer else np.any(valid & held)):
            raise ValueError("OUTER_HOUSEHOLD_OVERLAP")
        validation_groups = set(groups[valid])
        if set(groups[train]).intersection(validation_groups):
            raise ValueError("HOUSEHOLD_CROSSES_MODEL_ROLE")
        if split is not outer:
            if inner_validation_groups.intersection(validation_groups):
                raise ValueError("INNER_HOUSEHOLD_OVERLAP")
            inner_validation_groups.update(validation_groups)
        if not hasattr(split, "pca_training_rows"):
            continue
        pca = _row_mask(split.pca_training_rows, len(groups))
        if np.any(pca & held) or np.any(train & ~pca) or set(groups[pca]).intersection(validation_groups):
            raise ValueError("PCA_TRAINING_ROLE_LEAK")
        if split is outer:
            if not np.array_equal(pca, ~held):
                raise ValueError("OUTER_PCA_TRAINING_ROLE_INCOMPLETE")
        else:
            # The complement includes people ineligible for clinical prediction.
            # All members of each family belong on the same side of this split.
            validation = ~held & ~pca
            if set(groups[pca]).intersection(groups[validation]):
                raise ValueError("PCA_HOUSEHOLD_CROSSES_INNER_ROLE")
            pca_held_count += validation
    if any(hasattr(split, "pca_training_rows") for split in (outer, *inner)):
        if not all(hasattr(split, "pca_training_rows") for split in (outer, *inner)):
            raise ValueError("MISSING_PCA_TRAINING_ROLE")
        if not np.array_equal(pca_held_count, (~held).astype(int)):
            raise ValueError("INNER_PCA_ROLES_DO_NOT_PARTITION_OUTER_TRAINING")


def run_fold(source, repeat, fold):
    from v7_models import ANCHOR, evaluate_split, families

    family = families()
    keys = [key for settings in family.values() for key in settings]
    reference_key = ANCHOR + "@0.1"
    inner, outer = source.roles(repeat, fold)
    groups = np.asarray(source.groups)
    assignment = np.asarray(source.reference["outer"][repeat])
    _check_roles(groups, assignment, fold, inner, outer)
    pieces = []
    for split in inner:
        off, on = _pack_predictions(evaluate_split(source, split), keys, len(split.rows))
        # The frozen adapter transforms each inner validation target using only
        # its own training role. Read split.yv instead of raw or outer targets.
        pieces.append((split.yv, groups[split.rows], off, on))
    y, g, off, on = [np.concatenate(parts) for parts in zip(*pieces, strict=True)]
    choices, audit = _select_inner(y, g, off, on, keys, family, ANCHOR, reference_key,
                                   _original_settings(family))
    # Outer targets have not entered any baseline, penalty or family choice.
    view_audit = {}
    off, on = _pack_predictions(evaluate_split(source, outer, view_audit=view_audit), keys, len(outer.rows))
    cols = {key: i for i, key in enumerate(keys)}
    names = route_names()
    nulls, fulls = [], []
    for name in names:
        key, enabled = choices[name]
        i = cols[key]
        nulls.append(off[:, i])
        fulls.append((on if enabled else off)[:, i])
    arrays = dict(
        rows=np.asarray(outer.rows),
        y=np.asarray(outer.yv),
        groups=groups[outer.rows].astype(str),
        off=np.column_stack(nulls),
        on=np.column_stack(fulls),
    )
    meta = dict(
        repeat=repeat,
        fold=fold,
        names=names,
        people=len(outer.rows),
        choices={n: dict(setting=k, enabled=e) for n, (k, e) in choices.items()},
        selection_audit=audit,
        modifier_view_audit=view_audit,
    )
    return arrays, meta


def _ci(values):
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def paired_scores(y, groups, off, on, names, *, draws=1999, seed=BOOT_SEED,
                  reference=REFERENCE, family_names=()):
    """Jointly resample entire households and recompute SST and all paired R²s.

    Each column has its own main-only prediction. Identical household draws
    across columns propagate covariance for differences from the PC10 anchor.
    """
    if not isinstance(draws, (int, np.integer)) or draws < 99:
        raise ValueError("BOOTSTRAP_REQUIRES_AT_LEAST_99_DRAWS")
    stats = _matrices(y, groups, off, on)
    k, h = len(names), len(stats)
    if len(set(names)) != k or reference not in names or stats.shape[1] != 3 + 2 * k:
        raise ValueError("INVALID_PAIR_NAMES")
    _, b, f, delta = _score_components(stats, k)
    boot = np.full((draws, 3, k), np.nan)
    rng = np.random.default_rng(seed)
    for start in range(0, draws, 64):
        stop = min(draws, start + 64)
        sampled = rng.integers(0, h, size=(stop - start, h))
        totals = np.stack([np.bincount(s, minlength=h) for s in sampled]) @ stats
        sst = totals[:, 2] - totals[:, 1] ** 2 / totals[:, 0]
        valid = sst > 64 * np.finfo(float).eps * totals[:, 2]
        den = sst[valid, None]
        off_sse, on_sse = totals[valid, 3 : 3 + k], totals[valid, 3 + k :]
        at = np.arange(start, stop)[valid]
        boot[at, 0] = 1 - off_sse / den
        boot[at, 1] = 1 - on_sse / den
        boot[at, 2] = (off_sse - on_sse) / den
    ok = np.isfinite(boot).all(axis=(1, 2))
    boot = boot[ok]
    reliable = len(boot) >= np.ceil(0.95 * draws)
    ratio_samples, records = {}, {}
    for i, name in enumerate(names):
        intervals = [_ci(boot[:, j, i]) if reliable else None for j in range(3)]
        off_ci, full_ci, delta_ci = intervals
        arithmetic = float(100 * delta[i] / f[i]) if abs(f[i]) > FLOOR else None
        ratio = arithmetic if b[i] > FLOOR and f[i] > FLOOR else None
        if ratio is None:
            status = "nonpositive_or_near_zero_R2"
        elif not reliable:
            status = "insufficient_valid_draws"
        elif off_ci[0] <= FLOOR or full_ci[0] <= FLOOR:
            status = "R2_interval_reaches_zero"
        elif np.any(boot[:, 1, i] <= FLOOR):
            status = "nonpositive_full_R2_in_bootstrap"
        else:
            status = "stable"
            ratio_samples[name] = 100 * boot[:, 2, i] / boot[:, 1, i]
        ci = _ci(ratio_samples[name]) if name in ratio_samples else None
        records[name] = dict(
            baseline_r2=float(b[i]),
            full_r2=float(f[i]),
            baseline_r2_ci95=off_ci,
            full_r2_ci95=full_ci,
            delta_r2=float(delta[i]),
            delta_r2_percentage_points=float(100 * delta[i]),
            delta_r2_ci95=delta_ci,
            ratio_over_full_pct=ratio,
            ratio_over_full_ci95_pct=ci,
            arithmetic_ratio_over_full_pct=arithmetic,
            ratio_over_full_status=status,
            beats_historical_point=None if ratio is None else ratio > BENCHMARK + BENCHMARK_TOL_PCT,
            beats_historical_ci=None if ci is None else ci[0] > BENCHMARK + BENCHMARK_TOL_PCT,
        )
    for name, record in records.items():
        ratio, anchor = record["ratio_over_full_pct"], records[reference]["ratio_over_full_pct"]
        record["ratio_minus_reference_pp"] = (
            None if ratio is None or anchor is None else ratio - anchor
        )
        record["ratio_minus_reference_ci95_pp"] = (
            _ci(ratio_samples[name] - ratio_samples[reference])
            if name in ratio_samples and reference in ratio_samples
            else None
        )
    addition = {}
    if PRIMARY in records and ORIGINAL in records:
        primary = records[PRIMARY]["ratio_over_full_pct"]
        original = records[ORIGINAL]["ratio_over_full_pct"]
        i, j = names.index(PRIMARY), names.index(ORIGINAL)
        addition["expanded_vs_original35"] = dict(
            expanded_ratio_pct=primary,
            original35_ratio_pct=original,
            ratio_difference_pp=None if primary is None or original is None else primary - original,
            ratio_difference_ci95_pp=(
                _ci(ratio_samples[PRIMARY] - ratio_samples[ORIGINAL])
                if PRIMARY in ratio_samples and ORIGINAL in ratio_samples else None
            ),
            delta_r2_difference_pp=float(100 * (delta[i] - delta[j])),
            delta_r2_difference_ci95_pp=(
                _ci(100 * (boot[:, 2, i] - boot[:, 2, j])) if reliable else None
            ),
        )
    if family_names:
        from v7_threshold import annotate
        addition["threshold_screen"] = annotate(records, names, boot, tuple(family_names), reliable)
    return dict(
        **addition,
        people=len(y),
        households=h,
        bootstrap_draws=int(draws),
        bootstrap_valid_draws=int(ok.sum()),
        routes=records,
    )


def summarize(source, folds, *, draws=1999):
    from v7_models import families

    family_names = list(families())
    names = route_names()
    reports = []
    for repeat in (0, 1):
        parts = [folds[(repeat, f)] for f in range(5)]
        expected = np.isfinite(source.reference["y"][repeat])
        for fold, (arrays, meta) in enumerate(parts):
            rows = arrays["rows"]
            expected_rows = np.flatnonzero(expected & (source.reference["outer"][repeat] == fold))
            correct = (
                meta["repeat"] == repeat
                and meta["fold"] == fold
                and meta["names"] == names
                and np.array_equal(rows, expected_rows)
                and np.array_equal(arrays["y"], source.reference["y"][repeat, rows])
                and np.array_equal(arrays["groups"], np.asarray(source.groups)[rows].astype(str))
            )
            if not correct:
                raise ValueError("OUTER_FOLD_ALIGNMENT_CHANGED")
        y, groups, off, on = [
            np.concatenate([a[key] for a, _ in parts]) for key in ("y", "groups", "off", "on")
        ]
        rep = paired_scores(y, groups, off, on, names, draws=draws,
                            seed=BOOT_SEED + repeat, family_names=family_names)
        rep["repeat"] = repeat
        for name, rec in rep["routes"].items():
            choices = [m["choices"][name] for _, m in parts]
            rec["active_folds"] = sum(c["enabled"] for c in choices)
            rec["choices"] = dict(
                Counter(c["setting"] if c["enabled"] else c["setting"] + " (off)" for c in choices)
            )

        def maximum(stable):
            candidates = []
            for n in family_names:
                r = rep["routes"][n]
                if (
                    r["ratio_over_full_pct"] is not None
                    and r["ratio_over_full_pct"] > 0
                    and (not stable or r["ratio_over_full_status"] == "stable")
                ):
                    candidates.append((n, r))
            if not candidates:
                return None
            n, r = max(candidates, key=lambda pair: pair[1]["ratio_over_full_pct"])
            return dict(route=n, **r)

        rep["exploratory_observed_max"] = maximum(False)
        rep["exploratory_stable_max"] = maximum(True)
        reports.append(rep)
    both = []
    for n in family_names:
        recs = [r["routes"][n] for r in reports]
        ratios = [r["ratio_over_full_pct"] for r in recs]
        if all(r is not None and r > 0 for r in ratios):
            both.append(
                dict(
                    route=n,
                    ratios_pct=ratios,
                    minimum_ratio_pct=min(ratios),
                    both_exceed_benchmark=min(ratios) > BENCHMARK + BENCHMARK_TOL_PCT,
                    stable_both=all(r["ratio_over_full_status"] == "stable" for r in recs),
                )
            )
    both.sort(key=lambda row: row["minimum_ratio_pct"], reverse=True)
    return dict(
        repeats=reports,
        positive_both_repeats=both,
        benchmark_pct=BENCHMARK,
        benchmark_tie_tolerance_pct=BENCHMARK_TOL_PCT,
        primary_procedure=PRIMARY,
        reference_route=REFERENCE,
        guarded_procedure=GUARDED,
        original35_procedure=ORIGINAL,
        maximum_scope="Individually tuned families; excludes pooled selectors and fixed reference.",
        family_count=len(family_names),
        route_count=len(names),
        reserved_outcomes_read=False,
        ratio_definition="100*(R2_on - R2_off)/R2_on",
        inference="Household percentile bootstrap conditional on fixed outer-fold predictions. "
        "Route maxima are exploratory. Route percentile intervals are pointwise. Each repeat also "
        "reports a simultaneous one-sided 95% lower bound on the family-wise margin above the fixed "
        "historical ratio threshold. Intervals condition on the fitted predictions. "
        "The historical percentage is a fixed descriptive comparator without a historical ratio CI; "
        "both repeats use the same people. Paired reference differences use the current PC10 @0.1 on these same rows.",
    )
