"""Screen every fixed hyperparameter in both repeats with one joint error bound.

Use blocks of columns and common household multiplicities. Person x setting
predictions stay in the TRE. The returned records contain aggregates only.
"""

import numpy as np

import v8_scores as stats


def by_household(values, group, h):
    return np.column_stack([np.bincount(group, weights=v, minlength=h) for v in values.T])


def evaluate(batches, keys, off_indices, target, draws, seed, width=64):
    """Apply the joint threshold over the named procedures AND all fixed settings."""
    a, b = batches
    if (
        not np.array_equal(a["rows"], b["rows"])
        or not np.array_equal(a["groups"], b["groups"])
        or len(keys) != len(set(keys))
        or len(keys) != len(off_indices)
        or draws < 99
        or width < 1
    ):
        raise ValueError("Unpaired or invalid fixed-setting evaluation")
    groups, group = np.unique(a["groups"], return_inverse=True)
    h = len(groups)
    off_indices = np.asarray(off_indices)
    if (
        off_indices.dtype.kind not in "iu"
        or off_indices.shape != (len(keys),)
        or not len(keys)
        or any(
            batch["setting_on"].shape != (len(group), len(keys))
            or min(off_indices) < 0
            or max(off_indices) >= batch["setting_off"].shape[1]
            for batch in batches
        )
    ):
        raise ValueError("Fixed settings lack matched off columns")
    if h < 2:
        raise ValueError("At least two households required")
    rng = np.random.default_rng(seed)
    weights = np.empty((draws, h), dtype=float)
    # Same draw stream as the V8 paired household bootstrap.
    for i in range(draws):
        weights[i] = np.bincount(rng.integers(0, h, size=h), minlength=h)
    variances, good = [], np.ones(draws, dtype=bool)
    for batch in batches:
        y = batch["y"]
        yc = y - y.mean()
        totals = weights @ by_household(np.column_stack((np.ones(len(y)), yc, yc**2)), group, h)
        ss = totals[:, 2] - totals[:, 1] ** 2 / totals[:, 0]
        var = yc @ yc
        if not np.isfinite(var) or var <= 0:
            raise ValueError("No usable outcome variance")
        good &= ss > 64 * np.finfo(float).eps * totals[:, 2]
        variances.append((var, ss))
    weights = weights[good]
    reliable = len(weights) >= np.ceil(0.95 * draws)
    errors = np.full(len(weights), -np.inf)
    results = {k: dict(repeats=[]) for k in keys}
    for r, batch in enumerate(batches):
        sst, ss = variances[r][0], variances[r][1][good, None]
        y = batch["y"][:, None]

        def squared_error(prediction):
            if not np.isfinite(prediction).all():
                raise ValueError("Nonfinite fixed-setting prediction")
            return by_household((y - prediction) ** 2, group, h)

        # Full search protection includes adaptive workflows as well as static settings.
        for start in range(0, batch["on"].shape[1], width):
            part = slice(start, start + width)
            margin_sse = 100 * squared_error(batch["off"][:, part]) - (100 - target) * squared_error(
                batch["on"][:, part]
            )
            margin = margin_sse.sum(axis=0) / sst - target
            sampled = (weights @ margin_sse) / ss - target
            if len(weights):
                errors = np.maximum(errors, np.max(sampled - margin, axis=1))

        # Every penalty of a structure shares the same off fit. Bootstrap these
        # few distinct columns once and index them when scoring the fixed settings.
        base_err = squared_error(batch["setting_off"])
        base = 1.0 - base_err.sum(axis=0) / sst
        base_draws = 1.0 - (weights @ base_err) / ss
        for start in range(0, len(keys), width):
            end = min(len(keys), start + width)
            idx = off_indices[start:end]
            err = squared_error(batch["setting_on"][:, start:end])
            full = 1.0 - err.sum(axis=0) / sst
            full_draws = 1.0 - (weights @ err) / ss
            delta = full - base[idx]
            ds = full_draws - base_draws[:, idx]
            margin = 100.0 * delta - target * full
            simulated = 100.0 * ds - target * full_draws
            if len(weights):
                errors = np.maximum(errors, np.max(simulated - margin, axis=1))
            for j, key in enumerate(keys[start:end]):
                point = np.array([base[idx[j]], full[j], delta[j]])
                sample = np.column_stack((base_draws[:, idx[j]], full_draws[:, j], ds[:, j]))
                ratio, ratio_draws, status = stats.ratios(point, sample, reliable)
                results[key]["repeats"].append(
                    dict(
                        repeat=r,
                        off_r2=float(point[0]),
                        on_r2=float(full[j]),
                        delta_r2_pp=float(100 * delta[j]),
                        delta_r2_ci95_pp=stats.interval(100 * ds[:, j]) if reliable else None,
                        ratio_pct=ratio,
                        ratio_ci95_pct=stats.interval(ratio_draws) if ratio_draws is not None else None,
                        ratio_ci97_5_pct=(
                            stats.interval(ratio_draws, (0.0125, 0.9875))
                            if ratio_draws is not None
                            else None
                        ),
                        ratio_status=status,
                        threshold_margin_r2_pp=float(margin[j]),
                    )
                )
    critical = max(0.0, float(np.quantile(errors, 0.95))) if reliable else None
    for row in results.values():
        scores = row["repeats"]
        values = [s["ratio_pct"] for s in scores]
        row["minimum_ratio_pct"] = min(values) if None not in values else None
        row["above_target_both_point"] = None if None in values else bool(min(values) > target + stats.EPS)
        row["stable_both"] = all(s["ratio_status"] == "stable" for s in scores)
        bounds = [s["ratio_ci97_5_pct"] for s in scores]
        row["minimum_ratio_ci95_pct"] = (
            [min(pair[0] for pair in bounds), min(pair[1] for pair in bounds)]
            if row["stable_both"]
            else None
        )
        for s in scores:
            s["simultaneous_lower_margin_r2_pp"] = (
                None if critical is None else s["threshold_margin_r2_pp"] - critical
            )
        row["above_target_both_adjusted"] = (
            all(s["simultaneous_lower_margin_r2_pp"] > stats.EPS for s in scores)
            if row["stable_both"] and reliable
            else None
        )
    return results, critical, int(good.sum())
