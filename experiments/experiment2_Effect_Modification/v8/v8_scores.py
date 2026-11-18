"""Paired discovery R² and a joint threshold screen across routes and both repeats."""
from collections import Counter

import numpy as np

import v7_eval
from v8_models import FAMILIES
from v8_selection import NAMES, REFERENCE


TARGET_PCT = 6.46
BOOT_SEED = 8182026
EPS = 1e-8


def ordered_predictions(source, folds):
    """Match each fold to its source row ids, then put both repeats in the same order."""
    full_groups = np.asarray(source.groups).astype(str)
    batches = []
    for repeat in (0, 1):
        parts = []
        eligible = np.isfinite(source.reference["y"][repeat])
        for fold in range(5):
            arrays, meta = folds[repeat, fold]
            expected = np.flatnonzero(eligible & (source.reference["outer"][repeat] == fold))
            if (meta["repeat"] != repeat or meta["fold"] != fold or meta["people"] != len(expected)
                    or meta["names"] != list(NAMES) or not np.array_equal(arrays["rows"], expected)
                    or not np.array_equal(arrays["y"], source.reference["y"][repeat, expected])
                    or not np.array_equal(arrays["groups"], full_groups[expected])
                    or arrays["off"].shape != (len(expected), len(NAMES))
                    or arrays["on"].shape != arrays["off"].shape):
                raise ValueError("Outer fold mismatches frozen people, targets, models or families")
            parts.append(arrays)
        all_rows = np.concatenate([a["rows"] for a in parts])
        order = np.argsort(all_rows)
        batch = {key: np.concatenate([a[key] for a in parts])[order]
                 for key in ("rows", "groups", "y", "off", "on")}
        if not np.array_equal(batch["rows"], np.flatnonzero(eligible)):
            raise ValueError("Outer predictions do not cover every eligible person once")
        batches.append(batch)
    for key in ("rows", "groups"):
        if not np.array_equal(batches[0][key], batches[1][key]):
            raise ValueError("Outer repeats must score exactly the same people and households")
    return batches


def scores_from_totals(totals, k):
    """R²_off, R²_on and paired gain for one or more household resamples."""
    sst = totals[:, 2] - totals[:, 1] ** 2 / totals[:, 0]
    good = sst > 64 * np.finfo(float).eps * totals[:, 2]
    out = np.full((len(totals), 3, k), np.nan)
    denom = sst[good, None]
    off_sse = totals[good, 3:3+k]
    on_sse = totals[good, 3+k:]
    out[good, 0] = 1 - off_sse / denom
    out[good, 1] = 1 - on_sse / denom
    out[good, 2] = (off_sse - on_sse) / denom
    return out


def paired_bootstrap(batches, draws, seed):
    if not isinstance(draws, (int, np.integer)) or draws < 99 or len(batches) != 2:
        raise ValueError("Two repeats and at least 99 bootstrap draws are required")
    if (not np.array_equal(batches[0]["rows"], batches[1]["rows"])
            or not np.array_equal(batches[0]["groups"], batches[1]["groups"])):
        raise ValueError("Bootstrap requires identically aligned rows and households")
    stats = [v7_eval._matrices(batch["y"], batch["groups"], batch["off"], batch["on"]) for batch in batches]
    if stats[0].shape != stats[1].shape:
        raise ValueError("Invalid paired household matrices")
    h = len(stats[0])
    k = batches[0]["off"].shape[1]
    point = np.stack([scores_from_totals(s.sum(axis=0, keepdims=True), k)[0] for s in stats])
    if not np.isfinite(point).all():
        raise ValueError("Outer evaluation has insufficient finite outcome variance")
    bootstrap = np.empty((draws, 2, 3, k))
    rng = np.random.default_rng(seed)
    for start in range(0, draws, 64):
        stop = min(start + 64, draws)
        samples = rng.integers(0, h, size=(stop - start, h))
        counts = np.stack([np.bincount(row, minlength=h) for row in samples])
        for repeat, s in enumerate(stats):
            bootstrap[start:stop, repeat] = scores_from_totals(counts @ s, k)
    ok = np.isfinite(bootstrap).all(axis=(1, 2, 3))
    return point, bootstrap[ok]


def interval(sample, probabilities=(.025, .975)):
    return np.quantile(sample, probabilities).tolist()


def ratios(point, draws, reliable):
    off, full, delta = point
    if off <= EPS or full <= EPS:
        return None, None, "nonpositive_or_near_zero_R2"
    ratio = float(100 * delta / full)
    if not reliable:
        return ratio, None, "insufficient_valid_draws"
    if np.any(np.quantile(draws[:, :2], .025, axis=0) <= EPS):
        return ratio, None, "R2_interval_reaches_zero"
    if np.any(draws[:, 1] <= EPS):
        return ratio, None, "nonpositive_full_R2_in_bootstrap"
    return ratio, 100 * draws[:, 2] / draws[:, 1], "stable"


def compare_routes(batches, names, families, reference, draws=1999, seed=BOOT_SEED):
    if (batches[0]["off"].shape[1] != len(names) or len(names) != len(set(names))
            or reference not in names or not set(families).issubset(names)):
        raise ValueError("Route names do not identify paired prediction columns")
    point, boot = paired_bootstrap(batches, draws, seed)
    reliable = len(boot) >= np.ceil(.95 * draws)
    # At positive R²_on this margin is positive iff the ratio exceeds the target.
    margin = 100 * point[:, 2] - TARGET_PCT * point[:, 1]
    critical = None
    if reliable:
        simulated = 100 * boot[:, :, 2] - TARGET_PCT * boot[:, :, 1]
        largest_error = np.max(simulated - margin, axis=(1, 2))
        critical = max(0., float(np.quantile(largest_error, .95)))
    j = names.index(reference)
    results = {}
    for i, name in enumerate(names):
        records, projected = [], []
        for r in (0, 1):
            b, f, d = point[r, :, i]
            sample = boot[:, r, :, i]
            value, ratio_draws, status = ratios(point[r, :, i], sample, reliable)
            anchor, anchor_draws, _ = ratios(point[r, :, j], boot[:, r, :, j], reliable)
            paired_ratio = None if value is None or anchor is None else value - anchor
            paired_ci = (interval(ratio_draws - anchor_draws)
                         if ratio_draws is not None and anchor_draws is not None else None)
            lower = None if critical is None else float(margin[r, i] - critical)
            projected.append(None if ratio_draws is None else interval(ratio_draws, (.0125, .9875)))
            records.append(dict(
                repeat=r, off_r2=float(b), on_r2=float(f), delta_r2=float(d),
                off_r2_ci95=interval(sample[:, 0]) if reliable else None,
                on_r2_ci95=interval(sample[:, 1]) if reliable else None,
                delta_r2_pp=float(100 * d), delta_r2_ci95_pp=interval(100 * sample[:, 2]) if reliable else None,
                ratio_pct=value, ratio_ci95_pct=interval(ratio_draws) if ratio_draws is not None else None,
                ratio_status=status, ratio_vs_v7_pp=paired_ratio, ratio_vs_v7_ci95_pp=paired_ci,
                off_r2_vs_v7_pp=float(100 * (b - point[r, 0, j])),
                on_r2_vs_v7_pp=float(100 * (f - point[r, 1, j])),
                delta_r2_vs_v7_pp=float(100 * (d - point[r, 2, j])),
                delta_r2_vs_v7_ci95_pp=interval(100 * (sample[:, 2] - boot[:, r, 2, j])) if reliable else None,
                threshold_margin_r2_pp=float(margin[r, i]), simultaneous_lower_margin_r2_pp=lower,
            ))
        values = [row["ratio_pct"] for row in records]
        minimum = min(values) if None not in values else None
        stable = all(row["ratio_status"] == "stable" for row in records)
        # Project a Bonferroni rectangle onto the minimum. This also handles ties.
        min_ci = ([min(bounds[0] for bounds in projected), min(bounds[1] for bounds in projected)]
                  if stable else None)
        results[name] = dict(repeats=records, minimum_ratio_pct=minimum,
                             minimum_ratio_ci95_pct=min_ci, stable_both=stable,
                             above_target_both_point=None if minimum is None else bool(minimum > TARGET_PCT + EPS),
                             above_target_both_adjusted=(
                                 all(row["simultaneous_lower_margin_r2_pp"] > EPS for row in records)
                                 if stable else None))
    baseline = results[reference]["minimum_ratio_pct"]
    for result in results.values():
        m = result["minimum_ratio_pct"]
        result["minimum_ratio_vs_v7_pp"] = None if m is None or baseline is None else m - baseline
    ordered = sorted((name for name in families if results[name]["minimum_ratio_pct"] is not None),
                     key=lambda n: -results[n]["minimum_ratio_pct"])
    return dict(people=len(batches[0]["y"]), households=len(np.unique(batches[0]["groups"])),
                inference="Household bootstrap conditional on fitted predictions; same resampled families "
                          "in both discovery repeats. The threshold margin lower bounds cover all reported "
                          "routes and both repeats jointly. Other intervals are per procedure.",
                repeats_same_people=True, reserved_outcomes_read=False, benchmark_pct=TARGET_PCT,
                bootstrap_draws=draws, valid_draws=len(boot), simultaneous_critical_r2_pp=critical,
                screened_route_count=len(names), routes=results,
                exploratory_ranking_by_minimum=ordered, best_stable_family=next(
                    (name for name in ordered if results[name]["stable_both"]), None),
                both_above_point=[n for n in names if results[n]["above_target_both_point"]],
                both_above_adjusted=[n for n in names if results[n]["above_target_both_adjusted"]])


def summarize(source, folds, draws):
    result = compare_routes(ordered_predictions(source, folds), NAMES, FAMILIES, REFERENCE, draws)
    for name, row in result["routes"].items():
        for r, score in enumerate(row["repeats"]):
            choices = [folds[r, f][1]["choices"][name] for f in range(5)]
            score["active_folds"] = sum(c["enabled"] for c in choices)
            score["choices"] = dict(Counter("+".join(c["settings"]) if c["enabled"] else "off" for c in choices))
    return result
