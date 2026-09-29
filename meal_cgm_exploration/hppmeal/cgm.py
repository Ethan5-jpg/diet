"""Exact piecewise-linear areas, conservative gaps, no merging across sensors."""
import numpy as np
import pandas as pd

OUTCOMES = ["iauc_net_0_120", "iauc_positive_0_120", "peak_rise_0_120", "peak_time_minutes"]


def areas(minutes, deviations):
    """Net and positive-only integrals, splitting segments at baseline crossings."""
    x, y = np.asarray(minutes, float), np.asarray(deviations, float)
    width, a, b = np.diff(x), y[:-1], y[1:]
    net = np.sum(width * (a + b) / 2)
    positive = 0.0
    for w, ya, yb in zip(width, a, b):
        if ya >= 0 and yb >= 0:
            positive += w * (ya + yb) / 2
        elif ya > 0 > yb:
            positive += w * ya * ya / (2 * (ya - yb))
        elif yb > 0 > ya:
            positive += w * yb * yb / (2 * (yb - ya))
    return float(net), float(positive)


def prepare_connections(raw):
    connections, stats = {}, []
    for key, group in raw.groupby("connection_id", sort=True):
        grouped = group.groupby("t", sort=True)
        g = grouped.agg(glucose=("glucose", "first"), low=("glucose", "min"),
                        high=("glucose", "max"), valid=("valid", "min"), n=("valid", "size"))
        conflict = g.low.ne(g.high) & g.low.notna() & g.high.notna()
        bad = conflict | g.valid.ne(1) | g.glucose.isna()
        g.loc[bad, "glucose"] = np.nan
        times = g.index.to_numpy(float)
        connections[str(key)] = (times, g.glucose.to_numpy(float))
        gaps = np.diff(times) / 60
        stats.append(dict(connection_id=str(key), unique_timepoints=len(g),
                          duplicate_excess_rows=int((g.n - 1).sum()),
                          conflicting_timepoints=int(conflict.sum()),
                          invalid_timepoints=int(bad.sum()),
                          observed_span_days=(times[-1] - times[0]) / 86400 if len(times) else 0,
                          gaps_over_20_minutes=int((gaps > 20).sum()),
                          max_gap_minutes=float(gaps.max()) if len(gaps) else np.nan))
    return connections, stats


def empty(reason, **extra):
    row = {k: np.nan for k in OUTCOMES + ["baseline_glucose", "pre_slope_per_min", "post_points", "max_gap_minutes"]}
    row.update(cgm_valid=False, cgm_reason=reason, connection_id="")
    row.update(extra)
    return row, None


def response(t, glucose, start, cfg):
    rel = (np.asarray(t) - start) / 60
    glucose = np.asarray(glucose)
    if not len(rel):
        return empty("no_connection")
    if start < t[0] + cfg["warmup_minutes"] * 60:
        return empty("warmup_or_before_record")
    lo, hi = np.searchsorted(rel, [-cfg["baseline_minutes"], cfg["post_minutes"]], side="left")
    stop = np.searchsorted(rel, cfg["post_minutes"], side="right")
    pre = (rel >= -cfg["baseline_minutes"]) & (rel < 0)
    if pre.sum() < cfg["baseline_min_points"]:
        return empty("insufficient_baseline")
    if not np.isfinite(glucose[pre]).all():
        return empty("invalid_baseline_reading")
    if -rel[pre][-1] > cfg["baseline_nearest_minutes"]:
        return empty("baseline_too_distant")
    base = float(np.median(glucose[pre]))
    pre_times, pre_values = rel[pre], glucose[pre]
    slope = (pre_values[-1] - pre_values[0]) / (pre_times[-1] - pre_times[0])
    if np.diff(pre_times).max(initial=0) > cfg["max_gap_minutes"]:
        return empty("baseline_gap")
    if not np.isfinite(glucose[lo:stop]).all():
        return empty("invalid_window_reading")
    post = (rel > 0) & (rel <= cfg["post_minutes"])
    if post.sum() < cfg["minimum_post_points"]:
        return empty("insufficient_post_points")
    endpoints = []
    for point in [0, cfg["post_minutes"]]:
        right = np.searchsorted(rel, point)
        if right < len(rel) and abs(rel[right] - point) < 1e-8:
            value = glucose[right]
        else:
            if right == 0 or right == len(rel):
                return empty("unbracketed_endpoint")
            if rel[right] - rel[right - 1] > cfg["max_gap_minutes"]:
                return empty("endpoint_gap")
            if not np.isfinite(glucose[right - 1:right + 1]).all():
                return empty("invalid_endpoint_bracket")
            value = np.interp(point, rel[right - 1:right + 1], glucose[right - 1:right + 1])
        if not np.isfinite(value):
            return empty("invalid_endpoint")
        endpoints.append(value)
    inside = (rel > 0) & (rel < cfg["post_minutes"])
    x = np.r_[0.0, rel[inside], float(cfg["post_minutes"])]
    y = np.r_[endpoints[0], glucose[inside], endpoints[1]]
    max_gap = float(np.diff(x).max())
    if max_gap > cfg["max_gap_minutes"]:
        return empty("post_gap", max_gap_minutes=max_gap)
    net, positive = areas(x, y - base)
    # Peak restricted to ACTUAL sensor observations, including t=0 if observed.
    observed = (rel >= 0) & (rel <= cfg["post_minutes"])
    peak_index = int(np.argmax(glucose[observed]))
    row = dict(cgm_valid=True, cgm_reason="valid", connection_id="",
               baseline_glucose=base, pre_slope_per_min=float(slope),
               post_points=int(post.sum()), max_gap_minutes=max_gap,
               iauc_net_0_120=net, iauc_positive_0_120=positive,
               peak_rise_0_120=float(glucose[observed][peak_index] - base),
               peak_time_minutes=float(rel[observed][peak_index]))
    return row, (x, y, base)


def match_meal(connections, start, cfg):
    overlapping = [(key, t, g) for key, (t, g) in connections.items()
                   if len(t) and t[0] <= start + cfg["post_minutes"] * 60
                   and t[-1] >= start - cfg["baseline_minutes"] * 60]
    if not overlapping:
        return empty("no_connection_window")
    if len(overlapping) > 1:
        return empty("multiple_connections_in_window")
    key, t, g = overlapping[0]
    row, curve = response(t, g, start, cfg)
    row["connection_id"] = key
    return row, curve
