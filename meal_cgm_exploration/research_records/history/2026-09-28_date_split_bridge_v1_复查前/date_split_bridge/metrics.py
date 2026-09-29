"""Bounded, within-date grid metrics. Never claim equality to legacy iglu exports."""
import numpy as np
import pandas as pd
from hppmeal.common import require

METRICS = ['Mean', 'CV', 'SDw', 'MODD15']
LEGACY = dict(Mean='cgm_mean_z', CV='cgm_cv_z', SDw='cgm_sdw_z', MODD15='cgm_modd_z')


def day_grid(t, g, day, timezone, warmup_end, gap_minutes=45):
    start = pd.Timestamp(day).tz_localize(timezone, ambiguous=False, nonexistent='shift_forward')
    stop = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(timezone, ambiguous=False, nonexistent='shift_forward')
    a, b = start.timestamp(), stop.timestamp()
    target = np.arange(a, b, 900.)
    require(len(target) in [92, 96, 100], 'Unsupported local-day duration')
    mask = (t >= a) & (t < b) & (t >= warmup_end)
    td, gd = t[mask], g[mask]
    values = np.full(len(target), np.nan)
    left_used = np.full(len(target), np.nan); right_used = left_used.copy()
    for i, q in enumerate(target):
        j = int(np.searchsorted(td, q))
        if j < len(td) and abs(td[j] - q) <= 1e-6:
            if np.isfinite(gd[j]):
                values[i] = gd[j]; left_used[i] = right_used[i] = td[j]
        elif 0 < j < len(td) and td[j] - td[j-1] <= gap_minutes * 60 + 1e-6:
            # Explicit invalid readings break a bracket; do not skip them to interpolate.
            if np.isfinite(gd[j-1:j+1]).all():
                values[i] = np.interp(q, td[j-1:j+1], gd[j-1:j+1])
                left_used[i], right_used[i] = td[j-1], td[j]
    finite = np.isfinite(values)
    coverage = float(finite.mean())
    record = dict(local_date=str(day), expected_grid_points=len(target), finite_grid_points=int(finite.sum()),
        grid_coverage=coverage, grid_day_eligible=coverage >= .9, dst_day=len(target) != 96,
        first_support=float(left_used[finite].min()) if finite.any() else np.nan,
        last_support=float(right_used[finite].max()) if finite.any() else np.nan)
    if finite.any():
        require(record['first_support'] >= max(a, warmup_end) and record['last_support'] < b, 'Grid read escaped its date')
    return record, values


def grid_days(t, g, allowed_dates, timezone, warmup_minutes):
    t, g = np.asarray(t, float), np.asarray(g, float)
    require(len(t) == len(g) and len(t) > 0 and np.isfinite(t).all() and (np.diff(t) > 0).all(), 'Invalid CGM time series')
    require(not np.isinf(g).any(), 'Infinite glucose')
    records, values = [], {}
    for day in sorted(set(allowed_dates)):
        row, v = day_grid(t, g, day, timezone, t[0] + warmup_minutes * 60)
        records.append(row)
        if row['grid_day_eligible']: values[day] = v
    return records, values


def summarize_window(days):
    n = len(days)
    row = {k: np.nan for k in METRICS}
    row.update(days=n, finite_points=sum(int(np.isfinite(v).sum()) for v in days.values()),
        MODD15_pairs=0, MODD15_matches=0, dst_days=sum(len(v) != 96 for v in days.values()),
        basic_reason='' if n >= 3 else 'Fewer than3 eligible grid dates', MODD15_reason='')
    diffs = []
    for day, v in sorted(days.items()):
        tomorrow = str((pd.Timestamp(day) + pd.Timedelta(days=1)).date())
        if len(v) != 96 or tomorrow not in days or len(days[tomorrow]) != 96: continue
        other = days[tomorrow]; paired = np.isfinite(v) & np.isfinite(other)
        if paired.sum() >= 48:
            diffs.extend(np.abs(other[paired] - v[paired]).tolist())
            row['MODD15_pairs'] += 1; row['MODD15_matches'] += int(paired.sum())
    if n >= 3:
        values = np.concatenate([v[np.isfinite(v)] for v in days.values()])
        require(len(values) > 1 and (values > 0).all(), 'Invalid physiological grid glucose')
        row['Mean'] = float(values.mean())
        row['CV'] = float(100 * values.std(ddof=1) / values.mean())
        row['SDw'] = float(np.mean([np.std(v[np.isfinite(v)], ddof=1) for v in days.values()]))
        if row['MODD15_pairs'] >= 2: row['MODD15'] = float(np.mean(diffs))
    if not np.isfinite(row['MODD15']): row['MODD15_reason'] = 'Need3 grid dates and2 adjacent non-DST pairs with48 matches'
    return row

