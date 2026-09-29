"""Calendar coverage and exact early/late areas; never infer missing meals."""
import sqlite3
from contextlib import closing
import numpy as np
import pandas as pd
from hppmeal.common import require, read_csv, unique
from hppmeal.cgm import prepare_connections, match_meal, areas
from .inputs import zone

DAY_COLS = ['participant_id', 'connection_id', 'local_date', 'full_day', 'expected_bins', 'valid_bins', 'coverage']
TIME_COLS = ['meal_id', 'meal_zone_ok', 'early_mean', 'late_mean', 'late_minus_early']


def daily_coverage(connections, timezone, cadence=15, warmup=0):
    rows = []
    for connection, (t, g) in connections.items():
        if not len(t):
            continue
        first = pd.to_datetime(t[0], unit='s', utc=True).tz_convert(timezone).date()
        last = pd.to_datetime(t[-1], unit='s', utc=True).tz_convert(timezone).date()
        dates = pd.date_range(first, last + pd.Timedelta(days=1), freq='D')
        bounds = dates.tz_localize(timezone, ambiguous=False, nonexistent='shift_forward').asi8 / 1e9
        for i in range(len(bounds) - 1):
            a, b = bounds[i:i+2]
            expected = int(round((b - a) / (cadence * 60)))
            require(expected > 0, 'Invalid local day duration')
            lo, hi = np.searchsorted(t, [a, b])
            valid = np.isfinite(g[lo:hi]) & (t[lo:hi] >= t[0] + warmup * 60)
            bins = np.floor((t[lo:hi][valid] - a) / (cadence * 60)).astype(int)
            count = len(np.unique(bins))
            rows.append(dict(connection_id=connection, local_date=str(dates[i].date()),
                             full_day=bool(a >= t[0] and b <= t[-1] + cadence * 60),
                             expected_bins=expected, valid_bins=count, coverage=count / expected))
    return pd.DataFrame(rows, columns=DAY_COLS[1:])


def split_curve(curve):
    x, y, base = curve
    require(x[0] == 0 and x[-1] == 120 and np.all(np.diff(x) > 0), 'Invalid 120-minute curve')
    midpoint = np.interp(60, x, y)
    early = areas(np.r_[x[x < 60], 60.], np.r_[y[x < 60], midpoint] - base)[0] / 60
    late = areas(np.r_[60., x[x > 60]], np.r_[midpoint, y[x > 60]] - base)[0] / 60
    return early, late, late - early


def meal_zone_at_start(food, start):
    # CSV round trips can move a large epoch float by a few sub-microseconds.
    near = food.loc[np.abs(food.t.to_numpy(float) - start) <= 1e-5, 'tz']
    zones = {zone(x) for x in near}
    require(len(zones) == 1 and None not in zones, 'Cannot reproduce original meal timezone')
    return next(iter(zones))


def attach_quality(meals, days, timing, diary, timezone):
    d = days.copy()
    d['connections_on_date'] = d.groupby(['participant_id', 'local_date']).connection_id.transform('nunique')
    d = d.merge(diary, on=['participant_id', 'local_date'], how='left', validate='many_to_one', indicator='diary_match')
    d['date_recorded'] = d.diary_match.eq('both') & d.food_items.gt(0) & d.diary_zone.eq(zone(timezone))
    d['reference_day'] = d.full_day & d.connections_on_date.eq(1)
    ref = d.loc[d.reference_day]
    people = ref.groupby('participant_id').agg(reference_days=('local_date', 'size'), logged_reference_days=('date_recorded', 'sum'))
    all_ids = pd.Index(meals.participant_id.unique(), name='participant_id')
    people = people.reindex(all_ids).fillna(0)
    people['diary_person_present'] = people.index.isin(diary.participant_id)
    people['actual_diary_dates'] = diary.groupby('participant_id').local_date.nunique().reindex(all_ids)
    people['logging_fraction'] = people.logged_reference_days.div(people.reference_days.replace(0, np.nan))
    people.loc[~people.diary_person_present, 'logging_fraction'] = np.nan
    people['log80_person'] = people.reference_days.ge(7) & people.logging_fraction.ge(.8)
    reference = d.loc[d.reference_day, ['participant_id', 'local_date', 'connection_id', 'coverage', 'date_recorded']]
    reference = reference.rename(columns={'connection_id': 'day_connection_id', 'coverage': 'day_cgm_coverage'})
    out = meals.merge(timing, on='meal_id', how='left', validate='one_to_one')
    out = out.merge(reference, on=['participant_id', 'local_date'], how='left', validate='many_to_one')
    out = out.merge(people[['log80_person', 'diary_person_present', 'actual_diary_dates']], left_on='participant_id', right_index=True, how='left', validate='many_to_one')
    probes = diary[['participant_id', 'local_date', 'diary_zone']].rename(columns={'diary_zone': 'meal_diary_zone'})
    out = out.merge(probes, on=['participant_id', 'local_date'], how='left', validate='many_to_one', indicator='meal_diary_match')
    out['diary_date_present'] = out.meal_diary_match.eq('both')
    out['aligned'] = out.meal_zone_ok.eq(True) & out.date_recorded.eq(True) & out.connection_id.eq(out.day_connection_id)
    out['cgm90'] = out.aligned & out.day_cgm_coverage.ge(.9)
    out['log80'] = out.aligned & out.log80_person.eq(True)
    out['joint'] = out.cgm90 & out.log80
    out['source_primary'] = True
    return out, d, people.reset_index()


def build(run, meals, paths, diary, progress=print):
    extra = read_csv(paths['meals'], ['meal_id', 'connection_id'], usecols=['meal_id', 'connection_id'])
    unique(extra, ['meal_id'], 'meal connection mapping')
    meals = meals.merge(extra, on='meal_id', how='left', validate='one_to_one')
    require(meals.connection_id.notna().all(), 'Ready meal has missing connection identifier')
    timezone = run.c['time']['local_timezone']
    day_parts, time_rows = [], []
    # Completed, hash-pinned snapshot; immutable prevents WAL/SHM sidecar creation.
    with closing(sqlite3.connect(paths['cache'].as_uri() + '?mode=ro&immutable=1', uri=True)) as con:
        con.execute('PRAGMA query_only=ON')
        require(con.execute('PRAGMA quick_check').fetchone()[0] == 'ok', 'Cache database integrity check failed')
        groups = meals.groupby('participant_id', sort=True)
        for i, (pid, group) in enumerate(groups, 1):
            key = (pid, run.c['cohort'], run.c['research_stage'])
            raw = pd.read_sql_query('SELECT connection_id,t,glucose,valid FROM cgm WHERE participant_id=? AND cohort=? AND research_stage=? ORDER BY connection_id,t', con, params=key)
            connections, _ = prepare_connections(raw)
            day_parts.append(daily_coverage(connections, timezone, run.c['cgm']['sampling_minutes'], run.c['cgm']['warmup_minutes']).assign(participant_id=pid))
            food = pd.read_sql_query('SELECT DISTINCT t,tz FROM food WHERE participant_id=? AND cohort=? AND research_stage=?', con, params=key)
            for m in group.itertuples(index=False):
                result, curve = match_meal(connections, m.start, run.c['cgm'])
                require(result['cgm_valid'] and curve is not None and result['connection_id'] == str(m.connection_id), 'Original valid CGM meal no longer reproduces')
                for field in ['baseline_glucose', 'pre_slope_per_min', 'iauc_net_0_120', 'peak_rise_0_120']:
                    require(np.isclose(result[field], getattr(m, field), rtol=1e-7, atol=1e-5), 'Original curve differs: ' + field)
                early, late, diff = split_curve(curve)
                require(np.isclose((early + late) / 2, m.mean_increment, rtol=1e-7, atol=1e-6), 'Early/late integral does not reproduce original mean')
                meal_zone = meal_zone_at_start(food, m.start)
                local = pd.to_datetime(m.start, unit='s', utc=True).tz_convert(meal_zone)
                require(str(local.date()) == m.local_date, 'Original local meal date differs from cached food timezone')
                time_rows.append(dict(meal_id=m.meal_id, meal_zone_ok=meal_zone == zone(timezone), early_mean=early, late_mean=late, late_minus_early=diff))
            if i % 50 == 0 or i == len(groups):
                progress('COVERAGE and original-curve verification {}/{} people'.format(i, len(groups)))
    days = pd.concat(day_parts, ignore_index=True) if day_parts else pd.DataFrame(columns=DAY_COLS)
    timing = pd.DataFrame(time_rows, columns=TIME_COLS)
    out, day_info, person_info = attach_quality(meals, days, timing, diary, timezone)
    run.table(day_info, '02_daily_coverage.csv', private=True)
    run.table(person_info, '02_person_coverage.csv', private=True)
    run.table(out[['participant_id', 'meal_id', 'local_date', 'connection_id'] + TIME_COLS[1:] + ['diary_person_present', 'diary_date_present', 'day_cgm_coverage', 'aligned', 'cgm90', 'log80', 'joint']], '03_meal_quality_timecourse.csv', private=True)
    metrics = dict(ready_meals=len(out), ready_people=out.participant_id.nunique(), source_curves_reproduced=len(timing),
                   diary_people=len(diary.participant_id.unique()), ready_people_absent_from_diary=int((~person_info.diary_person_present).sum()),
                   full_reference_person_days=int(day_info.reference_day.sum()),
                   reference_dates_with_diary=int((day_info.reference_day & day_info.date_recorded).sum()),
                   overlapping_connection_person_dates=len(day_info.loc[day_info.connections_on_date.gt(1), ['participant_id', 'local_date']].drop_duplicates()))
    run.table(pd.DataFrame([metrics]), '02_coverage_overview.csv')
    run.table(person_info[['reference_days', 'logging_fraction']].quantile([0, .1, .25, .5, .75, .9, 1]).rename_axis('quantile').reset_index(), '02_coverage_quantiles.csv')
    return out, person_info
