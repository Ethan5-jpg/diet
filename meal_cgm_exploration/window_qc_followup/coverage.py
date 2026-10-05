"""Fixed denominator coverage; no interpolation and no sensor stitching."""
import sqlite3
from contextlib import closing

import numpy as np
import pandas as pd

from hppmeal.common import KEYS, require, unique
from quality_timecourse.inputs import zone

CADENCE = 900
EXPECTED = 1344
DURATION = EXPECTED * CADENCE
TOLERANCE = 180


def longest_run(mask):
    a = np.r_[False, np.asarray(mask, bool), False].astype(int)
    edges = np.flatnonzero(np.diff(a))
    return int((edges[1::2] - edges[::2]).max()) if len(edges) else 0


def slots(raw, start, low=40, high=500):
    """Duplicates at exact timestamps and within slots cannot inflate coverage."""
    t = raw.t.to_numpy(float)
    g = pd.to_numeric(raw.glucose, errors="coerce").to_numpy(float)
    flags = pd.to_numeric(raw.valid, errors="coerce").to_numpy(float)
    require(np.isfinite(t).all() and np.isfinite(start), "Invalid cached/declared timestamp")
    # Sensor activation need not coincide with the device's 15-minute clock.
    # Infer that clock's phase from the first timestamp, never from glucose or
    # the best coverage; map its regular readings into the fixed planned bins.
    phase = float(t.min()) if len(t) else start
    relative = (t - phase) / CADENCE
    index = np.floor((phase - start) / CADENCE).astype(np.int64) + np.rint(relative).astype(np.int64)
    inside = (t >= start) & (t < start + DURATION)
    on_grid = np.abs(relative - np.rint(relative)) * CADENCE <= TOLERANCE
    assignable = inside & on_grid & (index >= 0) & (index < EXPECTED)
    d = pd.DataFrame(dict(slot=index[assignable], glucose=g[assignable],
                         valid=((flags == 1) & np.isfinite(g) & (g >= low) & (g <= high))[assignable]))
    seen = np.zeros(EXPECTED, bool)
    good = np.zeros(EXPECTED, bool)
    conflict_slots = 0
    if len(d):
        groups = d.groupby("slot").agg(minimum=("glucose", "min"), maximum=("glucose", "max"),
                                      all_valid=("valid", "all"), count=("valid", "size"))
        conflict = groups.minimum.ne(groups.maximum) & groups.minimum.notna() & groups.maximum.notna()
        seen[groups.index.to_numpy(int)] = True
        good[groups.index.to_numpy(int)] = (groups.all_valid & ~conflict).to_numpy(bool)
        conflict_slots = int(conflict.sum())
    missing = ~good
    nz = np.flatnonzero(good)
    leading = int(nz[0]) if len(nz) else 0
    trailing = int(EXPECTED - 1 - nz[-1]) if len(nz) else 0
    internal = int(missing[nz[0]:nz[-1]+1].sum()) if len(nz) else 0
    whole_window = 0 if len(nz) else EXPECTED
    row = dict(expected_slots=EXPECTED, raw_rows=len(raw), raw_present_slots=int(seen.sum()),
        valid_slots=int(good.sum()), absent_slots=int((~seen).sum()),
        invalid_or_conflict_slots=int((seen & ~good).sum()), conflicting_slots=conflict_slots,
        missing_slots=int(missing.sum()), cgm_missing_rate=float(missing.mean()),
        leading_missing_slots=leading, trailing_missing_slots=trailing, internal_missing_slots=internal,
        whole_window_missing_slots=whole_window,
        longest_missing_minutes=longest_run(missing) * 15,
        off_grid_or_unassignable_rows=int((inside & ~assignable).sum()),
        rows_before_window=int((t < start).sum()), rows_after_window=int((t >= start + DURATION).sum()),
        duplicate_excess_rows=int(raw.duplicated("t").sum()),
        observed_span_days=float((t.max() - t.min()) / 86400) if len(t) else np.nan,
        cadence_phase_seconds=float((phase - start) % CADENCE))
    require(row["missing_slots"] == row["absent_slots"] + row["invalid_or_conflict_slots"], "Coverage identity failed")
    require(row["missing_slots"] == leading + trailing + internal + whole_window, "Missing-location partition failed")
    return row, good


def diet_days(food, diary, start, timezone, declared_date=None):
    """Presence of dated logs; never label this as a true missing-meal rate."""
    local_start = pd.to_datetime(start, unit="s", utc=True).tz_convert(timezone)
    first = pd.Timestamp(declared_date) if declared_date is not None and pd.notna(declared_date) else pd.Timestamp(local_start.date())
    dates = [str((first + pd.Timedelta(days=i)).date()) for i in range(14)]
    expected_zone = zone(timezone)
    f = food.copy()
    if len(f):
        f["local_date"] = pd.to_datetime(f.t, unit="s", utc=True).dt.tz_convert(timezone).dt.strftime("%Y-%m-%d")
        f["zone_ok"] = f.tz.map(zone).eq(expected_zone)
        events = f.groupby("local_date").agg(event_count=("t", "size"), zone_ok=("zone_ok", "all"))
    else:
        events = pd.DataFrame(columns=["event_count", "zone_ok"])
    daily = diary.set_index("local_date")
    daily = daily.reindex(dates)
    d = pd.DataFrame(dict(local_date=dates)).set_index("local_date")
    d["event_count"] = events.event_count.reindex(dates).fillna(0).astype(int)
    d["event_zone_ok"] = events.zone_ok.reindex(dates).eq(True)
    d["metadata_present"] = daily.food_items.notna()
    d["metadata_items"] = daily.food_items
    d["metadata_zone_ok"] = daily.diary_zone.eq(expected_zone)
    d["recorded"] = d.event_count.gt(0) & d.event_zone_ok & d.metadata_items.gt(0) & d.metadata_zone_ok
    d["source_presence_disagreement"] = d.event_count.gt(0).ne(d.metadata_items.gt(0))
    d["timezone_problem"] = (d.event_count.gt(0) & ~d.event_zone_ok) | (d.metadata_present & ~d.metadata_zone_ok)
    bounds = pd.date_range(first, first + pd.Timedelta(days=14), freq="D").tz_localize(
        timezone, ambiguous=False, nonexistent="shift_forward").asi8 / 1e9
    d["fully_within_cgm_window"] = (bounds[:-1] >= start) & (bounds[1:] <= start + DURATION)
    nfull = int(d.fully_within_cgm_window.sum())
    nfull_recorded = int((d.fully_within_cgm_window & d.recorded).sum())
    row = dict(diet_first_date=dates[0], diet_last_date=dates[-1], diet_expected_days=14,
        diet_recorded_days=int(d.recorded.sum()), diet_missing_days=int((~d.recorded).sum()),
        diet_missing_day_rate=float((~d.recorded).mean()),
        diet_metadata_absent_days=int((~d.metadata_present).sum()),
        diet_source_disagreement_days=int(d.source_presence_disagreement.sum()),
        diet_timezone_problem_days=int(d.timezone_problem.sum()),
        diet_first_day_partial_in_cgm=bool(bounds[0] < start),
        diet_full_cgm_days=nfull, diet_recorded_full_cgm_days=nfull_recorded,
        diet_full_day_missing_rate=1 - nfull_recorded / nfull if nfull else np.nan,
        diet_window_source="DECLARED_DATE" if declared_date is not None and pd.notna(declared_date) else "CGM_START_LOCAL_DATE_PROXY")
    return row, d.reset_index()


def blank_quality(pid, conn, config):
    return dict(participant_id=pid, connection_id=conn, cohort=config["cohort"], research_stage=config["research_stage"],
        status="NO_CGM_RECORDS_OR_WINDOW", anchor_source="UNKNOWN", window_start=np.nan, window_end=np.nan,
        expected_slots=EXPECTED, raw_rows=0, valid_slots=np.nan, missing_slots=np.nan, cgm_missing_rate=np.nan,
        diet_expected_days=14, diet_recorded_days=np.nan, diet_missing_days=np.nan, diet_missing_day_rate=np.nan,
        diet_first_date=None, diet_last_date=None, unresolved_diet_time=False, connection_unambiguous=False)


def build(run, paths, diary, plan, metadata, progress=print):
    config = run.c
    scope = (config["cohort"], config["research_stage"])
    declared = {} if plan is None else {(r.participant_id, r.connection_id): r for r in plan.itertuples(index=False)}
    diary_groups = {pid: frame for pid, frame in diary.groupby("participant_id", sort=False)}
    empty_diary = diary.iloc[:0]
    conn_rows, daily_rows = [], []
    with closing(sqlite3.connect(paths["cache"].as_uri() + "?mode=ro&immutable=1", uri=True)) as con:
        con.execute("PRAGMA query_only=ON")
        require(con.execute("PRAGMA quick_check").fetchone()[0] == "ok", "Frozen cache integrity failed")
        pairs = con.execute("SELECT DISTINCT participant_id,connection_id FROM cgm WHERE cohort=? AND research_stage=?", scope).fetchall()
        food_ids = {r[0] for r in con.execute("SELECT DISTINCT participant_id FROM food WHERE cohort=? AND research_stage=?", scope)}
        issue_ids = {r[0] for r in con.execute("SELECT DISTINCT participant_id FROM diet_time_issues WHERE cohort=? AND research_stage=?", scope)}
        from_cache = {}
        for pid, cid in pairs:
            from_cache.setdefault(str(pid), set()).add(str(cid))
        meta_connections = {}
        for r in metadata.itertuples(index=False):
            meta_connections.setdefault(r.participant_id, set()).add(r.connection_id)
        planned_connections = {}
        for pid, cid in declared:
            planned_connections.setdefault(pid, set()).add(cid)
        participants = sorted(set(from_cache) | food_ids | set(diary.participant_id) | set(meta_connections) | set(planned_connections) | issue_ids)
        require(participants, "No target-visit participants available for coverage audit")
        for i, pid in enumerate(participants, 1):
            key = (pid,) + scope
            raw = pd.read_sql_query("SELECT connection_id,t,glucose,valid FROM cgm WHERE participant_id=? AND cohort=? AND research_stage=? ORDER BY connection_id,t", con, params=key)
            food = pd.read_sql_query("SELECT t,tz FROM food WHERE participant_id=? AND cohort=? AND research_stage=?", con, params=key)
            connections = sorted(from_cache.get(pid, set()) | meta_connections.get(pid, set()) | planned_connections.get(pid, set()))
            for cid in connections or [""]:
                r = blank_quality(pid, cid, config)
                r["unresolved_diet_time"] = pid in issue_ids
                r["connection_unambiguous"] = len(connections) == 1
                series = raw.loc[raw.connection_id.eq(cid)].copy()
                r["raw_rows"] = len(series)
                supplied = declared.get((pid, cid))
                if plan is not None and supplied is None:
                    r["status"] = "DECLARED_WINDOW_UNMATCHED"
                elif len(series) or supplied is not None:
                    start = float(supplied.window_start) if supplied is not None else float(series.t.min())
                    r.update(window_start=start, window_end=start+DURATION,
                             status="AUDITED", anchor_source="DECLARED_PLAN" if supplied is not None else "FIRST_RECORDED_POINT_PROXY")
                    cg, _ = slots(series, start, config["cgm"]["plausible_min"], config["cgm"]["plausible_max"])
                    diet_date = supplied.diet_start_date if supplied is not None else None
                    dt, days = diet_days(food, diary_groups.get(pid, empty_diary), start,
                                         config["time"]["local_timezone"], diet_date)
                    r.update(cg)
                    r.update(dt)
                    daily_rows.append(days.assign(participant_id=pid, connection_id=cid))
                conn_rows.append(r)
            if i % 100 == 0 or i == len(participants):
                progress("TWO_WEEK_QC {}/{} participants".format(i, len(participants)))
    quality = pd.DataFrame(conn_rows)
    unique(quality, ["participant_id", "connection_id"], "connection quality")
    day = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame(columns=["participant_id", "connection_id", "local_date", "recorded"])
    run.table(quality, "connection_quality.csv", private=True)
    run.table(day, "diet_window_days.csv", private=True)
    audit = pd.DataFrame([dict(people=len(participants), connections=len(quality),
        audited_connections=int(quality.status.eq("AUDITED").sum()),
        connections_without_auditable_window=int(quality.status.ne("AUDITED").sum()),
        connections_with_records_but_unmatched_window=int((quality.raw_rows.gt(0) & quality.status.ne("AUDITED")).sum()),
        unambiguous_people=int(quality.loc[quality.connection_unambiguous].participant_id.nunique()),
        no_cgm_people=int(quality.groupby("participant_id").raw_rows.sum().eq(0).sum()),
        multiple_connection_people=int(quality.loc[~quality.connection_unambiguous & quality.connection_id.ne("")].participant_id.nunique()),
        known_plan_connections=int(quality.anchor_source.eq("DECLARED_PLAN").sum()),
        first_record_proxy_connections=int(quality.anchor_source.eq("FIRST_RECORDED_POINT_PROXY").sum()),
        unresolved_diet_time_people=len(issue_ids),
        population_boundary="Frozen target-visit cache + scoped diary + usable CGM metadata + optional declared plan; not all enrolled HPP participants")])
    run.table(audit, "01_coverage_population.csv")
    audited = quality.loc[quality.status.eq("AUDITED")]
    extra = dict(audited_connections_with_any_cgm_missing=int(audited.cgm_missing_rate.gt(0).sum()),
        audited_connections_with_zero_cgm_missing=int(audited.cgm_missing_rate.eq(0).sum()),
        audited_people_with_any_cgm_missing=int(audited.loc[audited.cgm_missing_rate.gt(0)].participant_id.nunique()),
        audited_connections_with_all_14_diet_days=int(audited.diet_recorded_days.eq(14).sum()))
    for field, value in extra.items():
        audit[field] = value
    run.table(audit, "01_coverage_population.csv")
    hist = []
    for field in ["cgm_missing_rate", "diet_missing_day_rate"]:
        values = audited[field]
        bins = [("zero", values.eq(0)), ("above0_below5pct", values.gt(0) & values.lt(.05)),
                ("5_to_below10pct", values.ge(.05) & values.lt(.10)),
                ("10_to_below20pct", values.ge(.10) & values.lt(.20)),
                ("20pct_or_more", values.ge(.20))]
        for label, mask in bins:
            hist.append(dict(metric=field, interval=label, connections=int(mask.sum()),
                             participants=int(audited.loc[mask].participant_id.nunique())))
    run.table(pd.DataFrame(hist), "01_missing_rate_distribution.csv")
    numeric = [c for c in ["cgm_missing_rate", "diet_missing_day_rate", "valid_slots", "longest_missing_minutes",
                           "trailing_missing_slots", "observed_span_days", "diet_full_day_missing_rate"] if c in quality]
    run.table(audited[numeric].quantile([0, .05, .1, .25, .5, .75, .9, .95, 1]).rename_axis("quantile").reset_index(), "01_coverage_quantiles.csv")
    return quality


def attach(meals, quality):
    fields = ["participant_id", "connection_id", "status", "anchor_source", "window_start", "window_end",
              "cgm_missing_rate", "diet_missing_day_rate", "diet_first_date", "diet_last_date",
              "unresolved_diet_time", "connection_unambiguous"]
    out = meals.merge(quality[fields], on=["participant_id", "connection_id"], how="left", validate="many_to_one")
    out["quality_audited"] = out.status.eq("AUDITED") & out.connection_unambiguous.eq(True) & out.unresolved_diet_time.eq(False)
    meal_date = pd.to_datetime(out.local_date, errors="coerce")
    diet_first = pd.to_datetime(out.diet_first_date, errors="coerce")
    diet_last = pd.to_datetime(out.diet_last_date, errors="coerce")
    out["inside_window"] = (out.start.ge(out.window_start + 30*60) & out.start.lt(out.window_end - 120*60)
                            & meal_date.ge(diet_first) & meal_date.le(diet_last))
    return out


def quality_mask(frame, cgm_threshold, diet_threshold, meal=False):
    require(0 < cgm_threshold <= 1 and 0 < diet_threshold <= 1, "Invalid quality thresholds")
    if meal:
        valid = frame.quality_audited & frame.inside_window
    else:
        valid = frame.status.eq("AUDITED") & frame.connection_unambiguous.eq(True) & frame.unresolved_diet_time.eq(False)
    return valid & frame.cgm_missing_rate.lt(cgm_threshold) & frame.diet_missing_day_rate.lt(diet_threshold)
