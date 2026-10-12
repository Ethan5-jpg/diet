"""Stream raw observations; derive calendar-day QC without assuming 14 complete days."""
import hashlib
import json
import sqlite3
from collections import Counter
from datetime import timedelta

import numpy as np
import pandas as pd

from .common import (KEYS, NUTRIENTS, date_only, epoch, ids, norm, number,
                     parse_times, read_csv, require)
from .inputs import load_mapping


def select_chunk(chunk, config, people, counts):
    require(isinstance(chunk.index, pd.RangeIndex), "Malformed raw CSV row width")
    counts["input_rows"] += len(chunk)
    for col in KEYS:
        chunk[col] = ids(chunk[col]) if col == "participant_id" else norm(chunk[col])
    scope = chunk.cohort.eq(config["cohort"]) & chunk.research_stage.eq(config["research_stage"])
    require(not (scope & chunk.participant_id.isna()).any(), "Missing participant ID in scoped raw input")
    counts["other_cohort_visit"] += int((~scope).sum())
    keep = scope & chunk.participant_id.isin(people)
    counts["outside_selected_connections_people"] += int((scope & ~keep).sum())
    return chunk.loc[keep].copy()


def ingest_diet(con, path, mapping, people, config):
    counts = Counter()
    required = KEYS + ["collection_timestamp", "collection_date", "food_id"] + NUTRIENTS
    tz = config["time"]["local_timezone"]
    for chunk in read_csv(path, required, chunksize=config["chunksize"]):
        d = select_chunk(chunk, config, people, counts)
        if d.empty:
            continue
        times = parse_times(d.collection_timestamp, config["time"]["naive_timezone"])
        unknown = d.loc[times.isna(), ["participant_id"]].drop_duplicates()
        if not unknown.empty:
            unknown.assign(source="diet_time").to_sql("bad_people", con, if_exists="append", index=False)
        counts["invalid_or_unresolved_time"] += int(times.isna().sum())
        d["t"] = epoch(times)
        d["date"] = times.dt.tz_convert(tz).dt.strftime("%Y-%m-%d")
        source_date = date_only(d.collection_date, tz)
        d["bad_day"] = source_date.isna() | source_date.ne(d.date)
        counts["source_date_conflicts"] += int((d.bad_day & times.notna()).sum())
        if "timezone" in d:
            z = norm(d.timezone)
            conflict = z.notna() & z.str.lower().ne(tz.lower())
            d["bad_day"] |= conflict
            counts["non_cohort_timezone_rows"] += int(conflict.sum())
            counts["timezone_missing_defaulted"] += int(z.isna().sum())
        else:
            counts["timezone_missing_defaulted"] += len(d)
        d = d.loc[d.t.notna()].copy()
        d["food_id"] = ids(d.food_id)
        for col in NUTRIENTS:
            n = number(d[col])
            d[col] = n.where(n.ge(0))
            counts[col + "_missing_or_invalid"] += int(d[col].isna().sum())
        d["energy"] = d.calories_kcal
        macros = d[["carbohydrate_g", "lipid_g", "protein_g"]].notna().all(axis=1)
        fallback = d.energy.isna() & macros
        d.loc[fallback, "energy"] = 4 * d.loc[fallback, "carbohydrate_g"] + 9 * d.loc[fallback, "lipid_g"] + 4 * d.loc[fallback, "protein_g"]
        d["formula_energy"] = fallback.astype(int)
        counts["formula_energy_rows"] += int(fallback.sum())
        d["nova"] = d.food_id.map(mapping)
        identity = ["participant_id", "t", "food_id"] + NUTRIENTS
        d["rowhash"] = [hashlib.sha256(json.dumps(row, default=str).encode()).hexdigest()
                        for row in d[identity].fillna("MISSING").itertuples(index=False, name=None)]
        columns = ["participant_id", "date", "t", "food_id", "bad_day", "rowhash", "energy", "formula_energy", "nova"] + NUTRIENTS
        d[columns].to_sql("food", con, if_exists="append", index=False, chunksize=10000)
        counts["cached_rows"] += len(d)
        print("DIET cached {:,} rows".format(counts["cached_rows"]), flush=True)
    require(counts["cached_rows"] > 0, "No diet records matched final CGM people with usable timestamps")
    con.execute("CREATE INDEX food_person ON food(participant_id,date,t)")
    return dict(counts)


def ingest_cgm(con, files, connections, config):
    counts = Counter()
    required = KEYS + ["connection_id", "collection_timestamp", "glucose"]
    for index, path in enumerate(files, 1):
        for chunk in read_csv(path, required, chunksize=config["chunksize"]):
            d = select_chunk(chunk, config, connections, counts)
            if d.empty:
                continue
            d["connection_id"] = ids(d.connection_id)
            require(d.connection_id.notna().all(), "Missing raw CGM connection identity")
            selected = d.connection_id.eq(d.participant_id.map(connections))
            counts["other_connection_rows"] += int((~selected).sum())
            d = d.loc[selected].copy()
            if d.empty:
                # The participant is selected but this file/chunk can contain
                # only their other connections. They are already counted above.
                continue
            times = parse_times(d.collection_timestamp, config["time"]["naive_timezone"])
            bad = times.isna()
            if "timezone" in d:
                z = norm(d.timezone)
                bad |= z.notna() & z.str.lower().ne(config["time"]["local_timezone"].lower())
            if bad.any():
                d.loc[bad, ["participant_id"]].drop_duplicates().assign(source="cgm_time").to_sql("bad_people", con, if_exists="append", index=False)
            counts["unresolved_time_or_timezone"] += int(bad.sum())
            d["t"] = epoch(times)
            d = d.loc[~bad].copy()
            if d.empty:
                continue
            d["glucose"] = number(d.glucose)
            valid = d.glucose.between(config["cgm"]["minimum_glucose"], config["cgm"]["maximum_glucose"])
            counts["invalid_glucose_rows"] += int((~valid).sum())
            d["valid"] = valid.astype(int)
            d[["participant_id", "connection_id", "t", "glucose", "valid"]].to_sql("cgm", con, if_exists="append", index=False, chunksize=10000)
            counts["cached_rows"] += len(d)
        if index % 20 == 0 or index == len(files):
            print("CGM files {}/{}; cached {:,} rows".format(index, len(files), counts["cached_rows"]), flush=True)
    require(counts["cached_rows"] > 0, "No raw CGM records matched the frozen final connections")
    con.execute("CREATE INDEX cgm_person ON cgm(participant_id,t)")
    return dict(counts)


def day_bounds(date, timezone):
    local = pd.Timestamp(date).date()
    left = pd.Timestamp(local, tz=timezone).timestamp()
    right = pd.Timestamp(local + timedelta(days=1), tz=timezone).timestamp()
    return left, right


def cgm_day(t, glucose, date, config):
    """One equally weighted median per observed 15-min bin; never interpolate."""
    cfg = config["cgm"]
    left, right = day_bounds(date, config["time"]["local_timezone"])
    width = cfg["bin_minutes"] * 60
    expected = int(round((right - left) / width))
    t, glucose = np.asarray(t, float), np.asarray(glucose, float)
    inside = (t >= left) & (t < right) & np.isfinite(glucose)
    times, values = t[inside], glucose[inside]
    if len(values):
        bins = np.floor((times - left) / width).astype(int)
        grouped = pd.Series(values).groupby(bins).median()
        vals = grouped.to_numpy(float)
        occupied = grouped.index.to_numpy(int)
        max_gap = float(np.diff(np.r_[left, np.sort(times), right]).max() / 60)
    else:
        vals, occupied, max_gap = np.array([]), np.array([], int), (right - left) / 60
    quarters = np.array_split(np.arange(expected), 4)
    qcoverage = min(float(np.isin(q, occupied).mean()) for q in quarters)
    mean = float(np.mean(vals)) if len(vals) else np.nan
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
    coverage = len(vals) / expected
    row = dict(date=date, day_hours=(right - left) / 3600, expected_bins=expected,
               observed_bins=len(vals), cgm_coverage=coverage, min_quarter_coverage=qcoverage,
               max_gap_minutes=max_gap, daily_mean=mean, daily_sd=sd,
               daily_cv=100 * sd / mean if mean > 0 else np.nan)
    row["cgm_pass"] = bool(coverage >= cfg["minimum_coverage"] and max_gap <= cfg["maximum_gap_minutes"] and qcoverage >= cfg["minimum_quarter_coverage"] and np.isfinite(sd))
    row["cgm_strict"] = bool(row["cgm_pass"] and coverage >= cfg["strict_coverage"] and max_gap <= cfg["strict_gap_minutes"])
    return row


def cgm_person(raw, config, bad_person=False):
    grouped = raw.groupby("t", sort=True)
    g = grouped.agg(glucose=("glucose", "first"), low=("glucose", "min"), high=("glucose", "max"), valid=("valid", "min"), rows=("valid", "size"))
    conflict = g.low.notna() & g.high.notna() & g.low.ne(g.high)
    g.loc[conflict | g.valid.ne(1), "glucose"] = np.nan
    g["date"] = pd.to_datetime(g.index.to_numpy(), unit="s", utc=True).tz_convert(config["time"]["local_timezone"]).strftime("%Y-%m-%d")
    rows = []
    for date, day in g.groupby("date", sort=True):
        row = cgm_day(day.index.to_numpy(float), day.glucose.to_numpy(float), date, config)
        row["cgm_time_issue"] = bool(bad_person)
        if bad_person:
            row["cgm_pass"], row["cgm_strict"] = False, False
        rows.append(row)
    stats = dict(duplicate_excess=int((g.rows - 1).sum()), conflicting_times=int(conflict.sum()), invalid_times=int(g.glucose.isna().sum()))
    return rows, stats


def diet_day(d, config, bad_person=False):
    cfg = config["diet"]
    energy = number(d.energy)
    complete_energy = energy.notna().all()
    total = float(energy.sum()) if complete_energy else np.nan
    classified = d.nova.isin([1, 2, 3, 4])
    class_energy = float(energy[classified].sum())
    nova4_energy = float(energy[d.nova.eq(4)].sum())
    caloric = energy.gt(0)
    times = sorted(d.loc[caloric, "t"].astype(float).unique())
    occasions, occasion_start = 0, None
    for t in times:
        if occasion_start is None or t - occasion_start > cfg["occasion_span_minutes"] * 60:
            occasions += 1
            occasion_start = t
    span = (times[-1] - times[0]) / 3600 if times else 0.0
    duplicate = d.rowhash.duplicated(keep=False).any()
    reasons = []
    if bad_person or d.bad_day.astype(bool).any(): reasons.append("TIME_OR_DATE")
    if duplicate: reasons.append("SUSPECT_DUPLICATE_FOOD")
    if not complete_energy: reasons.append("INCOMPLETE_ENERGY")
    if not np.isfinite(total) or not cfg["minimum_energy_kcal"] <= total <= cfg["maximum_energy_kcal"]: reasons.append("ENERGY_RANGE")
    if occasions < cfg["minimum_occasions"]: reasons.append("FEW_EATING_OCCASIONS")
    if span < cfg["minimum_span_hours"]: reasons.append("SHORT_LOGGING_SPAN")
    coverage = class_energy / total if total > 0 else np.nan
    row = dict(energy_kcal=total, events=len(d), occasions=occasions, logging_span_hours=span,
               diet_pass=not reasons, diet_reasons=";".join(reasons), nova_coverage=coverage,
               nova4_classified_pct=100 * nova4_energy / class_energy if class_energy > 0 and complete_energy else np.nan,
               nova4_total_pct=100 * nova4_energy / total if total > 0 else np.nan,
               formula_energy_events=int(d.formula_energy.sum()), suspect_duplicate=bool(duplicate))
    for source, name in [("dietary_fiber_g", "fiber_g"), ("carbohydrate_g", "carb_g"), ("protein_g", "protein_g"), ("alcohol_g", "alcohol_g")]:
        observed = d[source].notna()
        full = observed[caloric].all()
        # Unknown nutrient in a positive-energy food invalidates its daily sum.
        value = float(d[source].sum()) if full and complete_energy else np.nan
        row[name] = value
        row[name + "_energy_coverage"] = float(energy.where(observed, 0).sum() / total) if total > 0 else np.nan
    row["fiber_density"] = row["fiber_g"] * 1000 / total if total > 0 else np.nan
    row["carb_density"] = row["carb_g"] * 1000 / total if total > 0 else np.nan
    row["protein_density"] = row["protein_g"] * 1000 / total if total > 0 else np.nan
    return row


def construct_panel(out, paths, people, config):
    cache = out / "private" / "cache.sqlite"
    require(not cache.exists(), "Existing cache cannot be overwritten; create a new run")
    con = sqlite3.connect(str(cache))
    con.execute("CREATE TABLE bad_people (participant_id TEXT, source TEXT)")
    connections = dict(zip(people.participant_id, people.connection_id))
    try:
        diet_counts = ingest_diet(con, paths["diet"], load_mapping(paths["nova_mapping"]), connections, config)
        cgm_counts = ingest_cgm(con, paths["cgm_files"], connections, config)
        con.commit()
        bad = pd.read_sql_query("SELECT DISTINCT participant_id,source FROM bad_people", con)
        bad_diet = set(bad.loc[bad.source.eq("diet_time"), "participant_id"])
        bad_cgm = set(bad.loc[bad.source.eq("cgm_time"), "participant_id"])
        day_food, day_cgm, duplicate_counts = [], [], Counter()
        for i, pid in enumerate(people.participant_id, 1):
            raw = pd.read_sql_query("SELECT * FROM food WHERE participant_id=? ORDER BY t", con, params=(pid,))
            for date, group in raw.groupby("date", sort=True):
                day_food.append(dict(participant_id=pid, date=date, **diet_day(group, config, pid in bad_diet)))
            raw = pd.read_sql_query("SELECT t,glucose,valid FROM cgm WHERE participant_id=? ORDER BY t", con, params=(pid,))
            if not raw.empty:
                rows, stats = cgm_person(raw, config, pid in bad_cgm)
                duplicate_counts.update(stats)
                for row in rows:
                    day_cgm.append(dict(participant_id=pid, **row))
            if i % 500 == 0 or i == len(people):
                print("DAILY summaries {}/{} people".format(i, len(people)), flush=True)
        require(day_food and day_cgm, "No daily summaries available")
        foods, cg = pd.DataFrame(day_food), pd.DataFrame(day_cgm)
        foods.to_csv(out / "private" / "diet_days.csv", index=False)
        cg.to_csv(out / "private" / "cgm_days.csv", index=False)
        joined = foods.merge(cg, on=["participant_id", "date"], how="outer", validate="one_to_one", indicator=True)
        audit = dict(diet=diet_counts, cgm=cgm_counts, cgm_duplicates=dict(duplicate_counts),
                     diet_days=len(foods), cgm_days=len(cg), matched_days=int(joined._merge.eq("both").sum()),
                     diet_only_days=int(joined._merge.eq("left_only").sum()), cgm_only_days=int(joined._merge.eq("right_only").sum()),
                     diet_time_issue_people=len(bad_diet), cgm_time_issue_people=len(bad_cgm))
        panel = joined.loc[joined._merge.eq("both")].drop(columns="_merge").merge(people, on="participant_id", how="left", validate="many_to_one")
        # Zero overlap is a feasibility result, not a reason to discard the
        # useful day/identity audit or omit all planned hypothesis rows.
        for col in ["diet_pass", "cgm_pass", "cgm_strict", "primary_eligible", "strict_eligible"]:
            panel[col] = panel[col].fillna(False).astype(bool)
        dates = pd.to_datetime(panel.date)
        panel["weekday"] = dates.dt.dayofweek
        first = pd.to_datetime(cg.date).groupby(cg.participant_id).min()
        panel["observation_day"] = (dates - panel.participant_id.map(first)).dt.days
        panel["day_id"] = panel.participant_id + "|" + panel.connection_id + "|" + panel.date
        panel["sample_minus_day"] = (pd.to_datetime(panel.sample_date, errors="coerce") - dates).dt.days
        panel.to_csv(out / "private" / "daily_panel.csv", index=False)
        return panel, audit
    finally:
        con.close()
