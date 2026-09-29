"""Stage two: iterate participant-visits and align meals with CGM."""
import sqlite3

import pandas as pd

from .cgm import match_meal, prepare_connections
from .common import KEYS, require, sha256
from .meals import build_meals
from .report import build_tables, example_plot


def build(run):
    require(run.status.get("audit", {}).get("status") == "PASS", "Successful audit required")
    cache = run.out / "private" / "cache.sqlite"
    require(cache.is_file(), "Cache missing")
    expected = run.status["audit"].get("cache_sha256")
    require(expected and sha256(cache) == expected, "Cache changed since audit")
    con = sqlite3.connect(cache.as_uri() + "?mode=ro", uri=True)
    all_meals, stats, examples, example_ids = [], [], [], []
    example_people = set()
    try:
        keys = con.execute("SELECT DISTINCT participant_id,cohort,research_stage FROM food ORDER BY participant_id,cohort,research_stage").fetchall()
        time_issues = set(con.execute("SELECT DISTINCT participant_id,cohort,research_stage FROM diet_time_issues").fetchall())
        for i, key in enumerate(keys, 1):
            where = "f.participant_id=? AND f.cohort=? AND f.research_stage=?"
            food = pd.read_sql_query("SELECT f.*, CASE WHEN dup.rowhash IS NULL THEN 0 ELSE 1 END AS duplicate_item FROM food f LEFT JOIN food_duplicates dup ON f.rowhash=dup.rowhash WHERE " + where + " ORDER BY f.t", con, params=key)
            meals = build_meals(food, run.c)
            raw = pd.read_sql_query("SELECT connection_id,t,glucose,valid FROM cgm WHERE participant_id=? AND cohort=? AND research_stage=? ORDER BY connection_id,t", con, params=key)
            connections, quality = prepare_connections(raw)
            for item in quality:
                item.update(dict(zip(KEYS, key)))
                t, g = connections[item["connection_id"]]
                dates = pd.to_datetime(t, unit="s", utc=True).tz_convert(run.c["time"]["local_timezone"]).strftime("%Y-%m-%d")
                counts = pd.Series(dates[pd.notna(g)]).value_counts()
                item["days_with_valid_readings"] = len(counts)
                item["days_with_at_least_80pct_nominal_readings"] = int((counts >= 0.8 * 24 * 60 / run.c["cgm"]["sampling_minutes"]).sum())
                stats.append(item)
            outcomes = []
            for record in meals.to_dict("records"):
                outcome, curve = match_meal(connections, record["start"], run.c["cgm"])
                outcomes.append(outcome)
                if (curve is not None and record["isolated_window"] and record["meal_eligible"] and key not in time_issues
                        and key[0] not in example_people and len(examples) < run.c["plots"]["example_meals"]):
                    in_meal = food.t.between(record["start"], record["end"])
                    examples.append(dict(curve=curve, carbs=record["carbohydrate_g"], nova4=record["nova4_pct"], coverage=record["nova_coverage"],
                                         food_minutes=sorted(set((food.loc[in_meal, "t"] - record["start"]) / 60))))
                    example_ids.append(dict(example=len(examples), meal_id=record["meal_id"], participant_id=key[0]))
                    example_people.add(key[0])
            merged = pd.concat([meals.reset_index(drop=True), pd.DataFrame(outcomes)], axis=1)
            merged["diet_time_issue"] = key in time_issues
            merged["analysis_ready"] = merged.meal_eligible & merged.isolated_window & merged.cgm_valid & ~merged.diet_time_issue
            all_meals.append(merged)
            if i % 100 == 0 or i == len(keys):
                print("BUILD person-visits {}/{}".format(i, len(keys)), flush=True)
    finally:
        con.close()
    require(all_meals, "No eating episodes")
    output = pd.concat(all_meals, ignore_index=True)
    require(not output.meal_id.duplicated().any(), "Meal ID collision")
    run.table(output, "02_meals.csv", private=True)
    if example_ids:
        run.table(pd.DataFrame(example_ids), "02_example_identity.csv", private=True)
    flow = build_tables(run, output, stats)
    example_plot(run, examples)
    return dict(status="PASS", meals=len(output), people=output.participant_id.nunique(),
                analysis_ready_meals=int(output.analysis_ready.sum()), flow=flow,
                meals_sha256=sha256(run.out / "private" / "02_meals.csv")), output
