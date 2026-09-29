"""Stream source CSVs into a run-local indexed SQLite cache."""
import glob
import hashlib
import json
import sqlite3
from collections import Counter

import numpy as np
import pandas as pd

from .common import KEYS, NUTRIENTS, epoch, norm, number, parse_times, read_csv, require, unique


def load_nova(run):
    p = run.path("nova_mapping")
    if p is None or not p.is_file():
        return {}, "MISSING_NOVA_MAPPING"
    run.record(p, "nova_mapping")
    df = read_csv(p, ["food_id", "nova_group", "nova_mapping_status"])
    df["food_id"] = norm(df.food_id)
    unique(df, ["food_id"], "NOVA mapping")
    g = number(df.nova_group)
    mapped = df.nova_mapping_status.eq("mapped")
    require(g[mapped].isin([1, 2, 3, 4]).all(), "Invalid mapped NOVA group")
    if "nova_analysis_status" in df:
        mapped &= ~df.nova_analysis_status.fillna("").str.startswith("excluded")
    return dict(zip(df.loc[mapped, "food_id"], g[mapped])), "AVAILABLE"


def select(chunk, c, counts):
    require(isinstance(chunk.index, pd.RangeIndex), "Malformed CSV row width")
    counts["input_rows"] += len(chunk)
    for k in KEYS:
        chunk[k] = norm(chunk[k])
    keep = chunk.cohort.eq(c["cohort"]) & chunk.research_stage.eq(c["research_stage"])
    counts["other_cohort_or_visit"] += int((~keep).sum())
    chunk = chunk.loc[keep].copy()
    counts["selected_rows"] += len(chunk)
    missing = chunk[KEYS].isna().any(axis=1)
    counts["missing_identity"] += int(missing.sum())
    return chunk.loc[~missing].copy()


def ingest_diet(run, con, mapping):
    c = run.c
    p = run.record(run.path("diet"), "diet")
    counts = Counter()
    timestamp = c["time"]["diet_column"]
    required = KEYS + [timestamp, "food_id", "calories_kcal", "carbohydrate_g"]
    for chunk in read_csv(p, required, chunksize=c["ingest"]["chunksize"]):
        d = select(chunk, c, counts)
        d["t"] = epoch(parse_times(d[timestamp], c["time"]["naive_timezone"]))
        counts["invalid_or_naive_time"] += int(d.t.isna().sum())
        issues = d.loc[d.t.isna(), KEYS].drop_duplicates()
        if not issues.empty:
            issues.to_sql("diet_time_issues", con, if_exists="append", index=False)
        d = d.loc[d.t.notna()].copy()
        if d.empty:
            continue
        d["food_id"] = norm(d.food_id)
        counts["missing_food_id"] += int(d.food_id.isna().sum())
        for col in NUTRIENTS:
            if col not in d:
                d[col] = np.nan
            d[col] = number(d[col])
            counts[col + "_negative"] += int(d[col].lt(0).sum())
            d[col] = d[col].where(d[col].ge(0))
            counts[col + "_missing"] += int(d[col].isna().sum())
        d["nova"] = d.food_id.map(mapping)
        if "timezone" in d:
            tz = d.timezone.fillna("").str.strip()
        else:
            tz = pd.Series("", index=d.index)
        counts["timezone_defaulted"] += int(tz.eq("").sum())
        d["tz"] = tz.mask(tz.eq(""), c["time"]["local_timezone"])
        # Detect duplicate source records globally, including across reader chunks.
        identity = KEYS + ["t", "food_id"] + NUTRIENTS
        d["rowhash"] = [hashlib.sha256(json.dumps(row, default=str).encode()).hexdigest()
                        for row in d[identity].fillna("MISSING").itertuples(index=False, name=None)]
        out = d[KEYS + ["t", "food_id", "tz", "nova", "rowhash"] + NUTRIENTS]
        out.to_sql("food", con, if_exists="append", index=False, chunksize=10000)
        counts["cached_rows"] += len(d)
        print("DIET rows cached: {:,}".format(counts["cached_rows"]), flush=True)
    require(counts["cached_rows"] > 0, "No usable diet rows for configured cohort/visit/time semantics")
    con.execute("CREATE INDEX food_key ON food(participant_id,cohort,research_stage,t)")
    con.execute("CREATE TABLE food_duplicates AS SELECT rowhash, COUNT(*) AS n FROM food GROUP BY rowhash HAVING COUNT(*)>1")
    con.execute("CREATE UNIQUE INDEX food_dup_key ON food_duplicates(rowhash)")
    counts["exact_duplicate_excess_rows"] = con.execute("SELECT COALESCE(SUM(n-1),0) FROM food_duplicates").fetchone()[0]
    counts["people"] = con.execute("SELECT COUNT(DISTINCT participant_id) FROM food").fetchone()[0]
    return dict(counts)


def ingest_cgm(run, con):
    c = run.c
    pattern = str(run.path("cgm_glob"))
    files = sorted(glob.glob(pattern))
    require(files, "No CGM files match " + pattern)
    timestamp = c["time"]["cgm_column"]
    required = KEYS + ["connection_id", timestamp, "glucose"]
    counts = Counter()
    file_stats = []
    for i, name in enumerate(files, 1):
        p = run.record(name, "cgm")
        local = Counter()
        for chunk in read_csv(p, required, chunksize=c["ingest"]["chunksize"]):
            d = select(chunk, c, local)
            d["connection_id"] = norm(d.connection_id)
            local["missing_connection"] += int(d.connection_id.isna().sum())
            d["t"] = epoch(parse_times(d[timestamp], c["time"]["naive_timezone"]))
            local["invalid_or_naive_time"] += int(d.t.isna().sum())
            d = d.loc[d.t.notna() & d.connection_id.notna()].copy()
            if d.empty:
                continue
            d["glucose"] = number(d.glucose)
            valid = d.glucose.between(c["cgm"]["plausible_min"], c["cgm"]["plausible_max"])
            local["invalid_glucose_rows"] += int((~valid).sum())
            d["valid"] = valid.astype(int)
            d[KEYS + ["connection_id", "t", "glucose", "valid"]].to_sql(
                "cgm", con, if_exists="append", index=False, chunksize=10000)
            local["cached_rows"] += len(d)
        counts.update(local)
        file_stats.append(dict(file_number=i, **dict(local)))
        if i % 20 == 0 or i == len(files):
            print("CGM files {}/{}; cached rows {:,}".format(i, len(files), counts["cached_rows"]), flush=True)
    require(counts["cached_rows"] > 0, "No timestamped CGM records in configured cohort/visit")
    con.execute("CREATE INDEX cgm_key ON cgm(participant_id,cohort,research_stage,connection_id,t)")
    counts["files"] = len(files)
    counts["people"] = con.execute("SELECT COUNT(DISTINCT participant_id) FROM cgm").fetchone()[0]
    run.table(pd.DataFrame(file_stats), "01_cgm_file_counts.csv")
    return dict(counts)


def audit(run):
    cache = run.out / "private" / "cache.sqlite"
    require(not cache.exists(), "Cache already exists; use a new run directory")
    con = sqlite3.connect(str(cache))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE diet_time_issues (participant_id TEXT, cohort TEXT, research_stage TEXT)")
    try:
        mapping, mapping_status = load_nova(run)
        diet = ingest_diet(run, con, mapping)
        cgm = ingest_cgm(run, con)
        con.commit()
        overlap = con.execute("SELECT COUNT(*) FROM (SELECT DISTINCT participant_id,cohort,research_stage FROM food INTERSECT SELECT DISTINCT participant_id,cohort,research_stage FROM cgm)").fetchone()[0]
        require(overlap > 0, "No common participant/cohort/visit between food and CGM")
        rows = [{"source": source, "metric": key, "value": value}
                for source, data in [("diet", diet), ("cgm", cgm)] for key, value in data.items()]
        rows += [{"source": "join", "metric": "common_person_visits", "value": overlap}]
        run.table(pd.DataFrame(rows), "01_input_audit.csv")
        return dict(status="PASS", mapping_status=mapping_status, common_person_visits=overlap,
                    consumption_time_verified=run.c["time"]["consumption_time_verified"],
                    diet=diet, cgm=cgm)
    finally:
        con.close()
