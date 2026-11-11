"""Reconstruct discovery-only, meal-log-anchored CGM windows inside TRE."""

from collections import Counter
from hashlib import sha256
import importlib.util
from importlib.metadata import version
from io import BytesIO
import argparse
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "diet_daily_modification_20261112_v1"
KEY = ["participant_id", "cohort", "research_stage"]
NUTRIENTS = ["calories_kcal_clean", "carbohydrate_g_clean", "lipid_g_clean",
             "protein_g_clean", "dietary_fiber_g_clean"]
TZ = "Asia/Jerusalem"
MINUTE = 60


def digest(path):
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_bytes())


def helper(cfg):
    local = ROOT.parents[1] / "meal_cgm_exploration/hppmeal/cgm.py"
    p = ROOT / "pinned_cgm.py"
    path = p if p.is_file() else local
    content = path.read_bytes()
    if path == local:
        content = content.replace(b"\r\n", b"\n")
    require(sha256(content).hexdigest() == cfg["cgm_helper_sha"], "CGM helper differs from the registered source")
    spec = importlib.util.spec_from_file_location("hpp_pinned_meal_cgm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def base_modules(cfg):
    for filename, key in (("inputs.py", "base_inputs_sha"), ("models.py", "base_models_sha"),
                          ("run.py", "base_runner_sha")):
        require(digest(BASE / filename) == cfg[key], "Pinned daily analysis sources differ")
    sys.path.insert(0, str(BASE))
    try:
        import inputs
        import run
    finally:
        sys.path.pop(0)
    require(Path(inputs.__file__).parent.resolve() == BASE.resolve()
            and Path(run.__file__).parent.resolve() == BASE.resolve(), "Unexpected analysis dependencies")
    return inputs, run


def scoped_read(path, cols, people, inputs):
    groups = people.groupby(KEY[1:], observed=True, sort=False)
    conditions = [[("participant_id", "in", part.participant_id.tolist()),
                   ("cohort", "=", cohort), ("research_stage", "=", stage)]
                  for (cohort, stage), part in groups]
    table = pq.read_table(path, columns=KEY + cols, filters=conditions, use_threads=False, pre_buffer=False)
    frame = inputs.frame(table)
    result = frame.merge(people[KEY + ["input_row"]], on=KEY, validate="many_to_one")
    require(len(result) == len(frame), "Projected input contains an out-of-scope identity")
    return result


def verified_roster(package, cfg, smoke, inputs):
    h = package / "inputs/discovery_v2"
    x = h / "discovery_covariates_food_TRE_ONLY.parquet"
    v = package / "experiments/diet_nested_stability_20261103_v4/results/TRE_ONLY/final_predictions.npz"
    f = package / "experiment2_Effect_Modification/v1/runs_TRE_ONLY" / inputs.EXP2
    reg = f / "registration.json"
    reg_sha = digest(reg)
    require(reg_sha == load(f / "COMPLETE.json")["registration"], "Experiment 2 receipt changed")
    anchor = load(reg)["sources"]
    fingerprints = {str(p): digest(p) for p in (x, v)}
    require(all(anchor.get(p) == sha for p, sha in fingerprints.items()), "Frozen cohort source changed")
    index = inputs.frame(pq.read_table(x, columns=inputs.INDEX, use_threads=False, pre_buffer=False))
    inputs.unique(index, KEY)
    require(len(index) == 4427 and index.final_role.eq("discovery").all()
            and np.array_equal(index.input_row, np.arange(len(index))), "Unexpected discovery index")
    with np.load(BytesIO(v.read_bytes()), allow_pickle=False) as data:
        masks = np.isfinite(data["y"])
    require(masks.shape == (2, len(index)) and np.array_equal(*masks), "Frozen membership differs by repeat")
    roster = index.loc[masks[0]].copy()
    require(len(roster) == 3343 and roster.split_unit_id.nunique() == 3059, "Frozen cohort changed")
    if smoke:
        selected = np.random.default_rng(20261112).choice(roster.index, cfg["smoke_people"], replace=False)
        roster = roster.loc[sorted(selected)]
    return roster, {**fingerprints, str(reg): reg_sha}


def epoch(times):
    arr = times.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    return arr.astype("int64").astype(float) / 1e9


def food_groups(events):
    stamps = pd.to_datetime(events.collection_timestamp, errors="coerce", utc=True).dt.tz_convert(TZ)
    wall = stamps.dt.tz_localize(None)
    local = pd.to_datetime(events.local_timestamp, errors="coerce")
    valid_clock = (local - wall).abs().dt.total_seconds().le(60)
    events = events.assign(
        t=np.where(stamps.notna(), epoch(stamps), np.nan), date=wall.dt.strftime("%Y-%m-%d"),
        clock_agrees=valid_clock,
    )
    return {row: g for row, g in events.groupby("input_row", sort=False)}


def episodes(food, eligible_days, cfg):
    food = food.loc[np.isfinite(food.t)].sort_values("t", kind="stable").reset_index(drop=True)
    if food.empty:
        return []
    tags = np.empty(len(food), int)
    origin = -np.inf
    tag = -1
    for i, value in enumerate(food.t):
        if value - origin > cfg["max_food_episode_span_minutes"] * MINUTE:
            tag += 1
            origin = value
        tags[i] = tag
    entries = list(food.groupby(tags, sort=True))
    result = []
    for j in range(1, len(entries) - 1):
        items = entries[j][1]
        start = float(items.t.iloc[0])
        prior = (start - entries[j - 1][1].t.iloc[-1]) / MINUTE
        next_gap = (entries[j + 1][1].t.iloc[0] - start) / MINUTE
        day = items.date.iloc[0]
        if (prior < cfg["isolation_minutes"] or next_gap <= cfg["isolation_minutes"]
                or day not in eligible_days or items.date.nunique() != 1
                or not items.clock_agrees.all() or not items.qc_primary_eligible.eq(True).all()
                or items.qc_same_food_time_n.gt(1).any() or items.qc_exact_signature_n.gt(1).any()):
            continue
        nutrients = items[NUTRIENTS].to_numpy(float)
        if not (np.isfinite(nutrients).all() and (nutrients >= 0).all()):
            continue
        energy, carbs, fat, protein, fiber = nutrients.sum(axis=0)
        if energy < cfg["minimum_kcal"] or carbs < cfg["minimum_carbohydrate_g"]:
            continue
        kcal = 4 * carbs + 9 * fat + 4 * protein
        wall = pd.Timestamp(start, unit="s", tz="UTC").tz_convert(TZ)
        result.append(dict(start=start, date=day, prior_gap=prior, next_gap=next_gap,
                           local_hour=wall.hour + wall.minute / 60, weekday=wall.day_of_week,
                           energy=energy, carbs=carbs, fat=fat, protein=protein, fiber=fiber,
                           carb_share=100 * 4 * carbs / kcal, protein_share=100 * 4 * protein / kcal))
    return result


def points(entry, person, inputs):
    path = Path(entry["source_path"]).resolve()
    sensor_root = Path("/home/ec2-user/studies/hpp_datasets/cgm/timeseries").resolve()
    require(path.parent == sensor_root and entry["status"] == "processed" and entry["unit"] == "mg/dL",
            "Unverified point source path or units")
    payload = path.read_bytes()
    require(sha256(payload).hexdigest() == entry["source_sha256"], "W09 point file changed")
    data = inputs.frame(pq.read_table(BytesIO(payload), columns=KEY + ["connection_id", "collection_timestamp", "glucose"],
                                     pre_buffer=False, use_threads=False))
    for key in KEY + ["connection_id"]:
        require(data[key].notna().all() and data[key].astype(str).eq(str(person[key])).all(),
                "Sensor identity differs from selected visit connection")
    stamp = pd.to_datetime(data.collection_timestamp, errors="coerce", utc=True)
    a = data.loc[stamp.notna(), ["glucose"]].assign(t=epoch(stamp.loc[stamp.notna()]))
    a = a.drop_duplicates(["t", "glucose"])
    a.loc[a.t.duplicated(keep=False) | ~a.glucose.between(40, 500), "glucose"] = np.nan
    a = a.drop_duplicates("t").sort_values("t")
    return a.t.to_numpy(float), a.glucose.to_numpy(float)


def align(person, food, days, fragment, cfg, inputs, cgm):
    candidates = episodes(food, days, cfg)
    if not candidates:
        return [], Counter({"candidate_episodes": 0, "valid_t0": 0, "valid_all_offsets": 0})
    times, glucose = points(fragment, person, inputs)
    rows = []
    counts = Counter(candidate_episodes=len(candidates), valid_t0=0, valid_all_offsets=0)
    for meal in candidates:
        rows_by_offset = {}
        for offset in cfg["time_offsets_minutes"]:
            rr, _ = cgm.response(times, glucose, meal["start"] + offset * MINUTE, cfg["cgm"])
            rows_by_offset[offset] = rr
        primary = rows_by_offset[0]
        if not primary["cgm_valid"]:
            counts["excluded_" + primary["cgm_reason"]] += 1
            continue
        counts["valid_t0"] += 1
        both = all(r["cgm_valid"] for r in rows_by_offset.values())
        counts["valid_all_offsets"] += both
        result = dict(meal, input_row=int(person.input_row), full_offsets=both)
        for offset, row in rows_by_offset.items():
            tag = f"m{abs(offset)}" if offset < 0 else f"p{offset}"
            result.update({
                f"net_{tag}": row["iauc_net_0_120"] / 120,
                f"positive_{tag}": row["iauc_positive_0_120"] / 120,
                f"peak_{tag}": row["peak_rise_0_120"],
                f"pre_{tag}": row["baseline_glucose"],
                f"trend_{tag}": row["pre_slope_per_min"],
            })
        rows.append(result)
    return rows, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    package = args.package.resolve()
    require((package / "experiments" / ROOT.name).resolve() == ROOT
            and re.fullmatch(r"[0-9a-f]{40}", args.commit)
            and re.fullmatch(r"[A-Za-z0-9_-]+", args.run), "Unexpected release or run name")
    cfg = load(ROOT / "config.json")
    inputs, io = base_modules(cfg)
    cgm, helper_path = helper(cfg)
    roster, provenance = verified_roster(package, cfg, args.smoke, inputs)
    project = package.parent
    shared = project / "analysis/shared"
    events_path = shared / "food_events_clean_TRE_ONLY.parquet"
    visits_path = shared / "cgm_visit_phenotypes_TRE_ONLY.parquet"
    daily_path = package / "inputs/daily/food_days_discovery_TRE_ONLY.parquet"
    w09 = project / "data_qc/w09_cgm_point_qc_20261028/fragment_processing_manifest_TRE_ONLY.json"
    static = [events_path, visits_path, daily_path, w09, helper_path, ROOT / "config.json", ROOT / "prepare.py",
              *(BASE / name for name in ("inputs.py", "run.py", "models.py"))]
    provenance.update({str(p): digest(p) for p in static})
    reg = dict(commit=args.commit, smoke=args.smoke, sources=provenance, roster_people=len(roster),
               python=sys.version.split()[0], versions={n: version(n) for n in ("numpy", "pandas", "pyarrow")},
               time_basis="recorded meal-log proxy, not verified ingestion time")
    path = ROOT / "runs_TRE_ONLY" / args.run
    path.mkdir(parents=True, exist_ok=args.resume)
    lock = path / ".running"
    lock.mkdir()
    try:
        (lock / "pid").write_text(str(os.getpid()), encoding="ascii")
        receipt = path / "registration.json"
        if args.resume:
            require(load(receipt) == reg and not (path / "COMPLETE.json").exists(), "Run changed or is complete")
        else:
            io.write(receipt, reg)
        fingerprint = digest(receipt)
        visits = scoped_read(visits_path, ["connection_id", "point_qc_verified", "unit"], roster, inputs)
        require(visits.input_row.is_unique and visits.connection_id.is_unique
                and visits.point_qc_verified.eq(True).all() and set(visits.unit.dropna()) == {"mg/dL"},
                "Unexpected selected-connection contract")
        visits = visits.set_index("input_row", drop=False).sort_index()
        fragments = {str(entry["connection_id"]): entry for entry in load(w09) if entry.get("status") == "processed"}
        require(set(visits.connection_id.astype(str)).issubset(fragments),
                "A selected connection lacks a verified W09 fragment")
        cols = ["collection_timestamp", "local_timestamp", "qc_primary_eligible",
                "qc_same_food_time_n", "qc_exact_signature_n", *NUTRIENTS]
        event = scoped_read(events_path, cols, roster, inputs)
        foods = food_groups(event)
        days = scoped_read(daily_path, ["date", "energy_eligible"], roster, inputs)
        days = days.loc[days.energy_eligible.eq(True)].copy()
        days["date"] = pd.to_datetime(days.date, errors="raise").dt.strftime("%Y-%m-%d")
        valid_dates = {row: set(g.date) for row, g in days.groupby("input_row", sort=False)}
        io.emit("MEAL_PREP_START", commit=args.commit, roster=len(roster), selected_visits=len(visits),
                scoped_event_rows=len(event), smoke=args.smoke)
        all_rows = []
        total = Counter()
        for start in range(0, len(visits), cfg["batch_size"]):
            part = visits.iloc[start:start + cfg["batch_size"]]
            batch = path / f"batch_{start:04d}_TRE_ONLY.parquet"
            checkpoint = path / f"batch_{start:04d}.json"
            if checkpoint.is_file():
                status = load(checkpoint)
                require(status["registration"] == fingerprint
                        and (not status["rows"] or digest(batch) == status["sha"]), "Invalid saved batch")
            else:
                records, counts = [], Counter()
                for row, person in part.iterrows():
                    if row not in foods or row not in valid_dates:
                        continue
                    meals, diagnostics = align(person, foods[row], valid_dates[row],
                                               fragments[str(person.connection_id)], cfg, inputs, cgm)
                    records.extend(meals)
                    counts.update(diagnostics)
                status = dict(registration=fingerprint, rows=len(records), counts=dict(counts))
                if records:
                    b = BytesIO()
                    pq.write_table(pa.Table.from_pylist(records), b, compression="zstd")
                    bdata = b.getvalue()
                    with batch.open("wb") as f:
                        f.write(bdata)
                    require(digest(batch) == sha256(bdata).hexdigest(), "Batch write verification failed")
                    status["sha"] = sha256(bdata).hexdigest()
                io.write(checkpoint, status)
                io.emit("MEAL_BATCH_DONE", cumulative_visits=start + len(part), window_rows=status["rows"])
            total.update(status["counts"])
            if status["rows"]:
                all_rows.append(pq.read_table(batch, pre_buffer=False, use_threads=False).to_pandas())
        require(bool(all_rows), "No CGM windows satisfy the meal-log quality specification")
        combined = pd.concat(all_rows, ignore_index=True)
        require(not combined.duplicated(["input_row", "start"]).any(), "Duplicate meal-log anchors")
        by_person = combined.groupby("input_row")
        ok = by_person.size().ge(cfg["minimum_meals"]) & by_person.date.nunique().ge(cfg["minimum_days"])
        full = combined.loc[combined.input_row.isin(ok.index[ok])].copy()
        require(not full.empty, "No qualifying people after meal/day minimums")
        table = path / "meal_windows_TRE_ONLY.parquet"
        b = BytesIO()
        pq.write_table(pa.Table.from_pandas(full, preserve_index=False), b, compression="zstd")
        blob = b.getvalue()
        with table.open("wb") as f:
            f.write(blob)
        require(digest(table) == sha256(blob).hexdigest()
                and all(digest(Path(p)) == v for p, v in provenance.items()), "Output or source verification failed")
        summary = dict(people=int(full.input_row.nunique()), meals=len(full),
                       households=int(roster.loc[roster.input_row.isin(full.input_row), "split_unit_id"].nunique()),
                       same_meals_all_offsets=int(full.full_offsets.sum()), counts=dict(total),
                       registration=fingerprint, smoke=args.smoke)
        io.write(path / "summary.json", summary)
        io.write(path / "COMPLETE.json", dict(summary=digest(path / "summary.json"), registration=fingerprint,
                                             windows=sha256(blob).hexdigest()))
        io.emit("MEAL_PREP_COMPLETE", **summary)
    finally:
        (lock / "pid").unlink(missing_ok=True)
        lock.rmdir()


if __name__ == "__main__":
    main()
