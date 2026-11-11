"""Rebuild exact time-resolved summaries of the archived meal CGM windows."""

import argparse
from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parent
MEAL = ROOT.parent / "meal_carbohydrate_modification_20261112_v1"
sys.path.insert(0, str(MEAL))
try:
    from paired import verified_result
    from prepare import KEY, base_modules, digest, helper, load, points, require, scoped_read
finally:
    sys.path.pop(0)

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


TAGS = {-15: "m15", 0: "p0", 15: "p15"}
EDGES = np.arange(0, 121, 30, dtype=float)
VERIFY_FIELDS = tuple(f"{name}_{tag}" for tag in TAGS.values() for name in ("net", "pre", "peak"))
PAIRED_SHA = "6495e7bf2fe831a971be31a1dbc1ddaf582773c13a04d89dbf7641a478239b0a"


def segment_means(x, y, baseline):
    x, y = np.asarray(x, float), np.asarray(y, float)
    require(x[0] == 0 and x[-1] == 120 and np.all(np.diff(x) > 0)
            and np.isfinite(y).all() and np.isfinite(baseline), "Invalid accepted glucose curve")
    out = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        knots = np.r_[lo, x[(x > lo) & (x < hi)], hi]
        values = np.interp(knots, x, y) - baseline
        out.append(float(np.diff(knots) @ ((values[:-1] + values[1:]) / 2) / (hi - lo)))
    return out


def window_segments(rows, times, glucose, cfg, cgm):
    records = []
    error = 0.
    for meal in rows.itertuples(index=False):
        row = dict(input_row=int(meal.input_row), start=float(meal.start))
        for offset, tag in TAGS.items():
            actual, trace = cgm.response(times, glucose, meal.start + offset * 60, cfg)
            require(actual["cgm_valid"] and trace is not None, "Previously accepted CGM window failed replay")
            segments = segment_means(*trace)
            anchors = [getattr(meal, f"{name}_{tag}") for name in ("net", "pre", "peak")]
            observed = [float(np.mean(segments)), actual["baseline_glucose"], actual["peak_rise_0_120"]]
            errors = np.abs(np.asarray(anchors) - observed)
            require(errors.max() <= 1e-8, "Rebuilt glucose phases disagree with the archived outcome")
            error = max(error, float(errors.max()))
            row.update({f"mean_{tag}_{i}": value for i, value in enumerate(segments)})
        records.append(row)
    return records, error


def write_table(path, table):
    buffer = BytesIO()
    pq.write_table(table, buffer, compression="zstd")
    data = buffer.getvalue()
    path.write_bytes(data)
    value = sha256(data).hexdigest()
    require(digest(path) == value, "Saved trajectory bytes differ")
    return value


def load_archived(package):
    require(digest(MEAL / "paired.py") == PAIRED_SHA, "Pinned replay code differs")
    archive = MEAL / "runs_TRE_ONLY/models-1"
    fitted, summary = verified_result(archive)
    prepared = MEAL / "runs_TRE_ONLY" / fitted["prepared"]
    registration = prepared / "registration.json"
    receipt = prepared / "COMPLETE.json"
    windows = prepared / "meal_windows_TRE_ONLY.parquet"
    m = load(receipt)
    require(not load(registration)["smoke"] and digest(registration) == m["registration"]
            and digest(windows) == m["windows"], "Archived meal windows differ")
    src = load(registration)["sources"]
    cfg = load(MEAL / "config.json")
    require(tuple(cfg["time_offsets_minutes"]) == tuple(TAGS) and cfg["cgm"]["post_minutes"] == 120,
            "Archived time grid differs")
    cgm, helper_path = helper(cfg)
    inputs, io = base_modules(cfg)
    p = package.parent
    visits_path = p / "analysis/shared/cgm_visit_phenotypes_TRE_ONLY.parquet"
    w09 = p / "data_qc/w09_cgm_point_qc_20261028/fragment_processing_manifest_TRE_ONLY.json"
    paths = [MEAL / "config.json", MEAL / "prepare.py", helper_path, visits_path, w09]
    source = {str(p): digest(p) for p in paths}
    require(all(src.get(p) == v for p, v in source.items()), "An archived reconstruction source changed")
    idx = package / "inputs/discovery_v2/discovery_covariates_food_TRE_ONLY.parquet"
    source[str(idx)] = digest(idx)
    require(fitted["sources"][str(idx)] == source[str(idx)], "Frozen participant index changed")
    source.update({str(p): digest(p) for p in (windows, registration, receipt, archive / "summary.json",
                                             MEAL / "paired.py", Path(__file__).resolve())})
    cols = ["input_row", "start", "date", "full_offsets", *VERIFY_FIELDS]
    meals = pq.read_table(windows, columns=cols, use_threads=False, pre_buffer=False).to_pandas()
    meals = meals.loc[meals.full_offsets].copy()
    counts = meals.groupby("input_row").agg(meals=("start","size"),days=("date","nunique"))
    keep = counts.meals.ge(cfg["minimum_meals"]) & counts.days.ge(cfg["minimum_days"])
    meals = meals.loc[meals.input_row.isin(counts.index[keep])].copy()
    require(len(meals) == summary["flow"]["meals"] and meals.input_row.nunique() == summary["flow"]["people"]
            and not meals.duplicated(["input_row","start"]).any(), "Archived model population changed")
    index = inputs.frame(pq.read_table(idx, columns=KEY + ["input_row","final_role"],
                                       pre_buffer=False, use_threads=False))
    require(index.final_role.eq("discovery").all() and index.input_row.is_unique, "Unexpected discovery index")
    return meals, index, inputs, io, cfg, cgm, visits_path, w09, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    package = args.package.resolve()
    require(package / "experiments" / ROOT.name == ROOT and re.fullmatch(r"[a-zA-Z0-9_-]+", args.run)
            and re.fullmatch(r"[0-9a-f]{40}", args.commit), "Unexpected release or run path")
    out = ROOT / "runs_TRE_ONLY" / args.run
    require(out.exists() == args.resume, "Run existence does not match the requested mode")
    meals, index, inputs, io, cfg, cgm, visits_path, w09, sources = load_archived(package)
    people = np.sort(meals.input_row.unique())
    if args.smoke:
        people = people[:cfg["smoke_people"]]
        meals = meals.loc[meals.input_row.isin(people)].copy()
    by_person = dict(tuple(meals.groupby("input_row")))
    roster = index.loc[index.input_row.isin(people)]
    require(len(roster) == len(people), "Archived participant is missing from the discovery index")
    visits = scoped_read(visits_path, ["connection_id","point_qc_verified","unit"], roster, inputs)
    require(visits.input_row.is_unique and visits.connection_id.is_unique
            and visits.point_qc_verified.eq(True).all() and set(visits.unit) == {"mg/dL"}
            and set(visits.input_row) == set(people), "Archived connection contract changed")
    visits = visits.sort_values("input_row").reset_index(drop=True)
    fragments = {str(v["connection_id"]): v for v in load(w09) if v.get("status") == "processed"}
    require(set(visits.connection_id.astype(str)).issubset(fragments), "A verified sensor file is missing")
    out.mkdir(parents=True, exist_ok=args.resume)
    lock = out / ".running"
    lock.mkdir()
    try:
        (lock / "pid").write_text(str(os.getpid()), encoding="ascii")
        reg = dict(commit=args.commit, smoke=args.smoke, sources=sources, people=len(people), meals=len(meals),
                   offsets=list(TAGS), edges=EDGES.tolist(), batch_size=cfg["batch_size"])
        reg_path = out / "registration.json"
        if args.resume:
            require(load(reg_path) == reg and not (out / "COMPLETE.json").exists(), "Run changed or is already complete")
        else:
            io.write(reg_path, reg)
        fingerprint = digest(reg_path)
        io.emit("CURVES_START", people=len(people), meals=len(meals), smoke=args.smoke)
        all_parts, maximum = [], 0.
        for start in range(0, len(visits), cfg["batch_size"]):
            group = visits.iloc[start:start + cfg["batch_size"]]
            file = out / f"batch_{start:04d}_TRE_ONLY.parquet"
            checkpoint = out / f"batch_{start:04d}.json"
            if checkpoint.exists():
                record = load(checkpoint)
                require(record["registration"] == fingerprint and record["sha"] == digest(file), "Invalid saved batch")
            else:
                values, max_error = [], 0.
                for _, person in group.iterrows():
                    times, glucose = points(fragments[str(person.connection_id)], person, inputs)
                    rows, error = window_segments(by_person[person.input_row], times, glucose, cfg["cgm"], cgm)
                    values.extend(rows)
                    max_error = max(max_error, error)
                record = dict(registration=fingerprint, rows=len(values), maximum_error=max_error,
                              sha=write_table(file, pa.Table.from_pylist(values)))
                io.write(checkpoint, record)
                io.emit("CURVES_BATCH", start=start, rows=record["rows"], error=record["maximum_error"])
            table = pq.read_table(file, use_threads=False, pre_buffer=False)
            require(table.num_rows == record["rows"], "Checkpoint row count differs")
            all_parts.append(table)
            maximum = max(maximum, record["maximum_error"])
        result = pa.concat_tables(all_parts)
        keys = ["input_row", "start"]
        actual = result.select(keys).to_pandas().sort_values(keys).reset_index(drop=True)
        expected = meals[keys].sort_values(keys).reset_index(drop=True)
        require(np.array_equal(actual.to_numpy(), expected.to_numpy()), "Meal keys differ from the archived input")
        table_sha = write_table(out / "phase_windows_TRE_ONLY.parquet", result)
        require(all(digest(Path(p)) == v for p, v in sources.items()), "An analysis source changed")
        summary = dict(people=len(people), meals=result.num_rows, batches=len(all_parts), maximum_error=maximum)
        io.write(out / "summary.json", summary)
        io.write(out / "COMPLETE.json", dict(registration=fingerprint, summary=digest(out / "summary.json"), phases=table_sha))
        io.emit("CURVES_COMPLETE", **summary, smoke=args.smoke)
    finally:
        (lock / "pid").unlink(missing_ok=True)
        lock.rmdir()


if __name__ == "__main__":
    main()
