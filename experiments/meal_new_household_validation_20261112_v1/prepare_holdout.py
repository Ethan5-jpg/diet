"""Rebuild original meal outcomes in previously unused discovery households."""

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from contracts import (
    FEASIBILITY, KEYS, MEAL, NUTRIENTS, ROOT, align, archived_contract, base_modules,
    check_sources, completed, digest, episodes, finish, food_groups, helper, load,
    require, run_path, scoped_read, validate_args, verified_roster,
)

WINDOWS = "meal_windows_TRE_ONLY.parquet"
WEIGHTS = "frozen_model_TRE_ONLY.npz"


def write_parquet(path, table):
    buffer = BytesIO()
    pq.write_table(table, buffer, compression="zstd")
    blob = buffer.getvalue()
    path.write_bytes(blob)
    value = sha256(blob).hexdigest()
    require(digest(path) == value, "Saved parquet differs")
    return value


def eligible_windows(meals, cfg):
    full = meals.loc[meals.full_offsets].copy()
    counts = full.groupby("input_row").date.agg(["size", "nunique"])
    okay = counts["size"].ge(cfg["minimum_meals"]) & counts["nunique"].ge(cfg["minimum_days"])
    return full.loc[full.input_row.isin(counts.index[okay])].copy()


def candidate_inputs(package, cfg, inputs, original):
    old, sources = verified_roster(package, cfg, False, inputs)
    blind = load(FEASIBILITY)
    require(not blind["new_glucose_outcomes_read"], "Expected outcome-blind meal feasibility")
    check_sources(blind["sources"])
    shared = package.parent / "analysis/shared"
    events = shared / "food_events_clean_TRE_ONLY.parquet"
    visits_path = shared / "cgm_visit_phenotypes_TRE_ONLY.parquet"
    food_days = package / "inputs/daily/food_days_discovery_TRE_ONLY.parquet"
    fragment_manifest = package.parent / "data_qc/w09_cgm_point_qc_20261028/fragment_processing_manifest_TRE_ONLY.json"
    index = package / "inputs/discovery_v2/discovery_covariates_food_TRE_ONLY.parquet"
    paths = (events, visits_path, food_days, fragment_manifest, index, MEAL / "prepare.py", MEAL / "config.json")
    sources.update({str(p): digest(p) for p in paths})
    require(all(original["sources"][str(p)] == sources[str(p)] for p in paths), "A meal input differs from archive")
    people = inputs.frame(pq.read_table(index, columns=inputs.INDEX, pre_buffer=False, use_threads=False))
    require(people.final_role.eq("discovery").all()
            and np.array_equal(people.input_row, np.arange(len(people))), "Unexpected discovery index")
    extra = people.loc[~people.split_unit_id.isin(old.split_unit_id)
                       & np.isfinite(people[["age_calendar_years", "sex_code"]]).all(axis=1)]
    cols = ["connection_id", "point_qc_verified", "unit", "first_observed", "last_observed"]
    visits = scoped_read(visits_path, cols, extra, inputs)
    visits = visits.loc[visits.point_qc_verified & visits.unit.eq("mg/dL")].sort_values("input_row")
    require(visits.input_row.is_unique and visits.connection_id.is_unique, "Unexpected selected connection")
    roster = extra.loc[extra.input_row.isin(visits.input_row)]
    fields = ["collection_timestamp", "local_timestamp", "qc_primary_eligible",
              "qc_same_food_time_n", "qc_exact_signature_n", *NUTRIENTS]
    food = food_groups(scoped_read(events, fields, roster, inputs))
    daily = scoped_read(food_days, ["date", "energy_eligible"], roster, inputs)
    daily = daily.loc[daily.energy_eligible].assign(date=lambda d: pd.to_datetime(d.date).dt.strftime("%Y-%m-%d"))
    dates = {r: set(frame.date) for r, frame in daily.groupby("input_row")}
    before = cfg["cgm"]["baseline_minutes"] - min(cfg["time_offsets_minutes"])
    after = cfg["cgm"]["post_minutes"] + max(cfg["time_offsets_minutes"])
    eligible = {}
    for visit in visits.itertuples(index=False):
        row = visit.input_row
        if row not in food or row not in dates:
            continue
        lower = visit.first_observed.timestamp() + before * 60
        upper = visit.last_observed.timestamp() - after * 60
        candidates = [z for z in episodes(food[row], dates[row], cfg) if lower <= z["start"] <= upper]
        if len(candidates) >= cfg["minimum_meals"] and len({z["date"] for z in candidates}) >= cfg["minimum_days"]:
            eligible[row] = len(candidates)
    roster = roster.loc[roster.input_row.isin(eligible)]
    visits = visits.loc[visits.input_row.isin(eligible)]
    proof = blind["counts"]
    require(len(roster) == proof["outside_original_households_people"]
            and roster.split_unit_id.nunique() == proof["outside_original_households"]
            and sum(eligible.values()) == proof["outside_original_households_meals"],
            "New-household population differs from feasibility")
    fragments = {str(z["connection_id"]): z for z in load(fragment_manifest) if z["status"] == "processed"}
    require(set(visits.connection_id.astype(str)).issubset(fragments), "A verified W09 fragment is missing")
    sources[str(FEASIBILITY)] = digest(FEASIBILITY)
    return roster, visits, food, dates, fragments, sources


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--package", required=True, type=Path)
    cli.add_argument("--commit", required=True)
    cli.add_argument("--run", required=True)
    cli.add_argument("--frozen", required=True)
    cli.add_argument("--smoke", action="store_true")
    cli.add_argument("--resume", action="store_true")
    args = cli.parse_args()
    package = validate_args(args)
    out = run_path(args.run)
    require(out.exists() == args.resume, "Run path and resume flag disagree")
    frozen = run_path(args.frozen)
    training, _, _, frozen_receipt = completed(frozen, WEIGHTS)
    require(training["commit"] == args.commit, "Use the registered model release")
    check_sources(training["sources"])
    _, archived_reg, original = archived_contract()
    cfg = load(MEAL / "config.json")
    inputs, io = base_modules(cfg)
    cgm, helper_path = helper(cfg)
    roster, visits, foods, dates, fragments, sources = candidate_inputs(package, cfg, inputs, original)
    if args.smoke:
        selected = np.random.default_rng(20261112).choice(visits.input_row, cfg["smoke_people"], replace=False)
        visits = visits.loc[visits.input_row.isin(selected)]
    paths = [archived_reg, helper_path, Path(__file__).resolve(), ROOT / "contracts.py", frozen / "COMPLETE.json"]
    sources.update({str(p): digest(p) for p in paths})
    out.mkdir(parents=True, exist_ok=args.resume)
    lock = out / ".running"
    lock.mkdir()
    try:
        (lock / "pid").write_text(str(os.getpid()), encoding="ascii")
        reg = out / "registration.json"
        content = dict(commit=args.commit, smoke=args.smoke, sources=sources, frozen=args.frozen,
                       frozen_receipt=frozen_receipt, people=len(visits), batch_size=cfg["batch_size"])
        if args.resume:
            require(load(reg) == content and not (out / "COMPLETE.json").exists(), "Run changed or is complete")
        else:
            io.write(reg, content)
        fingerprint = digest(reg)
        all_rows, counts = [], Counter()
        for start in range(0, len(visits), cfg["batch_size"]):
            group = visits.iloc[start:start + cfg["batch_size"]]
            batch = out / f"batch_{start:04d}_TRE_ONLY.parquet"
            checkpoint = out / f"batch_{start:04d}.json"
            if checkpoint.exists():
                record = load(checkpoint)
                require(record["registration"] == fingerprint
                        and (not record["rows"] or digest(batch) == record["sha"]), "Saved batch differs")
            else:
                rows, counters = [], Counter()
                for _, visit in group.iterrows():
                    records, diagnostics = align(visit, foods[visit.input_row], dates[visit.input_row],
                                                 fragments[str(visit.connection_id)], cfg, inputs, cgm)
                    rows.extend(records)
                    counters.update(diagnostics)
                record = dict(registration=fingerprint, rows=len(rows), counts=dict(counters))
                if rows:
                    record["sha"] = write_parquet(batch, pa.Table.from_pylist(rows))
                io.write(checkpoint, record)
                io.emit("NEW_MEAL_BATCH", visits=start + len(group), window_rows=len(rows), smoke=args.smoke)
            counts.update(record["counts"])
            if record["rows"]:
                all_rows.append(pq.read_table(batch, pre_buffer=False, use_threads=False).to_pandas())
        require(bool(all_rows), "No qualifying meal windows")
        windows = pd.concat(all_rows, ignore_index=True)
        require(not windows.duplicated(KEYS).any(), "Duplicate meal anchors")
        windows = eligible_windows(windows, cfg)
        require(not windows.empty, "No participants satisfy the original full-offset meal rules")
        table = out / WINDOWS
        write_parquet(table, pa.Table.from_pandas(windows, preserve_index=False))
        check_sources(sources)
        summary = dict(people=int(windows.input_row.nunique()), meals=len(windows),
                       households=int(roster.loc[roster.input_row.isin(windows.input_row), "split_unit_id"].nunique()),
                       diagnostics=dict(counts), batches=len(range(0, len(visits), cfg["batch_size"])))
        finish(out, summary, table, io)
        io.emit("NEW_MEAL_PREP_COMPLETE", smoke=args.smoke, **summary)
    finally:
        (lock / "pid").unlink(missing_ok=True)
        lock.rmdir()


if __name__ == "__main__":
    main()
