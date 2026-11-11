"""Assess classification and repeated-meal support without reading CGM values."""

import argparse
import os
from pathlib import Path
import re
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from processing import CARBS, TOTALS, coverage, nearby_pair_count, reconstruct, repeated_support

ROOT = Path(__file__).resolve().parent
MEAL = ROOT.parent / "meal_carbohydrate_modification_20261112_v1"
sys.path.insert(0, str(MEAL))
try:
    from paired import verified_result
    from prepare import KEY, NUTRIENTS, base_modules, digest, food_groups, load, require, scoped_read
    from time_transfer import assign_periods
finally:
    sys.path.pop(0)

PINS = {
    MEAL / "paired.py": "6495e7bf2fe831a971be31a1dbc1ddaf582773c13a04d89dbf7641a478239b0a",
    MEAL / "time_transfer.py": "c9295f753b4f74e9f6cade505bd35c248e6c7fdbf89a6d16263148411b0b20a8",
}


def projected_meals(package):
    require(all(digest(p) == sha for p, sha in PINS.items()), "Pinned archival helpers changed")
    cfg = load(MEAL / "config.json")
    inputs, io = base_modules(cfg)
    archive = MEAL / "runs_TRE_ONLY/models-1"
    model, summary = verified_result(archive)
    prepared = MEAL / "runs_TRE_ONLY" / model["prepared"]
    prep = prepared / "registration.json"
    window = prepared / "meal_windows_TRE_ONLY.parquet"
    receipt = load(prepared / "COMPLETE.json")
    require(not load(prep)["smoke"] and digest(prep) == receipt["registration"]
            and digest(window) == receipt["windows"], "Meal archive changed")
    cols = ["input_row", "start", "date", "local_hour", "full_offsets", *TOTALS]
    meals = pq.read_table(window, columns=cols, use_threads=False, pre_buffer=False).to_pandas()
    meals = meals.loc[meals.full_offsets].copy()
    counts = meals.groupby("input_row").agg(meals=("start", "size"), days=("date", "nunique"))
    keep = counts.meals.ge(cfg["minimum_meals"]) & counts.days.ge(cfg["minimum_days"])
    meals = meals.loc[meals.input_row.isin(counts.index[keep])].copy()
    idx = package / "inputs/discovery_v2/discovery_covariates_food_TRE_ONLY.parquet"
    require(digest(idx) == model["sources"][str(idx)], "Frozen index changed")
    index = inputs.frame(pq.read_table(idx, columns=KEY + ["input_row", "split_unit_id", "final_role"],
                                       use_threads=False, pre_buffer=False))
    require(index.final_role.eq("discovery").all() and index.input_row.is_unique,
            "Unexpected discovery population")
    meals = meals.merge(index[["input_row", "split_unit_id"]], on="input_row", validate="many_to_one")
    require(flow(meals) == summary["flow"], "Projected population differs from model archive")
    paths = [prep, window, idx, archive / "summary.json", MEAL / "config.json", *PINS]
    return meals, index, cfg, inputs, io, prep, {str(path): digest(path) for path in paths}


def classified_sources(package, meals, index, cfg, inputs, prep):
    project = package.parent
    qc = project / "data_qc/w07_w08_diet_views_20261028"
    crosswalk = project / "config/food_id_crosswalk_versioned.parquet"
    w07 = qc / "summary_TRE_ONLY.json"
    verification = qc / "verification.json"
    food = project / "analysis/shared/food_events_clean_TRE_ONLY.parquet"
    sources = {str(p): digest(p) for p in (crosswalk, w07, verification, food)}
    manifest = load(w07)
    proofs = [p for p in manifest["artifacts"] if p["cloud"] == str(crosswalk)]
    verified = load(verification)
    require(verified["verified"] and verified["W07_crosswalk_with_unknowns_built"]
            and len(proofs) == 1 and proofs[0]["sha256"] == sources[str(crosswalk)],
            "W07 crosswalk lacks a matching verification receipt")
    require(sources[str(food)] == load(prep)["sources"][str(food)], "Archived food input changed")
    lookup = pq.read_table(crosswalk, columns=["food_id", "NOVA", "mapping_version"],
                           use_threads=False, pre_buffer=False).to_pandas()
    require(len(lookup) == verified["crosswalk_rows"] and lookup.mapping_version.nunique() == 1,
            "W07 lookup size or version changed")
    roster = index.loc[index.input_row.isin(meals.input_row)]
    cols = ["collection_timestamp", "local_timestamp", "food_id", "food_category", "qc_primary_eligible",
            "qc_same_food_time_n", "qc_exact_signature_n", *NUTRIENTS]
    events = scoped_read(food, cols, roster, inputs)
    events = pd.concat(food_groups(events).values(), ignore_index=True)
    result = reconstruct(meals, events, lookup, NUTRIENTS, cfg["max_food_episode_span_minutes"] * 60)
    return result, sources, str(lookup.mapping_version.iloc[0])


def flow(meal):
    return dict(people=int(meal.input_row.nunique()), households=int(meal.split_unit_id.nunique()), meals=len(meal))


def support(meal, sources):
    eligible = meal.loc[repeated_support(meal, sources)].copy()
    eligible["person_index"] = eligible.input_row
    dates, late = assign_periods(eligible)
    dated = eligible.loc[dates].copy()
    dated["period"] = 2 * dated.input_row + late[dates]
    period_ok = dated.loc[repeated_support(dated, sources, days=1, unit="period")]
    two_halves = period_ok.groupby("input_row").period.nunique().eq(2)
    return dict(flow=flow(meal), repeated=flow(eligible), date_transfer=flow(dated),
                both_sources_each_period=flow(dated.loc[dated.input_row.isin(two_halves.index[two_halves])]),
                nearby_pair_people=nearby_pair_count(eligible, *sources))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--package", type=Path, required=True)
    cli.add_argument("--run", required=True)
    cli.add_argument("--commit", required=True)
    cli.add_argument("--smoke", action="store_true")
    args = cli.parse_args()
    package = args.package.resolve()
    require(package / "experiments" / ROOT.name == ROOT and re.fullmatch(r"[a-zA-Z0-9_-]+", args.run)
            and re.fullmatch(r"[0-9a-f]{40}", args.commit), "Unexpected release or run path")
    out = ROOT / "runs_TRE_ONLY" / args.run
    require(not out.exists(), "Use a fresh audit identifier")
    meals, index, cfg, inputs, io, prep, sources = projected_meals(package)
    if args.smoke:
        people = np.sort(meals.input_row.unique())[:cfg["smoke_people"]]
        meals = meals.loc[meals.input_row.isin(people)]
    meals, food_hashes, mapping = classified_sources(package, meals, index, cfg, inputs, prep)
    sources.update(food_hashes)
    sources.update({str(p): digest(p) for p in (Path(__file__).resolve(), ROOT / "processing.py")})
    result = {name: support(meals.loc[mask], CARBS[:2]) for name, mask in coverage(meals).items()}
    result["provider_wholewheat_vs_bread"] = support(meals, ("wholewheat_carbs", "bread_carbs"))
    result["all_meals"] = dict(**flow(meals), linked_events=int(meals.linked_events.sum()),
                               unknown_events=int(meals.unknown_events.sum()),
                               unknown_carbohydrate_fraction=float(meals.unknown_carbs.sum() / meals.carbs.sum()),
                               unknown_energy_fraction=float(meals.unknown_kcal.sum() / meals.energy.sum()))
    require(all(digest(Path(p)) == sha for p, sha in sources.items()), "Audit inputs changed")
    out.mkdir(parents=True, exist_ok=False)
    reg = out / "registration.json"
    summary = out / "summary.json"
    io.write(reg, dict(commit=args.commit, smoke=args.smoke, sources=sources, mapping_version=mapping,
                       outcome_columns_read=False))
    io.write(summary, result)
    io.write(out / "COMPLETE.json", dict(registration=digest(reg), summary=digest(summary)))
    io.emit("PROCESSING_AUDIT_COMPLETE", smoke=args.smoke, **result)


if __name__ == "__main__":
    main()
