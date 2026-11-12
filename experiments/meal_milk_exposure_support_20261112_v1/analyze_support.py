"""Audit milk product labels and meal-dose support without opening glucose values."""

import argparse
from io import BytesIO
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

from design import genus_detection, overlapping_rows, reconstruct


ROOT = Path(__file__).resolve().parent
MEAL = ROOT.parent / "meal_carbohydrate_modification_20261112_v1"
PROCESSING = ROOT.parent / "meal_processing_modification_20261112_v1"
COFFEE = ROOT.parent / "meal_coffee_modification_20261112_v1"
PINS = {
    PROCESSING / "audit.py": "d093b86be0d041fc2eb1530329216518eb76ddae54615fcdcc2c3d0ca3463295",
    PROCESSING / "processing.py": "f249264bcdec0ed84b299cd3702ff5ed1e61937a6baf33ea8ed883fce697c567",
    COFFEE / "coffee_design.py": "e5943010e99f7fcbc4a264387b123d5bcc945166d9fc44fad238f2d8420e5390",
}
CONTRASTS = (
    ("free", "unspecified"),
    ("reduced", "unspecified"),
    ("unspecified", "no_listed_milk"),
)


def flow(meal):
    return dict(people=int(meal.input_row.nunique()), households=int(meal.split_unit_id.nunique()),
                meals=len(meal), positive_people=int(meal.loc[meal.detected, "input_row"].nunique()))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--package", type=Path, required=True)
    cli.add_argument("--commit", required=True)
    cli.add_argument("--run", required=True)
    cli.add_argument("--smoke", action="store_true")
    args = cli.parse_args()
    package = args.package.resolve()
    if (package / "experiments" / ROOT.name != ROOT or not re.fullmatch(r"[0-9a-f]{40}", args.commit)
            or not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run)):
        raise ValueError("Unexpected study path, release or run")
    sys.path[:0] = [str(MEAL), str(PROCESSING), str(COFFEE)]
    try:
        from audit import projected_meals
        from coffee_design import eligible_events
        from paired import verified_result
        from prepare import NUTRIENTS, digest, food_groups, load, require, scoped_read
    finally:
        del sys.path[:3]
    require(all(digest(path) == sha for path, sha in PINS.items()), "Pinned helpers differ")
    meals, index, cfg, inputs, io, prep, sources = projected_meals(package)
    if args.smoke:
        selected = np.sort(meals.input_row.unique())[:48]
        meals = meals.loc[meals.input_row.isin(selected)].copy()
    food_path = package.parent / "analysis/shared/food_events_clean_TRE_ONLY.parquet"
    food_sha = digest(food_path)
    require(food_sha == load(prep)["sources"][str(food_path)], "Food events differ from archived inputs")
    columns = ["collection_timestamp", "local_timestamp", "short_food_name", "product_name",
               "qc_primary_eligible", "qc_same_food_time_n", "qc_exact_signature_n", *NUTRIENTS]
    roster = index.loc[index.input_row.isin(meals.input_row)]
    events = scoped_read(food_path, columns, roster, inputs)
    events = pd.concat(food_groups(events).values(), ignore_index=True)
    meals = reconstruct(meals, events, NUTRIENTS, cfg["max_food_episode_span_minutes"] * 60, eligible_events)

    archived, _ = verified_result(MEAL / "runs_TRE_ONLY/models-1")
    input_dir = package / "inputs/discovery_v2"
    raw_path = input_dir / "gut_raw_TRE_ONLY.npz"
    taxa_path = input_dir / "gut_taxa.json"
    for path in (raw_path, taxa_path):
        sources[str(path)] = digest(path)
        require(sources[str(path)] == archived["sources"][str(path)], "Archived gut profile changed")
    require(np.array_equal(index.input_row, np.arange(len(index))), "Archived row order changed")
    with np.load(BytesIO(raw_path.read_bytes()), allow_pickle=False) as contents:
        detected, taxa = genus_detection(contents["abundance"], load(taxa_path), len(index))
    meals["detected"] = detected[meals.input_row.to_numpy(int)]
    contexts = {label: flow(rows) for label, rows in meals.groupby("context")}
    contrasts = {}
    for first, second in CONTRASTS:
        common = overlapping_rows(meals, first, second)
        contrasts[f"{first}_vs_{second}"] = dict(
            **flow(common), first_meals=int(common.context.eq(first).sum()),
            second_meals=int(common.context.eq(second).sum()))
    sources[str(food_path)] = food_sha
    for path in (Path(__file__), ROOT / "design.py", *PINS):
        sources[str(path)] = digest(path)
    result = dict(outcome_columns_read=False, whole=flow(meals), exact_terminal_taxa=taxa,
                  contexts=contexts, common_support=contrasts)
    out = ROOT / "runs_TRE_ONLY" / args.run
    out.mkdir(parents=True, exist_ok=False)
    io.write(out / "registration.json", dict(commit=args.commit, smoke=args.smoke, sources=sources))
    io.write(out / "summary.json", result)
    io.write(out / "COMPLETE.json", dict(registration=digest(out / "registration.json"),
                                         summary=digest(out / "summary.json")))
    io.emit("MILK_EXPOSURE_SUPPORT_COMPLETE", smoke=args.smoke, **result)


if __name__ == "__main__":
    main()
