"""Test carbohydrate-by-microbiome modification within exact meal templates."""

import argparse
from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
import re
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd

from template_design import DIRECTIONS, KEYS, paired_summary, reconstruct, scopes, summary

ROOT = Path(__file__).resolve().parent
MEAL = ROOT.parent / "meal_carbohydrate_modification_20261112_v1"
PROCESSING = ROOT.parent / "meal_processing_modification_20261112_v1"
COFFEE = ROOT.parent / "meal_coffee_modification_20261112_v1"
PINS = {
    MEAL / "paired.py": "6495e7bf2fe831a971be31a1dbc1ddaf582773c13a04d89dbf7641a478239b0a",
    MEAL / "analyze.py": "319cf53ec1a4414153ef2b929c74c071d09e4365e95bc8fb0f712d7c6aaf2ae6",
    PROCESSING / "audit.py": "d093b86be0d041fc2eb1530329216518eb76ddae54615fcdcc2c3d0ca3463295",
    PROCESSING / "processing.py": "f249264bcdec0ed84b299cd3702ff5ed1e61937a6baf33ea8ed883fce697c567",
    COFFEE / "coffee_design.py": "e5943010e99f7fcbc4a264387b123d5bcc945166d9fc44fad238f2d8420e5390",
}
SEED = 202611137

sys.path[:0] = [str(MEAL), str(PROCESSING), str(COFFEE)]
try:
    from prepare import NUTRIENTS, digest, food_groups, load, require, scoped_read
    from analyze import ENDPOINTS, bases
    from audit import projected_meals
    from coffee_design import eligible_events
    from paired import archived_inputs, family_totals
finally:
    del sys.path[:3]


def save_fold(path, values, io, fingerprint, repeat, fold):
    buffer = BytesIO()
    np.savez_compressed(buffer, **values)
    payload = buffer.getvalue()
    path.write_bytes(payload)
    io.write(path.with_suffix(".json"), dict(registration=fingerprint, repeat=repeat, fold=fold,
                                            sha=sha256(payload).hexdigest()))


def add_totals(totals, parts):
    for name, value in parts.items():
        totals[name] = totals.get(name, 0) + value


def reduce_totals(totals, repeat):
    return {key: paired_summary(values, SEED + repeat) for key, values in totals.items()}


def prepare_food(package):
    meals, index, cfg, inputs, io, prep, sources = projected_meals(package)
    food_path = package.parent / "analysis/shared/food_events_clean_TRE_ONLY.parquet"
    food_sha = digest(food_path)
    require(food_sha == load(prep)["sources"][str(food_path)], "Food events differ from archived input")
    columns = ["collection_timestamp", "local_timestamp", "food_id", "qc_primary_eligible",
               "qc_same_food_time_n", "qc_exact_signature_n", *NUTRIENTS]
    roster = index.loc[index.input_row.isin(meals.input_row)]
    events = scoped_read(food_path, columns, roster, inputs)
    events = pd.concat(food_groups(events).values(), ignore_index=True)
    complete = reconstruct(meals, events, NUTRIENTS, cfg["max_food_episode_span_minutes"] * 60, eligible_events)
    pooled, calendar = scopes(complete)
    support = dict(archived_meals=len(meals), complete_food_id_meals=len(complete),
                   pooled=summary(pooled), calendar=summary(calendar, calendar=True))
    sources[str(food_path)] = food_sha
    for path in (Path(__file__), ROOT / "template_design.py", *PINS):
        sources[str(path)] = digest(path)
    return pooled, calendar, support, sources, io


def dataset(outcomes, rows, population, model, calendar=False):
    take = outcomes.merge(rows[[*KEYS, "group", "late"]], on=KEYS, validate="one_to_one")
    if len(take) != len(rows):
        raise ValueError("Template keys differ from archived outcomes")
    actual_person = np.searchsorted(population, take.input_row)
    if not np.array_equal(population[actual_person], take.input_row.to_numpy()):
        raise ValueError("Template participants differ from frozen population")
    take["person_index"] = take.group
    matrices, dose = bases(take, model)
    person_period = ["input_row", "late"] if calendar else "input_row"
    weights = 1 / take.groupby(person_period).date.transform("size").to_numpy()
    return take, actual_person, matrices, dose["carbs"], weights


def design(data, score, model):
    meal, person, _, dose, _ = data
    z = score[person]
    energy = np.log(meal.energy.to_numpy(float) / 100)
    protein = (meal.protein.to_numpy(float) - 15) / 10
    blocks = np.column_stack([z * term[:, None] for term in (energy, protein, dose)])
    centered, _ = model.center(blocks, meal.group.to_numpy())
    return centered[:, :-z.shape[1]], centered[:, -z.shape[1]:]


def matched_losses(data, score, train, direction, model, groups):
    meal, person, matrices, _, weights = data
    baseline, tested = design(data, score, model)
    training, valid = train[person], ~train[person]
    if direction != "pooled":
        later = meal.late.to_numpy(bool)
        early_to_late = direction == "early_to_late"
        training = training & (later != early_to_late)
        valid = valid & (later == early_to_late)
    family = np.searchsorted(groups, meal.split_unit_id.astype(str))
    result = {}
    for endpoint in ENDPOINTS:
        y, x, _ = matrices[endpoint, "carbs"]
        parts = family_totals(y, np.column_stack([x, baseline]), weights, {"interaction": tested},
                             training, family, len(groups), model, valid=valid)
        result[f"{direction}__{endpoint}"] = np.stack([parts[key] for key in ("none", "interaction", "tss")])
    return result


def associations(data, score, model):
    meal, _, matrices, _, weights = data
    baseline, tested = design(data, score, model)
    households = meal.split_unit_id.astype(str).to_numpy()
    groups = meal.group.nunique()
    result = {}
    for endpoint in ENDPOINTS:
        y, x, _ = matrices[endpoint, "carbs"]
        fitted = model.fit_background(np.column_stack([x, baseline]), y, weights)
        residual, _, _ = model.orthogonal_block(tested, fitted)
        _, _, rank, _ = np.linalg.lstsq(residual, np.zeros(len(y)), rcond=1e-10)
        if (rank < tested.shape[1] or len(y) - groups - fitted[1].shape[1] <= rank
                or len(np.unique(households)) <= rank):
            result[endpoint] = {"status": "insufficient_rank_or_residual_df", "rank": int(rank)}
            continue
        try:
            beta, _, p = model.household_test(tested, fitted, households, groups)
        except np.linalg.LinAlgError:
            result[endpoint] = {"status": "singular_household_covariance", "rank": int(rank)}
            continue
        used = np.unique(data[1])
        result[endpoint] = dict(p=p, effect_sd_mg_dl=float(np.std(score[used] @ beta, ddof=1)))
    # Keep the five-endpoint family even if a diagnostic endpoint is unavailable.
    adjusted = model.holm([row.get("p", 1.) for row in result.values()])
    for row, p in zip(result.values(), adjusted):
        if "p" in row:
            row["holm_p"] = float(p)
    return result


def verify(run, commit):
    out = ROOT / "runs_TRE_ONLY" / run
    meta, result = load(out / "registration.json"), load(out / "summary.json")
    receipt = load(out / "COMPLETE.json")
    fingerprint = digest(out / "registration.json")
    require(receipt == dict(registration=fingerprint, summary=digest(out / "summary.json")),
            "Template completion receipt differs")
    require(meta["commit"] == commit and meta["support"] == result["support"]
            and all(digest(Path(p)) == sha for p, sha in meta["sources"].items()),
            "Template sources or outcome-blind support differ")
    if meta["audit"]:
        require(not result["prediction"] and not result["direct"], "Audit contains outcome results")
        print("TEMPLATE_READBACK_PASS audit", flush=True)
        return
    repeats, folds = ((0,), (0,)) if meta["smoke"] else (range(2), range(5))
    names = {f"{scope}__{ep}" for scope in meta["routes"] for ep in ENDPOINTS}
    require(set(result["direct"]) == (set() if meta["smoke"] else set(ENDPOINTS)), "Direct endpoints differ")
    for repeat in repeats:
        totals = {}
        for fold in folds:
            part = out / f"r{repeat}_f{fold}.npz"
            payload = part.read_bytes()
            require(load(part.with_suffix(".json")) == dict(registration=fingerprint, repeat=repeat,
                    fold=fold, sha=sha256(payload).hexdigest()), "Template fold receipt differs")
            with np.load(BytesIO(payload), allow_pickle=False) as arrays:
                require(set(arrays.files) == names, "Unexpected fold comparisons")
                add_totals(totals, arrays)
        require(reduce_totals(totals, repeat) == result["prediction"][f"r{repeat}"], "Template pooling differs")
    print(f"TEMPLATE_READBACK_PASS folds={len(repeats) * len(folds)}", flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--package", type=Path, required=True)
    cli.add_argument("--commit", required=True)
    cli.add_argument("--run", required=True)
    cli.add_argument("--support-run", default="support-1")
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument("--audit", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = cli.parse_args()
    package = args.package.resolve()
    if (package / "experiments" / ROOT.name != ROOT or not re.fullmatch(r"[0-9a-f]{40}", args.commit)
            or not all(re.fullmatch(r"[a-zA-Z0-9_-]+", name) for name in (args.run, args.support_run))):
        raise ValueError("Unexpected release, run or package")
    require(all(digest(path) == sha for path, sha in PINS.items()), "Pinned helper differs")
    if args.verify:
        verify(args.run, args.commit)
        return
    out = ROOT / "runs_TRE_ONLY" / args.run
    require(not out.exists(), "Use a fresh run name")
    pooled, calendar, support, sources, io = prepare_food(package)
    io.emit("TEMPLATE_SUPPORT", **support)
    prediction, direct, routes = {}, {}, []
    if not args.audit:
        support_path = ROOT / "runs_TRE_ONLY" / args.support_run
        support_meta = load(support_path / "registration.json")
        support_receipt = load(support_path / "COMPLETE.json")
        require(support_meta["audit"] and support_meta["commit"] == args.commit
                and support_meta["sources"] == sources and support_meta["support"] == support
                and support_receipt["registration"] == digest(support_path / "registration.json")
                and support_receipt["summary"] == digest(support_path / "summary.json"),
                "Outcome-blind support receipt differs")
        sources[str(support_path / "COMPLETE.json")] = digest(support_path / "COMPLETE.json")
        require(support["pooled"]["effective_households"] > 1 and support["pooled"]["adjusted_carb_sd_g"] > 1e-6,
                "Insufficient energy- and protein-adjusted support")
        receipt, _, _, meal, index, population, groups, _, outer, raw, taxa, clr, _ = archived_inputs(
            package, MEAL / "runs_TRE_ONLY/models-1")
        require(all(p not in sources or sources[p] == sha for p, sha in receipt["sources"].items()),
                "Projected and full sources differ")
        sources.update(receipt["sources"])
        model = sys.modules["models"]
        household = index.iloc[population].split_unit_id.astype(str).to_numpy()
        datasets = {"pooled": dataset(meal, pooled, population, model)}
        # Require every held-out fold to contain household outcome contrasts in both periods.
        if support["calendar"]["effective_households"] > 1:
            data = dataset(meal, calendar, population, model, calendar=True)
            if all(np.bincount(outer[r, data[1][data[0].late.eq(late)]], minlength=5).min() > 0
                   for r in (0, 1) for late in (False, True)):
                datasets.update({name: data for name in DIRECTIONS})
        routes = list(datasets)
    out.mkdir(parents=True, exist_ok=False)
    reg = out / "registration.json"
    io.write(reg, dict(commit=args.commit, audit=args.audit, smoke=args.smoke, support=support, sources=sources,
                       routes=routes, primary="pooled__net_p0", seed=SEED, bootstrap=1999))
    fingerprint = digest(reg)
    if not args.audit:
        for repeat in ((0,) if args.smoke else (0, 1)):
            totals = {}
            for fold in ((0,) if args.smoke else range(5)):
                train = outer[repeat] != fold
                require(not set(household[train]) & set(household[~train]), "Households cross a frozen fold")
                score = model.site_scores(raw, taxa, index, population, train, clr, 5)["both_pc"]
                values = {}
                for route, data in datasets.items():
                    values.update(matched_losses(data, score, train, route, model, groups))
                save_fold(out / f"r{repeat}_f{fold}.npz", values, io, fingerprint, repeat, fold)
                add_totals(totals, values)
                io.emit("TEMPLATE_FOLD", repeat=repeat, fold=fold)
            prediction[f"r{repeat}"] = reduce_totals(totals, repeat)
        if not args.smoke:
            used = np.unique(datasets["pooled"][1])
            train = np.isin(np.arange(len(population)), used)
            score = model.site_scores(raw, taxa, index, population, train, clr, 5)["both_pc"]
            direct = associations(datasets["pooled"], score, model)
    io.write(out / "summary.json", dict(support=support, prediction=prediction, direct=direct))
    io.write(out / "COMPLETE.json", dict(registration=fingerprint, summary=digest(out / "summary.json")))
    io.emit("TEMPLATE_COMPLETE", audit=args.audit, smoke=args.smoke, routes=routes)


if __name__ == "__main__":
    main()
