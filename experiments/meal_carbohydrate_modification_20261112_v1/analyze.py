"""Compare within-person carbohydrate modification on the same meal windows."""

import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import t

from prepare import ROOT, base_modules, digest, load, require


ENDPOINTS = ("net_p0", "peak_p0", "net_m15", "net_p15", "pre_p0")
EXPOSURES = ("carb_share", "carbs")
PRIMARY = "net_p0_carb_share_external_sum7"
EXTERNAL_CONFIG_SHA = "8cfe32a8ab2c64e05b805f1f9c61c3aa22dd17b0a6d81cf255b87e9487c4d897"


def contrasts(meals):
    share = (meals.carb_share.to_numpy(float) - 50) / 10
    grams = (meals.carbs.to_numpy(float) - 40) / 10
    return {"carb_share": share, "carbs": grams}


def bases(meals, model):
    person = meals.person_index.to_numpy()
    doses = contrasts(meals)
    age = (meals.age_calendar_years.to_numpy(float) - 50) / 10
    sex = meals.sex_code.to_numpy(float)
    hour = 2 * np.pi * meals.local_hour.to_numpy(float) / 24
    gap = np.log1p(meals.prior_gap.to_numpy(float) / 60)
    energy = np.log(meals.energy.to_numpy(float) / 100)
    common = [energy, energy ** 2, meals.fiber.to_numpy(float) / 5, gap, np.sin(hour), np.cos(hour),
              *(meals.weekday.to_numpy() == i for i in range(1, 7))]
    matrices = {}
    for exposure, dose in doses.items():
        protein = ((meals.protein_share if exposure == "carb_share" else meals.protein).to_numpy(float) - 15) / 10
        background = [*common, protein, protein ** 2, dose, dose ** 2, dose * protein,
                      dose * age, dose * sex]
        for endpoint in ENDPOINTS:
            tag = endpoint.rsplit("_", 1)[1]
            a = background if endpoint == "pre_p0" else [
                *background, (meals[f"pre_{tag}"].to_numpy(float) - 100) / 10,
                meals[f"trend_{tag}"].to_numpy(float),
            ]
            centered, weight = model.center(np.column_stack([meals[endpoint].to_numpy(float), *a]), person)
            matrices[endpoint, exposure] = (centered[:, 0], centered[:, 1:], weight)
    return matrices, doses


def panel(raw, taxa, mapping):
    lookup = [name.rsplit("|", 1)[-1] for name in taxa]
    keys = [name for name in mapping.values() if name is not None]
    require(len(keys) == len(set(keys)) == 7 and all(lookup.count(key) == 1 for key in keys),
            "External literature panel no longer aligns")
    return raw[:, [lookup.index(key) for key in keys]].sum(axis=1, keepdims=True)


def markers(model, raw, taxa, index, pop, train, clr, mapping):
    result = model.site_scores(raw, taxa, index, pop, train, clr, 5)
    return dict(gut_shannon=result["gut_shannon"], oral_shannon=result["oral_shannon"],
                gut_pc5=result["gut_pc"], oral_pc5=result["oral_pc"], both_pc5=result["both_pc"],
                external_sum7=model.train_standardize(panel(raw["gut"][pop], taxa["gut"], mapping), train))


def compare(meals, matrices, dose, scores, train, model, direct=False):
    person = meals.person_index.to_numpy()
    groups = meals.split_unit_id.astype(str).to_numpy()
    take = train[person]
    check = {}
    interaction = {(e, name): model.center(d[:, None] * vals[person], person)[0]
                   for e, d in dose.items() for name, vals in scores.items()}
    for (endpoint, exposure), (y, x, w) in matrices.items():
        fitted = model.fit_background(x[take], y[take], w[take])
        for name, score in scores.items():
            z = interaction[exposure, name]
            key = f"{endpoint}_{exposure}_{name}"
            if direct:
                beta, cov, p = model.household_test(z, fitted, groups, len(train))
                row = dict(p=p, contrast_sd=float(np.std(score @ beta, ddof=1)))
                if len(beta) == 1:
                    q = float(np.diff(np.quantile(score[:, 0], [0.25, 0.75]))[0])
                    effect = float(q * beta[0])
                    margin = float(q * np.sqrt(cov[0, 0]) * t.ppf(0.975, len(set(groups)) - 1))
                    row.update(q75_minus_q25=effect, ci95=[effect - margin, effect + margin])
                check[key] = row
            else:
                beta, sse0, sse = model.fit_interaction(x, z, y, w, take, ~take, fitted, 0.03)
                check[key] = dict(sse=sse, sse_null=sse0, tss=float(np.sum(w[~take] * y[~take] ** 2)))
    if direct:
        for row, adjusted in zip(check.values(), model.holm([row["p"] for row in check.values()])):
            row["holm_p"] = float(adjusted)
    return check


def pool(parts):
    result = {}
    for repeat in sorted({part["repeat"] for part in parts}):
        rows = [part["results"] for part in parts if part["repeat"] == repeat]
        for key in rows[0]:
            sse0, sse, tss = [sum(row[key][name] for row in rows) for name in ("sse_null", "sse", "tss")]
            result.setdefault(key, {})[f"r{repeat}"] = dict(
                r2=1 - sse / tss, null_r2=1 - sse0 / tss, delta_r2=(sse0 - sse) / tss)
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--package", required=True, type=Path)
    cli.add_argument("--prepared", required=True)
    cli.add_argument("--commit", required=True)
    cli.add_argument("--run", required=True)
    cli.add_argument("--smoke", action="store_true")
    args = cli.parse_args()
    package = args.package.resolve()
    require((package / "experiments" / ROOT.name).resolve() == ROOT
            and re.fullmatch(r"[a-zA-Z0-9_-]+", args.run)
            and re.fullmatch(r"[a-zA-Z0-9_-]+", args.prepared)
            and re.fullmatch(r"[0-9a-f]{40}", args.commit), "Invalid release or run paths")
    config = load(ROOT / "config.json")
    inputs, io = base_modules(config)
    model = __import__("models")
    prepared = ROOT / "runs_TRE_ONLY" / args.prepared
    m = load(prepared / "COMPLETE.json")
    table = prepared / "meal_windows_TRE_ONLY.parquet"
    require(digest(table) == m["windows"] and digest(prepared / "registration.json") == m["registration"]
            and digest(prepared / "summary.json") == m["summary"], "Prepared meal provenance differs")
    require(args.smoke or not load(prepared / "summary.json")["smoke"], "Full models need a full preparation run")
    meal = pq.read_table(table, use_threads=False, pre_buffer=False).to_pandas()
    meal = meal.loc[meal.full_offsets].copy()
    support = meal.groupby("input_row").agg(meals=("start", "size"), days=("date", "nunique"))
    use = support.meals.ge(config["minimum_meals"]) & support.days.ge(config["minimum_days"])
    meal = meal.loc[meal.input_row.isin(support.index[use])].copy()
    require(len(meal) and np.isfinite(meal[list(ENDPOINTS)].to_numpy()).all(), "Missing timing-matched outcomes")
    h = package / "inputs/discovery_v2"
    x = h / "discovery_covariates_food_TRE_ONLY.parquet"
    split = package / "experiments/diet_nested_stability_20261103_v4/results/TRE_ONLY/final_predictions.npz"
    exp2 = package / "experiment2_Effect_Modification/v1/runs_TRE_ONLY" / inputs.EXP2
    anchor = exp2 / "registration.json"
    require(digest(anchor) == load(exp2 / "COMPLETE.json")["registration"], "Experiment 2 registration differs")
    refs = load(anchor)["sources"]
    raw_paths = {s: h / f"{s}_raw_TRE_ONLY.npz" for s in ("oral", "gut")}
    names = {s: h / f"{s}_taxa.json" for s in raw_paths}
    prep_code = package / "preprocessing/code/preprocessing_v1_20261030.py"
    src = {str(p): digest(p) for p in (x, split, prep_code, *raw_paths.values(), *names.values())}
    require(all(refs.get(p) == d for p, d in src.items()), "Frozen microbiome/partition source differs")
    index = inputs.frame(pq.read_table(x, columns=inputs.INDEX, pre_buffer=False, use_threads=False))
    require(index.final_role.eq("discovery").all() and np.array_equal(index.input_row, np.arange(len(index))),
            "Expected discovery index order")
    with np.load(BytesIO(split.read_bytes()), allow_pickle=False) as ar:
        folds, eligible = ar["outer"], np.isfinite(ar["y"])
    require(np.array_equal(*eligible) and eligible[0, meal.input_row.to_numpy(int)].all(),
            "Prepared meals include identities outside the frozen population")
    meal = meal.merge(index[["input_row", "age_calendar_years", "sex_code", "split_unit_id"]],
                      on="input_row", validate="many_to_one")
    pop = np.sort(meal.input_row.unique()).astype(int)
    meal["person_index"] = meal.input_row.map({row: j for j, row in enumerate(pop)})
    raw, taxa = {}, {}
    for site, path in raw_paths.items():
        with np.load(BytesIO(path.read_bytes()), allow_pickle=False) as ar:
            raw[site] = ar["abundance"]
        taxa[site] = load(names[site])
    spec = spec_from_file_location("hpp_meal_clr", prep_code)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    external = ROOT.parent / "diet_carb_external_score_20261112_v1/config.json"
    require(digest(external) == EXTERNAL_CONFIG_SHA, "External score dictionary changed")
    mapping = load(external)["taxa"]
    src.update({str(p): digest(p) for p in (ROOT / "analyze.py", ROOT / "prepare.py",
                                          ROOT / "config.json", external, table, anchor, prepared / "COMPLETE.json")})
    path = ROOT / "runs_TRE_ONLY" / args.run
    path.mkdir(parents=True, exist_ok=False)
    flow = dict(people=len(pop), households=int(meal.split_unit_id.nunique()), meals=len(meal))
    reg = dict(commit=args.commit, prepared=args.prepared, sources=src, smoke=args.smoke, flow=flow)
    io.write(path / "registration.json", reg)
    fingerprint = digest(path / "registration.json")
    matrices, doses = bases(meal, model)
    parts = []
    schedule = [(0, 0)] if args.smoke else [(r, f) for r in (0, 1) for f in range(5)]
    io.emit("MEAL_MODEL_START", **flow)
    for repeat, fold in schedule:
        train = folds[repeat, pop] != fold
        families = index.iloc[pop].split_unit_id.to_numpy()
        require(not (set(families[train]) & set(families[~train])), "Households overlap")
        marker = markers(model, raw, taxa, index, pop, train, mod.MicrobiomeCLR, mapping)
        row = dict(registration=fingerprint, repeat=repeat, fold=fold,
                   results=compare(meal, matrices, doses, marker, train, model))
        io.write(path / f"r{repeat}_f{fold}.json", row)
        parts.append(row)
        io.emit("MEAL_MODEL_FOLD", repeat=repeat, fold=fold, held=int((~train).sum()))
    report = dict(flow=flow, folds=len(parts), prediction=pool(parts))
    if not args.smoke:
        all_train = np.ones(len(pop), bool)
        marker = markers(model, raw, taxa, index, pop, all_train, mod.MicrobiomeCLR, mapping)
        report["direct"] = compare(meal, matrices, doses, marker, all_train, model, direct=True)
        io.emit("MEAL_MODEL_PRIMARY", **report["direct"][PRIMARY], prediction=report["prediction"][PRIMARY])
    require(all(digest(Path(p)) == value for p, value in src.items()), "Analysis source changed")
    io.write(path / "summary.json", report)
    io.write(path / "COMPLETE.json", dict(registration=fingerprint, summary=digest(path / "summary.json")))
    io.emit("MEAL_MODEL_COMPLETE", smoke=args.smoke, folds=len(parts))


if __name__ == "__main__":
    main()
