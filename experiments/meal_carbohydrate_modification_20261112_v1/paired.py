"""Replay and pair-bootstrap matched meal models for the carbohydrate-grams signal.

Intervals describe the archived fitted discovery predictions. Each of the two
household partitions is resampled separately. The models are not refit in the
bootstrap, and these intervals are exploratory after route selection.
"""

import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq

from analyze import ENDPOINTS, EXTERNAL_CONFIG_SHA, bases, markers
from prepare import ROOT, base_modules, digest, load, require


SITES = ("oral_pc5", "gut_pc5", "both_pc5")
COMPARE = dict(oral_vs_none=("oral_pc5", "none"), gut_vs_none=("gut_pc5", "none"),
               both_vs_none=("both_pc5", "none"), both_vs_oral=("both_pc5", "oral_pc5"),
               both_vs_gut=("both_pc5", "gut_pc5"))


def family_totals(y, x, w, z, training, family, n_family, model):
    valid = ~training
    fitted = model.fit_background(x[training], y[training], w[training])
    background = x[valid] @ fitted[0]
    loss = dict(none=(y[valid] - background) ** 2)
    for key, terms in z.items():
        zr, yr, projection = model.orthogonal_block(terms[training], fitted)
        gram = zr.T @ zr + 0.03 * w[training].sum() * np.eye(terms.shape[1])
        beta = np.linalg.solve(gram, zr.T @ yr)
        change = (terms[valid] - x[valid] @ projection) @ beta
        loss[key] = (y[valid] - background - change) ** 2
    loss["tss"] = y[valid] ** 2
    return {k: np.bincount(family[valid], weights=w[valid] * values, minlength=n_family)
            for k, values in loss.items()}


def summarize(totals, sampled_households):
    tss = totals["tss"]
    denominator = tss[sampled_households].sum(axis=1)
    if not (tss.sum() > 0 and np.all(denominator > 0)):
        raise ValueError("Nonpositive matched prediction denominator")
    out = {}
    for name, (model, referent) in COMPARE.items():
        gains = totals[referent] - totals[model]
        stats = gains[sampled_households].sum(axis=1) / denominator
        out[name] = dict(delta_r2=float(gains.sum() / tss.sum()),
                         interval=np.quantile(stats, [0.025, 0.975]).tolist())
    return out


def verified_result(path):
    receipt = load(path / "COMPLETE.json")
    require(digest(path / "registration.json") == receipt["registration"]
            and digest(path / "summary.json") == receipt["summary"], "Archived analysis hash differs")
    summary = load(path / "summary.json")
    require(summary["folds"] == 10 and not load(path / "registration.json")["smoke"],
            "A complete full model run is required")
    require(len(list(path.glob("r*_f*.json"))) == 10 and all(
        load(p)["registration"] == receipt["registration"] for p in path.glob("r*_f*.json")),
        "Archived fold records differ")
    return load(path / "registration.json"), summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--archived", default="models-1")
    parser.add_argument("--run", required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    package = args.package.resolve()
    require((package / "experiments" / ROOT.name).resolve() == ROOT
            and re.fullmatch(r"[a-zA-Z0-9_-]+", args.archived)
            and re.fullmatch(r"[a-zA-Z0-9_-]+", args.run)
            and re.fullmatch(r"[0-9a-f]{40}", args.commit), "Unexpected run or source")
    config = load(ROOT / "config.json")
    inputs, io = base_modules(config)
    model = __import__("models")
    archived = ROOT / "runs_TRE_ONLY" / args.archived
    receipt, original = verified_result(archived)
    source_hashes = receipt["sources"]
    require(all(digest(Path(p)) == h for p, h in source_hashes.items()), "An archived model input changed")
    prepared = ROOT / "runs_TRE_ONLY" / receipt["prepared"]
    path = prepared / "meal_windows_TRE_ONLY.parquet"
    meal = pq.read_table(path, use_threads=False, pre_buffer=False).to_pandas()
    meal = meal.loc[meal.full_offsets].copy()
    counts = meal.groupby("input_row").agg(meals=("start", "size"), days=("date", "nunique"))
    ok = counts.meals.ge(config["minimum_meals"]) & counts.days.ge(config["minimum_days"])
    meal = meal.loc[meal.input_row.isin(counts.index[ok])].copy()
    h = package / "inputs/discovery_v2"
    idx_path = h / "discovery_covariates_food_TRE_ONLY.parquet"
    index = inputs.frame(pq.read_table(idx_path, columns=inputs.INDEX, pre_buffer=False, use_threads=False))
    require(index.final_role.eq("discovery").all() and np.array_equal(index.input_row, np.arange(len(index))),
            "Archived discovery index changed")
    meal = meal.merge(index[["input_row", "age_calendar_years", "sex_code", "split_unit_id"]],
                      on="input_row", validate="many_to_one")
    pop = np.sort(meal.input_row.unique()).astype(int)
    meal["person_index"] = meal.input_row.map({row: j for j, row in enumerate(pop)})
    family_ids, family = np.unique(meal.split_unit_id.astype(str), return_inverse=True)
    require(dict(people=len(pop), households=len(family_ids), meals=len(meal)) == original["flow"],
            "Meal cohort differs from the archived model")
    v4 = package / "experiments/diet_nested_stability_20261103_v4/results/TRE_ONLY/final_predictions.npz"
    with np.load(BytesIO(v4.read_bytes()), allow_pickle=False) as ar:
        outer = ar["outer"][:, pop]
    raw, taxa = {}, {}
    for site in ("oral", "gut"):
        with np.load(BytesIO((h / f"{site}_raw_TRE_ONLY.npz").read_bytes()), allow_pickle=False) as ar:
            raw[site] = ar["abundance"]
        taxa[site] = load(h / f"{site}_taxa.json")
    p = package / "preprocessing/code/preprocessing_v1_20261030.py"
    spec = spec_from_file_location("hpp_paired_clr", p)
    preprocess = module_from_spec(spec)
    spec.loader.exec_module(preprocess)
    external = ROOT.parent / "diet_carb_external_score_20261112_v1/config.json"
    require(digest(external) == EXTERNAL_CONFIG_SHA, "Literature panel definition differs")
    mapping = load(external)["taxa"]
    matrices, dose = bases(meal, model)
    person = meal.person_index.to_numpy()
    result = {name: {} for name in ENDPOINTS}
    out = ROOT / "runs_TRE_ONLY" / args.run
    out.mkdir(exist_ok=False)
    registration = dict(commit=args.commit, source=source_hashes, archived=args.archived,
                        archived_summary=digest(archived / "summary.json"), replicates=1999,
                        paired_code=digest(ROOT / "paired.py"), seed=202611122)
    io.write(out / "registration.json", registration)
    for repeat in (0, 1):
        by_family = {ep: {k: np.zeros(len(family_ids)) for k in ("none", *SITES, "tss")}
                     for ep in ENDPOINTS}
        for fold in range(5):
            training = outer[repeat] != fold
            groups = index.iloc[pop].split_unit_id
            require(not set(groups[training]) & set(groups[~training]), "Household crossing")
            marker = markers(model, raw, taxa, index, pop, training, preprocess.MicrobiomeCLR, mapping)
            z = {name: model.center(dose["carbs"][:, None] * marker[name][person], person)[0] for name in SITES}
            for endpoint in ENDPOINTS:
                y, x, w = matrices[endpoint, "carbs"]
                values = family_totals(y, x, w, z, training[person], family, len(family_ids), model)
                for k, vector in values.items():
                    by_family[endpoint][k] += vector
            io.emit("PAIRED_FOLD_DONE", repeat=repeat, fold=fold)
        rng = np.random.default_rng(registration["seed"] + repeat)
        boot = rng.integers(0, len(family_ids), (registration["replicates"], len(family_ids)), dtype=np.int32)
        for endpoint, totals in by_family.items():
            rep = f"r{repeat}"
            report = summarize(totals, boot)
            for s, key in (("oral", "oral_pc5"), ("gut", "gut_pc5"), ("both", "both_pc5")):
                ref = original["prediction"][f"{endpoint}_carbs_{key}"][rep]["delta_r2"]
                require(np.isclose(ref, report[f"{s}_vs_none"]["delta_r2"], atol=1e-10, rtol=0),
                        "Failed to reproduce the archived matched prediction")
            result[endpoint][rep] = report
        io.write(out / f"{rep}.json", {ep: rows[rep] for ep, rows in result.items()})
    require(all(digest(Path(p)) == h for p, h in source_hashes.items()), "An archived model input changed during replay")
    io.write(out / "summary.json", dict(flow=original["flow"], estimates=result))
    io.write(out / "COMPLETE.json", dict(registration=digest(out / "registration.json"),
                                         summary=digest(out / "summary.json")))
    for k in ("net_p0", "peak_p0", "pre_p0"):
        io.emit("PAIRED_RESULT", outcome=k,
                **{rep: reports["both_vs_none"] for rep, reports in result[k].items()})
    io.emit("PAIRED_COMPLETE", folds=10, comparisons=25)


if __name__ == "__main__":
    main()
