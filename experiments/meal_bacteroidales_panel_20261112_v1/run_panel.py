"""Test a prespecified three-species gut score in full HPP meal windows."""

import argparse
from io import BytesIO
import os
from pathlib import Path
import re
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import t

from panel_design import MODELS, SPECIES, panel_mass, score_losses

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT.parent / "meal_template_modification_20261112_v1"
TRANSFER = ROOT.parent / "meal_template_household_transfer_20261112_v1"
MEAL = ROOT.parent / "meal_carbohydrate_modification_20261112_v1"
PINS = {
    TEMPLATE / "analyze_template.py": "273dab1dc271c88e090d2fac91b720d26ad32af1f04821a71f87894e4ef55371",
    TEMPLATE / "template_design.py": "dee91e52e425485cc6976380e1599761bb79cf2f91c07e2d30f676b934ca682c",
    TRANSFER / "run_transfer.py": "6ab3de9b99f4b8d0354b390dbb023927e504be847de999daa7e135de41c90783",
    MEAL / "time_transfer.py": "c9295f753b4f74e9f6cade505bd35c248e6c7fdbf89a6d16263148411b0b20a8",
}
SEED = 202611147
DIRECTIONS = ("early_to_late", "late_to_early")

sys.path[:0] = [str(TEMPLATE), str(TRANSFER), str(MEAL)]
try:
    from analyze_template import (
        ENDPOINTS, PINS as TEMPLATE_PINS, add_totals, archived_inputs, bases, design, digest, family_totals,
        load, require, save_fold,
    )
    from run_transfer import ARCHIVE, NEW_FROZEN, NEW_PREPARED, calibration, check_sources, completed
    from run_transfer import fit_frozen, prepared, read_arrays, save_arrays
    from time_transfer import assign_periods
    from prepare import base_modules
finally:
    del sys.path[:3]


def data(frame, model, half=None):
    meal = frame.copy()
    people = meal.input_row.to_numpy(int)
    meal["group"] = people if half is None else 2 * people + half
    meal["person_index"] = meal.group
    matrices, dose = bases(meal, model)
    return meal, people, matrices, dose["carbs"], matrices["net_p0", "carbs"][2]


def blocks(dataset, score, model):
    guards, terms = design(dataset, score, model)
    return guards, dict(panel=terms[:, :1], pcs=terms[:, 1:], joint=terms)


def prediction(dataset, score, train, valid, family, count, model):
    _, _, matrices, _, weights = dataset
    guards, terms = blocks(dataset, score, model)
    results = {}
    for ep in ENDPOINTS:
        y, x, _ = matrices[ep, "carbs"]
        parts = family_totals(y, np.column_stack([x, guards]), weights, terms, train, family, count, model, valid=valid)
        results[ep] = np.stack([parts[key] for key in ("none", *MODELS, "tss")])
    return results


def direct(dataset, score, model):
    meal, people, matrices, _, weights = dataset
    guards, terms = blocks(dataset, score, model)
    hh = meal.split_unit_id.astype(str).to_numpy()
    n_people = meal.input_row.nunique()
    quartile_span = np.diff(np.quantile(score[np.unique(people), 0], [.25, .75]))[0]
    result = {}
    for ep in ENDPOINTS:
        y, x, _ = matrices[ep, "carbs"]
        base = np.column_stack([x, guards])
        for name, covariates in (("panel", base), ("panel_after_pcs", np.column_stack([base, terms["pcs"]]))):
            fitted = model.fit_background(covariates, y, weights)
            beta, covariance, p = model.household_test(terms["panel"], fitted, hh, n_people)
            estimate = quartile_span * beta[0]
            width = quartile_span * np.sqrt(covariance[0, 0]) * t.ppf(.975, len(set(hh)) - 1)
            result[f"{ep}__{name}"] = dict(p=p, q75_minus_q25_mg_dl=float(estimate),
                                         interval=[float(estimate - width), float(estimate + width)])
    for row, adjusted in zip(result.values(), model.holm([row["p"] for row in result.values()])):
        row["holm_p"] = float(adjusted)
    return result


def summarize(values, seed):
    return {key: score_losses(value, seed) for key, value in values.items()}


def freeze(out, dataset, score, model, io, fingerprint):
    _, _, matrices, _, weights = dataset
    guards, terms = blocks(dataset, score, model)
    arrays = {"scores": score}
    for ep in ENDPOINTS:
        y, x, _ = matrices[ep, "carbs"]
        x = np.column_stack([x, guards])
        for key, term in terms.items():
            fit = fit_frozen(y, x, weights, term, model)
            for name, value in zip(("background", "projection", "beta"), fit):
                arrays[f"{ep}__{key}__{name}"] = value
    path = out / "frozen_model_TRE_ONLY.npz"
    save_arrays(path, arrays)
    marker = dict(registration=fingerprint, payload=digest(path), additional_outcomes_decoded=False)
    io.write(out / "FROZEN.json", marker)
    return digest(out / "FROZEN.json")


def read_frozen(path, summary, fingerprint):
    reg = load(path / "FROZEN.json")
    model = path / "frozen_model_TRE_ONLY.npz"
    require(digest(path / "FROZEN.json") == summary["frozen"]
            and reg == dict(registration=fingerprint, payload=digest(model), additional_outcomes_decoded=False),
            "Frozen panel weights differ")
    return model


def discover(package, out, commit, smoke):
    archived, _, io, meal, index, pop, _, _, outer, raw, taxa, clr, _ = archived_inputs(package, ARCHIVE)
    model = sys.modules["models"]
    prep, flow, window, old, old_weights = prepared()
    check_sources(prep["sources"])
    check_sources(old["sources"])
    projection = pq.read_table(window, columns=["input_row"], use_threads=False, pre_buffer=False).to_pandas()
    extra = np.sort(projection.input_row.unique())
    reference = read_arrays(old_weights)
    require(np.array_equal(reference["training_rows"], pop) and reference["scores"].shape == (len(index), 10)
            and len(extra) == flow["people"] and len(projection) == flow["meals"]
            and index.iloc[extra].split_unit_id.nunique() == flow["households"]
            and not set(index.iloc[extra].split_unit_id) & set(index.iloc[reference["excluded_rows"]].split_unit_id),
            "Additional household eligibility differs")
    mass = np.zeros(len(index))
    mass[pop], seen = panel_mass(raw["gut"][pop], taxa["gut"])
    mass[extra], seen_extra = panel_mass(raw["gut"][extra], taxa["gut"])
    support = {"discovery": dict(people=len(pop), meals=len(meal), detected_by_species=seen.tolist()),
               "additional": dict(people=len(extra), meals=len(projection), detected_by_species=seen_extra.tolist())}
    pooled = data(meal, model)
    keep, late = assign_periods(meal)
    calendar = data(meal.loc[keep], model, late[keep])
    require(len(calendar[0]) > 0 and len(np.unique(late[keep])) == 2, "Insufficient recording periods")
    support["calendar"] = dict(people=int(calendar[0].input_row.nunique()), meals=len(calendar[0]))
    household = index.iloc[pop].split_unit_id.astype(str).to_numpy()
    groups = np.unique(household)
    families = {"pooled": np.searchsorted(groups, pooled[0].split_unit_id.astype(str)),
                "calendar": np.searchsorted(groups, calendar[0].split_unit_id.astype(str))}
    sources = {**archived["sources"], **prep["sources"], **old["sources"]}
    files = (NEW_FROZEN / "COMPLETE.json", NEW_PREPARED / "COMPLETE.json", window, old_weights)
    sources.update({str(path): digest(path) for path in files})
    record(out, "discovery", commit, io, sources, support, smoke=smoke)
    fingerprint = digest(out / "registration.json")
    split = np.full(len(index), -1)
    pred = {}
    for repeat in ((0,) if smoke else (0, 1)):
        split[pop] = outer[repeat]
        totals = {}
        for fold in ((0,) if smoke else range(5)):
            train = outer[repeat] != fold
            require(not set(household[train]) & set(household[~train]), "Frozen fold crosses households")
            score = np.zeros((len(index), 11))
            training_people = np.isin(np.arange(len(index)), pop[train])
            score[:, :1] = model.train_standardize(mass[:, None], training_people)
            score[pop, 1:] = model.site_scores(raw, taxa, index, pop, train, clr, 5)["both_pc"]
            train_pooled = split[pooled[1]] != fold
            parts = {f"pooled__{ep}": value for ep, value in prediction(
                pooled, score, train_pooled, ~train_pooled, families["pooled"], len(groups), model).items()}
            train_calendar = split[calendar[1]] != fold
            period = late[keep]
            for direction in DIRECTIONS:
                forward = direction == "early_to_late"
                values = prediction(calendar, score, train_calendar & (period != forward),
                                    ~train_calendar & (period == forward), families["calendar"], len(groups), model)
                parts.update({f"{direction}__{ep}": value for ep, value in values.items()})
            save_fold(out / f"r{repeat}_f{fold}.npz", parts, io, fingerprint, repeat, fold)
            add_totals(totals, parts)
            io.emit("BACTEROIDALES_FOLD", repeat=repeat, fold=fold)
        pred[f"r{repeat}"] = summarize(totals, SEED + repeat)
    associations, frozen = {}, None
    if not smoke:
        score = np.column_stack([model.train_standardize(mass[:, None], np.isin(np.arange(len(index)), pop)),
                                 reference["scores"]])
        associations = direct(pooled, score, model)
        frozen = freeze(out, pooled, score, model, io, fingerprint)
    finish(out, io, dict(support=support, prediction=pred, direct=associations, frozen=frozen))
    io.emit("BACTEROIDALES_COMPLETE", stage="discovery", smoke=smoke, **support)


def transfer(package, out, discovery, commit):
    reg, summary, fingerprint = verify_receipt(discovery, commit, "discovery")
    require(not reg["smoke"], "Transfer requires a full discovery fit")
    model_file = read_frozen(discovery, summary, fingerprint)
    arrays = read_arrays(model_file)
    prep, flow, window, _, _ = prepared()
    require(reg["sources"][str(window)] == digest(window), "Additional meal archive differs")
    cfg = load(MEAL / "config.json")
    inputs, io = base_modules(cfg)
    model = sys.modules["models"]
    index_path = package / "inputs/discovery_v2/discovery_covariates_food_TRE_ONLY.parquet"
    index = inputs.frame(pq.read_table(index_path, columns=inputs.INDEX, pre_buffer=False, use_threads=False))
    meals = pq.read_table(window, pre_buffer=False, use_threads=False).to_pandas().merge(
        index[["input_row", "split_unit_id", "age_calendar_years", "sex_code"]], on="input_row", validate="many_to_one")
    require(len(meals) == flow["meals"] and meals.input_row.nunique() == flow["people"]
            and meals.split_unit_id.nunique() == flow["households"], "Additional flow differs")
    dataset = data(meals, model)
    guards, terms = blocks(dataset, arrays["scores"], model)
    hh = meals.split_unit_id.astype(str).to_numpy()
    groups, family = np.unique(hh, return_inverse=True)
    _, _, matrices, _, weights = dataset
    losses, calibrated = {}, {}
    for ep in ENDPOINTS:
        y, x, _ = matrices[ep, "carbs"]
        x = np.column_stack([x, guards])
        baseline = x @ arrays[f"{ep}__panel__background"]
        effects = {}
        for name, term in terms.items():
            proj, beta = [arrays[f"{ep}__{name}__{part}"] for part in ("projection", "beta")]
            effects[name] = (term - x @ proj) @ beta
        parts = ((y - baseline) ** 2, *((y - baseline - effects[name]) ** 2 for name in MODELS), y ** 2)
        losses[ep] = np.stack([np.bincount(family, weights=weights * part, minlength=len(groups)) for part in parts])
        calibrated[ep] = calibration(dataset[0], y, x, weights, effects["panel"], hh, model)
    for row, p in zip(calibrated.values(), model.holm([row.get("p", 1.) for row in calibrated.values()])):
        if "p" in row:
            row["holm_p"] = float(p)
    sources = {**reg["sources"], **prep["sources"]}
    sources.update({str(path): digest(path) for path in (discovery / "COMPLETE.json", discovery / "FROZEN.json", model_file)})
    record(out, "transfer", commit, io, sources, summary["support"], smoke=False)
    payload = out / "additional_losses_TRE_ONLY.npz"
    save_arrays(payload, losses)
    finish(out, io, dict(support=summary["support"], prediction=summarize(losses, SEED + 2),
                         calibration=calibrated, payload=digest(payload)))
    io.emit("BACTEROIDALES_COMPLETE", stage="transfer", people=flow["people"], meals=flow["meals"])


def record(out, stage, commit, io, sources, support, smoke):
    out.mkdir(parents=True, exist_ok=False)
    files = (Path(__file__).resolve(), ROOT / "panel_design.py", *PINS, *TEMPLATE_PINS)
    sources.update({str(path): digest(path) for path in files})
    io.write(out / "registration.json", dict(stage=stage, commit=commit, smoke=smoke, support=support,
             sources=sources, species=list(SPECIES), primary="net_p0", seed=SEED))


def finish(out, io, summary):
    io.write(out / "summary.json", summary)
    io.write(out / "COMPLETE.json", dict(registration=digest(out / "registration.json"),
                                         summary=digest(out / "summary.json")))


def verify_receipt(out, commit, stage):
    reg = load(out / "registration.json")
    summary = load(out / "summary.json")
    fingerprint = digest(out / "registration.json")
    require(reg["stage"] == stage and reg["commit"] == commit and summary["support"] == reg["support"]
            and load(out / "COMPLETE.json") == dict(registration=fingerprint, summary=digest(out / "summary.json")),
            "Panel completion receipt differs")
    check_sources(reg["sources"])
    return reg, summary, fingerprint


def verify(out, commit, stage):
    reg, summary, fingerprint = verify_receipt(out, commit, stage)
    if stage == "transfer":
        path = out / "additional_losses_TRE_ONLY.npz"
        losses = read_arrays(path)
        require(digest(path) == summary["payload"] and set(losses) == set(ENDPOINTS)
                and summarize(losses, SEED + 2) == summary["prediction"], "Transfer readback differs")
    else:
        repeats, folds = ((0,), (0,)) if reg["smoke"] else (range(2), range(5))
        for repeat in repeats:
            totals = {}
            for fold in folds:
                path = out / f"r{repeat}_f{fold}.npz"
                require(load(path.with_suffix(".json")) == dict(
                    registration=fingerprint, repeat=repeat, fold=fold, sha=digest(path)), "Fold receipt differs")
                parts = read_arrays(path)
                require(set(parts) == {f"{scope}__{ep}" for scope in ("pooled", *DIRECTIONS) for ep in ENDPOINTS},
                        "Incomplete fold scopes")
                add_totals(totals, parts)
            require(summarize(totals, SEED + repeat) == summary["prediction"][f"r{repeat}"], "Discovery readback differs")
        if not reg["smoke"]:
            read_frozen(out, summary, fingerprint)
    print(f"BACTEROIDALES_READBACK_PASS {stage}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--stage", choices=("discovery", "transfer"), default="discovery")
    parser.add_argument("--discovery-run", default="discovery-1")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    package = args.package.resolve()
    require(package / "experiments" / ROOT.name == ROOT and re.fullmatch(r"[0-9a-f]{40}", args.commit)
            and all(re.fullmatch(r"[a-zA-Z0-9_-]+", name) for name in (args.run, args.discovery_run))
            and not (args.smoke and args.stage == "transfer"), "Unexpected study, release or run")
    check_sources({str(path): sha for path, sha in {**PINS, **TEMPLATE_PINS}.items()})
    out = ROOT / "runs_TRE_ONLY" / args.run
    if args.verify:
        verify(out, args.commit, args.stage)
    else:
        require(not out.exists(), "Use a fresh run name")
        if args.stage == "discovery":
            discover(package, out, args.commit, args.smoke)
        else:
            transfer(package, out, ROOT / "runs_TRE_ONLY" / args.discovery_run, args.commit)


if __name__ == "__main__":
    main()
