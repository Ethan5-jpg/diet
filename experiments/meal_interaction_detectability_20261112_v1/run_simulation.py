"""Inject known microbiome-by-carbohydrate signals into HPP outcome residuals."""

import argparse
from io import BytesIO
import os
from pathlib import Path
import re
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from simulation_design import AMPLITUDES, METHODS, aggregate, household_moments, prediction_errors, recoverability

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT.parent / "meal_template_modification_20261112_v1"
TRANSFER = ROOT.parent / "meal_template_household_transfer_20261112_v1"
TRANSFER_COMMIT = "43d6fccb1be8bb97978bdf28d71dfc7088a41fba"
PINS = {
    TEMPLATE / "analyze_template.py": "273dab1dc271c88e090d2fac91b720d26ad32af1f04821a71f87894e4ef55371",
    TEMPLATE / "template_design.py": "dee91e52e425485cc6976380e1599761bb79cf2f91c07e2d30f676b934ca682c",
    TRANSFER / "run_transfer.py": "6ab3de9b99f4b8d0354b390dbb023927e504be847de999daa7e135de41c90783",
    TRANSFER / "transfer_design.py": "ffa1cdd04ddddfb78934bbe4cc45c9af842243d4869ad015eda0d4d699d15403",
}
ENDPOINTS = ("net_p0", "peak_p0")
SCOPES = ("archived", "template", "transfer")
DIRECTIONS = ("gut_pc1", "distributed")
SEED = 202611143

sys.path[:0] = [str(TEMPLATE), str(TRANSFER)]
try:
    from analyze_template import archived_inputs, bases, dataset, design, digest, load, prepare_food, require
    from run_transfer import ARCHIVE, checked_stage, check_sources, prepared, read_arrays, save_arrays, finish
finally:
    del sys.path[:2]


def matrices(data, score, endpoint, model, guarded):
    _, _, base, _, weights = data
    y, x, _ = base[endpoint, "carbs"]
    controls, tested = design(data, score, model)
    return y, np.column_stack([x, controls]) if guarded else x, tested, weights


def truth(data, direction, model):
    meal, person, _, dose, _ = data
    centered, _ = model.center((dose * direction[person])[:, None], meal.group.to_numpy())
    return centered[:, 0]


def route(scope, panels, scores, canonical, endpoint, model, assignment, family, signs, n_boot, seed):
    training_data = panels["archived" if scope == "archived" else "template"]
    evaluation_data = panels["additional"] if scope == "transfer" else training_data
    guarded = scope != "archived"
    trans = scope == "transfer"
    train_y, train_x, _, train_w = matrices(training_data, canonical, endpoint, model, guarded)
    test_y, test_x, _, test_w = matrices(evaluation_data, canonical, endpoint, model, guarded)
    nuisance = model.fit_background(train_x, train_y, train_w)[0]
    train_mean, test_mean = train_x @ nuisance, test_x @ nuisance

    legs, residuals = [], np.zeros_like(train_y)
    for fold, score in enumerate(scores):
        y, x, z, w = matrices(training_data, score, endpoint, model, guarded)
        group = assignment[training_data[1]] != fold
        val = ~group
        require(group.any() and val.any(), "Empty frozen training or validation fold")
        residuals[val] = y[val] - x[val] @ model.fit_background(x[group], y[group], w[group])[0]
        if not trans:
            legs.append((group, val, x[group], x[val], z[group], z[val]))
    if trans:
        _, _, z_train, _ = matrices(training_data, canonical, endpoint, model, guarded)
        _, _, z_test, _ = matrices(evaluation_data, canonical, endpoint, model, guarded)
        legs = [(np.ones(len(train_y), bool), np.ones(len(test_y), bool), train_x, test_x, z_train, z_test)]

    train_family = family[training_data[1]]
    test_family = family[evaluation_data[1]]
    _, compact = np.unique(test_family, return_inverse=True)
    n_households = int(compact.max()) + 1
    train_noise = train_mean[:, None] + residuals[:, None] * signs[train_family]
    test_noise = test_mean[:, None] + (test_y - test_mean)[:, None] * signs[test_family] if trans else train_noise

    directions = {
        "gut_pc1": canonical[:, 5],
        "distributed": canonical @ np.array([1., -1., 1., -1., 1., -1., 1., -1., 1., -1.]),
    }
    training_people = np.unique(training_data[1])
    result = {}
    for name, value in directions.items():
        direction = value / value[training_people].std()
        u_train, u_test = truth(training_data, direction, model), truth(evaluation_data, direction, model)
        synthetic_train = np.column_stack([train_noise, u_train])
        synthetic_test = np.column_stack([test_noise, u_test])
        totals = {}
        for train, valid, x_tr, x_va, z_tr, z_va in legs:
            errors = prediction_errors(x_tr, x_va, z_tr, z_va, u_train[train], u_test[valid],
                                       synthetic_train[train], synthetic_test[valid], train_w[train])
            parts = household_moments(errors, compact[valid], test_w[valid], n_households)
            for key, values in parts.items():
                totals[key] = totals.get(key, 0) + values
        metrics = recoverability(totals, seed, n_boot)
        result.update({f"{scope}__{endpoint}__{name}__{method}": value for method, value in metrics.items()})
    return result


def sources_and_panels(package):
    audit_path = TRANSFER / "runs_TRE_ONLY/support-1"
    score_path = TRANSFER / "runs_TRE_ONLY/validation-1"
    reg, _, selection = checked_stage(audit_path, "audit", TRANSFER_COMMIT)
    score_reg, _, _ = checked_stage(score_path, "score", TRANSFER_COMMIT)
    receipt, _, io, meal, index, pop, _, _, outer, raw, taxa, clr, _ = archived_inputs(package, ARCHIVE)
    model = sys.modules["models"]
    pooled, _, support, sources, _ = prepare_food(package)
    _, _, window, _, weights_file = prepared()
    require(digest(window) == reg["sources"][str(window)], "Additional simulation window differs")
    canonical = read_arrays(weights_file)
    require(np.array_equal(pop, canonical["training_rows"]) and canonical["scores"].shape == (len(index), 10),
            "Canonical axes differ")
    archived_meal = meal.assign(group=meal.person_index)
    base, dose = bases(archived_meal, model)
    archived = (archived_meal, meal.input_row.to_numpy(int), base, dose["carbs"], base["net_p0", "carbs"][2])
    everyone = np.arange(len(index))
    template = dataset(meal, pooled, everyone, model)
    extra = pq.read_table(window, use_threads=False, pre_buffer=False).to_pandas().merge(
        index[["input_row", "split_unit_id", "age_calendar_years", "sex_code"]],
        on="input_row", validate="many_to_one",
    )
    additional = dataset(extra, pd.DataFrame(read_arrays(selection)), everyone, model)
    assignment = np.full(len(index), -1)
    assignment[pop] = outer[0]
    require((assignment[template[1]] >= 0).all() and (assignment[additional[1]] < 0).all(),
            "Simulation evaluation population crosses discovery")
    _, family = np.unique(index.split_unit_id.astype(str), return_inverse=True)
    require(not set(family[pop]) & set(family[additional[1]]), "Simulation households overlap")
    scores = []
    for fold in range(5):
        train = outer[0] != fold
        require(not set(family[pop[train]]) & set(family[pop[~train]]), "Frozen fold crosses households")
        score = np.zeros_like(canonical["scores"])
        score[pop] = model.site_scores(raw, taxa, index, pop, train, clr, 5)["both_pc"]
        scores.append(score)
    sources = {**score_reg["sources"], **reg["sources"], **receipt["sources"], **sources}
    files = (audit_path / "COMPLETE.json", score_path / "COMPLETE.json", weights_file,
             window, Path(__file__).resolve(), ROOT / "simulation_design.py", *PINS)
    sources.update({str(path): digest(path) for path in files})
    panels = dict(archived=archived, template=template, additional=additional)
    flows = {name: dict(people=int(len(np.unique(data[1]))), meals=len(data[1]),
                        households=int(len(np.unique(family[data[1]])))) for name, data in panels.items()}
    return panels, scores, canonical["scores"], model, assignment, family, sources, flows, support, io


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--run", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    package = args.package.resolve()
    require(package / "experiments" / ROOT.name == ROOT and re.fullmatch(r"[0-9a-f]{40}", args.commit)
            and re.fullmatch(r"[a-zA-Z0-9_-]+", args.run), "Unexpected study, release or run")
    check_sources({str(path): sha for path, sha in PINS.items()})
    out = ROOT / "runs_TRE_ONLY" / args.run
    payload = out / "simulation_replicates_TRE_ONLY.npz"
    if args.verify:
        from run_transfer import completed
        reg, report, _, _ = completed(out, payload.name)
        check_sources(reg["sources"])
        arrays = read_arrays(payload)
        names = {f"{scope}__{ep}__{direction}__{method}" for scope in SCOPES for ep in ENDPOINTS
                 for direction in DIRECTIONS for method in METHODS}
        require(reg["commit"] == args.commit and report["flows"] == reg["flows"] and set(arrays) == names
                and tuple(reg["amplitudes"]) == AMPLITUDES
                and all(x.shape == (len(AMPLITUDES), 2, reg["replicates"]) and np.isfinite(x).all()
                        for x in arrays.values())
                and aggregate(arrays) == report["recoverability"], "Simulation readback differs")
        print("DETECTABILITY_READBACK_PASS", flush=True)
        return
    require(not out.exists(), "Use a fresh simulation run")
    panels, scores, canonical, model, assignment, family, sources, flows, support, io = sources_and_panels(package)
    replicates, n_boot = (8, 99) if args.smoke else (128, 399)
    rng = np.random.default_rng(SEED)
    signs = rng.choice([-1., 1.], size=(int(family.max()) + 1, replicates))
    out.mkdir(parents=True, exist_ok=False)
    io.write(out / "registration.json", dict(commit=args.commit, smoke=args.smoke, sources=sources,
              replicates=replicates, paired_draws=n_boot, amplitudes=AMPLITUDES, seed=SEED, flows=flows))
    values = {}
    for scope in SCOPES:
        for ep in ENDPOINTS:
            values.update(route(scope, panels, scores, canonical, ep, model, assignment,
                                family, signs, n_boot, SEED + 1))
            io.emit("DETECTABILITY_CELL", scope=scope, endpoint=ep, smoke=args.smoke)
    save_arrays(payload, values)
    finish(out, dict(flows=flows, template_support=support, recoverability=aggregate(values)), payload, io)
    io.emit("DETECTABILITY_COMPLETE", smoke=args.smoke, replicates=replicates, paired_draws=n_boot)


if __name__ == "__main__":
    main()
