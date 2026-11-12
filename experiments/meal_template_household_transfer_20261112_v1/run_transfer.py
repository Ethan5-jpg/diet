"""Audit, freeze, and evaluate the meal-template interaction in separate HPP households."""

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
from scipy.stats import t

from transfer_design import household_losses, predict, score_summary

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT.parent / "meal_template_modification_20261112_v1"
NEW = ROOT.parent / "meal_new_household_validation_20261112_v1"
ARCHIVE = TEMPLATE.parent / "meal_carbohydrate_modification_20261112_v1" / "runs_TRE_ONLY/models-1"
NEW_FROZEN = NEW / "runs_TRE_ONLY/frozen-1"
NEW_PREPARED = NEW / "runs_TRE_ONLY/prep-discovery-1"
TEMPLATE_DISCOVERY = TEMPLATE / "runs_TRE_ONLY/discovery-1"
PINS = {
    TEMPLATE / "analyze_template.py": "273dab1dc271c88e090d2fac91b720d26ad32af1f04821a71f87894e4ef55371",
    TEMPLATE / "template_design.py": "dee91e52e425485cc6976380e1599761bb79cf2f91c07e2d30f676b934ca682c",
    NEW / "contracts.py": "b191f6a86aa648f9d3760ce58efabe5aba5b874ae3014c1c638582b90177b903",
    NEW / "frozen.py": "e49fe0e495fa212e0cda6accea59fdaae44a8e2def159d88a6a855a1747d79f9",
    NEW / "prepare_holdout.py": "952143d739cc80f296d8129879f955b50ff6696ea66df5cd891b44e65509598a",
}
PAYLOADS = {"audit": "selection_TRE_ONLY.npz", "freeze": "coefficients_TRE_ONLY.npz",
            "score": "household_losses_TRE_ONLY.npz"}
SEED = 202611141

sys.path[:0] = [str(TEMPLATE), str(NEW)]
try:
    from analyze_template import (
        ENDPOINTS, KEYS, MEAL, NUTRIENTS, PINS as TEMPLATE_PINS, archived_inputs, dataset, design,
        digest, eligible_events, food_groups, load, prepare_food, require, scoped_read,
    )
    from contracts import base_modules, check_sources, completed, finish
    from frozen import fit_frozen, save_arrays
    from prepare_holdout import WEIGHTS, WINDOWS
    from template_design import TOTALS, reconstruct, scopes, summary
finally:
    del sys.path[:2]


def stage_path(name):
    return ROOT / "runs_TRE_ONLY" / name


def checked_stage(path, stage, commit):
    reg, result, payload, _ = completed(path, PAYLOADS[stage])
    require(reg["stage"] == stage and reg["commit"] == commit, "Unexpected source stage or release")
    check_sources(reg["sources"])
    return reg, result, payload


def read_arrays(path):
    with np.load(BytesIO(path.read_bytes()), allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


def write_stage(out, stage, commit, sources, arrays, results, io):
    out.mkdir(parents=True, exist_ok=False)
    local = (Path(__file__).resolve(), ROOT / "transfer_design.py", *PINS, *TEMPLATE_PINS)
    sources = {**sources, **{str(path): digest(path) for path in local}}
    io.write(out / "registration.json", dict(
        stage=stage, commit=commit, sources=sources,
        primary="net_p0", secondary="peak_p0", seed=SEED,
    ))
    payload = out / PAYLOADS[stage]
    save_arrays(payload, arrays)
    finish(out, results, payload, io)
    io.emit("TEMPLATE_TRANSFER_STAGE", stage=stage, **results)


def prepared():
    prep, flow, window, prep_receipt = completed(NEW_PREPARED, WINDOWS)
    ref, _, weights, ref_receipt = completed(NEW_FROZEN, WEIGHTS)
    require(not prep["smoke"] and prep["frozen_receipt"] == ref_receipt,
            "Additional meals do not match the frozen reference")
    return prep, flow, window, ref, weights


def audit(package, out, commit):
    prep, flow, window, reference, weights_path = prepared()
    check_sources(prep["sources"])
    check_sources(reference["sources"])
    cfg = load(MEAL / "config.json")
    inputs, io = base_modules(cfg)
    index_path = package / "inputs/discovery_v2/discovery_covariates_food_TRE_ONLY.parquet"
    require(digest(index_path) == reference["sources"][str(index_path)], "Archived index changed")
    index = inputs.frame(pq.read_table(index_path, columns=inputs.INDEX, use_threads=False, pre_buffer=False))
    require(index.final_role.eq("discovery").all() and np.array_equal(index.input_row, np.arange(len(index))),
            "Unexpected archived index order")
    columns = [*KEYS, "date", "local_hour", "full_offsets", *TOTALS]
    meals = pq.read_table(window, columns=columns, use_threads=False, pre_buffer=False).to_pandas()
    meals = meals.merge(index[["input_row", "split_unit_id"]], on="input_row", validate="many_to_one")
    roster = index.loc[index.input_row.isin(meals.input_row)]
    weights = read_arrays(weights_path)
    excluded = index.iloc[weights["excluded_rows"]].split_unit_id
    require(meals.full_offsets.all() and len(meals) == flow["meals"] and len(roster) == flow["people"]
            and roster.split_unit_id.nunique() == flow["households"]
            and not roster.split_unit_id.isin(excluded).any(), "Additional meal or household contract differs")
    food_path = package.parent / "analysis/shared/food_events_clean_TRE_ONLY.parquet"
    food_sha = digest(food_path)
    require(food_sha == prep["sources"][str(food_path)], "Additional food event source changed")
    fields = ["collection_timestamp", "local_timestamp", "food_id", "qc_primary_eligible",
              "qc_same_food_time_n", "qc_exact_signature_n", *NUTRIENTS]
    food = pd.concat(food_groups(scoped_read(food_path, fields, roster, inputs)).values(), ignore_index=True)
    complete = reconstruct(meals, food, NUTRIENTS, cfg["max_food_episode_span_minutes"] * 60, eligible_events)
    pooled, _ = scopes(complete)
    support = summary(pooled)
    sources = {**prep["sources"], **reference["sources"], str(food_path): food_sha}
    for path in (NEW_PREPARED / "COMPLETE.json", NEW_FROZEN / "COMPLETE.json", window, weights_path):
        sources[str(path)] = digest(path)
    selection = {name: pooled[name].to_numpy() for name in (*KEYS, "group", "late")}
    write_stage(out, "audit", commit, sources, selection, dict(
        archive_meals=len(meals), complete_food_id_meals=len(complete), support=support,
        additional_outcomes_decoded=False,
    ), io)


def freeze(package, out, commit, audit_path):
    audit_reg, audited, _ = checked_stage(audit_path, "audit", commit)
    support = audited["support"]
    require(support["effective_households"] > 1 and support["adjusted_carb_sd_g"] > 1e-6,
            "Insufficient independent carbohydrate variation in the additional households")
    locked = load(TEMPLATE_DISCOVERY / "COMPLETE.json")
    training_reg = load(TEMPLATE_DISCOVERY / "registration.json")
    require(locked["registration"] == digest(TEMPLATE_DISCOVERY / "registration.json")
            and locked["summary"] == digest(TEMPLATE_DISCOVERY / "summary.json")
            and not training_reg["smoke"] and not training_reg["audit"], "Discovery template run differs")
    check_sources(training_reg["sources"])
    pooled, _, current_support, sources, io = prepare_food(package)
    require(current_support == training_reg["support"], "Discovery template support changed")
    archive, _, _, meals, index, population, _, _, _, _, _, _, _ = archived_inputs(package, ARCHIVE)
    model = sys.modules["models"]
    _, _, _, _, weights_path = prepared()
    reference = read_arrays(weights_path)
    scores = reference["scores"]
    require(scores.shape == (len(index), 10) and np.array_equal(population, reference["training_rows"]),
            "Frozen score axes do not match the discovery population")
    data = dataset(meals, pooled, np.arange(len(index)), model)
    _, _, matrices, _, weights = data
    controls, interaction = design(data, scores, model)
    arrays = {"scores": scores}
    for endpoint in ENDPOINTS:
        y, x, _ = matrices[endpoint, "carbs"]
        coefficients = fit_frozen(y, np.column_stack([x, controls]), weights, interaction, model)
        for name, part in zip(("background", "projection", "beta"), coefficients):
            arrays[f"{endpoint}_{name}"] = part
        arrays[f"{endpoint}_reference_beta"] = reference[f"{endpoint}_beta"]
    sources = {**audit_reg["sources"], **training_reg["sources"], **archive["sources"], **sources}
    for path in (audit_path / "COMPLETE.json", TEMPLATE_DISCOVERY / "COMPLETE.json"):
        sources[str(path)] = digest(path)
    write_stage(out, "freeze", commit, sources, arrays, dict(
        discovery=current_support["pooled"], additional=support, additional_outcomes_decoded=False,
    ), io)


def calibration(data, y, x, weights, correction, household, model):
    fitted = model.fit_background(x, y, weights)
    group_count = data.group.nunique()
    cluster_count = len(np.unique(household))
    residual, _, _ = model.orthogonal_block(correction[:, None], fitted)
    if (cluster_count < 2 or len(y) - group_count - fitted[1].shape[1] <= 1
            or np.linalg.norm(residual) < 1e-10):
        return {"status": "insufficient_calibration_information"}
    try:
        beta, covariance, p = model.household_test(correction[:, None], fitted, household, group_count)
    except np.linalg.LinAlgError:
        return {"status": "singular_calibration_covariance"}
    margin = np.sqrt(covariance[0, 0]) * t.ppf(.975, cluster_count - 1)
    return dict(coefficient=float(beta[0]), interval=[float(beta[0] - margin), float(beta[0] + margin)], p=p)


def evaluate(package, out, commit, audit_path, frozen_path):
    audit_reg, audited, selection_path = checked_stage(audit_path, "audit", commit)
    reg, frozen, coefficients = checked_stage(frozen_path, "freeze", commit)
    require(frozen["additional_outcomes_decoded"] is False and audited["additional_outcomes_decoded"] is False
            and frozen["additional"] == audited["support"]
            and reg["sources"][str(audit_path / "COMPLETE.json")] == digest(audit_path / "COMPLETE.json"),
            "Frozen model does not match the outcome-blind selection")
    selection = pd.DataFrame(read_arrays(selection_path))
    params = read_arrays(coefficients)
    # The selected rows and model have both been verified before decoding any additional outcomes.
    _, _, window, _, _ = prepared()
    require(digest(window) == audit_reg["sources"][str(window)], "Prepared outcome window changed")
    outcomes = pq.read_table(window, use_threads=False, pre_buffer=False).to_pandas()
    cfg = load(MEAL / "config.json")
    inputs, io = base_modules(cfg)
    model = sys.modules["models"]
    index_path = package / "inputs/discovery_v2/discovery_covariates_food_TRE_ONLY.parquet"
    index = inputs.frame(pq.read_table(index_path, columns=inputs.INDEX, use_threads=False, pre_buffer=False))
    outcomes = outcomes.merge(index[["input_row", "split_unit_id", "age_calendar_years", "sex_code"]],
                              on="input_row", validate="many_to_one")
    data = dataset(outcomes, selection, np.arange(len(index)), model)
    meal, _, matrices, _, weights = data
    controls, interaction = design(data, params["scores"], model)
    household = meal.split_unit_id.astype(str).to_numpy()
    ids, family = np.unique(household, return_inverse=True)
    require(len(ids) == audited["support"]["households"], "Additional selected households changed")
    losses, scores, calibrated = {}, {}, {}
    for endpoint in ENDPOINTS:
        y, x, _ = matrices[endpoint, "carbs"]
        x = np.column_stack([x, controls])
        names = ("background", "projection", "beta", "reference_beta")
        baseline, change, reference = predict(x, interaction, *(params[f"{endpoint}_{name}"] for name in names))
        losses[endpoint] = household_losses(y, weights, family, baseline, change, reference, len(ids))
        scores[endpoint] = score_summary(losses[endpoint], SEED)
        calibrated[endpoint] = calibration(meal, y, x, weights, change, household, model)
    for row, adjusted in zip(calibrated.values(), model.holm([row.get("p", 1.) for row in calibrated.values()])):
        if "p" in row:
            row["holm_p"] = float(adjusted)
    sources = {**audit_reg["sources"], **reg["sources"]}
    sources[str(frozen_path / "COMPLETE.json")] = digest(frozen_path / "COMPLETE.json")
    write_stage(out, "score", commit, sources, losses, dict(
        support=audited["support"], scores=scores, calibration=calibrated,
    ), io)


def verify(path, stage, commit):
    _, result, payload = checked_stage(path, stage, commit)
    arrays = read_arrays(payload)
    if stage == "score":
        require(set(arrays) == set(ENDPOINTS) and set(result["calibration"]) == set(ENDPOINTS)
                and {ep: score_summary(arrays[ep], SEED) for ep in ENDPOINTS} == result["scores"],
                "Frozen template scoring readback differs")
    elif stage == "audit":
        require(result["additional_outcomes_decoded"] is False
                and set(arrays) == {*KEYS, "group", "late"}
                and len(arrays["start"]) == result["support"]["meals"],
                "Blinded selection readback differs")
    else:
        expected = {"scores"} | {f"{ep}_{key}" for ep in ENDPOINTS
                                 for key in ("background", "projection", "beta", "reference_beta")}
        require(result["additional_outcomes_decoded"] is False and set(arrays) == expected
                and all(np.isfinite(array).all() for array in arrays.values()), "Frozen coefficient readback differs")
    print(f"TEMPLATE_TRANSFER_READBACK_PASS {stage}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--stage", choices=tuple(PAYLOADS), required=True)
    parser.add_argument("--audit-run", default="support-1")
    parser.add_argument("--frozen-run", default="frozen-1")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    package = args.package.resolve()
    require(package / "experiments" / ROOT.name == ROOT and re.fullmatch(r"[0-9a-f]{40}", args.commit)
            and all(re.fullmatch(r"[a-zA-Z0-9_-]+", value)
                    for value in (args.run, args.audit_run, args.frozen_run)), "Unexpected stage, path or release")
    check_sources({str(path): sha for path, sha in {**PINS, **TEMPLATE_PINS}.items()})
    out, audit_path, frozen_path = map(stage_path, (args.run, args.audit_run, args.frozen_run))
    if args.verify:
        verify(out, args.stage, args.commit)
    else:
        require(not out.exists(), "Use a fresh stage identifier")
        if args.stage == "audit":
            audit(package, out, args.commit)
        elif args.stage == "freeze":
            freeze(package, out, args.commit, audit_path)
        else:
            evaluate(package, out, args.commit, audit_path, frozen_path)


if __name__ == "__main__":
    main()
