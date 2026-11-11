"""Fit the original interaction on archived households and evaluate once on new ones."""

import argparse
from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import t

from contracts import (
    ARCHIVE, ENDPOINTS, KEYS, MEAL, ROOT, archived_contract, archived_inputs, base_modules, bases,
    check_sources, completed, digest, finish, load, require, run_path, validate_args, verified_roster,
)
from prepare_holdout import WEIGHTS, WINDOWS, eligible_windows

LOSS = "paired_household_losses_TRE_ONLY.npz"
PENALTY = .03
SEED = 202611130


def save_arrays(path, arrays):
    stream = BytesIO()
    np.savez_compressed(stream, **arrays)
    data = stream.getvalue()
    path.write_bytes(data)
    require(digest(path) == sha256(data).hexdigest(), "Saved arrays differ")


def fit_frozen(y, x, w, z, model):
    background = model.fit_background(x, y, w)
    residual, target, projection = model.orthogonal_block(z, background)
    gram = residual.T @ residual + PENALTY * w.sum() * np.eye(z.shape[1])
    return background[0], projection, np.linalg.solve(gram, residual.T @ target)


def prepare_design(meals, scores, model):
    matrices, dose = bases(meals, model)
    person = meals.person_index.to_numpy()
    terms, _ = model.center(dose["carbs"][:, None] * scores[meals.input_row.to_numpy(int)], person)
    return matrices, terms


def freeze(args):
    package = validate_args(args)
    out = run_path(args.run)
    require(not out.exists(), "Use a fresh model identifier")
    archived_contract()
    (receipt, report, io, meal, index, pop, _, _, _, raw, taxa, clr, _) = archived_inputs(package, ARCHIVE)
    cfg = load(MEAL / "config.json")
    inputs, _ = base_modules(cfg)
    excluded, support = verified_roster(package, cfg, False, inputs)
    require(set(pop).issubset(excluded.input_row), "Archived training rows differ from the registered roster")
    model = __import__("models")
    everyone = index.input_row.to_numpy(int)
    training = np.isin(everyone, pop)
    scores = model.site_scores(raw, taxa, index, everyone, training, clr, 5)["both_pc"]
    matrices, z = prepare_design(meal, scores, model)
    arrays = dict(scores=scores, training_rows=pop, excluded_rows=excluded.input_row.to_numpy(int))
    for endpoint in ENDPOINTS:
        y, x, w = matrices[endpoint, "carbs"]
        background, projection, beta = fit_frozen(y, x, w, z, model)
        arrays.update({f"{endpoint}_background": background, f"{endpoint}_projection": projection,
                       f"{endpoint}_beta": beta, f"{endpoint}_slope_sd": np.std(scores[pop] @ beta, ddof=1)})
    files = (Path(__file__).resolve(), ROOT / "prepare_holdout.py", ROOT / "contracts.py", MEAL / "paired.py",
             ARCHIVE / "registration.json", ARCHIVE / "summary.json",
             *(MEAL.parent / "diet_daily_modification_20261112_v1" / name for name in ("inputs.py", "models.py", "run.py")))
    sources = {**receipt["sources"], **support, **{str(p): digest(p) for p in files}}
    out.mkdir(parents=True, exist_ok=False)
    io.write(out / "registration.json", dict(commit=args.commit, sources=sources, primary="net_p0",
                                            exposures="carbohydrate_grams", marker="both_pc5", penalty=PENALTY,
                                            endpoint_order=list(ENDPOINTS), no_new_outcomes_read=True))
    data = out / WEIGHTS
    save_arrays(data, arrays)
    check_sources(sources)
    summary = dict(flow=report["flow"], score_width=scores.shape[1], excluded_people=len(excluded),
                   archived_oof={k: report["prediction"][f"{k}_carbs_both_pc5"] for k in ENDPOINTS})
    finish(out, summary, data, io)
    io.emit("NEW_MEAL_MODEL_FROZEN", **summary["flow"], score_width=scores.shape[1])


def check_prepared(path):
    reg, summary, payload, receipt = completed(path, WINDOWS)
    check_sources(reg["sources"])
    batches = []
    parts = sorted(path.glob("batch_*.json"))
    require(len(parts) == summary["batches"], "Incomplete preparation checkpoint count")
    for part in parts:
        record = load(part)
        require(record["registration"] == receipt["registration"], "Preparation batch registration differs")
        if record["rows"]:
            batch = path / f"{part.stem}_TRE_ONLY.parquet"
            require(digest(batch) == record["sha"], "Meal batch bytes differ")
            frame = pq.read_table(batch, use_threads=False, pre_buffer=False).to_pandas()
            require(len(frame) == record["rows"], "Meal batch count differs")
            batches.append(frame)
    require(bool(batches), "No verified meal batches")
    windows = pd.concat(batches, ignore_index=True)
    require(not windows.duplicated(KEYS).any(), "Archived batches contain duplicate meal anchors")
    selected = eligible_windows(windows, load(MEAL / "config.json")).reset_index(drop=True)
    final = pq.read_table(payload, use_threads=False, pre_buffer=False).to_pandas()
    pd.testing.assert_frame_equal(selected, final, check_exact=True)
    require(len(final) == summary["meals"] and final.input_row.nunique() == summary["people"],
            "Reaggregated meal flow differs")
    return reg, summary, final, receipt


def summarize_losses(arrays):
    results = {}
    for endpoint in ENDPOINTS:
        null, alt, total = arrays[endpoint]
        require(total.sum() > 0 and np.isfinite(arrays[endpoint]).all(), "Invalid held-out household loss")
        draws = np.random.default_rng(SEED).integers(0, len(total), (1999, len(total)), dtype=np.int32)
        denom = total[draws].sum(axis=1)
        require((denom > 0).all(), "Nonpositive paired bootstrap denominator")
        gain = null - alt
        results[endpoint] = dict(baseline_r2=float(1 - null.sum() / total.sum()),
                                 augmented_r2=float(1 - alt.sum() / total.sum()),
                                 delta_r2=float(gain.sum() / total.sum()),
                                 interval=np.quantile(gain[draws].sum(axis=1) / denom, [.025, .975]).tolist())
    return results


def evaluate(args):
    package = validate_args(args)
    out = run_path(args.run)
    require(not out.exists(), "Use a fresh evaluation identifier")
    prep, flow, meals, window_receipt = check_prepared(run_path(args.prepared))
    frozen = run_path(args.frozen)
    model_reg, trained, weights_file, model_receipt = completed(frozen, WEIGHTS)
    require(not prep["smoke"] and prep["frozen"] == args.frozen and prep["frozen_receipt"] == model_receipt
            and prep["commit"] == model_reg["commit"] == args.commit and model_reg["penalty"] == PENALTY,
            "Evaluation requires the registered training-only model and full preparation")
    check_sources(model_reg["sources"])
    cfg = load(MEAL / "config.json")
    inputs, io = base_modules(cfg)
    index_path = package / "inputs/discovery_v2/discovery_covariates_food_TRE_ONLY.parquet"
    index = inputs.frame(pq.read_table(index_path, columns=inputs.INDEX, use_threads=False, pre_buffer=False))
    require(index.final_role.eq("discovery").all() and np.array_equal(index.input_row, np.arange(len(index))),
            "Frozen discovery index order changed")
    meals = meals.merge(index[["input_row", "age_calendar_years", "sex_code", "split_unit_id"]],
                        on="input_row", validate="many_to_one")
    require(len(meals) == flow["meals"] and meals.split_unit_id.nunique() == flow["households"],
            "A validation meal lost its original household")
    people, person = np.unique(meals.input_row, return_inverse=True)
    meals["person_index"] = person
    household = meals.split_unit_id.astype(str).to_numpy()
    family_ids, family = np.unique(household, return_inverse=True)
    with np.load(BytesIO(weights_file.read_bytes()), allow_pickle=False) as stored:
        weights = {k: stored[k] for k in stored.files}
    require(weights["scores"].shape == (len(index), 10) and len(weights["training_rows"]) == trained["flow"]["people"]
            and not set(household) & set(index.iloc[weights["excluded_rows"]].split_unit_id.astype(str)),
            "A validation household crossed the original eligibility boundary")
    model = __import__("models")
    matrices, z = prepare_design(meals, weights["scores"], model)
    sources = {**model_reg["sources"], **prep["sources"], **{str(p): digest(p) for p in
               (frozen / "COMPLETE.json", run_path(args.prepared) / "COMPLETE.json", weights_file)}}
    out.mkdir(parents=True, exist_ok=False)
    io.write(out / "registration.json", dict(commit=args.commit, sources=sources, frozen=args.frozen,
                                            prepared=args.prepared, frozen_receipt=model_receipt,
                                            windows_receipt=window_receipt, seed=SEED, primary="net_p0",
                                            calibration_scope="held-out association of the frozen direction"))
    losses, calibration = {}, {}
    for endpoint in ENDPOINTS:
        y, x, w = matrices[endpoint, "carbs"]
        null = x @ weights[f"{endpoint}_background"]
        change = (z - x @ weights[f"{endpoint}_projection"]) @ weights[f"{endpoint}_beta"]
        alt = null + change
        losses[endpoint] = np.stack([np.bincount(family, weights=w * value, minlength=len(family_ids))
                                    for value in ((y - null) ** 2, (y - alt) ** 2, y ** 2)])
        fitted = model.fit_background(x, y, w)
        beta, covariance, p = model.household_test(change[:, None], fitted, household, len(people))
        interval = beta[0] + np.array([-1., 1.]) * np.sqrt(covariance[0, 0]) * t.ppf(.975, len(family_ids) - 1)
        scale = weights[f"{endpoint}_slope_sd"]
        calibration[endpoint] = dict(coefficient=float(beta[0]), ci95=interval.tolist(), p=p,
                                     effect_mg_dl_per_training_slope_sd=float(beta[0] * scale),
                                     effect_ci95=(interval * scale).tolist())
    for row, adjusted in zip(calibration.values(), model.holm([row["p"] for row in calibration.values()])):
        row["holm_p"] = float(adjusted)
    payload = out / LOSS
    save_arrays(payload, losses)
    check_sources(sources)
    predictions = summarize_losses(losses)
    finish(out, dict(flow=flow, prediction=predictions, calibration=calibration, original_oof=trained["archived_oof"]),
           payload, io)
    io.emit("NEW_MEAL_VALIDATION_COMPLETE", **{k: flow[k] for k in ("people", "households", "meals")},
            primary_prediction=predictions["net_p0"], primary_calibration=calibration["net_p0"])


def verify(args):
    out = run_path(args.run)
    reg, result, data, _ = completed(out, LOSS)
    check_sources(reg["sources"])
    training, _, _, frozen = completed(run_path(reg["frozen"]), WEIGHTS)
    _, flow, _, prepared = check_prepared(run_path(reg["prepared"]))
    require(reg["frozen_receipt"] == frozen and reg["windows_receipt"] == prepared
            and training["commit"] == reg["commit"] and flow == result["flow"], "Stage receipt differs")
    with np.load(BytesIO(data.read_bytes()), allow_pickle=False) as saved:
        require(set(saved.files) == set(ENDPOINTS), "Paired outcome family is incomplete")
        require(summarize_losses(saved) == result["prediction"], "Paired household readback differs")
    print("NEW_MEAL_VALIDATION_READBACK_PASS", flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="stage", required=True)
    for name in ("freeze", "evaluate", "verify", "check-prep"):
        sub = commands.add_parser(name)
        sub.add_argument("--run", required=True)
        if name in ("freeze", "evaluate"):
            sub.add_argument("--package", type=Path, required=True)
            sub.add_argument("--commit", required=True)
        if name == "evaluate":
            sub.add_argument("--prepared", required=True)
            sub.add_argument("--frozen", required=True)
    args = cli.parse_args()
    if args.stage == "check-prep":
        _, flow, _, _ = check_prepared(run_path(args.run))
        print("NEW_MEAL_PREP_READBACK_PASS", flow["batches"], flush=True)
    else:
        {"freeze": freeze, "evaluate": evaluate, "verify": verify}[args.stage](args)


if __name__ == "__main__":
    main()
