"""Evaluate shared timing-offset coefficients against the exact original meal model."""

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

from shared_fit import ENDPOINTS, PENALTY, SEED, fit_shared, household_losses, summarize

ROOT = Path(__file__).resolve().parent
MEAL = ROOT.parent / "meal_carbohydrate_modification_20261112_v1"
NEW = ROOT.parent / "meal_new_household_validation_20261112_v1"
sys.path[:0] = [str(MEAL), str(NEW)]
try:
    from analyze import bases
    from contracts import archived_contract, check_sources, completed, load, require
    from frozen import LOSS, WEIGHTS, WINDOWS, prepare_design, save_arrays
    from paired import archived_inputs
    from prepare import digest
finally:
    del sys.path[:2]

PINS = {
    NEW / "contracts.py": "b191f6a86aa648f9d3760ce58efabe5aba5b874ae3014c1c638582b90177b903",
    NEW / "frozen.py": "e49fe0e495fa212e0cda6accea59fdaae44a8e2def159d88a6a855a1747d79f9",
    NEW / "prepare_holdout.py": "952143d739cc80f296d8129879f955b50ff6696ea66df5cd891b44e65509598a",
}


def archive_losses(parts, path):
    receipt = load(path)
    for endpoint in ENDPOINTS:
        earlier = receipt["results"][f"{endpoint}_carbs_both_pc5"]
        computed = parts[endpoint][[0, 1, 3]].sum(axis=1)
        np.testing.assert_allclose(computed, [earlier[key] for key in ("sse_null", "sse", "tss")],
                                   rtol=1e-10, atol=1e-9)


def cohort_transfer(out, meal, index, pop, model, io, fingerprint):
    root = NEW / "runs_TRE_ONLY"
    frozen, prepared, validated = [root / name for name in ("frozen-1", "prep-discovery-1", "validation-1")]
    training, _, wfile, frozen_receipt = completed(frozen, WEIGHTS)
    prep, flow, path, prep_receipt = completed(prepared, WINDOWS)
    previous, reference, lfile, _ = completed(validated, LOSS)
    require(not prep["smoke"] and previous["frozen_receipt"] == frozen_receipt
            and previous["windows_receipt"] == prep_receipt and reference["flow"] == flow,
            "Earlier validation stages differ")
    check_sources(training["sources"])
    check_sources(prep["sources"])
    check_sources(previous["sources"])
    with np.load(BytesIO(wfile.read_bytes()), allow_pickle=False) as saved:
        weights = {key: saved[key] for key in saved.files}
    require(np.array_equal(weights["training_rows"], pop), "Frozen training population differs")
    matrices, term = prepare_design(meal, weights["scores"], model)
    fits, shared = fit_shared(matrices, term, np.ones(len(meal), bool), model)
    for ep, fit in fits.items():
        for suffix, values in zip(("background", "projection", "beta"), fit):
            np.testing.assert_allclose(values, weights[f"{ep}_{suffix}"], atol=1e-10, rtol=1e-10)
    snapshot = out / "shared_coefficients_TRE_ONLY.npz"
    save_arrays(snapshot, {"shared": shared})
    freeze = out / "FROZEN.json"
    io.write(freeze, dict(registration=fingerprint, weights=digest(snapshot), additional_outcomes_read=False))
    hold = pq.read_table(path, use_threads=False, pre_buffer=False).to_pandas()
    hold = hold.merge(index[["input_row", "split_unit_id", "age_calendar_years", "sex_code"]],
                      on="input_row", validate="many_to_one")
    require(len(hold) == flow["meals"] and hold.input_row.nunique() == flow["people"]
            and not set(hold.split_unit_id) & set(index.iloc[weights["excluded_rows"]].split_unit_id),
            "Additional households cross the frozen eligibility boundary")
    _, person = np.unique(hold.input_row, return_inverse=True)
    hold["person_index"] = person
    households, family = np.unique(hold.split_unit_id.astype(str), return_inverse=True)
    matrices, term = prepare_design(hold, weights["scores"], model)
    parts = household_losses(matrices, term, fits, shared, np.ones(len(hold), bool), family, len(households))
    with np.load(BytesIO(lfile.read_bytes()), allow_pickle=False) as recorded:
        for ep in ENDPOINTS:
            np.testing.assert_allclose(parts[ep][[0, 1, 3]], recorded[ep], atol=1e-8, rtol=1e-10)
    losses = out / "additional_losses_TRE_ONLY.npz"
    save_arrays(losses, parts)
    checks = (frozen / "COMPLETE.json", prepared / "COMPLETE.json", validated / "COMPLETE.json", wfile, path, lfile)
    sources = {**training["sources"], **prep["sources"], **previous["sources"],
               **{str(p): digest(p) for p in checks}}
    return dict(flow=flow, prediction=summarize(parts, SEED + 2)), sources, (snapshot, freeze, losses)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--package", type=Path, required=True)
    cli.add_argument("--run", required=True)
    cli.add_argument("--commit", required=True)
    cli.add_argument("--smoke", action="store_true")
    args = cli.parse_args()
    package = args.package.resolve()
    require(package / "experiments" / ROOT.name == ROOT and re.fullmatch(r"[0-9a-f]{40}", args.commit)
            and re.fullmatch(r"[a-zA-Z0-9_-]+", args.run), "Unexpected study path or release")
    require(all(digest(p) == sha for p, sha in PINS.items()), "Archived model helper changed")
    out = ROOT / "runs_TRE_ONLY" / args.run
    require(not out.exists(), "Use a fresh run identifier")
    archived_contract()
    archive = MEAL / "runs_TRE_ONLY/models-1"
    (receipt, summary, io, meal, index, pop, family_ids, family, outer, raw, taxa, clr, _) = archived_inputs(package, archive)
    model = __import__("models")
    matrices, dose = bases(meal, model)
    person = meal.person_index.to_numpy()
    households = index.iloc[pop].split_unit_id.astype(str).to_numpy()
    references = [archive / f"r{r}_f{f}.json" for r in (0, 1) for f in range(5)]
    files = [Path(__file__).resolve(), ROOT / "shared_fit.py", archive / "COMPLETE.json", *PINS, *references]
    sources = {**receipt["sources"], **{str(p): digest(p) for p in files}}
    out.mkdir(parents=True, exist_ok=False)
    reg = out / "registration.json"
    io.write(reg, dict(commit=args.commit, sources=sources, smoke=args.smoke, flow=summary["flow"],
                       primary=["net_p0", "shared_vs_separate"], weights=[1/3] * 3, penalty=PENALTY, seed=SEED))
    fingerprint = digest(reg)
    predictions = {}
    for repeat in ((0,) if args.smoke else (0, 1)):
        sums = {ep: np.zeros((4, len(family_ids))) for ep in ENDPOINTS}
        for fold in ((0,) if args.smoke else range(5)):
            train = outer[repeat] != fold
            require(not set(households[train]) & set(households[~train]), "Households cross the archived split")
            score = model.site_scores(raw, taxa, index, pop, train, clr, 5)["both_pc"]
            term, _ = model.center(dose["carbs"][:, None] * score[person], person)
            fits, shared = fit_shared(matrices, term, train[person], model)
            values = household_losses(matrices, term, fits, shared, ~train[person], family, len(family_ids))
            archive_losses(values, archive / f"r{repeat}_f{fold}.json")
            for ep in ENDPOINTS:
                sums[ep] += values[ep]
            path = out / f"r{repeat}_f{fold}.npz"
            save_arrays(path, values)
            io.write(path.with_suffix(".json"), dict(registration=fingerprint, repeat=repeat, fold=fold, sha=digest(path)))
            io.emit("SHARED_OFFSET_FOLD", repeat=repeat, fold=fold)
        predictions[f"r{repeat}"] = summarize(sums, SEED + repeat)
    additional, artifacts, extra_sources = {}, {}, {}
    if not args.smoke:
        additional, extra_sources, files = cohort_transfer(out, meal, index, pop, model, io, fingerprint)
        artifacts = {p.name: digest(p) for p in files}
    check_sources({**sources, **extra_sources})
    result = out / "summary.json"
    io.write(result, dict(flow=summary["flow"], prediction=predictions, additional=additional))
    io.write(out / "COMPLETE.json", dict(registration=fingerprint, summary=digest(result),
                                         artifacts=artifacts, extra_sources=extra_sources))
    io.emit("SHARED_OFFSET_COMPLETE", smoke=args.smoke, folds=1 if args.smoke else 10)


if __name__ == "__main__":
    main()
