"""Evaluate held-out community predictions of recurrent carbohydrate slopes."""

import argparse
from io import BytesIO
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
MEAL = ROOT.parent / "meal_carbohydrate_modification_20261112_v1"
SLOPE = ROOT.parent / "meal_carb_slope_reliability_20261112_v1"

sys.path[:0] = [str(MEAL), str(SLOPE)]
try:
    from analyze import bases, contrasts
    from evaluate import dose_support, period_stats
    from paired import archived_inputs
    from prepare import digest, load, require
    from time_transfer import assign_periods
finally:
    del sys.path[:2]

import numpy as np

from community import PENALTIES, clr_kernels, covariance_moments, household_totals, predict, summarize


ENDPOINTS = ("net_p0", "peak_p0", "pre_p0")
DIRECTIONS = ("early_to_late", "late_to_early")
SEED = 2026111208
BOOTSTRAPS = 1999
PINS = {
    MEAL / "analyze.py": "319cf53ec1a4414153ef2b929c74c071d09e4365e95bc8fb0f712d7c6aaf2ae6",
    MEAL / "paired.py": "6495e7bf2fe831a971be31a1dbc1ddaf582773c13a04d89dbf7641a478239b0a",
    MEAL / "time_transfer.py": "c9295f753b4f74e9f6cade505bd35c248e6c7fdbf89a6d16263148411b0b20a8",
    SLOPE / "evaluate.py": "f896e787335da2736a3ef4422d4195b45b409833bb3c9373e7f7e5af5b37a14c",
    SLOPE / "slopes.py": "4cae8a06fefcc7fada7c1923970ee9b98aadb31bbe715151473b6bf9cd6c7570",
}


def geometries(raw, taxa, index, pop, training, clr, model):
    profile = {}
    ids = index.iloc[pop[training]].participant_id.tolist()
    for site in ("oral", "gut"):
        abundance = raw[site][pop]
        transform = clr().fit(abundance[training], taxa[site], ids, site=site)
        result = transform.transform(abundance, taxa[site], site=site)
        require(result["supported"].all(), f"Unsupported CLR profile in {site}")
        profile[site] = result["values"]
    pcs, kernels = [], []
    for values in profile.values():
        centered = values - values[training].mean(axis=0)
        _, singular, axes = np.linalg.svd(centered[training], full_matrices=False)
        require(len(singular) >= 5 and singular[4] > 1e-8, "Insufficient training CLR rank")
        pcs.append(model.train_standardize(centered @ axes[:5].T, training))
        kernels.extend(clr_kernels(values, training))
    scores = np.column_stack(pcs)
    return dict(pc=[(scores @ scores[training].T, (scores ** 2).sum(axis=1))], community=kernels)


def checkpoints(data, estimates, held, household, count):
    family = household[held]
    output = {}
    for col, ep in enumerate(ENDPOINTS):
        periods = [x[held, col, :] for x in data]
        observed = [x[:, 1] / x[:, 0] for x in periods]
        moments = [covariance_moments(*observed, family, count)]
        for name in ("constant", "pc", "community"):
            residual = [observed[side] - estimates[side][name][held, col] for side in (0, 1)]
            moments.append(covariance_moments(*residual, family, count))
        output[f"{ep}__covariance"] = np.column_stack(moments)
        for source, direction in enumerate(DIRECTIONS):
            target = 1 - source
            d = periods[target]
            slope = observed[target]
            prediction = [estimates[source][name][held, col] for name in ("constant", *PENALTIES)]
            output[f"{ep}__{direction}__slope"] = household_totals(
                [slope ** 2, *((slope - value) ** 2 for value in prediction)], family, count)
            output[f"{ep}__{direction}__meal"] = household_totals(
                [d[:, 5], *(d[:, 4] - 2 * value * d[:, 3] + value ** 2 * d[:, 2] for value in prediction)],
                family, count)
    return output


def combined(out, registration):
    fingerprint = digest(out / "registration.json")
    smoke = registration["smoke"]
    count = registration["flow"]["households"]
    result = {}
    for repeat in ((0,) if smoke else range(2)):
        totals = {}
        for fold in ((0,) if smoke else range(5)):
            path = out / f"r{repeat}_f{fold}.npz"
            receipt = load(path.with_suffix(".json"))
            require(receipt == dict(registration=fingerprint, repeat=repeat, fold=fold, sha256=digest(path)),
                    "Fold receipt differs")
            with np.load(BytesIO(path.read_bytes()), allow_pickle=False) as block:
                for key in block.files:
                    values = block[key]
                    totals.setdefault(key, np.zeros_like(values))
                    totals[key] += values
        generator = np.random.default_rng(SEED + repeat)
        draws = np.empty((BOOTSTRAPS + 1, count), dtype=float)
        draws[0] = 1
        sampled = generator.integers(0, count, (BOOTSTRAPS, count), dtype=np.int32)
        for dest, sample in zip(draws[1:], sampled):
            dest[:] = np.bincount(sample, minlength=count)
        result[f"r{repeat}"] = summarize(totals, draws)
    return dict(flow=registration["flow"], estimates=result)


def verify(out):
    registration = load(out / "registration.json")
    n = 1 if registration["smoke"] else 10
    require(len(list(out.glob("r*_f*.json"))) == len(list(out.glob("r*_f*.npz"))) == n,
            "Unexpected fold count")
    require(load(out / "COMPLETE.json") == dict(registration=digest(out / "registration.json"),
                                               summary=digest(out / "summary.json")), "Completion receipt differs")
    require(all(digest(Path(p)) == value for p, value in registration["sources"].items()),
            "Registered input or source differs")
    require(combined(out, registration) == load(out / "summary.json"), "Household reaggregation differs")
    print(f"COMMUNITY_SLOPE_READBACK_PASS folds={n}", flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--package", type=Path, required=True)
    cli.add_argument("--commit", required=True)
    cli.add_argument("--run", required=True)
    cli.add_argument("--smoke", action="store_true")
    cli.add_argument("--verify", action="store_true")
    args = cli.parse_args()
    package = args.package.resolve()
    require(package / "experiments" / ROOT.name == ROOT and re.fullmatch(r"[a-zA-Z0-9_-]+", args.run)
            and re.fullmatch(r"[0-9a-f]{40}", args.commit), "Unexpected study, run or source release")
    out = ROOT / "runs_TRE_ONLY" / args.run
    if args.verify:
        require(load(out / "registration.json")["commit"] == args.commit, "Requested source release differs")
        verify(out)
        return
    require(not out.exists(), "Use a fresh run identifier")
    require(all(digest(path) == sha for path, sha in PINS.items()), "Archived analysis sources differ")
    archive = MEAL / "runs_TRE_ONLY/models-1"
    receipt, _, io, meal, index, pop, _, _, outer, raw, taxa, clr, _ = archived_inputs(package, archive)
    model = __import__("models")
    eligible, late = assign_periods(meal)
    people = meal.person_index.to_numpy(int)
    dose_ok = dose_support(contrasts(meal)["carbs"], people, late, len(pop), model)
    keep = eligible & dose_ok[people]
    meal, late = meal.loc[keep].copy(), late[keep]
    rows, person = np.unique(meal.person_index, return_inverse=True)
    pop, outer = pop[rows], outer[:, rows]
    households, family = np.unique(index.iloc[pop].split_unit_id.astype(str), return_inverse=True)
    count = len(households)
    require(count >= 100, "Insufficient households with repeated carbohydrate support")
    meal["person_index"] = 2 * person + late
    matrices, doses = bases(meal, model)
    dose = model.center(doses["carbs"][:, None], meal.person_index.to_numpy())[0][:, 0]
    flow = dict(people=len(pop), households=count, meals=len(meal), early=int((~late).sum()), late=int(late.sum()))
    sources = {**receipt["sources"], **{str(p): sha for p, sha in PINS.items()},
               **{str(p): digest(p) for p in (*ROOT.glob("*.py"), archive / "summary.json", archive / "COMPLETE.json")}}
    registration = dict(commit=args.commit, smoke=args.smoke, flow=flow, sources=sources, seed=SEED,
                        bootstraps=BOOTSTRAPS, penalties=PENALTIES, primary="net_p0 early_to_late meal community_vs_pc")
    out.mkdir(parents=True, exist_ok=False)
    io.write(out / "registration.json", registration)
    fingerprint = digest(out / "registration.json")
    io.emit("COMMUNITY_SLOPE_START", **flow, smoke=args.smoke)
    for repeat in ((0,) if args.smoke else range(2)):
        folds = outer[repeat]
        for fold in ((0,) if args.smoke else range(5)):
            train = folds != fold
            require(not set(family[train]) & set(family[~train]), "Household crossing")
            kernels = geometries(raw, taxa, index, pop, train, clr, model)
            data = [period_stats(matrices[ep, "carbs"], dose, person, late, folds, fold, model) for ep in ENDPOINTS]
            data = [np.stack(period, axis=1) for period in zip(*data)]
            estimates = [predict(kernels, period[:, 0, 2], period[:, :, 1] / period[:, :, 0], train) for period in data]
            del kernels
            fold_values = checkpoints(data, estimates, ~train, family, count)
            path = out / f"r{repeat}_f{fold}.npz"
            buffer = BytesIO()
            np.savez_compressed(buffer, **fold_values)
            path.write_bytes(buffer.getvalue())
            io.write(path.with_suffix(".json"), dict(registration=fingerprint, repeat=repeat, fold=fold,
                                                    sha256=digest(path)))
            io.emit("COMMUNITY_SLOPE_FOLD", repeat=repeat, fold=fold)
    io.write(out / "summary.json", combined(out, registration))
    io.write(out / "COMPLETE.json", dict(registration=fingerprint, summary=digest(out / "summary.json")))
    io.emit("COMMUNITY_SLOPE_COMPLETE", smoke=args.smoke, **flow)


if __name__ == "__main__":
    main()
