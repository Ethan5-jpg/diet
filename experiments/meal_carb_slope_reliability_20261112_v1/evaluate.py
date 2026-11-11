"""Assess repeated meal slopes and precision-weighted microbial prediction."""

import argparse
from io import BytesIO
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parent
MEAL = ROOT.parent / "meal_carbohydrate_modification_20261112_v1"
PINS = {
    MEAL / "paired.py": "6495e7bf2fe831a971be31a1dbc1ddaf582773c13a04d89dbf7641a478239b0a",
    MEAL / "time_transfer.py": "c9295f753b4f74e9f6cade505bd35c248e6c7fdbf89a6d16263148411b0b20a8",
}
sys.path.insert(0, str(MEAL))
try:
    from analyze import bases, contrasts, markers
    from paired import archived_inputs
    from prepare import digest, require
    from time_transfer import assign_periods
finally:
    sys.path.pop(0)

import numpy as np

from slopes import family_prediction, family_repeatability, fold_models, read_prediction, read_repeatability, sums


ENDPOINTS = ("net_p0", "peak_p0", "pre_p0")
MARKERS = ("both_pc5", "external_sum7")
DIRECTIONS = ("early_to_late", "late_to_early")
SEED = 202611126
BOOTSTRAPS = 1999
MIN_DOSE_INFORMATION = 4.


def dose_support(dose, person, late, count, model):
    group = 2 * person + late
    centered = model.center(dose[:, None], group)[0][:, 0]
    info = np.bincount(group, weights=centered ** 2, minlength=2 * count).reshape(count, 2)
    return np.all(info >= MIN_DOSE_INFORMATION, axis=1)


def period_stats(matrices, dose, person, late, folds, held_fold, model):
    y, x, weights = matrices
    fold_by_meal = folds[person]
    periods = []
    for side in (0, 1):
        total = np.zeros((len(folds), 6))
        period = late == side
        for fold in range(5):
            use = period & (fold_by_meal != held_fold) & (fold_by_meal != fold)
            target = period & (fold_by_meal == fold)
            require(use.any() and target.any(), "Insufficient household fold support")
            beta = model.fit_background(x[use], y[use], weights[use])[0]
            residual = y[target] - x[target] @ beta
            total += sums(person[target], dose[target], residual, y[target], weights[target], len(folds))
        require((total[:, 0] >= MIN_DOSE_INFORMATION - 1e-9).all(), "Meal-dose information changed")
        periods.append(total)
    return periods


def aggregate(totals, draws):
    repeatability = {ep: read_repeatability(totals[f"{ep}__repeatability"], draws) for ep in ENDPOINTS}
    prediction = {}
    for ep in ENDPOINTS:
        prediction[ep] = {}
        for direction in DIRECTIONS:
            prediction[ep][direction] = {}
            for site in MARKERS:
                prefix = f"{ep}__{direction}__{site}__"
                parts = {k[len(prefix):]: v for k, v in totals.items() if k.startswith(prefix)}
                prediction[ep][direction][site] = read_prediction(parts, draws)
    return dict(repeatability=repeatability, prediction=prediction)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    package = args.package.resolve()
    require(package / "experiments" / ROOT.name == ROOT and re.fullmatch(r"[a-zA-Z0-9_-]+", args.run)
            and re.fullmatch(r"[0-9a-f]{40}", args.commit), "Unexpected release or run path")
    require(all(digest(p) == sha for p, sha in PINS.items()), "Pinned meal source differs")
    out = ROOT / "runs_TRE_ONLY" / args.run
    require(not out.exists(), "Use a fresh run identifier")
    archive = MEAL / "runs_TRE_ONLY/models-1"
    receipt, original, io, meal, index, pop, _, _, outer, raw, taxa, clr, mapping = archived_inputs(package, archive)
    model = __import__("models")
    supported, late = assign_periods(meal)
    all_people = meal.person_index.to_numpy(int)
    dose_ok = dose_support(contrasts(meal)["carbs"], all_people, late, len(pop), model)
    keep = supported & dose_ok[all_people]
    meal, late = meal.loc[keep].copy(), late[keep]
    rows, person = np.unique(meal.person_index, return_inverse=True)
    pop, outer = pop[rows], outer[:, rows]
    households = index.iloc[pop].split_unit_id.astype(str).to_numpy()
    groups, family = np.unique(households, return_inverse=True)
    require(len(groups) >= 100, "Insufficient households with repeated dose variation")
    meal["person_index"] = 2 * person + late
    matrices, doses = bases(meal, model)
    dose = model.center(doses["carbs"][:, None], meal.person_index.to_numpy())[0][:, 0]
    flow = dict(people=len(pop), households=len(groups), meals=len(meal),
                early=int((~late).sum()), late=int(late.sum()))
    sources = {**receipt["sources"], **{str(p): h for p, h in PINS.items()},
               **{str(p): digest(p) for p in (Path(__file__).resolve(), ROOT / "slopes.py", archive / "summary.json")}}
    out.mkdir(parents=True, exist_ok=False)
    reg = dict(commit=args.commit, smoke=args.smoke, flow=flow, archived_flow=original["flow"], sources=sources,
               seed=SEED, bootstraps=BOOTSTRAPS, dose_information=MIN_DOSE_INFORMATION,
               primary=["net_p0", "early_to_late", "both_pc5", "precision_marker"])
    io.write(out / "registration.json", reg)
    fingerprint = digest(out / "registration.json")
    io.emit("SLOPE_RELIABILITY_START", **flow, smoke=args.smoke)
    result = {}
    for repeat in ((0,) if args.smoke else (0, 1)):
        totals = {}
        folds = outer[repeat]
        for fold in ((0,) if args.smoke else range(5)):
            train = folds != fold
            require(not set(households[train]) & set(households[~train]), "Household crossing")
            score = markers(model, raw, taxa, index, pop, train, clr, mapping)
            checkpoint = {}
            for ep in ENDPOINTS:
                data = period_stats(matrices[ep, "carbs"], dose, person, late, folds, fold, model)
                early, later = [side[:, 1] / side[:, 0] for side in data]
                information = np.column_stack([side[:, 0] for side in data])
                checkpoint[f"{ep}__repeatability"] = family_repeatability(family, early, later, ~train, len(groups))
                for h, (source, target) in enumerate(((early, later), (later, early))):
                    for site in MARKERS:
                        pred = fold_models(score[site], source, target, information, train, h)
                        stats = family_prediction(family, data[1 - h], pred, ~train, len(groups))
                        checkpoint.update({f"{ep}__{DIRECTIONS[h]}__{site}__{key}": values
                                           for key, values in stats.items()})
            for key, values in checkpoint.items():
                totals.setdefault(key, np.zeros_like(values))
                totals[key] += values
            buffer = BytesIO()
            np.savez_compressed(buffer, **checkpoint)
            file = out / f"r{repeat}_f{fold}.npz"
            file.write_bytes(buffer.getvalue())
            io.write(file.with_suffix(".json"), dict(registration=fingerprint, repeat=repeat, fold=fold,
                     held=int((~train).sum()), losses=digest(file)))
            io.emit("SLOPE_RELIABILITY_FOLD", repeat=repeat, fold=fold)
        boot = np.random.default_rng(SEED + repeat).integers(0, len(groups), (BOOTSTRAPS, len(groups)), dtype=np.int32)
        result[f"r{repeat}"] = aggregate(totals, boot)
    require(all(digest(Path(p)) == sha for p, sha in sources.items()), "Analysis source changed")
    io.write(out / "summary.json", dict(flow=flow, estimates=result))
    io.write(out / "COMPLETE.json", dict(registration=fingerprint, summary=digest(out / "summary.json")))
    io.emit("SLOPE_RELIABILITY_COMPLETE", smoke=args.smoke, repeats=1 if args.smoke else 2)


if __name__ == "__main__":
    main()
