"""Test whether the selected discovery interaction transfers between recording periods."""

from paired import ROOT, SITES, archived_inputs, family_totals, summarize
from analyze import bases, markers
from prepare import digest, require

import argparse
from pathlib import Path
import re

import numpy as np


ENDPOINTS = ("net_p0", "peak_p0", "pre_p0")
DIRECTIONS = ("early_to_late", "late_to_early")


def assign_periods(meal):
    dates = meal.groupby("person_index").date
    rank = dates.rank(method="dense")
    count = dates.transform("nunique")
    late = rank > count // 2
    by_period = meal.groupby([meal.person_index, late]).date.transform("size")
    smallest = by_period.groupby(meal.person_index).transform("min")
    keep = count.ge(4) & smallest.ge(4)
    return keep.to_numpy(), late.to_numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--archived", default="models-1")
    parser.add_argument("--run", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    package = args.package.resolve()
    require((package / "experiments" / ROOT.name).resolve() == ROOT
            and re.fullmatch(r"[a-zA-Z0-9_-]+", args.archived)
            and re.fullmatch(r"[a-zA-Z0-9_-]+", args.run)
            and re.fullmatch(r"[0-9a-f]{40}", args.commit), "Unexpected run or source")
    archive = ROOT / "runs_TRE_ONLY" / args.archived
    out = ROOT / "runs_TRE_ONLY" / args.run
    require(not out.exists(), "Use a new run identifier")
    (receipt, _, io, meal, index, pop, _, _, outer,
     raw, taxa, clr, mapping) = archived_inputs(package, archive)
    model = __import__("models")

    keep, late = assign_periods(meal)
    meal = meal.loc[keep].copy()
    late = late[keep]
    eligible, person = np.unique(meal.person_index, return_inverse=True)
    pop = pop[eligible]
    outer = outer[:, eligible]
    household = index.iloc[pop].split_unit_id.to_numpy()
    family_ids, family = np.unique(household[person], return_inverse=True)
    require(len(family_ids) > 20 and late.any() and (~late).any(), "Insufficient date-split support")
    # The held-out half's observed mean defines the same within-person deviation target.
    # No centering quantities are shared across periods.
    period_person = 2 * person + late
    meal["person_index"] = period_person
    matrices, dose = bases(meal, model)
    flow = dict(people=len(pop), households=len(family_ids), meals=len(meal),
                early=int((~late).sum()), late=int(late.sum()))
    files = [Path(__file__).resolve(), ROOT / "paired.py"]
    sources = {**receipt["sources"], **{str(p): digest(p) for p in files}}
    out.mkdir(exist_ok=False)
    registration = dict(commit=args.commit, archived=args.archived, smoke=args.smoke,
                        archived_summary=digest(archive / "summary.json"), sources=sources,
                        flow=flow, replicates=1999, seed=202611123)
    io.write(out / "registration.json", registration)
    fingerprint = digest(out / "registration.json")
    results = {name: {} for name in DIRECTIONS}
    io.emit("TIME_TRANSFER_START", **flow, smoke=args.smoke)
    for repeat in ((0,) if args.smoke else (0, 1)):
        totals = {direction: {ep: {k: np.zeros(len(family_ids)) for k in ("none", *SITES, "tss")}
                              for ep in ENDPOINTS} for direction in DIRECTIONS}
        for fold in ((0,) if args.smoke else range(5)):
            training = outer[repeat] != fold
            require(not set(household[training]) & set(household[~training]), "Household crossing")
            scores = markers(model, raw, taxa, index, pop, training, clr, mapping)
            interactions = {site: model.center(dose["carbs"][:, None] * scores[site][person],
                                               period_person)[0] for site in SITES}
            for train_late, direction in enumerate(DIRECTIONS):
                train = training[person] & (late == train_late)
                valid = ~training[person] & (late != train_late)
                for endpoint in ENDPOINTS:
                    y, x, weights = matrices[endpoint, "carbs"]
                    parts = family_totals(y, x, weights, interactions, train,
                                          family, len(family_ids), model, valid=valid)
                    for key, vector in parts.items():
                        totals[direction][endpoint][key] += vector
            io.write(out / f"r{repeat}_f{fold}.json", dict(registration=fingerprint,
                     repeat=repeat, fold=fold, held=int((~training).sum())))
            io.emit("TIME_TRANSFER_FOLD", repeat=repeat, fold=fold)
        rng = np.random.default_rng(registration["seed"] + repeat)
        boot = rng.integers(0, len(family_ids), (registration["replicates"], len(family_ids)), dtype=np.int32)
        for direction, endpoints in totals.items():
            results[direction][f"r{repeat}"] = {ep: summarize(parts, boot) for ep, parts in endpoints.items()}
    require(all(digest(Path(p)) == value for p, value in sources.items()), "An analysis source changed")
    io.write(out / "summary.json", dict(flow=flow, estimates=results))
    io.write(out / "COMPLETE.json", dict(registration=fingerprint, summary=digest(out / "summary.json")))
    io.emit("TIME_TRANSFER_COMPLETE", smoke=args.smoke, repeats=1 if args.smoke else 2)


if __name__ == "__main__":
    main()
