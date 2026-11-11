"""Test an externally motivated bread-source-by-Lachnoclostridium contrast."""

import argparse
from io import BytesIO
import os
from pathlib import Path
import re
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from scipy.stats import t

from design import BREAD, comparable_pairs, covariates, genus_fraction, zero_aware

ROOT = Path(__file__).resolve().parent
MEAL = ROOT.parent / "meal_carbohydrate_modification_20261112_v1"
SOURCE = ROOT.parent / "meal_carb_source_modification_20261112_v1"
PROCESSING = ROOT.parent / "meal_processing_modification_20261112_v1"
sys.path[:0] = [str(MEAL), str(SOURCE), str(PROCESSING)]
try:
    from analyze import bases
    from analyze_sources import collect_sources, summarize
    from paired import archived_inputs, family_totals
    from prepare import digest, load, require
    from processing import repeated_support
    from time_transfer import assign_periods
finally:
    del sys.path[:3]

PINS = {
    MEAL / "paired.py": "6495e7bf2fe831a971be31a1dbc1ddaf582773c13a04d89dbf7641a478239b0a",
    MEAL / "time_transfer.py": "c9295f753b4f74e9f6cade505bd35c248e6c7fdbf89a6d16263148411b0b20a8",
    SOURCE / "exposure.py": "ff192f2169aedce795eb706905e18d012e639ba3b7a9786c6556849bd7ee3884",
    SOURCE / "analyze_sources.py": "594be0c49a953e27884f41e8c58643a05a8c640d0d2fb3f12d2416ad699e5683",
    PROCESSING / "processing.py": "f249264bcdec0ed84b299cd3702ff5ed1e61937a6baf33ea8ed883fce697c567",
}
ENDPOINTS = ("net_p0", "peak_p0", "net_m15", "net_p15", "pre_p0")
PERIODS = ("pooled", "early_to_late", "late_to_early", "comparable")
MARKERS = ("detection", "positive_dose")
SEED = 202611127


def cohort(meal, receipt, model):
    audit = receipt["provider_wholewheat_vs_bread"]
    meal = meal.loc[repeated_support(meal, BREAD)].copy()
    people, person = np.unique(meal.person_index, return_inverse=True)
    meal["person_index"] = person
    require(audit["repeated"]["people"] == len(people) and audit["repeated"]["meals"] == len(meal),
            "Archived provider-category support differs")
    comparable = comparable_pairs(meal)
    require(audit["nearby_pair_people"] * 2 == len(comparable), "Comparable bread pair support differs")
    date_ok, late = assign_periods(meal)
    dated = meal.loc[date_ok].copy()
    dated["period"] = 2 * dated.input_row + late[date_ok]
    per_period = dated.loc[repeated_support(dated, BREAD, days=1, unit="period")]
    present = per_period.groupby("input_row").period.nunique().eq(2)
    balanced = dated.input_row.isin(present.index[present])
    dated = dated.loc[balanced].copy()
    late = late[date_ok][balanced.to_numpy()]
    dated_people = dated.person_index.to_numpy()
    require(audit["both_sources_each_period"]["people"] == len(np.unique(dated_people))
            and audit["both_sources_each_period"]["meals"] == len(dated), "Calendar support differs")
    dated["person_index"] = 2 * dated_people + late
    return people, dict(
        pooled=(meal, person, None, bases(meal, model)[0]),
        chronological=(dated, dated_people, late, bases(dated, model)[0]),
        comparable=(comparable, comparable.person_index.to_numpy(), None, bases(comparable, model)[0]),
    )


def active_markers(period):
    return MARKERS[:1] if period == "comparable" else MARKERS


def period_data(period, datasets):
    return datasets["chronological"] if period in ("early_to_late", "late_to_early") else datasets[period]


def matrices(dataset, score, model):
    meal, person, _, raw_matrices = dataset
    z, exchange = covariates(meal, score, person, meal.person_index.to_numpy(), model)
    return {ep: (y, np.column_stack([x, z]), w) for (ep, exposure), (y, x, w)
            in raw_matrices.items() if exposure == "carbs"}, exchange


def marker_design(name, x, exchange):
    if name == "detection":
        return x, exchange[:, :1]
    return np.column_stack([x, exchange[:, 0]]), exchange[:, 1:]


def evaluate(period, dataset, score, training, family, n_family, model):
    _, person, late, _ = dataset
    take, valid = training[person], ~training[person]
    if late is not None:
        later = period == "late_to_early"
        take, valid = take & (late == later), valid & (late != later)
    m, exchange = matrices(dataset, score, model)
    losses = {}
    for endpoint, (y, x, w) in m.items():
        for marker in active_markers(period):
            baseline, interaction = marker_design(marker, x, exchange)
            parts = family_totals(y, baseline, w, {"exchange": interaction}, take, family[person], n_family,
                                  model, valid=valid)
            losses.update({f"{period}__{endpoint}__{marker}__{key}": value for key, value in parts.items()})
    return losses


def paired_summary(flat, repeat):
    prediction = {}
    for period in PERIODS:
        pairs = {f"{period}__{ep}__{name}": {part: flat[f"{period}__{ep}__{name}__{part}"]
                 for part in ("none", "exchange", "tss")} for ep in ENDPOINTS for name in active_markers(period)}
        evaluated = np.flatnonzero(np.logical_or.reduce([row["tss"] > 0 for row in pairs.values()]))
        require(len(evaluated) > 0, "No households evaluated")
        samples = np.random.default_rng(SEED + repeat).choice(evaluated, (1999, len(evaluated)), replace=True)
        prediction.update({name: summarize(parts, samples) for name, parts in pairs.items()})
    return prediction


def associations(datasets, score, separation, households, model):
    estimates = {}
    for period in ("pooled", "comparable"):
        dataset = datasets[period]
        person = dataset[1]
        m, exchange = matrices(dataset, score, model)
        group = households[person]
        dof = len(np.unique(group)) - 1
        for endpoint, (y, x, w) in m.items():
            for name in active_markers(period):
                base, interaction = marker_design(name, x, exchange)
                fitted = model.fit_background(base, y, w)
                beta, cov, p = model.household_test(interaction, fitted, group, len(np.unique(person)))
                units = separation if name == "detection" else 1.
                effect = float(beta[0] * units)
                half = float(np.sqrt(cov[0, 0]) * units * t.ppf(.975, dof))
                estimates[f"{period}__{endpoint}__{name}"] = dict(p=p, effect=effect, ci95=[effect-half, effect+half])
    for row, adjusted in zip(estimates.values(), model.holm([row["p"] for row in estimates.values()])):
        row["holm_p"] = float(adjusted)
    return estimates


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--package", required=True, type=Path)
    cli.add_argument("--run", required=True)
    cli.add_argument("--commit", required=True)
    cli.add_argument("--smoke", action="store_true")
    args = cli.parse_args()
    package = args.package.resolve()
    require(package / "experiments" / ROOT.name == ROOT and re.fullmatch(r"[a-zA-Z0-9_-]+", args.run)
            and re.fullmatch(r"[0-9a-f]{40}", args.commit), "Unexpected release or run path")
    require(all(digest(p) == sha for p, sha in PINS.items()), "Pinned meal dependencies differ")
    out = ROOT / "runs_TRE_ONLY" / args.run
    require(not out.exists(), "Use a fresh run identifier")
    audit = PROCESSING / "runs_TRE_ONLY/audit-discovery-1"
    summary_path, registration = audit / "summary.json", audit / "registration.json"
    completed = load(audit / "COMPLETE.json")
    registered = load(registration)
    require(digest(summary_path) == completed["summary"] and digest(registration) == completed["registration"]
            and not registered["smoke"] and not registered["outcome_columns_read"], "Unverified bread support")
    archive = MEAL / "runs_TRE_ONLY/models-1"
    (receipt, _, io, meal, index, pop, _, _, outer, raw, taxa, _, _) = archived_inputs(package, archive)
    food, food_sources = collect_sources(package, archive, receipt, meal, index)
    require(all(registered["sources"][p] == v for p, v in food_sources.items()),
            "Sources differ from outcome-blind bread audit")
    model = __import__("models")
    frozen, data = cohort(food, load(summary_path), model)
    people = pop[frozen]
    outer = outer[:, frozen]
    fraction, n_leaves = genus_fraction(raw["gut"][people], taxa["gut"])
    households = index.iloc[people].split_unit_id.astype(str).to_numpy()
    family_ids, family = np.unique(households, return_inverse=True)
    flow = {name: dict(people=len(np.unique(p)), households=len(np.unique(family[p])), meals=len(m),
                       detected=int((fraction[np.unique(p)] > 0).sum())) for name, (m, p, _, _) in data.items()}
    here = [Path(__file__).resolve(), ROOT / "design.py", registration, summary_path, *PINS]
    sources = {**receipt["sources"], **food_sources, **{str(p): digest(p) for p in here}}
    out.mkdir(parents=True, exist_ok=False)
    reg_path = out / "registration.json"
    io.write(reg_path, dict(commit=args.commit, smoke=args.smoke, flow=flow, species_leaves=n_leaves,
                           primary="pooled__net_p0__detection", seed=SEED, bootstraps=1999, sources=sources))
    fingerprint = digest(reg_path)
    prediction = {}
    io.emit("WHOLEWHEAT_START", smoke=args.smoke, **flow)
    for repeat in ((0,) if args.smoke else (0, 1)):
        totals = {}
        for fold in ((0,) if args.smoke else range(5)):
            training = outer[repeat] != fold
            require(not set(households[training]) & set(households[~training]), "Household crossing")
            score, _ = zero_aware(fraction, training)
            parts = {}
            for period in PERIODS:
                parts.update(evaluate(period, period_data(period, data), score, training, family, len(family_ids), model))
            for key, v in parts.items():
                if key not in totals:
                    totals[key] = np.zeros_like(v)
                totals[key] += v
            buffer = BytesIO()
            np.savez_compressed(buffer, **parts)
            saved = out / f"r{repeat}_f{fold}.npz"
            saved.write_bytes(buffer.getvalue())
            io.write(saved.with_suffix(".json"), dict(registration=fingerprint, repeat=repeat, fold=fold, sha=digest(saved)))
            io.emit("WHOLEWHEAT_FOLD", repeat=repeat, fold=fold)
        prediction[f"r{repeat}"] = paired_summary(totals, repeat)
    direct = {}
    if not args.smoke:
        score, separation = zero_aware(fraction, np.ones(len(fraction), bool))
        direct = associations(data, score, separation, households, model)
    require(all(digest(Path(p)) == sha for p, sha in sources.items()), "An analysis input changed")
    summary = out / "summary.json"
    io.write(summary, dict(flow=flow, prediction=prediction, direct=direct))
    io.write(out / "COMPLETE.json", dict(registration=fingerprint, summary=digest(summary)))
    io.emit("WHOLEWHEAT_COMPLETE", smoke=args.smoke, tests=len(direct))


if __name__ == "__main__":
    main()
