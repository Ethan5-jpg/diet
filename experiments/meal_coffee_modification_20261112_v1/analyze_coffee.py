"""Test whether a named gut taxon modifies the carbohydrate slope in coffee meals."""

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
from scipy.stats import t

from coffee_design import KEYS, SPECIES, STRATA, interaction_design, nearby_people, reconstruct, supported, taxon_fraction

ROOT = Path(__file__).resolve().parent
MEAL = ROOT.parent / "meal_carbohydrate_modification_20261112_v1"
PROCESSING = ROOT.parent / "meal_processing_modification_20261112_v1"
BREAD = ROOT.parent / "meal_wholewheat_modification_20261112_v1"
TIMING = ROOT.parent / "meal_sample_timing_modification_20261112_v1"
sys.path[:0] = [str(MEAL), str(PROCESSING), str(BREAD)]
try:
    from analyze import bases
    from audit import projected_meals
    from design import zero_aware
    from paired import archived_inputs, family_totals
    from prepare import NUTRIENTS, digest, food_groups, load, require, scoped_read
    from time_transfer import assign_periods
finally:
    del sys.path[:3]

PINS = {
    PROCESSING / "audit.py": "d093b86be0d041fc2eb1530329216518eb76ddae54615fcdcc2c3d0ca3463295",
    PROCESSING / "processing.py": "f249264bcdec0ed84b299cd3702ff5ed1e61937a6baf33ea8ed883fce697c567",
    BREAD / "design.py": "9d43980c48467ef7e3ced5e1616e4901f9f597a2ebfb3c5d438ce3a84af13e2f",
}
ENDPOINTS = ("net_p0", "peak_p0", "net_m15", "net_p15", "pre_p0")
MARKERS = ("detection", "positive_dose")
PERIODS = ("pooled", "early_to_late", "late_to_early")
SEED = 202611129


def exposures(package):
    require(all(digest(p) == sha for p, sha in PINS.items()), "Pinned exposure dependencies differ")
    blind = TIMING / "runs_TRE_ONLY/COFFEE_LABEL_SUPPORT_TRE_ONLY.json"
    require(digest(blind) == "d8255f9ff42c624400e6f1504b3aa7f38000b231c162da6428dca1c58edd5ec7",
            "Outcome-blind feasibility record differs")
    evidence = load(blind)
    require(not evidence["outcome_columns_read"], "Expected outcome-blind feasibility")
    meals, index, cfg, inputs, io, prep, sources = projected_meals(package)
    food_path = package.parent / "analysis/shared/food_events_clean_TRE_ONLY.parquet"
    food_sha = digest(food_path)
    require(food_sha == load(prep)["sources"][str(food_path)] == evidence["source_sha256"],
            "Archived and audited food source differ")
    columns = ["collection_timestamp", "local_timestamp", "short_food_name", "qc_primary_eligible",
               "qc_same_food_time_n", "qc_exact_signature_n", *NUTRIENTS]
    food = scoped_read(food_path, columns, index.loc[index.input_row.isin(meals.input_row)], inputs)
    food = pd.concat(food_groups(food).values(), ignore_index=True)
    meals = reconstruct(meals, food, NUTRIENTS, cfg["max_food_episode_span_minutes"] * 60)
    rows = {s: supported(meals, s) for s in STRATA}
    primary = rows["documented"]
    require(dict(people=int(primary.input_row.nunique()), households=int(primary.split_unit_id.nunique()),
                 meals=len(primary), coffee_meals=int(primary.context.sum())) == evidence["support"]["certain"],
            "Documented coffee support differs from the frozen audit")
    files = [Path(__file__).resolve(), ROOT / "coffee_design.py", blind, *PINS]
    return rows, io, {**sources, str(food_path): food_sha, **{str(p): digest(p) for p in files}}


def population(meal, model):
    frozen, person = np.unique(meal.person_index, return_inverse=True)
    meal["person_index"] = person
    dated, late = assign_periods(meal)
    split = meal.loc[dated].copy()
    split["late"] = late[dated]
    counts = split.groupby(["person_index", "late", "context"]).size()
    stable = counts[counts.ge(2)].groupby(level="person_index").size()
    split = split.loc[split.person_index.isin(stable.index[stable.eq(4)])].copy()
    split_person = split.person_index.to_numpy()
    late = split.late.to_numpy()
    split["person_index"] = 2 * split_person + late
    return frozen, {
        "pooled": (meal, person, None, bases(meal, model)[0]),
        "calendar": (split, split_person, late, bases(split, model)[0]),
    }


def model_matrices(dataset, score, model):
    meal, person, _, matrices = dataset
    covariates, interaction = interaction_design(meal, score, person, model)
    return {e: (y, np.column_stack([x, covariates]), w) for (e, d), (y, x, w) in matrices.items()
            if d == "carbs"}, interaction


def add_other_marker(x, interaction, marker):
    column = MARKERS.index(marker)
    return np.column_stack([x, interaction[:, 1 - column]]), interaction[:, [column]]


def evaluate(dataset, period, score, family, train, model):
    _, person, late, _ = dataset
    take, valid = train[person], ~train[person]
    if late is not None:
        later = period == "late_to_early"
        take, valid = take & (late == later), valid & (late != later)
    matrices, interaction = model_matrices(dataset, score, model)
    result = {}
    for ep, (y, x, w) in matrices.items():
        for marker in MARKERS:
            baseline, term = add_other_marker(x, interaction, marker)
            parts = family_totals(y, baseline, w, {"coffee": term}, take, family[person],
                                  family.max() + 1, model, valid=valid)
            result.update({f"{period}__{ep}__{marker}__{k}": v for k, v in parts.items()})
    return result


def predictions(total, repeat):
    result = {}
    for s in STRATA:
        for period in PERIODS:
            data = {f"{s}__{period}__{ep}__{m}": {k: total[f"{s}__{period}__{ep}__{m}__{k}"]
                    for k in ("none", "coffee", "tss")} for ep in ENDPOINTS for m in MARKERS}
            support = np.flatnonzero(np.logical_or.reduce([v["tss"] > 0 for v in data.values()]))
            require(len(support) > 0, "No held-out households")
            draws = np.random.default_rng(SEED + repeat).choice(support, (1999, len(support)), replace=True)
            for key, parts in data.items():
                tss = parts["tss"]
                denom = tss[draws].sum(axis=1)
                require(tss.sum() > 0 and (denom > 0).all(), "Invalid paired denominator")
                gain = parts["none"] - parts["coffee"]
                result[key] = dict(delta_r2=float(gain.sum() / tss.sum()),
                                   baseline_r2=float(1 - parts["none"].sum() / tss.sum()),
                                   ci95=np.quantile(gain[draws].sum(axis=1) / denom, [.025, .975]).tolist())
    return result


def associations(dataset, fraction, households, model):
    _, person, _, _ = dataset
    score, detection_gap = zero_aware(fraction, np.ones(len(fraction), bool))
    dose_gap = float(np.diff(np.quantile(score[fraction > 0, 1], [.25, .75]))[0])
    matrices, interaction = model_matrices(dataset, score, model)
    rows = {}
    for ep, (y, x, w) in matrices.items():
        for marker, scale in zip(MARKERS, (detection_gap, dose_gap)):
            baseline, term = add_other_marker(x, interaction, marker)
            fitted = model.fit_background(baseline, y, w)
            beta, cov, p = model.household_test(term, fitted, households[person], len(fraction))
            estimate = float(beta[0] * scale)
            half = float(np.sqrt(cov[0, 0]) * scale * t.ppf(.975, len(np.unique(households)) - 1))
            rows[f"{ep}__{marker}"] = dict(p=p, effect=estimate, ci95=[estimate - half, estimate + half])
    return rows


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--package", type=Path, required=True)
    cli.add_argument("--run", required=True)
    cli.add_argument("--commit", required=True)
    cli.add_argument("--smoke", action="store_true")
    args = cli.parse_args()
    package = args.package.resolve()
    require(package / "experiments" / ROOT.name == ROOT and re.fullmatch(r"[a-zA-Z0-9_-]+", args.run)
            and re.fullmatch(r"[0-9a-f]{40}", args.commit), "Unexpected release or run path")
    out = ROOT / "runs_TRE_ONLY" / args.run
    require(not out.exists(), "Use a fresh run identifier")
    candidate, io, sources = exposures(package)
    archive = MEAL / "runs_TRE_ONLY/models-1"
    (receipt, _, _, meal, index, pop, _, _, outer, raw, taxa, _, _) = archived_inputs(package, archive)
    fraction = taxon_fraction(raw["gut"][pop], taxa["gut"])
    model = __import__("models")
    datasets, flows = {}, {}
    for s, rows in candidate.items():
        fields = [*KEYS, "context", "coffee_kcal", "coffee_carbs", "prior_coffee_share", "prior_log_days"]
        joined = meal.merge(rows[fields], on=KEYS, validate="one_to_one")
        require(len(joined) == len(rows), "A selected coffee meal is absent from the model archive")
        frozen, data = population(joined, model)
        households = index.iloc[pop[frozen]].split_unit_id.astype(str).to_numpy()
        _, family = np.unique(households, return_inverse=True)
        datasets[s] = data, households, family, outer[:, frozen], fraction[frozen]
        flows[s] = {name: dict(people=len(np.unique(p)), households=len(np.unique(family[p])), meals=len(m),
                               coffee_meals=int(m.context.sum())) for name, (m, p, _, _) in data.items()}
        flows[s]["nearby_pair_people"] = nearby_people(data["pooled"][0])
        require(flows[s]["calendar"]["households"] >= 10, "Insufficient balanced calendar support")
    io.emit("COFFEE_START", smoke=args.smoke, **flows)
    sources = {**receipt["sources"], **sources}
    out.mkdir(parents=True, exist_ok=False)
    reg = out / "registration.json"
    io.write(reg, dict(commit=args.commit, smoke=args.smoke, flow=flows, sources=sources, seed=SEED, bootstraps=1999,
                       taxon=SPECIES, primary="documented__pooled__net_p0__positive_dose"))
    fingerprint = digest(reg)
    prediction = {}
    for repeat in ((0,) if args.smoke else (0, 1)):
        total = {}
        for fold in ((0,) if args.smoke else range(5)):
            part = {}
            for s, (data, households, family, splits, abund) in datasets.items():
                train = splits[repeat] != fold
                require(not set(households[train]) & set(households[~train]), "Household crossing")
                score, _ = zero_aware(abund, train)
                for period in PERIODS:
                    local = data["pooled"] if period == "pooled" else data["calendar"]
                    part.update({f"{s}__{k}": v for k, v in evaluate(local, period, score, family, train, model).items()})
            for key, value in part.items():
                if key not in total:
                    total[key] = np.zeros_like(value)
                total[key] += value
            blob = BytesIO()
            np.savez_compressed(blob, **part)
            saved = out / f"r{repeat}_f{fold}.npz"
            saved.write_bytes(blob.getvalue())
            io.write(saved.with_suffix(".json"), dict(registration=fingerprint, repeat=repeat, fold=fold, sha=digest(saved)))
            io.emit("COFFEE_FOLD", repeat=repeat, fold=fold)
        prediction[f"r{repeat}"] = predictions(total, repeat)
    direct = {}
    if not args.smoke:
        for s, (data, households, _, _, abund) in datasets.items():
            direct.update({f"{s}__{k}": v for k, v in associations(data["pooled"], abund, households, model).items()})
        for row, adjusted in zip(direct.values(), model.holm([row["p"] for row in direct.values()])):
            row["holm_p"] = float(adjusted)
    require(all(digest(Path(p)) == sha for p, sha in sources.items()), "A registered source changed")
    summary = out / "summary.json"
    io.write(summary, dict(flow=flows, prediction=prediction, direct=direct))
    io.write(out / "COMPLETE.json", dict(registration=fingerprint, summary=digest(summary)))
    io.emit("COFFEE_COMPLETE", smoke=args.smoke, tests=len(direct))


if __name__ == "__main__":
    main()
