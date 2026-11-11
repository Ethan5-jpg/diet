"""Test a fruit-for-bread carbohydrate contrast within discovery meal records."""

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
    from analyze import bases, markers
    from paired import archived_inputs, family_totals
    from prepare import NUTRIENTS, base_modules, digest, food_groups, load, require, scoped_read
    from time_transfer import assign_periods
finally:
    sys.path.pop(0)

import numpy as np
import pandas as pd
from scipy.stats import t

from exposure import nearby_pair_support, reconstruct, supported


ENDPOINTS = ("net_p0", "peak_p0", "pre_p0")
MARKERS = ("both_pc5", "oral_pc5", "gut_pc5", "external_sum7")
PERIODS = ("pooled", "early_to_late", "late_to_early")
SEED = 202611125


def exposures(meal):
    exchange = (meal.fruits.to_numpy(float) - meal.bread.to_numpy(float)) / 20
    pair = (meal.fruits.to_numpy(float) + meal.bread.to_numpy(float)) / 10
    total = (meal.carbs.to_numpy(float) - 40) / 10
    age = (meal.age_calendar_years.to_numpy(float) - 50) / 10
    sex = meal.sex_code.to_numpy(float)
    extra = np.column_stack((pair, exchange, pair ** 2, exchange ** 2, pair * exchange,
                            pair * total, exchange * total, exchange * age, exchange * sex,
                            meal.bread_wholewheat.to_numpy(float) / 10, meal.unlabelled.to_numpy(float) / 10))
    return exchange, pair, total, extra


def design(score, person, center_id, exchange, pair, total, model):
    values = score[person]
    all_terms = np.column_stack((values * total[:, None], values * pair[:, None],
                                 values * exchange[:, None]))
    centered, _ = model.center(all_terms, center_id)
    return centered[:, :-values.shape[1]], centered[:, -values.shape[1]:]


def summarize(parts, draws):
    gain = parts["none"] - parts["exchange"]
    tss = parts["tss"]
    den = tss[draws].sum(axis=1)
    require(tss.sum() > 0 and (den > 0).all(), "Invalid paired denominator")
    return dict(delta_r2=float(gain.sum() / tss.sum()), null_r2=float(1 - parts["none"].sum() / tss.sum()),
                ci95=np.quantile(gain[draws].sum(axis=1) / den, [.025, .975]).tolist())


def collect_sources(package, archive, receipt, meal, index):
    cfg = load(MEAL / "config.json")
    inputs, _ = base_modules(cfg)
    prepared = MEAL / "runs_TRE_ONLY" / receipt["prepared"]
    reg = prepared / "registration.json"
    require(digest(reg) == load(prepared / "COMPLETE.json")["registration"],
            "Original meal preparation registration changed")
    path = package.parent / "analysis/shared/food_events_clean_TRE_ONLY.parquet"
    sha = digest(path)
    require(load(reg)["sources"][str(path)] == sha, "Food event source differs from archived meal preparation")
    fields = ["collection_timestamp", "local_timestamp", "food_category", "qc_primary_eligible",
              "qc_same_food_time_n", "qc_exact_signature_n", *NUTRIENTS]
    food = scoped_read(path, fields, index.loc[index.input_row.isin(meal.input_row)], inputs)
    food = pd.concat(food_groups(food).values(), ignore_index=True)
    require(digest(path) == sha, "Food events changed during read")
    linked = reconstruct(meal, food, NUTRIENTS, cfg["max_food_episode_span_minutes"] * 60)
    return linked, {str(path): sha, str(reg): digest(reg), str(archive / "summary.json"): digest(archive / "summary.json")}


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
    (receipt, original, io, meal, index, pop, _, _, outer, raw, taxa, clr, mapping) = archived_inputs(package, archive)
    model = __import__("models")
    meal, input_sources = collect_sources(package, archive, receipt, meal, index)
    date_support, late = assign_periods(meal)
    keep = supported(meal).to_numpy() & date_support
    meal, late = meal.loc[keep].copy(), late[keep]
    selected, person = np.unique(meal.person_index, return_inverse=True)
    pop, outer = pop[selected], outer[:, selected]
    households = index.iloc[pop].split_unit_id.astype(str).to_numpy()
    family_ids, family = np.unique(households[person], return_inverse=True)
    require(len(family_ids) >= 100, "Insufficient households with repeated consumption of both sources")
    flow = dict(people=len(pop), households=len(family_ids), meals=len(meal),
                early=int((~late).sum()), late=int(late.sum()), nearby_pair_people=nearby_pair_support(meal))
    period_person = 2 * person + late
    meal["person_index"] = person
    pooled, _ = bases(meal, model)
    meal["person_index"] = period_person
    split, _ = bases(meal, model)
    exchange, pair, total, extra = exposures(meal)
    period_setup = [
        ("pooled", person, pooled, model.center(extra, person)[0]),
        ("early_to_late", period_person, split, model.center(extra, period_person)[0]),
        ("late_to_early", period_person, split, model.center(extra, period_person)[0]),
    ]
    scripts = {str(p): digest(p) for p in (Path(__file__).resolve(), ROOT / "exposure.py")}
    sources = {**receipt["sources"], **input_sources, **{str(p): sha for p, sha in PINS.items()}, **scripts}
    out.mkdir(parents=True, exist_ok=False)
    reg = dict(commit=args.commit, smoke=args.smoke, flow=flow, archived_flow=original["flow"], sources=sources,
               primary=["pooled", "net_p0", "both_pc5"], bootstraps=1999, seed=SEED)
    io.write(out / "registration.json", reg)
    fp = digest(out / "registration.json")
    io.emit("CARB_SOURCE_START", **flow, smoke=args.smoke)
    prediction = {}
    names = ("none", "exchange", "tss")
    for repeat in ((0,) if args.smoke else (0, 1)):
        totals = {period: {ep: {site: {key: np.zeros(len(family_ids)) for key in names}
                                for site in MARKERS} for ep in ENDPOINTS} for period in PERIODS}
        for fold in ((0,) if args.smoke else range(5)):
            training = outer[repeat] != fold
            require(not set(households[training]) & set(households[~training]), "Household crossing")
            scores = markers(model, raw, taxa, index, pop, training, clr, mapping)
            fold_stats = {}
            for period, center_id, matrices, covars in period_setup:
                take, valid = training[person], ~training[person]
                if period != "pooled":
                    train_late = period == "late_to_early"
                    take, valid = take & (late == train_late), valid & (late != train_late)
                for site in MARKERS:
                    generic, replacement = design(scores[site], person, center_id, exchange, pair, total, model)
                    for ep in ENDPOINTS:
                        y, x, weights = matrices[ep, "carbs"]
                        x = np.column_stack((x, covars, generic))
                        parts = family_totals(y, x, weights, {"exchange": replacement}, take,
                                              family, len(family_ids), model, valid=valid)
                        for key, values in parts.items():
                            totals[period][ep][site][key] += values
                            fold_stats[f"{period}__{ep}__{site}__{key}"] = values
            buffer = BytesIO()
            np.savez_compressed(buffer, **fold_stats)
            part = out / f"r{repeat}_f{fold}.npz"
            part.write_bytes(buffer.getvalue())
            io.write(part.with_suffix(".json"), dict(registration=fp, fold=fold, repeat=repeat,
                     held=int((~training).sum()), losses=digest(part)))
            io.emit("CARB_SOURCE_FOLD", repeat=repeat, fold=fold)
        boot = np.random.default_rng(SEED + repeat).integers(0, len(family_ids), (1999, len(family_ids)), dtype=np.int32)
        prediction[f"r{repeat}"] = {period: {ep: {site: summarize(row, boot) for site, row in sites.items()}
                                            for ep, sites in endpoints.items()} for period, endpoints in totals.items()}
    direct = {}
    if not args.smoke:
        scores = markers(model, raw, taxa, index, pop, np.ones(len(pop), bool), clr, mapping)
        covars = period_setup[0][3]
        for site in MARKERS:
            generic, replacement = design(scores[site], person, person, exchange, pair, total, model)
            for ep in ENDPOINTS:
                y, x, weights = pooled[ep, "carbs"]
                fitted = model.fit_background(np.column_stack((x, covars, generic)), y, weights)
                beta, covariance, p = model.household_test(replacement, fitted, households[person], len(pop))
                row = dict(p=p, contrast_sd=float(np.std(scores[site] @ beta, ddof=1)))
                if len(beta) == 1:
                    dq = float(np.diff(np.quantile(scores[site][:, 0], [.25, .75]))[0])
                    delta = float(dq * beta[0])
                    half = dq * np.sqrt(covariance[0, 0]) * t.ppf(.975, len(family_ids) - 1)
                    row.update(q75_minus_q25=delta, ci95=[float(delta - half), float(delta + half)])
                direct[f"{ep}__{site}"] = row
        for row, p in zip(direct.values(), model.holm([row["p"] for row in direct.values()])):
            row["holm_p"] = float(p)
    require(all(digest(Path(p)) == sha for p, sha in sources.items()), "Analysis source changed")
    io.write(out / "summary.json", dict(flow=flow, prediction=prediction, direct=direct))
    io.write(out / "COMPLETE.json", dict(registration=fp, summary=digest(out / "summary.json")))
    io.emit("CARB_SOURCE_COMPLETE", smoke=args.smoke, tests=len(direct))


if __name__ == "__main__":
    main()
