"""Evaluate NOVA carbohydrate replacement beyond generic microbial modification."""

import argparse
from io import BytesIO
from pathlib import Path
import re
import sys

from audit import MEAL, PINS, ROOT, assign_periods, classified_sources, digest, flow, load, projected_meals, require
from processing import CARBS, KEYS, coverage, repeated_support

sys.path.insert(0, str(MEAL))
try:
    from analyze import bases, contrasts, markers
    from paired import archived_inputs, family_totals
finally:
    sys.path.pop(0)

import numpy as np
from scipy.stats import t

STRATA = ("classified_carb", "high_coverage")
ENDPOINTS = ("net_p0", "peak_p0", "net_m15", "net_p15", "pre_p0")
MARKERS = ("gut_shannon", "both_pc5")
SEED = 202611126


def design(meals, score, person, center_id, model):
    total = contrasts(meals)["carbs"]
    upf = meals.upf_carbs.to_numpy(float) / 10
    unknown = meals.unknown_carbs.to_numpy(float) / 10
    unknown_energy = meals.unknown_kcal.to_numpy(float) / 100
    age = (meals.age_calendar_years.to_numpy(float) - 50) / 10
    sex = meals.sex_code.to_numpy(float)
    profile = score[person]
    nuisance = np.column_stack([upf, upf ** 2, upf * total, upf * age, upf * sex, unknown, unknown_energy,
                                profile * total[:, None], profile * unknown[:, None]])
    return model.center(nuisance, center_id)[0], model.center(profile * upf[:, None], center_id)[0]


def analysis_set(meal, audit, model):
    meal = meal.loc[repeated_support(meal, CARBS[:2])].copy()
    require(flow(meal) == audit["repeated"], "Model population differs from outcome-blind support")
    frozen, person = np.unique(meal.person_index, return_inverse=True)
    meal["person_index"] = person
    date_ok, late = assign_periods(meal)
    dated = meal.loc[date_ok].copy()
    dated["period"] = 2 * dated.input_row + late[date_ok]
    per_period = dated.loc[repeated_support(dated, CARBS[:2], days=1, unit="period")]
    present = per_period.groupby("input_row").period.nunique().eq(2)
    balanced = dated.input_row.isin(present.index[present])
    dated = dated.loc[balanced].copy()
    late = late[date_ok][balanced.to_numpy()]
    require(flow(dated) == audit["both_sources_each_period"], "Date-transfer support differs")
    pooled = (meal, person, person, None, bases(meal, model)[0])
    period_center = 2 * dated.person_index.to_numpy() + late
    dated_person = dated.person_index.to_numpy()
    dated["person_index"] = period_center
    chrono = (dated, dated_person, period_center, late, bases(dated, model)[0])
    return frozen, pooled, chrono


def paired_summary(parts, draws):
    gain = parts["none"] - parts["processing"]
    tss = parts["tss"]
    den = tss[draws].sum(axis=1)
    require(tss.sum() > 0 and (den > 0).all(), "Invalid held-out denominator")
    return dict(delta_r2=float(gain.sum() / tss.sum()),
                baseline_r2=float(1 - parts["none"].sum() / tss.sum()),
                ci95=np.quantile(gain[draws].sum(axis=1) / den, [.025, .975]).tolist())


def summarize_stratum(values, repeat):
    results = {}
    for direction in ("pooled", "early_to_late", "late_to_early"):
        matching = {k: v for k, v in values.items() if k.startswith(direction + "__")}
        # Date transfer evaluates a smaller household set. Draw only from
        # households with evaluable outcomes in the corresponding direction.
        support = np.flatnonzero(np.logical_or.reduce([v["tss"] > 0 for v in matching.values()]))
        require(len(support) > 0, "No held-out households")
        draws = np.random.default_rng(SEED + repeat).choice(support, (1999, len(support)), replace=True)
        results.update({name: paired_summary(parts, draws) for name, parts in matching.items()})
    return results


def evaluate(dataset, site, training, family, groups, model, direction):
    meal, person, center, late, matrices = dataset
    take, valid = training[person], ~training[person]
    if direction != "pooled":
        late_train = direction == "late_to_early"
        take, valid = take & (late == late_train), valid & (late != late_train)
    nuisance, exchange = design(meal, site, person, center, model)
    losses = {}
    for endpoint in ENDPOINTS:
        y, x, w = matrices[endpoint, "carbs"]
        losses[endpoint] = family_totals(y, np.column_stack([x, nuisance]), w, {"processing": exchange},
                                         take, family[person], groups, model, valid=valid)
    return losses


def direct(dataset, score, households, model):
    meal, person, center, _, matrices = dataset
    nuisance, exchange = design(meal, score, person, center, model)
    rows = {}
    for endpoint in ENDPOINTS:
        y, x, w = matrices[endpoint, "carbs"]
        fitted = model.fit_background(np.column_stack([x, nuisance]), y, w)
        beta, cov, p = model.household_test(exchange, fitted, households[person], len(score))
        row = dict(p=p, contrast_sd=float(np.std(score @ beta, ddof=1)))
        if len(beta) == 1:
            iqr = float(np.diff(np.quantile(score[:, 0], [.25, .75]))[0])
            delta = float(iqr * beta[0])
            margin = float(iqr * np.sqrt(cov[0, 0]) * t.ppf(.975, len(np.unique(households)) - 1))
            row.update(q75_minus_q25=delta, ci95=[delta - margin, delta + margin])
        rows[endpoint] = row
    return rows


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--package", type=Path, required=True)
    cli.add_argument("--audit", default="audit-discovery-1")
    cli.add_argument("--run", required=True)
    cli.add_argument("--commit", required=True)
    cli.add_argument("--smoke", action="store_true")
    args = cli.parse_args()
    package = args.package.resolve()
    require(package / "experiments" / ROOT.name == ROOT and re.fullmatch(r"[a-zA-Z0-9_-]+", args.run)
            and re.fullmatch(r"[a-zA-Z0-9_-]+", args.audit) and re.fullmatch(r"[0-9a-f]{40}", args.commit),
            "Unexpected release or run path")
    out = ROOT / "runs_TRE_ONLY" / args.run
    require(not out.exists(), "Use a fresh run identifier")
    audit = ROOT / "runs_TRE_ONLY" / args.audit
    record = load(audit / "COMPLETE.json")
    registered, summary = audit / "registration.json", audit / "summary.json"
    require(digest(registered) == record["registration"] and digest(summary) == record["summary"],
            "Outcome-blind audit receipt changed")
    audit_reg, audited = load(registered), load(summary)
    require(not audit_reg["smoke"] and not audit_reg["outcome_columns_read"]
            and all(digest(Path(path)) == sha for path, sha in audit_reg["sources"].items()),
            "Use a complete and verified outcome-blind audit")
    projected, index, cfg, inputs, io, prep, _ = projected_meals(package)
    processing, food_hashes, _ = classified_sources(package, projected, index, cfg, inputs, prep)
    require(all(audit_reg["sources"][path] == sha for path, sha in food_hashes.items()),
            "Processing lookup differs from the verified audit")
    archive = MEAL / "runs_TRE_ONLY/models-1"
    (receipt, _, _, meal, index, pop, _, _, outer, raw, taxa, clr, mapping) = archived_inputs(package, archive)
    new_cols = [col for col in processing.columns if col not in meal.columns]
    meal = meal.merge(processing[[*KEYS, *new_cols]], on=KEYS, validate="one_to_one")
    require(len(meal) == len(projected), "Meal model and audit rows differ")
    model = __import__("models")
    model_sets = {}
    flags = coverage(meal)
    for stratum in STRATA:
        frozen, pooled, chrono = analysis_set(meal.loc[flags[stratum]], audited[stratum], model)
        households = index.iloc[pop[frozen]].split_unit_id.astype(str).to_numpy()
        require(len(np.unique(households)) > len(MARKERS), "Insufficient supported households")
        model_sets[stratum] = frozen, households, pooled, chrono
    files = (Path(__file__).resolve(), registered, summary, *PINS)
    sources = {**receipt["sources"], **audit_reg["sources"], **{str(p): digest(p) for p in files}}
    out.mkdir(parents=True, exist_ok=False)
    registration = out / "registration.json"
    io.write(registration, dict(commit=args.commit, smoke=args.smoke, sources=sources, seed=SEED, bootstraps=1999,
                                primary=["classified_carb", "pooled", "net_p0", "gut_shannon"]))
    fingerprint = digest(registration)
    prediction = {}
    directions = ("pooled", "early_to_late", "late_to_early")
    io.emit("PROCESSING_MODEL_START", smoke=args.smoke)
    for repeat in ((0,) if args.smoke else (0, 1)):
        totals = {stratum: {} for stratum in STRATA}
        for fold in ((0,) if args.smoke else range(5)):
            fold_losses = {}
            for stratum, (frozen, households, pooled, chrono) in model_sets.items():
                family_ids, family = np.unique(households, return_inverse=True)
                training = outer[repeat, frozen] != fold
                require(not set(households[training]) & set(households[~training]), "Household crossing")
                scores = markers(model, raw, taxa, index, pop[frozen], training, clr, mapping)
                for direction in directions:
                    dataset = pooled if direction == "pooled" else chrono
                    for marker in MARKERS:
                        losses = evaluate(dataset, scores[marker], training, family, len(family_ids), model, direction)
                        for endpoint, parts in losses.items():
                            key = f"{direction}__{endpoint}__{marker}"
                            if key not in totals[stratum]:
                                totals[stratum][key] = {name: np.zeros(len(family_ids)) for name in parts}
                            for name, loss in parts.items():
                                totals[stratum][key][name] += loss
                                fold_losses[f"{stratum}__{key}__{name}"] = loss
            buffer = BytesIO()
            np.savez_compressed(buffer, **fold_losses)
            checkpoint = out / f"r{repeat}_f{fold}.npz"
            checkpoint.write_bytes(buffer.getvalue())
            io.write(checkpoint.with_suffix(".json"), dict(registration=fingerprint, repeat=repeat,
                                                           fold=fold, sha=digest(checkpoint)))
            io.emit("PROCESSING_MODEL_FOLD", repeat=repeat, fold=fold)
        prediction[f"r{repeat}"] = {stratum: summarize_stratum(values, repeat) for stratum, values in totals.items()}
    estimates = {}
    if not args.smoke:
        for stratum, (frozen, households, pooled, _) in model_sets.items():
            scores = markers(model, raw, taxa, index, pop[frozen], np.ones(len(frozen), bool), clr, mapping)
            for marker in MARKERS:
                for endpoint, row in direct(pooled, scores[marker], households, model).items():
                    estimates[f"{stratum}__{endpoint}__{marker}"] = row
        for row, adjusted in zip(estimates.values(), model.holm([row["p"] for row in estimates.values()])):
            row["holm_p"] = float(adjusted)
    require(all(digest(Path(p)) == sha for p, sha in sources.items()), "Model inputs changed")
    summary_path = out / "summary.json"
    io.write(summary_path, dict(prediction=prediction, direct=estimates, support={s: audited[s] for s in STRATA}))
    io.write(out / "COMPLETE.json", dict(registration=fingerprint, summary=digest(summary_path)))
    io.emit("PROCESSING_MODEL_COMPLETE", smoke=args.smoke, tests=len(estimates))


if __name__ == "__main__":
    main()
