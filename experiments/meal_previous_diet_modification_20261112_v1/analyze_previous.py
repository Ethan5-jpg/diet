"""Test whether a previous recorded meal changes the microbiome modification of current carbohydrate."""

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

from history import KEYS, PREVIOUS, design, flow, preceding, repeated, rows_with_context

ROOT = Path(__file__).resolve().parent
MEAL = ROOT.parent / "meal_carbohydrate_modification_20261112_v1"
PROCESSING = ROOT.parent / "meal_processing_modification_20261112_v1"
VALIDATION = ROOT.parent / "meal_new_household_validation_20261112_v1"
sys.path[:0] = [str(MEAL), str(PROCESSING), str(VALIDATION)]
try:
    from analyze import bases
    from audit import projected_meals
    from contracts import check_sources, completed
    from frozen import fit_frozen, save_arrays
    from paired import archived_inputs
    from prepare import NUTRIENTS, digest, food_groups, load, require, scoped_read
    from time_transfer import assign_periods
finally:
    del sys.path[:3]

PINS = {
    PROCESSING / "audit.py": "d093b86be0d041fc2eb1530329216518eb76ddae54615fcdcc2c3d0ca3463295",
    PROCESSING / "processing.py": "f249264bcdec0ed84b299cd3702ff5ed1e61937a6baf33ea8ed883fce697c567",
    VALIDATION / "contracts.py": "b191f6a86aa648f9d3760ce58efabe5aba5b874ae3014c1c638582b90177b903",
    VALIDATION / "frozen.py": "e49fe0e495fa212e0cda6accea59fdaae44a8e2def159d88a6a855a1747d79f9",
    VALIDATION / "prepare_holdout.py": "952143d739cc80f296d8129879f955b50ff6696ea66df5cd891b44e65509598a",
}
ENDPOINTS = ("net_p0", "peak_p0", "net_m15", "net_p15", "pre_p0")
PERIODS = ("pooled", "early_to_late", "late_to_early")
MODELS = ("joint", "direction")
SEED = 202611131


def projected_history(package):
    require(all(digest(p) == sha for p, sha in PINS.items()), "Pinned model and exposure helpers differ")
    old, index, cfg, inputs, io, prep, sources = projected_meals(package)
    archive = prep.parent / "meal_windows_TRE_ONLY.parquet"
    old = old.merge(pq.read_table(archive, columns=[*KEYS, "prior_gap"], use_threads=False).to_pandas(),
                    on=KEYS, validate="one_to_one")
    additional = VALIDATION / "runs_TRE_ONLY/prep-discovery-1"
    hreg, hsummary, hold_path, _ = completed(additional, "meal_windows_TRE_ONLY.parquet")
    require(not hreg["smoke"], "Use full additional household preparation")
    check_sources(hreg["sources"])
    columns = [*KEYS, "date", "carbs", "prior_gap"]
    hold = pq.read_table(hold_path, columns=columns, use_threads=False).to_pandas()
    hold = hold.merge(index[["input_row", "split_unit_id"]], on="input_row", validate="many_to_one")
    require(len(hold) == hsummary["meals"] and not set(hold.split_unit_id) & set(old.split_unit_id),
            "Additional household projection overlaps discovery training")
    roster = index.loc[index.input_row.isin(pd.concat([old.input_row, hold.input_row]))]
    events = package.parent / "analysis/shared/food_events_clean_TRE_ONLY.parquet"
    previous_audit = VALIDATION / "runs_TRE_ONLY/PREVIOUS_MEAL_FEASIBILITY_TRE_ONLY.json"
    audit = load(previous_audit)
    require(not audit["outcome_columns_read"], "Expected outcome-blind preceding meal audit")
    event_sha = digest(events)
    require(event_sha == load(prep)["sources"][str(events)] == audit["sources"][str(events)],
            "Food event input differs from the feasibility audit")
    fields = ["collection_timestamp", "local_timestamp", "qc_primary_eligible",
              "qc_same_food_time_n", "qc_exact_signature_n", *NUTRIENTS]
    food = food_groups(scoped_read(events, fields, roster, inputs))
    span = cfg["max_food_episode_span_minutes"] * 60
    histories = {label: rows_with_context(pd.concat(
        [preceding(part, food[row], NUTRIENTS, span) for row, part in table.groupby("input_row")],
        ignore_index=True)) for label, table in (("discovery", old), ("additional", hold))}
    audit_rows = histories["discovery"].loc[histories["discovery"].pair_verified]
    repeated_rows = audit_rows.loc[repeated(audit_rows)]
    require(dict(people=int(audit_rows.input_row.nunique()), meals=len(audit_rows),
                 both_people=int(repeated_rows.input_row.nunique()), both_meals=len(repeated_rows)) == audit["support"],
            "Previous episode reconstruction differs from feasibility")
    selected = {}
    for name, meal in histories.items():
        eligible = meal.loc[meal.complete_previous & meal.prior_energy.ge(cfg["minimum_kcal"])]
        selected[name] = eligible.loc[repeated(eligible)].copy()
    files = [Path(__file__).resolve(), ROOT / "history.py", previous_audit, hold_path,
             additional / "COMPLETE.json", *PINS]
    return selected, hold_path, io, {**sources, **hreg["sources"], str(events): event_sha,
                                     **{str(p): digest(p) for p in files}}


def with_history(meals, history):
    merged = meals.merge(history[[*KEYS, *PREVIOUS]], on=KEYS, validate="one_to_one")
    require(len(merged) == len(history), "A selected history row is absent from its archived CGM windows")
    selected, person = np.unique(merged.input_row, return_inverse=True)
    merged["person_index"] = person
    return merged, selected


def calendar(meal):
    keep, late = assign_periods(meal)
    half = meal.loc[keep].assign(late=late[keep])
    half["center"] = 2 * half.person_index + half.late
    support = half.loc[repeated(half, unit="center", minimum_dates=1)].groupby("input_row").center.nunique()
    half = half.loc[half.input_row.isin(support.index[support.eq(2)])].copy()
    half["person_index"] = half.center
    return half


def fit_scores(raw, taxa, index, population, train, clr, model):
    block = model.site_scores(raw, taxa, index, population, train, clr, 5)["both_pc"]
    scores = np.full((len(index), block.shape[1]), np.nan)
    scores[population] = block
    return scores


def fit_blocks(meal, matrices, scores, train, model):
    nuisance, current, interaction = design(meal, scores, model)
    out = {}
    ids = meal.input_row.to_numpy(int)
    for endpoint in ENDPOINTS:
        y, x, w = matrices[endpoint, "carbs"]
        without_current = np.column_stack([x, nuisance])
        direction = fit_frozen(y[train], without_current[train], w[train], current[train], model)[2]
        spread = np.std(scores[np.unique(ids[train])] @ direction)
        require(spread > 1e-12, "The training-only current meal direction is degenerate")
        direction = direction / spread
        x = np.column_stack([without_current, current])
        blocks = dict(joint=interaction, direction=(interaction @ direction)[:, None])
        out[endpoint] = {}
        for key, z in blocks.items():
            out[endpoint][key] = fit_frozen(y[train], x[train], w[train], z[train], model)
        out[endpoint]["direction_vector"] = direction
    return out


def losses(meal, matrices, scores, params, valid, family, n_family, model):
    nuisance, current, interaction = design(meal, scores, model)
    saved = {}
    for endpoint in ENDPOINTS:
        y, x, w = matrices[endpoint, "carbs"]
        x = np.column_stack([x, nuisance, current])[valid]
        y, w = y[valid], w[valid]
        term = interaction[valid]
        parts = {"tss": y ** 2}
        for key, z in (("joint", term), ("direction", (term @ params[endpoint]["direction_vector"])[:, None])):
            base, proj, beta = params[endpoint][key]
            null = x @ base
            change = (z - x @ proj) @ beta
            parts[key] = (y - null - change) ** 2
            parts["none"] = (y - null) ** 2
        saved.update({f"{endpoint}__{k}": np.bincount(family[valid], weights=w * v, minlength=n_family)
                      for k, v in parts.items()})
    return saved


def summarize(flat, seed):
    totals = [flat[f"{ep}__tss"] for ep in ENDPOINTS]
    use = np.flatnonzero(np.logical_or.reduce([z > 0 for z in totals]))
    require(len(use) > 1, "Insufficient paired household outcomes")
    draws = np.random.default_rng(seed).choice(use, (1999, len(use)), replace=True)
    result = {}
    for endpoint, tss in zip(ENDPOINTS, totals):
        denom = tss[draws].sum(axis=1)
        require((denom > 0).all(), "Invalid paired household denominator")
        row = {}
        for key in MODELS:
            gain = flat[f"{endpoint}__none"] - flat[f"{endpoint}__{key}"]
            row[key] = dict(delta_r2=float(gain.sum() / tss.sum()),
                            ci95=np.quantile(gain[draws].sum(axis=1) / denom, [.025, .975]).tolist())
        result[endpoint] = row
    return result


def direct(meal, matrices, scores, model):
    nuisance, current, interaction = design(meal, scores, model)
    households = meal.split_unit_id.astype(str).to_numpy()
    people = meal.input_row.unique()
    result = {}
    for endpoint in ENDPOINTS:
        y, x, w = matrices[endpoint, "carbs"]
        fitted = model.fit_background(np.column_stack([x, nuisance, current]), y, w)
        beta, _, p = model.household_test(interaction, fitted, households, len(people))
        result[endpoint] = dict(p=p, effect_sd=float(np.std(scores[people] @ beta, ddof=1)))
    for row, adjusted in zip(result.values(), model.holm([row["p"] for row in result.values()])):
        row["holm_p"] = float(adjusted)
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--package", required=True, type=Path)
    cli.add_argument("--commit", required=True)
    cli.add_argument("--run", required=True)
    cli.add_argument("--smoke", action="store_true")
    args = cli.parse_args()
    package = args.package.resolve()
    require(package / "experiments" / ROOT.name == ROOT and re.fullmatch(r"[0-9a-f]{40}", args.commit)
            and re.fullmatch(r"[a-zA-Z0-9_-]+", args.run), "Unexpected study path or release")
    out = ROOT / "runs_TRE_ONLY" / args.run
    require(not out.exists(), "Use a fresh run identifier")
    histories, hold_path, io, sources = projected_history(package)
    archive = MEAL / "runs_TRE_ONLY/models-1"
    (receipt, _, _, meal, index, pop, _, _, outer, raw, taxa, clr, _) = archived_inputs(package, archive)
    model = __import__("models")
    pooled, people = with_history(meal, histories["discovery"])
    dated = calendar(pooled)
    household = index.iloc[people].split_unit_id.astype(str).to_numpy()
    groups, families = np.unique(household, return_inverse=True)
    person = {row: k for k, row in enumerate(people)}
    splits = np.searchsorted(pop, people)
    require(np.array_equal(pop[splits], people), "Frozen training rows differ")
    datasets = {}
    for name, rows in (("pooled", pooled), ("calendar", dated)):
        units = rows.input_row.map(person).to_numpy(int)
        datasets[name] = rows, bases(rows, model)[0], units
    support = dict(discovery=flow(pooled), calendar=flow(dated), additional=flow(histories["additional"]))
    require(support["calendar"]["households"] >= 10, "Insufficient chronological household support")
    sources = {**receipt["sources"], **sources}
    out.mkdir(parents=True, exist_ok=False)
    reg = out / "registration.json"
    io.write(reg, dict(commit=args.commit, smoke=args.smoke, sources=sources, support=support,
                       seed=SEED, primary=["net_p0", "joint"], bootstrap=1999))
    fingerprint = digest(reg)
    io.emit("PREVIOUS_MEAL_START", smoke=args.smoke, **support)
    prediction = {}
    for repeat in ((0,) if args.smoke else (0, 1)):
        sums = {p: {} for p in PERIODS}
        for fold in ((0,) if args.smoke else range(5)):
            train = outer[repeat, splits] != fold
            require(not set(household[train]) & set(household[~train]), "Household crossing")
            scores = fit_scores(raw, taxa, index, people, train, clr, model)
            checkpoint = {}
            for period in PERIODS:
                rows, matrices, units = datasets["pooled" if period == "pooled" else "calendar"]
                take, valid = train[units], ~train[units]
                if period != "pooled":
                    later = period == "late_to_early"
                    take, valid = take & rows.late.eq(later).to_numpy(), valid & rows.late.ne(later).to_numpy()
                params = fit_blocks(rows, matrices, scores, take, model)
                part = losses(rows, matrices, scores, params, valid, families[units], len(groups), model)
                for name, value in part.items():
                    if name not in sums[period]:
                        sums[period][name] = np.zeros_like(value)
                    sums[period][name] += value
                    checkpoint[f"{period}__{name}"] = value
            part = out / f"r{repeat}_f{fold}.npz"
            save_arrays(part, checkpoint)
            io.write(part.with_suffix(".json"), dict(registration=fingerprint, repeat=repeat, fold=fold, sha=digest(part)))
            io.emit("PREVIOUS_MEAL_FOLD", repeat=repeat, fold=fold)
        prediction[f"r{repeat}"] = {k: summarize(v, SEED + repeat) for k, v in sums.items()}
    estimates, external = {}, {}
    artifacts = {}
    if not args.smoke:
        can_transfer = support["additional"]["households"] >= 10
        population = np.union1d(people, histories["additional"].input_row.unique()) if can_transfer else people
        scores = fit_scores(raw, taxa, index, population, np.isin(population, people), clr, model)
        matrices = datasets["pooled"][1]
        estimates = direct(pooled, matrices, scores, model)
    if not args.smoke and can_transfer:
        params = fit_blocks(pooled, matrices, scores, np.ones(len(pooled), bool), model)
        frozen = dict(scores=scores)
        for ep, blocks in params.items():
            frozen[f"{ep}__direction_vector"] = blocks["direction_vector"]
            for key in MODELS:
                frozen.update({f"{ep}__{key}__{name}": z for name, z in zip(("base", "projection", "beta"), blocks[key])})
        path = out / "frozen_history_TRE_ONLY.npz"
        save_arrays(path, frozen)
        io.write(out / "FROZEN.json", dict(registration=fingerprint, frozen=digest(path), new_outcomes_read=False))
        artifacts.update({p.name: digest(p) for p in (path, out / "FROZEN.json")})
        hold = pq.read_table(hold_path, use_threads=False, pre_buffer=False).to_pandas()
        hold = hold.merge(index[["input_row", "age_calendar_years", "sex_code", "split_unit_id"]],
                          on="input_row", validate="many_to_one")
        hold, _ = with_history(hold, histories["additional"])
        require(not set(hold.split_unit_id) & set(household), "Additional household crosses training")
        family, inv = np.unique(hold.split_unit_id.astype(str), return_inverse=True)
        parts = losses(hold, bases(hold, model)[0], scores, params, np.ones(len(hold), bool), inv, len(family), model)
        path = out / "additional_losses_TRE_ONLY.npz"
        save_arrays(path, parts)
        artifacts[path.name] = digest(path)
        external = summarize(parts, SEED + 2)
    check_sources(sources)
    summary = out / "summary.json"
    io.write(summary, dict(support=support, prediction=prediction, direct=estimates, additional=external))
    io.write(out / "COMPLETE.json", dict(registration=fingerprint, summary=digest(summary), artifacts=artifacts))
    io.emit("PREVIOUS_MEAL_COMPLETE", smoke=args.smoke, tests=len(estimates))


if __name__ == "__main__":
    main()
