"""Test oral microbial pathway profiles as modifiers of within-person meal response."""

import argparse
import contextlib
from io import BytesIO, StringIO
from pathlib import Path
import re
import runpy
import signal
import sys


ROOT = Path(__file__).resolve().parent
MEAL = ROOT.parent / "meal_carbohydrate_modification_20261112_v1"
PRIOR = ROOT.parent / "diet_effect_modification_20261109_v12/functional_followup_20261110_v1"
RAW = Path("/home/ec2-user/studies/hpp_datasets")
PINS = {
    PRIOR / "functional.py": "fcefca3fa10d03ac95abe9b286c573dfbcbd280371ed8d3fc5e7dc7c280d3d3a",
    PRIOR / "audit_driver.py": "0c2186ecfc4b42456437f5145847addf807d4cff9979088e30f57bcccfbcf92c",
    PRIOR / "config.json": "5f4fe8a03443b1623a9c2a980e3e8aadf3188148ab65290982ddf3cd898e3102",
    MEAL / "paired.py": "6495e7bf2fe831a971be31a1dbc1ddaf582773c13a04d89dbf7641a478239b0a",
    MEAL / "time_transfer.py": "c9295f753b4f74e9f6cade505bd35c248e6c7fdbf89a6d16263148411b0b20a8",
}

sys.path.insert(0, str(MEAL))
try:
    from paired import archived_inputs, family_totals
    from analyze import bases, markers
    from prepare import digest, load, require
    from time_transfer import assign_periods
finally:
    sys.path.pop(0)

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


OUTCOMES = ("net_p0", "peak_p0", "pre_p0")
MODES = ("abundance_relative", "coverage_confidence", "abundance_unknown_null")
PERIODS = ("pooled", "early_to_late", "late_to_early")


def audited_profiles(package, index):
    saved = sys.argv[:], signal.getsignal(signal.SIGALRM)
    log = StringIO()
    try:
        sys.argv = [str(PRIOR / "audit_driver.py"), str(RAW), str(package.parent)]
        with contextlib.redirect_stdout(log):
            audit = runpy.run_path(sys.argv[0], run_name="__hpp_meal_function_audit__")
    except SystemExit as error:
        raise RuntimeError("Prior pathway audit did not complete: " + log.getvalue()[-300:]) from error
    finally:
        sys.argv = saved[0]
        signal.alarm(0)
        signal.signal(signal.SIGALRM, saved[1])
    require(audit.get("complete") and "PATHWAY_NULL_CONTRACT_DONE" in log.getvalue(),
            "Prior pathway audit did not complete")
    keys = audit["K"]
    require(np.array_equal(audit["x"][keys].to_numpy(), index[keys].to_numpy()),
            "Meal and function index order differ")
    records, matrices, masks = audit["records"], audit["matrices"], audit["ARROW_NULL_MASKS"]
    require(records["abundance"]["columns"] == records["coverage"]["columns"]
            and len(records["abundance"]["columns"]) == 485
            and np.array_equal(masks["abundance"], masks["coverage"]), "Pathway encoding changed")
    for name, values in matrices.items():
        require(np.array_equal(np.isnan(values), masks[name]) and not np.isinf(values).any()
                and (values[np.isfinite(values)] >= 0).all(), "Invalid provided pathway signal")
    require((matrices["coverage"][np.isfinite(matrices["coverage"])] <= 1 + 1e-8).all(),
            "Unexpected coverage scale")
    positions = audit["target"].input_row.to_numpy(int)
    usable = np.nansum(matrices["abundance"], axis=1) > 0
    require(usable.sum() == 3337, "Previously audited functional population changed")
    available = np.zeros(len(index), bool)
    available[positions[usable]] = True
    signals = {}
    for name, values in matrices.items():
        signals[name] = np.full((len(index), values.shape[1]), np.nan)
        signals[name][positions] = values
    source = audit["src_path"]
    require(digest(source) == audit["src_hash"], "Oral index changed")
    columns = ["raw_read_count", "metaphlan4_aligned_read_count"]
    with pq.ParquetFile(source, pre_buffer=False) as pf:
        meta = pf.read(columns=columns, use_threads=False).to_pandas()
    counts = meta.iloc[audit["approved"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    ratio = np.full(len(counts), np.nan)
    np.divide(counts[:, 1], counts[:, 0], out=ratio, where=counts[:, 0] > 0)
    tech = np.full((len(index), 3), np.nan)
    tech[positions] = np.column_stack((counts, ratio))
    paths = [source, audit["reg_path"], *(record["path"] for record in records.values())]
    sources = {str(path): digest(path) for path in paths}
    require(all(audit["signature"](r["path"]) == r["signature"] for r in records.values()),
            "A pathway table changed after linkage")
    require(sources[str(source)] == audit["src_hash"], "Oral index changed during technical read")
    return available, signals, tech, sources, log.getvalue()


def summarize(totals, boot):
    result = {}
    tss = totals["tss"]
    require(tss.sum() > 0 and (tss[boot].sum(axis=1) > 0).all(), "Invalid evaluation denominator")
    denominator = tss[boot].sum(axis=1)
    comparisons = {f"{mode}_vs_additive": ("none", mode) for mode in MODES}
    comparisons["function_plus_taxonomy_vs_taxonomy"] = ("taxonomy", "combined")
    for key, (reference, extension) in comparisons.items():
        gain = totals[reference] - totals[extension]
        result[key] = dict(delta_r2=float(gain.sum() / tss.sum()),
                           ci95=np.quantile(gain[boot].sum(axis=1) / denominator, [.025, .975]).tolist())
    return result


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
    require(all(digest(p) == sha for p, sha in PINS.items()), "Pinned research source differs")
    out = ROOT / "runs_TRE_ONLY" / args.run
    require(not out.exists(), "Use a new run identifier")
    archived = MEAL / "runs_TRE_ONLY/models-1"
    (registration, _, io, meal, index, pop, _, _, outer, raw, taxa,
     clr, mapping) = archived_inputs(package, archived)
    model = __import__("models")
    sys.path.insert(0, str(PRIOR))
    try:
        from functional import FunctionState, TechnicalState
    finally:
        sys.path.pop(0)
    available, signals, technical, input_sources, audit_log = audited_profiles(package, index)
    eligible_dates, late = assign_periods(meal)
    keep = available[meal.input_row.to_numpy(int)] & eligible_dates
    meal, late = meal.loc[keep].copy(), late[keep]
    rows, person = np.unique(meal.person_index, return_inverse=True)
    pop, outer = pop[rows], outer[:, rows]
    households = index.iloc[pop].split_unit_id.astype(str).to_numpy()
    groups, family = np.unique(households[person], return_inverse=True)
    require(len(groups) > 20, "Insufficient households")
    flow = dict(people=len(pop), households=len(groups), meals=len(meal),
                early=int((~late).sum()), late=int(late.sum()))
    period_person = person * 2 + late
    meal["person_index"] = person
    pooled, doses = bases(meal, model)
    meal["person_index"] = period_person
    split, _ = bases(meal, model)
    config = load(PRIOR / "config.json")
    names = ("none", *MODES, "taxonomy", "combined", "tss")
    sources = {**registration["sources"], **input_sources,
               **{str(p): sha for p, sha in PINS.items()}, str(Path(__file__).resolve()): digest(Path(__file__))}
    out.mkdir(parents=True, exist_ok=False)
    (out / "PATHWAY_SOURCE_AUDIT.log").write_text(audit_log, encoding="utf-8")
    record = dict(commit=args.commit, smoke=args.smoke, flow=flow, sources=sources,
                  archived_summary=digest(archived / "summary.json"), seed=202611124, bootstraps=1999)
    io.write(out / "registration.json", record)
    fingerprint = digest(out / "registration.json")
    io.emit("MEAL_FUNCTION_START", **flow, smoke=args.smoke)
    results = {}
    for repeat in ((0,) if args.smoke else (0, 1)):
        totals = {period: {ep: {key: np.zeros(len(groups)) for key in names} for ep in OUTCOMES}
                  for period in PERIODS}
        for fold in ((0,) if args.smoke else range(5)):
            training = outer[repeat] != fold
            require(not set(households[training]) & set(households[~training]), "Household crossing")
            tech = TechnicalState().fit(technical[pop[training]]).transform(technical[pop])
            tax = markers(model, raw, taxa, index, pop, training, clr, mapping)["both_pc5"]
            scores = {"taxonomy": tax}
            for mode in MODES:
                profile = signals["coverage" if mode == "coverage_confidence" else "abundance"][pop]
                score = FunctionState(mode, config).fit(profile[training]).transform(profile)
                require(score.shape[1] >= 3, "A functional route has insufficient training rank")
                scores[mode] = score[:, :3]
            scores["combined"] = np.column_stack((tax, scores["abundance_relative"]))
            checkpoint = {}
            for period, people, matrices in (("pooled", person, pooled),
                                             ("early_to_late", period_person, split),
                                             ("late_to_early", period_person, split)):
                take = training[person]
                hold = ~take
                if period != "pooled":
                    train_late = period == "late_to_early"
                    take = take & (late == train_late)
                    hold = hold & (late != train_late)
                tech_terms = model.center(doses["carbs"][:, None] * tech[person], people)[0]
                z = {key: model.center(doses["carbs"][:, None] * score[person], people)[0]
                     for key, score in scores.items()}
                for endpoint in OUTCOMES:
                    y, x, weights = matrices[endpoint, "carbs"]
                    x = np.column_stack((x, tech_terms))
                    rows = family_totals(y, x, weights, z, take, family, len(groups), model, valid=hold)
                    for key, values in rows.items():
                        totals[period][endpoint][key] += values
                        checkpoint[f"{period}__{endpoint}__{key}"] = values
            buffer = BytesIO()
            np.savez_compressed(buffer, **checkpoint)
            part = out / f"r{repeat}_f{fold}.npz"
            part.write_bytes(buffer.getvalue())
            io.write(out / f"r{repeat}_f{fold}.json", dict(registration=fingerprint, repeat=repeat,
                     fold=fold, held=int((~training).sum()), losses=digest(part)))
            io.emit("MEAL_FUNCTION_FOLD", repeat=repeat, fold=fold)
        rng = np.random.default_rng(record["seed"] + repeat)
        boot = rng.integers(0, len(groups), (record["bootstraps"], len(groups)), dtype=np.int32)
        results[f"r{repeat}"] = {period: {ep: summarize(parts, boot) for ep, parts in endpoints.items()}
                                for period, endpoints in totals.items()}
    require(all(digest(Path(p)) == sha for p, sha in sources.items()), "An analysis source changed")
    io.write(out / "summary.json", dict(flow=flow, estimates=results))
    io.write(out / "COMPLETE.json", dict(registration=fingerprint, summary=digest(out / "summary.json")))
    io.emit("MEAL_FUNCTION_COMPLETE", smoke=args.smoke)


if __name__ == "__main__":
    main()
