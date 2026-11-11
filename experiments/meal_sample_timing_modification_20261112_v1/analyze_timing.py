"""Test whether the carbohydrate interaction varies with stool handoff proximity."""

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

from inputs_timing import MEAL, ROOT, archived_inputs, digest, load, require, timing_inputs
from timing import KEYS, STRATA, design, overlap, supported

sys.path.insert(0, str(MEAL))
try:
    from analyze import bases
    from paired import family_totals
finally:
    sys.path.pop(0)

MARKERS = ("gut_pc", "gut_shannon")
ENDPOINTS = ("net_p0", "peak_p0", "net_m15", "net_p15", "pre_p0")
SEED = 202611128


def predictions(data, family, training, score, model):
    meal, person, matrices = data
    nuisance, timing = design(meal, score, person, model)
    near = meal.near.to_numpy()
    fit = training[person]
    n_family = family.max() + 1
    result = {}
    for endpoint in ENDPOINTS:
        y, x, weights = matrices[endpoint, "carbs"]
        x = np.column_stack((x, nuisance))
        for context, valid in (("near", near), ("far", ~near)):
            result[context, endpoint] = family_totals(
                y, x, weights, {"timing": timing}, fit, family[person], n_family, model, valid=~fit & valid,
            )
    return result


def pooled_summary(flat, repeat):
    result = {}
    for s in STRATA:
        for marker in MARKERS:
            for endpoint in ENDPOINTS:
                context = {k: {v: flat[f"{s}__{marker}__{k}__{endpoint}__{v}"] for v in ("none", "timing", "tss")}
                           for k in ("near", "far")}
                context["pooled"] = {k: context["near"][k] + context["far"][k] for k in ("none", "timing", "tss")}
                support = np.flatnonzero(context["pooled"]["tss"] > 0)
                require(len(support) > 0, "No held-out households")
                draws = np.random.default_rng(SEED + repeat).choice(support, (1999, len(support)), replace=True)
                for name, row in context.items():
                    denom = row["tss"][draws].sum(axis=1)
                    require(row["tss"].sum() > 0 and (denom > 0).all(), "Nonpositive matched denominator")
                    gain = row["none"] - row["timing"]
                    result[f"{s}__{marker}__{name}__{endpoint}"] = dict(
                        delta_r2=float(gain.sum() / row["tss"].sum()),
                        baseline_r2=float(1 - row["none"].sum() / row["tss"].sum()),
                        ci95=np.quantile(gain[draws].sum(axis=1) / denom, [.025, .975]).tolist(),
                    )
    return result


def associations(data, households, score, model):
    meal, person, matrices = data
    nuisance, timing = design(meal, score, person, model)
    rows = {}
    dof = len(np.unique(households)) - 1
    for endpoint in ENDPOINTS:
        y, x, weights = matrices[endpoint, "carbs"]
        fitted = model.fit_background(np.column_stack((x, nuisance)), y, weights)
        residualized, _, _ = model.orthogonal_block(timing, fitted)
        beta, cov, p = model.household_test(timing, fitted, households[person], len(score))
        row = dict(p=p, contrast_sd=float(np.std(score @ beta, ddof=1)),
                   remaining_term_fraction=float((residualized ** 2).sum() / (timing ** 2 * weights[:, None]).sum()))
        if len(beta) == 1:
            spread = float(np.diff(np.quantile(score[:, 0], [.25, .75]))[0])
            delta = float(spread * beta[0])
            half = float(spread * np.sqrt(cov[0, 0]) * t.ppf(.975, dof))
            row.update(q75_minus_q25=delta, ci95=[delta - half, delta + half])
        rows[endpoint] = row
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
    aligned, io, sources = timing_inputs(package)
    selected = {s: supported(aligned, s) for s in STRATA}
    flows = {s: overlap(rows) for s, rows in selected.items()}
    io.emit("HANDOFF_SUPPORT", identity_matched_people=int(aligned.input_row.nunique()), **flows)
    require(all(v["households"] >= 10 and v["common_monitor_days"] >= 2 for v in flows.values()),
            "Insufficient shared monitoring-day support")
    archive = MEAL / "runs_TRE_ONLY/models-1"
    (receipt, _, _, meal, index, pop, _, _, outer, raw, taxa, clr, _) = archived_inputs(package, archive)
    model = __import__("models")
    datasets = {}
    for s, timing in selected.items():
        joined = meal.merge(timing[[*KEYS, "monitor_day", "handoff_lag", "near"]], on=KEYS, validate="one_to_one")
        require(len(joined) == len(timing), "Aligned meal is missing from the archived outcomes")
        subset, person = np.unique(joined.person_index, return_inverse=True)
        joined["person_index"] = person
        people = pop[subset]
        households = index.iloc[people].split_unit_id.astype(str).to_numpy()
        _, family = np.unique(households, return_inverse=True)
        datasets[s] = (joined, person, bases(joined, model)[0]), people, outer[:, subset], households, family
    sources = {**receipt["sources"], **sources, str(Path(__file__).resolve()): digest(Path(__file__).resolve())}
    out.mkdir(parents=True, exist_ok=False)
    reg_path = out / "registration.json"
    io.write(reg_path, dict(commit=args.commit, smoke=args.smoke, flow=flows, sources=sources, seed=SEED, bootstraps=1999,
                           primary=["all_doses", "gut_pc", "pooled", "net_p0"],
                           clock="gut dictionary sample-given date and verified first observed CGM local date"))
    fingerprint = digest(reg_path)
    prediction = {}
    for repeat in ((0,) if args.smoke else (0, 1)):
        total = {}
        for fold in ((0,) if args.smoke else range(5)):
            checkpoint = {}
            for s, (data, people, splits, households, family) in datasets.items():
                train = splits[repeat] != fold
                require(not set(households[train]) & set(households[~train]), "Household crossing")
                scores = model.site_scores(raw, taxa, index, people, train, clr, 5)
                for marker in MARKERS:
                    parts = predictions(data, family, train, scores[marker], model)
                    for (context, endpoint), values in parts.items():
                        for key, v in values.items():
                            name = f"{s}__{marker}__{context}__{endpoint}__{key}"
                            if name not in total:
                                total[name] = np.zeros_like(v)
                            total[name] += v
                            checkpoint[name] = v
            buffer = BytesIO()
            np.savez_compressed(buffer, **checkpoint)
            saved = out / f"r{repeat}_f{fold}.npz"
            saved.write_bytes(buffer.getvalue())
            io.write(saved.with_suffix(".json"), dict(registration=fingerprint, repeat=repeat, fold=fold, sha=digest(saved)))
            io.emit("HANDOFF_FOLD", repeat=repeat, fold=fold)
        prediction[f"r{repeat}"] = pooled_summary(total, repeat)
    estimates = {}
    if not args.smoke:
        for s, (data, people, _, households, _) in datasets.items():
            scores = model.site_scores(raw, taxa, index, people, np.ones(len(people), bool), clr, 5)
            for marker in MARKERS:
                for endpoint, row in associations(data, households, scores[marker], model).items():
                    estimates[f"{s}__{marker}__{endpoint}"] = row
        for row, adjusted in zip(estimates.values(), model.holm([row["p"] for row in estimates.values()])):
            row["holm_p"] = float(adjusted)
    require(all(digest(Path(p)) == sha for p, sha in sources.items()), "Model inputs changed")
    summary = out / "summary.json"
    io.write(summary, dict(flow=flows, prediction=prediction, direct=estimates))
    io.write(out / "COMPLETE.json", dict(registration=fingerprint, summary=digest(summary)))
    io.emit("HANDOFF_COMPLETE", smoke=args.smoke, tests=len(estimates))


if __name__ == "__main__":
    main()
