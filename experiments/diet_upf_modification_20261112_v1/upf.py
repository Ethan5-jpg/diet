"""Conditional modification of the UPF–glucose association in discovery households."""

import os

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from importlib.metadata import version
from pathlib import Path
import re
import sys

import numpy as np
from scipy.stats import t

from sources import calibration, checked_import, read


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "diet_daily_modification_20261112_v1"
FOOD = "scripts/preprocessing_20261030_v1"
CONTEXT = "diet_model_context_20261108_v10"
ROUTES = (
    ("gut_shannon", False), ("oral_shannon", False), ("gut_pc5", False),
    ("gut_pc10", False), ("oral_pc5", False), ("both_pc5", False),
    ("gut_pc5", True), ("both_pc5", True),
)


def dose_design(expected, draws, lower, upper, center):
    def point(values):
        u = (values - center) / 0.1
        return np.column_stack([u, u ** 2])

    integrated = point(expected)
    integrated[:, 1] += np.var(draws / 0.1, axis=0)
    return dict(integrated=integrated, expectation=point(expected),
                lower=point(lower), upper=point(upper))


def microbial(data, training, clr, width, model):
    sites = model.site_scores(data.raw, data.taxa, data.index, data.rows, training, clr, width)
    gut = sites["gut_pc"]
    oral = sites["oral_pc"]
    markers = {site: sites[site] for site in ("gut_shannon", "oral_shannon")}
    markers.update(gut_pc5=gut[:, :5], gut_pc10=gut, oral_pc5=oral[:, :5],
                   both_pc5=np.column_stack([gut[:, :5], oral[:, :5]]))
    ecology = np.column_stack([markers["oral_shannon"], markers["gut_shannon"]])
    nuisance = np.column_stack([oral, gut, ecology, ecology ** 2, gut[:, :3] ** 2, oral[:, :3] ** 2])
    return markers, nuisance


def host_design(data, training):
    frame = data.index.iloc[data.rows]
    age = (frame.age_calendar_years.to_numpy(float) - 50) / 10
    sex = frame.sex_code.to_numpy(float)
    smoking = frame.smoking_status.fillna("__UNKNOWN__").astype(str).to_numpy()
    levels = sorted(set(smoking[training]))
    categorical = [(smoking == val).astype(float) for val in levels[1:]]
    base = np.column_stack([np.ones(len(age)), age, sex, frame.mean_daily_energy_UPF / 1000,
                            (data.upper - data.lower) / 0.1, *categorical])
    extra = data.context.copy()
    present = np.isfinite(extra)
    keep = present[training].any(axis=0)
    extra, present = extra[:, keep], present[:, keep]
    medians = np.nanmedian(np.where(present[training], extra[training], np.nan), axis=0)
    extra = np.where(present, extra, medians)
    scale = extra[training].std(axis=0)
    scale[scale <= 1e-12] = 1
    extension = np.column_stack([(extra - medians) / scale, ~present])
    return base, extension


def designs(host, nuisance, dose):
    core, extension = host
    combined = np.column_stack([core, nuisance, dose, dose[:, :1] * core[:, 1:3]])
    return {"core": combined, "joint": np.column_stack([combined, extension])}


def block(dose, marker, curve):
    width = 2 if curve else 1
    return (dose[:, :width, None] * marker[:, None, :]).reshape(len(dose), -1)


def inference(z, fitted, groups, marker, model):
    beta, cov, p = model.household_test(z, fitted, groups, n_people=0)
    # For center - 0.05 to center + 0.05 the linear basis changes by one
    # and the quadratic basis is equal at both endpoints.
    width = marker.shape[1]
    changes = marker @ beta[:width]
    result = dict(p=p, contrast_sd=float(changes.std(ddof=1)))
    if width == 1:
        spread = float(np.subtract(*np.quantile(marker[:, 0], [0.75, 0.25])))
        effect = float(spread * beta[0])
        half = float(t.ppf(0.975, len(set(groups)) - 1) * np.sqrt(cov[0, 0]) * spread)
        result.update(q75_minus_q25=effect, ci95=[effect - half, effect + half])
    return result


def analyze(data, train_full, cfg, group_draws, clr, model, seed, direct=False):
    train = train_full[data.rows]
    mean, draws, food_meta = calibration(data, train_full, group_draws, cfg, seed)
    center = float(np.median(mean[train]))
    if not 0.05 <= center <= 0.95:
        raise ValueError("The prespecified contrast leaves the valid UPF domain")
    dose_versions = dose_design(mean, draws, data.lower, data.upper, center)
    markers, nuisance = microbial(data, train, clr, cfg["pc_width"], model)
    host = host_design(data, train)
    groups = data.index.iloc[data.rows].split_unit_id.astype(str).to_numpy()
    weight = np.ones(len(data.y))
    results = {}
    for name, dose in dose_versions.items():
        for bg, x in designs(host, nuisance, dose).items():
            fitted = model.fit_background(x[train], data.y[train], weight[train])
            for marker_name, curved in ROUTES:
                z = block(dose, markers[marker_name], curved)
                key = f"{name}_{bg}_{marker_name}_{'curve' if curved else 'linear'}"
                if direct:
                    results[key] = inference(z, fitted, groups, markers[marker_name], model)
                else:
                    for penalty in cfg["penalties"]:
                        beta, sse_null, sse = model.fit_interaction(
                            x, z, data.y, weight, train, ~train, fitted, penalty)
                        m = markers[marker_name][~train]
                        change = m @ beta[:m.shape[1]]
                        results[f"{key}:{penalty}"] = dict(
                            sse=sse, sse_null=sse_null,
                            contrast=[len(change), float(change.sum()), float(change @ change)])
    common = dict(food_reference_groups=int(food_meta["distinct_reference_groups"]), center=center)
    if not direct:
        return dict(**common, results=results, held_people=int((~train).sum()))
    for row, adjusted in zip(results.values(), model.holm([x["p"] for x in results.values()])):
        row["holm_p"] = float(adjusted)
    scenarios = []
    marker = markers["gut_shannon"]
    for j in np.linspace(0, len(draws) - 1, cfg["scenario_fits"], dtype=int):
        u = (draws[j] - center) / 0.1
        dose = np.column_stack([u, u ** 2])
        x = designs(host, nuisance, dose)["joint"]
        fitted = model.fit_background(x, data.y, weight)
        scenarios.append(inference(block(dose, marker, False), fitted, groups, marker, model)["q75_minus_q25"])
    spread = (data.upper - data.lower) * 100
    return dict(**common, results=results,
                median_width_pp=float(np.median(spread)),
                calibrated_quartiles=np.quantile(mean, [0.25, 0.5, 0.75]).tolist(),
                label_scenario_contrast_quantiles=np.quantile(scenarios, [0.05, 0.5, 0.95]).tolist())


def pool(parts, tss):
    metrics = {}
    for repeat in sorted(set(fold["repeat"] for fold in parts)):
        folds = [part["results"] for part in parts if part["repeat"] == repeat]
        for key in folds[0]:
            sse0 = sum(f[key]["sse_null"] for f in folds)
            sse = sum(f[key]["sse"] for f in folds)
            n, total, square = np.sum([f[key]["contrast"] for f in folds], axis=0)
            metrics.setdefault(key, {})[f"r{repeat}"] = dict(
                delta_r2=(sse0 - sse) / tss, null_r2=1 - sse0 / tss, r2=1 - sse / tss,
                contrast_sd=float(np.sqrt(max(0, (square - total * total / n) / (n - 1)))))
    return metrics


def main():
    from hashlib import sha256
    import json

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--package", type=Path, required=True)
    cli.add_argument("--run", required=True)
    cli.add_argument("--commit", required=True)
    cli.add_argument("--smoke", action="store_true")
    cli.add_argument("--resume", action="store_true")
    args = cli.parse_args()
    package = args.package.resolve()
    if ((package / "experiments" / ROOT.name).resolve() != ROOT
            or not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run)
            or not re.fullmatch(r"[0-9a-f]{40}", args.commit)):
        raise ValueError("Invalid release directory, run name or commit")
    config = json.loads((ROOT / "config.json").read_bytes())
    inputs, model, io = checked_import(
        BASE, config["base_code"], ("inputs", "models", "run"),
        lambda path: sha256(path.read_bytes()).hexdigest())
    prep_dir = package.parent / FOOD
    food, prep = checked_import(
        prep_dir, config["food_code"], ("upf_catalogue_calibration_v2", "preprocessing_v1_20261030"),
        inputs.digest)
    context, = checked_import(
        package / "experiments" / CONTEXT, {"context.py": config["context_sha"]}, ("context",),
        inputs.digest)
    code = {str(ROOT / name): inputs.digest(ROOT / name) for name in ("sources.py", "upf.py", "config.json")}
    data = read(package, inputs, context, [prep_dir / f for f in config["food_code"]],
                [BASE / f for f in config["base_code"]])
    provenance = {**data.provenance, **code, str(Path(context.__file__)): config["context_sha"]}
    flow = dict(people=len(data.rows), households=int(data.index.iloc[data.rows].split_unit_id.nunique()),
                mean_bound_width_pp=float(np.mean(data.upper - data.lower) * 100))
    io.emit("UPF_START", commit=args.commit, **flow)
    reg = dict(commit=args.commit, sources=provenance, smoke=args.smoke, flow=flow,
               python=sys.version.split()[0], versions={n: version(n) for n in ("numpy", "pandas", "pyarrow", "scipy")})
    path = ROOT / "runs_TRE_ONLY" / args.run
    path.mkdir(parents=True, exist_ok=args.resume)
    lock = path / ".running"
    lock.mkdir()
    try:
        (lock / "pid").write_text(str(os.getpid()), encoding="ascii")
        registration = path / "registration.json"
        if args.resume:
            inputs.require(inputs.json_read(registration) == reg and not (path / "COMPLETE.json").exists(),
                           "Run sources changed or already complete")
        else:
            io.write(registration, reg)
        sha = inputs.digest(registration)
        parts = []
        schedule = [(0, 0)] if args.smoke else [(r, f) for r in (0, 1) for f in range(5)]
        for repeat, fold in schedule:
            checkpoint = path / f"r{repeat}_f{fold}.json"
            if checkpoint.is_file():
                result = inputs.json_read(checkpoint)
                inputs.require((result["repeat"], result["fold"], result["registration"]) == (repeat, fold, sha),
                               "Saved fold does not match the release")
            else:
                result = analyze(data, data.folds[repeat] != fold, config, food.grouped_probability_draws,
                                 prep.MicrobiomeCLR, model, config["seed"] + 100 * repeat + fold)
                result.update(repeat=repeat, fold=fold, registration=sha)
                io.write(checkpoint, result)
                io.emit("UPF_FOLD_DONE", repeat=repeat, fold=fold, held=result["held_people"],
                        food_groups=result["food_reference_groups"])
            parts.append(result)
        result = dict(registration=sha, flow=flow, folds=len(parts))
        if not args.smoke:
            result["prediction"] = pool(parts, np.sum((data.y - data.y.mean()) ** 2))
            result["direct"] = analyze(data, np.ones(len(data.index), bool), config,
                                       food.grouped_probability_draws, prep.MicrobiomeCLR,
                                       model, config["seed"] + 999, direct=True)
            primary = config["primary"]
            io.emit("UPF_PRIMARY", **result["direct"]["results"][primary],
                    scenarios=result["direct"]["label_scenario_contrast_quantiles"])
            io.emit("UPF_PRIMARY_PREDICTION", **result["prediction"][f'{primary}:{config["penalties"][0]}'])
        inputs.require(all(inputs.digest(Path(p)) == h for p, h in provenance.items()),
                       "Source bytes changed during analysis")
        io.write(path / "summary.json", result)
        io.write(path / "COMPLETE.json", dict(registration=sha, summary=inputs.digest(path / "summary.json")))
        io.emit("UPF_COMPLETE", run=args.run, smoke=args.smoke, folds=len(parts))
    finally:
        (lock / "pid").unlink(missing_ok=True)
        lock.rmdir()


if __name__ == "__main__":
    main()
