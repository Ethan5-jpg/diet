"""Run a registered discovery-only daily effect-modification comparison."""

import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
from scipy import stats

from inputs import digest, json_read, read, require
from models import (CONTRAST, DIETS, background, design, fit_background,
                    fit_interaction, holm, household_test, site_scores)


SOURCE = Path(__file__).resolve().parent
CODE = ("config.json", "inputs.py", "models.py", "run.py")


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def emit(tag, **values):
    print(tag, json.dumps(values, separators=(",", ":"), allow_nan=False), flush=True)


def routes(cfg):
    for exposure, marker, curve in cfg["routes"]:
        name = f"{exposure}_{marker}_{'curve' if curve else 'linear'}"
        yield name, DIETS if exposure == "joint" else (exposure,), marker, curve


def compare(daily, designs, score, train_people, validate_people, cfg):
    y, x, w = designs
    people = daily.person_index.to_numpy()
    train, valid = train_people[people], validate_people[people]
    fitted = fit_background(x[train], y[train], w[train])
    total = float(np.sum(w[valid] * y[valid] ** 2))
    findings = {}
    for name, diets, marker, curve in routes(cfg):
        z, contrasts = design(daily, diets, score[marker][people], curve)
        for penalty in cfg["penalties"]:
            beta, sse0, sse = fit_interaction(x, z, y, w, train, valid, fitted, penalty)
            changes = {}
            for exposure, (columns, step) in contrasts.items():
                difference = (score[marker][validate_people] @ beta[columns]) * step
                changes[exposure] = [len(difference), float(difference.sum()), float(difference @ difference)]
            findings[f"{name}:{penalty}"] = dict(sse=sse, sse_null=sse0, changes=changes)
    return dict(tss=total, people=int(validate_people.sum()), results=findings)


def direct_tests(daily, designs, score, cfg):
    y, x, w = designs
    persons = daily.person_index.to_numpy()
    households = daily.split_unit_id.astype(str).to_numpy()
    n_people, n_households = len(set(persons)), len(set(households))
    fitted = fit_background(x, y, w)
    rows = []
    for name, diets, marker, curve in routes(cfg):
        z, contrasts = design(daily, diets, score[marker][persons], curve)
        beta, cov, p = household_test(z, fitted, households, n_people)
        estimates = {}
        for exposure, (columns, step) in contrasts.items():
            mod = score[marker] @ beta[columns] * step
            estimates[exposure] = dict(model_contrast_sd=float(mod.std(ddof=1)))
            if mod.shape[0] and score[marker].shape[1] == 1:
                diff = float(np.subtract(*np.quantile(score[marker][:, 0], [0.75, 0.25])) * step)
                point = float(beta[columns][0] * diff)
                se = float(np.sqrt(cov[columns, columns][0, 0]) * abs(diff))
                critical = stats.t.ppf(0.975, n_households - 1)
                estimates[exposure].update(q75_minus_q25=point, ci95=[point - critical * se, point + critical * se])
        rows.append(dict(route=name, p=p, contrasts=estimates))
    for row, q in zip(rows, holm([row["p"] for row in rows])):
        row["holm_p"] = float(q)
    return rows


def combine(checkpoints):
    results = {}
    for rep in (0, 1):
        folds = [part for part in checkpoints if part["repeat"] == rep]
        if not folds:
            continue
        tss = sum(part["tss"] for part in folds)
        for key in folds[0]["results"]:
            pieces = [part["results"][key] for part in folds]
            null = 1 - sum(part["sse_null"] for part in pieces) / tss
            alt = 1 - sum(part["sse"] for part in pieces) / tss
            contrasts = {}
            for exposure in pieces[0]["changes"]:
                n, first, second = np.sum([part["changes"][exposure] for part in pieces], axis=0)
                contrasts[exposure] = float(np.sqrt(max(0, (second - first * first / n) / (n - 1))))
            results.setdefault(key, {})[f"r{rep}"] = dict(
                null_r2=null, r2=alt, delta_r2=alt - null, contrast_sd=contrasts
            )
    return results


def support(daily):
    per_person = daily.groupby("input_row", sort=False)
    return {
        exposure: {
            "q10_median_q90": daily[exposure].quantile([0.1, 0.5, 0.9]).tolist(),
            "median_within_sd": float(per_person[exposure].std().median()),
            "people_spanning_contrast": int(
                ((per_person[exposure].min() <= low) & (per_person[exposure].max() >= high)).sum()
            ),
        }
        for exposure, (low, high) in CONTRAST.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    require(re.fullmatch(r"[a-zA-Z0-9_-]+", args.run), "Invalid run name")
    require(re.fullmatch(r"[0-9a-f]{40}", args.commit), "A full release commit is required")
    cfg = json_read(SOURCE / "config.json")
    package = args.package.resolve()
    require((package / "experiments" / SOURCE.name).resolve() == SOURCE, "Unexpected release directory")
    code = {name: digest(SOURCE / name) for name in CODE}
    emit("DAILY_START", release=args.commit, smoke=args.smoke)
    daily, index, raw, taxa, folds, clr, inputs = read(package, cfg)
    population = daily.input_row.drop_duplicates().to_numpy(dtype=int)
    daily["person_index"] = daily.input_row.map({row: j for j, row in enumerate(population)})
    flow = dict(people=len(population), households=int(daily.split_unit_id.nunique()), days=len(daily),
                support=support(daily))
    emit("DAILY_ALIGNED", **flow)
    reg = dict(commit=args.commit, code=code, sources=inputs, smoke=args.smoke, flow=flow,
               python=sys.version.split()[0],
               versions={n: version(n) for n in ("numpy", "scipy", "pandas", "pyarrow")})
    out = SOURCE / "runs_TRE_ONLY" / args.run
    out.mkdir(parents=True, exist_ok=args.resume)
    lock = out / ".running"
    lock.mkdir()
    try:
        (lock / "pid").write_text(str(os.getpid()), encoding="ascii")
        reg_path = out / "registration.json"
        if args.resume:
            require(reg_path.is_file() and json_read(reg_path) == reg, "Cannot resume different sources or code")
            require(not (out / "COMPLETE.json").exists(), "Run already complete")
        else:
            write(reg_path, reg)
        reg_sha = digest(reg_path)
        designs = background(daily)
        parts = []
        schedule = [(0, 0)] if args.smoke else [(r, f) for r in range(2) for f in range(5)]
        for repeat, fold in schedule:
            saved = out / f"r{repeat}_f{fold}.json"
            if saved.is_file():
                result = json_read(saved)
                require((result["repeat"], result["fold"], result["registration"]) == (repeat, fold, reg_sha),
                        "Unexpected saved fold provenance")
            else:
                start = time.monotonic()
                valid = folds[repeat, population] == fold
                score = site_scores(raw, taxa, index, population, ~valid, clr, cfg["pc_width"])
                result = compare(daily, designs, score, ~valid, valid, cfg)
                result.update(repeat=repeat, fold=fold, registration=reg_sha)
                write(saved, result)
                emit("DAILY_FOLD_DONE", repeat=repeat, fold=fold, people=result["people"],
                     seconds=round(time.monotonic() - start, 2))
            parts.append(result)
        summary = dict(status="smoke" if args.smoke else "complete", registration=reg_sha,
                       folds=len(parts), flow=flow, prediction=combine(parts))
        if not args.smoke:
            score = site_scores(raw, taxa, index, population, np.ones(len(population), bool), clr, cfg["pc_width"])
            summary["direct_tests"] = direct_tests(daily, designs, score, cfg)
            row = next(row for row in summary["direct_tests"] if row["route"] == cfg["primary_test"])
            emit("DAILY_PRIMARY", **row)
            emit("DAILY_PRIMARY_PREDICTION",
                 **summary["prediction"][f'{cfg["primary_test"]}:{cfg["penalties"][0]}'])
        require(all(digest(Path(path)) == value for path, value in inputs.items()),
                "Input changed during run")
        require(all(digest(SOURCE / name) == value for name, value in code.items()),
                "Code changed during run")
        write(out / "summary.json", summary)
        write(out / "COMPLETE.json", dict(registration=reg_sha, summary=digest(out / "summary.json"),
                                          smoke=args.smoke))
        emit("DAILY_COMPLETE", run=args.run, smoke=args.smoke, folds=len(parts))
    finally:
        (lock / "pid").unlink(missing_ok=True)
        lock.rmdir()


if __name__ == "__main__":
    main()
