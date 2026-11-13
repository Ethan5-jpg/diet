"""Validate the matched-day gut Shannon contrast across disjoint periods."""

import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
from pathlib import Path
import re
import sys

import numpy as np
from scipy.stats import t

from date_split import COLUMNS, PAIRINGS, design, effect, gap, period_means


ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "diet_fiber_within_between_20261113_v1"
PINS = {"run.py": "b853a57bf93a31eed019577717e8fd4012727c6570340e35493952d37f089fcd",
        "estimate.py": "dc24d36e94ab862e194f14e95b4badddbc18dd1754e9f8a67ccba5fb6c19539b"}

sys.path.insert(0, str(PRIOR))
try:
    spec = spec_from_file_location("hpp_fiber_prior", PRIOR / "run.py")
    prior = module_from_spec(spec)
    spec.loader.exec_module(prior)
    from estimate import CONTRASTS, fold_losses, infer, summarize
finally:
    sys.path.pop(0)


digest, read_json, require, io = prior.digest, prior.json_read, prior.require, prior.io
SEED = 2026111302
BOOTSTRAPS = 1999
CROSS = ("early_to_late", "late_to_early")


def check_prior():
    require(all(digest(PRIOR / name) == value for name, value in PINS.items()), "Prior fiber sources changed")
    root = PRIOR / "runs_TRE_ONLY/discovery-1"
    prior.verify(root, "7cf7993d60b790ef4e75926c17c547c5a1871486")
    return root, read_json(root / "registration.json")


def support(periods, marker):
    q25, q75 = np.quantile(marker, [.25, .75])
    require(q75 > q25, "Shannon interquartile range is zero")
    results = {}
    for label, mask in (("low", marker <= q25), ("high", marker >= q75)):
        results[label] = dict(people=int(mask.sum()), periods={})
        for key, values in periods.items():
            fiber = values[mask, 1]
            results[label]["periods"][key] = dict(q05_q95=np.quantile(fiber, [.05, .95]).tolist(),
                                                  near=[int((np.abs(fiber - dose) <= 1).sum()) for dose in (10, 12.5, 15)])
    return dict(q25=float(q25), q75=float(q75), tails=results)


def roles(periods):
    return {key: design(periods[source], periods[target], prior.dose_terms)
            for key, (source, target) in PAIRINGS.items()}


def score_design(role, ecology, pcs, marker):
    y, x, dose = role
    return y, np.column_stack((x, ecology, ecology ** 2, pcs)), dose * marker[:, None]


def bootstrap_totals(out, registration):
    fingerprint = digest(out / "registration.json")
    count = registration["flow"]["households"]
    smoke = registration["smoke"]
    results = {}
    for repeat in ((0,) if smoke else range(2)):
        totals = {key: np.zeros((count, 4)) for key in PAIRINGS}
        for fold in ((0,) if smoke else range(5)):
            file = out / f"r{repeat}_f{fold}.npz"
            require(read_json(file.with_suffix(".json")) == {"registration": fingerprint, "sha": digest(file)},
                    "Fold payload changed")
            with np.load(file, allow_pickle=False) as payload:
                require(set(payload.files) == set(PAIRINGS), "Unexpected fold pairings")
                for key in totals:
                    totals[key] += payload[key]
        rng = np.random.default_rng(SEED + repeat)
        weights = np.ones((BOOTSTRAPS + 1, count))
        for row in weights[1:]:
            row[:] = np.bincount(rng.integers(count, size=count), minlength=count)
        results[f"r{repeat}"] = {key: summarize(value, weights) for key, value in totals.items()}
    return results


def verify(out, commit):
    reg = read_json(out / "registration.json")
    n = 1 if reg["smoke"] else 10
    require(reg["commit"] == commit and len(list(out.glob("r*_f*.npz"))) == n
            and len(list(out.glob("r*_f*.json"))) == n, "Unexpected release or fold count")
    require(all(digest(Path(path)) == sha for path, sha in reg["sources"].items()), "Source hash differs")
    require(read_json(out / "COMPLETE.json") == dict(registration=digest(out / "registration.json"),
                                                    summary=digest(out / "summary.json")), "Completion hash differs")
    require(bootstrap_totals(out, reg) == read_json(out / "summary.json")["prediction"], "Prediction readback differs")
    io.emit("FIBER_DATES_READBACK_PASS", folds=n)


def direct(roles_by_direction, eco, pcs, marker, family, count, dosage):
    iqr = dosage["q75"] - dosage["q25"]
    critical = float(t.ppf(.975, count - 1))
    joint_critical = float(t.ppf(1 - .05 / (2 * len(CROSS)), count - 1))
    one = np.ones(len(family))
    findings, linear_fits = {}, {}
    for key, role in roles_by_direction.items():
        y, x, z = score_design(role, eco, pcs, marker)
        linear = infer(y, x, z[:, :1], one, family, count, 0)
        hinge = infer(y, x, z, one, family, count, 0)
        linear_fits[key] = linear
        findings[key] = dict(linear={"full": effect(linear, [1.], iqr, critical)},
                             hinge={name: effect(hinge, vector, iqr, critical)
                                    for name, vector in CONTRASTS.items()})
        if key in CROSS:
            entry = findings[key]["linear"]["full"]
            point, margin = entry["mg_dl"], joint_critical * entry["se"]
            entry["family_ci95"] = [point - margin, point + margin]
    return findings, {key: gap(linear_fits[key], linear_fits["all_same"], iqr, critical) for key in CROSS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    package = args.package.resolve()
    require(package / "experiments" / ROOT.name == ROOT and re.fullmatch(r"[0-9a-f]{40}", args.commit)
            and re.fullmatch(r"[a-zA-Z0-9_-]+", args.run), "Unexpected package, source or run ID")
    out = ROOT / "runs_TRE_ONLY" / args.run
    if args.verify:
        verify(out, args.commit)
        return
    require(not out.exists(), "Use a fresh run ID")
    prior_root, receipt = check_prior()
    daily, index, raw, taxa, folds, clr, sources = prior.read(package, read_json(prior.DAILY / "config.json"))
    require(all(receipt["sources"].get(path) == sha for path, sha in sources.items()), "Prior matched-day sources differ")
    pop, periods, early, late = period_means(daily.input_row.to_numpy(int), daily.date.to_numpy(dtype="datetime64[D]"),
                                             daily[list(COLUMNS)].to_numpy(float))
    households, family = np.unique(index.iloc[pop].split_unit_id.astype(str), return_inverse=True)
    count = len(households)
    require(count >= 100, "Insufficient date-split household support")
    ecology = np.column_stack([prior.raw_ecology(raw[site][pop]) for site in ("oral", "gut")])
    marker = ecology[:, 2]
    dosage = support(periods, marker)
    roles_by_direction = roles(periods)
    flow = dict(people=len(pop), households=count, days=int(early.sum() + late.sum()), early=int(early.sum()),
                late=int(late.sum()), support=dosage)
    sources.update(receipt["sources"])
    sources.update({str(path): digest(path) for path in (*ROOT.glob("*.py"), prior_root / "COMPLETE.json",
                                                       prior_root / "registration.json", prior_root / "summary.json")})
    registration = dict(commit=args.commit, smoke=args.smoke, flow=flow, sources=sources, seed=SEED,
                        bootstraps=BOOTSTRAPS, primary="early_to_late linear Shannon Q75-Q25 at fiber 10 to 15")
    out.mkdir(parents=True, exist_ok=False)
    io.write(out / "registration.json", registration)
    fingerprint = digest(out / "registration.json")
    io.emit("FIBER_DATES_START", people=len(pop), households=count, days=flow["days"], smoke=args.smoke)
    one = np.ones(len(pop))
    for repeat in ((0,) if args.smoke else range(2)):
        for fold in ((0,) if args.smoke else range(5)):
            held = folds[repeat, pop] == fold
            require(not set(family[held]) & set(family[~held]), "Household crosses roles")
            pcs = prior.site_scores(raw, taxa, index, pop, ~held, clr, 5)["both_pc"]
            standardized = (marker - marker[~held].mean()) / marker[~held].std()
            results = {key: fold_losses(*score_design(role, ecology, pcs, standardized), one, family, count, ~held, held)
                       for key, role in roles_by_direction.items()}
            file = out / f"r{repeat}_f{fold}.npz"
            buffer = BytesIO()
            np.savez_compressed(buffer, **results)
            file.write_bytes(buffer.getvalue())
            io.write(file.with_suffix(".json"), dict(registration=fingerprint, sha=digest(file)))
            io.emit("FIBER_DATES_FOLD", repeat=repeat, fold=fold)
    associations, paired = {}, {}
    if not args.smoke:
        pcs = prior.site_scores(raw, taxa, index, pop, np.ones(len(pop), dtype=bool), clr, 5)["both_pc"]
        associations, paired = direct(roles_by_direction, ecology, pcs, marker, family, count, dosage)
    stability = dict(fiber_r=float(np.corrcoef(periods["early"][:, 1], periods["late"][:, 1])[0, 1]),
                     glucose_r=float(np.corrcoef(periods["early"][:, 0], periods["late"][:, 0])[0, 1]))
    io.write(out / "summary.json", dict(flow=flow, stability=stability, direct=associations,
                                       linear_vs_same_dates=paired, prediction=bootstrap_totals(out, registration)))
    io.write(out / "COMPLETE.json", dict(registration=fingerprint, summary=digest(out / "summary.json")))
    io.emit("FIBER_DATES_COMPLETE", smoke=args.smoke)


if __name__ == "__main__":
    main()
