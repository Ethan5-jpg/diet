"""Compare fixed gut Shannon fiber contrasts on the same HPP days and people."""

import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from io import BytesIO
from pathlib import Path
import re
import sys

import numpy as np
from scipy.stats import t

from estimate import contrasts, fold_losses, group_mean, infer, summarize


ROOT = Path(__file__).resolve().parent
DAILY = ROOT.parent / "diet_daily_modification_20261112_v1"
SEED = 20261113
BOOTSTRAPS = 1999
PINS = {
    "inputs.py": "c5d28bb133105ed914de8d05f2e2a2402c13223164d4d05902ce0aba592f9f6b",
    "models.py": "13078e7de66f1b140a4ac776d6ee5572475a66abb9c798ca743a4bbe564ad432",
    "config.json": "775f55167c9564176225b50268d55a3cd6942a5d1e672f458237e26b183f9276",
    "run.py": "056cef596a6d402a6ab17e3c5406ecea4f7d08a30e3bc22606f74e503a758e38",
}

sys.path.insert(0, str(DAILY))
try:
    from inputs import digest, json_read, read, require
    from models import background, center, site_scores
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("hpp_daily_fiber_io", DAILY / "run.py")
    io = module_from_spec(spec)
    spec.loader.exec_module(io)
finally:
    sys.path.pop(0)


def dose_terms(fiber):
    return np.column_stack(((fiber - 10) / 5, np.maximum(fiber - 12.5, 0) / 5))


def raw_ecology(abundance):
    require(np.isfinite(abundance).all() and (abundance >= 0).all() and (abundance.sum(axis=1) > 0).all(),
            "Unsupported raw ecological profile")
    relative = abundance / abundance.sum(axis=1, keepdims=True)
    logged = np.log(relative, out=np.zeros_like(relative), where=relative > 0)
    return np.column_stack((-(relative * logged).sum(axis=1), relative.max(axis=1)))


def design(daily, person, count):
    fiber = daily.fiber.to_numpy(float)
    dose = dose_terms(fiber)
    age = (daily.age_calendar_years.to_numpy(float) - 50) / 10
    sex = daily.sex_code.to_numpy(float)
    target = daily.Y_value.to_numpy(float)
    y, x, w = background(daily)
    terms = np.column_stack((dose, dose * age[:, None], dose * sex[:, None]))
    within = y, np.column_stack((x, center(terms, person)[0])), center(dose, person)[0], w
    means = group_mean(np.column_stack((target, fiber, age, sex, daily.energy_kcal,
                                         daily.carb, daily.protein)), person, count)
    target, fiber, age, sex, energy, carb, protein = means.T
    dose = dose_terms(fiber)
    between = (target, np.column_stack((np.ones(count), age, sex, energy / 1000, carb / 10, protein / 10,
                                       dose, dose * age[:, None], dose * sex[:, None])), dose, np.ones(count))
    return {"within": within, "between": between}


def estimate_level(level, designs, marker, eco, pcs, person):
    y, x, dose, weight = designs[level]
    if level == "within":
        z = dose * marker[person, None]
    else:
        z = dose * marker[:, None]
        x = np.column_stack((x, eco, eco ** 2, pcs))
    return y, x, z, weight


def support(daily, marker, person, count):
    grouped = daily.groupby("person_index").fiber
    lows, highs = grouped.min().to_numpy(), grouped.max().to_numpy()
    means = grouped.mean().to_numpy()
    require(len(means) == count, "Person support alignment changed")
    q25, q75 = np.quantile(marker, [.25, .75])
    require(q75 > q25, "Degenerate Shannon contrast")
    result = {}
    for name, mask in (("low", marker <= q25), ("high", marker >= q75)):
        result[name] = dict(people=int(mask.sum()), days=int(mask[person].sum()),
                            people_spanning=int((mask & (lows <= 10) & (highs >= 15)).sum()),
                            mean_fiber_q05_q95=np.quantile(means[mask], [.05, .95]).tolist(),
                            days_near_10_12_5_15=[int((mask[person] & ((daily.fiber - dose).abs() <= 1)).sum())
                                                  for dose in (10, 12.5, 15)])
    return dict(q25=float(q25), q75=float(q75), tails=result)


def prediction(out, registration):
    reg_sha = digest(out / "registration.json")
    smoke = registration["smoke"]
    count = registration["flow"]["households"]
    result = {}
    for repeat in ((0,) if smoke else range(2)):
        totals = {level: np.zeros((count, 4)) for level in ("within", "between")}
        for fold in ((0,) if smoke else range(5)):
            file = out / f"r{repeat}_f{fold}.npz"
            require(json_read(file.with_suffix(".json")) == dict(registration=reg_sha, sha=digest(file)),
                    "A fold payload differs from its receipt")
            with np.load(BytesIO(file.read_bytes()), allow_pickle=False) as payload:
                for name in totals:
                    totals[name] += payload[name]
        rng = np.random.default_rng(SEED + repeat)
        weights = np.ones((BOOTSTRAPS + 1, count))
        for row in weights[1:]:
            row[:] = np.bincount(rng.integers(count, size=count), minlength=count)
        result[f"r{repeat}"] = {level: summarize(totals[level], weights) for level in totals}
    return result


def verify(out, commit):
    reg = json_read(out / "registration.json")
    n = 1 if reg["smoke"] else 10
    require(reg["commit"] == commit and len(list(out.glob("r*_f*.npz"))) == n
            and len(list(out.glob("r*_f*.json"))) == n, "Unexpected release or fold count")
    require(all(digest(Path(p)) == sha for p, sha in reg["sources"].items()), "An input or source changed")
    require(json_read(out / "COMPLETE.json") == dict(registration=digest(out / "registration.json"),
                                                    summary=digest(out / "summary.json")), "Completion hash mismatch")
    require(prediction(out, reg) == json_read(out / "summary.json")["prediction"], "Prediction reaggregation differs")
    io.emit("FIBER_LEVEL_READBACK_PASS", folds=n)


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
            and re.fullmatch(r"[a-zA-Z0-9_-]+", args.run), "Unexpected package, run ID or release")
    out = ROOT / "runs_TRE_ONLY" / args.run
    if args.verify:
        verify(out, args.commit)
        return
    require(not out.exists(), "Use a fresh run ID")
    require(all(digest(DAILY / name) == sha for name, sha in PINS.items()), "Daily analysis source changed")
    daily, index, raw, taxa, folds, clr, source = read(package, json_read(DAILY / "config.json"))
    pop, person = np.unique(daily.input_row.to_numpy(int), return_inverse=True)
    daily["person_index"] = person
    houses, family = np.unique(index.iloc[pop].split_unit_id.astype(str), return_inverse=True)
    count = len(houses)
    ecology = np.column_stack([raw_ecology(raw[site][pop]) for site in ("oral", "gut")])
    marker = ecology[:, 2]
    dosage = support(daily, marker, person, len(pop))
    designs = design(daily, person, len(pop))
    flow = dict(people=len(pop), households=count, days=len(daily), support=dosage)
    source.update({str(DAILY / name): sha for name, sha in PINS.items()})
    source.update({str(path): digest(path) for path in ROOT.glob("*.py")})
    reg = dict(commit=args.commit, smoke=args.smoke, flow=flow, sources=source, seed=SEED, bootstraps=BOOTSTRAPS,
               primary="linear within-minus-between gut Shannon Q75-Q25 at fiber 10 to 15")
    out.mkdir(parents=True, exist_ok=False)
    io.write(out / "registration.json", reg)
    reg_sha = digest(out / "registration.json")
    io.emit("FIBER_LEVEL_START", people=len(pop), households=count, days=len(daily), smoke=args.smoke)
    for repeat in ((0,) if args.smoke else range(2)):
        for fold in ((0,) if args.smoke else range(5)):
            valid = folds[repeat, pop] == fold
            require(not set(family[valid]) & set(family[~valid]), "Household crosses evaluation roles")
            pcs = site_scores(raw, taxa, index, pop, ~valid, clr, 5)["both_pc"]
            standardized = (marker - marker[~valid].mean()) / marker[~valid].std()
            payload = {}
            for level, held, members in (("within", valid[person], family[person]), ("between", valid, family)):
                values = estimate_level(level, designs, standardized, ecology, pcs, person)
                payload[level] = fold_losses(*values, members, count, ~held, held, absorbed=level == "within")
            path = out / f"r{repeat}_f{fold}.npz"
            buffer = BytesIO()
            np.savez_compressed(buffer, **payload)
            path.write_bytes(buffer.getvalue())
            io.write(path.with_suffix(".json"), dict(registration=reg_sha, sha=digest(path)))
            io.emit("FIBER_LEVEL_FOLD", repeat=repeat, fold=fold)
    direct = {}
    if not args.smoke:
        pcs = site_scores(raw, taxa, index, pop, np.ones(len(pop), dtype=bool), clr, 5)["both_pc"]
        iqr = dosage["q75"] - dosage["q25"]
        critical = float(t.ppf(.975, count - 1))
        for width, name in ((1, "linear"), (2, "hinge")):
            values = []
            for level, members, absorbed in (("within", family[person], len(pop)), ("between", family, 0)):
                y, x, z, weight = estimate_level(level, designs, marker, ecology, pcs, person)
                values.append(infer(y, x, z[:, :width], weight, members, count, absorbed))
            direct[name] = contrasts(*values, iqr, critical)
    io.write(out / "summary.json", dict(flow=flow, direct=direct, prediction=prediction(out, reg)))
    io.write(out / "COMPLETE.json", dict(registration=reg_sha, summary=digest(out / "summary.json")))
    io.emit("FIBER_LEVEL_COMPLETE", smoke=args.smoke)


if __name__ == "__main__":
    main()
