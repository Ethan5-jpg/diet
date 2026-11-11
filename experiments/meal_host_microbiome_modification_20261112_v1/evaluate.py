"""Benchmark microbial meal effect modification against antecedent host markers."""

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
    from prepare import digest, load, require
    from time_transfer import assign_periods
finally:
    sys.path.pop(0)

import numpy as np
from scipy.stats import t

from clinical import RAW, read_measurements, scale


PANELS = {"hba1c_bmi": ("hba1c", "bmi"), "bmi": ("bmi",)}
ENDPOINTS = ("net_p0", "peak_p0", "pre_p0")
PERIODS = ("pooled", "early_to_late", "late_to_early")
CONTRASTS = {
    "host": ("original", "host"),
    "microbe_without_host": ("original", "microbe_alone"),
    "microbe_beyond_host": ("host", "host_microbe"),
    "host_microbe_interaction": ("host_microbe", "combined"),
}
SEED = 202611127
BOOTSTRAPS = 1999


def terms(dose, person, center_id, host, microbial, model):
    by_host = dose[:, None] * host[person]
    by_microbe = dose[:, None] * microbial[person]
    # The first host variable is the registered clinical context for the three-way block.
    triple = by_microbe * host[person, :1]
    return tuple(model.center(z, center_id)[0] for z in (by_host, by_microbe, triple))


def fold_prediction(matrices, design, training, heldout, family, groups, model):
    y, x, weights = matrices
    host, micro, triple = design
    original = family_totals(y, x, weights, {"microbe": micro}, training, family, groups, model, valid=heldout)
    changed = family_totals(y, np.column_stack((x, host)), weights,
                           {"microbe": micro, "combined": np.column_stack((micro, triple))},
                           training, family, groups, model, valid=heldout)
    return dict(original=original["none"], microbe_alone=original["microbe"], host=changed["none"],
                host_microbe=changed["microbe"], combined=changed["combined"], tss=original["tss"])


def summarize(values, draws):
    tss = values["tss"]
    den = tss[draws].sum(axis=1)
    require(tss.sum() > 0 and (den > 0).all(), "Invalid prediction denominator")
    report = {}
    for name, (base, full) in CONTRASTS.items():
        delta = values[base] - values[full]
        report[name] = dict(delta_r2=float(delta.sum() / tss.sum()),
                            reference_r2=float(1 - values[base].sum() / tss.sum()),
                            ci95=np.quantile(delta[draws].sum(axis=1) / den, [.025, .975]).tolist())
    return report


def aggregate(totals, draws):
    result = {}
    for period in PERIODS:
        result[period] = {}
        for endpoint in ENDPOINTS:
            prefix = f"{period}__{endpoint}__"
            values = {key[len(prefix):]: v for key, v in totals.items() if key.startswith(prefix)}
            result[period][endpoint] = summarize(values, draws)
    return result


def direct_tests(matrices, design, groups, person, microbial, names, model):
    host, micro, triple = design
    result = {}
    for endpoint in ENDPOINTS:
        y, x, weights = matrices[endpoint, "carbs"]
        w = np.column_stack((x, host))
        with_micro = np.column_stack((w, micro))
        models = (("host", x, host), ("microbe_beyond_host", w, micro),
                  ("host_microbe_interaction", with_micro, triple))
        for name, baseline, z in models:
            fitted = model.fit_background(baseline, y, weights)
            b, covariance, p = model.household_test(z, fitted, groups[person], len(groups))
            row = dict(p=p)
            if name == "host":
                margin = t.ppf(.975, len(np.unique(groups)) - 1) * np.sqrt(np.diag(covariance))
                row["contrast"] = {key: dict(value=float(v), ci95=[float(v - m), float(v + m)])
                                   for key, v, m in zip(names, b, margin)}
            else:
                row["contrast_sd"] = float(np.std(microbial @ b, ddof=1))
            result[f"{endpoint}__{name}"] = row
    for row, p in zip(result.values(), model.holm([v["p"] for v in result.values()])):
        row["holm_p"] = float(p)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--panel", required=True, choices=PANELS)
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
    receipt, original, io, meal, index, pop, _, _, outer, raw, taxa, clr, mapping = archived_inputs(package, archive)
    model = __import__("models")
    names = PANELS[args.panel]
    clinical, audit, source = read_measurements(Path(RAW), meal, index, names, digest)
    complete = clinical.dropna()
    support, late = assign_periods(meal)
    keep = meal.input_row.isin(complete.index).to_numpy() & support
    meal, late = meal.loc[keep].copy(), late[keep]
    positions, person = np.unique(meal.person_index, return_inverse=True)
    pop, outer = pop[positions], outer[:, positions]
    host = scale(complete.loc[pop, list(names)].to_numpy(float), names)
    households = index.iloc[pop].split_unit_id.astype(str).to_numpy()
    family_ids, family = np.unique(households[person], return_inverse=True)
    require(len(family_ids) >= 250 and np.isfinite(host).all(), "Insufficient clinical and time support")
    flow = dict(people=len(pop), households=len(family_ids), meals=len(meal),
                early=int((~late).sum()), late=int(late.sum()))
    period_person = 2 * person + late
    meal["person_index"] = person
    pooled, doses = bases(meal, model)
    meal["person_index"] = period_person
    split, _ = bases(meal, model)
    setup = (("pooled", person, pooled), ("early_to_late", period_person, split),
             ("late_to_early", period_person, split))
    scripts = {str(p): digest(p) for p in (Path(__file__).resolve(), ROOT / "clinical.py", archive / "summary.json")}
    sources = {**receipt["sources"], **source, **{str(p): sha for p, sha in PINS.items()}, **scripts}
    out.mkdir(parents=True, exist_ok=False)
    reg = dict(commit=args.commit, panel=args.panel, smoke=args.smoke, flow=flow, clinical_audit=audit,
               archived_flow=original["flow"], sources=sources, seed=SEED, bootstraps=BOOTSTRAPS)
    io.write(out / "registration.json", reg)
    fingerprint = digest(out / "registration.json")
    io.emit("HOST_MOD_START", panel=args.panel, **flow, smoke=args.smoke)
    predictions = {}
    for repeat in ((0,) if args.smoke else (0, 1)):
        totals = {}
        for fold in ((0,) if args.smoke else range(5)):
            train = outer[repeat] != fold
            require(not set(households[train]) & set(households[~train]), "Household crossing")
            microbial = markers(model, raw, taxa, index, pop, train, clr, mapping)["both_pc5"]
            checkpoint = {}
            for period, center_id, matrices in setup:
                training, heldout = train[person], ~train[person]
                if period != "pooled":
                    train_late = period == "late_to_early"
                    training, heldout = training & (late == train_late), heldout & (late != train_late)
                design = terms(doses["carbs"], person, center_id, host, microbial, model)
                for endpoint in ENDPOINTS:
                    values = fold_prediction(matrices[endpoint, "carbs"], design, training,
                                             heldout, family, len(family_ids), model)
                    for key, val in values.items():
                        checkpoint[f"{period}__{endpoint}__{key}"] = val
            for key, val in checkpoint.items():
                totals.setdefault(key, np.zeros_like(val))
                totals[key] += val
            buffer = BytesIO()
            np.savez_compressed(buffer, **checkpoint)
            part = out / f"r{repeat}_f{fold}.npz"
            part.write_bytes(buffer.getvalue())
            io.write(part.with_suffix(".json"), dict(registration=fingerprint, repeat=repeat, fold=fold,
                     held=int((~train).sum()), losses=digest(part)))
            io.emit("HOST_MOD_FOLD", repeat=repeat, fold=fold)
        boot = np.random.default_rng(SEED + repeat).integers(0, len(family_ids), (BOOTSTRAPS, len(family_ids)), dtype=np.int32)
        predictions[f"r{repeat}"] = aggregate(totals, boot)
    direct = {}
    if not args.smoke:
        microbial = markers(model, raw, taxa, index, pop, np.ones(len(pop), bool), clr, mapping)["both_pc5"]
        design = terms(doses["carbs"], person, person, host, microbial, model)
        direct = direct_tests(pooled, design, households, person, microbial, names, model)
    require(all(digest(Path(p)) == sha for p, sha in sources.items()), "Analysis source changed")
    io.write(out / "summary.json", dict(flow=flow, prediction=predictions, direct=direct, clinical_audit=audit))
    io.write(out / "COMPLETE.json", dict(registration=fingerprint, summary=digest(out / "summary.json")))
    io.emit("HOST_MOD_COMPLETE", panel=args.panel, smoke=args.smoke, tests=len(direct))


if __name__ == "__main__":
    main()
