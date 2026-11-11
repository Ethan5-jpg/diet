"""Evaluate microbial modification across two supported recorded meal contexts."""

import argparse
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parent
MEAL = ROOT.parent / "meal_carbohydrate_modification_20261112_v1"
HOST = ROOT.parent / "meal_host_microbiome_modification_20261112_v1"
PINS = {
    MEAL / "paired.py": "6495e7bf2fe831a971be31a1dbc1ddaf582773c13a04d89dbf7641a478239b0a",
    MEAL / "time_transfer.py": "c9295f753b4f74e9f6cade505bd35c248e6c7fdbf89a6d16263148411b0b20a8",
    HOST / "clinical.py": "d4defd07e430768989f17392ade8f04b166ac5f691f87dd07665cd8e3c662784",
}
sys.path[:0] = [str(MEAL), str(HOST)]
try:
    from analyze import bases, markers
    from paired import archived_inputs, family_totals
    from prepare import digest, require
    from time_transfer import assign_periods
    from clinical import RAW, read_measurements, scale
finally:
    del sys.path[:2]

import numpy as np

from context import interaction_design, require_support, select_contexts, split_background


ENDPOINTS = ("net_p0", "peak_p0", "pre_p0")
COMPARISONS = {"shared": ("none", "shared"), "context_specific": ("none", "context"),
               "context_beyond_shared": ("shared", "context")}
SEED = 202611128
BOOTSTRAPS = 1999
CLINICAL = ("hba1c", "bmi")


@dataclass
class Scope:
    person: np.ndarray
    morning: np.ndarray
    late: np.ndarray
    dose: np.ndarray
    matrices: dict
    host: np.ndarray | None
    family: np.ndarray
    groups: int


def build_scope(frame, households, model, *, split=False, host=None):
    person = frame.person_index.to_numpy(int)
    morning = frame.morning.to_numpy(bool)
    late = frame.late.to_numpy(bool)
    groups = 2 * person + morning
    if split:
        groups = 2 * groups + late
    matrices, doses = bases(frame.assign(person_index=groups), model)
    dose = model.center(doses["carbs"][:, None], groups)[0][:, 0]
    _, family = np.unique(households[person], return_inverse=True)
    return Scope(person, morning, late, dose, matrices, host, family, int(family.max()) + 1)


def prepare_scope(meal, households, model, panel, host):
    if panel == "clinical":
        frame = meal.assign(late=False)
        return {"clinical_matched": build_scope(frame, households, model),
                "clinical_adjusted": build_scope(frame, households, model, host=host)}
    support, late = assign_periods(meal)
    frame = meal.assign(late=late)
    balanced = require_support(frame.loc[support], ("morning", "late"), 2, 5, 4)
    require(balanced.person_index.nunique() >= 100, "Insufficient balanced date support")
    chronological = build_scope(balanced, households, model, split=True)
    return {"pooled": build_scope(frame, households, model),
            "balanced_pooled": build_scope(balanced, households, model),
            "early_to_late": chronological, "late_to_early": chronological}


def design(scope, marker):
    host = None if scope.host is None else scope.host[scope.person]
    return interaction_design(scope.dose, marker[scope.person], scope.morning, host)


def flow(scope):
    return dict(people=len(np.unique(scope.person)), households=scope.groups, meals=len(scope.person),
                morning=int(scope.morning.sum()), daytime=int((~scope.morning).sum()))


def scores_for_fold(name, scope, marker, train, model):
    common, difference, clinical = design(scope, marker)
    blocks = {"shared": common, "context": np.column_stack((common, difference))}
    training, valid = train[scope.person], ~train[scope.person]
    if name in ("early_to_late", "late_to_early"):
        train_late = name == "late_to_early"
        training, valid = training & (scope.late == train_late), valid & (scope.late != train_late)
    checkpoint = {}
    for endpoint in ENDPOINTS:
        y, x, weights = scope.matrices[endpoint, "carbs"]
        x = split_background(x, scope.morning)
        if clinical is not None:
            x = np.column_stack((x, clinical))
        for context, m in (("daytime", ~scope.morning), ("morning", scope.morning)):
            losses = family_totals(y, x, weights, blocks, training, scope.family, scope.groups, model,
                                   valid=valid & m)
            checkpoint.update({f"{name}__{endpoint}__{context}__{key}": val for key, val in losses.items()})
    return checkpoint


def summarize(parts, draws):
    tss = parts["tss"]
    denominator = tss[draws].sum(axis=1)
    require(tss.sum() > 0 and (denominator > 0).all(), "Invalid matched score denominator")
    output = {}
    for name, (null, alt) in COMPARISONS.items():
        gain = parts[null] - parts[alt]
        output[name] = dict(delta_r2=float(gain.sum() / tss.sum()),
                            ci95=np.quantile(gain[draws].sum(axis=1) / denominator, [.025, .975]).tolist())
    return output


def aggregate(totals, scopes, repeat):
    output = {}
    keys = ("none", "shared", "context", "tss")
    for name, metadata in scopes.items():
        groups = metadata["households"]
        draws = np.random.default_rng(SEED + repeat).integers(0, groups, (BOOTSTRAPS, groups), dtype=np.int32)
        output[name] = {}
        for endpoint in ENDPOINTS:
            parts = {context: {k: totals[f"{name}__{endpoint}__{context}__{k}"] for k in keys}
                     for context in ("daytime", "morning")}
            parts["joint"] = {k: parts["morning"][k] + parts["daytime"][k] for k in keys}
            output[name][endpoint] = {context: summarize(v, draws) for context, v in parts.items()}
    return output


def direct(scope, marker, households, model):
    common, difference, clinical = design(scope, marker)
    output = {}
    fixed_effects = len(np.unique(2 * scope.person + scope.morning))
    width = marker.shape[1]
    for ep in ENDPOINTS:
        y, x, weights = scope.matrices[ep, "carbs"]
        x = split_background(x, scope.morning)
        if clinical is not None:
            x = np.column_stack((x, clinical))
        b, _, p_all = model.household_test(np.column_stack((common, difference)),
                                            model.fit_background(x, y, weights),
                                            households[scope.person], fixed_effects)
        _, _, p_diff = model.household_test(difference, model.fit_background(np.column_stack((x, common)), y, weights),
                                           households[scope.person], fixed_effects)
        day = marker @ (b[:width] - .5 * b[width:])
        morning = marker @ (b[:width] + .5 * b[width:])
        output[ep] = dict(p_all=p_all, p_difference=p_diff, morning_sd=float(np.std(morning, ddof=1)),
                          daytime_sd=float(np.std(day, ddof=1)), difference_sd=float(np.std(morning - day, ddof=1)))
    tests = [(row, key) for row in output.values() for key in ("p_all", "p_difference")]
    for (row, key), p in zip(tests, model.holm([row[key] for row, key in tests])):
        row[f"{key}_holm"] = float(p)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--panel", choices=("broad", "clinical"), required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    package = args.package.resolve()
    require(package / "experiments" / ROOT.name == ROOT and re.fullmatch(r"[a-zA-Z0-9_-]+", args.run)
            and re.fullmatch(r"[0-9a-f]{40}", args.commit), "Unexpected release or run path")
    require(all(digest(p) == sha for p, sha in PINS.items()), "Pinned source differs")
    out = ROOT / "runs_TRE_ONLY" / args.run
    require(not out.exists(), "Use a fresh run identifier")
    archive = MEAL / "runs_TRE_ONLY/models-1"
    receipt, original, io, meal, index, pop, _, _, outer, raw, taxa, clr, mapping = archived_inputs(package, archive)
    model = __import__("models")
    data = select_contexts(meal)
    assay, assay_sources, audit = None, {}, {}
    if args.panel == "clinical":
        assay, audit, assay_sources = read_measurements(Path(RAW), meal, index, CLINICAL, digest)
        assay = assay.dropna()
        data = data.loc[data.input_row.isin(assay.index)].copy()
    positions, person = np.unique(data.person_index, return_inverse=True)
    pop, outer = pop[positions], outer[:, positions]
    data["person_index"] = person
    households = index.iloc[pop].split_unit_id.astype(str).to_numpy()
    require(len(np.unique(households)) >= 250, "Insufficient paired meal contexts")
    host = None if assay is None else scale(assay.loc[pop, list(CLINICAL)].to_numpy(float), CLINICAL)
    scopes = prepare_scope(data, households, model, args.panel, host)
    metadata = {key: flow(scope) for key, scope in scopes.items()}
    scripts = {str(p): digest(p) for p in (Path(__file__).resolve(), ROOT / "context.py", archive / "summary.json")}
    sources = {**receipt["sources"], **assay_sources, **{str(p): v for p, v in PINS.items()}, **scripts}
    out.mkdir(parents=True, exist_ok=False)
    reg = dict(commit=args.commit, smoke=args.smoke, panel=args.panel, scopes=metadata,
               archived_flow=original["flow"], sources=sources, seed=SEED, bootstraps=BOOTSTRAPS,
               clinical_audit=audit)
    io.write(out / "registration.json", reg)
    fingerprint = digest(out / "registration.json")
    io.emit("CONTEXT_MOD_START", panel=args.panel, scopes=metadata, smoke=args.smoke)
    prediction = {}
    for repeat in ((0,) if args.smoke else (0, 1)):
        totals = {}
        for fold in ((0,) if args.smoke else range(5)):
            train = outer[repeat] != fold
            require(not set(households[train]) & set(households[~train]), "Household crossing")
            marker = markers(model, raw, taxa, index, pop, train, clr, mapping)["both_pc5"]
            checkpoint = {}
            for name, scope in scopes.items():
                checkpoint.update(scores_for_fold(name, scope, marker, train, model))
            for key, values in checkpoint.items():
                totals.setdefault(key, np.zeros_like(values))
                totals[key] += values
            buf = BytesIO()
            np.savez_compressed(buf, **checkpoint)
            part = out / f"r{repeat}_f{fold}.npz"
            part.write_bytes(buf.getvalue())
            io.write(part.with_suffix(".json"), dict(registration=fingerprint, repeat=repeat, fold=fold,
                     held=int((~train).sum()), losses=digest(part)))
            io.emit("CONTEXT_MOD_FOLD", repeat=repeat, fold=fold)
        prediction[f"r{repeat}"] = aggregate(totals, metadata, repeat)
    tests = {}
    if not args.smoke:
        marker = markers(model, raw, taxa, index, pop, np.ones(len(pop), bool), clr, mapping)["both_pc5"]
        primary = "pooled" if args.panel == "broad" else "clinical_adjusted"
        tests = direct(scopes[primary], marker, households, model)
    require(all(digest(Path(p)) == sha for p, sha in sources.items()), "Analysis source changed")
    io.write(out / "summary.json", dict(scopes=metadata, prediction=prediction, direct=tests))
    io.write(out / "COMPLETE.json", dict(registration=fingerprint, summary=digest(out / "summary.json")))
    io.emit("CONTEXT_MOD_COMPLETE", panel=args.panel, smoke=args.smoke)


if __name__ == "__main__":
    main()
