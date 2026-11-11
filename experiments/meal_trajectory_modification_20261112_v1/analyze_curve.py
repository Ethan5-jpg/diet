"""Separate level and time-shape components of microbial meal modification."""

import argparse
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parent
MEAL = ROOT.parent / "meal_carbohydrate_modification_20261112_v1"
PINS = {
    MEAL / "paired.py": "6495e7bf2fe831a971be31a1dbc1ddaf582773c13a04d89dbf7641a478239b0a",
    MEAL / "time_transfer.py": "c9295f753b4f74e9f6cade505bd35c248e6c7fdbf89a6d16263148411b0b20a8",
    ROOT / "prepare_curve.py": "a0974b66fe06c94bc8bd1572f5c07e1d178618ef7edb3834c3d425304fa98f8c",
}
sys.path.insert(0, str(MEAL))
try:
    from analyze import bases, markers
    from paired import archived_inputs
    from prepare import digest, load, require
    from time_transfer import assign_periods
finally:
    sys.path.pop(0)

import numpy as np
import pyarrow.parquet as pq

from curve_models import MODES, clustered_test, household_losses, summarize


TAGS = ("p0", "m15", "p15")
SEED = 202611129
BOOTSTRAPS = 1999


@dataclass
class Scope:
    person: np.ndarray
    late: np.ndarray
    family: np.ndarray
    groups: int
    data: dict
    dose: np.ndarray


def prepare_scope(frame, households, model, *, separate_periods=False):
    person = frame.person_index.to_numpy(int)
    late = frame.late.to_numpy(bool)
    groups = 2 * person + late if separate_periods else person
    matrices, doses = bases(frame.assign(person_index=groups), model)
    dose = model.center(doses["carbs"][:, None], groups)[0][:, 0]
    _, family = np.unique(households[person], return_inverse=True)
    data = {}
    for tag in TAGS:
        y, x, weight = matrices[f"net_{tag}", "carbs"]
        phase = frame[[f"mean_{tag}_{i}" for i in range(4)]].to_numpy(float)
        centered = model.center(phase @ MODES, groups)[0]
        require(np.allclose(centered[:, 0] / 2, y, atol=1e-8, rtol=0), "Curve level disagrees with archived net outcome")
        data[tag] = (centered, x, weight)
    return Scope(person, late, family, int(family.max()) + 1, data, dose)


def aggregate(totals, metadata, repeat):
    output = {}
    for name, flow in metadata.items():
        n = flow["households"]
        draws = np.random.default_rng(SEED + repeat).integers(0, n, (BOOTSTRAPS,n), dtype=np.int32)
        output[name] = {}
        for tag in TAGS:
            prefix = f"{name}__{tag}__"
            parts = {key[len(prefix):]: v for key, v in totals.items() if key.startswith(prefix)}
            output[name][tag] = summarize(parts, draws)
    return output


def direct(scope, score, households, model):
    result = {}
    z = scope.dose[:, None] * score[scope.person]
    groups = households[scope.person]
    n = len(np.unique(scope.person))
    for tag, (y, x, w) in scope.data.items():
        level_p, level = clustered_test(y[:, :1], x, z, w, groups, n, model)
        shape_p, shape = clustered_test(y[:, 1:3], x, z, w, groups, n, model)
        beta = np.column_stack((level, shape)) @ MODES[:, :3].T
        result[tag] = dict(level_p=level_p, shape_p=shape_p,
                           smooth_phase_sd=np.std(score @ beta, axis=0, ddof=1).tolist())
    tests = [(row, key) for row in result.values() for key in ("level_p","shape_p")]
    for (row, key), p in zip(tests, model.holm([row[key] for row,key in tests])):
        row[key + "_holm"] = float(p)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--prepared", default="phase-discovery-1")
    parser.add_argument("--run", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    package = args.package.resolve()
    require(package / "experiments" / ROOT.name == ROOT and re.fullmatch(r"[a-zA-Z0-9_-]+", args.run)
            and re.fullmatch(r"[a-zA-Z0-9_-]+", args.prepared)
            and re.fullmatch(r"[0-9a-f]{40}", args.commit), "Unexpected release or run path")
    require(all(digest(p) == sha for p, sha in PINS.items()), "Pinned source differs")
    out = ROOT / "runs_TRE_ONLY" / args.run
    require(not out.exists(), "Use a fresh run identifier")
    archive = MEAL / "runs_TRE_ONLY/models-1"
    receipt, original, io, meal, index, pop, _, _, outer, raw, taxa, clr, mapping = archived_inputs(package, archive)
    model = __import__("models")
    prepared = ROOT / "runs_TRE_ONLY" / args.prepared
    phase = prepared / "phase_windows_TRE_ONLY.parquet"
    files = (prepared / "registration.json", prepared / "summary.json", phase)
    completion = load(prepared / "COMPLETE.json")
    require(not load(files[0])["smoke"] and all(digest(p) == completion[key]
            for p, key in zip(files, ("registration","summary","phases"))), "Prepared phases differ")
    table = pq.read_table(phase, use_threads=False, pre_buffer=False).to_pandas()
    require(len(table) == len(meal), "Prepared curve rows differ from the archived meal model")
    meal = meal.merge(table,on=["input_row","start"],how="left",validate="one_to_one")
    require(np.isfinite(meal[[f"mean_{tag}_{i}" for tag in TAGS for i in range(4)]]).all().all(),
            "Missing prepared curve")
    support, late = assign_periods(meal)
    meal["late"] = late
    households = index.iloc[pop].split_unit_id.astype(str).to_numpy()
    balanced = meal.loc[support]
    chrono = prepare_scope(balanced, households, model, separate_periods=True)
    scopes = dict(pooled=prepare_scope(meal, households, model),
                  balanced_pooled=prepare_scope(balanced, households, model),
                  early_to_late=chrono, late_to_early=chrono)
    metadata = {key: dict(people=len(np.unique(v.person)), households=v.groups, meals=len(v.person))
                for key,v in scopes.items()}
    sources = {**receipt["sources"], **{str(p): sha for p,sha in PINS.items()},
               **{str(p): digest(p) for p in (*files, prepared/"COMPLETE.json", archive/"summary.json",
                                              Path(__file__).resolve(),ROOT/"curve_models.py")}}
    out.mkdir(parents=True,exist_ok=False)
    reg = dict(commit=args.commit, prepared=args.prepared, smoke=args.smoke, sources=sources,
               scopes=metadata, seed=SEED, bootstraps=BOOTSTRAPS)
    io.write(out/"registration.json",reg)
    fingerprint = digest(out/"registration.json")
    io.emit("CURVE_MODEL_START", scopes=metadata, smoke=args.smoke)
    prediction = {}
    for repeat in ((0,) if args.smoke else (0,1)):
        totals = {}
        for fold in ((0,) if args.smoke else range(5)):
            train = outer[repeat] != fold
            require(not set(households[train]) & set(households[~train]), "Household crossing")
            score = markers(model, raw, taxa, index, pop, train, clr, mapping)["both_pc5"]
            checkpoint = {}
            for name,scope in scopes.items():
                take, held = train[scope.person], ~train[scope.person]
                if name in ("early_to_late","late_to_early"):
                    train_late = name == "late_to_early"
                    take, held = take & (scope.late == train_late), held & (scope.late != train_late)
                z = scope.dose[:, None] * score[scope.person]
                for tag,(y,x,w) in scope.data.items():
                    parts = household_losses(y,x,z,w,take,held,scope.family,scope.groups,model)
                    checkpoint.update({f"{name}__{tag}__{k}": v for k,v in parts.items()})
            for k,v in checkpoint.items():
                totals.setdefault(k,np.zeros_like(v))
                totals[k] += v
            buf = BytesIO()
            np.savez_compressed(buf, **checkpoint)
            part = out/f"r{repeat}_f{fold}.npz"
            part.write_bytes(buf.getvalue())
            io.write(part.with_suffix(".json"),dict(registration=fingerprint,repeat=repeat,fold=fold,
                     held=int((~train).sum()),losses=digest(part)))
            io.emit("CURVE_MODEL_FOLD",repeat=repeat,fold=fold)
        rep = f"r{repeat}"
        prediction[rep] = aggregate(totals,metadata,repeat)
        if not args.smoke:
            observed = prediction[rep]["pooled"]["p0"]["level"]["level"]["delta_r2"]
            reference = original["prediction"]["net_p0_carbs_both_pc5"][rep]["delta_r2"]
            require(abs(observed-reference)<1e-10,"Level model failed to reproduce the original fold-out result")
    tests = {}
    if not args.smoke:
        score = markers(model, raw, taxa, index, pop, np.ones(len(pop),bool), clr, mapping)["both_pc5"]
        tests = direct(scopes["pooled"],score,households,model)
    require(all(digest(Path(p)) == v for p,v in sources.items()), "Analysis source changed")
    io.write(out/"summary.json",dict(scopes=metadata,prediction=prediction,direct=tests))
    io.write(out/"COMPLETE.json",dict(registration=fingerprint,summary=digest(out/"summary.json")))
    io.emit("CURVE_MODEL_COMPLETE", smoke=args.smoke)


if __name__ == "__main__":
    main()
