"""Replay archived three-arm scores and test lower-variance diet main effects."""

import argparse
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
V5 = HERE.parent / "Full_pheno_v5"
sys.path.insert(1, str(V5))

import run_v5 as archive
from run_v2 import directory_id, require
from run_v3 import v1_run


ANCHOR = ("fiber_density", "D_Glu")


def emit(event, fields):
    print("FULL_PHENO_V6_" + event, json.dumps(fields, allow_nan=False), flush=True)


def evaluate(study, cached, draws):
    import numpy as np
    import protocol as p
    from breadth import basis
    from diet_audit import METRICS, RECIPES, fit_fold, household_moments, simultaneous

    tasks = p.MODES["all"]
    _, groups = np.unique(study.family, return_inverse=True)
    count = int(groups.max()) + 1
    outcomes = {endpoint: study.outcome(endpoint) for endpoint in p.ENDPOINTS}
    fits = {task: [] for task in tasks}
    for fold in p.FOLDS:
        for exposure in p.EXPOSURES:
            shared = basis(study, exposure, fold)
            for endpoint, outcome in outcomes.items():
                fits[exposure, endpoint].append(fit_fold(study, exposure, endpoint, outcome, fold, shared))
        emit("FOLD_DONE", {"fold": fold, "pairs": len(tasks)})

    moments = np.stack([household_moments(fits[key], groups, count) for key in tasks])
    r2, points, radius = simultaneous(moments, draws)
    rows, replay_errors = [], []
    for task, (exposure, endpoint) in enumerate(tasks):
        prior = cached[exposure, endpoint]
        ref = len(RECIPES) - 1 if (exposure, endpoint) == ANCHOR else 0
        members = moments[task, :, 0] > 0
        people, families = int(moments[task, :, 0].sum()), int(members.sum())
        require(people == prior["people"] and families == prior["households"], "V6_ARCHIVE_COUNTS")
        legacy = [prior[x] for x in ("r2_no_diet", "r2_off", "r2_on")]
        replay_errors.append(float(np.max(abs(r2[task, ref] - legacy))))
        for j, recipe in enumerate(RECIPES):
            a, b, c = (float(v) for v in r2[task, j])
            out = dict(exposure=exposure, endpoint=endpoint, recipe=recipe, people=people, households=families,
                       r2_no_diet=a, r2_off=b, r2_on=c)
            for k, label in enumerate(METRICS):
                center, width = float(points[task, j, k]), float(radius[task, j, k])
                out[label] = center
                out[label + "_95"] = [center - width, center + width]
            change = moments[task, members, 4 + j] - moments[task, members, 4 + len(RECIPES) + j]
            retain = np.argsort(abs(change), kind="stable")[:-max(1, int(np.ceil(0.01 * families)))]
            stripped = moments[task, members][retain].sum(axis=0)
            sst = stripped[2] - stripped[1]**2 / stripped[0]
            out["interaction_pp_excluding_top_1pct_families"] = float(100. * change[retain].sum() / sst)
            out["positive_interaction_folds"] = sum(
                int(np.sum((fit.outcome - fit.off[:, j])**2 - (fit.outcome - fit.on[:, j])**2) > 0)
                for fit in fits[exposure, endpoint]
            )
            rows.append(out)
    error = max(replay_errors)
    require(error <= 2e-8, "V6_ARCHIVE_REPLAY_MISMATCH")
    return dict(rows=rows, replay_max_r2_error=error, bootstrap_draws=draws,
                scope="discovery only", reserved_outcomes_read=False)


def report(result):
    rows = result["rows"]
    for recipe in dict.fromkeys(row["recipe"] for row in rows):
        group = [row for row in rows if row["recipe"] == recipe]
        rest = [row for row in group if (row["exposure"], row["endpoint"]) != ANCHOR]
        delta = [row["interaction_pp"] for row in rest]
        emit("RECIPE", dict(recipe=recipe, main_negative=sum(x["diet_main_pp"] < 0 for x in group),
                            total_negative=sum(x["diet_total_pp"] < 0 for x in group),
                            positive17=sum(x > 0 for x in delta), ge_010=sum(x >= .1 for x in delta),
                            le_neg_010=sum(x <= -.1 for x in delta),
                            adjusted_positive17=sum(x["interaction_pp_95"][0] > 0 for x in rest)))
    targets = {ANCHOR, ("UPF", "D_TotalFat"), ("UPF", "D_Liver_viscosity")}
    for row in rows:
        if (row["exposure"], row["endpoint"]) in targets:
            print(" | ".join((row["exposure"], row["endpoint"], row["recipe"],
                             *(f"{row[k]:+.4f}" for k in ("diet_main_pp", "interaction_pp", "diet_total_pp")),
                             f"{row['positive_interaction_folds']}/5")), flush=True)
    emit("COMPLETE", dict(pairs=len({(x["exposure"], x["endpoint"]) for x in rows}),
                         replay_max_r2_error=result["replay_max_r2_error"], reserved_outcomes_read=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, default=HERE.parents[2])
    parser.add_argument("--run-id", type=directory_id, default="diet-main-1")
    parser.add_argument("--commit", required=False, default="synthetic-or-development")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    result_dir = HERE / "runs/TRE_ONLY" / args.run_id
    result_path = result_dir / "diet_audit.json"
    if args.report:
        report(v1_run.read_json(result_path))
        return

    runtime = v1_run.init_runtime(2)
    import protocol as p
    from source import Snapshot, Study, required_files

    root = args.handoff.resolve(strict=True)
    require(root.name == p.HANDOFF, "V6_HANDOFF_ROOT")
    snapshot = Snapshot(root, required_files(p.MODES["all"]))
    release, _ = archive.verify_release()
    old = V5 / "runs/TRE_ONLY/discovery-1"
    receipt = archive.read_complete(old, release)
    registration = v1_run.read_json(old / "registration.json")
    require(registration["snapshot_sha"] == snapshot.manifest_sha and registration["sources"] == snapshot.report(),
            "V6_ARCHIVE_INPUTS_MISMATCH")
    rows = receipt["rows"]
    previous = {(x["exposure"], x["endpoint"]): x for x in rows if x["workflow"] == "v4_pc5_half"}
    require(set(previous) == set(p.MODES["all"]), "V6_ARCHIVE_REFERENCE_MISSING")
    identity = dict(commit=args.commit, snapshot=snapshot.manifest_sha, runtime=runtime,
                    v5_complete=v1_run.file_sha(old / "COMPLETE.json"),
                    code={path.name: v1_run.file_sha(path) for path in sorted(HERE.glob("*.py"))})
    with v1_run.run_lock(result_dir):
        meta = result_dir / "registration.json"
        if meta.exists():
            require(v1_run.read_json(meta) == identity, "V6_RUN_IDENTITY_CHANGED")
        else:
            v1_run.atomic_write(meta, v1_run.encode(identity))
        if result_path.exists():
            result = v1_run.read_json(result_path)
        else:
            emit("START", {"run_id": args.run_id, "pairs": len(previous), "scope": "discovery only"})
            result = evaluate(Study(snapshot, with_upf=True), previous, p.BOOTSTRAP_DRAWS)
            require(set(snapshot.paths) <= snapshot.verified, "V6_UNVERIFIED_INPUTS")
            snapshot.verify_manifest()
            v1_run.atomic_write(result_path, v1_run.encode(result))
    report(result)


if __name__ == "__main__":
    main()
