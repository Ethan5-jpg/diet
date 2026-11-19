"""Evaluate protected-anchor workflows on the registered discovery folds."""

import argparse
from collections import defaultdict
from hashlib import sha256
from importlib.metadata import PackageNotFoundError
from pathlib import Path
import json
import re
import sys


HERE = Path(__file__).resolve().parent
V2 = HERE.parent / "Full_pheno_v2"
V1 = HERE.parent / "v1"
sys.path[1:1] = [str(V2), str(V1)]

import v3_design as d
import run as v1_run
import run_v2
from run_v2 import digest, read_json, require, write_csv


FILES = ("README.md", "preprocessing_v1_20261030.py", "run_v3.py", "server_run.sh",
         "v3_design.py", "v3_inputs.py", "v3_models.py", "v3_blend.py", "v3_engine.py",
         "v3_score.py", "v3_report.py", "test_v3.py")
TABLES = ("workflow_scores.csv", "workflow_comparisons.csv", "library_screen.csv",
          "fold_choices.csv", "diet_overlap.csv")


def emit(name, payload):
    print("FULL_PHENO_V3_" + name, json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)


def verify_release():
    release = read_json(HERE / "upload_manifest.json")
    require(release["version"] == d.VERSION and set(release["files"]) == set(FILES), "V3_MANIFEST")
    for name, expected in release["files"].items():
        require(digest(HERE / name) == expected, "V3_UPLOAD_HASH " + name)
    require(digest(V2 / "upload_manifest.json") == d.V2_MANIFEST_SHA, "V3_V2_MANIFEST_HASH")
    release_v2, release_v1 = run_v2.verify_release()
    return release, release_v2, release_v1


def v2_references(path, release, snapshot, v1_ref):
    import protocol as p

    seal_bytes = (path / "COMPLETE.json").read_bytes()
    seal = sha256(seal_bytes).hexdigest()
    summary = run_v2.load_completed(path, release)
    reg_bytes = (path / "registration.json").read_bytes()
    reg_hash = sha256(reg_bytes).hexdigest()
    require(reg_hash == json.loads(seal_bytes)["registration_sha256"], "V3_V2_REGISTRATION_HASH")
    registration = json.loads(reg_bytes)
    require(registration["snapshot_sha"] == snapshot.manifest_sha and registration["sources"] == snapshot.report(),
            "V3_V2_DISCOVERY_SNAPSHOT_DIFF")
    require(registration["v1_receipt"] == v1_ref, "V3_V2_HISTORICAL_V1_RECEIPT_DIFF")
    require(summary["scope"] == "discovery only" and summary["confirmation_outcomes_read"] is False,
            "V3_V2_NOT_DISCOVERY_ONLY")
    refs = defaultdict(dict)
    for row in summary["details"]:
        key = row["exposure"], row["endpoint"]
        require(row["recipe"] not in refs[key], "V3_DUPLICATE_V2_RECIPE")
        refs[key][row["recipe"]] = row
    require(set(refs) == set(p.MODES["all"]) and all(set(x) == set(d.V2_NAMES) for x in refs.values()),
            "V3_INCOMPLETE_V2_GRID")
    require(digest(path / "COMPLETE.json") == seal and digest(path / "registration.json") == reg_hash,
            "V3_V2_RECEIPT_CHANGED")
    return refs, seal


def execute(study, refs_v1, refs_v2, draws, progress=None):
    import numpy as np
    from v3_engine import fit_all
    from v3_report import summarize
    from v3_score import collect, diet_overlap, inference

    groups, codes = np.unique(study.family, return_inverse=True)
    parts = fit_all(study, progress=progress)
    tasks = [collect(pair, folds, study, codes, len(groups), refs_v1[pair], refs_v2[pair])
             for pair, folds in parts.items()]
    replay_error = float(max(t.v2_replay_error for t in tasks))
    if progress:
        progress("REPLAY_OK", dict(pairs=len(tasks), recipes=len(d.V2_NAMES), max_abs_r2=replay_error))
    summary = inference(tasks, draws=draws)
    summary.update(version=d.VERSION, scope="discovery only", confirmation_outcomes_read=False,
                   v2_replay_max_abs_r2=replay_error, library_settings=len(d.LIBRARY), primary=d.PRIMARY)
    library = summarize(summary, tasks)
    overlap = diet_overlap(parts, study)
    summary["diet_overlap"] = overlap
    choices = []
    for (exposure, endpoint), folds in parts.items():
        for fold, result in enumerate(folds):
            for workflow, weights in result.weights.items():
                choices.append(dict(exposure=exposure, endpoint=endpoint, fold=fold, workflow=workflow,
                                    nonzero_settings=len(weights), interaction_weight=sum(weights.values()),
                                    weights=json.dumps(weights, sort_keys=True)))
    return summary, (summary["results"], summary["comparisons"], library, choices, overlap)


def read_complete(path, release):
    require((path / "COMPLETE.json").is_file(), "V3_RUN_NOT_COMPLETE")
    receipt = read_json(path / "COMPLETE.json")
    require(receipt["version"] == d.VERSION and receipt["release"] == release, "V3_RUN_RELEASE_DIFFERS")
    require(digest(path / "registration.json") == receipt["registration_sha256"], "V3_RESULT_REGISTRATION_HASH")
    for name in TABLES:
        require(digest(path / name) == receipt["artifacts"][name], "V3_RESULT_HASH " + name)
    summary = (path / "summary.json").read_bytes()
    require(sha256(summary).hexdigest() == receipt["artifacts"]["summary.json"], "V3_RESULT_HASH summary.json")
    return json.loads(summary)


def write_complete(out, reg_path, release, summary, tables):
    for name, rows in zip(TABLES, tables, strict=True):
        write_csv(v1_run.atomic_write, out / name, rows)
    v1_run.atomic_write(out / "summary.json", v1_run.encode(summary))
    stamp = dict(version=d.VERSION, release=release, registration_sha256=digest(reg_path),
                 artifacts={name: digest(out / name) for name in (*TABLES, "summary.json")})
    v1_run.atomic_write(out / "COMPLETE.json", v1_run.encode(stamp))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--report", action="store_true")
    cli.add_argument("--threads", type=int, default=2)
    cli.add_argument("--run-id", default="discovery-1", type=run_v2.directory_id)
    cli.add_argument("--v1-run-id", default="discovery-1", type=run_v2.directory_id)
    cli.add_argument("--v2-run-id", default="discovery-1", type=run_v2.directory_id)
    cli.add_argument("--handoff", default=HERE.parents[2], type=Path)
    args = cli.parse_args()
    require(1 <= args.threads <= 16, "V3_THREADS")
    release, release_v2, release_v1 = verify_release()
    runtime = v1_run.init_runtime(args.threads)
    emit("RELEASE_OK", dict(version=d.VERSION, library=len(d.LIBRARY), workflows=len(d.WORKFLOWS)))
    if args.preflight:
        import unittest
        import test_v3

        tests = unittest.defaultTestLoader.loadTestsFromModule(test_v3)
        result = unittest.TextTestRunner(verbosity=1).run(tests)
        if result.wasSuccessful():
            emit("PREFLIGHT_OK", dict(tests=result.testsRun, scope="generated data"))
            return 0
        return 1
    from v3_report import render

    out = HERE / "runs/TRE_ONLY" / args.run_id
    require(out.resolve().is_relative_to(HERE), "V3_OUTPUT_OUTSIDE_VERSION")
    if args.report:
        print(render(read_complete(out, release)), end="", flush=True)
        return 0

    import protocol as p
    from source import Snapshot, Study, inventory, required_files

    root = args.handoff.resolve(strict=True)
    require(root.name == p.HANDOFF, "EXPECTED_REGISTERED_COLLABORATOR_HANDOFF_DIRECTORY")
    scope = p.MODES["all"]
    snapshot = Snapshot(root, required_files(scope))
    refs_v1, v1_hashes = run_v2.v1_records(V1 / "runs/TRE_ONLY" / args.v1_run_id, snapshot, release_v1)
    refs_v2, v2_seal = v2_references(V2 / "runs/TRE_ONLY" / args.v2_run_id, release_v2, snapshot, v1_hashes)
    emit("INVENTORY", inventory(snapshot, scope))
    registration = dict(version=d.VERSION, release=release, runtime=runtime, scope="discovery only",
                        snapshot_sha=snapshot.manifest_sha, sources=snapshot.report(),
                        v1_receipt=v1_hashes, v2_complete_sha=v2_seal)
    with v1_run.run_lock(out):
        reg_path = out / "registration.json"
        if reg_path.exists():
            require(read_json(reg_path) == registration, "V3_RUN_REGISTRATION_DIFFERS")
        else:
            require(not (out / "COMPLETE.json").exists(), "V3_RUN_REGISTRATION_MISSING")
            v1_run.atomic_write(reg_path, v1_run.encode(registration))
        if (out / "COMPLETE.json").exists():
            summary = read_complete(out, release)
        else:
            study = Study(snapshot, with_upf=True)
            summary, tables = execute(study, refs_v1, refs_v2, p.BOOTSTRAP_DRAWS, progress=emit)
            require(set(snapshot.paths) <= snapshot.verified, "V3_UNVERIFIED_INPUT")
            snapshot.verify_manifest()
            write_complete(out, reg_path, release, summary, tables)
    emit("COMPLETE", dict(pairs=len(scope), workflows=len(d.WORKFLOWS), reserved_outcomes_read=False))
    print(render(summary), end="", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, TypeError, KeyError, OSError, ArithmeticError, PackageNotFoundError) as exc:
        msg = str(exc)
        if isinstance(exc, FileNotFoundError) and exc.filename:
            msg = "FILE_NOT_FOUND " + Path(exc.filename).name
        elif not re.fullmatch(r"[A-Z][A-Za-z0-9_.; =@/-]*", msg):
            msg = type(exc).__name__
        emit("STOP", dict(reason=msg))
        raise SystemExit(1) from None
