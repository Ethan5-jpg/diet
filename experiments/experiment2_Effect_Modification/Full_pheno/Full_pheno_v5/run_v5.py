"""Run the discovery-only Full Pheno v5 split and three-arm comparison."""

import argparse
from hashlib import sha256
from importlib.metadata import PackageNotFoundError
from pathlib import Path
import json
import re
import sys


HERE = Path(__file__).resolve().parent
V4 = HERE.parent / "Full_pheno_v4"
sys.path.insert(1, str(V4))

import run_v4 as previous
import v5_design as d
from run_v2 import Stop, digest, directory_id, read_json, require, write_csv
from run_v3 import v1_run


FILES = ("README.md", "run_v5.py", "server_run.sh", "test_v5.py", "v5_design.py",
         "v5_models.py", "v5_select.py", "v5_engine.py", "v5_score.py", "v5_report.py")
TABLES = ("workflow_scores.csv", "workflow_comparisons.csv", "library_screen.csv", "fold_choices.csv")


def emit(event, values):
    print("FULL_PHENO_V5_"+event, json.dumps(values, ensure_ascii=False, allow_nan=False), flush=True)


def verify_release():
    release = read_json(HERE / "upload_manifest.json")
    require(release["version"] == d.VERSION and set(release["files"]) == set(FILES), "V5_UPLOAD_MANIFEST")
    for filename, expected in release["files"].items():
        require(digest(HERE / filename) == expected, "V5_UPLOAD_HASH " + filename)
    require(digest(V4 / "upload_manifest.json") == d.V4_MANIFEST_SHA, "V5_V4_MANIFEST_HASH")
    prior, _ = previous.verify_release()
    return release, prior


def references(path, release, snapshot):
    import protocol as p

    receipt = (path / "COMPLETE.json").read_bytes()
    seal = sha256(receipt).hexdigest()
    result = previous.read_complete(path, release)
    recorded = (path / "registration.json").read_bytes()
    require(sha256(recorded).hexdigest() == json.loads(receipt)["registration_sha256"], "V5_V4_REGISTRATION_HASH")
    reg = json.loads(recorded)
    require(reg["snapshot_sha"] == snapshot.manifest_sha and reg["sources"] == snapshot.report(),
            "V5_V4_DISCOVERY_SNAPSHOT_DIFF")
    require(result["scope"] == "discovery only" and result["confirmation_outcomes_read"] is False, "V5_V4_SCOPE_DIFF")
    names = set(d.REPLAY.values())
    refs = {}
    for row in result["rows"]:
        if row["workflow"] in names:
            key = ((row["exposure"], row["endpoint"]), row["workflow"])
            require(key not in refs, "V5_DUPLICATE_V4_REFERENCE")
            refs[key] = row
    require(set(refs) == {(pair, name) for pair in p.MODES["all"] for name in names}, "V5_INCOMPLETE_V4_REFERENCE")
    require(digest(path / "COMPLETE.json") == seal, "V5_V4_RECEIPT_CHANGED")
    return refs, seal


def execute(study, refs, draws, progress=None):
    import numpy as np
    from v5_engine import fit_all
    from v5_report import summarize
    from v5_score import collect, inference

    families, codes = np.unique(study.family, return_inverse=True)
    parts = fit_all(study, progress=progress)
    tasks = [collect(key, folds, study, codes, len(families), refs) for key, folds in parts.items()]
    gap = max(t.replay_error for t in tasks)
    if progress:
        progress("V4_REPLAY_OK", dict(pairs=len(tasks), workflows=len(d.REPLAY), max_abs_r2=gap))
    summary = inference(tasks, draws)
    summary.update(version=d.VERSION, scope="discovery only", confirmation_outcomes_read=False,
                   v4_replay_max_abs_r2=gap, primary=d.PRIMARY, library_settings=len(d.LIBRARY))
    library = summarize(summary, tasks)
    choices = [dict(exposure=key[0], endpoint=key[1], fold=i, workflow=method,
                    interaction_weight=sum(weights.values()), weights=json.dumps(weights, sort_keys=True))
               for key, folds in parts.items() for i, f in enumerate(folds) for method, weights in f.weights.items()]
    return summary, (summary["rows"], summary["comparisons"], library, choices)


def read_complete(out, release):
    require((out / "COMPLETE.json").is_file(), "V5_RUN_NOT_COMPLETE")
    stamp = read_json(out / "COMPLETE.json")
    require(stamp["version"] == d.VERSION and stamp["release"] == release, "V5_RUN_RELEASE_DIFFERS")
    require(digest(out / "registration.json") == stamp["registration_sha256"], "V5_RESULT_REGISTRATION_HASH")
    for name in TABLES:
        require(digest(out / name) == stamp["artifacts"][name], "V5_RESULT_HASH " + name)
    payload = (out / "summary.json").read_bytes()
    require(sha256(payload).hexdigest() == stamp["artifacts"]["summary.json"], "V5_RESULT_HASH summary.json")
    return json.loads(payload)


def write_complete(out, registration, release, summary, tables):
    for name, rows in zip(TABLES, tables, strict=True):
        write_csv(v1_run.atomic_write, out / name, rows)
    v1_run.atomic_write(out / "summary.json", v1_run.encode(summary))
    stamp = dict(version=d.VERSION, release=release, registration_sha256=digest(registration),
                 artifacts={name: digest(out / name) for name in (*TABLES, "summary.json")})
    v1_run.atomic_write(out / "COMPLETE.json", v1_run.encode(stamp))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--report", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--run-id", type=directory_id, default="discovery-1")
    parser.add_argument("--v4-run-id", type=directory_id, default="discovery-1")
    parser.add_argument("--handoff", type=Path, default=HERE.parents[2])
    args = parser.parse_args()
    require(1 <= args.threads <= 16, "V5_THREADS")
    release, prior_release = verify_release()
    runtime = v1_run.init_runtime(args.threads)
    emit("RELEASE_OK", dict(version=d.VERSION, workflows=len(d.WORKFLOWS), library=len(d.LIBRARY)))
    if args.preflight:
        import unittest
        import test_v5

        tests = unittest.defaultTestLoader.loadTestsFromModule(test_v5)
        outcome = unittest.TextTestRunner(verbosity=1).run(tests)
        if outcome.wasSuccessful():
            emit("PREFLIGHT_OK", dict(tests=outcome.testsRun, scope="generated data"))
            return 0
        return 1
    from v5_report import render

    out = HERE / "runs/TRE_ONLY" / args.run_id
    require(out.resolve().is_relative_to(HERE), "V5_OUTPUT_OUTSIDE_VERSION")
    if args.report:
        print(render(read_complete(out, release)), end="", flush=True)
        return 0

    import protocol as p
    from source import Snapshot, Study, inventory, required_files

    root = args.handoff.resolve(strict=True)
    require(root.name == p.HANDOFF, "EXPECTED_REGISTERED_COLLABORATOR_HANDOFF_DIRECTORY")
    scope = p.MODES["all"]
    snapshot = Snapshot(root, required_files(scope))
    refs, seal = references(V4 / "runs/TRE_ONLY" / args.v4_run_id, prior_release, snapshot)
    emit("INVENTORY", inventory(snapshot, scope))
    registration = dict(version=d.VERSION, release=release, runtime=runtime, scope="discovery only",
                        snapshot_sha=snapshot.manifest_sha, sources=snapshot.report(), v4_complete_sha=seal)
    with v1_run.run_lock(out):
        path = out / "registration.json"
        if path.exists():
            require(read_json(path) == registration, "V5_RUN_REGISTRATION_DIFFERS")
        else:
            require(not (out / "COMPLETE.json").exists(), "V5_RUN_REGISTRATION_MISSING")
            v1_run.atomic_write(path, v1_run.encode(registration))
        if (out / "COMPLETE.json").exists():
            summary = read_complete(out, release)
        else:
            study = Study(snapshot, with_upf=True)
            summary, tables = execute(study, refs, p.BOOTSTRAP_DRAWS, progress=emit)
            require(set(snapshot.paths) <= snapshot.verified, "V5_UNVERIFIED_INPUT")
            snapshot.verify_manifest()
            write_complete(out, path, release, summary, tables)
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
        elif not isinstance(exc, Stop) and not re.fullmatch(r"[A-Z][A-Za-z0-9_.; =@/-]*", msg):
            msg = type(exc).__name__
        emit("STOP", dict(reason=msg))
        raise SystemExit(1) from None
