"""Run the discovery-only three-arm Full Pheno screen."""

import argparse
from hashlib import sha256
from importlib.metadata import PackageNotFoundError
from pathlib import Path
import json
import re
import sys


HERE = Path(__file__).resolve().parent
V3 = HERE.parent / "Full_pheno_v3"
sys.path.insert(1, str(V3))

import run_v3 as previous
import v4_design as d
from run_v2 import Stop, digest, directory_id, read_json, require, write_csv
from run_v3 import v1_run


FILES = ("README.md", "run_v4.py", "server_run.sh", "test_v4.py", "v4_design.py", "v4_models.py",
         "v4_select.py", "v4_engine.py", "v4_score.py", "v4_report.py")
TABLES = ("workflow_scores.csv", "workflow_comparisons.csv", "library_screen.csv", "fold_choices.csv")


def emit(name, payload):
    print("FULL_PHENO_V4_" + name, json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)


def verify_release():
    release = read_json(HERE / "upload_manifest.json")
    require(release["version"] == d.VERSION and set(release["files"]) == set(FILES), "V4_UPLOAD_MANIFEST")
    for name, expected in release["files"].items():
        require(digest(HERE / name) == expected, "V4_UPLOAD_HASH " + name)
    require(digest(V3 / "upload_manifest.json") == d.V3_MANIFEST_SHA, "V4_V3_MANIFEST_HASH")
    prior, _, _ = previous.verify_release()
    return release, prior


def reference(path, release, snapshot):
    import protocol as p

    receipt = (path / "COMPLETE.json").read_bytes()
    seal = sha256(receipt).hexdigest()
    aggregate = previous.read_complete(path, release)
    data = (path / "registration.json").read_bytes()
    require(sha256(data).hexdigest() == json.loads(receipt)["registration_sha256"], "V4_V3_REGISTRATION_HASH")
    reg = json.loads(data)
    require(reg["snapshot_sha"] == snapshot.manifest_sha and reg["sources"] == snapshot.report(),
            "V4_V3_DISCOVERY_SNAPSHOT_DIFF")
    require(aggregate["scope"] == "discovery only" and aggregate["confirmation_outcomes_read"] is False,
            "V4_V3_SCOPE_DIFF")
    refs = {}
    for row in aggregate["results"]:
        if row["workflow"] in ("fixed_pc5", "blend_guarded"):
            key = ((row["exposure"], row["endpoint"]), row["workflow"])
            require(key not in refs, "V4_DUPLICATE_V3_REFERENCE")
            refs[key] = row
    require(set(refs) == {(key, name) for key in p.MODES["all"] for name in ("fixed_pc5", "blend_guarded")},
            "V4_INCOMPLETE_V3_REFERENCE")
    require(digest(path / "COMPLETE.json") == seal, "V4_V3_RECEIPT_CHANGED")
    return refs, seal


def execute(study, refs, draws, progress=None):
    import numpy as np
    from v4_engine import fit_all
    from v4_report import summarize
    from v4_score import collect, inference

    families, codes = np.unique(study.family, return_inverse=True)
    scores = fit_all(study, progress=progress)
    tasks = [collect(key, folds, study, codes, len(families), refs) for key, folds in scores.items()]
    gap = max(t.replay_error for t in tasks)
    if progress:
        progress("V3_REPLAY_OK", dict(pairs=len(tasks), workflows=2, max_abs_r2=gap))
    summary = inference(tasks, draws)
    summary.update(version=d.VERSION, scope="discovery only", confirmation_outcomes_read=False,
                   v3_replay_max_abs_r2=gap, primary=d.PRIMARY, library_settings=len(d.LIBRARY))
    library = summarize(summary, tasks)
    choices = [dict(exposure=key[0], endpoint=key[1], fold=i, workflow=method, interaction_weight=sum(weights.values()),
                    weights=json.dumps(weights, sort_keys=True))
               for key, folds in scores.items() for i, f in enumerate(folds) for method, weights in f.weights.items()]
    return summary, (summary["rows"], summary["comparisons"], library, choices)


def read_complete(out, release):
    require((out / "COMPLETE.json").is_file(), "V4_RUN_NOT_COMPLETE")
    receipt = read_json(out / "COMPLETE.json")
    require(receipt["version"] == d.VERSION and receipt["release"] == release, "V4_RUN_RELEASE_DIFFERS")
    require(digest(out / "registration.json") == receipt["registration_sha256"], "V4_RESULT_REGISTRATION_HASH")
    for name in TABLES:
        require(digest(out / name) == receipt["artifacts"][name], "V4_RESULT_HASH " + name)
    value = (out / "summary.json").read_bytes()
    require(sha256(value).hexdigest() == receipt["artifacts"]["summary.json"], "V4_RESULT_HASH summary.json")
    return json.loads(value)


def write_complete(out, registration, release, summary, tables):
    for name, rows in zip(TABLES, tables, strict=True):
        write_csv(v1_run.atomic_write, out / name, rows)
    v1_run.atomic_write(out / "summary.json", v1_run.encode(summary))
    receipt = dict(version=d.VERSION, release=release, registration_sha256=digest(registration),
                   artifacts={name: digest(out / name) for name in (*TABLES, "summary.json")})
    v1_run.atomic_write(out / "COMPLETE.json", v1_run.encode(receipt))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--report", action="store_true")
    cli.add_argument("--threads", type=int, default=2)
    cli.add_argument("--run-id", type=directory_id, default="discovery-1")
    cli.add_argument("--v3-run-id", type=directory_id, default="discovery-1")
    cli.add_argument("--handoff", type=Path, default=HERE.parents[2])
    args = cli.parse_args()
    require(1 <= args.threads <= 16, "V4_THREADS")
    release, old_release = verify_release()
    runtime = v1_run.init_runtime(args.threads)
    emit("RELEASE_OK", dict(version=d.VERSION, workflows=len(d.WORKFLOWS), library=len(d.LIBRARY)))
    if args.preflight:
        import test_v4
        import unittest

        tests = unittest.defaultTestLoader.loadTestsFromModule(test_v4)
        result = unittest.TextTestRunner(verbosity=1).run(tests)
        if result.wasSuccessful():
            emit("PREFLIGHT_OK", dict(tests=result.testsRun, scope="generated data"))
            return 0
        return 1
    from v4_report import render

    out = HERE / "runs/TRE_ONLY" / args.run_id
    require(out.resolve().is_relative_to(HERE), "V4_OUTPUT_OUTSIDE_VERSION")
    if args.report:
        print(render(read_complete(out, release)), end="", flush=True)
        return 0

    import protocol as p
    from source import Snapshot, Study, inventory, required_files

    root = args.handoff.resolve(strict=True)
    require(root.name == p.HANDOFF, "EXPECTED_REGISTERED_COLLABORATOR_HANDOFF_DIRECTORY")
    scope = p.MODES["all"]
    snapshot = Snapshot(root, required_files(scope))
    refs, prior = reference(V3 / "runs/TRE_ONLY" / args.v3_run_id, old_release, snapshot)
    emit("INVENTORY", inventory(snapshot, scope))
    registration = dict(version=d.VERSION, release=release, runtime=runtime, scope="discovery only",
                        snapshot_sha=snapshot.manifest_sha, sources=snapshot.report(), v3_complete_sha=prior)
    with v1_run.run_lock(out):
        path = out / "registration.json"
        if path.exists():
            require(read_json(path) == registration, "V4_RUN_REGISTRATION_DIFFERS")
        else:
            require(not (out / "COMPLETE.json").exists(), "V4_RUN_REGISTRATION_MISSING")
            v1_run.atomic_write(path, v1_run.encode(registration))
        if (out / "COMPLETE.json").exists():
            summary = read_complete(out, release)
        else:
            study = Study(snapshot, with_upf=True)
            summary, tables = execute(study, refs, p.BOOTSTRAP_DRAWS, progress=emit)
            require(set(snapshot.paths) <= snapshot.verified, "V4_UNVERIFIED_INPUT")
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
