"""Run the cross-phenotype breadth screen on the registered discovery snapshot."""

import argparse
import csv
from hashlib import file_digest, sha256
import io
import json
from pathlib import Path
import re
import sys
import time
from importlib.metadata import PackageNotFoundError

import design as d


HERE = Path(__file__).resolve().parent
V1 = HERE.parent / "v1"
FILES = ("README.md", "design.py", "breadth.py", "run_v2.py", "server_run.sh", "test_v2.py")


def digest(path):
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def emit(name, payload):
    print("FULL_PHENO_V2_" + name, json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)


class Stop(ValueError):
    pass


def require(ok, code):
    if not ok:
        raise Stop(code)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_release():
    check = read_json(HERE / "upload_manifest.json")
    require(check["version"] == d.VERSION and set(check["files"]) == set(FILES), "V2_UPLOAD_MANIFEST")
    for name in FILES:
        require(digest(HERE / name) == check["files"][name], "V2_UPLOAD_HASH " + name)
    manifest = V1 / "upload_manifest.json"
    if not manifest.is_file():
        raise Stop("V1_SIBLING_UPLOAD_MANIFEST_NOT_FOUND")
    require(digest(manifest) == d.V1_MANIFEST_SHA, "V1_SIBLING_UPLOAD_MANIFEST_HASH")
    expected = read_json(manifest)
    for name, sha in expected["files"].items():
        require(digest(V1 / name) == sha, "V1_SIBLING_FILE_HASH " + name)
    return check, expected


def v1_records(path, snapshot, old_release):
    import run as v1_run
    import protocol as p
    from source import required_files

    require((path / "registration.json").is_file(), "V1_RUN_REGISTRATION_NOT_FOUND")
    require(not (path / "RUNNING.lock").exists(), "V1_RUN_STILL_ACTIVE")
    stamp = digest(path / "registration.json")
    record = read_json(path / "registration.json")
    require(record["snapshot_sha256"] == snapshot.manifest_sha, "V1_RUN_SNAPSHOT_DIFFERS")
    require(record["scope"] == "discovery only", "V1_NOT_DISCOVERY_RUN")
    results, hashes = {}, {}
    for pair in p.MODES["all"]:
        x, e = pair
        file = v1_run.task_path(path, x, e)
        require(file.is_file(), f"V1_RESULT_MISSING {x} {e}")
        old_hash = digest(file)
        item = v1_run.read_task(file, x, e, old_release)
        require(digest(file) == old_hash, f"V1_RESULT_CHANGED {x} {e}")
        identity = item["identity"]
        require({k: identity[k] for k in v1_run.COMMON_IDENTITY} == record, "V1_RESULT_REGISTRATION_DIFFERS")
        require(identity["inputs"] == {k: snapshot.paths[k][1] for k in required_files((pair,))},
                f"V1_RESULT_INPUT_HASHES_DIFFER {x} {e}")
        results[pair] = item["result"]
        hashes[file.name] = old_hash
    require(not (path / "RUNNING.lock").exists() and digest(path / "registration.json") == stamp,
            "V1_RUN_CHANGED_DURING_READ")
    return results, dict(registration_sha=stamp, task_shas=hashes)


def registered_results(study, refs, progress):
    import numpy as np
    import protocol as p
    from breadth import evaluate_split, collect, replay

    outcomes = {e: study.outcome(e) for e in p.ENDPOINTS}
    family, codes = np.unique(study.family, return_inverse=True)
    tasks, discrepancies = [], {}
    for x in p.EXPOSURES:
        folds = []
        for f in p.FOLDS:
            start = time.monotonic()
            folds.append(evaluate_split(study, x, outcomes, f))
            if progress:
                emit("DONE_FOLD", dict(exposure=x, fold=f, endpoints=len(outcomes),
                                       seconds=round(time.monotonic() - start, 2)))
        for e in p.ENDPOINTS:
            task = collect(x, e, [fold[e] for fold in folds], study.fold, codes, len(family))
            discrepancies[f"{x}/{e}"] = replay(task, refs[x, e])
            tasks.append(task)
    return tasks, discrepancies


def render(summary):
    lines = [
        "FULL_PHENO_V2_SCREEN | same fixed recipe across 2 diets x 9 phenotypes | discovery only",
        "Delta R2 is in percentage points; majority = >=10/18 with gain >=0.10 pp.",
        "Pos = delta > 0; adjusted = simultaneous lower bound across all recipe/endpoint pairs.",
        "recipe | pos | >=.10 | >=.25 | <=-.10 | median | fiber/UPF | glucose/fat/liver | adjusted >0 / >=.10",
    ]
    top = list(summary["ranking"][:10])
    top.extend(r for r in summary["ranking"] if r["recipe"] in d.REFS and r not in top)
    for r in top:
        lines.append(f"{r['recipe']:41} | {r['positive']:2}/18 | {r['at_least_010']:2}/18 |"
                     f" {r['at_least_025']:2}/18 | {r['material_loss']:2}/18 | {r['median_delta_r2_pp']:+.4f} |"
                     f" {r['fiber_hits']}/{r['upf_hits']} |"
                     f" {r['glucose_hits']}/{r['adiposity_hits']}/{r['liver_hits']} |"
                     f" {r['corrected_positive']}/{r['corrected_at_least_010']}")
    best = summary["ranking"][0]
    lines.append(f"Point-estimate winner: {best['recipe']}; majority={best['majority']}; "
                 f"gains in both diets={best['at_least_010_both_diets']}; "
                 f"gains in all domains={best['at_least_010_all_domains']}.")
    point = sum(r["majority"] for r in summary["ranking"])
    confirmed = sum(r["corrected_majority"] for r in summary["ranking"])
    lines.append(f"Recipes with >=10/18 meaningful gains: point={point}, simultaneous lower bound={confirmed}.")
    lines.append("exposure | endpoint | N | R2 off/on | delta pp [simultaneous 95%] | share %")
    for row in summary["details"]:
        if row["recipe"] == best["recipe"]:
            share = "NA" if row["share_pct"] is None else f"{row['share_pct']:+.2f}"
            lines.append(f"{row['exposure']:13} | {row['endpoint']:25} | {row['people']:4} | "
                         f"{row['r2_off']:+.5f}/{row['r2_on']:+.5f} | {row['delta_r2_pp']:+.4f} "
                         f"[{row['simultaneous95_low_pp']:+.4f},{row['simultaneous95_high_pp']:+.4f}] | {share}")
    lines.extend(["Intervals condition on fixed out-of-fold predictions; no independent validation.",
                  "FULL_PHENO_V2_SCREEN_END | summary.json, breadth_ranking.csv, all_pairs.csv"])
    return "\n".join(lines) + "\n"


def write_csv(atomic, path, rows):
    buf = io.StringIO(newline="")
    csv_file = csv.DictWriter(buf, fieldnames=tuple(rows[0]))
    csv_file.writeheader()
    csv_file.writerows(rows)
    atomic(path, buf.getvalue().encode("utf-8"))


def load_completed(path, release):
    complete = path / "COMPLETE.json"
    if not complete.is_file():
        require(not (path / "RUNNING.lock").exists(), "V2_RUN_STILL_ACTIVE")
        raise Stop("V2_RUN_NOT_COMPLETE")
    receipt = read_json(complete)
    require(receipt["version"] == d.VERSION and receipt["release"] == release, "V2_RUN_RELEASE_DIFFERS")
    require(digest(path / "registration.json") == receipt["registration_sha256"], "V2_RESULT_REGISTRATION_HASH")
    for n in ("breadth_ranking.csv", "all_pairs.csv"):
        require(digest(path / n) == receipt["artifacts"][n], "V2_RESULT_HASH " + n)
    summary = (path / "summary.json").read_bytes()
    require(sha256(summary).hexdigest() == receipt["artifacts"]["summary.json"], "V2_RESULT_HASH summary.json")
    return json.loads(summary)


def directory_id(text):
    require(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,48}", text) is not None, "INVALID_RUN_ID")
    return text


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--report", action="store_true")
    cli.add_argument("--run-id", default="discovery-1", type=directory_id)
    cli.add_argument("--v1-run-id", default="discovery-1", type=directory_id)
    cli.add_argument("--threads", type=int, default=2)
    cli.add_argument("--handoff", type=Path, default=HERE.parents[2])
    args = cli.parse_args()
    require(1 <= args.threads <= 16, "V2_THREADS")
    release, old_release = verify_release()
    sys.path.insert(1, str(V1))
    import run as v1_run
    import protocol as p

    runtime = v1_run.init_runtime(args.threads)
    emit("RELEASE_OK", dict(version=d.VERSION, recipes=len(d.NAMES), tasks=len(p.MODES["all"])))
    if args.preflight:
        import unittest
        import test_v2

        result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(test_v2))
        if not result.wasSuccessful():
            return 1
        emit("PREFLIGHT_OK", dict(tests=result.testsRun, inputs="generated fixtures only"))
        return 0
    out = HERE / "runs/TRE_ONLY" / args.run_id
    require(out.resolve().is_relative_to(HERE), "V2_OUTPUT_OUTSIDE_VERSION")
    if args.report:
        print(render(load_completed(out, release)), end="", flush=True)
        return 0
    from breadth import make_summary
    from source import Snapshot, Study, inventory, required_files

    root = args.handoff.resolve(strict=True)
    require(root.name == p.HANDOFF, "EXPECTED_REGISTERED_COLLABORATOR_HANDOFF_DIRECTORY")
    combinations = p.MODES["all"]
    snapshot = Snapshot(root, required_files(combinations))
    emit("INVENTORY", inventory(snapshot, combinations))
    refs, hashes = v1_records(V1 / "runs/TRE_ONLY" / args.v1_run_id, snapshot, old_release)
    reg = dict(release=release, runtime=runtime, snapshot_sha=snapshot.manifest_sha,
               sources=snapshot.report(), v1_receipt=hashes, scope="discovery only")
    with v1_run.run_lock(out):
        registration = out / "registration.json"
        if registration.exists():
            require(read_json(registration) == reg, "V2_RUN_REGISTRATION_DIFFERS")
        else:
            require(not (out / "COMPLETE.json").exists(), "V2_RUN_REGISTRATION_NOT_FOUND")
            v1_run.atomic_write(registration, v1_run.encode(reg))
        if (out / "COMPLETE.json").exists():
            summary = load_completed(out, release)
        else:
            study = Study(snapshot, with_upf=True)
            tasks, replay = registered_results(study, refs, progress=True)
            require(set(snapshot.paths) <= snapshot.verified, "V2_UNVERIFIED_INPUT")
            emit("V1_REPLAY_OK", dict(tasks=len(tasks), max_abs_r2=max(replay.values())))
            summary = make_summary(tasks)
            snapshot.verify_manifest()
            summary.update(v1_replay_max_abs_r2=max(replay.values()), scope="discovery only",
                           confirmation_outcomes_read=False, recipe_count=len(d.NAMES))
            for filename, document in (("breadth_ranking.csv", summary["ranking"]), ("all_pairs.csv", summary["details"])):
                write_csv(v1_run.atomic_write, out / filename, document)
            v1_run.atomic_write(out / "summary.json", v1_run.encode(summary))
            seal = dict(version=d.VERSION, release=release, registration_sha256=digest(registration), artifacts={
                n: digest(out / n) for n in ("summary.json", "breadth_ranking.csv", "all_pairs.csv")})
            v1_run.atomic_write(out / "COMPLETE.json", v1_run.encode(seal))
    emit("COMPLETE", dict(tasks=len(combinations), recipes=len(d.NAMES), reserved_outcomes_read=False))
    print(render(summary), end="", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, TypeError, KeyError, OSError, ArithmeticError, PackageNotFoundError) as exc:
        message = str(exc)
        if isinstance(exc, FileNotFoundError) and exc.filename:
            message = "FILE_NOT_FOUND " + Path(exc.filename).name
        elif not isinstance(exc, Stop) and not re.fullmatch(r"[A-Z][A-Za-z0-9_.; =@/-]*", message):
            message = type(exc).__name__
        emit("STOP", dict(reason=message))
        raise SystemExit(1) from None
