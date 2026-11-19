"""Run a fixed discovery-only full phenotype v1 without exporting person-level records."""

import argparse
import contextlib
import csv
from hashlib import file_digest, sha256
from importlib.metadata import PackageNotFoundError, version
import io
import json
import os
from pathlib import Path
import platform
import re
import sys
from tempfile import NamedTemporaryFile
import time

import protocol as p


RELEASE_FILES = (
    "README.md", "protocol.py", "source.py", "models.py", "metrics.py", "reference.py",
    "run.py", "server_run.sh", "test_full_pheno.py",
)


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + "\n").encode("utf-8")


def emit(event, value):
    print("FULL_PHENO_V1_" + event, json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def file_sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_release():
    manifest = read_json(p.HERE / "upload_manifest.json")
    require(manifest["version"] == p.VERSION and set(manifest["files"]) == set(RELEASE_FILES),
            "INVALID_UPLOAD_MANIFEST")
    for name in RELEASE_FILES:
        require(file_sha(p.HERE / name) == manifest["files"][name], "UPLOAD_FILE_MISMATCH " + name)
    return manifest


def init_runtime(threads):
    require(sys.version_info[:2] == (3, 11), "EXPECTED_PYTHON_3_11; use the pinned V12 interpreter")
    actual = {name: version(name) for name in p.PINS}
    for name, expected in p.PINS.items():
        require(actual[name] == expected, f"VERSION_MISMATCH {name}={actual[name]} expected={expected}")
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = str(threads)
    return dict(python=platform.python_version(), packages=actual, threads=threads,
                platform=platform.system(), architecture=platform.machine())


def atomic_write(path, value):
    """One sequential write and same-directory rename (works on the TRE mount)."""
    with NamedTemporaryFile(prefix=".pending_", dir=path.parent, delete=False) as f:
        temp = Path(f.name)
        try:
            f.write(value)
            f.close()
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


@contextlib.contextmanager
def run_lock(out):
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = out / "RUNNING.lock"
    try:
        with lock.open("x", encoding="utf-8") as f:
            f.write(json.dumps({"pid": os.getpid(), "host": platform.node()}) + "\n")
            lock_inode = os.fstat(f.fileno()).st_ino
    except FileExistsError as exc:
        raise ValueError("RUN_LOCK_EXISTS; verify owner has exited before removing the run lock") from exc
    try:
        yield
    finally:
        if lock.exists() and lock.stat().st_ino == lock_inode:
            lock.unlink()


def task_identity(snapshot, exposure, endpoint, release, runtime, draws, v10_root):
    from reference import receipts
    from source import required_files

    registered = snapshot.report()
    result = dict(run_identity(snapshot, release, runtime, draws), endpoint=endpoint, exposure=exposure,
                  inputs={k: registered[k] for k in required_files(((exposure, endpoint),))})
    if v10_root is not None and (exposure, endpoint) in p.MODES["bridge"]:
        result["v10_reference_receipts"] = receipts(v10_root)
    return result


def task_path(out, exposure, endpoint):
    return out / f"{exposure}__{endpoint}.json"


COMMON_IDENTITY = ("version", "scope", "release", "runtime", "bootstrap_draws", "snapshot_sha256")


def run_identity(snapshot, release, runtime, draws):
    return dict(version=p.VERSION, scope="discovery only", release=sha256(encode(release)).hexdigest(),
                runtime=runtime, bootstrap_draws=draws, snapshot_sha256=snapshot.manifest_sha)


def save_task(path, identity, result):
    payload = dict(identity=identity, result=result)
    atomic_write(path, encode(dict(payload=payload, sha256=sha256(encode(payload)).hexdigest())))


def read_task(path, exposure, endpoint, release):
    item = read_json(path)
    payload = item["payload"]
    reg = payload["identity"]
    require(item["sha256"] == sha256(encode(payload)).hexdigest(), "RESULT_HASH_MISMATCH " + path.name)
    require(reg["version"] == p.VERSION and reg["exposure"] == exposure and reg["endpoint"] == endpoint
            and reg["release"] == sha256(encode(release)).hexdigest(), "RESULT_REGISTRATION_MISMATCH " + path.name)
    return payload


def read_report(out, tasks, release):
    require(not (out / "RUNNING.lock").exists(), "RUN_IN_PROGRESS")
    record = read_json(out / "registration.json")
    result = []
    for x, endpoint in tasks:
        task = read_task(task_path(out, x, endpoint), x, endpoint, release)
        require({k: task["identity"][k] for k in COMMON_IDENTITY} == record, "RESULT_RUN_MISMATCH")
        result.append(task["result"])
    return result


def evaluate(study, exposure, endpoint, outcome, *, draws, v10_root=None, progress=False):
    import numpy as np
    from metrics import association, prediction
    from models import fit_fold
    from reference import load_v10

    parts = []
    for f in p.FOLDS:
        fold = fit_fold(study, exposure, endpoint, outcome, f)
        parts.append(fold)
        if progress:
            emit("FOLD", dict(exposure=exposure, endpoint=endpoint, **fold.counts()))
    rows = np.concatenate([f.rows for f in parts])
    groups = study.family[rows]
    require(len(np.unique(rows)) == len(rows), "DUPLICATED_OUT_OF_FOLD_ROWS")
    require(all(np.all(study.fold[f.rows] == f.fold) for f in parts), "OOF_FOLD_MISMATCH")
    g = len(np.unique(groups))
    require(sum(f.held_households for f in parts) == g, "DUPLICATED_HELD_HOUSEHOLD")
    if (exposure, endpoint) == ("fiber_density", "D_Glu") and study.native:
        require(tuple(len(f.rows) for f in parts) == p.EXPECTED_GLU_FOLDS and g == p.EXPECTED_GLU_HOUSEHOLDS,
                "HISTORICAL_GLUCOSE_BRIDGE_COUNTS_DIFFER")
    y = np.concatenate([f.y for f in parts])
    candidates = np.vstack([f.predictions for f in parts])
    scores = prediction(y, candidates, groups, draws, p.BOOTSTRAP_SEED)
    effects = association(study, exposure, rows, outcome)
    result = dict(exposure=exposure, endpoint=endpoint, people=len(rows), households=g,
                  unit=outcome.unit, folds=[f.counts() for f in parts], score=scores, association=effects)
    if v10_root is not None and (exposure, endpoint) == ("fiber_density", "D_Glu"):
        result["v10_reference"] = load_v10(v10_root, study.snapshot, parts, study.family, draws)
    return result


def header_numbers(result):
    from metrics import CONTRASTS

    score = result["score"]
    models = score.get("models", {})
    pairs = score.get("pairs", {})
    markers = result["association"]["markers"]
    fields = {k: result[k] for k in ("exposure", "endpoint", "people", "households", "unit")}
    fields.update(prediction_status=score["status"], prediction_interval_status=score.get("interval_status"))
    fields.update({f"r2_{k}": models.get(k, {}).get("r2") for k in p.PREDICTORS})
    for label in CONTRASTS:
        pair = pairs.get(label, {})
        fields[f"{label}_delta_r2_pp"] = times_100(pair.get("delta_r2"))
        for field, stem in (("delta_ci95", "ci95"), ("delta_simultaneous95", "family95")):
            ci = pair.get(field)
            fields[f"{label}_{stem}_low_pp"] = None if ci is None else 100 * ci[0]
            fields[f"{label}_{stem}_high_pp"] = None if ci is None else 100 * ci[1]
        fields[f"{label}_share_pct"] = pair.get("share_pct")
        fields[f"{label}_share_status"] = pair.get("share_status")
        ratio = pair.get("share_ci95")
        fields[f"{label}_share_ci95_low"] = None if ratio is None else ratio[0]
        fields[f"{label}_share_ci95_high"] = None if ratio is None else ratio[1]
    for label in p.MARKERS:
        marker = markers[label]
        fields[f"{label}_contrast_status"] = marker["status"]
        fields[f"{label}_contrast_reason"] = marker.get("reason")
        fields[f"{label}_contrast"] = marker.get("estimate")
        ci = marker.get("simultaneous_ci95")
        fields[f"{label}_contrast_family95_low"] = None if ci is None else ci[0]
        fields[f"{label}_contrast_family95_high"] = None if ci is None else ci[1]
    return fields


def times_100(value):
    return None if value is None else 100 * value


def fmt(value, digits=5):
    return "NA" if value is None else f"{value:+.{digits}f}"


def render(records):
    lines = ["# Full phenotype v1 | discovery only", "",
             "Paired held-out predictions on the same people per endpoint.",
             "Delta is in R2 percentage points. Share uses the enhanced model's R2; NA means unavailable.",
             "Family intervals adjust across two prediction contrasts (C-A, C-B) and nine outcomes per exposure.",
             "B-A is descriptive and has a pointwise interval.", "",
             "|Exposure|Endpoint|N / households|R2 off / panel / fusion|"
             "Panel delta [95%]|All interaction delta [family 95%]|All share %|Shannon extra delta [family 95%]|",
             "|---|---|---:|---|---:|---:|---:|---:|"]
    for item in records:
        r = header_numbers(item)
        scores = " / ".join(fmt(r[f"r2_{k}"]) for k in p.PREDICTORS)
        panel, total, shannon = [
            fmt(r[f"{label}_delta_r2_pp"], 4) + " [" +
            ", ".join(fmt(r[f"{label}_{ci}_{k}_pp"], 4) for k in ("low", "high")) + "]"
            for label, ci in (("panel_minus_off", "ci95"), ("all_interactions", "family95"),
                              ("shannon_increment", "family95"))]
        lines.append(f"|{r['exposure']}|{r['endpoint']}|{r['people']} / {r['households']}|{scores}|"
                     f"{panel}|{total}|{fmt(r['all_interactions_share_pct'], 2)}|{shannon}|")
    lines += ["", "|Exposure|Endpoint|Outcome unit|Three-species contrast [family 95%]|"
              "Shannon contrast [family 95%]|", "|---|---|---|---:|---:|"]
    for item in records:
        r = header_numbers(item)
        term = [fmt(r[f"{name}_contrast"], 4) + " [" + ", ".join(
            fmt(r[f"{name}_contrast_family95_{k}"], 4) for k in ("low", "high")) + "]"
                 if r[f"{name}_contrast_status"] == "ok" else "NA"
                 for name in p.MARKERS]
        unit = r["unit"].replace("|", r"\|").replace("\n", " ").replace("\r", " ")
        lines.append(f"|{r['exposure']}|{r['endpoint']}|{unit}|{term[0]}|{term[1]}|")
    for item in records:
        ref = item.get("v10_reference")
        if ref:
            lines += ["", "V10 bridge: " + ref["status"]]
            if ref["status"] == "matched":
                scored = ref["scoring"]
                if scored["status"] == "ok":
                    off, on = (scored["models"][key]["r2"] for key in ("v10_off", "v10_on"))
                    pair = scored["pairs"]["all_interactions"]
                    lines.append(f"V10 {ref['setting']} r0: {fmt(off)} -> {fmt(on)}; "
                                 f"delta {fmt(100 * pair['delta_r2'], 4)} R2 pp; "
                                 f"share {fmt(pair['share_pct'], 2)}%.")
                else:
                    lines.append("V10 score unavailable: " + scored["reason"])
            else:
                lines.append(ref["reason"])
    lines += ["", "Association contrasts compare marker Q75 vs Q25 for fiber 10 to 15 g/1000 kcal "
              "or calibrated UPF 20% to 30%. Association estimates are descriptive, with two markers "
              "by nine endpoint adjustment per exposure.",
              "Paired prediction intervals condition on fitted predictions.", ""]
    return "\n".join(lines)


def batch(study, tasks, out, scope, release, runtime, *, draws, v10_root):
    records = []
    with run_lock(out):
        run_reg = run_identity(study.snapshot, release, runtime, draws)
        path = out / "registration.json"
        if path.exists():
            require(read_json(path) == run_reg, "RUN_REGISTRATION_DIFFERS; use matching flags or a new --run-id")
        else:
            require(not any(out.glob("*__D_*.json")), "TASK_RESULTS_WITHOUT_RUN_REGISTRATION")
            atomic_write(path, encode(run_reg))
        for exposure, endpoint in tasks:
            path = task_path(out, exposure, endpoint)
            reg = task_identity(study.snapshot, exposure, endpoint, release, runtime, draws, v10_root)
            outcome = study.outcome(endpoint)  # checks the raw outcome even when reusing an aggregate
            if path.exists():
                prior = read_task(path, exposure, endpoint, release)
                require(reg == prior["identity"], "EXISTING_TASK_REGISTRATION_DIFFERS " + path.name)
                record = prior["result"]
                emit("REUSE", dict(exposure=exposure, endpoint=endpoint))
            else:
                start = time.monotonic()
                record = evaluate(study, exposure, endpoint, outcome, draws=draws, v10_root=v10_root, progress=True)
                study.snapshot.verify_manifest()
                require(reg == task_identity(study.snapshot, exposure, endpoint, release, runtime, draws, v10_root),
                        "BRIDGE_REFERENCE_CHANGED_DURING_RUN")
                save_task(path, reg, record)
                emit("DONE", dict(exposure=exposure, endpoint=endpoint, people=record["people"],
                                  households=record["households"], seconds=round(time.monotonic() - start, 2)))
            records.append(record)
        require(set(study.snapshot.paths) <= study.snapshot.verified, "SOME_REGISTERED_INPUTS_NOT_VERIFIED")
        study.snapshot.verify_manifest()
        csv_io = io.StringIO(newline="")
        outputs = [header_numbers(r) for r in records]
        writer = csv.DictWriter(csv_io, fieldnames=tuple(outputs[0]))
        writer.writeheader()
        writer.writerows(outputs)
        csv_name, md_name = f"{scope}_summary.csv", f"{scope}_summary.md"
        atomic_write(out / csv_name, csv_io.getvalue().encode("utf-8"))
        atomic_write(out / md_name, render(records).encode("utf-8"))
        emit("COMPLETE", dict(scope=scope, combinations=len(records), directory=str(out),
                              csv=csv_name, markdown=md_name, reserved_outcomes_read=False))
    return records


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="synthetic tests, without reading native inputs")
    mode.add_argument("--inventory", action="store_true", help="check registered discovery file metadata and schemas")
    mode.add_argument("--report", action="store_true", help="read aggregate results only")
    cli.add_argument("--scope", default="bridge", choices=tuple(p.MODES))
    cli.add_argument("--threads", default=2, type=int)
    cli.add_argument("--run-id", default="discovery-1")
    cli.add_argument("--handoff", type=Path, default=p.HERE.parents[2],
                     help="path to the same registered TRE discovery handoff")
    return cli


def main(argv=None):
    # Set BLAS before loading NumPy/SciPy (incl. on the synthetic test path).
    args = parser().parse_args(argv)
    require(1 <= args.threads <= 16, "THREADS_MUST_BE_1_TO_16")
    require(re.fullmatch("[a-zA-Z0-9][a-zA-Z0-9_-]{0,48}", args.run_id) is not None, "INVALID_RUN_ID")
    release = read_release()
    emit("RELEASE_OK", dict(files=len(release["files"]), version=p.VERSION))
    runtime = init_runtime(args.threads)
    if args.preflight:
        import unittest
        import test_full_pheno
        suite = unittest.defaultTestLoader.loadTestsFromModule(test_full_pheno)
        result = unittest.TextTestRunner(verbosity=1).run(suite)
        if not result.wasSuccessful():
            return 1
        emit("PREFLIGHT_OK", dict(tests=result.testsRun, source="generated fixtures only"))
        return 0

    tasks = p.MODES[args.scope]
    out = p.HERE / "runs/TRE_ONLY" / args.run_id
    require(out.resolve().is_relative_to(p.HERE), "RUN_PATH_OUTSIDE_VERSION")
    if args.report:
        records = read_report(out, tasks, release)
    else:
        from source import Snapshot, Study, inventory, required_files

        root = args.handoff.resolve(strict=True)
        require(root.name == p.HANDOFF, "EXPECTED_REGISTERED_COLLABORATOR_HANDOFF_DIRECTORY")
        snapshot = Snapshot(root, required_files(tasks))
        emit("INVENTORY", inventory(snapshot, tasks))
        if args.inventory:
            return 0
        study = Study(snapshot, with_upf=any(x == "UPF" for x, _ in tasks))
        emit("READY", dict(people=study.n, discovery_households=study.role_counts[0][1],
                           species=study.panel_qc, reserved_roster_metadata_only=True))
        records = batch(study, tasks, out, args.scope, release, runtime, draws=p.BOOTSTRAP_DRAWS,
                        v10_root=p.HERE.parents[1] / "v10")
    print(render(records), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, TypeError, KeyError, OSError, ArithmeticError, PackageNotFoundError) as exc:
        # Expected audit errors have controlled messages. Never print table rows or a traceback of native data.
        msg = str(exc) if isinstance(exc, ValueError) and re.fullmatch(r"[A-Z][A-Za-z0-9_.; =@/-]*", str(exc)) else type(exc).__name__
        if isinstance(exc, FileNotFoundError) and exc.filename:
            msg = f"FILE_NOT_FOUND {Path(exc.filename).name}"
        emit("STOP", dict(reason=msg))
        raise SystemExit(1) from None
