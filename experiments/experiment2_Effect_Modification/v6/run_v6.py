#!/usr/bin/env python3
"""V6 entry point: preflight, historical audit, discovery evaluation, report."""
import argparse
from contextlib import contextmanager, redirect_stdout
import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parent
VERSION = "effect_modification_v6_20261118_expanded"
RUN = ROOT / "runs/TRE_ONLY" / VERSION
PINS = {"numpy": "1.26.4", "scipy": "1.11.4", "pandas": "2.1.0", "pyarrow": "15.0.2",
        "scikit-learn": "1.4.1.post1", "threadpoolctl": "3.3.0"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def encode(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()


def write_json(path, value):
    # Only new V6 run artifacts may be written by this entry point.
    path = Path(path)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(encode(value))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def emit(prefix, value):
    print(prefix + " " + json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def release():
    manifest = read_json(ROOT / "upload_manifest.json")
    expected = manifest["files"]
    for relative, sha in expected.items():
        path = ROOT / relative
        if not path.is_file() or digest(path) != sha:
            raise RuntimeError("UPLOAD_FILE_MISMATCH " + relative)
    if manifest["version"] != VERSION:
        raise RuntimeError("UPLOAD_VERSION_MISMATCH")
    emit("EXP2_V6_RELEASE_OK", dict(files=len(expected), version=VERSION))
    return expected


def environment():
    actual = {name: importlib.metadata.version(name) for name in PINS}
    bad = {name: {"actual": actual[name], "expected": value} for name, value in PINS.items()
           if actual[name] != value}
    if sys.version_info[:2] != (3, 11) or bad:
        raise RuntimeError("PINNED_ENVIRONMENT_MISMATCH " + json.dumps(bad) +
                           "; use the existing V12 .python_path interpreter")
    emit("EXP2_V6_ENV_OK", actual)


def runtime_identity(threads):
    """Stable runtime fields that must agree before reusing fitted folds."""
    from threadpoolctl import threadpool_info
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("REQUESTED_THREADS_MUST_BE_POSITIVE_INTEGER")
    pools = [dict(api=p.get("internal_api"), prefix=p.get("prefix"),
                  version=p.get("version"), num_threads=p.get("num_threads"))
             for p in threadpool_info()]
    pools.sort(key=lambda p: json.dumps(p, sort_keys=True))
    return dict(python_patch=".".join(map(str, sys.version_info[:3])),
                package_versions={name: importlib.metadata.version(name) for name in PINS},
                requested_threads=threads, threadpools=pools)


def cohort_identity(source):
    """Record the evaluated source size without assuming native fixture counts."""
    import numpy as np
    y, groups = np.asarray(source.reference["y"]), np.asarray(source.groups)
    if y.ndim != 2 or y.shape[0] != 2 or groups.shape != (y.shape[1],):
        raise ValueError("INVALID_SOURCE_COHORT_SHAPE")
    result = []
    for repeat in (0, 1):
        mask = np.isfinite(y[repeat])
        people, households = int(mask.sum()), int(len(np.unique(groups[mask])))
        if people < 2 or households < 2:
            raise ValueError("INSUFFICIENT_SOURCE_COHORT")
        result.append(dict(repeat=repeat, people=people, households=households))
    return result


@contextmanager
def lock(run):
    run.mkdir(parents=True, exist_ok=True)
    path = run / "RUNNING.lock"
    owner = dict(pid=os.getpid(), host=socket.gethostname(), token=uuid.uuid4().hex)
    if path.exists():
        old = read_json(path)
        # Never automatically unlink a stale pathname: simultaneous recovery
        # processes could otherwise delete a newly acquired lock (TOCTOU).
        raise RuntimeError("RUN_LOCK_EXISTS " + json.dumps(old) +
                           "; verify that process has ended before manually removing RUNNING.lock")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(owner, stream)
    try:
        yield
    finally:
        if path.exists() and read_json(path).get("token") == owner["token"]:
            path.unlink()


def save_fold(run, repeat, fold, arrays, metadata):
    import numpy as np
    final = run / f"r{repeat}_f{fold}"
    if final.exists():
        raise RuntimeError("REFUSING_TO_OVERWRITE_COMPLETED_FOLD")
    temporary = Path(tempfile.mkdtemp(prefix=".pending_fold_", dir=run))
    try:
        # ZIP seeks only in RAM. Filesystem receives a single sequential write.
        memory = io.BytesIO()
        np.savez_compressed(memory, **arrays)
        with (temporary / "arrays.npz").open("xb") as stream:
            stream.write(memory.getvalue())
        write_json(temporary / "metadata.json", metadata)
        write_json(temporary / "receipt.json", dict(
            arrays_sha256=digest(temporary / "arrays.npz"),
            metadata_sha256=digest(temporary / "metadata.json")))
        temporary.rename(final)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def read_fold(run, repeat, fold):
    import numpy as np
    final = run / f"r{repeat}_f{fold}"
    receipt = read_json(final / "receipt.json")
    for name in ("arrays", "metadata"):
        ext = "npz" if name == "arrays" else "json"
        if digest(final / f"{name}.{ext}") != receipt[name + "_sha256"]:
            raise RuntimeError("FOLD_CHECKPOINT_HASH_MISMATCH")
    metadata = read_json(final / "metadata.json")
    if metadata["repeat"] != repeat or metadata["fold"] != fold:
        raise RuntimeError("CHECKPOINT_FOLD_ID_MISMATCH")
    with np.load(io.BytesIO((final / "arrays.npz").read_bytes()), allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    return arrays, metadata


def completed(run, *, expected_code_hashes=None):
    receipt = read_json(run / "COMPLETE.json")
    if receipt.get("version") != VERSION:
        raise RuntimeError("COMPLETED_RUN_VERSION_MISMATCH")
    expected_files = {"registration.json", "summary.json", "all_routes.csv"}
    expected_files.update(f"r{r}_f{f}/{name}" for r in (0, 1) for f in range(5)
                          for name in ("arrays.npz", "metadata.json", "receipt.json"))
    if not isinstance(receipt.get("artifacts"), dict) or set(receipt["artifacts"]) != expected_files:
        raise RuntimeError("COMPLETED_RUN_ARTIFACT_SET_INCOMPLETE")
    for filename, sha in receipt["artifacts"].items():
        if not (run / filename).is_file() or digest(run / filename) != sha:
            raise RuntimeError("COMPLETED_RUN_ARTIFACT_CHANGED " + filename)
    registration, summary = read_json(run / "registration.json"), read_json(run / "summary.json")
    if registration.get("version") != VERSION:
        raise RuntimeError("COMPLETED_REGISTRATION_VERSION_MISMATCH")
    if expected_code_hashes is not None and registration.get("code_hashes") != expected_code_hashes:
        raise RuntimeError("COMPLETED_RUN_CODE_MISMATCH")
    config = registration.get("config", {})
    routes, settings = config.get("routes"), config.get("settings")
    if (config.get("repeats") != 2 or config.get("outer_folds") != 5
            or config.get("inner_folds") != 3 or config.get("outcome") != "D_Glu"
            or config.get("exposure") != "fiber_density" or config.get("contrast") != [10, 15]
            or config.get("reserved_outcomes_read") is not False
            or not isinstance(routes, list) or not routes
            or not all(isinstance(name, str) and name for name in routes)
            or len(routes) != len(set(routes)) or "nested_selector" not in routes
            or not isinstance(settings, list) or not settings):
        raise RuntimeError("COMPLETED_REGISTRATION_SCHEMA_MISMATCH")
    cohorts, repeats = registration.get("cohort"), summary.get("repeats")
    if (not isinstance(cohorts, list) or not isinstance(repeats, list)
            or [row.get("repeat") for row in cohorts] != [0, 1]
            or [row.get("repeat") for row in repeats] != [0, 1]
            or summary.get("route_count") != len(routes) or summary.get("settings") != len(settings)
            or summary.get("outcome") != config["outcome"]
            or summary.get("exposure") != config["exposure"]
            or summary.get("reserved_outcomes_read") is not False):
        raise RuntimeError("COMPLETED_SUMMARY_SCHEMA_MISMATCH")
    for repeat, (cohort, result) in enumerate(zip(cohorts, repeats, strict=True)):
        if (set(result.get("routes", {})) != set(routes)
                or result.get("people") != cohort.get("people")
                or result.get("households") != cohort.get("households")
                or not isinstance(cohort.get("people"), int) or cohort["people"] < 2
                or not isinstance(cohort.get("households"), int) or cohort["households"] < 2):
            raise RuntimeError("COMPLETED_SUMMARY_COHORT_OR_ROUTE_MISMATCH")
        fold_people = 0
        for fold in range(5):
            metadata = read_json(run / f"r{repeat}_f{fold}/metadata.json")
            if (metadata.get("repeat") != repeat or metadata.get("fold") != fold
                    or metadata.get("names") != routes
                    or not isinstance(metadata.get("people"), int) or metadata["people"] < 1):
                raise RuntimeError("COMPLETED_FOLD_SCHEMA_MISMATCH")
            fold_people += metadata["people"]
        if fold_people != cohort["people"]:
            raise RuntimeError("COMPLETED_FOLD_COHORT_INCOMPLETE")
    return summary


def compact(summary):
    for result in summary["repeats"]:
        emit("EXP2_V6_PRIMARY", dict(repeat=result["repeat"], baseline_r2=result["baseline_r2"],
                                     **result["routes"]["nested_selector"]))
        emit("EXP2_V6_EXPLORATORY_MAX", dict(repeat=result["repeat"],
                                               result=result["exploratory_observed_max"]))
        emit("EXP2_V6_ADDED_METHODS_COMPARE", dict(repeat=result["repeat"],
             comparison=result["added_methods_vs_original_selector"]))
    emit("EXP2_V6_SUMMARY", dict(
        scope=summary["scope"], routes=summary["route_count"], settings=summary["settings"],
        positive_in_both_repeats=summary["exploratory_positive_both_repeats"][:10],
        repeats_are_same_people=True, reserved_outcomes_read=False,
        interpretation="relative predictive R2 gain; not a causal effect proportion"))


def result_files(run, summary):
    write_json(run / "summary.json", summary)
    path = run / "all_routes.csv"
    fields = ["repeat", "route", "baseline_r2", "full_r2", "delta_r2", "delta_r2_percentage_points",
              "delta_r2_ci95", "ratio_over_full_pct", "ratio_over_full_ci95_pct",
              "ratio_over_full_status", "relative_r2_change_over_base_pct", "active_folds",
              "contrast_sd_10_to_15_mg_dl"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for result in summary["repeats"]:
            for name, data in result["routes"].items():
                row = dict(repeat=result["repeat"], route=name, baseline_r2=result["baseline_r2"], **data)
                writer.writerow({k: json.dumps(row[k]) if isinstance(row[k], list) else row[k] for k in fields})
    artifacts = {str(p.relative_to(run)): digest(p) for p in
                 [run / "registration.json", run / "summary.json", path,
                  *sorted(run.glob("r*_f*/*"))] if p.is_file()}
    write_json(run / "COMPLETE.json", dict(version=VERSION, artifacts=artifacts))


def execute(source, run, code_hashes, *, draws=1999, bases=None, emit_progress=True, threads=2):
    from experiment import settings, route_names, run_fold, summarize
    from adapters import moderation_models as mm
    bases = mm.BASES if bases is None else bases
    registration = dict(version=VERSION, code_hashes=code_hashes,
                        source_hashes={str(k): v for k, v in source.source_hashes.items()},
                        runtime=runtime_identity(threads), cohort=cohort_identity(source),
                        config=dict(exposure="fiber_density", outcome="D_Glu", contrast=[10, 15],
                                    repeats=2, outer_folds=5, inner_folds=3, draws=draws,
                                    bases=[list(b) for b in bases], settings=[dict(name=n, **c.to_dict()) for n, c in settings()],
                                    routes=route_names(), reserved_outcomes_read=False))
    with lock(run):
        path = run / "registration.json"
        if path.exists():
            if read_json(path) != registration:
                raise RuntimeError("RUN_REGISTRATION_CHANGED; keep this run and use a new version folder")
        else:
            if any(run.glob("r*_f*")) or (run / "COMPLETE.json").exists():
                raise RuntimeError("UNREGISTERED_RUN_ARTIFACTS")
            write_json(path, registration)
        if (run / "COMPLETE.json").exists():
            result = completed(run, expected_code_hashes=code_hashes)
            compact(result)
            return result
        folds = {}
        for repeat in (0, 1):
            for fold in range(5):
                key = (repeat, fold)
                if (run / f"r{repeat}_f{fold}").exists():
                    folds[key] = read_fold(run, repeat, fold)
                    emit("EXP2_V6_REUSED_FOLD", dict(repeat=repeat, fold=fold))
                    continue
                arrays, metadata = run_fold(source, repeat, fold, bases=bases,
                                              emit=print if emit_progress else lambda _: None)
                source.verify_sources()
                save_fold(run, repeat, fold, arrays, metadata)
                folds[key] = arrays, metadata
                emit("EXP2_V6_OUTER_COMPLETE", {k: metadata[k] for k in
                                               ("repeat", "fold", "people", "baseline", "seconds")})
        result = summarize(source, folds, draws=draws)
        source.verify_sources()
        result_files(run, result)
        result = completed(run, expected_code_hashes=code_hashes)
        compact(result)
        return result


def audit():
    from history_reader import scan, paired_row
    package = ROOT.parent.parent
    current, issues = scan(package)
    local = []
    for item in read_json(ROOT / "historical_pairs.json"):
        row = paired_row(item["label"], item["repeat"], item["base"], item["full"], item["delta"],
                          item["source"], item["stage"], item.get("ci"))
        row.update(origin="shipped historical archive", notes=item.get("notes", ""))
        local.append(row)
    for row in current:
        row["origin"] = "server saved aggregate"
    for row in [*local, *current]:
        label = row["label"]
        row["task_scope"] = ("fiber_to_person_D_Glu" if label.startswith(
            ("legacy_V1 ", "legacy_V10 fiber_", "v1:", "v2:", "v5:", "exp2_v", "legacy_v5:"))
            else "other_or_not_harmonized_historical_task")
    unique = {}
    for row in [*local, *current]:
        # Duplicate filenames/log copies do not count as independent evidence.
        key = tuple(row[k] for k in ("label", "repeat", "base_r2", "full_r2", "delta_r2"))
        unique[key] = row
    rows = list(unique.values())
    eligible = sorted((r for r in rows if r["positive_r2_denominators"] and r["delta_r2"] > 0),
                      key=lambda r: r["ratio_over_full_pct"], reverse=True)
    target = [r for r in eligible if r["task_scope"] == "fiber_to_person_D_Glu"]
    result = dict(scope="saved aggregate audit; no raw participant data or model fits", rows=rows,
                  shipped_rows=len(local), server_rows=len(current), issues=issues,
                  historical_missing_groups=read_json(ROOT / "historical_missing.json"),
                  exploratory_max=eligible[0] if eligible else None,
                  fiber_person_D_Glu_exploratory_max=target[0] if target else None,
                  archive_scope="heterogeneous endpoints, populations and validation stages; do not pool as one experiment",
                  interpretation="ratio of matched prediction scores; not a causal diet effect share")
    folder = ROOT / "runs/TRE_ONLY/history_audit"
    with lock(folder):
        write_json(folder / "history_ratios.json", result)
    for row in eligible[:10]:
        emit("EXP2_V6_HISTORY_TOP", row)
    emit("EXP2_V6_HISTORY_COMPLETE", dict(shipped_pairs=len(local), server_pairs=len(current),
                                          combined_pairs=len(rows), issues=len(issues),
                                          maximum=result["exploratory_max"],
                                          fiber_person_D_Glu_maximum=result["fiber_person_D_Glu_exploratory_max"],
                                          archive_scope=result["archive_scope"],
                                          missing_summaries_are_not_zero=True, raw_rows_read=False))


def preflight():
    import unittest
    suite = unittest.defaultTestLoader.discover(str(ROOT), pattern="test_*.py")
    # Test lifecycle fixtures may emit summary-shaped synthetic records. Keep
    # them out of the server log so they cannot be mistaken for HPP estimates.
    with redirect_stdout(io.StringIO()):
        result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("SYNTHETIC_PREFLIGHT_FAILED")
    from experiment import grid, settings, route_names
    emit("EXP2_V6_PREFLIGHT_DONE", dict(tests=result.testsRun, single_families=len(grid()),
                                        settings=len(settings()), routes=len(route_names()),
                                        scope="generated data only; no native HPP rows"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--audit", action="store_true")
    group.add_argument("--report", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    hashes = release()
    if args.audit:
        audit()
        return
    environment()
    # threadpoolctl limits only libraries already loaded when its context is
    # entered. Import the full numerical stack first, including sklearn's
    # OpenMP library; a fresh CLI process otherwise ignores --threads.
    importlib.import_module("experiment")
    from adapters.person_inputs import load_discovery
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=args.threads):
        if args.preflight:
            preflight()
        elif args.report:
            compact(completed(RUN, expected_code_hashes=hashes))
        else:
            source = load_discovery(ROOT)
            emit("EXP2_V6_SOURCE_READY", dict(people=3343, households=3059,
                                               confirmation_outcomes_read=False))
            execute(source, RUN, hashes, threads=args.threads)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Full traceback is useful for a concrete repair; no participant values
        # are deliberately printed. Preserve the failed run and checkpoints.
        emit("EXP2_V6_STOP", dict(error=type(exc).__name__, detail=str(exc)))
        raise
