"""V9 discovery experiment, restartable by outer household fold."""

import argparse
import contextlib
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parent
V8 = ROOT.parent / "v8"
sys.path[1:1] = [str(V8), str(ROOT.parent / "v7"), str(ROOT.parent / "v6")]
import run_v8 as prior
import run_v6 as store

VERSION = "effect_modification_v9_paired_broad_20261118_r2"
RUN = ROOT / "runs/TRE_ONLY" / VERSION
ARTIFACTS = (*prior.ARTIFACTS, "parent_comparisons.csv", "fixed_settings.csv")


def emit(name, data):
    store.emit("EXP2_V9_" + name, data)


def release_check():
    release = store.read_json(ROOT / "upload_manifest.json")
    if (
        release["version"] != VERSION
        or set(release["packages"]) != {"v6", "v7", "v8", "v9"}
        or release["pins"] != store.PINS
    ):
        raise ValueError("Incomplete V9 release manifest")
    for folder, items in release["packages"].items():
        for name, digest in items.items():
            path = ROOT.parent / folder / name
            if not path.is_file() or store.digest(path) != digest:
                raise ValueError("Missing or changed upload file: " + folder + "/" + name)
    return release


def protocol():
    import v9_cv as cv
    import v9_features as features
    import v9_grid as grid
    import v8_scores as stats

    data = dict(
        version=VERSION,
        scope="discovery",
        exposure="fiber_density",
        outcome="D_Glu",
        population="unchanged V11/V8 evaluated people",
        paired_own_off=True,
        objective="100 * (R2_on - R2_off) / R2_on; smaller of two household repartitions",
        threshold_pct=stats.TARGET_PCT,
        primary=cv.PRIMARY,
        reference=grid.ANCHOR,
        family_count=len(grid.FAMILIES),
        settings_count=len(grid.KEYS),
        routes=list(cv.NAMES),
        fixed_setting_keys=grid.KEYS,
        simultaneous_scope="all named procedures and fixed settings x both repeats",
        folds=dict(outer=[2, 5], inner=[2, 3], second_seed=cv.v8_selection.SECOND_INNER_SEED),
        bootstrap_draws=1999,
        seed=stats.BOOT_SEED,
        feature_seed=features.SEED,
        fixed_pc10=grid.FIXED_PC10,
        stack_weights=cv.MIX_WEIGHTS,
        stack_pool_per_group=cv.POOL_PER_GROUP,
        stack_max_components=cv.MAX_COMPONENTS,
        reserved_outcomes_read=False,
        **grid.design(),
    )
    return json.loads(json.dumps(data, allow_nan=False))


def replay(folds, previous):
    import numpy as np
    import v8_models as models

    if previous is None or not (previous / "COMPLETE.json").is_file():
        return dict(status="V8_archive_unavailable_controls_recomputed", folds_checked=0)
    stamp = store.digest(previous / "COMPLETE.json")
    prior.completed(previous, store.read_json(V8 / "upload_manifest.json"))
    maximum = np.zeros(2)
    for r in (0, 1):
        for f in range(5):
            a, m = store.read_fold(previous, r, f)
            b, n = folds[r, f]
            if any(not np.array_equal(a[k], b[k]) for k in ("rows", "y", "groups")):
                raise ValueError(f"V8 control cohort/target mismatch in r{r} f{f}")
            i, j = [], []
            for name in models.FAMILIES:
                old, new = m["choices"][name], n["choices"][name]
                if (
                    old["settings"] != new["settings"]
                    or old["enabled"] != new["enabled"]
                    or new["weights"] != [1.0]
                ):
                    raise ValueError(f"V8 control settings differ at {name} r{r} f{f}")
                i.append(m["names"].index(name))
                j.append(n["names"].index(name))
            difference = [np.max(abs(a[key][:, i] - b[key][:, j])) for key in ("off", "on")]
            if not np.all(np.isfinite(difference)) or max(difference) > 2e-8:
                raise ValueError(f"V8 paired predictions differ at r{r} f{f}: {difference}")
            maximum = np.maximum(maximum, difference)
    if store.digest(previous / "COMPLETE.json") != stamp:
        raise ValueError("V8 receipt changed during comparison")
    return dict(
        status="V8_predictions_match",
        families_checked=len(models.FAMILIES),
        folds_checked=10,
        max_abs_off_on_mg_dl=maximum.tolist(),
        tolerance_mg_dl=2e-8,
        receipt_sha256=stamp,
    )


def write_parents(path, rows):
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", dir=path.parent, delete=False
    ) as out:
        temp = Path(out.name)
        try:
            writer = csv.DictWriter(out, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()})
            out.close()
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


def completed(run, release):
    receipt = store.read_json(run / "COMPLETE.json")
    if receipt["version"] != VERSION or set(receipt["artifacts"]) != set(ARTIFACTS):
        raise ValueError("Incomplete V9 receipt")
    for name, expected in receipt["artifacts"].items():
        if store.digest(run / name) != expected:
            raise ValueError("Completed artifact changed: " + name)
    reg = store.read_json(run / "registration.json")
    if reg["release"] != release or reg["protocol"]["version"] != VERSION:
        raise ValueError("Completed run belongs to another release")
    return store.read_json(run / "summary.json")


def execute(source, run, release, threads, *, draws=1999, previous=None):
    import v9_cv as cv
    import v9_report as report

    registration = dict(
        protocol=protocol(),
        release=release,
        bootstrap_draws=draws,
        runtime=store.runtime_identity(threads),
        cohort=store.cohort_identity(source),
        source_hashes={str(k): v for k, v in source.source_hashes.items()},
    )
    with store.lock(run):
        path = run / "registration.json"
        if path.is_file():
            if store.read_json(path) != registration:
                raise ValueError("V9 registration differs from existing checkpoints")
        else:
            if any(run.glob("r*_f*")) or (run / "COMPLETE.json").exists():
                raise ValueError("Existing checkpoint has no V9 registration")
            store.write_json(path, registration)
        if (run / "COMPLETE.json").is_file():
            return completed(run, release)
        folds = {}
        for r in (0, 1):
            for f in range(5):
                if (run / f"r{r}_f{f}").is_dir():
                    folds[r, f] = store.read_fold(run, r, f)
                    emit("REUSED", dict(repeat=r, fold=f))
                else:
                    start = time.monotonic()
                    emit("START", dict(repeat=r, fold=f))
                    arrays, meta = cv.fold_predictions(source, r, f)
                    source.verify_sources()
                    store.save_fold(run, r, f, arrays, meta)
                    folds[r, f] = arrays, meta
                    emit(
                        "OUTER_COMPLETE",
                        dict(
                            repeat=r,
                            fold=f,
                            people=meta["people"],
                            seconds=round(time.monotonic() - start, 1),
                        ),
                    )
        control = replay(folds, previous)
        emit("SCORING", dict(draws=draws, controls=control["status"]))
        result = report.summarize(source, folds, draws)
        result["v8_replay"] = control
        source.verify_sources()
        store.write_json(run / "summary.json", result)
        prior.write_csv(run / "all_routes.csv", result)
        prior.write_csv(run / "fixed_settings.csv", {"routes": result["fixed_settings"]})
        write_parents(run / "parent_comparisons.csv", result["parent_comparisons"])
        store.write_json(
            run / "COMPLETE.json",
            dict(version=VERSION, artifacts={name: store.digest(run / name) for name in ARTIFACTS}),
        )
        return completed(run, release)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--report", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    release = release_check()
    store.environment()
    import v9_report as report

    if args.report:
        if not (RUN / "COMPLETE.json").is_file():
            raise FileNotFoundError("No completed V9 run in " + str(RUN))
        report.compact(completed(RUN, release))
        return
    from adapters.person_inputs import load_discovery
    from v7_source import wrap_source
    from threadpoolctl import threadpool_limits

    plan = protocol()
    if store.read_json(ROOT / "design.json") != plan:
        raise ValueError("Stored design differs from executable model grid")
    emit(
        "DESIGN",
        dict(
            families=plan["family_count"],
            settings=plan["settings_count"],
            reported=len(plan["routes"]),
            primary=plan["primary"],
            target_pct=plan["threshold_pct"],
        ),
    )
    with threadpool_limits(limits=args.threads):
        if args.preflight:
            suite = unittest.defaultTestLoader.discover(str(ROOT), pattern="test_v9*.py")
            with open(os.devnull, "w") as mute, contextlib.redirect_stdout(mute):
                result = unittest.TextTestRunner(verbosity=1).run(suite)
            if not result.wasSuccessful() or not result.testsRun or result.skipped:
                raise RuntimeError("V9 synthetic preflight failed")
            emit("PREFLIGHT_DONE", dict(tests=result.testsRun, scope="generated only"))
        else:
            source = wrap_source(load_discovery(ROOT))
            emit("SOURCE_READY", store.cohort_identity(source))
            report.compact(
                execute(
                    source, RUN, release, args.threads, draws=plan["bootstrap_draws"], previous=prior.RUN
                )
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        emit("STOP", dict(error=type(exc).__name__, detail=str(exc)))
        raise
