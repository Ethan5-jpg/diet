"""V10 paired interaction discovery run. Work is restartable at completed outer folds."""

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parent
for n in ("v6", "v7", "v8", "v9"):
    sys.path.insert(1, str(ROOT.parent / n))
import run_v6 as store
import run_v9 as prior
from run_v8 import write_csv
import v10_grid as grid

VERSION = "v10"
RUN = ROOT / "runs" / "TRE_ONLY" / VERSION
ARTIFACTS = ("registration.json", "summary.json", "fixed_settings.csv", "workflows.csv", "vs_v9.csv")
FILES = (
    "README.md",
    "design.json",
    "run_all.sh",
    "server_run.sh",
    "run_v10.py",
    "v10_grid.py",
    "v10_features.py",
    "v10_fit.py",
    "v10_cv.py",
    "v10_report.py",
    "v10_audit.py",
    "test_v10_fit.py",
    "test_v10_inference.py",
    "test_v10_audit.py",
)
REPLAY = {
    grid.ANCHOR: grid.ANCHOR,
    "legacy_joint_surface": "standalone_score_surface_joint",
    "legacy_shannon_surface": "standalone_score_surface_shannon",
}


def emit(n, x):
    store.emit("EXP2_V10_" + n, x)


def protocol():
    from v8_selection import SECOND_INNER_SEED

    return json.loads(
        json.dumps(
            dict(
                version=VERSION,
                **grid.design(),
                population="unchanged V11 evaluated discovery people",
                outcome="D_Glu (mg/dL), archived training-role preprocessing",
                exposure="fiber_density (g/1000 kcal)",
                outcome_scope="discovery only",
                objective="100*(R2_on-R2_off)/R2_on; min across repeats",
                per_model_additive_pair=True,
                independent_point_check="compensated sums of squared errors of held-out people; named baseline per setting",
                strict_threshold="positive unrounded 100*delta_R2 - target*R2_on, beyond floating-point roundoff; require same key twice",
                adaptive_population="intersection of eligible outer-training validation rows in both inner repartitions",
                outer_repeats=2,
                outer_folds=5,
                inner_repartitions=2,
                inner_folds=3,
                second_inner_seed=SECOND_INNER_SEED,
                bootstrap_draws=1999,
                bootstrap_seed=grid.SEED,
                simultaneous_scope="all fixed settings and adaptive procedures across both repeats",
                previous_version=prior.VERSION,
                exact_replay=REPLAY,
                replay_penalties=grid.LEGACY_PENALTIES,
            ),
            allow_nan=False,
        )
    )


def release_check():
    release = store.read_json(ROOT / "upload_manifest.json")
    if release.get("version") != VERSION or set(release.get("files", {})) != set(FILES):
        raise ValueError("Incomplete V10 release manifest")
    for name, digest in release["files"].items():
        if store.digest(ROOT / name) != digest:
            raise ValueError("Changed V10 upload file: " + name)
    if store.digest(ROOT.parent / "v9/upload_manifest.json") != release["v9_manifest_sha256"]:
        raise ValueError("V9 release differs from the dependency registered for v10")
    prior.release_check()
    if store.read_json(ROOT / "design.json") != protocol():
        raise ValueError("Declared and executable V10 designs differ")
    return release


def replay(folds, previous):
    import numpy as np
    import v9_cv as old_cv
    import v9_grid as old

    if previous is None or not (previous / "COMPLETE.json").is_file():
        return dict(status="V9_archive_unavailable_controls_refit", folds_checked=0)
    receipt = previous / "COMPLETE.json"
    stamp = store.digest(receipt)
    prior.completed(previous, store.read_json(ROOT.parent / "v9/upload_manifest.json"))
    max_abs = np.zeros(2)
    for (r, f), (a, m) in sorted(folds.items()):
        b, _ = store.read_fold(previous, r, f)
        if any(not np.array_equal(a[k], b[k]) for k in ("rows", "y", "groups")):
            raise ValueError(f"V9 replay differs in rows, groups or outcome r{r} f{f}")
        for name, old_name in REPLAY.items():
            new = [grid.INDEX[grid.key(name, p)] for p in grid.LEGACY_PENALTIES]
            previous_ix = [old_cv.INDEX[grid.key(old_name, p)] for p in grid.LEGACY_PENALTIES]
            errors = [
                np.max(
                    abs(
                        a["setting_off"][:, grid.FAMILY_INDEX[name]]
                        - b["setting_off"][:, old.FAMILIES.index(old_name)]
                    )
                ),
                np.max(abs(a["setting_on"][:, new] - b["setting_on"][:, previous_ix])),
            ]
            if not np.isfinite(errors).all() or max(errors) > 2e-8:
                raise ValueError(f"Archived V9 prediction mismatch {name} r{r} f{f}: {errors}")
            max_abs = np.maximum(max_abs, errors)
    if store.digest(receipt) != stamp:
        raise RuntimeError("V9 receipt changed during replay")
    return dict(
        status="V9_predictions_match",
        folds_checked=len(folds),
        paired_settings=len(REPLAY) * len(grid.LEGACY_PENALTIES),
        max_abs_off_on_mg_dl=max_abs.tolist(),
        tolerance_mg_dl=2e-8,
        receipt_sha256=stamp,
    )


def completed(run, release):
    receipt = store.read_json(run / "COMPLETE.json")
    if receipt.get("version") != VERSION or set(receipt.get("artifacts", {})) != set(ARTIFACTS):
        raise ValueError("Invalid V10 completion receipt")
    for name, digest in receipt["artifacts"].items():
        if store.digest(run / name) != digest:
            raise ValueError("Completed aggregate changed: " + name)
    reg = store.read_json(run / "registration.json")
    if reg["release"] != release or reg["protocol"] != protocol():
        raise ValueError("Completed run differs from uploaded release")
    return store.read_json(run / "summary.json")


def execute(source, run, release, threads, draws, previous):
    import v10_cv as cv
    import v10_report as report

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
        if path.exists():
            if store.read_json(path) != registration:
                raise ValueError("Existing V10 checkpoint registration differs")
        else:
            if any(run.glob("r*_f*")) or (run / "COMPLETE.json").exists():
                raise ValueError("Existing run has no matching registration")
            store.write_json(path, registration)
        if (run / "COMPLETE.json").exists():
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
                    a, m = cv.fold_predictions(source, r, f)
                    source.verify_sources()
                    store.save_fold(run, r, f, a, m)
                    folds[r, f] = a, m
                    emit(
                        "OUTER_COMPLETE",
                        dict(
                            repeat=r, fold=f, people=m["people"], seconds=round(time.monotonic() - start, 1)
                        ),
                    )
        control = replay(folds, previous)
        emit("SCORING", dict(draws=draws, control=control["status"], fixed=len(grid.KEYS)))
        result = report.summarize(source, folds, draws)
        result["v9_replay"] = control
        source.verify_sources()
        store.write_json(run / "summary.json", result)
        for name, keys in (("fixed_settings.csv", grid.KEYS), ("workflows.csv", grid.ADAPTIVE)):
            write_csv(run / name, dict(routes={k: result["routes"][k] for k in keys}))
        prior.write_parents(run / "vs_v9.csv", result["vs_anchor"])
        store.write_json(
            run / "COMPLETE.json",
            dict(version=VERSION, artifacts={n: store.digest(run / n) for n in ARTIFACTS}),
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
    emit(
        "DESIGN",
        dict(
            families=len(grid.FAMILIES),
            fixed_settings=len(grid.KEYS),
            adaptive=len(grid.ADAPTIVE),
            target_pct=grid.TARGET,
            primary_fixed=grid.PRIMARY,
        ),
    )
    from threadpoolctl import threadpool_limits
    import v10_report as report

    if args.report:
        report.compact(completed(RUN, release))
        return
    with threadpool_limits(limits=args.threads):
        if args.preflight:
            suite = unittest.defaultTestLoader.discover(str(ROOT), pattern="test_v10*.py")
            with open(os.devnull, "w") as mute, contextlib.redirect_stdout(mute):
                result = unittest.TextTestRunner(verbosity=1).run(suite)
            if not result.wasSuccessful() or not result.testsRun or result.skipped:
                raise RuntimeError("V10 generated-data preflight failed")
            emit("PREFLIGHT_DONE", dict(tests=result.testsRun, scope="generated only"))
            return
        from adapters.person_inputs import load_discovery
        from v7_source import wrap_source

        source = wrap_source(load_discovery(ROOT))
        emit("SOURCE_READY", store.cohort_identity(source))
        report.compact(
            execute(source, RUN, release, args.threads, protocol()["bootstrap_draws"], prior.RUN)
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        emit("STOP", dict(error=type(err).__name__, detail=str(err)))
        raise
