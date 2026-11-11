"""Experiment2: preflight, train, resume, or print aggregate reports."""

import argparse
import os
import platform
import re
import sys

from exp2_inputs import (ROOT, adapters, environment, fingerprint, native_paths,
                         preflight, source_state, validate_population)
from exp2_store import (load_unit, locked, publish, read_json, read_summary,
                        recover_stale, register, save_unit, unit_path)


def output(message):
    print(message, flush=True)


def audit_sources():
    output("EXP2_SOURCE_AUDIT | discovery allowlist, no confirmation outcomes")
    paths = native_paths()
    state = source_state(paths)
    hashes = fingerprint(paths)
    verify_sources(paths, state)
    return paths, hashes, state


def verify_sources(paths, expected):
    if source_state(paths) != expected:
        raise ValueError("Discovery inputs changed during the run")


def verify_release(code_id):
    _, current, _ = preflight()
    if current != code_id:
        raise ValueError("Upload manifest changed during the run")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--run-id")
    parser.add_argument("--preflight", action="store_true", help="No HPP data or output writes")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--recover-stale-lock", action="store_true")
    parser.add_argument("--limit-outer", type=int, default=0, help="Stop cleanly after N new folds")
    parser.add_argument("--screen", type=int, choices=range(1, 11))
    args = parser.parse_args()
    if not (1 <= args.threads <= 16) or args.limit_outer < 0:
        parser.error("Threads must be 1..16 and --limit-outer nonnegative")
    if args.recover_stale_lock and (not args.resume or args.preflight or args.screen):
        parser.error("--recover-stale-lock is only for resuming a failed worker")
    if (args.preflight or args.screen) and (args.resume or args.limit_outer or (args.preflight and args.screen)):
        parser.error("Choose exactly one of preflight, report, or run")
    if args.screen:
        cfg = read_json(ROOT / "config.json")
    else:
        file_count, code_id, cfg = preflight()
        output("EXP2_UPLOAD_CHECK_OK files=%d" % file_count)
    args.run_id = args.run_id or cfg["version"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_id):
        parser.error("Invalid run ID")
    run = ROOT / "runs_TRE_ONLY" / args.run_id
    if args.screen:
        from exp2_report import display
        display(read_summary(run), args.screen)
        return
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    versions = environment(cfg)
    output("EXP2_ENV_OK " + " ".join("%s=%s" % (k, v) for k, v in versions.items()))
    bridge, features = adapters()
    if args.preflight:
        from exp2_test import quick
        quick()
        from exp2_models import specifications
        output("EXP2_CONFIG %s routes=%d" % (cfg["version"], len(specifications(cfg))))
        output("EXP2_PREFLIGHT_DONE | synthetic checks only; no native rows read or writes")
        return

    from threadpoolctl import threadpool_limits
    from exp2_engine import Engine
    from exp2_report import display
    from exp2_score import summarize, validate_fold

    if args.resume:
        if not run.is_dir():
            raise FileNotFoundError("Cannot resume a run that does not exist")
        if args.recover_stale_lock:
            recover_stale(run)
    else:
        # A missing or invalid discovery source must not reserve a fresh run ID.
        paths, hashes, state = audit_sources()
        run.mkdir(parents=True, exist_ok=False, mode=0o700)
    with locked(run), threadpool_limits(limits=args.threads):
        if args.resume:
            paths, hashes, state = audit_sources()
        verify_release(code_id)
        schedule = [(r, f) for r in range(cfg["outer_repeats"]) for f in range(cfg["outer_folds"])]
        reg = register(run, dict(code=code_id, sources=hashes, threads=args.threads,
                                 python=platform.python_version(), versions=versions,
                                 schedule=[list(k) for k in schedule]), args.resume)
        if (run / "COMPLETE.json").exists():
            display(read_summary(run))
            return
        from exp2_inputs import PACKAGE
        dataset = bridge.load(PACKAGE, cfg)
        if set(dataset.paths) != set(paths):
            raise ValueError("Adapter returned a different source allowlist")
        validate_population(dataset, cfg)
        verify_sources(paths, state)
        # The shared adapter loads two old score columns with the reference archive.
        # Remove those immediately: only y and household assignments enter the experiment.
        dataset.reference = {key: dataset.reference[key] for key in ("y", "outer")}
        output("EXP2_SOURCE_READY discovery=%d people=%d households=%d" %
               (len(dataset.index), cfg["expected_people"], cfg["expected_households"]))
        engine, units, fresh = Engine(cfg, features, output), {}, 0
        for repeat, fold in schedule:
            path = unit_path(run, repeat, fold)
            if path.exists():
                units[(repeat, fold)] = load_unit(run, repeat, fold, reg)
                validate_fold(dataset, engine, repeat, fold, units[(repeat, fold)][0])
                output("RESUMED r%d f%d" % (repeat, fold))
                continue
            if args.limit_outer and fresh >= args.limit_outer:
                continue
            arrays, meta = engine.run_unit(dataset, repeat, fold)
            validate_fold(dataset, engine, repeat, fold, arrays)
            verify_sources(paths, state)
            verify_release(code_id)
            save_unit(run, repeat, fold, reg, arrays, meta)
            units[(repeat, fold)] = load_unit(run, repeat, fold, reg)
            fresh += 1
            output("OUTER_COMPLETE r%d f%d people=%d seconds=%.1f" %
                   (repeat, fold, len(arrays["rows"]), meta["seconds"]))
        if hashes != fingerprint(paths):
            raise ValueError("Discovery inputs changed during the run")
        verify_sources(paths, state)
        verify_release(code_id)
        if len(units) != len(schedule):
            output("EXP2_PAUSED %d/%d folds committed; resume with same run ID and threads." %
                   (len(units), len(schedule)))
            return
        summary = summarize(dataset, engine, units)
        verify_sources(paths, state)
        verify_release(code_id)
        publish(run, summary, schedule, reg)
        verified = read_summary(run)
        output("EXP2_COMPLETE | all folds verified; use --screen 9 for the new routes")
        display(verified)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print("EXP2_STOP %s: %s" % (type(exc).__name__, exc), file=sys.stderr, flush=True)
        sys.exit(1)
