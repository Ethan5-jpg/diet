"""Run V11 inside the handoff experiments folder with TTY output, never a log file."""

import os

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import argparse
import json
import signal
import sys
import time
from pathlib import Path

from threadpoolctl import threadpool_limits

from bridge import Memo, Reference, load
from engine import Engine
from report import aggregate, compact, completed_results
from runtime import (
    audit,
    check_config,
    environment,
    identity,
    publish_json,
    publish_text,
    register,
    run_lock,
    valid_id,
)

ROOT = Path(__file__).resolve().parent


def log(message):
    print(time.strftime("%H:%M:%S ") + message, flush=True)


def execute(args):
    package = ROOT.parent.parent
    if package.name != "collaborator_handoff_20261102_v1" or ROOT.parent.name != "experiments":
        raise ValueError("Upload V11 under collaborator_handoff_20261102_v1/experiments/")
    if not sys.stdout.isatty() or not sys.stderr.isatty():
        raise RuntimeError("A terminal is required. Use launch.py --console; do not redirect logs on FUSE")
    config = json.loads((ROOT / "config.json").read_text())
    check_config(config)
    versions = environment(config)
    run = ROOT / "runs" / valid_id(args.run_id)
    if run.exists() and not args.resume:
        raise FileExistsError(
            "Run already exists; use --resume for identical code/data or choose a new run-id"
        )
    if (run / "RUNNING.lock").exists():
        raise RuntimeError("Existing run lock; inspect worker before retrying")
    log("LOAD discovery inputs and joint-day distributions")
    dataset = load(package)
    reference = Reference(package, config, dataset, log=log)
    note = dataset.extra["audit"]
    log(
        "DISTRIBUTION fixed_N=%d energy_days=%d joint_macro_ge3=%d bag_days=%d; no new outcomes"
        % (
            note["discovery_people"],
            note["energy_days"],
            note["joint_macro_people_ge3"],
            note["joint_distribution_days"],
        )
    )
    log("HASH sources, inputs and references; no confirmation data")
    current = identity(ROOT, config, dataset, reference, versions, args.threads)
    register(run, current, args.resume)
    with run_lock(run):
        memo = Memo(
            run / "TRE_ONLY",
            log=lambda message: log(message) if not message.startswith("DONE ") else None,
        )
        total_started, new_count = time.monotonic(), 0
        for repeat in range(config["outer_repeats"]):
            for fold in range(config["outer_folds"]):
                engine, before = Engine(config, memo, args.threads, reference, log), memo.created
                memo.get(
                    "r%d_f%d_result" % (repeat, fold),
                    lambda engine=engine, repeat=repeat, fold=fold: engine.outer(dataset, repeat, fold),
                )
                log(
                    "OUTER_COMPLETE r%d f%d total_elapsed=%.1fmin"
                    % (repeat, fold, (time.monotonic() - total_started) / 60)
                )
                new_count += memo.created > before
                if args.limit_outer and new_count >= args.limit_outer:
                    log("STOPPED_AT_REQUESTED_OUTER_LIMIT; rerun with --resume")
                    return
        log("FINAL_HASH_AUDIT")
        audit(current)
        result = aggregate(config, completed_results(run, config))
        publish_json(run / "summary.json", result)
        publish_text(run / "SUMMARY.txt", compact(result, config))
        print(compact(result, config), end="", flush=True)
        log("BATCH_COMPLETE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="distribution_v11_20261109")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-outer", type=int, default=0)
    args = parser.parse_args()
    if args.threads < 1 or args.limit_outer < 0:
        parser.error("Positive threads and nonnegative limit-outer required")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit("Worker received SIGTERM; completed units retained"))
    with threadpool_limits(limits=args.threads):
        execute(args)
