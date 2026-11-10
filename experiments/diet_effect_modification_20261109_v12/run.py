"""Native discovery execution. Run only after V11 and the prior-summary review."""

import argparse
import json
import signal
import sys
import time
from pathlib import Path

from threadpoolctl import threadpool_limits

from bridge import Memo, load, sha
from engine import Engine
from report import aggregate, completed_results, screen
from runtime import audit, check_config, environment, identity, publish_json, readiness, register, run_lock, valid_id

ROOT = Path(__file__).resolve().parent


def log(message):
    print(time.strftime("%H:%M:%S ") + message, flush=True)


def execute(args):
    package = ROOT.parent.parent
    if package.name != "collaborator_handoff_20261102_v1" or ROOT.parent.name != "experiments":
        raise ValueError("Native run requires the established server handoff/experiments directory")
    if not sys.stdout.isatty() or not sys.stderr.isatty():
        raise RuntimeError("Use the foreground server terminal; do not redirect mounted-drive logs")
    config = json.loads((ROOT / "config.json").read_text())
    check_config(config)
    gate = readiness(config, args.prior_summary_reviewed)
    versions = environment(config)
    run = ROOT / "runs" / valid_id(args.run_id)
    if (run / "RUNNING.lock").exists():
        raise RuntimeError("Run lock exists; inspect recorded PID before recovery")
    if run.exists() and not args.resume:
        raise FileExistsError("Run exists; use --resume with identical settings or a new run ID")
    log("LOAD existing discovery-only sources; V11 completion gate passed")
    dataset = load(package, config)
    current = identity(ROOT, config, dataset, versions, gate, args.threads)
    register(run, current, args.resume)
    with run_lock(run):
        memo = Memo(run / "TRE_ONLY")
        engine, new_units = Engine(config, memo, log), 0
        for repeat in range(2):
            for fold in range(5):
                before = memo.created
                engine.checkpoint(dataset, repeat, fold)
                new_units += memo.created > before
                log("OUTER_COMPLETE r%d f%d" % (repeat, fold))
                if args.limit_outer and new_units >= args.limit_outer:
                    log("REQUESTED_OUTER_LIMIT_REACHED; use --resume later")
                    return
        log("FINAL_SOURCE_INPUT_AUDIT")
        audit(current)
        result = aggregate(completed_results(run, config), config,
                           dataset.index.split_unit_id.astype(str).to_numpy(), "DISCOVERY_SERVER")
        if result["n"] != config["expected_people"]:
            raise ValueError("Final participant count differs")
        publish_json(run / "summary.json", result)
        files = {}
        for folder in sorted((run / "TRE_ONLY").glob("r*_f*_COMPLETE")):
            for path in sorted(folder.glob("*")):
                if path.is_file():
                    files[str(path.relative_to(run / "TRE_ONLY"))] = sha(path)
        publish_json(run / "completion.json", dict(
            summary_sha256=sha(run / "summary.json"), registration_sha256=sha(run / "registration.json"),
            outer_folds_complete=10, checkpoint_files=files, final_source_input_audit="PASS",
        ))
        print(screen(result, 1), end="", flush=True)
        log("BATCH_COMPLETE; server_report.sh --screen 2 and --screen 3 show remaining aggregate screens")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="effect_modification_v12_20261109")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--prior-summary-reviewed", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-outer", type=int, default=0)
    args = parser.parse_args()
    if args.threads < 1 or args.limit_outer < 0:
        parser.error("Positive threads and nonnegative outer limit required")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit("SIGTERM: completed units retained; inspect run lock"))
    with threadpool_limits(limits=args.threads):
        execute(args)
