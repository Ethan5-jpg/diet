#!/usr/bin/env python3
"""Portable entry point. Raw sources are read-only; each run has its own output."""
import argparse
import copy
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))


def code_fingerprint():
    from hppmeal.common import sha256
    files = sorted((PACKAGE / "hppmeal").glob("*.py")) + sorted((PACKAGE / "scripts").glob("*.py"))
    return {str(p.relative_to(PACKAGE)): sha256(p) for p in files}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default_data = PACKAGE.parent if (PACKAGE.parent / "Transfer").is_dir() else PACKAGE.parent / "Data"
    parser.add_argument("--data-root", default=str(default_data), help="Server Data directory containing Transfer/")
    parser.add_argument("--config", default=str(PACKAGE / "config" / "default.json"))
    parser.add_argument("--stage", choices=["check", "audit", "build", "analyze", "all"], default="build")
    parser.add_argument("--output-root", default=str(PACKAGE / "outputs"))
    parser.add_argument("--run-dir", help="Resume a successful audit/build directory; never overwrite completed stages")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 8):
        print("Python 3.8 or newer is required; select your HPP analysis environment.", file=sys.stderr)
        return 2
    try:
        from hppmeal.common import Run, acquire_run_lock, check_inputs_unchanged, check_source_inventory, check_source_stats, require, sha256, validate_config, write_json
    except ImportError as e:
        print("Dependency import failed: {}\nUse the HPP Python environment or install with:\n{} -m pip install -r {}".format(
            e, sys.executable, PACKAGE / "requirements.txt"), file=sys.stderr)
        return 2
    from hppmeal.report import summary
    config = validate_config(json.loads(Path(args.config).read_text(encoding="utf-8")))
    data_root = Path(args.data_root).resolve()
    require(data_root.is_dir(), "Data root does not exist: " + str(data_root))
    marker = data_root / "SYNTHETIC_TEST_ONLY.json"
    require(not marker.exists() or config["data_kind"] == "SYNTHETIC", "Synthetic fixtures cannot be run with REAL config")
    new = not args.run_dir
    require(new or args.stage != "check", "check uses a fresh output directory; omit --run-dir")
    if new:
        require(args.stage != "analyze", "analyze requires --run-dir from a successful build")
        out = Path(args.output_root) / datetime.now(timezone.utc).strftime("meal_%Y%m%dT%H%M%S_%fZ")
        out.mkdir(parents=True, exist_ok=False)
    else:
        out = Path(args.run_dir).resolve()
        require((out / "manifest.json").is_file(), "Resume manifest missing")
    try:
        lock = acquire_run_lock(out)
    except ValueError as e:
        print("REQUEST REJECTED: " + str(e), file=sys.stderr)
        return 2
    resume_verified = new
    active = None
    run = None

    def begin_stage(name):
        nonlocal active
        active = name
        run.status[name] = dict(status="RUNNING")
        run.save()
        summary(run)

    def finish_stage(name, result):
        # Keep 'active' until the final checks and summary succeed. A late error
        # must invalidate the attempted stage even after its computation returned.
        run.status[name] = result
        check_source_stats(run)
        run.save()
        summary(run)

    try:
        run = Run(data_root, out, config)
        fingerprint = code_fingerprint()
        if not new:
            old = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            require(not old["config"]["time"]["consumption_time_verified"] or config["time"]["consumption_time_verified"],
                    "Cannot revoke timestamp verification in an existing run; use a new run")
            old_config, current = copy.deepcopy(old["config"]), copy.deepcopy(config)
            old_config["time"].pop("consumption_time_verified")
            current["time"].pop("consumption_time_verified")
            require(old_config == current, "Config changed beyond time verification; start a new run")
            require(old["root"] == str(data_root), "Resume data root mismatch")
            require(json.loads((out / "code_fingerprint.json").read_text(encoding="utf-8")) == fingerprint, "Code changed; start a new run")
            check_inputs_unchanged(old["inputs"])
            run.inputs, run.status = old["inputs"], old["status"]
            check_source_inventory(run)
            if args.stage in ["build", "all"]:
                require("build" not in run.status, "Build already attempted; analyze that run or start a new run")
            if args.stage == "analyze":
                require(run.status.get("build", {}).get("status") == "PASS", "Successful build required")
                require(run.status.get("analyze", {}).get("status") in [None, "REVIEW_REQUIRED"],
                        "Models already attempted; start a new run for revised analysis")
                require(sha256(out / "private" / "02_meals.csv") == run.status["build"]["meals_sha256"],
                        "Meal table changed after build")
            if "audit" in run.status:
                require(run.status["audit"]["status"] == "PASS", "Audit failed; start a new run after fixing source/config")
            resume_verified = True
        write_json(out / "effective_config.json", config)
        write_json(out / "code_fingerprint.json", fingerprint)
        print("RUN_DIR=" + str(out), flush=True)
        if "audit" not in run.status:
            from hppmeal.preflight import check
            begin_stage("check")
            finish_stage("check", check(run))
            if args.stage == "check" or run.status["check"]["status"] != "PASS":
                summary(run)
                print("CHECK={}\nSUMMARY={}".format(run.status["check"]["status"], out / "SUMMARY.md"), flush=True)
                return 0 if run.status["check"]["status"] == "PASS" else 2
        if "audit" not in run.status:
            from hppmeal.ingest import audit
            begin_stage("audit")
            result = audit(run)
            result["cache_sha256"] = sha256(out / "private" / "cache.sqlite")
            finish_stage("audit", result)
        require(run.status["audit"]["status"] == "PASS", "Audit failed; start a new run after fixing source/config")
        meals = None
        if args.stage in ["build", "all"]:
            require("build" not in run.status, "Build already attempted; analyze that run or start a new run")
            from hppmeal.pipeline import build
            begin_stage("build")
            result, meals = build(run)
            finish_stage("build", result)
        if args.stage in ["analyze", "all"]:
            require(run.status.get("build", {}).get("status") == "PASS", "Successful build required")
            require(run.status.get("analyze", {}).get("status") in [None, "REVIEW_REQUIRED"],
                    "Models already attempted; start a new run for revised analysis")
            require(sha256(out / "private" / "02_meals.csv") == run.status["build"]["meals_sha256"], "Meal table changed after build")
            if meals is None:
                import pandas as pd
                meals = pd.read_csv(out / "private" / "02_meals.csv", dtype={k: str for k in ["participant_id", "cohort", "research_stage", "connection_id"]})
            from hppmeal.models import analyze
            begin_stage("analyze")
            finish_stage("analyze", analyze(run, meals))
        check_source_stats(run)
        summary(run)
        print("SUMMARY=" + str(out / "SUMMARY.md"), flush=True)
        for stage, status in run.status.items():
            print(stage.upper() + "=" + status["status"], flush=True)
        return 2 if any(x["status"] not in ["PASS"] for x in run.status.values()) else 0
    except (Exception, KeyboardInterrupt) as e:
        interrupted = isinstance(e, KeyboardInterrupt)
        reason = "Interrupted by user; incomplete stage; start a new run" if interrupted else str(e)
        if run is None:
            print("OUTPUT SETUP FAILED: " + reason, file=sys.stderr)
            return 130 if interrupted else 2
        if not resume_verified:
            print("RESUME REJECTED (existing run unchanged): " + reason, file=sys.stderr)
            return 130 if interrupted else 2
        if active is None and not new:
            print("REQUEST REJECTED (completed stages unchanged): " + reason, file=sys.stderr)
            return 130 if interrupted else 2
        active = active or "request"
        run.status[active] = dict(status="INTERRUPTED" if interrupted else "FAILED", reason=reason)
        (out / "logs" / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        run.save()
        summary(run)
        print("FAILED: {}\nSee {}".format(reason, out / "SUMMARY.md"), file=sys.stderr)
        return 130 if interrupted else 2
    finally:
        lock.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print("CONFIG/INPUT ERROR: " + str(e), file=sys.stderr)
        sys.exit(2)
