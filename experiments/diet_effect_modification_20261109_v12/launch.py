"""Verify the prepared upload and synthetic checks, then enter a foreground worker."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from runtime import readiness, valid_id
from upload_check import verify

ROOT = Path(__file__).resolve().parent


def launch(args):
    if not sys.stdout.isatty() or not sys.stderr.isatty():
        raise RuntimeError("Keep this foreground server terminal open; do not redirect logs")
    valid_id(args.run_id)
    run = ROOT / "runs" / args.run_id
    if (run / "RUNNING.lock").exists():
        raise RuntimeError("Run lock exists; inspect its PID before any recovery")
    if run.exists() and not args.resume:
        raise FileExistsError("Run already exists; use --resume with identical settings or a new run ID")
    readiness(json.loads((ROOT / "config.json").read_text()), args.prior_summary_reviewed)
    print("UPLOAD_CHECK_OK files=%d" % verify(ROOT), flush=True)
    env = os.environ.copy()
    env.update({name: str(args.threads) for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")})
    subprocess.run([sys.executable, "-u", "check.py"], cwd=ROOT, env=env, check=True)
    command = [sys.executable, "-u", str(ROOT / "run.py"), "--run-id", args.run_id,
               "--threads", str(args.threads), "--prior-summary-reviewed"]
    if args.resume:
        command.append("--resume")
    if args.limit_outer:
        command += ["--limit-outer", str(args.limit_outer)]
    os.execve(sys.executable, command, env)


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
    launch(args)
