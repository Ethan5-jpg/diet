"""Synthetic preflight followed by foreground execution in the current terminal."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from runtime import valid_id
from upload_check import verify

ROOT = Path(__file__).resolve().parent


def launch(args):
    if not sys.stdout.isatty() or not sys.stderr.isatty():
        raise RuntimeError("Open a terminal and use launch.py --console; do not redirect output")
    valid_id(args.run_id)
    print("UPLOAD_CHECK_OK files=%d" % verify(ROOT), flush=True)
    run = ROOT / "runs" / args.run_id
    if (run / "RUNNING.lock").exists():
        raise RuntimeError("Run lock exists; inspect its PID before any recovery")
    if run.exists() and not args.resume:
        raise FileExistsError("Run already exists; use --resume or a new run-id")
    env = os.environ.copy()
    env.update(
        {key: str(args.threads) for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")}
    )
    subprocess.run([sys.executable, str(ROOT / "check.py")], cwd=ROOT, env=env, check=True)
    command = [
        sys.executable,
        "-u",
        str(ROOT / "run.py"),
        "--run-id",
        args.run_id,
        "--threads",
        str(args.threads),
    ]
    if args.resume:
        command.append("--resume")
    print(
        "CONSOLE_WORKER pid=%d run=%s; do not interrupt this terminal" % (os.getpid(), args.run_id),
        flush=True,
    )
    os.execve(sys.executable, command, env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--console", action="store_true", help="Foreground execution (also the default)")
    parser.add_argument("--run-id", default="overnight_v8_20261108")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Positive threads required")
    launch(args)
