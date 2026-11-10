"""Test, then run V6 in this terminal without a persistent file-log writer."""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def launch(args):
    root = Path(__file__).resolve().parent
    if (
        root.parent.parent.name != "collaborator_handoff_20261102_v1"
        or root.parent.name != "experiments"
    ):
        raise ValueError("Upload this file into the original handoff experiment folder")
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.run_id)
        or args.threads < 1
    ):
        raise ValueError("Invalid run-id or threads")
    if not sys.stdout.isatty() or not sys.stderr.isatty():
        raise ValueError("Run directly in a terminal without file or pipe redirection")
    run = root / "runs" / args.run_id
    if run.is_symlink():
        raise ValueError("Run directory must not be a symlink")
    if (run / "RUNNING.lock").exists():
        raise ValueError("Existing run lock retained; use a fresh run-id")
    if run.exists() != args.resume:
        raise ValueError("Use --resume for an existing run, or a fresh run-id")
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        OPENBLAS_NUM_THREADS=str(args.threads),
        MKL_NUM_THREADS=str(args.threads),
        OMP_NUM_THREADS=str(args.threads),
    )
    subprocess.run(
        [sys.executable, "-B", "test_suite.py"], cwd=root, env=env, check=True
    )
    command = [
        sys.executable,
        "-u",
        "-B",
        "run.py",
        "--run-id",
        args.run_id,
        "--threads",
        str(args.threads),
    ]
    if args.resume:
        command.append("--resume")
    print(
        "CONSOLE_WORKER pid=%d run=%s; keep this terminal open"
        % (os.getpid(), args.run_id),
        flush=True,
    )
    os.chdir(root)
    os.execve(sys.executable, command, env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="console_20261107")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    launch(parser.parse_args())
