"""Test once, then start the full batch in the background without overwriting logs."""

import argparse
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path


def launch(args):
    root = Path(__file__).resolve().parent
    if (
        root.parent.parent.name != "collaborator_handoff_20261102_v1"
        or root.parent.name != "experiments"
    ):
        raise ValueError(
            "Upload the complete batch folder into the handoff experiments/ directory"
        )
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.run_id)
        or args.threads < 1
    ):
        raise ValueError("Invalid run-id or threads")
    run = root / "runs" / args.run_id
    if run.exists() != args.resume:
        raise ValueError(
            "Use --resume for an existing run, or a fresh run-id for a new run"
        )
    if (run / "RUNNING.lock").exists():
        raise ValueError(
            "Existing run lock: verify the previous process has stopped before resuming"
        )
    name = (
        args.run_id
        + ("_resume_" + uuid.uuid4().hex[:8] if args.resume else "")
        + ".log"
    )
    log = root / name
    if log.exists():
        raise FileExistsError("Existing log retained: " + str(log))
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
        "run.py",
        "--run-id",
        args.run_id,
        "--threads",
        str(args.threads),
    ]
    if args.resume:
        command.append("--resume")
    with log.open("xb") as stream:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print("Started batch PID=%d; log=%s" % (process.pid, log.name), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="main_20261107")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    launch(parser.parse_args())
