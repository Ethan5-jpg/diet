"""Run synthetic checks, then start one experiment in a persistent tmux terminal."""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from runtime import valid_id

ROOT = Path(__file__).resolve().parent


def command(args):
    value = [
        sys.executable,
        "-u",
        str(ROOT / "run.py"),
        "--run-id",
        args.run_id,
        "--threads",
        str(args.threads),
    ]
    if args.resume:
        value.append("--resume")
    return value


def launch(args):
    valid_id(args.run_id)
    run = ROOT / "runs" / args.run_id
    if (run / "RUNNING.lock").exists():
        raise RuntimeError("Existing RUNNING.lock: inspect its worker before recovery")
    if run.exists() and not args.resume:
        raise FileExistsError("Run already exists; choose --resume or a new run-id")
    if args.resume and not (run / "registration.json").is_file():
        raise FileNotFoundError("No registration exists for this resume")
    session = "hpp_v7_" + args.run_id
    if not args.console:
        if shutil.which("tmux") is None:
            raise RuntimeError("tmux is required for browser disconnects: sudo yum install -y tmux")
        if (
            subprocess.run(
                ["tmux", "has-session", "-t", session], capture_output=True, check=False
            ).returncode
            == 0
        ):
            raise RuntimeError("tmux session already exists; attach it instead of starting another")
    environment = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[key] = str(args.threads)
    subprocess.run([sys.executable, str(ROOT / "check.py")], cwd=ROOT, env=environment, check=True)
    worker = command(args)
    if args.console:
        if not sys.stdout.isatty() or not sys.stderr.isatty():
            raise RuntimeError("--console requires an interactive terminal")
        os.chdir(ROOT)
        os.execve(sys.executable, worker, environment)
    shell_command = (
        shlex.join(["cd", str(ROOT)])
        + " && "
        + shlex.join(worker)
        + '; printf "\\nWORKER_EXIT=%s\\n" "$?"; exec bash --noprofile --norc'
    )
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, shell_command],
        env=environment,
        check=True,
    )
    print("STARTED tmux session=" + session)
    print("Attach: tmux attach -t " + session)
    print("Detach: Ctrl+b, then d. Browser disconnects preserve this session.")
    print("Report: .venv/bin/python report.py --run-id " + args.run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="main_v7_20261107")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--console",
        action="store_true",
        help="Foreground only; browser disconnect safety is not guaranteed",
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Positive threads required")
    launch(args)
