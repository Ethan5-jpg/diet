"""Console launch must avoid old log handles and keep existing runs isolated."""

import contextlib
import errno
import json
import os
import pty
import select
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import launch_console


class ConsoleChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = (
            Path(self.temporary.name).resolve()
            / "collaborator_handoff_20261102_v1/experiments/batch"
        )
        self.root.mkdir(parents=True)
        self.args = SimpleNamespace(run_id="console_test", threads=4, resume=False)
        self.context = contextlib.ExitStack()
        self.addCleanup(self.context.close)
        self.context.enter_context(
            mock.patch.object(
                launch_console, "__file__", str(self.root / "launch_console.py")
            )
        )
        self.context.enter_context(
            mock.patch.object(sys.stdout, "isatty", return_value=True)
        )
        self.context.enter_context(
            mock.patch.object(sys.stderr, "isatty", return_value=True)
        )

    def test_exec_inherits_terminal_same_pid_and_avoids_old_blocked_log(self):
        source = Path(__file__).resolve().parents[1] / "launch_console.py"
        shutil.copy2(source, self.root / "launch_console.py")
        old = self.root / "runs/main_20261107"
        old.mkdir(parents=True)
        lock, checkpoint = old / "RUNNING.lock", old / "saved.pkl"
        lock.write_bytes(b"pid=24812\n")
        checkpoint.write_bytes(b"SYNTHETIC_OLD_CHECKPOINT")
        blocked = self.root / "main_20261107.log"
        os.mkfifo(blocked)
        (self.root / "test_suite.py").write_text(
            'from pathlib import Path\nPath("tests_ran").touch()\nprint("PREFLIGHT_OK")\n'
        )
        (self.root / "run.py").write_text(
            "import json, os, sys\nfrom pathlib import Path\n"
            'assert Path("tests_ran").exists()\n'
            "record=dict(pid=os.getpid(), argv=sys.argv[1:], "
            "stdout_tty=os.isatty(1), stderr_tty=os.isatty(2), "
            "threads=[os.environ[k] for k in "
            '("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")])\n'
            'Path("worker.json").write_text(json.dumps(record))\n'
            'print("CONSOLE_WORKER_DONE")\n'
        )
        master, slave = pty.openpty()
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    str(self.root / "launch_console.py"),
                    "--run-id",
                    self.args.run_id,
                ],
                cwd=self.temporary.name,
                stdin=subprocess.DEVNULL,
                stdout=slave,
                stderr=slave,
            )
            output = bytearray()
            try:
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    ready, _, _ = select.select([master], [], [], 0.1)
                    if not ready:
                        if process.poll() is not None:
                            break
                        continue
                    try:
                        block = os.read(master, 65536)
                    except OSError as error:
                        if error.errno == errno.EIO:
                            break
                        raise
                    if not block:
                        break
                    output.extend(block)
                self.assertEqual(process.wait(timeout=1), 0, output.decode())
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        finally:
            os.close(master)
            if slave is not None:
                os.close(slave)
        self.assertIn(b"PREFLIGHT_OK", output)
        self.assertIn(b"CONSOLE_WORKER_DONE", output)
        record = json.loads((self.root / "worker.json").read_text())
        self.assertEqual(record["pid"], process.pid)
        self.assertTrue(record["stdout_tty"] and record["stderr_tty"])
        self.assertEqual(record["threads"], ["4"] * 3)
        self.assertEqual(record["argv"], ["--run-id", "console_test", "--threads", "4"])
        self.assertEqual(lock.read_bytes(), b"pid=24812\n")
        self.assertEqual(checkpoint.read_bytes(), b"SYNTHETIC_OLD_CHECKPOINT")
        self.assertTrue(stat.S_ISFIFO(blocked.stat().st_mode))
        self.assertEqual(list(self.root.glob("*.log")), [blocked])

    def test_redirected_output_is_rejected_before_preflight(self):
        with (
            mock.patch.object(sys.stdout, "isatty", return_value=False),
            mock.patch.object(launch_console.subprocess, "run") as preflight,
        ):
            with self.assertRaisesRegex(ValueError, "terminal"):
                launch_console.launch(self.args)
            preflight.assert_not_called()

    def test_existing_run_or_lock_is_never_bypassed(self):
        run = self.root / "runs" / self.args.run_id
        run.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "fresh run-id"):
            launch_console.launch(self.args)
        self.args.resume = True
        lock = run / "RUNNING.lock"
        lock.write_text("pid=24812")
        with self.assertRaisesRegex(ValueError, "Existing run lock"):
            launch_console.launch(self.args)
        self.assertEqual(lock.read_text(), "pid=24812")

    def test_failed_preflight_never_starts_worker(self):
        with (
            mock.patch.object(
                launch_console.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(1, "tests"),
            ),
            mock.patch.object(launch_console.os, "execve") as execute,
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                launch_console.launch(self.args)
            execute.assert_not_called()
        self.assertFalse((self.root / "runs").exists())

    def test_resume_passes_original_worker_arguments_without_creating_log(self):
        run = self.root / "runs" / self.args.run_id
        run.mkdir(parents=True)
        self.args.resume = True
        with (
            mock.patch.object(launch_console.subprocess, "run") as preflight,
            mock.patch.object(launch_console.os, "chdir") as change,
            mock.patch.object(launch_console.os, "execve") as execute,
            mock.patch("builtins.print"),
        ):
            launch_console.launch(self.args)
            self.assertEqual(preflight.call_args.kwargs["cwd"], self.root)
            self.assertTrue(preflight.call_args.kwargs["check"])
            change.assert_called_once_with(self.root)
            command = execute.call_args.args[1]
            self.assertEqual(command[-1], "--resume")
            self.assertEqual(command[command.index("--threads") + 1], "4")
        self.assertFalse(list(self.root.glob("*.log")))

    def test_invalid_arguments_location_and_symlink_are_rejected(self):
        for name, threads in (("../old", 4), ("ok", 0), ("", 4)):
            with self.subTest(name=name, threads=threads):
                args = SimpleNamespace(run_id=name, threads=threads, resume=False)
                with self.assertRaisesRegex(ValueError, "Invalid"):
                    launch_console.launch(args)
        with mock.patch.object(
            launch_console, "__file__", str(Path(self.temporary.name) / "wrong.py")
        ):
            with self.assertRaisesRegex(ValueError, "Upload"):
                launch_console.launch(self.args)
        run = self.root / "runs" / self.args.run_id
        run.parent.mkdir()
        run.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            launch_console.launch(self.args)
