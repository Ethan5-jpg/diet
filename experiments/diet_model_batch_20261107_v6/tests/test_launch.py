"""Detached launch and log protection, using an empty synthetic worker."""

import contextlib
import io
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import launch


class LaunchChecks(unittest.TestCase):
    def test_detached_start_preserves_existing_log(self):
        with tempfile.TemporaryDirectory() as name:
            root = (
                Path(name).resolve()
                / "collaborator_handoff_20261102_v1/experiments/batch"
            )
            root.mkdir(parents=True)
            (root / "test_suite.py").write_text(
                'from pathlib import Path\nPath("tests_ran").write_text("yes")\n'
            )
            (root / "run.py").write_text(
                'from pathlib import Path\nassert Path("tests_ran").exists()\nprint("SYNTHETIC_WORKER_DONE", flush=True)\nPath("worker_done").write_text("yes")\n'
            )
            args = SimpleNamespace(run_id="test", threads=1, resume=False)
            with (
                mock.patch.object(launch, "__file__", str(root / "launch.py")),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                launch.launch(args)
                deadline = time.monotonic() + 5
                while (
                    not (root / "worker_done").exists() and time.monotonic() < deadline
                ):
                    time.sleep(0.01)
                self.assertTrue((root / "worker_done").exists())
                log = (root / "test.log").read_bytes()
                self.assertIn(b"SYNTHETIC_WORKER_DONE", log)
                with self.assertRaises(FileExistsError):
                    launch.launch(args)
                self.assertEqual((root / "test.log").read_bytes(), log)

    def test_live_or_stale_lock_is_not_silently_removed(self):
        with tempfile.TemporaryDirectory() as name:
            root = (
                Path(name).resolve()
                / "collaborator_handoff_20261102_v1/experiments/batch"
            )
            run = root / "runs/test"
            run.mkdir(parents=True)
            lock = run / "RUNNING.lock"
            lock.write_text("pid=12345")
            with mock.patch.object(launch, "__file__", str(root / "launch.py")):
                with self.assertRaisesRegex(ValueError, "Existing run lock"):
                    launch.launch(
                        SimpleNamespace(run_id="test", threads=1, resume=True)
                    )
            self.assertTrue(lock.exists())
