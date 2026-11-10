import contextlib
import io
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from support import ROOT

from bridge import V6, Memo, Reference, sha, write_json
from launch import command, launch
from report import show
from runtime import audit, check_config, register, run_lock, valid_id


class IntegrityChecks(unittest.TestCase):
    def test_frozen_counts_and_discovery_guard(self):
        from registry import names, specifications

        config = json.loads((ROOT / "config.json").read_text())
        self.assertEqual(len(names(config)), 30)
        self.assertEqual(sum(len(s["grid"]) for s in specifications(config)), 67)
        check_config(config)
        for key, value in (
            ("confirmation_outcomes_read", True),
            ("outcome", "D_VAT"),
            ("outer_seed", 2),
        ):
            changed = dict(config, **{key: value})
            with self.assertRaises(ValueError):
                check_config(changed)

    def test_receipt_corruption_and_incomplete_units(self):
        with tempfile.TemporaryDirectory() as folder:
            memo = Memo(folder)
            payload = memo.get("one", lambda: dict(a=1))
            self.assertEqual(payload, dict(a=1))
            (Path(folder) / "one" / "done_broken.json").write_text("{")
            self.assertEqual(memo.get("one"), payload)
            path = next((Path(folder) / "one").glob("*.pkl"))
            path.write_bytes(b"wrong")
            with self.assertRaises(ValueError):
                memo.get("one")
            with self.assertRaises(ValueError):
                memo.get("../escape")

    def test_registration_lock_and_hash_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "run"
            identity = dict(config="frozen", versions="test")
            register(run, identity, False)
            register(run, identity, True)
            with self.assertRaises(ValueError):
                register(run, dict(config="changed"), True)
            with run_lock(run):
                with self.assertRaises(RuntimeError):
                    with run_lock(run):
                        pass
            self.assertFalse((run / "RUNNING.lock").exists())
            write_json(run / "RUNNING.lock", dict(pid=-1))
            with self.assertRaises(RuntimeError):
                with run_lock(run):
                    pass
            self.assertTrue((run / "RUNNING.lock").exists())
            data = Path(folder) / "input.txt"
            data.write_text("old")
            value = dict(hashes={str(data): sha(data)})
            audit(value)
            data.write_text("changed")
            with self.assertRaises(ValueError):
                audit(value)

    def test_readonly_v6_reference_checks_rows_targets_and_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            package = Path(folder)
            root = package / "experiments" / V6.name / "runs/test"
            root.mkdir(parents=True)
            write_json(
                root / "registration.json",
                dict(
                    config=dict(
                        outcome="D_Glu",
                        exposure="fiber_density",
                        outer_seed=20261104,
                        confirmation_outcomes_read=False,
                    )
                ),
            )
            write_json(root / "summary.json", dict(status="COMPLETE"))
            rows, y = np.array([1, 2]), np.array([90.0, 93.0])
            memo = Memo(root / "TRE_ONLY")
            for repeat in range(2):
                for fold in range(5):
                    memo.get(
                        "r%d_f%d_result" % (repeat, fold),
                        lambda: dict(rows=rows, y=y, predictions=dict(fusion=y + 0.2)),
                    )
            before = {str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()}
            reference = Reference(package, "test")
            np.testing.assert_allclose(reference.get(0, 0, rows, y), y + 0.2)
            after = {str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            for bad_rows, bad_y in ((rows[::-1], y), (rows, y + 1)):
                with self.assertRaises(ValueError):
                    reference.get(0, 0, bad_rows, bad_y)
            path = reference.files[0, 0]
            path.write_bytes(pickle.dumps(dict(rows=rows)))
            with self.assertRaises(ValueError):
                Reference(package, "test")

    def test_report_unstarted_is_readonly_and_ascii(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "missing"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                show(path)
            output.getvalue().encode("ascii")
            self.assertFalse(path.exists())

    def test_launcher_quotes_arguments_checks_conflicts_and_has_no_file_log(self):
        args = SimpleNamespace(run_id="test-run", threads=2, resume=False, console=False)
        for value in ("../x", "a;exit", "a b", "v7.1", ""):
            with self.assertRaises(ValueError):
                valid_id(value)
        self.assertEqual(command(args)[-1], "2")
        calls = []

        def fake_run(value, **kwargs):
            calls.append(value)
            return SimpleNamespace(returncode=1 if "has-session" in value else 0)

        with (
            patch("launch.subprocess.run", side_effect=fake_run),
            patch("launch.shutil.which", return_value="/usr/bin/tmux"),
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                launch(args)
        tmux = next(c for c in calls if "new-session" in c)
        self.assertNotIn(">", tmux[-1])
        self.assertNotIn("tee", tmux[-1])
        self.assertIn("run.py", tmux[-1])
        self.assertIn("WORKER_EXIT", tmux[-1])
        self.assertNotIn("-c", tmux[1:-1])
        self.assertIn("cd ", tmux[-1])
