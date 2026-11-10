"""Upload, environment, reference, lock and checkpoint safeguards."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bridge import V6, Memo, Reference, sha, write_json
from registry import names, specifications
from report import show
from runtime import audit, check_config, publish_json, publish_text, register, run_lock, valid_id
from support import ROOT, fixture
from upload_check import verify


class IntegrityChecks(unittest.TestCase):
    def test_legacy_adapter_and_fixture_do_not_pollute_module_search_path(self):
        import sys
        from unittest.mock import patch

        import bridge

        previous = sys.path[:]
        fixture()
        self.assertEqual(sys.path, previous)

        def legacy(_):
            sys.path.insert(0, "/synthetic/old-pilot")
            return "loaded"

        with patch.object(bridge, "_frozen_load", legacy):
            self.assertEqual(bridge.load("unused"), "loaded")
        self.assertEqual(sys.path, previous)

    def test_frozen_counts_pins_ascii_and_discovery_scope(self):
        _, cfg = fixture()
        self.assertEqual(len(names(cfg)), 55)
        self.assertEqual(sum(len(v["grid"]) for v in specifications(cfg)), 138)
        check_config(cfg)
        required = {k + "==" + v for k, v in cfg["dependencies"].items()}
        self.assertEqual(set((ROOT / "requirements.txt").read_text().splitlines()), required)
        for path in (
            list(ROOT.glob("*.py"))
            + list((ROOT / "tests").glob("*.py"))
            + [ROOT / "setup.sh", ROOT / "config.json"]
        ):
            path.read_text().encode("ascii")
        for key, value in (("outcome", "D_VAT"), ("confirmation_outcomes_read", True), ("outer_seed", 4)):
            with self.assertRaises(ValueError):
                check_config(dict(cfg, **{key: value}))

    def test_receipt_corruption_and_path_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            memo = Memo(folder)
            self.assertEqual(memo.get("one", lambda: dict(test=True)), dict(test=True))
            (Path(folder) / "one/done_broken.json").write_text("{")
            self.assertEqual(memo.get("one"), dict(test=True))
            next((Path(folder) / "one").glob("*.pkl")).write_bytes(b"broken")
            with self.assertRaises(ValueError):
                memo.get("one")
            with self.assertRaises(ValueError):
                memo.get("../escape")

    def test_partial_final_publication_is_recovered_and_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "summary.json"
            path.write_bytes(b'{"broken":')
            publish_json(path, dict(status="COMPLETE"))
            self.assertEqual(json.loads(path.read_text())["status"], "COMPLETE")
            self.assertTrue(list(path.parent.glob("summary.json.incomplete_*")))
            with self.assertRaises(ValueError):
                publish_json(path, dict(status="DIFFERENT"))
            text = Path(folder) / "SUMMARY.txt"
            text.write_text("V8")
            publish_text(text, "V8 COMPLETE\n")
            self.assertEqual(text.read_text(), "V8 COMPLETE\n")

    def test_run_identity_hashes_and_existing_locks_are_not_bypassed(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "run"
            value = dict(frozen=True)
            register(run, value, False)
            register(run, value, True)
            with self.assertRaises(ValueError):
                register(run, dict(changed=True), True)
            with run_lock(run):
                with self.assertRaises(RuntimeError):
                    with run_lock(run):
                        pass
            write_json(run / "RUNNING.lock", dict(pid=-1))
            with self.assertRaises(RuntimeError):
                with run_lock(run):
                    pass
            self.assertTrue((run / "RUNNING.lock").exists())
            path = Path(folder) / "input"
            path.write_text("before")
            frozen = dict(hashes={str(path): sha(path)})
            audit(frozen)
            path.write_text("after")
            with self.assertRaises(ValueError):
                audit(frozen)
        for run_id in ("../x", "a;exit", "a b", "", "x.y"):
            with self.assertRaises(ValueError):
                valid_id(run_id)

    def test_historical_comparators_are_readonly_and_strictly_matched(self):
        with tempfile.TemporaryDirectory() as folder:
            package = Path(folder)
            rows, y = np.array([1, 2]), np.array([90.0, 93.0])
            for name, member in ((V6.name, "fusion"), ("diet_model_expansion_20261107_v7", "ridge_stack")):
                root = package / "experiments" / name / "runs/test"
                root.mkdir(parents=True)
                config = dict(
                    outcome="D_Glu",
                    exposure="fiber_density",
                    outer_seed=20261104,
                    confirmation_outcomes_read=False,
                    outer_repeats=2,
                    outer_folds=5,
                )
                write_json(root / "registration.json", dict(config=config))
                write_json(root / "summary.json", dict(status="COMPLETE"))
                memo = Memo(root / "TRE_ONLY")
                for r in range(2):
                    for f in range(5):
                        memo.get(
                            "r%d_f%d_result" % (r, f),
                            lambda member=member: dict(rows=rows, y=y, predictions={member: y + 0.2}),
                        )
            before = {str(p): sha(p) for p in package.rglob("*") if p.is_file()}
            reference = Reference(package, "test", "test")
            result = reference.get(0, 0, rows, y)
            np.testing.assert_allclose(result["v6_fusion"], y + 0.2)
            np.testing.assert_allclose(result["v7_ridge_stack"], y + 0.2)
            self.assertEqual(before, {str(p): sha(p) for p in package.rglob("*") if p.is_file()})
            with self.assertRaises(ValueError):
                reference.get(0, 0, rows[::-1], y)
            with self.assertRaises(ValueError):
                reference.get(0, 0, rows, y + 1)
            path, _ = reference.files["v7_ridge_stack", 0, 0]
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                Reference(package, "test", "test")

    def test_unstarted_report_is_readonly(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "missing"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                show(path)
            self.assertIn("NOT_REGISTERED", output.getvalue())
            self.assertFalse(path.exists())

    def test_upload_manifest_detects_changed_files_and_path_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "a.py"
            path.write_text("print('test')\n")
            (root / "upload_manifest.json").write_text(json.dumps(dict(sha256={"a.py": sha(path)})))
            self.assertEqual(verify(root), 1)
            path.write_text("changed\n")
            with self.assertRaises(ValueError):
                verify(root)
            (root / "upload_manifest.json").write_text(json.dumps(dict(sha256={"../escape": "bad"})))
            with self.assertRaises(ValueError):
                verify(root)

    def test_launcher_uses_terminal_preflight_and_never_opens_a_log(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        import launch as launcher

        class Terminal(io.StringIO):
            def isatty(self):
                return True

        args = SimpleNamespace(run_id="synthetic", threads=2, resume=False)
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(launcher, "ROOT", Path(folder)),
            patch.object(launcher, "verify", return_value=3),
            patch.object(launcher.subprocess, "run") as preflight,
            patch.object(launcher.os, "execve") as replace,
            patch.object(launcher.sys, "stdout", Terminal()),
            patch.object(launcher.sys, "stderr", Terminal()),
        ):
            launcher.launch(args)
            self.assertTrue(preflight.call_args.kwargs["check"])
            self.assertIn("check.py", preflight.call_args.args[0][-1])
            command = replace.call_args.args[1]
            self.assertIn("-u", command)
            self.assertIn("run.py", command[2])
            self.assertNotIn(">", " ".join(command))
            self.assertEqual(replace.call_args.args[2]["OMP_NUM_THREADS"], "2")
