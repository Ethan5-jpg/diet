"""Configuration, reference, upload, checkpoint and foreground-launch safeguards."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from bridge import V6, V8, Memo, Reference, sha, write_json
from registry import names, specifications
from report import show
from runtime import audit, check_config, publish_json, publish_text, register, run_lock, valid_id
from support import ROOT, fixture
from upload_check import verify


class IntegrityChecks(unittest.TestCase):
    def test_frozen_counts_pins_ascii_and_discovery_guards(self):
        _, cfg = fixture()
        self.assertEqual(len(names(cfg)), 64)
        self.assertEqual(sum(len(v["grid"]) for v in specifications(cfg)), 164)
        check_config(cfg)
        self.assertEqual(
            set((ROOT / "requirements.txt").read_text().splitlines()),
            {k + "==" + v for k, v in cfg["dependencies"].items()},
        )
        for p in (
            list(ROOT.glob("*.py"))
            + list((ROOT / "tests").glob("*.py"))
            + [ROOT / "setup.sh", ROOT / "config.json"]
        ):
            p.read_text().encode("ascii")
        for key, value in (("outcome", "D_VAT"), ("confirmation_outcomes_read", True), ("outer_seed", 1)):
            with self.assertRaises(ValueError):
                check_config(dict(cfg, **{key: value}))

    def test_old_sources_unchanged_and_frozen_loader_path_mutation_is_contained(self):
        import bridge
        import daily

        hashes = json.loads((ROOT / "research_records/prior_source_hashes.json").read_text())
        required = (
            V6.name + "/data.py",
            V6.name + "/numerics.py",
            V6.name + "/storage.py",
            V8.name + "/features.py",
        )
        for name in required:
            self.assertEqual(sha(ROOT.parent / name), hashes[name])
        before = sys.path[:]
        fixture()
        self.assertEqual(sys.path, before)
        dummy = SimpleNamespace(paths=[], index="synthetic", taxa={"oral": ["a"], "gut": ["b"]})

        def legacy(_):
            sys.path.insert(0, "/synthetic/old-pilot")
            return dummy

        with (
            patch.object(bridge, "_frozen_load", legacy),
            patch.object(daily, "read_daily", return_value=dict(audit={})) as reader,
        ):
            loaded = bridge.load(Path("synthetic_handoff"))
            self.assertEqual(loaded.extra, dict(audit={}))
            self.assertEqual(reader.call_args.args[0].name, "food_days_discovery_TRE_ONLY.parquet")
        self.assertEqual(sys.path, before)

    def test_historical_references_are_readonly_and_match_rows_targets_and_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            package = Path(folder)
            rows, y = np.array([1, 2]), np.array([90.0, 93.0])
            for directory, members in (
                (V6.name, ["fusion"]),
                ("diet_model_expansion_20261107_v7", ["ridge_stack"]),
                (V8.name, ["ridge_wide", "ridge_joint"]),
            ):
                root = package / "experiments" / directory / "runs/test"
                root.mkdir(parents=True)
                cfg = dict(
                    outcome="D_Glu",
                    exposure="fiber_density",
                    outer_seed=20261104,
                    confirmation_outcomes_read=False,
                    outer_repeats=2,
                    outer_folds=5,
                )
                write_json(root / "registration.json", dict(config=cfg))
                write_json(root / "summary.json", dict(status="COMPLETE"))
                memo = Memo(root / "TRE_ONLY")
                for r in range(2):
                    for f in range(5):
                        memo.get(
                            "r%d_f%d_result" % (r, f),
                            lambda members=members: dict(
                                rows=rows, y=y, predictions={n: y + 0.2 for n in members}
                            ),
                        )
            before = {str(p): sha(p) for p in package.rglob("*") if p.is_file()}
            ref = Reference(package, "test", "test", "test")
            values = ref.get(0, 0, rows, y)
            self.assertEqual(len(values), 4)
            for value in values.values():
                np.testing.assert_allclose(value, y + 0.2)
            self.assertEqual(before, {str(p): sha(p) for p in package.rglob("*") if p.is_file()})
            with self.assertRaises(ValueError):
                ref.get(0, 0, rows[::-1], y)
            with self.assertRaises(ValueError):
                ref.get(0, 0, rows, y + 1)
            path, _ = ref.files["v8_ridge_wide", 0, 0]
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                Reference(package, "test", "test", "test")

    def test_incomplete_and_corrupt_receipts_are_not_silently_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            memo = Memo(folder)
            self.assertEqual(memo.get("one", lambda: dict(ok=True)), dict(ok=True))
            (Path(folder) / "one/done_broken.json").write_text("{")
            self.assertEqual(memo.get("one"), dict(ok=True))
            next((Path(folder) / "one").glob("*.pkl")).write_bytes(b"bad")
            with self.assertRaises(ValueError):
                memo.get("one")
            with self.assertRaises(ValueError):
                memo.get("../escape")

    def test_identity_hash_and_existing_run_locks_are_enforced(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "run"
            register(run, dict(frozen=True), False)
            register(run, dict(frozen=True), True)
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
            path.write_text("old")
            value = dict(hashes={str(path): sha(path)})
            audit(value)
            path.write_text("changed")
            with self.assertRaises(ValueError):
                audit(value)
        for value in ("../x", "a;exit", "a b", "", "x.y"):
            with self.assertRaises(ValueError):
                valid_id(value)

    def test_partial_publication_is_recovered_but_complete_conflicts_are_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "summary.json"
            path.write_bytes(b'{"broken":')
            publish_json(path, dict(status="COMPLETE"))
            self.assertEqual(json.loads(path.read_text())["status"], "COMPLETE")
            self.assertTrue(list(path.parent.glob("summary.json.incomplete_*")))
            with self.assertRaises(ValueError):
                publish_json(path, dict(status="DIFFERENT"))
            text = Path(folder) / "SUMMARY.txt"
            text.write_text("V9")
            publish_text(text, "V9 COMPLETE\n")
            self.assertEqual(text.read_text(), "V9 COMPLETE\n")

    def test_upload_check_detects_changes_and_report_before_start_is_readonly(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "a.py"
            path.write_text("print('synthetic')\n")
            (root / "upload_manifest.json").write_text(json.dumps(dict(sha256={"a.py": sha(path)})))
            self.assertEqual(verify(root), 1)
            path.write_text("changed")
            with self.assertRaises(ValueError):
                verify(root)
            (root / "upload_manifest.json").write_text(json.dumps(dict(sha256={"../escape": "bad"})))
            with self.assertRaises(ValueError):
                verify(root)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                show(root / "missing")
            self.assertIn("NOT_REGISTERED", output.getvalue())
            self.assertFalse((root / "missing").exists())

    def test_foreground_launcher_checks_then_executes_without_file_logging(self):
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
