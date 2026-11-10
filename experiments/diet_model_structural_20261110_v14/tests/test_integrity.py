"""Scope, identity, publication, launch and prior-source invariants."""

import io
import contextlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bridge import Memo, sha
from registry import specifications, names, pools, additions
from runtime import check_config, publish_json, publish_text, run_lock, valid_id, register, identity
from support import ROOT, fixture


class Terminal(io.StringIO):
    def isatty(self):
        return True


class IntegrityChecks(unittest.TestCase):
    def test_reporting_before_start_is_readonly_and_ascii(self):
        from report import show

        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "not_started"
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                show(run)
            self.assertFalse(run.exists())
            self.assertIn("NOT_REGISTERED", out.getvalue())
            self.assertTrue(out.getvalue().isascii())

    def test_fixed_scope_counts_and_ascii(self):
        cfg = json.loads((ROOT / "config.json").read_text())
        check_config(cfg)
        self.assertEqual((len(specifications(cfg)), len(names(cfg)), len(additions(cfg))), (40, 66, 34))
        self.assertEqual(sum(len(s["grid"]) for s in specifications(cfg)), 80)
        self.assertEqual(
            {p: len(v) for p, v in pools(cfg).items()},
            dict(oral=18, balanced=22, non_tree=20, new=40, bank=128, combined=168),
        )
        cfg["confirmation_outcomes_read"] = True
        with self.assertRaises(ValueError):
            check_config(cfg)
        for p in list(ROOT.glob("*.py")) + list(ROOT.glob("*.sh")):
            self.assertTrue(p.read_text().isascii(), p.name)

    def test_older_runtime_sources_unchanged(self):
        expected = json.loads((ROOT / "research_records/prior_runtime_hashes.json").read_text())
        for name, digest in expected.items():
            self.assertEqual(sha(ROOT.parent / name), digest, name)

    def test_receipts_and_readonly_misses_and_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            memo = Memo(root)
            with self.assertRaises(FileNotFoundError):
                memo.get("not_here")
            self.assertFalse((root / "not_here").exists())
            for key in ("../bad", "/tmp/bad", "."):
                with self.assertRaises(ValueError):
                    memo.get(key, lambda: 1)
            self.assertEqual(memo.get("one", lambda: 2), 2)
            receipt = next((root / "one").glob("done_*.json"))
            (receipt.parent / json.loads(receipt.read_text())["file"]).write_bytes(b"changed")
            with self.assertRaises(ValueError):
                memo.get("one")

    def test_lock_registration_and_partial_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            register(root / "run", dict(source="one"), False)
            with self.assertRaises(ValueError):
                register(root / "run", dict(source="other"), True)
            with run_lock(root / "run"):
                with self.assertRaises(RuntimeError):
                    with run_lock(root / "run"):
                        pass
            p = root / "summary.json"
            p.write_text('{"st')
            publish_json(p, dict(status="COMPLETE"))
            publish_json(p, dict(status="COMPLETE"))
            with self.assertRaises(ValueError):
                publish_json(p, dict(status="OTHER"))
            t = root / "SUMMARY.txt"
            t.write_text("GO")
            publish_text(t, "GOOD\n")
            self.assertTrue(list(root.glob("*.incomplete_*")))
            for name in ("../x", " ", "", "a" * 65):
                with self.assertRaises(ValueError):
                    valid_id(name)

    def test_historical_expected_hashes_are_checked_before_cache_or_fit(self):
        data, cfg = fixture()
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "old_payload"
            p.write_text("original")
            ref = SimpleNamespace(paths=[p], expected_hashes={str(p): sha(p)})
            p.write_text("changed")
            with self.assertRaisesRegex(ValueError, "Historical V13 registered dependency changed"):
                identity(ROOT, cfg, data, ref, {}, 1)

    def test_shell_entrypoints_and_launch_forward_foreground_thread_settings(self):
        import launch

        for p in ROOT.glob("*.sh"):
            subprocess.run(["bash", "-n", str(p)], check=True)
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(run_id="synthetic", threads=1, resume=False)
            with (
                patch.object(launch, "ROOT", Path(folder)),
                patch.object(launch, "verify", return_value=1),
                patch.object(launch.sys, "stdout", Terminal()),
                patch.object(launch.sys, "stderr", Terminal()),
                patch.object(launch.subprocess, "run") as preflight,
                patch.object(launch.os, "execve") as execute,
            ):
                launch.launch(args)
            self.assertTrue(preflight.call_args.kwargs["check"])
            command = execute.call_args.args[1]
            self.assertIn("--threads", command)
            self.assertIn("1", command)
            self.assertEqual(execute.call_args.args[2]["OMP_NUM_THREADS"], "1")
        self.assertNotIn("open(", (ROOT / "launch.py").read_text())
