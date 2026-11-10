"""Configuration, reference, upload, checkpoint and foreground-launch safeguards."""

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from bridge import Memo, sha, write_json
from registry import names, specifications
from report import show
from runtime import audit, check_config, publish_json, publish_text, register, run_lock, valid_id
from support import ROOT, fixture
from upload_check import verify


class IntegrityChecks(unittest.TestCase):
    def test_shell_entrypoints_change_directory_verify_environment_and_forward_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "upload folder"
            root.mkdir()
            for name in ("setup.sh", "server_run.sh", "server_report.sh"):
                shutil.copy2(ROOT / name, root / name)
            python = root / "fake-python"
            python.write_text(
                "#!%s\n" % sys.executable
                + "import json,sys\nfrom pathlib import Path\n"
                + "p=Path('calls.jsonl')\n"
                + "with p.open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
                + "print('SYNTHETIC_DISPATCH_OK',flush=True)\n"
            )
            python.chmod(0o700)
            env = dict(os.environ, HPP_V13_PYTHON=str(python))
            subprocess.run(
                ["bash", str(root / "server_run.sh"), "--resume"],
                cwd=directory,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["bash", str(root / "server_report.sh"), "--progress"],
                cwd=directory,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
            self.assertEqual(calls[:2], [["upload_check.py"], ["-"]])
            self.assertEqual(
                calls[2],
                [
                    "launch.py",
                    "--console",
                    "--run-id",
                    "complementary_v13_20261109",
                    "--threads",
                    "2",
                    "--resume",
                ],
            )
            self.assertEqual(calls[3], ["report.py", "--run-id", "complementary_v13_20261109", "--progress"])
            self.assertEqual((root / ".python_path").read_text().strip(), str(python))

    def test_frozen_counts_pins_ascii_and_discovery_guards(self):
        _, cfg = fixture()
        self.assertEqual(len(names(cfg)), 55)
        self.assertEqual(sum(len(v["grid"]) for v in specifications(cfg)), 84)
        check_config(cfg)
        self.assertEqual(
            set((ROOT / "requirements.txt").read_text().splitlines()),
            {k + "==" + v for k, v in cfg["dependencies"].items()},
        )
        for p in (
            list(ROOT.glob("*.py"))
            + list((ROOT / "tests").glob("*.py"))
            + list(ROOT.glob("*.sh"))
            + [ROOT / "config.json"]
        ):
            p.read_text().encode("ascii")
        for key, value in (
            ("outcome", "D_VAT"),
            ("confirmation_outcomes_read", True),
            ("outer_seed", 1),
            ("contrast", [10, 20]),
        ):
            with self.assertRaises(ValueError):
                check_config(dict(cfg, **{key: value}))

    def test_old_sources_unchanged_and_frozen_loader_path_mutation_is_contained(self):
        import bridge
        import context

        hashes = json.loads((ROOT / "research_records/prior_source_hashes.json").read_text())
        for name, digest in hashes.items():
            self.assertEqual(sha(ROOT.parent / name), digest)
        before = sys.path[:]
        fixture()
        self.assertEqual(sys.path, before)
        dummy = SimpleNamespace(
            paths=[], index="synthetic", reference={"y": np.ones((2, 1))}, taxa={"oral": ["a"], "gut": ["b"]}
        )

        def legacy(_):
            sys.path.insert(0, "/synthetic/old-pilot")
            return dummy

        with (
            patch.object(bridge, "_frozen_load", legacy),
            patch.object(
                context, "read_context", return_value=(dict(audit={}), [Path("synthetic.parquet")])
            ) as reader,
        ):
            loaded = bridge.load(Path("synthetic_handoff"))
            self.assertEqual(loaded.extra, dict(audit={}))
            self.assertEqual(reader.call_args.args[0], Path("synthetic_handoff"))
        self.assertEqual(sys.path, before)

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

    def test_duplicate_receipts_fail_and_readonly_misses_do_not_create_units(self):
        from report import completed_results

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            memo = Memo(root / "TRE_ONLY")
            memo.get("r0_f0_result", lambda: dict(value=True))
            with self.assertRaises(FileNotFoundError):
                memo.get("not_started")
            self.assertFalse((memo.root / "not_started").exists())
            receipt = next((memo.root / "r0_f0_result").glob("done_*.json"))
            (receipt.parent / "done_broken.json").write_text("{")
            self.assertEqual(len(completed_results(root, dict(outer_repeats=2, outer_folds=5))), 1)
            shutil.copy2(receipt, receipt.parent / "done_duplicate.json")
            with self.assertRaisesRegex(ValueError, "Multiple closed"):
                completed_results(root, dict(outer_repeats=2, outer_folds=5))

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

    def test_resource_reader_honors_visible_cpu_quota_and_memory_limit(self):
        import resources

        files = {
            "/sys/fs/cgroup/cpu.max": "200000 100000",
            "/sys/fs/cgroup/memory.max": str(8 * 2**30),
            "/sys/fs/cgroup/memory.current": str(3 * 2**30),
            "/proc/meminfo": "MemTotal: 16777216 kB\nMemAvailable: 10485760 kB\n",
        }
        with patch.object(resources, "read", side_effect=lambda path: files.get(path)):
            value = resources.snapshot()
        self.assertEqual(value["quota_cpus"], 2.0)
        self.assertLessEqual(value["effective_cpus"], 2.0)
        self.assertAlmostEqual(value["available_gib"], 5.0)

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
            text.write_text("V10")
            publish_text(text, "V10 COMPLETE\n")
            self.assertEqual(text.read_text(), "V10 COMPLETE\n")

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
