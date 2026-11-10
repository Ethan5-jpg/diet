"""Numerical replay tolerance and explicit, reversible same-run repair."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bridge import Memo, sha
from engine import Engine
from support import ROOT, FakeReference, fixture, reduced

INCIDENT = ROOT / "research_records/2026-11-09_replay_failure"
spec = importlib.util.spec_from_file_location("v13_replay_repair", INCIDENT / "apply_replay_fix.py")
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)
ORIGINAL = (INCIDENT / "original/engine.py").read_bytes()


class OffsetReference(FakeReference):
    offset = 2.52560283798e-8

    def get(self, repeat, fold, rows, y, choices, role=None):
        values, notes = super().get(repeat, fold, rows, y, choices, role)
        if fold == 1:
            values["v11_positive_combined"] += self.offset
        return values, notes


class Terminal(io.StringIO):
    def isatty(self):
        return True


def package(folder):
    root = Path(folder).resolve()
    run = root / "runs/complementary_v13_20261109"
    run.mkdir(parents=True)
    (root / "engine.py").write_bytes(ORIGINAL)
    source = root / "immutable_input"
    source.write_text("SYNTHETIC_ONLY")
    code = root / "immutable_source.py"
    code.write_text("# SYNTHETIC_ONLY\n")
    registration = dict(
        hashes={
            str(root / "engine.py"): sha(root / "engine.py"),
            str(source): sha(source),
            str(code): sha(code),
        },
        threads=2,
        preserved="unchanged",
    )
    (run / "registration.json").write_text(json.dumps(registration))
    (root / "upload_manifest.json").write_text(
        json.dumps(dict(sha256={"engine.py": sha(root / "engine.py")}))
    )
    private = run / "TRE_ONLY/private_checkpoint.pkl"
    private.parent.mkdir()
    private.write_bytes(b"synthetic-private-payload")
    return root, run, private


class ReplayRepairChecks(unittest.TestCase):
    def test_observed_numeric_difference_passes_but_larger_difference_stops(self):
        data, cfg = fixture()
        cfg = reduced(cfg)
        reference = OffsetReference(cfg)
        with tempfile.TemporaryDirectory() as folder:
            memo = Memo(folder)
            value = Engine(cfg, memo, 1, reference, lambda _: None).outer(data, 0, 1)
            self.assertGreater(value["v11_replay_max_error"], 1e-8)
            self.assertLess(value["v11_replay_max_error"], 1e-6)
            reference.offset = 1e-3
            with patch.object(Engine, "fit", side_effect=AssertionError("cached fit called")):
                with self.assertRaisesRegex(ValueError, "max_abs_error=.*tolerance=1e-6"):
                    Engine(cfg, memo, 1, reference, lambda _: None).outer(data, 0, 1)

    def test_explicit_hash_amendment_preserves_originals_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            root, run, private = package(folder)
            before = [(root / n).read_bytes() for n in ("engine.py", "upload_manifest.json")]
            reg_before = (run / "registration.json").read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                repair.apply(root)
            self.assertEqual(private.read_bytes(), b"synthetic-private-payload")
            note_root = root / "research_records/2026-11-09_replay_failure/native_repair_backup"
            self.assertEqual((note_root / "engine.before.py").read_bytes(), before[0])
            self.assertEqual((note_root / "upload_manifest.before.json").read_bytes(), before[1])
            self.assertEqual((note_root / "registration.before.json").read_bytes(), reg_before)
            reg, previous = json.loads((run / "registration.json").read_text()), json.loads(reg_before)
            previous["hashes"][str(root / "engine.py")] = sha(root / "engine.py")
            self.assertEqual(reg, previous)
            self.assertEqual(json.loads((note_root / "repair.json").read_text())["state"], "COMMITTED")
            snapshot = {str(p): sha(p) for p in root.rglob("*") if p.is_file()}
            with contextlib.redirect_stdout(io.StringIO()):
                repair.apply(root)
            self.assertEqual(snapshot, {str(p): sha(p) for p in root.rglob("*") if p.is_file()})

    def test_code_drift_and_existing_locks_are_refused_without_patching(self):
        with tempfile.TemporaryDirectory() as folder:
            root, run, _ = package(folder)
            original = (root / "engine.py").read_bytes()
            (root / "immutable_source.py").write_text("DRIFT")
            with (
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(ValueError, "code/config changed"),
            ):
                repair.apply(root)
            self.assertEqual((root / "engine.py").read_bytes(), original)
            (run / "RUNNING.lock").write_text('{"pid":-1}')
            with self.assertRaises(RuntimeError):
                repair.apply(root)
            self.assertTrue((run / "RUNNING.lock").exists())
            self.assertEqual((root / "engine.py").read_bytes(), original)

    def test_repair_does_not_open_data_or_private_cache_and_retains_their_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            root, run, private = package(folder)
            old = json.loads((run / "registration.json").read_text())
            read = Path.read_bytes
            blocked = {root / "immutable_input", private}

            def guarded(path):
                if path in blocked:
                    raise AssertionError("Repair unexpectedly scanned data/cache")
                return read(path)

            with patch.object(Path, "read_bytes", guarded), contextlib.redirect_stdout(io.StringIO()):
                repair.apply(root)
            new = json.loads((run / "registration.json").read_text())
            self.assertEqual(
                new["hashes"][str(root / "immutable_input")], old["hashes"][str(root / "immutable_input")]
            )
            self.assertEqual(private.read_bytes(), b"synthetic-private-payload")

    def test_unchanged_real_resume_refuses_data_drift_before_using_any_cached_fold(self):
        import run as worker

        data, cfg = fixture()
        cfg, terminal = reduced(cfg), Terminal()
        with tempfile.TemporaryDirectory() as folder:
            root = (
                Path(folder)
                / "collaborator_handoff_20261102_v1/experiments/diet_model_complementary_20261109_v13"
            ).resolve()
            root.mkdir(parents=True)
            for path in list(ROOT.glob("*.py")) + [ROOT / "requirements.txt"]:
                shutil.copy2(path, root / path.name)
            (root / "engine.py").write_bytes(ORIGINAL)
            (root / "config.json").write_text(json.dumps(cfg))
            (root / "upload_manifest.json").write_text(
                json.dumps(dict(sha256={"engine.py": sha(root / "engine.py")}))
            )
            source = root / "synthetic_input.data"
            source.write_text("SYNTHETIC_BEFORE")
            data.paths = [source]
            args = SimpleNamespace(
                run_id="complementary_v13_20261109", threads=1, resume=False, limit_outer=1
            )
            with (
                patch.object(worker, "ROOT", root),
                patch.object(worker, "load", return_value=data),
                patch.object(worker, "Reference", return_value=OffsetReference(cfg)),
                contextlib.redirect_stdout(terminal),
                patch.object(worker.sys, "stderr", terminal),
            ):
                worker.execute(args)
                run = root / "runs/complementary_v13_20261109"
                snapshot = {str(p): sha(p) for p in (run / "TRE_ONLY").rglob("*") if p.is_file()}
                registration = json.loads((run / "registration.json").read_text())
                source.write_text("SYNTHETIC_DRIFT")
                repair.apply(root)
                updated = json.loads((run / "registration.json").read_text())
                self.assertEqual(updated["hashes"][str(source)], registration["hashes"][str(source)])
                args.resume, args.limit_outer = True, 0
                with patch.object(
                    Engine, "outer", side_effect=AssertionError("Data drift reached cached/model evaluation")
                ):
                    with self.assertRaisesRegex(ValueError, "Resume identity differs"):
                        worker.execute(args)
                self.assertTrue(all(sha(Path(p)) == h for p, h in snapshot.items()))

    def test_publication_failure_rolls_back_source_manifest_and_registration(self):
        with tempfile.TemporaryDirectory() as folder:
            root, run, private = package(folder)
            paths = (root / "engine.py", run / "registration.json", root / "upload_manifest.json", private)
            before = {str(p): p.read_bytes() for p in paths}
            original_put, failed = repair.put, []

            def fault(path, payload):
                if path == run / "registration.json" and not failed:
                    failed.append(True)
                    raise OSError("synthetic publication failure")
                original_put(path, payload)

            with patch.object(repair, "put", fault), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(OSError, "publication failure"):
                    repair.apply(root)
            self.assertEqual(before, {str(p): p.read_bytes() for p in paths})
            self.assertFalse((run / "RUNNING.lock").exists())
            with contextlib.redirect_stdout(io.StringIO()):
                repair.apply(root)
            self.assertIn(b"if error > 1e-6:", (root / "engine.py").read_bytes())

    def test_actual_stopped_worker_resumes_cached_failed_fold_without_any_refit(self):
        import run as worker

        old_spec = importlib.util.spec_from_file_location(
            "v13_original_engine_guard", INCIDENT / "original/engine.py"
        )
        old_engine = importlib.util.module_from_spec(old_spec)
        old_spec.loader.exec_module(old_engine)
        data, cfg = fixture()
        cfg, terminal = reduced(cfg), Terminal()
        with tempfile.TemporaryDirectory() as folder:
            root = (
                Path(folder)
                / "collaborator_handoff_20261102_v1/experiments/diet_model_complementary_20261109_v13"
            )
            root = root.resolve()
            root.mkdir(parents=True)
            for path in list(ROOT.glob("*.py")) + [ROOT / "requirements.txt"]:
                shutil.copy2(path, root / path.name)
            (root / "engine.py").write_bytes(ORIGINAL)
            (root / "config.json").write_text(json.dumps(cfg))
            (root / "upload_manifest.json").write_text(
                json.dumps(dict(sha256={"engine.py": sha(root / "engine.py")}))
            )
            args = SimpleNamespace(
                run_id="complementary_v13_20261109", threads=1, resume=False, limit_outer=0
            )
            with (
                patch.object(worker, "ROOT", root),
                patch.object(worker, "load", return_value=data),
                patch.object(worker, "Reference", return_value=OffsetReference(cfg)),
                contextlib.redirect_stdout(terminal),
                patch.object(worker.sys, "stderr", terminal),
            ):
                with patch.object(worker, "Engine", old_engine.Engine):
                    with self.assertRaisesRegex(ValueError, "V11 positive fusion replay differs"):
                        worker.execute(args)
                run = root / "runs/complementary_v13_20261109"
                memo = Memo(run / "TRE_ONLY")
                self.assertTrue((memo.root / "r0_f0_result").exists())
                with self.assertRaises(FileNotFoundError):
                    memo.get("r0_f1_result")
                before = {str(p): sha(p) for p in memo.root.rglob("*") if p.is_file()}
                repair.apply(root)
                args.resume, args.limit_outer = True, 1
                with (
                    patch.object(worker, "Engine", Engine),
                    patch.object(
                        Engine, "fit", side_effect=AssertionError("repair refitted a cached learner")
                    ),
                    patch("engine.meta_fit", side_effect=AssertionError("repair refitted cached fusion")),
                ):
                    worker.execute(args)
                value = memo.get("r0_f1_result")
                self.assertLess(value["v11_replay_max_error"], 1e-6)
                self.assertGreater(value["v11_replay_max_error"], 1e-8)
                self.assertTrue(all(sha(Path(p)) == h for p, h in before.items()))
                self.assertFalse((run / "RUNNING.lock").exists())
                self.assertIn("OUTER_COMPLETE r0 f1", terminal.getvalue())
