import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from numpy.testing import assert_allclose
from support import FakeReference, fixture, reduced_config

from bridge import V6, Memo, module
from engine import Engine
from registry import INTERACTIONS, names
from report import aggregate, compact, show


class PipelineChecks(unittest.TestCase):
    def test_negative_pooled_r2_and_cross_fold_household_rejection(self):
        import numpy as np

        from registry import REFERENCES

        _, config = fixture()
        completed = {}
        for repeat in range(2):
            for fold in range(5):
                y = np.array([float(fold), float(fold) + 1])
                completed[repeat, fold] = dict(
                    rows=np.array([2 * fold, 2 * fold + 1]),
                    y=y,
                    groups=np.array([str(fold), str(fold)]),
                    predictions={name: y + 3 for name in names(config) + list(REFERENCES)},
                    modifiers={name: np.zeros(2) for name in INTERACTIONS},
                    active={name: False for name in INTERACTIONS},
                    inner_selector="lgbm_host",
                    blend_weights=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    selected={},
                    outer_diagnostics={},
                )
        result = aggregate(config, completed)
        y = np.concatenate([completed[0, f]["y"] for f in range(5)])
        score = result["scores"][0]["R2"]
        self.assertLess(score, 0)
        self.assertAlmostEqual(score, 1 - 9 / np.var(y))
        self.assertNotAlmostEqual(score, 1 - 9 / 0.25)
        completed[0, 1]["groups"][:] = "0"
        with self.assertRaises(ValueError):
            aggregate(config, completed)

    def test_runtime_partial_then_complete_resume_and_readonly_report(self):
        import contextlib
        import io
        import shutil

        from support import ROOT

        import run as runtime_worker

        class Terminal(io.StringIO):
            def isatty(self):
                return True

        dataset, config = fixture()
        config = reduced_config(config)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "collaborator_handoff_20261102_v1/experiments/v7"
            root.mkdir(parents=True)
            for path in list(ROOT.glob("*.py")) + [ROOT / "requirements.txt"]:
                shutil.copy2(path, root / path.name)
            (root / "config.json").write_text(json.dumps(config))
            args = SimpleNamespace(run_id="synthetic", threads=1, resume=False, limit_outer=1)
            output = Terminal()
            with (
                patch.object(runtime_worker, "ROOT", root),
                patch.object(runtime_worker, "load", return_value=dataset),
                patch.object(runtime_worker, "Reference", return_value=FakeReference()),
                contextlib.redirect_stdout(output),
                patch.object(runtime_worker.sys, "stderr", output),
            ):
                runtime_worker.execute(args)
                run = root / "runs/synthetic"
                self.assertFalse((run / "summary.json").exists())
                self.assertFalse((run / "RUNNING.lock").exists())
                args.resume, args.limit_outer = True, 0
                runtime_worker.execute(args)
                self.assertEqual(
                    json.loads((run / "summary.json").read_text())["status"], "COMPLETE"
                )
                show(run)
                self.assertIn("BATCH_COMPLETE", output.getvalue())
                (run / "SUMMARY.txt").read_text().encode("ascii")
                # An interrupted final write must not invalidate completed model units.
                full_json = (run / "summary.json").read_bytes()
                full_text = (run / "SUMMARY.txt").read_bytes()
                (run / "summary.json").write_bytes(full_json[:50])
                (run / "SUMMARY.txt").write_bytes(full_text[:50])
                runtime_worker.execute(args)
                self.assertEqual((run / "summary.json").read_bytes(), full_json)
                self.assertEqual((run / "SUMMARY.txt").read_bytes(), full_text)
                self.assertTrue(list(run.glob("summary.json.incomplete_*")))
                self.assertTrue(list(run.glob("SUMMARY.txt.incomplete_*")))
                (root / "config.json").write_text(json.dumps(dict(config, model_seed=1)))
                with self.assertRaises(ValueError):
                    runtime_worker.execute(args)

    def test_complete_two_round_pipeline_cache_and_reporting(self):
        dataset, config = fixture()
        config = reduced_config(config)
        completed = {}
        with tempfile.TemporaryDirectory() as folder:
            memo = Memo(Path(folder) / "TRE_ONLY")
            for repeat in range(2):
                for fold in range(5):
                    engine = Engine(config, memo, 1, FakeReference(), lambda _: None)
                    key = "r%d_f%d_result" % (repeat, fold)
                    completed[repeat, fold] = memo.get(
                        key,
                        lambda engine=engine, repeat=repeat, fold=fold: engine.outer(
                            dataset, repeat, fold
                        ),
                    )
            result = aggregate(config, completed)
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(len(result["scores"]), 66)
            for name in INTERACTIONS:
                for repeat in range(2):
                    delta = next(
                        v
                        for v in result["comparisons"]
                        if v["model"] == name and v["repeat"] == repeat
                    )
                    scores = {
                        v["model"]: v["R2"] for v in result["scores"] if v["repeat"] == repeat
                    }
                    self.assertAlmostEqual(
                        delta["delta_R2"], scores[name] - scores[INTERACTIONS[name]], places=12
                    )
            text = compact(result, config)
            text.encode("ascii")
            self.assertLess(len(text.splitlines()), 40)
            self.assertIn("INCOMPLETE", self._show(Path(folder), config))
            count = memo.created
            with patch.object(
                Engine, "outer", side_effect=AssertionError("Completed outer must not refit")
            ):
                again = memo.get("r0_f0_result", lambda: None)
            assert_allclose(
                again["predictions"]["inner_selector"],
                completed[0, 0]["predictions"]["inner_selector"],
            )
            self.assertEqual(memo.created, count)
            with self.assertRaises(ValueError):
                aggregate(config, {(0, 0): completed[0, 0]})

    @staticmethod
    def _show(folder, config):
        import contextlib
        import io

        (folder / "registration.json").write_text(json.dumps(dict(config=config)))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            show(folder)
        return output.getvalue()

    def test_outer_labels_do_not_change_any_new_choice_prediction_or_fusion_weight(self):
        dataset, config = fixture()
        config = reduced_config(config)
        helper = module("v7_changed_fixture", V6 / "tests/support.py")
        changed = helper.changed_outer_labels(dataset)
        with tempfile.TemporaryDirectory() as folder:
            a = Engine(
                config, Memo(Path(folder) / "original"), 1, FakeReference(), lambda _: None
            ).outer(dataset, 0, 0)
            b = Engine(
                config, Memo(Path(folder) / "changed"), 1, FakeReference(), lambda _: None
            ).outer(changed, 0, 0)
        self.assertEqual(a["selected"], b["selected"])
        self.assertEqual(a["inner_losses"], b["inner_losses"])
        self.assertEqual(a["inner_selector"], b["inner_selector"])
        assert_allclose(a["blend_weights"], b["blend_weights"], atol=0, rtol=0)
        for name in names(config):
            assert_allclose(a["predictions"][name], b["predictions"][name], atol=0, rtol=0)

    def test_inner_cache_resume_needs_no_candidate_refitting(self):
        dataset, config = fixture()
        config = reduced_config(config)
        with tempfile.TemporaryDirectory() as folder:
            engine = Engine(config, Memo(folder), 1, FakeReference(), lambda _: None)
            original = engine.outer(dataset, 0, 0)
            resumed = Engine(config, Memo(folder), 1, FakeReference(), lambda _: None)
            with patch.object(
                resumed, "fit", side_effect=AssertionError("Cached unit unexpectedly refit")
            ):
                again = resumed.outer(dataset, 0, 0)
            self.assertEqual(original["selected"], again["selected"])
            assert_allclose(
                original["predictions"]["convex_blend"], again["predictions"]["convex_blend"]
            )
