"""Actual worker execution, nested household isolation and checkpoint recovery."""

import contextlib
import copy
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from bridge import V6, Memo, module
from engine import Engine

from registry import REFERENCES, names
from report import aggregate, compact, diagnostics, show
from support import ROOT, FakeReference, fixture, reduced


class Terminal(io.StringIO):
    def isatty(self):
        return True


def artificial(config):
    out = {}
    for r in range(2):
        for f in range(5):
            y = np.array([float(f), float(f) + 1])
            out[r, f] = dict(
                rows=np.array([2 * f, 2 * f + 1]),
                y=y,
                groups=np.array([str(f), str(f)]),
                predictions={n: y + 3 for n in names(config) + list(REFERENCES)},
                selected={},
                meta={},
                v11_replay_max_error=0.0,
                outer_diagnostics={},
            )
    return out


class PipelineChecks(unittest.TestCase):
    def test_actual_worker_two_repeat_runtime_resume_and_readonly_reports(self):
        import run as worker

        data, cfg = fixture()
        cfg, output = reduced(cfg), Terminal()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "collaborator_handoff_20261102_v1/experiments/v13"
            root.mkdir(parents=True)
            for p in list(ROOT.glob("*.py")) + [ROOT / "requirements.txt"]:
                shutil.copy2(p, root / p.name)
            (root / "config.json").write_text(json.dumps(cfg))
            args = SimpleNamespace(run_id="synthetic", threads=1, resume=False, limit_outer=1)
            with (
                patch.object(worker, "ROOT", root),
                patch.object(worker, "load", return_value=data),
                patch.object(worker, "Reference", return_value=FakeReference(cfg)),
                contextlib.redirect_stdout(output),
                patch.object(worker.sys, "stderr", output),
            ):
                worker.execute(args)
                run = root / "runs/synthetic"
                self.assertFalse((run / "summary.json").exists())
                original = Memo(run / "TRE_ONLY").get("r0_f0_result")
                with (
                    patch.object(
                        Engine, "fit", side_effect=AssertionError("Cached candidate unexpectedly refit")
                    ),
                    patch("engine.meta_fit", side_effect=AssertionError("Cached fusion unexpectedly refit")),
                ):
                    again = Engine(cfg, Memo(run / "TRE_ONLY"), 1, FakeReference(cfg), lambda _: None).outer(
                        data, 0, 0
                    )
                for n in names(cfg):
                    np.testing.assert_array_equal(original["predictions"][n], again["predictions"][n])
                args.resume, args.limit_outer = True, 0
                worker.execute(args)
                result = json.loads((run / "summary.json").read_text())
                self.assertEqual(result["status"], "COMPLETE")
                self.assertEqual(len(result["scores"]), 2 * (len(names(cfg)) + len(REFERENCES)))
                self.assertLess(len(compact(result, cfg).splitlines()), 40)
                self.assertLess(len(diagnostics(result).splitlines()), 45)
                with patch.object(
                    Engine, "outer", side_effect=AssertionError("Completed outer unexpectedly refit")
                ):
                    worker.execute(args)
                    show(run)
                    show(run, all_models=True)
                    show(run, fit_notes=True)
                full_json, full_text = (run / "summary.json").read_bytes(), (run / "SUMMARY.txt").read_bytes()
                (run / "summary.json").write_bytes(full_json[:30])
                (run / "SUMMARY.txt").write_bytes(full_text[:30])
                worker.execute(args)
                self.assertEqual((run / "summary.json").read_bytes(), full_json)
                self.assertEqual((run / "SUMMARY.txt").read_bytes(), full_text)
                self.assertIn("BATCH_COMPLETE", output.getvalue())
                (root / "config.json").write_text(json.dumps(dict(cfg, model_seed=1)))
                with self.assertRaises(ValueError):
                    worker.execute(args)

    def test_outer_labels_do_not_change_choices_fusion_or_any_new_prediction(self):
        data, cfg = fixture()
        cfg = reduced(cfg)
        helper = module("v11_changed_label_fixture", V6 / "tests/support.py")
        changed = helper.changed_outer_labels(data)
        with tempfile.TemporaryDirectory() as folder:
            a, b = [
                Engine(cfg, Memo(Path(folder) / str(j)), 1, FakeReference(cfg), lambda _: None).outer(d, 0, 0)
                for j, d in enumerate((data, changed))
            ]
        self.assertEqual(a["selected"], b["selected"])
        self.assertEqual(a["inner_losses"], b["inner_losses"])
        self.assertEqual(a["meta"], b["meta"])
        for n in names(cfg):
            np.testing.assert_array_equal(a["predictions"][n], b["predictions"][n])

    def test_outer_features_do_not_change_learned_meta_functions(self):
        data, cfg = fixture()
        cfg = reduced(cfg)
        changed = copy.copy(data)
        changed.raw, changed.extra = copy.deepcopy(data.raw), copy.deepcopy(data.extra)
        valid = data.reference["outer"][0] == 0
        for x in changed.raw.values():
            x[valid] = x[valid, ::-1]
        for value in changed.extra["blocks"].values():
            value[valid] = 999
        changed.extra["alcohol"][valid] = "outer_only"
        rng = np.random.default_rng(44)
        common = {
            n: rng.normal(90, 4, 20) for n in names(cfg) + ["bank__" + x for x in cfg["legacy_members"]]
        }
        with tempfile.TemporaryDirectory() as folder:
            blocks, fits = [], []
            for j, d in enumerate((data, changed)):
                memo = Memo(Path(folder) / str(j))
                blocks.append(Engine(cfg, memo, 1, FakeReference(cfg), lambda _: None).outer(d, 0, 0))
                fits.append(memo.get("r0_f0_meta_fit"))
            self.assertEqual(blocks[0]["selected"], blocks[1]["selected"])
            self.assertEqual(blocks[0]["inner_losses"], blocks[1]["inner_losses"])
            for n in fits[0]:
                np.testing.assert_array_equal(fits[0][n].predict(common), fits[1][n].predict(common))

    def test_negative_scores_partial_batches_and_household_leakage_are_handled(self):
        _, cfg = fixture()
        blocks = artificial(cfg)
        result = aggregate(cfg, blocks)
        y = np.concatenate([blocks[0, f]["y"] for f in range(5)])
        score = result["scores"][0]["R2"]
        self.assertLess(score, 0)
        self.assertAlmostEqual(score, 1 - 9 / np.var(y))
        with self.assertRaises(ValueError):
            aggregate(cfg, {(0, 0): blocks[0, 0]})
        blocks[0, 1]["groups"][:] = "0"
        with self.assertRaises(ValueError):
            aggregate(cfg, blocks)

    def test_goal_uses_same_pipeline_both_repeats_and_unrounded_precision(self):
        _, cfg = fixture()
        result = aggregate(cfg, artificial(cfg))
        for v in result["scores"]:
            if v["model"] == "hgb_pc_oral":
                v["R2"] = 0.200001 if v["repeat"] == 0 else 0.1999999
            elif v["model"] == "cat_pc_balanced":
                v["R2"] = 0.1999999 if v["repeat"] == 0 else 0.200001
        self.assertIn("Both repeats R2>=0.20: 0/55", compact(result, cfg))

    def test_repeat_populations_must_match_even_when_both_counts_are_equal(self):
        _, cfg = fixture()
        blocks = artificial(cfg)
        blocks[1, 4]["rows"][1] = 99
        with self.assertRaisesRegex(ValueError, "populations differ"):
            aggregate(cfg, blocks)

    def test_member_warning_reaches_contributing_fusion(self):
        data, cfg = fixture()
        cfg = reduced(cfg)
        original = Engine.evaluate

        def warned(engine, role, spec, case, keep=False):
            value = original(engine, role, spec, case, keep)
            if spec["name"] == "hgb_pc_oral":
                value["diagnostic"]["warnings"] = ["Synthetic input warning"]
            return value

        with tempfile.TemporaryDirectory() as folder, patch.object(Engine, "evaluate", warned):
            with patch("fusion.blend_weights", side_effect=lambda a, y: np.full(a.shape[1], 1 / a.shape[1])):
                block = Engine(cfg, Memo(folder), 1, FakeReference(cfg), lambda _: None).outer(data, 0, 0)
        for n in ("convex_new", "convex_combined", "convex_oral"):
            self.assertIn("hgb_pc_oral", block["outer_diagnostics"][n]["contributing_fit_warnings"])
