"""Family isolation, full nested workflow, resumability and reporting checks."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from support import changed_outer_labels, fixture

import run
from experiment import INTERACTIONS, REFERENCES, Engine, cases, model_names
from storage import Memo


class PipelineChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset, cls.config = fixture()

    def test_outer_labels_do_not_change_choices_predictions_or_fusion(self):
        with (
            tempfile.TemporaryDirectory() as first,
            tempfile.TemporaryDirectory() as second,
        ):
            a = Engine(self.config, Memo(first), 1).outer(self.dataset, 0, 0)
            b = Engine(self.config, Memo(second), 1).outer(
                changed_outer_labels(self.dataset), 0, 0
            )
            self.assertEqual(a["selected"], b["selected"])
            np.testing.assert_allclose(
                a["fusion_weights"], b["fusion_weights"], atol=1e-10, rtol=0
            )
            self.assertEqual(
                set(a["predictions"]), set(model_names(self.config) + list(REFERENCES))
            )
            for name in model_names(self.config):
                np.testing.assert_allclose(
                    a["predictions"][name], b["predictions"][name], atol=1e-9, rtol=0
                )
            for name in INTERACTIONS:
                if a["effective_weights"][name] == 0:
                    np.testing.assert_array_equal(
                        a["predictions"][name], a["predictions"]["kernel_main"]
                    )
                    np.testing.assert_array_equal(
                        a["modifiers"][name], np.zeros(len(a["rows"]))
                    )

    def test_cached_outer_fold_needs_no_model_refitting(self):
        with tempfile.TemporaryDirectory() as name:
            memo = Memo(name)
            first = Engine(self.config, memo, 1).outer(self.dataset, 0, 0)
            with (
                mock.patch(
                    "experiment.tree_fit",
                    side_effect=AssertionError("Unexpected refit"),
                ),
                mock.patch(
                    "experiment.KernelProfile",
                    side_effect=AssertionError("Unexpected kernel refit"),
                ),
            ):
                second_memo = Memo(name)
                second = Engine(self.config, second_memo, 1).outer(self.dataset, 0, 0)
            self.assertEqual(second_memo.created, 0)
            for model in first["predictions"]:
                np.testing.assert_array_equal(
                    first["predictions"][model], second["predictions"][model]
                )

    def test_every_default_candidate_can_fit_the_frozen_preprocessing_interface(self):
        config = json.loads((run.ROOT / "config.json").read_text())
        assignment = self.dataset.reference["outer"][0]
        split = self.dataset.split(assignment != 0, assignment == 0, config["contrast"])
        with tempfile.TemporaryDirectory() as name:
            engine = Engine(config, Memo(name), 1)
            for index in range(len(cases("kernel_main", config))):
                fitted = engine.fit("kernel_main", split, "default_grid", index)
                self.assertTrue(np.isfinite(fitted["prediction"]).all())
            engine.set_base(split, "default_grid", 0, False)
            for model in model_names(config)[1:-1]:
                for index in range(len(cases(model, config))):
                    with self.subTest(model=model, candidate=index):
                        fitted = engine.fit(model, split, "default_grid", index)
                        self.assertEqual(fitted["prediction"].shape, split.yv.shape)
                        self.assertTrue(np.isfinite(fitted["prediction"]).all())

    def test_complete_two_round_run_resume_and_configuration_guard(self):
        with tempfile.TemporaryDirectory() as name:
            package = Path(name).resolve() / "collaborator_handoff_20261102_v1"
            root = package / "experiments/diet_model_batch_20261107_v6"
            root.mkdir(parents=True)
            for filename in (
                "data.py",
                "experiment.py",
                "learners.py",
                "numerics.py",
                "run.py",
                "reporting.py",
                "storage.py",
            ):
                (root / filename).write_bytes((run.ROOT / filename).read_bytes())
            (root / "config.json").write_text(json.dumps(self.config))
            args = SimpleNamespace(
                run_id="synthetic_only",
                summary=False,
                resume=False,
                threads=1,
                limit_outer=1,
            )
            with (
                mock.patch.object(run, "ROOT", root),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                run.execute(args, loader=lambda package: self.dataset)
                output = root / "runs/synthetic_only"
                self.assertFalse((output / "summary.json").exists())
                self.assertFalse((output / "RUNNING.lock").exists())
                self.assertIsNone(run.report(output))
                args.resume, args.limit_outer = True, 0
                result = run.execute(args, loader=lambda package: self.dataset)
                self.assertEqual(result["status"], "COMPLETE")
                brief = (output / "SUMMARY.md").read_text()
                self.assertIn("# V6 结果摘要", brief)
                self.assertLessEqual(len(brief.splitlines()), 28)
                self.assertEqual(
                    len(result["scores"]),
                    2 * (len(model_names(self.config)) + len(REFERENCES)),
                )
                self.assertEqual(len(run.results(output, self.config)), 10)
                self.assertFalse(result["confirmation_outcomes_read"])
                count = len(list((output / "TRE_ONLY").glob("*/*.pkl")))
                run.execute(args, loader=lambda package: self.dataset)
                self.assertEqual(brief, (output / "SUMMARY.md").read_text())
                self.assertEqual(
                    count, len(list((output / "TRE_ONLY").glob("*/*.pkl")))
                )
                altered = copy.deepcopy(self.config)
                altered["model_seed"] += 1
                (root / "config.json").write_text(json.dumps(altered))
                with self.assertRaisesRegex(
                    ValueError, "Code, inputs or environment changed"
                ):
                    run.execute(args, loader=lambda package: self.dataset)

    def test_pooled_r2_and_matched_interaction_delta_use_identical_rows(self):
        completed = {}
        names = model_names(self.config) + list(REFERENCES)
        for repeat in range(2):
            for fold in range(5):
                y = np.array([80.0, 90.0, 100.0]) + fold * 20
                completed[(repeat, fold)] = dict(
                    rows=np.arange(fold * 3, fold * 3 + 3),
                    y=y,
                    predictions={
                        name: y + (j + 1) * 0.1 for j, name in enumerate(names)
                    },
                    modifiers={
                        name: np.array([-0.1, 0.0, 0.1]) for name in INTERACTIONS
                    },
                    effective_weights={name: 1.0 for name in INTERACTIONS},
                )
        result = run.aggregate(self.config, completed)
        target = np.concatenate([completed[(0, fold)]["y"] for fold in range(5)])
        score = next(
            row
            for row in result["scores"]
            if row["model"] == "kernel_main" and row["repeat"] == 0
        )
        self.assertAlmostEqual(score["R2"], 1 - 0.01 / target.var())
        name = INTERACTIONS[0]
        index = names.index(name)
        row = next(
            row
            for row in result["comparisons"]
            if row["model"] == name and row["repeat"] == 0
        )
        self.assertEqual(row["comparator"], "kernel_main")
        self.assertAlmostEqual(
            row["delta_R2"], (0.01 - ((index + 1) * 0.1) ** 2) / target.var()
        )
        self.assertLess(row["delta_R2"], 0)

    def test_partial_summary_suppresses_pooled_scores(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "registration.json").write_text(
                json.dumps(dict(config=self.config))
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertIsNone(run.report(root))
            self.assertIn("0/10", output.getvalue())
            self.assertNotIn("| 模型 |", output.getvalue())
