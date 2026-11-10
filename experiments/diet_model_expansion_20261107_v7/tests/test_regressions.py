"""Adversarial checks added during the second review, not performance benchmarks."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from sklearn.mixture import GaussianMixture
from support import fixture

from features import Role
from models import atomic, interaction
from registry import specifications
from runtime import publish_json, publish_text


class RegressionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dataset, cls.config = fixture()
        cls.dataset = copy.copy(dataset)
        rng = np.random.default_rng(17)
        cls.dataset.raw = {s: rng.dirichlet(np.ones(180), len(dataset.y)) for s in ("oral", "gut")}
        cls.dataset.taxa = {s: [str(j) for j in range(180)] for s in cls.dataset.raw}
        assignment = dataset.reference["outer"][0]
        cls.split = cls.dataset.split(assignment != 0, assignment == 0, cls.config["contrast"])

    def role(self):
        return Role(self.split, self.dataset.raw, self.config["model_seed"])

    def test_randomized_pca_training_scores_equal_replayed_projection(self):
        role = self.role()
        scores, _, pc, scale = role.pcs()
        assert_allclose(scores, pc.transform(self.split.gt) / scale, rtol=1e-12, atol=1e-12)

    def test_hybrid_design_replays_identically_for_training_and_validation(self):
        role = self.role()
        design = role.design("balanced", count=6, hybrid=True)
        for training, expected in ((True, design.xt), (False, design.xv)):
            actual, _ = role.replay(design.state, training)
            assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_mixture_training_prediction_uses_the_fitted_coordinates(self):
        role = self.role()
        spec = next(s for s in specifications(self.config) if s["name"] == "gut_mixture")
        fit, _ = atomic(role, spec, spec["grid"][0], self.config, 1)
        design = role.design("balanced", count=spec["grid"][0]["count"], hybrid=True)
        expected = fit.model.predict(design.xt, role.pcs()[0]) * fit.scale + fit.mean
        assert_allclose(fit.predict(role, True), expected, rtol=1e-12, atol=1e-12)

    def test_gate_convergence_warning_is_kept_with_the_candidate(self):
        role = self.role()
        specs = specifications(self.config)
        spec = next(s for s in specs if s["name"] == "additive_boost")
        base, _ = atomic(role, spec, spec["grid"][0], self.config, 1)
        spec = next(s for s in specs if s["name"] == "additive_gate_interaction")

        def unfinished_gate(**settings):
            settings["max_iter"] = 1
            return GaussianMixture(**settings)

        with patch("models.GaussianMixture", side_effect=unfinished_gate):
            _, diagnostic = interaction(
                role,
                spec,
                dict(weight=1.0, clusters=2, penalty=0.1),
                base,
                self.config,
                1,
                lambda _, compute: compute(),
            )
        self.assertTrue(diagnostic["warnings"])
        self.assertFalse(diagnostic["converged"])


class PublicationChecks(unittest.TestCase):
    def test_closed_result_publication_recovers_partial_files_and_keeps_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "summary.json"
            path.write_bytes(b'{"status":')
            value = dict(status="COMPLETE", value=2.0)
            publish_json(path, value)
            self.assertEqual(json.loads(path.read_text()), value)
            backup = next(root.glob("summary.json.incomplete_*"))
            self.assertEqual(backup.read_bytes(), b'{"status":')
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                publish_json(path, dict(status="DIFFERENT"))
            self.assertEqual(path.read_bytes(), original)
            text = root / "SUMMARY.txt"
            text.write_text("V7 COM")
            publish_text(text, "V7 COMPLETE\n")
            self.assertEqual(text.read_text(), "V7 COMPLETE\n")
            self.assertEqual(next(root.glob("SUMMARY.txt.incomplete_*")).read_text(), "V7 COM")

    def test_publish_failure_before_rename_leaves_no_completed_result(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "summary.json"
            with patch("runtime.Path.replace", side_effect=OSError("synthetic interruption")):
                with self.assertRaises(OSError):
                    publish_json(path, dict(status="COMPLETE"))
            self.assertFalse(path.exists())
            self.assertTrue(list(Path(folder).glob("summary.json.partial_*")))
            publish_json(path, dict(status="COMPLETE"))
            self.assertEqual(json.loads(path.read_text()), dict(status="COMPLETE"))
