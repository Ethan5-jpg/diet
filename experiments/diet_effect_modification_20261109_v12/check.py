"""Synthetic-only preflight. No discovery/confirmation input is opened."""

import copy
import argparse
import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from bridge import Memo
from background import render as background_render
import launch as launcher
from engine import Engine
from features import ecology, fixed_targets, partial_out
from inference import holm, paired_interval, partial_inference
from models import fit_augmented, fit_main, specifications
from report import aggregate, completed_results, ordered, screen, show
from runtime import check_config, publish_json, readiness, register, run_lock, valid_id
from synthetic import Dataset, separated

ROOT = Path(__file__).resolve().parent


def config():
    value = json.loads((ROOT / "config.json").read_text())
    value.update(main_components_per_site=8, modifier_components_per_site=3,
                 curved_components_per_site=3, main_penalties=[0.01, 0.1],
                 interaction_penalties=[0.03, 0.3], score_multipliers=199,
                 prediction_bootstraps=199, support_neighbors=40)
    return value


class CoreChecks(unittest.TestCase):
    def test_scope_is_fixed(self):
        value = json.loads((ROOT / "config.json").read_text())
        check_config(value)
        value["outcome"] = "another_endpoint"
        with self.assertRaises(ValueError):
            check_config(value)

    def test_run_id_cannot_escape(self):
        for value in ("../old", "/tmp/run", "x/y", "", "a" * 65):
            with self.assertRaises(ValueError):
                valid_id(value)

    def test_readiness_blocks_before_native_inputs(self):
        with self.assertRaisesRegex(RuntimeError, "WAIT"):
            readiness(config(), False, Path("/no_native_data_here"))

    def test_readiness_needs_completed_v11(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = config()
            with self.assertRaises(RuntimeError):
                readiness(value, True, root)
            run = root / "runs" / value["v11_run_id"]
            run.mkdir(parents=True)
            (run / "summary.json").write_text(json.dumps(dict(status="COMPLETE", confirmation_outcomes_read=False,
                                                              scores=[dict(repeat=r, N=3343) for r in (0, 1)])))
            self.assertTrue(readiness(value, True, root)["v6_v11_summary_reviewed"])
            (run / "RUNNING.lock").write_text("still_running")
            with self.assertRaisesRegex(RuntimeError, "lock"):
                readiness(value, True, root)

    def test_ecology_is_relative_and_finite_with_zeros(self):
        raw = np.array([[1., 3., 0.], [2., 2., 2.]])
        np.testing.assert_allclose(ecology(raw), ecology(100 * raw))
        self.assertTrue(np.isfinite(ecology(raw)).all())
        with self.assertRaises(ValueError):
            ecology(np.zeros((2, 3)))

    def test_fixed_target_units(self):
        markers = np.array([[2., .4, 3., .2], [1., .1, 2., .3]])
        delta = fixed_targets(np.full(2, 15), markers) - fixed_targets(np.full(2, 10), markers)
        np.testing.assert_allclose(delta, markers)

    def test_households_never_cross(self):
        data = Dataset()
        groups = data.index.split_unit_id.to_numpy()
        for assignment in data.reference["outer"]:
            for fold in range(5):
                separated(groups, assignment != fold, assignment == fold)

    def test_background_inspection_never_uses_outcome_values(self):
        data = Dataset()
        first = background_render(data)
        data.y += 1000000
        data.reference["y"] += 1000000
        self.assertEqual(first, background_render(data))
        self.assertNotIn("synthetic_household_", first)
        self.assertLessEqual(len(first.splitlines()), 24)

    def test_projection_handles_duplicate_main_columns(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(200, 4))
        x = np.column_stack([np.ones(200), x, x[:, 0]])
        y = x[:, 1] - 3 * x[:, 3] + 7
        residual, q, rank = partial_out(x, y)
        self.assertEqual(rank, 5)
        np.testing.assert_allclose(residual, 0, atol=1e-10)
        np.testing.assert_allclose(q.T @ q, np.eye(rank), atol=1e-10)

    def test_partial_regression_and_covariance_match_full_ols_reference(self):
        rng = np.random.default_rng(119)
        n, groups = 600, np.arange(600) // 2
        z = rng.normal(size=(n, 3))
        target = rng.normal(size=(n, 4)) + z[:, :1]
        nuisance = np.zeros((n, 15))
        blocks, ry, rt = [], np.zeros(n), np.zeros((n, 4))
        for fold in range(3):
            rows = np.arange(fold * 200, (fold + 1) * 200)
            nuisance[rows, 5 * fold:5 * fold + 5] = np.column_stack([np.ones(200), z[rows], z[rows, 0] ** 2])
        full = np.column_stack([nuisance, target])
        y = full @ rng.normal(size=19) + rng.normal(size=300)[groups] + rng.normal(size=n)
        for fold in range(3):
            rows = np.arange(fold * 200, (fold + 1) * 200)
            residual, q, _ = partial_out(nuisance[rows, 5 * fold:5 * fold + 5], np.column_stack([y[rows], target[rows]]))
            ry[rows], rt[rows] = residual[:, 0], residual[:, 1:]
            blocks.append(dict(rows=rows, q=q))
        result = partial_inference(ry, rt, groups, np.ones(4), 199, main_blocks=blocks)
        inverse = np.linalg.inv(full.T @ full)
        coefficient = inverse @ full.T @ y
        np.testing.assert_allclose([m["coefficient"] for m in result["markers"]], coefficient[-4:], atol=1e-10)
        # Independent dense full-design hat matrix and sandwich reference.
        hat = full @ inverse @ full.T
        residual = y - full @ coefficient
        meat = np.zeros((19, 19))
        for group in range(300):
            rows = np.flatnonzero(groups == group)
            values, vectors = np.linalg.eigh(np.eye(2) - hat[np.ix_(rows, rows)])
            adjusted = vectors @ ((vectors.T @ residual[rows]) / np.sqrt(values))
            score = full[rows].T @ adjusted
            meat += np.outer(score, score)
        reference = 300 / 299 * inverse @ meat @ inverse
        np.testing.assert_allclose(result["covariance"], reference[-4:, -4:], rtol=1e-9, atol=1e-10)

    def test_rank_deficiency_is_unavailable(self):
        x = np.ones((100, 4))
        result = partial_inference(np.arange(100), x, np.arange(100), np.ones(4), 199)
        self.assertEqual(result["status"], "UNAVAILABLE")
        self.assertNotIn("omnibus", result)

    def test_many_nominal_households_do_not_hide_concentrated_information(self):
        rng = np.random.default_rng(409)
        n, groups = 800, np.arange(800) // 2
        x = rng.normal(size=(n, 4))
        x[:, 0] = 0
        x[:8, 0] = rng.normal(size=8)
        self.assertEqual(np.linalg.matrix_rank(x), 4)
        result = partial_inference(rng.normal(size=n), x, groups, np.ones(4), 199)
        self.assertEqual(result["status"], "UNAVAILABLE")
        self.assertLess(result["effective_households_per_coordinate"][0], 10)
        self.assertNotIn("omnibus", result)

    def test_household_information_diagnostic_is_invariant_to_marker_units(self):
        rng = np.random.default_rng(410)
        x = rng.normal(size=(900, 4))
        y, groups = rng.normal(size=900), np.arange(900) // 2
        first = partial_inference(y, x, groups, np.ones(4), 199)
        x[:, 1] *= 1000
        second = partial_inference(y, x, groups, np.ones(4), 199)
        np.testing.assert_allclose(first["effective_households_per_coordinate"], second["effective_households_per_coordinate"], rtol=1e-9)

    def test_known_interaction_recovery_and_units(self):
        rng = np.random.default_rng(110)
        n = 1600
        x = rng.normal(size=(n, 4))
        theta = np.array([1.2, -0.9, 0., .5])
        groups = np.arange(n) // 2
        error = rng.normal(size=n) + rng.normal(size=n // 2)[groups]
        result = partial_inference(x @ theta + error, x, groups, np.array([1., .2, 1., .2]), 399)
        self.assertEqual(result["status"], "OK")
        estimates = np.array([row["coefficient"] for row in result["markers"]])
        np.testing.assert_allclose(estimates, theta, atol=.15)
        self.assertLess(result["omnibus"]["p_primary"], .05)
        self.assertAlmostEqual(result["markers"][1]["interaction_iqr_difference_mg_dl"], estimates[1] * .2)

    def test_household_dependence_changes_uncertainty(self):
        rng = np.random.default_rng(101)
        groups = np.arange(900) // 3
        x = rng.normal(size=(300, 4))[groups]
        y = rng.normal(size=300)[groups] + .05 * rng.normal(size=900)
        clustered = partial_inference(y, x, groups, np.ones(4), 199)
        naive = partial_inference(y, x, np.arange(900), np.ones(4), 199)
        self.assertGreater(clustered["markers"][0]["se"], 1.3 * naive["markers"][0]["se"])

    def test_holm_followup_family(self):
        np.testing.assert_allclose(holm([.01, .04, .02]), [.03, .04, .04])

    def test_negative_prediction_increment_has_negative_interval(self):
        rng = np.random.default_rng(91)
        y = rng.normal(size=400)
        good = y + rng.normal(0, .1, 400)
        bad = y + rng.normal(0, 2, 400)
        ci = paired_interval(y, good, bad, np.arange(400) // 2, 199)
        self.assertLess(ci[1], 0)

    def test_manifest_checkpoint_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            memo = Memo(Path(temporary))
            memo.get("unit", lambda: np.arange(10))
            payload = next((Path(temporary) / "unit").glob("*.pkl"))
            payload.write_bytes(payload.read_bytes() + b"corruption")
            with self.assertRaises(ValueError):
                memo.get("unit")

    def test_resume_identity_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            register(run, {"identity": "a"})
            register(run, {"identity": "a"}, True)
            with self.assertRaises(ValueError):
                register(run, {"identity": "b"}, True)
            publish_json(run / "summary.json", {"value": 1})
            with self.assertRaises(ValueError):
                publish_json(run / "summary.json", {"value": 2})

    def test_failure_lock_is_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            with self.assertRaises(RuntimeError):
                with run_lock(run):
                    raise RuntimeError("synthetic failure")
            self.assertTrue((run / "RUNNING.lock").exists())

    def test_progress_counts_verified_units_once_and_is_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            memo = Memo(run / "TRE_ONLY")
            memo.get("r0_f0_COMPLETE", lambda: {"synthetic": True})
            folder = memo.root / "r0_f0_COMPLETE"
            receipt = next(folder.glob("done_*.json"))
            shutil.copyfile(receipt, folder / "done_duplicate.json")
            (folder / "done_partial.json").write_text('{"file":')
            before = {str(p.relative_to(run)) for p in run.rglob("*")}
            output = io.StringIO()
            with redirect_stdout(output):
                show(run, progress=True)
            self.assertIn("completed_outer=1/10", output.getvalue())
            self.assertIn("final_aggregate=missing", output.getvalue())
            self.assertEqual(before, {str(p.relative_to(run)) for p in run.rglob("*")})

    def test_progress_rejects_corrupt_completed_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            memo = Memo(run / "TRE_ONLY")
            memo.get("r0_f0_COMPLETE", lambda: {"synthetic": True})
            payload = next((memo.root / "r0_f0_COMPLETE").glob("*.pkl"))
            payload.write_bytes(payload.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(ValueError, "Corrupted checkpoint"):
                show(run, progress=True)

    def test_missing_run_progress_does_not_create_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "not_created"
            output = io.StringIO()
            with redirect_stdout(output):
                show(run, progress=True)
            self.assertIn("completed_outer=0/10", output.getvalue())
            self.assertFalse(run.exists())

    def test_completed_results_reader_does_not_create_absent_units(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            memo = Memo(run / "TRE_ONLY")
            memo.get("r0_f0_COMPLETE", lambda: {"synthetic": True})
            before = {str(p.relative_to(run)) for p in run.rglob("*")}
            result = completed_results(run, config())
            self.assertEqual(set(result), {(0, 0)})
            self.assertEqual(before, {str(p.relative_to(run)) for p in run.rglob("*")})

    def test_duplicate_launch_is_blocked_before_synthetic_preflight(self):
        for locked, resume in ((True, True), (False, False)):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                run = root / "runs" / "duplicate"
                run.mkdir(parents=True)
                if locked:
                    (run / "RUNNING.lock").write_text('{"pid": 12345}')
                args = argparse.Namespace(run_id="duplicate", threads=2, resume=resume,
                                          prior_summary_reviewed=True, limit_outer=0)
                with patch.object(launcher, "ROOT", root), patch.object(launcher.sys.stdout, "isatty", return_value=True), patch.object(launcher.sys.stderr, "isatty", return_value=True), patch.object(launcher.subprocess, "run") as preflight, patch.object(launcher.os, "execve") as worker:
                    with self.assertRaises((RuntimeError, FileExistsError)):
                        launcher.launch(args)
                    preflight.assert_not_called()
                    worker.assert_not_called()


class FeatureChecks(unittest.TestCase):
    def setUp(self):
        self.data, self.cfg = Dataset(n=1200), config()
        self.train = self.data.reference["outer"][0] != 0
        self.valid = ~self.train
        self.engine = Engine(self.cfg, None, log=lambda _: None)

    def test_outer_changes_cannot_change_training_features(self):
        first = self.engine.design(self.data, self.train, self.valid)
        self.data.y[self.valid] += 1000
        self.data.raw["oral"][self.valid] = np.roll(self.data.raw["oral"][self.valid], 1, axis=1)
        self.data.index.loc[self.valid, "mean_daily_energy_fiber"] += 900
        second = self.engine.design(self.data, self.train, self.valid)
        np.testing.assert_allclose(first.xt, second.xt, atol=1e-12)
        for site in ("oral", "gut"):
            np.testing.assert_allclose(first.pc_states[site]["model"].components_, second.pc_states[site]["model"].components_, atol=1e-12)

    def test_observed_diet_replay_is_exact(self):
        design = self.engine.design(self.data, self.train, self.valid)
        value = 12.5
        # Check all retained host diet columns, including age/sex interactions.
        at = design.main(False, value)
        self.assertTrue(np.isfinite(at).all())
        self.assertFalse(np.allclose(at, design.main(False, 17.5)))
        np.testing.assert_allclose(design.main(False), design.xv)

    def test_fixed_diet_design_matches_independent_validation_rebuild(self):
        first = self.engine.design(self.data, self.train, self.valid)
        value = 12.5
        expected_main = first.main(False, value).copy()
        expected_interactions = {}
        for spec in specifications():
            _, _, mapping = first.interactions(spec)
            expected_interactions[spec["name"]] = mapping.transform(first, spec, False, value).copy()
        # Rebuild through the data adapter at a changed validation exposure,
        # independently of host_at or the counterfactual transform operation.
        self.data.index.loc[self.valid, "fiber_density"] = value
        second = self.engine.design(self.data, self.train, self.valid)
        np.testing.assert_allclose(first.xt, second.xt, atol=1e-12)
        np.testing.assert_allclose(second.xv, expected_main, atol=1e-10)
        for spec in specifications():
            _, validation, _ = second.interactions(spec)
            np.testing.assert_allclose(validation, expected_interactions[spec["name"]], atol=1e-10)

    def test_exact_zero_option_and_contrast_decomposition(self):
        design = self.engine.design(self.data, self.train, self.valid)
        base = fit_main(design, .1)
        self.assertEqual(len(specifications()), 12)
        for spec in specifications():
            zero = fit_augmented(design, spec, .1, None, base)
            self.assertIs(zero, base)
            np.testing.assert_array_equal(zero.predict(), base.predict())
        model = fit_augmented(design, specifications()[-1], .1, .03)
        total, component = model.contrasts()
        lo, hi = self.cfg["contrast"]
        xdelta = design.main(False, hi) - design.main(False, lo)
        product_coefficient = np.zeros(model.interaction_map.projection.shape[1])
        product_coefficient[model.interaction_map.keep] = model.coefficient[design.xt.shape[1]:] / model.interaction_map.scaler.scale_
        raw_main_coefficient = model.coefficient[:design.xt.shape[1]] - model.interaction_map.projection @ product_coefficient
        expected_background = xdelta @ raw_main_coefficient
        np.testing.assert_allclose(total, expected_background + component, atol=1e-10)

    def test_interactions_cannot_duplicate_penalised_main_effect_directions(self):
        design = self.engine.design(self.data, self.train, self.valid)
        base = fit_main(design, .1)
        for spec in specifications():
            zt, _, _ = design.interactions(spec)
            np.testing.assert_allclose(design.xt.T @ zt / len(design.xt), 0, atol=1e-8)
            model = fit_augmented(design, spec, .1, .03)
            np.testing.assert_allclose(model.coefficient[:design.xt.shape[1]], base.coefficient, atol=1e-8)

    def test_support_never_drops_evaluation_rows(self):
        self.cfg["contrast"] = [-500, 500]
        design = self.engine.design(self.data, self.train, self.valid)
        self.assertFalse(design.support().any())
        self.assertEqual(len(design.xv), self.valid.sum())

    def test_outer_outcome_cannot_choose_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = Engine(self.cfg, Memo(Path(temporary) / "a"), log=lambda _: None).checkpoint(self.data, 0, 0)
            self.data.y[self.valid] += 100
            self.data.reference["y"][:, self.valid] += 100
            second = Engine(self.cfg, Memo(Path(temporary) / "b"), log=lambda _: None).checkpoint(self.data, 0, 0)
            self.assertEqual(first["choices"], second["choices"])
            self.assertEqual(first["main_penalty"], second["main_penalty"])
            for name in first["prediction"]:
                np.testing.assert_allclose(first["prediction"][name], second["prediction"][name])


class EndToEndChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="hpp_v12_synthetic_")
        cls.cfg, cls.data = config(), Dataset(n=1200, seed=73)
        cls.memo = Memo(Path(cls.temporary.name))
        engine = Engine(cls.cfg, cls.memo, log=lambda _: None)
        cls.completed = {(r, f): engine.checkpoint(cls.data, r, f) for r in range(2) for f in range(5)}
        cls.result = aggregate(cls.completed, cls.cfg, cls.data.index.split_unit_id.to_numpy(), "SYNTHETIC_VERIFICATION")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_both_repeats_count_same_people_once(self):
        self.assertEqual(self.result["n"], 1200)
        self.assertEqual(len(self.result["scores"]), 26)
        self.assertIn("Same participants", self.result["repeat_scope"])

    def test_all_models_replay_after_closed_file_checkpoint(self):
        part = self.memo.get("r0_f0_COMPLETE")
        for name, model in part["models"].items():
            np.testing.assert_allclose(model.predict(), part["prediction"][name], atol=1e-12)
            self.assertEqual(model.design.role.raw, {})

    def test_resume_does_not_fit_complete_units(self):
        class NoFitEngine(Engine):
            def outer(self, *_):
                raise AssertionError("Completed unit was fitted again")
        value = NoFitEngine(self.cfg, self.memo).checkpoint(self.data, 0, 0)
        self.assertEqual(len(value["rows"]), len(self.completed[0, 0]["rows"]))

    def test_reports_are_compact_and_explicitly_synthetic(self):
        for mode in (1, 2, 3):
            text = screen(self.result, mode)
            self.assertLessEqual(len(text.splitlines()), 24)
            self.assertTrue(text.startswith("SYNTHETIC_VERIFICATION"))
            self.assertNotIn("synthetic_household_", text)

    def test_aggregate_is_strictly_json_serializable(self):
        encoded = json.dumps(self.result, allow_nan=False)
        self.assertEqual(json.loads(encoded)["n"], 1200)

    def test_primary_screen_uses_family_intervals(self):
        self.assertIn("family95%CI0", screen(self.result, 1))
        for info in self.result["inference"]:
            if info["status"] == "OK":
                for marker in info["markers"]:
                    pointwise = marker["interaction_iqr_ci_mg_dl"]
                    family = marker["interaction_iqr_ci_bonferroni95_mg_dl"]
                    self.assertLessEqual(family[0], pointwise[0])
                    self.assertGreaterEqual(family[1], pointwise[1])

    def test_selected_option_nonzero_sd_and_prediction_gain_are_separate(self):
        completed = {}
        name = "ecology_linear_oral"
        for key, original in self.completed.items():
            part = dict(original)
            part["choices"] = dict(original["choices"], **{name: .03})
            part["interaction_component"] = dict(original["interaction_component"], **{name: np.zeros(len(original["rows"]))})
            part["total_contrast"] = dict(original["total_contrast"], **{name: original["total_contrast"]["main"].copy()})
            completed[key] = part
        result = aggregate(completed, self.cfg, self.data.index.split_unit_id.to_numpy(), "SYNTHETIC_VERIFICATION")
        for row in result["scores"]:
            if row["model"] == name:
                self.assertEqual(row["selected_interaction_folds"], 5)
                self.assertEqual(row["nonzero_product_contrast_folds"], 0)
                self.assertEqual(row["nonzero_correction_contrast_folds"], 0)
                self.assertEqual(row["interaction_component_sd_mg_dl"], 0)
                actual = next(v for v in self.result["scores"] if v["model"] == name and v["repeat"] == row["repeat"])
                self.assertEqual(row["delta_R2"], actual["delta_R2"])

    def test_incomplete_and_duplicate_rows_are_rejected(self):
        incomplete = dict(self.completed)
        incomplete.pop((1, 4))
        with self.assertRaises(ValueError):
            aggregate(incomplete, self.cfg, self.data.index.split_unit_id.to_numpy(), "SYNTHETIC_VERIFICATION")
        part = self.completed[0, 0]
        with self.assertRaises(ValueError):
            ordered([part, part], "y")

    def test_negative_results_survive_report(self):
        value = copy.deepcopy(self.result)
        row = next(row for row in value["scores"] if row["model"] == "ecology_linear_oral" and row["repeat"] == 0)
        row["delta_R2"] = -.123456
        self.assertIn("-0.123456", screen(value, 2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        unittest.main(verbosity=1)
