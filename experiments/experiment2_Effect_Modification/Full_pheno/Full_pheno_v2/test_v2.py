"""Generated data tests for the v2 replay, shared fits, inference and ranking."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

import design as d
import protocol as p
import run as v1_run
from breadth import (Fold, Task, collect, evaluate_split, fit_profile, joint_bootstrap,
                     make_summary, point_scores, rank, replay)
from run_v2 import registered_results, render, v1_records
from source import AuditError, Outcome, Snapshot, Study, required_files
from test_full_pheno import fixture


class BreadthTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = TemporaryDirectory(prefix="hpp_breadth_generated_")
        cls.addClassCleanup(tmp.cleanup)
        cls.root = Path(tmp.name) / p.HANDOFF
        fixture(cls.root)
        cls.snapshot = Snapshot(cls.root, required_files(p.MODES["all"]))
        cls.study = Study(cls.snapshot, with_upf=True, expected=None)
        cls.outcomes = {e: cls.study.outcome(e) for e in p.ENDPOINTS}

    def test_adjusted_ridge_matches_direct_least_squares(self):
        rng = np.random.default_rng(46)
        x = np.column_stack((np.ones(500), rng.normal(size=(500, 4))))
        n = 375
        z = rng.normal(size=(500, 3)) + x[:, :3]
        y = 100 + z[:n] @ [1.5, -.9, .25] + x[:n] @ [0, 2, 1, 4, 0] + rng.normal(size=n)
        inverse = np.linalg.pinv(x[:n])
        residual = y - x[:n] @ (inverse @ y)
        prof = fit_profile(z, x, inverse, residual)
        penalty = .1
        joined = np.column_stack((x, z))
        a = joined[:n]
        beta = np.linalg.solve(a.T @ a + np.diag([0.] * x.shape[1] + [n * penalty] * z.shape[1]), a.T @ y)
        obs = x[n:] @ (inverse @ y) + prof.fit(penalty)
        np.testing.assert_allclose(obs, joined[n:] @ beta, atol=1e-10, rtol=0.)

    def test_other_endpoints_held_out_values_do_not_leak(self):
        fold = 0
        outcomes = dict(self.outcomes)
        first = evaluate_split(self.study, "fiber_density", outcomes, fold)
        for e, original in outcomes.items():
            y = original.y.copy()
            y[first[e].rows] += .001
            outcomes[e] = Outcome(y, original.unit)
        second = evaluate_split(self.study, "fiber_density", outcomes, fold)
        self.assertFalse(np.array_equal(first["D_Glu"].y, second["D_Glu"].y))
        for e in outcomes:
            np.testing.assert_array_equal(first[e].rows, second[e].rows)
            for name in (d.BASE, *d.NAMES):
                np.testing.assert_array_equal(first[e].preds[name], second[e].preds[name])

    def test_shared_prediction_uses_other_training_outcomes(self):
        fold = 1
        outcomes = dict(self.outcomes)
        first = evaluate_split(self.study, "fiber_density", outcomes, fold)
        e = "D_TotalFat"
        y = outcomes[e].y.copy()
        mask = self.study.backgrounds[fold].role & np.isfinite(y) & np.isfinite(self.study.numeric["fiber_density"])
        y[mask] += .2 * self.study.backgrounds[fold].pcs[mask, p.N_PC] * self.study.numeric["fiber_density"][mask]
        # This generated experiment holds admission fixed and changes a donor target only.
        old = self.study.eligible
        def fixed(x, endpoint, f, clean):
            if endpoint == e:
                clean = p_clean(self.outcomes[e].y, self.study.backgrounds[f].role)
            return old(x, endpoint, f, clean)
        from unittest.mock import patch
        from source import clean_outcome as p_clean
        with patch.object(self.study, "eligible", fixed):
            second = evaluate_split(self.study, "fiber_density", outcomes | {e: Outcome(y, outcomes[e].unit)}, fold)
        for name in (d.BASE, *d.REFS, d.RECIPES[0].name):
            np.testing.assert_array_equal(first["D_Glu"].preds[name], second["D_Glu"].preds[name])
        self.assertGreater(np.max(abs(first["D_Glu"].preds[d.SHARED[0]] -
                                      second["D_Glu"].preds[d.SHARED[0]])), 1e-5)

    def test_joint_resampling_preserves_identical_pair_deltas(self):
        a = np.zeros((100, 4 + len(d.NAMES)))
        rng = np.random.default_rng(10)
        y = rng.normal(size=100)
        a[:, :4] = np.column_stack((np.ones(len(y)), y, y**2, .16*y**2))
        a[:, 4:] = (-.09*y**2)[:, None]
        first = Task("fiber_density", "D_Glu", 100, 100, [], a)
        second = Task("UPF", "D_Glu", 100, 100, [], a.copy())
        boot, stable = joint_bootstrap((first, second), 100, 416)
        np.testing.assert_allclose(boot[:, 0, :], boot[:, 1, :], atol=1e-15, rtol=0.)
        self.assertTrue(stable.all())
        self.assertTrue(np.all(boot < 0))

    def test_paired_scores_preserve_small_losses_and_use_the_frozen_fold(self):
        rng = np.random.default_rng(94)
        y = rng.normal(size=150)
        base = y * .6
        on = base + y * .1
        tiny = np.nextafter(base, base + y)
        preds = {name: base for name in (d.BASE, *d.NAMES)}
        preds.update({d.REFS[0]: on, d.REFS[1]: tiny})
        codes = np.arange(len(y)) // 2
        fold_ids = codes % len(p.FOLDS)
        parts = []
        for fold in p.FOLDS:
            rows = np.flatnonzero(fold_ids == fold)
            parts.append(Fold(rows, y[rows], {k: q[rows] for k, q in preds.items()},
                              dict(fold=fold, held_people=len(rows), held_households=len(np.unique(codes[rows])))))
        task = collect("fiber_density", "D_Glu", parts, fold_ids, codes, len(np.unique(codes)))
        r2, delta = point_scores(task)
        sst = np.sum((y - y.mean())**2)
        np.testing.assert_allclose(r2[:2], [1 - np.sum((y - q)**2)/sst for q in (base, on)], atol=1e-14)
        self.assertAlmostEqual(delta[0], r2[1] - r2[0], places=14)
        self.assertGreater(delta[1], 0.)
        self.assertEqual(delta[2], 0.)
        boot, _ = joint_bootstrap([task], 120, 45)
        self.assertTrue(np.all(boot[:, 0, :2] > 0.))
        self.assertTrue(np.all(boot[:, 0, 2:] == 0.))
        altered_folds = fold_ids.copy()
        altered_folds[parts[0].rows[0]] = 1
        with self.assertRaisesRegex(AuditError, "V2_HELD_FOLD_MISMATCH"):
            collect("fiber_density", "D_Glu", parts, altered_folds, codes, len(np.unique(codes)))

    def test_breadth_outranks_one_large_result_and_demands_eighteen_cells(self):
        import itertools
        first, second = d.REFS
        rows = []
        for recipe, (i, (x, e)) in itertools.product(d.NAMES, enumerate(p.MODES["all"])):
            pp = 2. if recipe == first and i == 0 else .15 if recipe == second and i % 3 != 0 else -.05
            rows.append(dict(recipe=recipe, exposure=x, endpoint=e, domain=d.domain(e),
                             delta_r2_pp=pp, simultaneous95_low_pp=pp-.2))
        best = rank(rows)[0]
        self.assertEqual(best["recipe"], second)
        self.assertEqual(best["at_least_010"], 12)
        self.assertTrue(best["majority"])
        self.assertFalse(best["corrected_majority"])
        with self.assertRaisesRegex(AuditError, "V2_INCOMPLETE_BREADTH_DENOMINATOR"):
            rank(rows[:-1])

    def test_same_eighteen_comparisons_replay_separate_v1_run(self):
        draws = 60
        with TemporaryDirectory(prefix="hpp_v1_v2_replay_") as tmp, redirect_stdout(StringIO()):
            archive = Path(tmp)
            release = v1_run.read_release()
            v1_run.batch(self.study, p.MODES["all"], archive, "all", release, dict(synthetic=True),
                         draws=draws, v10_root=None)
            stored, _ = v1_records(archive, self.snapshot, release)
            tasks, errors = registered_results(self.study, stored, progress=False)
        self.assertEqual(len(tasks), 18)
        self.assertLess(max(errors.values()), 1e-13)
        summary = make_summary(tasks, draws=draws)
        self.assertEqual(summary["simultaneous_cells"], len(d.NAMES)*18)
        self.assertEqual(len(summary["ranking"]), len(d.NAMES))
        self.assertIn("FULL_PHENO_V2_SCREEN_END", render(summary))
        self.assertFalse(any("9100001" in str(r) for r in summary["details"]))
        old = stored[tasks[0].exposure, tasks[0].endpoint]
        with self.assertRaisesRegex(AuditError, "V1_REPLAY_COHORT_DIFFERS"):
            replay(tasks[0], dict(old, people=old["people"]+1))
        stats, _ = point_scores(tasks[0])
        self.assertAlmostEqual(stats[1], old["score"]["models"]["three_species"]["r2"], places=12)
