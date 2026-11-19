"""Generated-data checks for v4 estimates, nesting and release artifacts."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

import breadth
import protocol as p
import run
import run_v2
import run_v3
import v4_design as d
from run_v4 import execute, read_complete, write_complete
from source import AuditError, Outcome, Snapshot, Study, required_files
from models import products
from test_full_pheno import fixture
from v3_inputs import InnerContexts, read_sites
from v4_engine import fit_outer
from v4_models import no_diet_main, ridge_forecast, robust_weights
from v4_report import render
from v4_score import Task, inference
from v4_select import Loss, capped_mix, peer_choices, single


class FullPhenoV4Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp = TemporaryDirectory(prefix="hpp_v4_generated_")
        cls.addClassCleanup(temp.cleanup)
        root = Path(temp.name) / p.HANDOFF
        fixture(root)
        cls.snapshot = Snapshot(root, required_files(p.MODES["all"]))
        cls.study = Study(cls.snapshot, with_upf=True, expected=None)
        cls.outcomes = {e: cls.study.outcome(e) for e in p.ENDPOINTS}
        cls.contexts = InnerContexts(cls.study, read_sites(cls.study))

    def test_candidate_grid_and_workflows(self):
        self.assertEqual(len(d.INDEX), len(d.LIBRARY))
        self.assertEqual((len(d.CURVES), len(d.ROBUST), len(d.LIBRARY)), (64, 48, 314))
        self.assertEqual(len(d.WORKFLOWS), 13)
        self.assertTrue(set(d.MIX_CORE) <= set(d.LIBRARY))

    def test_no_diet_removes_exact_dose_and_age_sex_products(self):
        rng = np.random.default_rng(18)
        n, size = 50, 65
        host = rng.normal(size=(size, 3))
        smoke = rng.choice(["no", "yes", "__UNKNOWN__"], size=size)
        study = SimpleNamespace(smoking=smoke)
        rows = np.arange(size)
        head = np.column_stack((np.ones(size), host, smoke == "no", smoke == "yes"))
        tail = rng.normal(size=(size, 12))
        for _ in range(2):
            dose = rng.normal(size=(size, 2))
            full = np.column_stack((head, dose, products(dose, host[:, :2]), tail))
            reduced = no_diet_main(study, rows, n, full, dose)
            np.testing.assert_array_equal(reduced, np.column_stack((head, tail)))
        with self.assertRaisesRegex(AuditError, "V4_DIET_BLOCK_MISALIGNED"):
            no_diet_main(study, rows, n, full, dose+1.)

    def test_robust_regression_reduces_outlier_leverage(self):
        z = np.ones((13, 1))
        y = np.r_[np.tile([-2., 0., 2.], 4), 50.]
        w = robust_weights(y)
        self.assertLess(w[-1], .1)
        a = ridge_forecast(z, y, z[:1], [.01])[0]
        b = ridge_forecast(z, y, z[:1], [.01], weights=w)[0]
        self.assertLess(abs(b), abs(a))
        np.testing.assert_allclose(b, np.sum(w*y)/(1.01*w.sum()), atol=1e-13)

    def test_capped_mixture_and_peer_selection(self):
        k, j = [d.INDEX[name] for name in (d.PC5, d.ORAL)]
        shape = (4, len(d.LIBRARY))
        a, b = np.zeros(shape), np.zeros(shape)
        a[:, k], b[:, k] = .01, .02
        a[:, j], b[:, j] = .03, .04
        cross = np.zeros(shape)
        cross[:, k] = b[:, k]
        mask = np.ones(len(d.LIBRARY), dtype=bool)
        fraction = np.full(4, .25)
        info = Loss(a, b, cross, mask, fraction)
        combo = capped_mix(info)
        self.assertEqual(set(combo), {d.PC5, d.ORAL})
        self.assertLessEqual(combo[d.ORAL], .35)
        self.assertLessEqual(sum(combo.values()), 1.)
        null = Loss(-a, b, cross, mask, fraction)
        self.assertEqual(capped_mix(null), {})
        own = single(null, d.CORE)
        donor, checked = peer_choices("D_VAT", null, {"D_TotalFat": info, "D_VAT": null}, own)
        self.assertEqual(donor, {d.ORAL: single(info, d.CORE)[d.ORAL]})
        self.assertEqual(checked, {})

    def test_maximin_uses_each_held_folds_own_sst(self):
        first, second = d.NEW[:2]
        gains = np.zeros((4, len(d.LIBRARY)))
        gains[:, d.INDEX[first]] = [.00007, .0023, .0023, .0023]
        gains[:, d.INDEX[second]] = [.00011, .00060, .0032, .0032]
        fraction = np.array([.02, .40, .25, .30])
        loss = Loss(.75*gains, .50*gains, np.zeros_like(gains), np.ones(len(d.LIBRARY), bool), fraction)
        self.assertEqual(single(loss, (first, second), maximin=True), {first: 1.})

    def test_held_outcomes_do_not_change_weights_or_predictions(self):
        fold = 0
        before = fit_outer(self.study, self.contexts, self.outcomes, "UPF", fold)
        changed = {}
        for e, raw in self.outcomes.items():
            y = raw.y.copy()
            y[before[e].rows] += .0001
            changed[e] = Outcome(y, raw.unit)
        after = fit_outer(self.study, self.contexts, changed, "UPF", fold)
        for e in self.outcomes:
            a, b = before[e], after[e]
            self.assertEqual(a.weights, b.weights)
            self.assertFalse(np.array_equal(a.y, b.y))
            for key in ("rows", "off", "no_diet", "corrections"):
                np.testing.assert_array_equal(getattr(a, key), getattr(b, key))

    def test_three_arm_ratios_and_small_diet_denominator(self):
        tasks = []
        for endpoint, a, b, c in (("D_Glu", .40, .50, .55), ("D_VAT", .50, .50, .5002)):
            y = np.tile([-2., -1., 1., 2.], 20)
            groups = np.arange(len(y))//4
            err = (1.-b)*y
            diet = (b-a)*y
            interaction = (c-b)*y
            columns = np.column_stack((np.ones(len(y)), y, y**2, err**2, diet*(2.*err+diet),
                                       np.tile((interaction*(2.*err-interaction))[:, None], (1, len(d.WORKFLOWS)))))
            loss = np.column_stack([np.bincount(groups, weights=v) for v in columns.T])
            tasks.append(Task(("UPF", endpoint), len(y), 20, loss,
                              np.ones((5,len(d.WORKFLOWS))), [], 0.))
        result = inference(tasks, draws=60)["rows"]
        first = result[0]
        self.assertAlmostEqual(first["r2_no_diet"], .64)
        self.assertAlmostEqual(first["r2_off"], .75)
        self.assertAlmostEqual(first["r2_on"], .7975)
        self.assertAlmostEqual(first["full_share_pct"], 100*.0475/.7975)
        self.assertAlmostEqual(first["diet_share_pct"], 100*.0475/.1575)
        small = result[-1]
        self.assertAlmostEqual(small["diet_share_point_pct"], 100.)
        self.assertIsNone(small["diet_share_pct"])
        self.assertIsNotNone(small["full_share_pct"])

    def test_diet_share_fails_when_only_the_point_denominator_passes(self):
        y = np.tile([-2., -1., 1., 2.], (20, 1))
        e = .50*y
        diff = (.50 - np.r_[np.full(6, .40), np.full(14, .62)])[:, None]*y
        c = .05*y
        inc = np.sum(c*(2.*e-c), axis=1)
        moments = np.column_stack((np.full(20, 4), y.sum(axis=1), (y**2).sum(axis=1), (e**2).sum(axis=1),
                                   (diff*(2.*e+diff)).sum(axis=1),
                                   np.tile(inc[:, None], (1, len(d.WORKFLOWS)))))
        task = Task(("UPF", "D_Glu"), 80, 20, moments, np.ones((5, len(d.WORKFLOWS))), [], 0.)
        row = inference([task], draws=200)["rows"][0]
        self.assertGreater(row["total_diet_pp"], .10)
        self.assertIsNotNone(row["diet_share_point_pct"])
        self.assertIsNone(row["diet_share_pct"])
        self.assertIsNotNone(row["full_share_pct"])

    def test_registered_replay_and_complete_outputs(self):
        draws = 60
        with TemporaryDirectory() as dest, redirect_stdout(StringIO()):
            archive = Path(dest)
            v1_release = run.read_release()
            run.batch(self.study, p.MODES["all"], archive, "all", v1_release,
                      {"synthetic": True}, draws=draws, v10_root=None)
            v1, _ = run_v2.v1_records(archive, self.snapshot, v1_release)
            v2_tasks, _ = run_v2.registered_results(self.study, v1, progress=False)
            v2 = {key: {} for key in p.MODES["all"]}
            for row in breadth.make_summary(v2_tasks, draws=draws)["details"]:
                v2[row["exposure"], row["endpoint"]][row["recipe"]] = row
            v3, _ = run_v3.execute(self.study, v1, v2, draws)
            prior = {((r["exposure"], r["endpoint"]), r["workflow"]): r for r in v3["results"]}
            summary, tables = execute(self.study, prior, draws)
            release = {"generated": True}
            write_complete(archive, archive/"registration.json", release, summary, tables)
            saved = read_complete(archive, release)
            self.assertEqual(saved, summary)
            self.assertEqual(render(saved), render(summary))
        self.assertLess(summary["v3_replay_max_abs_r2"], 1e-12)
        self.assertEqual([len(t) for t in tables], [234, 216, 5302, 1170])
        self.assertEqual(summary["simultaneous_cells"], 684)
        self.assertEqual(len(summary["fixed_ranking"]), 278)
        self.assertEqual(set(summary["share_peaks"]), {"full_share_pct", "diet_share_pct"})
        anchor = [r for r in summary["rows"] if (r["exposure"], r["endpoint"]) == d.ANCHOR
                  and r["workflow"] not in d.UNIFORM]
        self.assertEqual(len({r["delta_pp"] for r in anchor}), 1)
        for r in summary["rows"]:
            self.assertAlmostEqual(r["total_diet_pp"], 100*(r["r2_on"]-r["r2_no_diet"]), places=11)
            self.assertAlmostEqual(r["total_diet_pp"], r["main_diet_pp"]+r["delta_pp"], places=11)
        self.assertIn("FULL_PHENO_V4_SCREEN_END", render(summary))
