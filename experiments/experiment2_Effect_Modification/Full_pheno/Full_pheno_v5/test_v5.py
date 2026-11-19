"""Generated-data checks of matching, nested selection and reporting."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from run_v5 import execute, read_complete, write_complete
import breadth
import protocol as p
import run
import run_v2
import run_v3
import run_v4
import v5_design as d
from source import Outcome, Snapshot, Study, required_files
from test_full_pheno import fixture
from v3_inputs import InnerContexts, read_sites
from v4_score import Task
from v5_engine import fit_outer
from v5_models import interaction_masks, orthogonalize
from v5_report import render
from v5_score import inference
from v5_select import Loss, anchored, choose


class FullPhenoV5Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp = TemporaryDirectory(prefix="hpp_v5_generated_")
        cls.addClassCleanup(temp.cleanup)
        root = Path(temp.name) / p.HANDOFF
        fixture(root)
        cls.snapshot = Snapshot(root, required_files(p.MODES["all"]))
        cls.study = Study(cls.snapshot, with_upf=True, expected=None)
        cls.outcomes = {e: cls.study.outcome(e) for e in p.ENDPOINTS}
        cls.contexts = InnerContexts(cls.study, read_sites(cls.study))

    def test_candidate_library(self):
        self.assertEqual((len(d.LIBRARY), len(d.INDEX), len(d.NEW)), (422, 422, 108))
        self.assertEqual(len(d.WORKFLOWS), 14)
        self.assertEqual(len(set(d.CORE)), len(d.CORE))
        self.assertTrue(set(d.CORE) <= set(d.LIBRARY))
        self.assertIn(d.FIXED_TENSOR, d.NEW)
        self.assertEqual(d.WORKFLOWS[0], d.REFERENCE)

    def test_interaction_masks_align_with_product_order(self):
        masks, charge = interaction_masks(np.zeros((9, 4)), np.zeros((9, 4)), 2, 2)
        np.testing.assert_array_equal(np.flatnonzero(masks["dose"]), [0, 1, 4, 5, 8, 9, 12, 13])
        np.testing.assert_array_equal(np.flatnonzero(masks["marker"]), np.arange(8))
        np.testing.assert_array_equal(charge.reshape(4, 4), [[1, 1, 4, 4], [1, 1, 4, 4],
                                                           [4, 4, 16, 16], [4, 4, 16, 16]])
        self.assertEqual(masks["joint"].sum(), 16)

    def test_training_projection_excludes_marginal_terms(self):
        rng = np.random.default_rng(49)
        n, size = 80, 100
        x, m = rng.normal(size=(2, size))
        base = np.column_stack((np.ones(size), x, m, abs(x), abs(m)))
        candidate = np.column_stack((x*m, x*abs(m), abs(x)*m))
        additive = base[:n] @ np.array([1., 2., -1., -.9, .8])
        w = rng.uniform(.15, 1., size=n)
        z, gram, rhs = orthogonalize(base, candidate, additive, n, w)
        self.assertGreater(float(np.linalg.norm(gram)), .05)
        np.testing.assert_allclose(rhs, np.zeros(3), atol=1e-13)
        target = additive + candidate[:n] @ [.8, -.2, .3]
        _, core, score = orthogonalize(base, candidate, target, n, w)
        np.testing.assert_allclose(np.linalg.solve(core, score), [.8, -.2, .3], atol=1e-12)
        base[n:, 1] += 1.
        candidate[n:] += .4
        test, other, rhs2 = orthogonalize(base, candidate, additive, n, w)
        self.assertFalse(np.array_equal(z, test))
        np.testing.assert_array_equal(gram, other)
        np.testing.assert_array_equal(rhs, rhs2)

    def test_three_and_four_fold_gates(self):
        candidate, other = d.NEW[:2]
        dot = np.zeros((4, len(d.LIBRARY)))
        norm = np.zeros_like(dot)
        j, k = d.INDEX[candidate], d.INDEX[other]
        dot[:, j], norm[:, j] = [.0004, .0004, .0004, -.00005], .0002
        dot[:, k], norm[:, k] = .000001, .0002
        mask = np.ones(len(d.LIBRARY), bool)
        fraction = np.full(4, .25)
        info = Loss(dot, norm, np.full(4, .001), fraction, mask)
        self.assertEqual(choose(info, (candidate, other)), (candidate, .5))
        self.assertEqual(anchored(choose(info, (candidate, other), strict=True)), {d.PC5: .5})
        self.assertEqual(anchored((candidate, .5)), {d.PC5: .25, candidate: .5})
        self.assertEqual(anchored((d.PC5, .5)), {d.PC5: .75})

    def test_outer_holdout_never_changes_predictions_or_selection(self):
        fold = 0
        before = fit_outer(self.study, self.contexts, self.outcomes, "UPF", fold)
        changed = {}
        for e, raw in self.outcomes.items():
            y = raw.y.copy()
            y[before[e].rows] += .0001
            changed[e] = Outcome(y, raw.unit)
        after = fit_outer(self.study, self.contexts, changed, "UPF", fold)
        for e, a in before.items():
            b = after[e]
            self.assertNotEqual(float(a.y.sum()), float(b.y.sum()))
            self.assertEqual(a.weights, b.weights)
            for key in ("rows", "no_diet", "off", "correction"):
                np.testing.assert_array_equal(getattr(a, key), getattr(b, key))

    def test_three_arm_ratios_and_small_diet_denominator(self):
        tasks = []
        for endpoint, a, b, c in (("D_Glu", .40, .50, .55), ("D_VAT", .50, .50, .5002)):
            y = np.tile([-2., -1., 1., 2.], 20)
            groups = np.arange(len(y))//4
            err, diet, interaction = (1.-b)*y, (b-a)*y, (c-b)*y
            gain = interaction*(2.*err-interaction)
            columns = np.column_stack((np.ones(len(y)), y, y**2, err**2, diet*(2.*err+diet),
                                       np.repeat(gain[:, None], len(d.WORKFLOWS), axis=1)))
            loss = np.column_stack([np.bincount(groups, weights=v) for v in columns.T])
            tasks.append(Task(("UPF", endpoint), len(y), 20, loss,
                              np.ones((5, len(d.WORKFLOWS))), [], 0.))
        first, *_, small = inference(tasks, draws=60)["rows"]
        self.assertAlmostEqual(first["r2_no_diet"], .64)
        self.assertAlmostEqual(first["r2_off"], .75)
        self.assertAlmostEqual(first["r2_on"], .7975)
        self.assertAlmostEqual(first["full_share_pct"], 100*.0475/.7975)
        self.assertAlmostEqual(first["diet_share_pct"], 100*.0475/.1575)
        self.assertAlmostEqual(small["diet_share_point_pct"], 100.)
        self.assertIsNone(small["diet_share_pct"])
        self.assertIsNotNone(small["full_share_pct"])

    def test_generated_chain_replays_and_writes_all_pairs(self):
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
            refs3 = {((r["exposure"], r["endpoint"]), r["workflow"]): r for r in v3["results"]}
            v4, _ = run_v4.execute(self.study, refs3, draws)
            refs4 = {((r["exposure"], r["endpoint"]), r["workflow"]): r for r in v4["rows"]}
            summary, tables = execute(self.study, refs4, draws)
            release = {"generated": True}
            write_complete(archive, archive/"registration.json", release, summary, tables)
            saved = read_complete(archive, release)
            self.assertEqual(saved, summary)
            self.assertEqual(render(saved), render(summary))
        self.assertLess(summary["v4_replay_max_abs_r2"], 1e-12)
        self.assertEqual([len(t) for t in tables], [252, 234, 7138, 1260])
        self.assertEqual(summary["simultaneous_cells"], 738)
        self.assertEqual(len(summary["fixed_ranking"]), 386)
        anchor = [r for r in summary["rows"] if (r["exposure"], r["endpoint"]) == d.ANCHOR]
        self.assertEqual(len(anchor), 14)
        self.assertEqual(len({r["delta_pp"] for r in anchor}), 1)
        self.assertEqual(len({r["total_diet_pp"] for r in anchor}), 1)
        for r in summary["rows"]:
            self.assertAlmostEqual(r["total_diet_pp"], 100*(r["r2_on"]-r["r2_no_diet"]), places=11)
            self.assertAlmostEqual(r["total_diet_pp"], r["main_diet_pp"]+r["delta_pp"], places=11)
        self.assertIn("FULL_PHENO_V5_SCREEN_END", render(summary))
