"""Generated-data regressions for the nested workflow and its comparison rules."""

from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

import breadth
import protocol as p
import run
import run_v2
import v3_design as d
from run_v3 import execute, read_complete, v2_references, write_complete
from source import Outcome, Snapshot, Study, required_files
from test_full_pheno import fixture
from v3_blend import InnerLoss, choose, diverse
from v3_engine import fit_outer
from v3_inputs import InnerContexts, Site, fit_site, read_sites
from v3_report import render
from v3_score import contrasts


class FullPhenoV3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = TemporaryDirectory(prefix="hpp_v3_generated_")
        cls.addClassCleanup(folder.cleanup)
        cls.root = Path(folder.name) / p.HANDOFF
        fixture(cls.root)
        cls.snapshot = Snapshot(cls.root, required_files(p.MODES["all"]))
        cls.study = Study(cls.snapshot, with_upf=True, expected=None)
        cls.sites = read_sites(cls.study)
        cls.contexts = InnerContexts(cls.study, cls.sites)
        cls.outcomes = {e: cls.study.outcome(e) for e in p.ENDPOINTS}

    def test_complete_candidate_grid_and_scope(self):
        self.assertEqual(len(set(d.LIBRARY)), 202)
        self.assertEqual(len(d.V2_NAMES), 54)
        self.assertEqual(len(d.DOMAIN_NAMES), 36)
        self.assertEqual(len(d.WORKFLOWS), 8)
        self.assertEqual(len(p.MODES["all"]), 18)

    def test_training_pca_unaffected_by_outside_abundances(self):
        train = (self.study.fold != 0) & (self.study.fold != 1)
        source = self.sites["gut"]
        base, support = fit_site("gut", source, self.study.keys[0], train)
        moved = source.abundance.copy()
        moved[~train] = moved[~train, ::-1]
        changed, sup = fit_site("gut", Site(moved, source.taxa), self.study.keys[0], train)
        np.testing.assert_array_equal(base[train], changed[train])
        np.testing.assert_array_equal(support[train], sup[train])
        self.assertGreater(np.max(abs(base[~train] - changed[~train])), .1)

    def test_households_are_disjoint_and_calibration_is_outer(self):
        for outer in p.FOLDS:
            valid = []
            for inner in p.FOLDS:
                if inner == outer:
                    continue
                view = self.contexts.split(outer, inner)
                background = view.backgrounds[inner]
                mask = view.active & ~background.role
                self.assertIs(background.archived["UPF"], self.study.backgrounds[outer].archived["UPF"])
                self.assertFalse(np.intersect1d(view.family[mask], view.family[background.role]).size)
                self.assertFalse(np.intersect1d(view.family[mask | background.role],
                                                 view.family[self.study.fold == outer]).size)
                valid.extend(np.flatnonzero(mask))
            self.assertEqual(sorted(valid), list(np.flatnonzero(self.study.fold != outer)))

    def test_blend_shrinks_and_combines_complementary_corrections(self):
        j, k = [d.LIBRARY_INDEX[name] for name in ("v1_panel", d.PC5_REFERENCE)]
        cols = len(d.LIBRARY)
        gram, b = np.zeros((cols, cols)), np.zeros(cols)
        gram[j, j] = gram[k, k] = .03
        b[[j, k]] = .015
        loss = InnerLoss(gram, b, np.tile(gram/4, (4, 1, 1)), np.tile(b/4, (4, 1)),
                         np.ones(cols, dtype=bool))
        fitted = choose(loss)
        w = fitted["blend_guarded"]
        self.assertEqual(set(np.flatnonzero(w)), {j, k})
        np.testing.assert_allclose(w[[j, k]], [.5/1.1]*2, atol=2e-6, rtol=0)
        self.assertGreater(2*b @ w - w @ gram @ w, .014)
        self.assertFalse(fitted["select"].any())  # The unscaled single corrections both break even.
        null = InnerLoss(gram, -b, loss.fold_gram, -loss.fold_rhs, loss.selectable)
        self.assertFalse(choose(null)["blend_guarded"].any())
        another = d.LIBRARY_INDEX["panel/linear/ridge_1"]
        mask = np.zeros(cols, bool)
        mask[[j, k, another]] = True
        self.assertEqual(set(diverse(np.ones(cols), mask, 3)), {j, k})

    def test_guarded_mixture_must_win_three_folds_after_weighting(self):
        from v3_blend import by_domain

        j, k = [d.LIBRARY_INDEX[name] for name in ("v1_panel", d.PC5_REFERENCE)]
        coeff = np.zeros((4, len(d.LIBRARY)))
        coeff[:, j] = [.10, .10, .10, -.22]
        coeff[:, k] = [-.22, .10, .10, .10]
        gram = np.einsum("fi,fj->fij", coeff, coeff)/4.
        rhs = coeff/4.
        allowed = np.isin(np.arange(len(d.LIBRARY)), (j, k))
        loss = InnerLoss(gram.sum(axis=0), rhs.sum(axis=0), gram, rhs, allowed)
        for index in np.flatnonzero(allowed):
            self.assertEqual(((2.*coeff[:, index]-coeff[:, index]**2) > 0.).sum(), 3)
        selected = choose(loss)
        self.assertEqual((loss.fold_gains(selected["blend"]) > 0.).sum(), 2)
        domain = by_domain({"D_TotalFat": loss, "D_VAT": loss})
        for chosen in (selected["blend_guarded"], domain["D_TotalFat"], domain["D_VAT"]):
            self.assertEqual(np.count_nonzero(chosen), 1)
            gains = loss.fold_gains(chosen)
            self.assertEqual((gains > 0.).sum(), 3)
            self.assertGreaterEqual(gains.sum(), d.GUARD_GAIN)

    def test_held_outcome_disappears_from_selection_and_prediction(self):
        for exposure in p.EXPOSURES:
            fold = 0
            first = fit_outer(self.study, self.contexts, self.outcomes, exposure, fold)
            changed = {}
            for e, o in self.outcomes.items():
                y = o.y.copy()
                y[first[e].candidate.rows] += .0001
                changed[e] = Outcome(y, o.unit)
            second = fit_outer(self.study, self.contexts, changed, exposure, fold)
            for e in self.outcomes:
                f, g = first[e], second[e]
                self.assertFalse(np.array_equal(f.candidate.y, g.candidate.y))
                self.assertEqual(f.weights, g.weights)
                np.testing.assert_array_equal(f.candidate.rows, g.candidate.rows)
                np.testing.assert_array_equal(f.candidate.off, g.candidate.off)
                np.testing.assert_array_equal(f.candidate.correction, g.candidate.correction)
                np.testing.assert_array_equal(f.correction, g.correction)
            self.assertFalse(first["D_Glu"].candidate.selectable[-len(d.DOMAIN_NAMES):].any())

    def test_registered_replay_anchor_and_paired_results(self):
        draws = 60
        with TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            archive = Path(tmp)
            run.batch(self.study, p.MODES["all"], archive, "all", run.read_release(), {"synthetic": True},
                      draws=draws, v10_root=None)
            v1, _ = run_v2.v1_records(archive, self.snapshot, run.read_release())
            old, _ = run_v2.registered_results(self.study, v1, progress=False)
            ref = {key: {} for key in p.MODES["all"]}
            for row in breadth.make_summary(old, draws=draws)["details"]:
                ref[row["exposure"], row["endpoint"]][row["recipe"]] = row
            summary, tables = execute(self.study, v1, ref, draws)
            registration = archive / "registration.json"
            release = {"generated": True}
            write_complete(archive, registration, release, summary, tables)
            self.assertEqual(read_complete(archive, release), summary)
        self.assertLess(summary["v2_replay_max_abs_r2"], 1e-12)
        self.assertEqual([len(t) for t in tables], [144, 144, 17*202-36, 720, 9])
        self.assertEqual(len(summary["library_peaks"]), 17)
        self.assertEqual(summary["simultaneous_cells"], 18*16)
        anchor = [r for r in summary["results"] if (r["exposure"], r["endpoint"]) == d.ANCHOR]
        self.assertEqual(len({r["delta_pp"] for r in anchor}), 1)
        self.assertAlmostEqual(anchor[0]["delta_pp"], ref[d.ANCHOR][d.ANCHOR_RECIPE]["delta_r2_pp"], places=12)
        for r in summary["results"]:
            self.assertAlmostEqual(r["delta_pp"], r["aligned_pp"] - r["correction_cost_pp"], places=11)
            self.assertAlmostEqual(r["delta_pp"], 100*(r["r2_on"] - r["r2_off"]), places=11)
        for r in summary["comparisons"]:
            if (r["exposure"], r["endpoint"]) == d.ANCHOR:
                self.assertEqual((r["difference_pp"], r["low_pp"], r["high_pp"]), (0., 0., 0.))
        for r in summary["diet_overlap"]:
            self.assertGreater(r["people"], 0)
            self.assertLessEqual(r["people"], min(v1[x, r["endpoint"]]["people"] for x in p.EXPOSURES))
            self.assertTrue(np.isfinite(r["fiber_pc5_delta_pp"]) and np.isfinite(r["upf_pc5_delta_pp"]))
        self.assertIn("FULL_PHENO_V3_SCREEN_END", render(summary))

    def test_comparisons_use_same_resample_for_both_workflows(self):
        # Test column order for all simultaneous contrasts on paired draws.
        gain = np.arange(48).reshape(3, 2, 8)/200.
        scored = contrasts(gain)
        self.assertEqual(scored.shape, (3, 2, 16))
        np.testing.assert_array_equal(scored[..., 8:15], gain[..., 1:] - gain[..., :1])
        np.testing.assert_array_equal(scored[..., -1], gain[..., 5] - gain[..., 1])

    def test_v2_registration_must_still_match_the_seal_after_report_read(self):
        with TemporaryDirectory() as tmp:
            archive = Path(tmp)
            registration = archive / "registration.json"
            registration.write_bytes(run.encode({"scope": "discovery only"}))
            receipt = {"registration_sha256": sha256(registration.read_bytes()).hexdigest()}
            (archive / "COMPLETE.json").write_bytes(run.encode(receipt))

            def replace_registration(path, release):
                registration.write_bytes(run.encode({"scope": "discovery only", "changed": True}))
                return {}

            with patch.object(run_v2, "load_completed", replace_registration):
                with self.assertRaisesRegex(run_v2.Stop, "V3_V2_REGISTRATION_HASH"):
                    v2_references(archive, {}, self.snapshot, {})
