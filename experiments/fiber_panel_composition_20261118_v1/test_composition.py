"""Synthetic checks for conditional interaction fitting and family-level scoring."""
from pathlib import Path
import sys
import unittest
import numpy as np

import composition as c
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"fiber_surface_dates_20261118_v1"))
import surface


class CompositionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(84)
        n = 480
        self.train = np.arange(n) < 350
        self.held = ~self.train
        self.eco = rng.normal(size=(n, 4))
        self.pc = rng.normal(size=(n, 3))
        self.scores = rng.normal(size=(n, 2))
        self.marker = self.scores[:, 0] + rng.normal(size=n)
        d = rng.normal(size=n)
        dose = np.column_stack((d, np.maximum(0, d-.5)))
        x = np.column_stack((np.ones(n), dose))
        self.role = (3*d*self.marker, x, dose)

    def fit(self, source, target, marker):
        return c.correction(surface, source, target, self.eco, self.pc, self.scores,
                            marker, self.train, self.held)

    def test_planted_conditional_signal_and_redundancy(self):
        _, extra = self.fit(self.role, self.role, self.marker)
        dose = self.role[2][:, 0]
        target = (dose*(self.marker-self.scores[:, 0]))[self.held]
        self.assertGreater(np.corrcoef(extra, target)[0, 1], .75)
        for marker in (self.scores[:, 0], np.ones(len(dose))):
            _, correction = self.fit(self.role, self.role, marker)
            np.testing.assert_allclose(correction, 0, atol=1e-8)

    def test_no_held_outcome_or_auxiliary_features(self):
        reference = self.fit(self.role, self.role, self.marker)
        changed = [a.copy() for a in self.role]
        for a in changed:
            a[self.held] = 1e6
        target = (changed[0], *self.role[1:])
        np.testing.assert_allclose(self.fit(changed, target, self.marker), reference)
        # Even a strong additive marker association must not generate an interaction.
        add_only = (7*self.marker + 2*self.role[2][:, 0], *self.role[1:])
        main, extra = self.fit(add_only, add_only, self.marker)
        self.assertGreater(main.std(), 2)
        np.testing.assert_allclose(extra, 0, atol=1e-8)

    def test_mapping_and_frozen_predictions(self):
        p = np.array([[.1, .2, .4], [.6, .1, 0]])
        names = dict.fromkeys(["Eubacterium rectale", "Ruminococcus bromii", "Bifidobacterium adolescentis"])
        np.testing.assert_allclose(c.scores_without_bromii(p, names, .1), np.log1p([5., 6.]))
        pred = c.assemble(*[np.array([x]) for x in (1., 2., 3., 4., 8.)])
        np.testing.assert_allclose(pred[0], [1., 2., 4.5, 4., 6., 4., 5., 7.5])

    def test_paired_metric(self):
        size = len(c.NAMES)
        a = np.ones((5, 2+2*size))
        a[:, 1] = 10
        a[:, 2:2+size] = [8, 7, 6, 6.5, 6.3, 9, 8, 7]
        rows = c.summarize([a, a], np.ones((15, 5)))
        self.assertAlmostEqual(rows["shared_half"]["delta"], .1)
        self.assertAlmostEqual(rows["shared_half"]["relative_pct"], 100)
        self.assertAlmostEqual(rows["matched_shared_half"]["delta"], .1)
        self.assertEqual(rows["shared_half"]["wins"], 2)


if __name__ == "__main__":
    unittest.main()
