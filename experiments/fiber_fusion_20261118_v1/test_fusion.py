import sys
import unittest
from pathlib import Path
import numpy as np
import fusion
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"fiber_surface_dates_20261118_v1"))
import surface


class FusionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(28)
        n = 650
        self.t = np.arange(n) < 470
        self.h = ~self.t
        d = rng.normal(size=n)
        self.dose = np.column_stack((d, np.maximum(d-.5, 0)))
        self.x = np.column_stack((np.ones(n), self.dose))
        self.b = rng.normal(size=(n, 2))
        self.args = (rng.normal(size=(n, 4)), rng.normal(size=(n, 10)), rng.normal(size=(n, 2)), self.b,
                     rng.normal(size=n), rng.normal(size=(n, 2)), self.t, self.h)

    def fit(self, y, reverse=False):
        return fusion.additions(surface, (y, self.x, self.dose), *self.args, reverse)

    def test_planted_balance_and_no_held_outcomes(self):
        y = self.dose[:, 0]*self.b[:, 0]
        main, delta = self.fit(y)
        self.assertGreater(np.corrcoef(delta[:, 0], y[self.h])[0, 1], .80)
        y[self.h] += 1e5
        main2, delta2 = self.fit(y)
        np.testing.assert_array_equal(delta, delta2)
        np.testing.assert_array_equal(main, main2)
        np.testing.assert_allclose(delta[:, -1], (delta[:, 2]+delta[:, 3]+delta[:, 5])/3)

    def test_matched_main_only(self):
        y = self.b[:, 0]+2*self.x[:, 1]
        for rev in (True, False):
            main, delta = self.fit(y, rev)
            self.assertGreater(main.std(), .5)
            np.testing.assert_allclose(delta, 0, atol=1e-8)
            zero = np.zeros_like(main)
            pred = fusion.assemble(zero, zero, main, delta)
            idx = fusion.NAMES.index("matched_off")
            np.testing.assert_allclose(pred[:, idx:]-pred[:, :idx], np.broadcast_to(main[:, None], pred[:, :idx].shape))

    def test_v8_species_contrasts_and_zeros(self):
        names = dict.fromkeys(["Eubacterium rectale", "Bifidobacterium adolescentis", "Ruminococcus bromii"])
        m = np.array([[3., 2., 4.], [0., 0., 0.]])
        v = fusion.panel_views(m, names, 1.)
        np.testing.assert_allclose(v[0], [(np.log(5)-np.log(3))/np.sqrt(2), np.log(15/16)/np.sqrt(6)])
        np.testing.assert_array_equal(v[1], [0., 0.])

    def test_percentages_computed_within_repeat(self):
        n = len(fusion.NAMES)
        a = np.ones((6, 2+2*n))
        a[:, 1] = 10
        a[:, 2:2+n] = 6.
        a[:, 2] = 8.
        b = a.copy()
        b[:, 3] = 5.
        summary = fusion.summarize([a, b], np.ones((16, 6)))
        self.assertAlmostEqual(summary["anchor"]["gain_pct"], 125)
        self.assertAlmostEqual(summary["anchor"]["fraction_pct"], 55)
        self.assertAlmostEqual(summary["anchor"]["worst_fraction"], 50)
        self.assertEqual(fusion.CONTRASTS, 32)


if __name__ == "__main__":
    unittest.main()
