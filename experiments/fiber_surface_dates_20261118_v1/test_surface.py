import unittest

import numpy as np

import surface as s


class SurfaceTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(110218)
        self.family = np.repeat(np.arange(150), 2)
        n = len(self.family)
        f, m, h, noise = rng.normal(size=(4, n))
        self.train = self.family < 100
        dose = np.column_stack((f, np.maximum(f-.1, 0)))
        self.role = (f + 3*f*m + 4*dose[:, 1]*np.maximum(m, 0) + .2*noise,
                     np.column_stack((np.ones(n), dose)), dose)
        self.args = (rng.normal(size=(n, 4)), rng.normal(size=(n, 5)), np.column_stack((m, h)))

    def test_signal_and_no_held_outcome_leakage(self):
        y, *x = s.design(self.role, *self.args, self.train)
        held = ~self.train
        preds = s.candidates(y, *x, self.train, held)
        error = lambda key: np.mean((y[held] - preds[key])**2)
        self.assertLess(error(("panel", s.FIXED)), .1*error(("baseline", None)))
        other = s.candidates(y+1000*held, *x, self.train, held)
        for key in preds:
            np.testing.assert_array_equal(preds[key], other[key])

    def test_folds(self):
        seen = np.zeros(len(self.train), int)
        for train, valid in s.inner_roles(self.family, self.train, 123):
            self.assertFalse(set(self.family[train]) & set(self.family[valid]))
            np.testing.assert_array_equal(train | valid, self.train)
            seen += valid
        np.testing.assert_array_equal(seen, self.train)

    def test_training_only_basis_and_shrinkage(self):
        marker = self.args[-1][:, 0]
        a = s.marker_basis(marker, self.train)
        b = s.marker_basis(marker + 1000*~self.train, self.train)
        np.testing.assert_array_equal(a[self.train], b[self.train])
        dose = s.dose_basis(self.role[-1], self.train)[self.train]
        self.assertAlmostEqual(float(dose[:, 0] @ dose[:, 1]), 0)
        losses = {("panel", pair): 1. for pair in s.GRID}
        losses["baseline", None] = 1.
        self.assertEqual(s.choose(losses, "panel"), ("baseline", None))
        losses["panel", (.1, 3.)] = .5
        self.assertEqual(s.choose(losses, "panel"), ("panel", (.1, 3.)))


if __name__ == "__main__":
    unittest.main()
