from types import SimpleNamespace
import unittest

import numpy as np

from run_precision import day_moments, features


class PrecisionTest(unittest.TestCase):
    def test_day_moments(self):
        got, count = day_moments(np.array([0,0,0,1,1]), np.array([1.,2.,3.,5.,7.]), 3)
        np.testing.assert_array_equal(count, [3,2,0])
        np.testing.assert_allclose(got, [[2.,1.],[6.,2.],[0.,0.]])

    def test_matching_scale_and_training_isolation(self):
        role = np.arange(10)<8
        bg = SimpleNamespace(role=role, supported=np.ones(10,bool))
        raw = np.linspace(-1.,2.,10)
        noise = np.linspace(.02,.35,10)
        count = np.arange(3,13)
        first, diag = features("fiber_density",bg,raw,((noise,count),None))
        next_raw, next_noise = raw.copy(), noise.copy()
        next_raw[~role] += 1000.
        next_noise[~role] *= 10000.
        second, test_diag = features("fiber_density",bg,next_raw,((next_noise,count),None))
        self.assertEqual(diag,test_diag)
        for name, values in first.items():
            np.testing.assert_allclose(values[role],second[name][role])
            self.assertAlmostEqual(values[role].std(),raw[role].std())
