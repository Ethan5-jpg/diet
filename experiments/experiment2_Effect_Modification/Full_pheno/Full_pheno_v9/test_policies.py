"""Exercise diet and interaction fallback without changing evaluation rows."""

from dataclasses import dataclass
import unittest

import numpy as np

from feature_policies import FIXED, adjusted, extend, pick


@dataclass(frozen=True)
class FixtureFit:
    rows: np.ndarray
    outcome: np.ndarray
    no_diet: np.ndarray
    off: np.ndarray
    on: np.ndarray


class PolicyTest(unittest.TestCase):
    def test_stop_diet_and_separate_interaction_stop(self):
        y = np.array([-2.,-1.,1.,2.])
        z = np.zeros_like(y)
        off = np.tile(-.20*y[:,None],(1,len(FIXED)))
        worse = FixtureFit(np.arange(4),y,z,off,1.5*off)
        self.assertEqual(pick([worse]*4),(0,"diet"))
        report = extend(worse,0,"diet","interaction")
        np.testing.assert_array_equal(report.rows,worse.rows)
        np.testing.assert_array_equal(report.off[:,-2],z)
        np.testing.assert_array_equal(report.on[:,-2],z)
        np.testing.assert_array_equal(report.on[:,-1],worse.off[:,0])

    def test_negative_interaction_keeps_supported_main(self):
        y = np.array([-2.,-1.,1.,2.])
        off = np.tile(.5*y[:,None],(1,len(FIXED)))
        fit = FixtureFit(np.arange(4),y,np.zeros_like(y),off,.5*off)
        self.assertEqual(pick([fit]*4),(0,"interaction"))
        b,c = adjusted(fit.no_diet,off[:,0],fit.on[:,0],"interaction",.5)
        np.testing.assert_allclose(b,.5*y)
        np.testing.assert_allclose(c,.375*y)
