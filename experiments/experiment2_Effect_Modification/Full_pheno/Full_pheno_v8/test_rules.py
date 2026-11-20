"""Prevent an apparent interaction increase from a degraded off arm alone."""

from types import SimpleNamespace
import unittest

import numpy as np

from stability_rules import FIXED, choose


class RuleTest(unittest.TestCase):
    def test_both_predictions_improve(self):
        y = np.array([-2.,-1.,1.,2.])
        b = np.tile(.2*y[:,None],(1,len(FIXED)))
        c = np.tile(.35*y[:,None],(1,len(FIXED)))
        j = FIXED.index("precision_linear_main")
        b[:,j],c[:,j] = .4*y,.85*y
        f = SimpleNamespace(outcome=y,no_diet=np.zeros_like(y),off=b,on=c)
        self.assertEqual(choose([f]*4,strict=True),(j,.5))

    def test_off_degradation_without_full_improvement_is_rejected(self):
        y = np.array([-2.,-1.,1.,2.])
        b = np.tile(.2*y[:,None],(1,len(FIXED)))
        c = np.tile(.35*y[:,None],(1,len(FIXED)))
        b[:,FIXED.index("blended_main")] = -.4*y
        f = SimpleNamespace(outcome=y,no_diet=np.zeros_like(y),off=b,on=c)
        self.assertEqual(choose([f]*4,strict=False),(0,0.))
        self.assertEqual(choose([f]*4,strict=True),(0,0.))
