"""Synthetic checks for the conditional selection criteria."""

from types import SimpleNamespace
import unittest

import numpy as np

from interaction_rules import MENU, METHODS, RULES, Upgrade, choose_upgrade, domain_prior


def simulated(off, on):
    y = np.array([-2., -1., 1., 2.])
    before = np.zeros((len(y), len(RULES)))
    after = np.tile(.15 * y[:, None], (1, len(RULES)))
    j = METHODS.index(MENU[0])
    before[:, j], after[:, j] = off*y, on*y
    return SimpleNamespace(outcome=y, off=before, on=after)


class UpgradeTest(unittest.TestCase):
    def test_joint_improvement_passes(self):
        fit = simulated(.1, .8)
        self.assertEqual(choose_upgrade([fit]*4), Upgrade(METHODS.index(MENU[0]), .5))

    def test_weakening_off_does_not_unlock_upgrade(self):
        fit = simulated(-.2, .15)
        self.assertEqual(choose_upgrade([fit]*4), Upgrade())

    def test_one_bad_fold_and_two_good_folds_default_to_reference(self):
        better, worse = simulated(.1, .8), simulated(0., -.1)
        self.assertEqual(choose_upgrade([better, better, worse, worse]), Upgrade())

    def test_domain_uses_average_normal_equations(self):
        one = SimpleNamespace(gram=np.diag([1., 3.]), rhs=np.array([2., 4.]))
        two = SimpleNamespace(gram=np.diag([3., 1.]), rhs=np.array([4., 2.]))
        np.testing.assert_allclose(domain_prior([one, two]), [1., 1.])
        self.assertIsNone(domain_prior([]))
