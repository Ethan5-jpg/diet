"""Exercise paired loss accounting for a frozen template direction."""

import unittest

import numpy as np

from transfer_design import household_losses, predict, score_summary


class TransferDesignTest(unittest.TestCase):
    def test_shared_background_and_projection(self):
        baseline, template, reference = predict(
            np.array([[2.], [-2.]]), np.array([[3.], [-3.]]),
            np.array([.5]), np.array([[1.]]), np.array([2.]), np.array([1.]),
        )
        np.testing.assert_allclose(baseline, [1., -1.])
        np.testing.assert_allclose(template, [2., -2.])
        np.testing.assert_allclose(reference, [1., -1.])

    def test_both_comparisons_use_the_same_households_and_denominator(self):
        y = np.array([2., 4.])
        parts = household_losses(y, np.array([.5, 1.]), np.array([0, 1]),
                                 np.zeros(2), y, y / 2, 2)
        result = score_summary(parts, 0)
        self.assertEqual(result["template_r2"], 1.)
        self.assertEqual(result["reference_r2"], .75)
        self.assertEqual(result["template_vs_reference"]["delta_r2"], .25)
        self.assertEqual(result["template_vs_reference"]["interval"], [.25, .25])
        self.assertEqual(result["leave_one_template_vs_baseline"], [1., 1.])
        np.testing.assert_allclose(parts[3], [2., 16.])
