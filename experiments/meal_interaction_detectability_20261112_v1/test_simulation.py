"""Synthetic checks for multi-outcome fitting and amplitude accounting."""

import unittest

import numpy as np

from simulation_design import AMPLITUDES, METHODS, aggregate, household_moments, prediction_errors, recoverability


class SimulationDesignTest(unittest.TestCase):
    def test_signal_response_is_linear_in_amplitude(self):
        x = np.arange(1., 9.)[:, None]
        z = np.tile([-1., 1.], 4)[:, None]
        y = np.column_stack([x[:, 0] + z[:, 0], 2 * z[:, 0]])
        w = np.ones(8)
        combined = prediction_errors(x, x, z, z, z[:, 0], z[:, 0], y, y, w)
        direct_y = y[:, [0]] + .7 * y[:, [1]]
        direct = prediction_errors(x, x, z, z, z[:, 0], z[:, 0], direct_y, direct_y, w)
        for key, value in combined.items():
            np.testing.assert_allclose(value[:, 0] + .7 * value[:, 1], direct[key][:, 0])
        np.testing.assert_allclose(combined["pc_ridge"], combined["known_direction"])

    def test_household_polynomial_matches_explicit_squared_errors(self):
        y = np.array([[3., 0., 1.], [-1., 2., 2.], [2., 4., -1.]])
        family, w = np.array([1, 0, 1]), np.array([.5, 1., .5])
        m = household_moments({"target": y}, family, w, 2)["target"]
        a = .25
        expected = np.vstack([np.sum(w[family == i, None] *
                                      (y[family == i, :-1] + a * y[family == i, -1:]) ** 2, axis=0)
                              for i in range(2)])
        np.testing.assert_allclose(m[0] + a * m[1] + a ** 2 * m[2], expected)

    def test_weighted_ridge_matches_the_joint_normal_equations(self):
        x = np.column_stack([np.ones(6), np.arange(6.)])
        z = np.array([0., 2., 1., 3., 2., 0.])[:, None]
        w = np.arange(1., 7.)
        y = np.column_stack([np.arange(6.) ** 2, np.arange(6.)[::-1] + z[:, 0]])
        joined = np.column_stack([x, z])
        train, valid = np.arange(5), np.array([5])
        weight = w[train]
        j = joined[train]
        expected = np.linalg.solve(j.T @ (weight[:, None] * j) + np.diag([0., 0., .03 * weight.sum()]),
                                   j.T @ (weight[:, None] * y[train]))
        errors = prediction_errors(x[train], x[valid], z[train], z[valid], z[train, 0], z[valid, 0],
                                   y[train], y[valid], weight)
        np.testing.assert_allclose(errors["pc_ridge"], y[valid] - joined[valid] @ expected)

    def test_paired_detection_requires_positive_interval(self):
        null = np.zeros((3, 2, 4))
        null[0] = 1.
        alt = null.copy()
        alt[2] = -.1
        samples = recoverability({"baseline": null, "target": null, **dict.fromkeys(METHODS, alt)}, 8, 39)
        out = aggregate(samples)["pc_ridge"]
        self.assertEqual(len(out["grid"]), len(AMPLITUDES))
        self.assertEqual(out["grid"][0]["positive_interval_rate"], 0.)
        self.assertEqual(out["first_persistent_80_percent"], .25)
