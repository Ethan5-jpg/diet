"""Test food-date coverage, a frozen spline and nested gate fallback."""

import unittest

import run_exposure

import numpy as np
import pandas as pd

from diet_audit import Fit
from exposure_models import apply_plans, choose_diet, guard, plans
from food_features import aggregate_average
from upf_refit import spline


class ExposureTest(unittest.TestCase):
    def test_average_requires_coverage_and_repeated_days(self):
        d = pd.DataFrame(dict(input_row=[0]*4+[1]*3, date=pd.date_range("2020-01-01", periods=7),
                              quantity=[1., 2., 3., 4., 5., 6., 7.]))
        values, reliability = aggregate_average(d, ["quantity"], np.array([5, 5, 0]), 3, reliable=True)
        np.testing.assert_allclose(values.ravel(), [2.5, np.nan, np.nan], equal_nan=True)
        self.assertEqual(reliability["quantity"]["people_ge4"], 1)

    def test_nested_fallback_and_no_unilateral_fold_selection(self):
        y = np.array([-1., 1.])
        off = np.repeat((.5*y)[:, None], 6, axis=1)
        on = np.repeat((.7*y)[:, None], 6, axis=1)
        fit = Fit(np.array([0, 1]), y, np.zeros(2), off, on)
        self.assertEqual(choose_diet([fit]*4, 8.), 0)
        self.assertEqual(guard(np.array([1., 0., 0., 0.]), np.ones(4), .4), 0.)
        plan = plans({"D_TotalFat": [fit]*4}, "UPF")["D_TotalFat"]
        selected = apply_plans(fit, plan, ((0, 0.), (-1, 0.), (-1, 0.)), False)
        np.testing.assert_allclose(selected.on[:, 0], on[:, 0])
        np.testing.assert_allclose(selected.off[:, -1], 0.)
        np.testing.assert_allclose(selected.on[:, -1], 0.)

    def test_registered_spline(self):
        np.testing.assert_allclose(spline(np.array([-2., -1., 0., 1., 2., 3.]), (-1., 0., 1.)),
                                   [0., 0., .25, 1.5, 3., 4.5])


if __name__ == "__main__":
    unittest.main()
