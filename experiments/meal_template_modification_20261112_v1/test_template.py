"""Synthetic checks for source linkage, date splits, and dose identification."""

import unittest

import numpy as np
import pandas as pd

from template_design import paired_summary, reconstruct, scopes, summary


class TemplateDesignTest(unittest.TestCase):
    def test_food_ids_are_a_multiset_and_missing_ids_do_not_form_templates(self):
        meals = pd.DataFrame([
            (0, 0., "a", 60, 15, 0, 0, 0),
            (0, 100., "b", 60, 15, 0, 0, 0),
            (0, 200., "c", 60, 15, 0, 0, 0),
            (0, 300., "d", 60, 15, 0, 0, 0),
        ], columns=["input_row", "start", "date", "energy", "carbs", "fat", "protein", "fiber"])
        nutrients = ["k", "c", "f", "p", "b"]
        food = pd.DataFrame([
            (0, float(start + offset), day, code, 20, 5, 0, 0, 0)
            for start, day, ids in [(0, "a", ["x", "x", "y"]), (100, "b", ["y", "x", "x"]),
                                     (200, "c", ["x", "y", "y"]), (300, "d", ["x", "", "y"])]
            for offset, code in enumerate(ids)
        ], columns=["input_row", "t", "date", "food_id", *nutrients])
        complete = reconstruct(meals, food, nutrients, 5, lambda rows: rows.t.notna())
        self.assertEqual(complete.template.tolist(), [0, 0, 1])
        pooled, calendar = scopes(complete)
        self.assertEqual(pooled.start.tolist(), [0, 100])
        self.assertTrue(calendar.empty)
        with self.assertRaisesRegex(ValueError, "nutrient totals"):
            reconstruct(meals.assign(carbs=16), food, nutrients, 5, lambda rows: rows.t.notna())

    def test_calendar_requires_the_same_template_in_both_halves(self):
        complete = pd.DataFrame([
            (0, 0, "a", False), (0, 0, "b", False), (0, 0, "c", True), (0, 0, "d", True),
            (0, 1, "a", False), (0, 1, "b", False), (0, 2, "c", True), (0, 2, "d", True),
        ], columns=["input_row", "template", "date", "late"])
        pooled, calendar = scopes(complete)
        self.assertEqual(len(pooled), 8)
        self.assertEqual(calendar.group.tolist(), [0, 0, 1, 1])

    def test_dose_collinear_with_energy_is_not_independent_information(self):
        rows = pd.DataFrame([
            (0, "a", 0, "a", 100., 15., 4., 7.), (0, "a", 0, "b", 200., 30., 4., 7.),
            (1, "b", 1, "a", 100., 15., 4., 7.), (1, "b", 1, "b", 200., 30., 4., 7.),
        ], columns=["input_row", "split_unit_id", "template", "date", "energy", "carbs", "protein", "local_hour"])
        rows["group"] = rows.template
        metrics = summary(rows)
        self.assertGreater(metrics["carb_sd_g"], 7.)
        self.assertLess(metrics["adjusted_carb_sd_g"], 1e-8)
        self.assertEqual(metrics["groups_with_matched_pair"], 0)
        self.assertGreater(summary(rows.assign(energy=100.))["adjusted_carb_sd_g"], 7.)

    def test_paired_bootstrap_uses_the_same_household_losses(self):
        values = np.array([[3., 4.], [2., 2.], [10., 20.]])
        result = paired_summary(values, seed=1)
        self.assertEqual(result["delta_r2"], .1)
        np.testing.assert_allclose(result["interval"], [.1, .1])
