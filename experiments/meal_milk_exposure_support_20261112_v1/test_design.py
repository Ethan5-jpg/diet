"""Exercise literal labels, conservative overlap, and exact taxonomic paths."""

import unittest

import numpy as np
import pandas as pd

from design import classify, genus_detection, overlapping_rows, reconstruct


class ExposureSupportTest(unittest.TestCase):
    def test_only_explicit_product_labels_indicate_lactose_status(self):
        self.assertEqual(classify("Milk", "3% milk"), "unspecified")
        self.assertEqual(classify("Low fat Milk", "Milk ללא לקטוז"), "free")
        self.assertEqual(classify("Milk", "Milk דל לקטוז"), "reduced")
        self.assertEqual(classify("Milk", "lactose-free and reduced-lactose"), "other")
        self.assertEqual(classify("Soymilk", "lactose-free"), "other")
        self.assertEqual(classify("Milk", "soy milk lactose-free"), "other")
        self.assertEqual(classify("Bread", ""), "no_listed_milk")

    def test_overlap_requires_separate_dates_and_common_dose_range(self):
        rows = pd.DataFrame([
            (person, context, day, carbs)
            for person, doses in [
                (0, (20, 30, 40, 25, 35, 45)),
                (1, (16, 20, 25, 45, 52, 58)),
                (2, (15, 16, 40, 25, 44, 45)),
            ]
            for context, day, carbs in zip(("free",) * 3 + ("unspecified",) * 3,
                                           (1, 2, 3, 1, 2, 3), doses)
        ], columns=["input_row", "context", "date", "carbs"])
        common = overlapping_rows(rows, "free", "unspecified")
        self.assertEqual(common.input_row.tolist(), [0, 0, 0, 0])
        self.assertEqual(common.carbs.tolist(), [30, 40, 25, 35])
        self.assertTrue(overlapping_rows(rows, "free", "reduced").empty)

    def test_mixed_milk_labels_are_not_a_single_context(self):
        names = ["input_row", "start", "date", "energy", "carbs", "fat", "protein", "fiber"]
        meals = pd.DataFrame([(0, 100.0, "day", 100, 20, 1, 2, 0)], columns=names)
        nutrients = ["kcal", "carb_g", "fat_g", "protein_g", "fiber_g"]
        food = pd.DataFrame([
            (0, 100.0, "day", "Milk", "Lactose-free", 50, 10, 0.5, 1, 0),
            (0, 120.0, "day", "Milk", "Milk", 50, 10, 0.5, 1, 0),
        ], columns=["input_row", "t", "date", "short_food_name", "product_name", *nutrients])
        result = reconstruct(meals, food, nutrients, 60, lambda rows: rows.t.notna())
        self.assertEqual(result.context.tolist(), ["ambiguous"])
        with self.assertRaisesRegex(ValueError, "nutrient totals"):
            reconstruct(meals.assign(carbs=21), food, nutrients, 60, lambda rows: rows.t.notna())

    def test_genus_detection_uses_exact_terminal_species_paths(self):
        raw = np.array([[0.2, 0.8], [0.0, 1.0]])
        taxa = ["g__Bifidobacterium|s__Example", "g__Other|s__Bifidobacterium_like"]
        mask, count = genus_detection(raw, taxa, expected_rows=2)
        self.assertEqual((mask.tolist(), count), ([True, False], 1))
