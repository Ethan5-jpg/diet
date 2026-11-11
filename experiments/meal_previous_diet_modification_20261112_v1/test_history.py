"""Synthetic checks for previous episode timing, quality and held-out isolation."""

from importlib.util import module_from_spec, spec_from_file_location
import unittest

import numpy as np
import pandas as pd

from analyze_previous import ENDPOINTS, ROOT, fit_blocks
from history import design, preceding, repeated

MODEL_PATH = ROOT.parent / "diet_daily_modification_20261112_v1/models.py"
SPEC = spec_from_file_location("original_meal_model", MODEL_PATH)
MODEL = module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL)


class PreviousMealTest(unittest.TestCase):
    def test_same_episode_boundary_and_exact_gap_with_incomplete_macro(self):
        meal = pd.DataFrame({"input_row": [1], "start": [10000.], "prior_gap": [135.]})
        food = pd.DataFrame({
            "t": [100., 1900., 10000.], "date": ["a"] * 3, "clock_agrees": True, "qc_primary_eligible": True,
            "qc_same_food_time_n": 1, "qc_exact_signature_n": 1, "e": [80., 100., 40.], "c": [10., 20., 20.],
            "f": 0., "p": [0., np.nan, 2.], "b": 0.,
        })
        out = preceding(meal, food, list("ecfpb"), 1800)
        self.assertEqual(out.prior_energy.iloc[0], 180.)
        self.assertEqual(out.prior_carbs.iloc[0], 30.)
        self.assertTrue(out.pair_verified.iloc[0])
        self.assertFalse(out.complete_previous.iloc[0])
        with self.assertRaisesRegex(ValueError, "preceding intervals"):
            preceding(meal.assign(prior_gap=145.), food, list("ecfpb"), 1800)

    def test_repeated_dates_are_required_for_both_previous_doses(self):
        data = pd.DataFrame({"input_row": [1] * 4 + [2] * 4, "prior_carbs": [20., 20., 40., 40.] * 2,
                             "date": ["a", "b", "a", "b", "a", "a", "a", "b"]})
        self.assertEqual(repeated(data).tolist(), [True] * 4 + [False] * 4)

    def test_interaction_and_training_only_direction(self):
        meals = pd.DataFrame({"input_row": [0] * 4, "person_index": [0] * 4, "carbs": [30., 50.] * 2,
                              "prior_carbs": [20., 20., 40., 40.], "prior_energy": 300.,
                              "prior_protein": 10., "prior_fiber": 3., "prior_gap": 180., "local_hour": 10.})
        _, _, term = design(meals, np.array([[2.]]), MODEL)
        np.testing.assert_allclose(term[:, 0], [2., -2., -2., 2.])
        rng = np.random.default_rng(82)
        data = pd.concat([meals.assign(input_row=i, person_index=i) for i in range(40)], ignore_index=True)
        data["prior_energy"] = rng.uniform(150., 600., len(data))
        data["prior_protein"] = rng.uniform(2., 25., len(data))
        data["prior_fiber"] = rng.uniform(.5, 8., len(data))
        data["prior_gap"] = rng.uniform(135., 360., len(data))
        data["local_hour"] = rng.uniform(9., 21., len(data))
        scores = rng.normal(size=(40, 3))
        x, y = rng.normal(size=(160, 4)), rng.normal(size=160)
        train = data.input_row.to_numpy() < 30
        w = np.full(160, .25)
        fit = fit_blocks(data, {(ep, "carbs"): (y, x, w) for ep in ENDPOINTS}, scores, train, MODEL)
        changed = y + 100 * ~train
        repeat = fit_blocks(data, {(ep, "carbs"): (changed, x, w) for ep in ENDPOINTS}, scores, train, MODEL)
        for ep in ENDPOINTS:
            np.testing.assert_array_equal(fit[ep]["direction_vector"], repeat[ep]["direction_vector"])
            np.testing.assert_array_equal(fit[ep]["joint"][2], repeat[ep]["joint"][2])


if __name__ == "__main__":
    unittest.main()
