"""Synthetic checks for source labels, prior logging history and dose interactions."""

from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from coffee_design import SPECIES, interaction_design, reconstruct, taxon_fraction


class CoffeeTest(unittest.TestCase):
    def test_exact_names_energy_and_prior_date_history(self):
        m = pd.DataFrame({
            "input_row": [1, 1, 1], "start": [100., 2000., 4000.], "date": ["a", "b", "c"],
            "energy": [225., 200., 205.], "carbs": [43., 40., 40.], "fat": 0.,
            "protein": 0., "fiber": 0.,
        })
        f = pd.DataFrame({
            "input_row": [1] * 7, "t": [100., 101., 102., 2000., 2001., 4000., 4001.],
            "date": ["a", "a", "a", "b", "b", "c", "c"],
            "short_food_name": ["Bread", "Coffee", " COFFEE ", "Bread", "Coffee substitute", "Bread", "Coffee"],
            "e": [200., 15., 10., 200., 0., 200., 5.], "c": [40., 1., 2., 40., 0., 40., 0.],
            "f": 0., "p": 0., "b": 0., "clock_agrees": True, "qc_primary_eligible": True,
            "qc_same_food_time_n": 1, "qc_exact_signature_n": 1,
        })
        meal = reconstruct(m, f, list("ecfpb"), 1800)
        self.assertEqual(meal.coffee.tolist(), [True, False, True])
        self.assertEqual(meal.pure.tolist(), [False, False, True])
        self.assertEqual(meal.questionable.tolist(), [False, True, False])
        self.assertEqual(meal.coffee_kcal.tolist(), [25., 0., 5.])
        self.assertEqual(meal.coffee_carbs.tolist(), [3., 0., 0.])
        np.testing.assert_allclose(meal.prior_coffee_share, [.5, 2 / 3, .5])
        self.assertEqual(meal.prior_log_days.tolist(), [0, 1, 2])

    def test_taxon_identity_is_exact_and_unique(self):
        abundance = np.array([[.1, .9], [0., 1.]])
        np.testing.assert_array_equal(taxon_fraction(abundance, [f"k__Bacteria|{SPECIES}", f"{SPECIES}_A"]), [.1, 0.])
        with self.assertRaisesRegex(ValueError, "unique exact"):
            taxon_fraction(abundance, [SPECIES, SPECIES])

    def test_three_way_term_uses_total_carbohydrate_and_context(self):
        def center(values, unit):
            frame = pd.DataFrame(values)
            return (frame - frame.groupby(unit).transform("mean")).to_numpy(), None

        m = pd.DataFrame({
            "person_index": 0, "carbs": [30., 50., 30., 50.], "context": [False, False, True, True],
            "local_hour": 9., "prior_gap": 300., "prior_coffee_share": .5, "prior_log_days": 4,
            "energy": 300., "protein": 10., "age_calendar_years": 50., "sex_code": 0,
            "coffee_kcal": [0., 0., 5., 5.], "coffee_carbs": 0.,
        })
        _, term = interaction_design(m, np.array([[2., 1.]]), np.zeros(4, dtype=int), SimpleNamespace(center=center))
        np.testing.assert_allclose(term[:, 0], [0., 0., -2., 2.])
        np.testing.assert_allclose(term[:, 1], [0., 0., -1., 1.])


if __name__ == "__main__":
    unittest.main()
