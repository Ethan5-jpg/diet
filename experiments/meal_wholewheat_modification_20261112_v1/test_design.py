"""Synthetic checks for taxonomic identity, fold isolation and diet matching."""

from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from design import comparable_pairs, covariates, genus_fraction, zero_aware


class DesignTest(unittest.TestCase):
    def test_genus_uses_only_exact_terminal_species(self):
        taxa = ["g__Lachnoclostridium|s__A", "g__Lachnoclostridium|s__B", "g__Lachnoclostridium_A|s__C"]
        raw = np.array([[.1, .2, .7], [.0, .0, 1.]])
        values, leaves = genus_fraction(raw, taxa)
        self.assertEqual(leaves, 2)
        np.testing.assert_allclose(values, [.3, 0.])
        with self.assertRaisesRegex(ValueError, "overlapping aggregate"):
            genus_fraction(raw, [taxa[0], "g__Lachnoclostridium", taxa[2]])

    def test_training_transform_excludes_held_out_abundance(self):
        training = np.array([True, True, True, True, False])
        first, difference = zero_aware(np.array([0., .01, .04, 0., .0001]), training)
        second, _ = zero_aware(np.array([0., .01, .04, 0., .70]), training)
        np.testing.assert_allclose(first[training], second[training])
        np.testing.assert_allclose(first[[0, 3], 1], 0.)
        self.assertEqual(difference, 2.)

    def test_pair_excludes_same_day_and_wraps_clock(self):
        meals = pd.DataFrame({
            "input_row": [0] * 4, "date": ["a", "a", "b", "c"], "start": [0, 1, 2, 3],
            "bread_wholewheat": [35., 0., 0., 0.], "bread": [0., 35., 35., 35.],
            "carbs": [40., 40., 49., 44.], "energy": [300.] * 4,
            "protein": [12.] * 4, "local_hour": [23., 23., 1., 23.],
        })
        self.assertEqual(comparable_pairs(meals).index.tolist(), [0, 3])

    def test_exchange_is_one_unit_per_ten_grams_at_fixed_total(self):
        def center(values, unit):
            frame = pd.DataFrame(values)
            return (frame - frame.groupby(unit).transform("mean")).to_numpy(), None

        meals = pd.DataFrame({
            "bread_wholewheat": [0., 10., 0., 10.], "bread": [20., 10., 20., 10.],
            "carbs": [40.] * 4, "age_calendar_years": [50.] * 4, "sex_code": [0.] * 4,
            "fruits": [10.] * 4, "unlabelled": [0.] * 4,
        })
        people = np.array([0, 0, 1, 1])
        score = np.array([[0., 0.], [1., 0.]])
        _, exchange = covariates(meals, score, people, people, SimpleNamespace(center=center))
        np.testing.assert_allclose(exchange[:, 0], [0., 0., -.5, .5])


if __name__ == "__main__":
    unittest.main()
