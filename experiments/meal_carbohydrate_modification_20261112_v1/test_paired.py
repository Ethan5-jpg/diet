import unittest

import numpy as np

from analyze import bases
from paired import family_totals, summarize
from test_analyze import models, sample


class PairTests(unittest.TestCase):
    def test_replay_matches_total_errors(self):
        meals, score = sample()
        matrices, doses = bases(meals, models)
        y, x, w = matrices["net_p0", "carb_share"]
        family = meals.split_unit_id.to_numpy()
        people = meals.person_index.to_numpy()
        train = family % 5 != 0
        z, _ = models.center(doses["carb_share"][:, None] * score[people], people)
        fitted = models.fit_background(x[train], y[train], w[train])
        _, null, alternative = models.fit_interaction(x, z, y, w, train, ~train, fitted, .03)
        replay = family_totals(y, x, w, {"both_pc5": z}, train, family, len(set(family)), models)
        np.testing.assert_allclose([replay["none"].sum(), replay["both_pc5"].sum()], [null, alternative])
        self.assertTrue((replay["none"][np.unique(family[train])] == 0).all())

    def test_pair_uses_same_draws_and_denominator(self):
        v = dict(tss=np.array([4., 8., 12.]), none=np.array([4., 7., 10.]),
                 oral_pc5=np.array([3., 6., 9.]), gut_pc5=np.array([3., 7., 9.]),
                 both_pc5=np.array([2., 5., 8.]))
        draws = np.array([[0, 0, 2], [1, 1, 1], [0, 2, 2]])
        result = summarize(v, draws)
        self.assertAlmostEqual(result["both_vs_none"]["delta_r2"], 6 / 24)
        self.assertAlmostEqual(result["both_vs_oral"]["delta_r2"], 3 / 24)
        self.assertAlmostEqual(result["both_vs_gut"]["delta_r2"], 4 / 24)


if __name__ == "__main__":
    unittest.main()
