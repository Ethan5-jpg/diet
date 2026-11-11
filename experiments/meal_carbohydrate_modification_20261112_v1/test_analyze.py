import json
import unittest

import numpy as np
import pandas as pd

from analyze import ENDPOINTS, PRIMARY, bases, compare, panel, pool
from prepare import ROOT, base_modules


CONFIG = json.loads((ROOT / "config.json").read_bytes())
base_modules(CONFIG)
import models


def sample():
    rng = np.random.default_rng(123)
    n = 180
    person = np.repeat(np.arange(n), 7)
    rows = len(person)
    m = rng.normal(0, 1, n)
    share = rng.uniform(25, 75, rows)
    net = 20 + 2 * (share - 50) / 10 * m[person] + rng.normal(0, 1, rows)
    df = pd.DataFrame(dict(person_index=person, split_unit_id=person // 2,
                           carb_share=share, carbs=rng.uniform(15, 90, rows),
                           age_calendar_years=rng.uniform(30, 70, n)[person],
                           sex_code=rng.integers(0, 2, n)[person], local_hour=rng.uniform(7, 23, rows),
                           prior_gap=rng.uniform(140, 400, rows), energy=rng.uniform(200, 950, rows),
                           protein=rng.uniform(5, 45, rows), protein_share=rng.uniform(7, 25, rows),
                           fiber=rng.uniform(1, 15, rows), weekday=rng.integers(0, 7, rows),
                           net_p0=net, net_m15=net + rng.normal(0, 1, rows), net_p15=net,
                           peak_p0=net * 2, pre_p0=100 + rng.normal(0, 1, rows),
                           pre_m15=101 + rng.normal(0, 1, rows), pre_p15=103 + rng.normal(0, 1, rows),
                           trend_p0=rng.normal(0, 1, rows), trend_m15=rng.normal(0, 1, rows),
                           trend_p15=rng.normal(0, 1, rows)))
    return df, m[:, None]


class AnalysisTests(unittest.TestCase):
    def test_signed_injected_response_and_pre_window(self):
        meals, score = sample()
        matrices, dose = bases(meals, models)
        scores = {"external_sum7": score}
        result = compare(meals, matrices, dose, scores, np.ones(len(score), bool), models, direct=True)
        self.assertEqual(len(result), 2 * len(ENDPOINTS))
        self.assertLess(result[PRIMARY]["holm_p"], .001)
        self.assertGreater(result[PRIMARY]["ci95"][0], 1.5)
        self.assertGreater(result["pre_p0_carb_share_external_sum7"]["p"], .05)
        train = np.arange(len(score)) // 2 % 5 != 0
        outcome = compare(meals, matrices, dose, scores, train, models)
        self.assertGreater(pool([dict(repeat=0, results=outcome)])[PRIMARY]["r0"]["delta_r2"], .65)

    def test_literal_panel_mapping(self):
        keys = {str(j): f"s__name{j}" for j in range(7)}
        keys["unsupported"] = None
        labels = [*list(keys.values())[:7], "s__unrelated"]
        np.testing.assert_allclose(panel(np.full((2, 8), 1 / 8), labels, keys), [[7 / 8], [7 / 8]])
        labels[0] = "s__not_the_requested_taxon"
        with self.assertRaisesRegex(ValueError, "aligns"):
            panel(np.full((2, 8), 1 / 8), labels, keys)


if __name__ == "__main__":
    unittest.main()
