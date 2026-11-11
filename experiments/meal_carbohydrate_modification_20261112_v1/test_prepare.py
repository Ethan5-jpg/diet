import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from prepare import NUTRIENTS, ROOT, align, episodes, food_groups, helper


CFG = json.loads((ROOT / "config.json").read_bytes())


def events(minutes):
    at = pd.Timestamp("2026-01-15 07:00", tz="Asia/Jerusalem") + pd.to_timedelta(minutes, unit="m")
    x = pd.DataFrame(dict(collection_timestamp=at, local_timestamp=at.tz_localize(None), input_row=7,
                          qc_primary_eligible=True, qc_exact_signature_n=1, qc_same_food_time_n=1))
    x[NUTRIENTS] = np.tile([320, 30, 10, 20, 3], (len(x), 1))
    return x


class PreparationTests(unittest.TestCase):
    def test_isolated_episode_and_nutrient_share(self):
        g = food_groups(events([0, 150, 160, 350]))[7]
        meals = episodes(g, {"2026-01-15"}, CFG)
        self.assertEqual(len(meals), 1)
        self.assertEqual(meals[0]["carbs"], 60)
        self.assertAlmostEqual(meals[0]["carb_share"], 100 * 120 / 290)
        self.assertEqual(meals[0]["energy"], 640)

    def test_clock_disagreement_and_overlapping_meals(self):
        source = events([0, 150, 160, 350])
        source.loc[2, "local_timestamp"] += pd.Timedelta(hours=1)
        self.assertFalse(episodes(food_groups(source)[7], {"2026-01-15"}, CFG))
        g = food_groups(events([0, 150, 160, 280]))[7]
        self.assertFalse(episodes(g, {"2026-01-15"}, CFG))

    def test_all_time_shifts_on_same_meal(self):
        g = food_groups(events([0, 150, 160, 350]))[7]
        t0 = g.t.iloc[1]
        rel = np.arange(-60, 166, 15)
        signal = np.where(rel > 0, np.minimum(rel / 2, 20), 0) + 100
        module, _ = helper(CFG)
        person = pd.Series(dict(input_row=7))
        with patch("prepare.points", return_value=(t0 + 60 * rel, signal)):
            rows, flow = align(person, g, {"2026-01-15"}, {}, CFG, None, module)
        self.assertEqual(flow["valid_all_offsets"], 1)
        self.assertTrue(rows[0]["full_offsets"])
        self.assertGreater(rows[0]["net_p0"], 10)
        self.assertAlmostEqual(rows[0]["pre_p0"], 100)


if __name__ == "__main__":
    unittest.main()
