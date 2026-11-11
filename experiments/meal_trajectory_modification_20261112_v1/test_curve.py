import unittest

import numpy as np
import pandas as pd

from prepare_curve import TAGS, segment_means, window_segments


class CurveTests(unittest.TestCase):
    def test_integration_respects_internal_knots_and_bin_boundaries(self):
        times = [0, 20, 50, 120]
        glucose = [100, 110, 105, 100]
        actual = segment_means(times, glucose, 100)
        expected_auc = 20 * 5 + 30 * 7.5 + 70 * 2.5
        self.assertAlmostEqual(sum(actual) * 30, expected_auc)
        self.assertAlmostEqual(actual[0], (20 * 5 + 10 * (10 + 25 / 3) / 2) / 30)

    def test_replay_checks_all_offsets_and_retains_original_window_key(self):
        class CGM:
            @staticmethod
            def response(times, glucose, start, cfg):
                return dict(cgm_valid=True, baseline_glucose=100., peak_rise_0_120=12.), (
                    np.array([0., 120.]), np.array([100., 112.]), 100.)

        row = dict(input_row=3, start=1234.)
        for tag in TAGS.values():
            row.update({f"net_{tag}": 6., f"pre_{tag}": 100., f"peak_{tag}": 12.})
        meals = pd.DataFrame([row])
        result, error = window_segments(meals, [], [], {}, CGM)
        self.assertEqual(error, 0.)
        self.assertEqual(result[0]["input_row"], 3)
        self.assertEqual(result[0]["start"], 1234.)
        self.assertEqual([result[0][f"mean_p0_{i}"] for i in range(4)], [1.5, 4.5, 7.5, 10.5])
        with self.assertRaisesRegex(ValueError, "disagree"):
            window_segments(meals.assign(net_m15=7.), [], [], {}, CGM)


if __name__ == "__main__":
    unittest.main()
