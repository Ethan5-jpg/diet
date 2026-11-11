import unittest

import numpy as np

from meal_function import MODES, summarize


class MatchedLossTests(unittest.TestCase):
    def test_functional_increment_uses_its_matched_reference(self):
        losses = dict(none=np.array([12., 14.]), tss=np.array([30., 40.]),
                      taxonomy=np.array([10., 12.]), combined=np.array([9., 13.]))
        for j, mode in enumerate(MODES):
            losses[mode] = losses["none"] - j
        draw = np.array([[0, 1], [1, 1], [0, 0]])
        result = summarize(losses, draw)
        self.assertEqual(result[f"{MODES[2]}_vs_additive"]["delta_r2"], 4 / 70)
        comparison = result["function_plus_taxonomy_vs_taxonomy"]
        self.assertEqual(comparison["delta_r2"], 0)
        self.assertLess(comparison["ci95"][0], 0)
        self.assertGreater(comparison["ci95"][1], 0)


if __name__ == "__main__":
    unittest.main()
