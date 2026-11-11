import json
import sys
import unittest

import numpy as np

from external import (BASE_NAME, SOURCE, aggregate, evaluate, panel_scores, person_means,
                      verified_base)


CFG = json.loads((SOURCE / "config.json").read_bytes())
BASE = SOURCE.parent / BASE_NAME
_, MODELS, _ = verified_base(BASE, CFG["base_code"])
sys.path.insert(0, str(BASE))
try:
    from test_daily import generated
finally:
    sys.path.pop(0)


class ExternalTests(unittest.TestCase):
    def test_panel_requires_unique_exact_columns(self):
        mapping = CFG["taxa"]
        names = [label for label in mapping.values() if label is not None] + ["s__Other"]
        abundance = np.array([[.04, .04, .04, .04, .04, .04, .04, .72]])
        scores = panel_scores(abundance, names, mapping)
        self.assertAlmostEqual(scores["sum7"][0], .28)
        self.assertAlmostEqual(scores["sum_without_coli"][0], .24)
        self.assertAlmostEqual(scores["log_sum7"][0], np.log1p(28))
        names[-1] = "s__Escherichia_coli"
        with self.assertRaisesRegex(ValueError, "distinct"):
            panel_scores(abundance, names, mapping)

    def test_within_recovery_and_between_counterpart(self):
        daily, marker = generated()
        persons = person_means(daily)
        scores = {"sum7": marker[:, 0]}
        design = MODELS.background(daily)
        direct = evaluate(daily, persons, scores, design, MODELS, .03)
        self.assertEqual(len(direct), 6)
        within = direct["within_sum7_carb"]["contrasts"]["carb"]
        self.assertGreater(within["ci95"][0], 1)
        self.assertLess(direct["within_sum7_carb"]["holm_p"], .001)
        self.assertGreater(direct["between_sum7_carb"]["p"], .05)
        train = np.arange(len(persons)) // 2 % 5 != 0
        fold = evaluate(daily, persons, scores, design, MODELS, .03, train)
        result = aggregate([dict(repeat=0, results=fold)])
        self.assertGreater(result["within_sum7_carb"]["r0"]["delta_r2"], .6)


if __name__ == "__main__":
    unittest.main()
