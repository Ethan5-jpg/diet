import unittest

import numpy as np

from v7_reference import HISTORICAL_RATIO
from v7_threshold import annotate


class ThresholdTest(unittest.TestCase):
    def test_paired_simultaneous_margin_uses_only_named_families(self):
        names = ["small", "selector", "large", "unstable", "duplicate"]
        full = np.full(len(names), .2)
        margin = np.array([.4, 999., 2., 3., 2.])
        delta = (margin + HISTORICAL_RATIO * full) / 100
        records = {n: dict(delta_r2=d, full_r2=f, ratio_over_full_status="stable")
                   for n, d, f in zip(names, delta, full, strict=True)}
        records["unstable"]["ratio_over_full_status"] = "R2_interval_reaches_zero"
        x = np.linspace(-1, 1, 101)
        noise = np.column_stack((x, x * 900, -x, x / 2, -x))
        boot = np.zeros((len(x), 3, len(names)))
        boot[:, 1] = full + .02*x[:, None]
        boot[:, 2] = (margin + noise + HISTORICAL_RATIO*boot[:, 1]) / 100
        family = ("large", "small", "unstable", "duplicate")
        out = annotate(records, names, boot, family, True)
        expected_critical = np.quantile(abs(x), .95)
        self.assertAlmostEqual(out["critical_margin_pp"], expected_critical)
        self.assertEqual(out["supported_families"], ["large", "duplicate"])
        self.assertIsNone(records["unstable"]["exceeds_benchmark_simul_lower"])
        self.assertNotIn("threshold_excess_pp", records["selector"])
        for name in family:
            row = records[name]
            self.assertAlmostEqual(row["threshold_excess_simul_lower95_pp"],
                                   margin[names.index(name)] - expected_critical)
        out = annotate(records, names, boot[:0], family, False)
        self.assertIsNone(out["critical_margin_pp"])
        self.assertIsNone(records["small"]["exceeds_benchmark_simul_lower"])
