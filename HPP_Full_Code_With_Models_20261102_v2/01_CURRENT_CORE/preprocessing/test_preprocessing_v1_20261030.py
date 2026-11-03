"""Synthetic checks for mathematical behavior and frozen validation parameters."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from preprocessing_v1_20261030 import MicrobiomeCLR, PhenotypeOutliers


class PreprocessingChecks(unittest.TestCase):
    def setUp(self):
        raw = np.column_stack([
            np.arange(1, 21), np.arange(21, 1, -1),
            np.r_[2.0, np.zeros(19)], np.zeros(20),
        ])
        self.x = raw / raw.sum(axis=1, keepdims=True)
        self.names = ["abundant_a", "abundant_b", "five_pct", "absent"]
        self.ids = [f"synthetic_{i}" for i in range(20)]

    def micro(self):
        return MicrobiomeCLR().fit(self.x, self.names, self.ids, site="oral")

    def test_prevalence_boundary_and_feature_specific_zero(self):
        obj = self.micro()
        self.assertEqual(obj.state["reference_indices"], [0, 1, 2])
        self.assertAlmostEqual(obj.state["pseudocounts"][2], self.x[0, 2]/2)
        z = obj.transform(self.x, self.names, site="oral")["values"]
        np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(z.std(axis=0), 1, atol=1e-12)

    def test_clr_hand_calculation_and_no_positive_censoring(self):
        a = np.array([[0.0, 0.2, 0.00001]])
        result = MicrobiomeCLR._clr(a, np.array([0.01, 0.15, 0.1]))
        expected = np.log([0.01, 0.2, 0.00001])
        np.testing.assert_allclose(result[0], expected-expected.mean())
        self.assertAlmostEqual(float(result.sum()), 0)

    def test_validation_is_pointwise_without_refitting(self):
        obj = self.micro()
        before = repr(obj.state)
        single = obj.transform(self.x[:1], self.names, site="oral")["values"]
        batch = obj.transform(self.x, self.names, site="oral")["values"]
        np.testing.assert_allclose(single[0], batch[0])
        self.assertEqual(before, repr(obj.state))

    def test_full_missing_profile_and_wrong_units_rejected(self):
        for bad in [np.zeros_like(self.x), self.x*100, np.full_like(self.x, np.nan)]:
            with self.assertRaises(ValueError):
                MicrobiomeCLR().fit(bad, self.names, self.ids, site="oral")

    def test_daily_repeat_ids_and_validation_fit_rejected(self):
        with self.assertRaises(ValueError):
            MicrobiomeCLR().fit(self.x, self.names, ["same"]*20, site="oral")
        with self.assertRaises(ValueError):
            MicrobiomeCLR().fit(self.x, self.names, self.ids, site="oral", role="confirmation")

    def test_wrong_taxon_order_and_site_rejected(self):
        obj = self.micro()
        with self.assertRaises(ValueError):
            obj.transform(self.x, self.names[::-1], site="oral")
        with self.assertRaises(ValueError):
            obj.transform(self.x, self.names, site="gut")

    def test_unsupported_reference_not_fabricated(self):
        obj = self.micro()
        v = np.array([[0, 0, 0, 1.0]])
        result = obj.transform(v, self.names, site="oral")
        self.assertFalse(result["supported"][0])
        self.assertTrue(np.isnan(result["values"]).all())

    def test_robust_core_is_shortest_interval_not_central_trim(self):
        a = np.r_[np.arange(95.0), np.arange(10000.0, 10005.0)]
        obj = PhenotypeOutliers().fit(a, range(100), endpoint="synthetic")
        self.assertEqual(obj.state["interval_n"], 95)
        self.assertEqual(obj.state["interval_low"], 0)
        self.assertEqual(obj.state["interval_high"], 94)
        self.assertEqual(obj.state["mean"], 47)
        self.assertAlmostEqual(obj.state["sd"], np.std(np.arange(95.0), ddof=1))

    def test_endpoint_boundaries_missing_and_flags(self):
        obj = PhenotypeOutliers().fit(np.arange(100.0), range(100), endpoint="synthetic")
        m, s = obj.state["mean"], obj.state["sd"]
        a = np.array([m, m+5*s, m+6*s, m+8*s, m+9*s, m-6*s, np.nan])
        raw = a.copy()
        result = obj.transform(a, endpoint="synthetic")
        np.testing.assert_array_equal(result["excluded_8sd"], [0,0,0,0,1,0,0])
        self.assertAlmostEqual(result["values"][2], m+5*s)
        self.assertAlmostEqual(result["values"][3], m+5*s)
        self.assertAlmostEqual(result["values"][5], m-5*s)
        self.assertTrue(np.isnan(result["values"][[4,6]]).all())
        np.testing.assert_array_equal(a, raw)

    def test_endpoint_scale_failure_and_identifier_guard(self):
        for a in [np.ones(20), np.array([1.0, np.nan, 2.0])]:
            with self.assertRaises(ValueError):
                PhenotypeOutliers().fit(a, range(len(a)), endpoint="synthetic")
        obj = PhenotypeOutliers().fit(np.arange(20.0), range(20), endpoint="synthetic")
        with self.assertRaises(ValueError):
            obj.transform([1.0], endpoint="different")

    def test_save_load_parity_for_both_transforms(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/"synthetic_parameters.json"
            obj = self.micro()
            obj.save(p)
            loaded = MicrobiomeCLR.load(p)
            np.testing.assert_array_equal(
                obj.transform(self.x, self.names, site="oral")["values"],
                loaded.transform(self.x, self.names, site="oral")["values"])
            obj = PhenotypeOutliers().fit(np.arange(100.0), range(100), endpoint="synthetic")
            obj.save(p)
            loaded = PhenotypeOutliers.load(p)
            a = [0.0, 10000.0, np.nan]
            np.testing.assert_array_equal(obj.transform(a, endpoint="synthetic")["values"],
                                          loaded.transform(a, endpoint="synthetic")["values"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
