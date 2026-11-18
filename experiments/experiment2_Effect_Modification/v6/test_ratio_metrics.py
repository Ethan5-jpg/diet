"""Independent score and direct person-resampling checks for v6 statistics."""

import json
import unittest

import numpy as np

from ratio_metrics import select_correction, summarize_paired


def direct_oracle(y, baseline, full, groups, draws, seed):
    """Rebuild complete person samples; does not use grouped sufficient statistics."""
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == value) for value in unique]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        selected = rng.integers(0, len(unique), size=len(unique))
        rows = np.concatenate([members[j] for j in selected])
        target = y[rows]
        sst = np.sum((target - target.mean()) ** 2)
        if sst <= 0:
            values.append([np.nan] * (1 + 2 * full.shape[1]))
            continue
        sse0 = np.sum((target - baseline[rows]) ** 2)
        sse1 = np.sum((target[:, None] - full[rows]) ** 2, axis=0)
        values.append(np.r_[1 - sse0 / sst, 1 - sse1 / sst, (sse0 - sse1) / sst])
    return np.asarray(values)


class RatioMetricsTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(18)
        self.groups = np.repeat(np.arange(65), rng.integers(1, 5, size=65))
        self.y = rng.normal(size=len(self.groups))
        self.base = self.y + rng.normal(0, 0.55, len(self.y))
        self.full = np.column_stack([
            self.y + rng.normal(0, .45, len(self.y)),
            self.y + rng.normal(0, .75, len(self.y)),
        ])

    def test_points_and_paired_household_bootstrap_against_direct_oracle(self):
        draws, seed = 499, 913
        got = summarize_paired(self.y, self.base, self.full, self.groups,
                               names=("good", "bad"), draws=draws, seed=seed)
        oracle = direct_oracle(self.y, self.base, self.full, self.groups, draws, seed)
        sst = np.sum((self.y - self.y.mean()) ** 2)
        b0 = 1 - np.sum((self.y - self.base) ** 2) / sst
        self.assertAlmostEqual(got["baseline_r2"], b0, places=13)
        np.testing.assert_allclose(got["baseline_r2_ci95"], np.quantile(oracle[:, 0], [.025, .975]), atol=1e-12)
        for j, name in enumerate(("good", "bad")):
            row = got["routes"][name]
            bj = 1 - np.sum((self.y - self.full[:, j]) ** 2) / sst
            self.assertAlmostEqual(row["full_r2"], bj, places=13)
            self.assertAlmostEqual(row["delta_r2"], bj - b0, places=13)
            delta = oracle[:, 3 + j]
            denominator = oracle[:, 1 + j]
            np.testing.assert_allclose(row["delta_r2_ci95"], np.quantile(delta, [.025, .975]), atol=1e-12)
            np.testing.assert_allclose(row["ratio_over_full_ci95_pct"],
                                       np.quantile(100 * delta / denominator, [.025, .975]), atol=1e-10)
            np.testing.assert_allclose(row["relative_r2_change_over_base_ci95_pct"],
                                       np.quantile(100 * delta / oracle[:, 0], [.025, .975]), atol=1e-10)
        json.dumps(got, allow_nan=False)
        self.assertLess(got["routes"]["bad"]["ratio_over_full_pct"], 0)

    def test_negative_r2_does_not_become_positive_interaction_share(self):
        # A constant shift makes the negative full R2 and negative delta ratio positive arithmetically.
        y = np.linspace(-1, 1, 100)
        groups = np.repeat(np.arange(50), 2)
        got = summarize_paired(y, y + 2, y + 3, groups, draws=199)
        row = got["routes"]["candidate_0"]
        self.assertLess(row["delta_r2"], 0)
        self.assertGreater(row["arithmetic_ratio_over_full_pct"], 0)
        self.assertIsNone(row["ratio_over_full_pct"])
        self.assertIsNone(row["ratio_over_full_ci95_pct"])
        self.assertFalse(row["rankable_positive_gain"])

    def test_zero_and_uncertain_denominators_withhold_ratio_interval(self):
        y = np.linspace(-1, 1, 120)
        groups = np.repeat(np.arange(60), 2)
        base = np.zeros_like(y)
        zero = summarize_paired(y, base, base, groups, draws=199)["routes"]["candidate_0"]
        self.assertIsNone(zero["ratio_over_full_pct"])
        self.assertIsNone(zero["ratio_over_full_ci95_pct"])
        # Very small positive R2: resampled denominators cross zero.
        uncertain = summarize_paired(y, base, y * .00001, groups, draws=499)["routes"]["candidate_0"]
        self.assertGreater(uncertain["full_r2"], 0)
        self.assertIsNotNone(uncertain["ratio_over_full_pct"])
        self.assertIsNone(uncertain["ratio_over_full_ci95_pct"])
        self.assertEqual(uncertain["ratio_over_full_status"], "denominator_ci_reaches_zero")

    def test_result_is_chunk_independent_and_outcome_shift_invariant(self):
        a = summarize_paired(self.y, self.base, self.full, self.groups, draws=199, chunk_size=1)
        b = summarize_paired(self.y, self.base, self.full, self.groups, draws=199, chunk_size=37)
        c = summarize_paired(self.y + 100000, self.base + 100000, self.full + 100000,
                             self.groups, draws=199, chunk_size=37)
        for name in a["routes"]:
            for key in ("delta_r2_ci95", "ratio_over_full_ci95_pct"):
                np.testing.assert_allclose(a["routes"][name][key], b["routes"][name][key], atol=1e-10)
                np.testing.assert_allclose(a["routes"][name][key], c["routes"][name][key], atol=1e-8)

    def test_constant_outcome_bootstrap_is_counted_not_fabricated(self):
        # Single-person households can produce constant samples; the point sample is valid.
        y = np.array([0., 1.])
        got = summarize_paired(y, y, y, np.array([0, 1]), draws=199)
        self.assertGreater(got["bootstrap_undefined_outcome_fraction"], .4)
        row = got["routes"]["candidate_0"]
        self.assertIsNone(row["delta_r2_ci95"])
        self.assertIsNone(row["ratio_over_full_ci95_pct"])

    def test_inner_selection_keeps_null_and_never_changes_baseline(self):
        y = np.arange(10, dtype=float)
        baseline = y + 1
        same = select_correction(y, baseline, np.zeros((10, 2)), names=("strong", "weak"))
        self.assertEqual(same["name"], "zero")
        self.assertIsNone(same["index"])
        better = select_correction(y, baseline, np.column_stack((np.full(10, -1.), np.full(10, 1.))),
                                   names=("better", "worse"))
        self.assertEqual(better["name"], "better")
        self.assertEqual(better["index"], 0)
        self.assertEqual(better["sse_gain"], 10.)
        empty = select_correction(y, baseline, np.empty((10, 0)), names=())
        self.assertTrue(empty["is_zero"])

    def test_exact_null_and_single_vector_prediction(self):
        got = summarize_paired(self.y, self.base, self.base, self.groups, draws=199)
        row = got["routes"]["candidate_0"]
        self.assertEqual(row["delta_r2"], 0.)
        self.assertEqual(row["delta_r2_ci95"], [0., 0.])
        self.assertEqual(row["ratio_over_full_pct"], 0.)
        self.assertEqual(row["ratio_over_full_ci95_pct"], [0., 0.])

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            summarize_paired(self.y, self.base, self.full, np.full(len(self.y), ""), draws=199)
        with self.assertRaises(ValueError):
            summarize_paired(self.y, self.base, self.full, np.zeros(len(self.y)), draws=199)
        bad = self.groups.astype(object)
        bad[0] = None
        with self.assertRaises(ValueError):
            summarize_paired(self.y, self.base, self.full, bad, draws=199)
        with self.assertRaises(ValueError):
            summarize_paired(np.ones(len(self.y)), self.base, self.full, self.groups, draws=199)
        with self.assertRaises(ValueError):
            summarize_paired(self.y, self.base, self.full, self.groups, names=("a", "a"), draws=199)
        with self.assertRaises(ValueError):
            summarize_paired(self.y, self.base, self.full, self.groups, draws=2)
        with self.assertRaises(ValueError):
            select_correction(self.y, self.base, self.full, names=("zero", "a"))
        mixed = self.groups.astype(object)
        mixed[0] = str(mixed[0])
        with self.assertRaises(ValueError):
            summarize_paired(self.y, self.base, self.full, mixed, draws=199)


if __name__ == "__main__":
    unittest.main()
