"""Independent numerical and isolation checks for generated compositions."""

import inspect
import json
import unittest

import numpy as np

from compositional_views import prepare_views


class CompositionalViewTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20261118)
        self.train = rng.dirichlet(np.ones(23), size=120)
        self.valid = rng.dirichlet(np.ones(23), size=12)
        self.taxa = [f"s__generated_{i}" for i in range(23)]

    def test_balance_and_two_part_match_direct_oracle(self):
        views, meta = prepare_views(self.train, self.valid, self.taxa, keep=4)
        selected = meta["selected_indices"]
        logs = np.log(np.maximum(self.train, 1e-6))
        valid_logs = np.log(np.maximum(self.valid, 1e-6))
        clr = logs - logs.mean(axis=1, keepdims=True)
        expected_order = np.argsort(-np.var(clr, axis=0), kind="stable")[:4]
        np.testing.assert_array_equal(selected, expected_order)
        pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        raw = np.array([[(row[selected[i]] - row[selected[j]]) / np.sqrt(2)
                         for i, j in pairs] for row in logs])
        raw_valid = np.array([[(row[selected[i]] - row[selected[j]]) / np.sqrt(2)
                               for i, j in pairs] for row in valid_logs])
        center, scale = np.mean(raw, axis=0), np.std(raw, axis=0)
        np.testing.assert_allclose(views["balance"][0], (raw - center) / scale, atol=1e-13)
        np.testing.assert_allclose(views["balance"][1], (raw_valid - center) / scale, atol=1e-13)
        train_two = np.column_stack((logs[:, selected], self.train[:, selected] > 0))
        valid_two = np.column_stack((valid_logs[:, selected], self.valid[:, selected] > 0))
        center, scale = train_two.mean(axis=0), train_two.std(axis=0)
        scale = np.where(scale <= 1e-12, 1, scale)
        np.testing.assert_allclose(views["two_part"][0], (train_two - center) / scale, atol=1e-13)
        np.testing.assert_allclose(views["two_part"][1], (valid_two - center) / scale, atol=1e-13)
        self.assertEqual(meta["column_counts"], dict(taxa=4, balance=6, two_part=8))
        self.assertNotIn("y", inspect.signature(prepare_views).parameters)
        json.dumps(meta, allow_nan=False)

    def test_validation_cannot_select_taxa_or_fit_any_other_row(self):
        original, meta = prepare_views(self.train, self.valid, self.taxa)
        changed_valid = self.valid.copy()
        changed_valid[-1] = np.r_[1.0, np.zeros(22)]
        changed, changed_meta = prepare_views(self.train, changed_valid, self.taxa)
        self.assertEqual(meta, changed_meta)
        for name in original:
            np.testing.assert_array_equal(original[name][0], changed[name][0])
            np.testing.assert_array_equal(original[name][1][:-1], changed[name][1][:-1])
        # Reordering validation rows just reorders their transformed values.
        shuffled, _ = prepare_views(self.train, self.valid[::-1], self.taxa)
        for name in original:
            np.testing.assert_array_equal(original[name][1][::-1], shuffled[name][1])

    def test_zeros_prevalence_fallback_and_constant_columns(self):
        train = np.zeros((20, 6))
        train[:, 0], train[:, 1] = 0.7, 0.3
        for row, col in enumerate(range(2, 6)):
            train[row, col] = .01
            train[row, 0] -= .01
        valid = np.array([[0.6, 0.35, .05, 0, 0, 0]])
        views, meta = prepare_views(train, valid, list("abcdef"), keep=16)
        self.assertEqual(meta["eligible_prevalent_taxa"], 2)
        self.assertEqual(meta["selected_taxa_count"], 2)
        self.assertEqual(set(meta["selected_indices"]), {0, 1})
        self.assertEqual(views["balance"][0].shape, (20, 1))
        self.assertTrue(all(np.isfinite(x).all() for values in views.values() for x in values))
        constant = np.tile([.2, .3, .5], (20, 1))
        fixed, audit = prepare_views(constant, np.array([[.4, .3, .3]]), list("abc"))
        # Zero-variance choices are stable dictionary-order ties and stay safe.
        self.assertEqual(audit["selected_indices"], [0, 1, 2])
        for name, (a, b) in fixed.items():
            np.testing.assert_allclose(a, 0, atol=1e-14)
            self.assertEqual(len(audit["constant_training_columns"][name]), a.shape[1])
            self.assertTrue(np.isfinite(b).all())

    def test_taxon_axes_and_units_are_validated(self):
        invalid = [
            (self.train, self.valid[:, :-1], self.taxa, "SHAPE"),
            (self.train, self.valid, self.taxa[:-1], "DICTIONARY"),
            (self.train, self.valid, ["same"] * 23, "DICTIONARY"),
            (self.train * 100, self.valid, self.taxa, "FRACTIONS"),
            (np.full_like(self.train, np.nan), self.valid, self.taxa, "ABUNDANCE"),
            (np.zeros_like(self.train), self.valid, self.taxa, "FRACTIONS"),
        ]
        for train, valid, taxa, error in invalid:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                prepare_views(train, valid, taxa)
        one_prevalent = np.zeros((20, 23))
        one_prevalent[:, 0] = 1
        with self.assertRaisesRegex(ValueError, "FEWER_THAN_TWO"):
            prepare_views(one_prevalent, self.valid, self.taxa)
        for keep in (1, True, 2.5):
            with self.assertRaisesRegex(ValueError, "KEEP"):
                prepare_views(self.train, self.valid, self.taxa, keep=keep)


if __name__ == "__main__":
    unittest.main()
