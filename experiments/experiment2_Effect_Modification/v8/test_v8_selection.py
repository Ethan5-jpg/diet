import unittest

import numpy as np
from threadpoolctl import threadpool_limits

import run_v8
import v8_selection as select
from integration_fixture import Generated
from v8_models import KEYS


class SelectionTests(unittest.TestCase):
    def test_maximin_uses_paired_ratio_and_full_old_grid_stays_separate(self):
        rows = np.arange(60)
        y = np.tile([-1., 1.], 30)
        groups = rows // 2
        a = "pc10_ecology@0.1"
        b = "marker_product@0.3"
        c = "fiber_curve_ecology@0.1"
        d = "marker_hinges@1"
        def predictions(ratios):
            # Synthetic predictions with analytically known ratios and unequal
            # family-specific baselines. Every family's own off is shared across λ.
            baseline = np.array([.3 if k.startswith("marker_product@") else .1 for k in KEYS])
            target = np.array([ratios.get(k, -2.) for k in KEYS])
            full = baseline / (1 - target/100)
            return (y, groups, y[:, None] + np.sqrt(1 - baseline), y[:, None] + np.sqrt(1 - full))
        p0 = predictions({a: 9., b: 7.1, c: 11., d: 6.6})
        p1 = predictions({a: 1., b: 7.5, c: -.2, d: 6.9})
        choices, audit = select.select((rows, p0), (rows[::-1], tuple(arr[::-1] for arr in p1)))
        self.assertEqual(choices[select.REFERENCE].describe()["settings"], [a])
        self.assertEqual(choices[select.PRIMARY].describe()["settings"], [b])
        self.assertEqual(choices[select.ENSEMBLE].describe()["settings"], [b, d, a])
        self.assertEqual(audit["eligible_positive_in_both"], 3)
        # Missing one person's entire inner prediction in repeat 2 only.
        _, limited = select.select((rows, p0), (rows[1:], tuple(arr[1:] for arr in p1)))
        self.assertEqual(limited["common_inner_people"], 59)
        negative = predictions({})
        disabled, _ = select.select((rows, negative), (rows, negative))
        self.assertTrue(all(not choice.enabled for choice in disabled.values()))

    def test_distinct_nested_partitions_never_share_outer_households(self):
        src = Generated(n=180)
        original, outer = src.roles(1, 3)
        other = select.second_inner(src, 1, 3)
        original_sets = {frozenset(src.groups[s.rows]) for s in original}
        new_sets = {frozenset(src.groups[s.rows]) for s in other}
        self.assertNotEqual(original_sets, new_sets)
        np.testing.assert_array_equal(np.sort(np.concatenate([s.rows for s in other])), np.sort(outer.train_rows))
        for s in other:
            self.assertFalse(set(src.groups[s.rows]) & set(src.groups[s.train_rows]))
            self.assertFalse(set(src.groups[outer.rows]) & set(src.groups[np.r_[s.train_rows, s.rows]]))

    def test_outer_targets_do_not_change_any_choice_or_prediction(self):
        src = Generated()
        y = src.y.copy()
        y[src.reference["outer"][0] == 2] += 1000
        with threadpool_limits(1):
            before, m0 = select.fold_predictions(src, 0, 2)
            after, m1 = select.fold_predictions(src.with_y(y), 0, 2)
        self.assertEqual(m0, m1)
        self.assertFalse(np.array_equal(before["y"], after["y"]))
        for key in ("off", "on"):
            np.testing.assert_array_equal(before[key], after[key])
