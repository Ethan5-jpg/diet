"""Numerical checks against independent solvers and the released V7 model."""
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

import run_v8
import v7_models
import v8_models as models
from integration_fixture import Generated
from test_v7_models import direct_joint


class PairedModelTests(unittest.TestCase):
    def setUp(self):
        self.source = Generated(n=180, seed=672)
        self.split = self.source.outer_split(0, 0)
        self.threads = threadpool_limits(1)
        self.addCleanup(self.threads.restore_original_limits)

    def test_each_legacy_anchor_setting_and_all_paired_blends(self):
        old = v7_models.evaluate_split(self.source, self.split)
        new = models.evaluate_split(self.source, self.split)
        self.assertEqual(len(new), len(models.KEYS))
        self.assertEqual(tuple(new), models.KEYS)
        for p in v7_models.PENALTIES:
            before = old[v7_models.setting_key(run_v8.V7_FAMILY, p)]
            after = new[v7_models.setting_key(models.CORE, p)]
            np.testing.assert_allclose(before, after, atol=2e-10, rtol=0)
        for family in models.FAMILIES:
            keys = [v7_models.setting_key(family, p) for p in models.PENALTIES]
            for key in keys:
                np.testing.assert_array_equal(new[key][0], new[keys[0]][0])
        for family, parents in models.BLENDS.items():
            for p in models.PENALTIES:
                expected = np.mean([new[v7_models.setting_key(parent, p)] for parent in parents], axis=0)
                np.testing.assert_allclose(new[v7_models.setting_key(family, p)], expected, atol=1e-12)

    def test_all_single_model_families_match_direct_hierarchical_ridge(self):
        s = self.split
        paired = models.evaluate_split(self.source, s)
        bank, (b, bv), doses = models.background_and_features(self.source, s)
        z = np.column_stack((s.ot, s.gt))
        zv = np.column_stack((s.ov, s.gv))
        p = .3
        for m in models.MODELS:
            # Construct the unprofiled design separately. Penalize its actual
            # lower-order terms equally in both arms. Solve via augmented SVD.
            main = models.ECO + m.extra
            x = [np.column_stack([bank[label][j] for label in main]) for j in (0, 1)]
            interaction = [np.column_stack([doses[j][d] * bank[label][j] for d, label in m.products])
                           for j in (0, 1)]
            off = direct_joint(b, bv, z, zv, s.yt, *x, np.ones(len(main)))
            full = [np.column_stack(pair) for pair in zip(x, interaction, strict=True)]
            on = direct_joint(b, bv, z, zv, s.yt, *full, np.r_[np.ones(len(main)), p*m.product_weights()])
            with self.subTest(model=m.name):
                np.testing.assert_allclose(paired[v7_models.setting_key(m.name, p)], (off, on),
                                           rtol=1e-10, atol=2e-9)

    def test_training_centered_panel_decomposition_and_dose_basis(self):
        s = self.split
        # Exercise actual zero detections and an entirely undetected panel.
        # Preserve each row's total. This test checks the raw-panel features.
        raw_gut = self.source.raw["gut"]
        for column, stride in enumerate((2, 3, 5)):
            rows = np.arange(0, len(raw_gut), stride)
            raw_gut[rows, 3] += raw_gut[rows, column]
            raw_gut[rows, column] = 0.
        bank, _, doses = models.background_and_features(self.source, s)
        expected = []
        for rows in (s.train_rows, s.rows):
            raw = self.source.raw["gut"][rows, :3]  # fixture uses frozen species order
            logs = np.log1p(raw / .0001)
            basis = np.array([[1/np.sqrt(2), 1/np.sqrt(6)], [-1/np.sqrt(2), 1/np.sqrt(6)],
                              [0., -2/np.sqrt(6)]])
            expected.append(np.column_stack((logs @ basis, np.count_nonzero(raw > 0, axis=1))))
        for j in (0, 1):
            actual = np.column_stack([bank[k][j] for k in models.BALANCES + ("panel_count",)])
            np.testing.assert_allclose(actual, expected[j] - expected[0].mean(axis=0), atol=1e-13)
        self.assertGreater(np.ptp(bank["panel_count"][0]), 0.)
        x = np.column_stack((np.ones(len(s.yt)), s.dt[:, 0]))
        np.testing.assert_allclose(x.T @ doses[0][1], 0., atol=1e-10)
        for name in ("shannon_hinge", "fermentation_hinge", "shannon_x_fermentation"):
            self.assertAlmostEqual(bank[name][0].mean(), 0., places=12)

    def test_changing_held_out_targets_and_one_profile_never_changes_peer_predictions(self):
        s = self.split
        before = models.evaluate_split(self.source, s)
        s.yv[:] = np.nan
        j = s.rows[0]
        self.source.raw["gut"][j] = np.roll(self.source.raw["gut"][j], 3)
        s.gv[0] += .7
        s.dv[0] += .5
        after = models.evaluate_split(self.source, s)
        for key in before:
            for a, b in zip(before[key], after[key], strict=True):
                np.testing.assert_allclose(a[1:], b[1:], atol=1e-10, rtol=0)
