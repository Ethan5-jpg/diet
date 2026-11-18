"""Check paired fit mathematics, recovered controls, inactive fits and fold isolation."""

import unittest
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

import run_v9  # makes the pinned sibling packages importable
import v7_models
import v8_models
import v9_cv as cv
import v9_features as features
import v9_fit as fit
import v9_grid as grid
from integration_fixture import Generated
from test_v7_models import direct_joint


class Models(unittest.TestCase):
    def setUp(self):
        self.src = Generated(n=180, seed=672)
        self.s = self.src.outer_split(0, 0)
        limits = threadpool_limits(1)
        self.addCleanup(limits.restore_original_limits)

    def test_unique_registered_grid_and_unrestricted_parents(self):
        self.assertEqual(len(grid.KEYS), len(set(grid.KEYS)))
        self.assertEqual(len(cv.NAMES), len(set(cv.NAMES)))
        self.assertEqual(set(grid.SETTINGS), set(grid.FAMILIES))
        self.assertTrue(set(v8_models.FAMILIES).issubset(grid.FAMILIES))
        triples = grid.PARENTS["factor_gate_light_curve"]
        self.assertEqual(len(triples), 3)
        required = {
            "factor_gate_light": {grid.ANCHOR, "ecology_both_light"},
            "factor_gate_curve": {grid.ANCHOR, "fiber_curve_fermentation"},
            "factor_light_curve": {"ecology_both_light", "fiber_curve_fermentation"},
        }
        self.assertEqual(set(triples), set(required))
        for name, parents in required.items():
            self.assertTrue(parents.issubset(grid.PARENTS[name]))
        for mask, name in enumerate(grid.MATCHED_CELLS):
            for j in range(len(grid.MATCHED_BLOCKS)):
                if mask & (1 << j):
                    self.assertIn(grid.MATCHED_CELLS[mask ^ (1 << j)], grid.PARENTS[name])
        for n, ps in grid.PARENTS.items():
            self.assertTrue(set(ps) < set(grid.FAMILIES))
            self.assertNotIn(n, ps)
        for model in grid.RECIPES.values():
            for d, m in model.terms:
                self.assertTrue(grid.is_pc(m) or m in model.mains)
                self.assertTrue(d not in (3, 4) or "dose_" + str(d) in model.mains)
        self.assertEqual(len(grid.ALTERNATIVES), 24)
        self.assertNotIn("v7_gate_shannon_g1", grid.RECIPES["standalone_score_surface_fermentation"].mains)

    def test_all_v8_settings_replay_and_ensembles_own_off(self):
        pred = fit.evaluate_split(self.src, self.s)
        old = v8_models.evaluate_split(self.src, self.s)
        self.assertEqual(set(pred), set(grid.KEYS))
        for key, pair in old.items():
            np.testing.assert_allclose(pred[key], pair, atol=2e-10, rtol=0)
        for name, pp in grid.SETTINGS.items():
            null = pred[v7_models.setting_key(name, pp[0])][0]
            for p in pp:
                np.testing.assert_array_equal(pred[v7_models.setting_key(name, p)][0], null)
        for n, ps in grid.BLENDS.items():
            for p in (0.1, 10.0):
                expected = np.mean([pred[v7_models.setting_key(parent, p)] for parent in ps], axis=0)
                np.testing.assert_allclose(pred[v7_models.setting_key(n, p)], expected, rtol=0, atol=1e-12)
        for name, (_, parent) in grid.ALTERNATIVES.items():
            k = v7_models.setting_key(name, grid.SETTINGS[name][0])
            np.testing.assert_array_equal(pred[k][0], pred[v7_models.setting_key(parent, 1.0)][0])

    def test_common_main_cells_and_attenuation_preserve_their_off_predictions(self):
        pred = fit.evaluate_split(self.src, self.s)
        self.assertEqual(grid.MATCHED_CELLS[3], grid.ANCHOR)
        self.assertEqual(len(set(grid.MATCHED_CELLS)), 8)
        self.assertEqual(len(grid.ATTENUATION), 18)
        for p in grid.PENALTIES:
            anchor = pred[v7_models.setting_key(grid.ANCHOR, p)][0]
            for name in grid.MATCHED_CELLS:
                np.testing.assert_array_equal(pred[v7_models.setting_key(name, p)][0], anchor)
                self.assertEqual(grid.RECIPES[name].mains, grid.RECIPES[grid.ANCHOR].mains)
            for name, (parent, weight) in grid.ATTENUATION.items():
                off, on = pred[v7_models.setting_key(parent, p)]
                pair = pred[v7_models.setting_key(name, p)]
                np.testing.assert_array_equal(pair[0], off)
                np.testing.assert_allclose(pair[1], (1.0 - weight) * off + weight * on, rtol=0, atol=1e-12)

    def test_unprofiled_svd_for_distinct_feature_families(self):
        s = self.s
        pred = fit.evaluate_split(self.src, s)
        bank, (b, bv), doses = features.build(self.src, s)
        z, zv = np.column_stack((s.ot, s.gt)), np.column_stack((s.ov, s.gv))
        for n in (
            grid.ANCHOR,
            "factor_gate_light_curve",
            "v7_gut_pc10_ridge",
            "composition_gate_pivot",
            "dose_lower_score",
            "partition_gut3",
            "partition_gate_gut5",
            "site_joint_gate",
            "three_log_species_gate",
            "standalone_score_surface_joint",
            "common_main_fermentation_gate",
            "common_main_fermentation_dose_curve",
        ):
            spec = grid.RECIPES[n]
            mats = [
                np.column_stack(
                    [
                        *(bank[m][side] for m in spec.mains),
                        *(doses[side][d] * bank[m][side] for d, m in spec.terms),
                    ]
                )
                for side in (0, 1)
            ]
            penalties = np.r_[np.ones(len(spec.mains)), 0.3 * np.array(spec.weights)]
            pair = pred[v7_models.setting_key(n, 0.3)]
            for arm, end in enumerate((len(spec.mains), len(penalties))):
                oracle = direct_joint(
                    b, bv, z, zv, s.yt, mats[0][:, :end], mats[1][:, :end], penalties[:end]
                )
                np.testing.assert_allclose(pair[arm], oracle, rtol=0, atol=2e-8, err_msg=n)

    def test_three_alternative_solvers_against_oracles(self):
        # Identity validation rows expose the interaction coefficients themselves.
        x = np.array([[1.4, 0.3, -0.1], [0.1, 0.8, 0.2], [0.3, 0.1, 1.6]])
        q, c = x.T @ x, np.array([0.8, -0.65, 0.04])
        state = SimpleNamespace(gram=q, rhs=c, projected_valid=np.eye(3), null=np.zeros(3))
        b = fit.InteractionBlock(state, [], list(range(3)), [1.0, 1.0, 1.0])
        target = np.linalg.solve(q + grid.ALT_RIDGE * np.eye(3), c)
        np.testing.assert_allclose(b.pcr([3])[:, 0], target, atol=1e-12)
        np.testing.assert_allclose(b.krylov([1, 3])[:, -1], target, atol=1e-12)
        self.assertGreater(np.linalg.norm(b.krylov([1])[:, 0] - target), 1e-3)
        for j, frac in enumerate(grid.OPTIONS["elastic"]):
            coef = b.elastic(grid.OPTIONS["elastic"])[:, j]
            grad = (q + grid.ALT_RIDGE * np.eye(3)) @ coef - c
            lam = frac * max(abs(c))
            for v, g in zip(coef, grad, strict=True):
                if v:
                    self.assertAlmostEqual(g, -np.sign(v) * lam, places=7)
                else:
                    self.assertLessEqual(abs(g), lam + 1e-8)
        state.rhs[:] = 0.0
        zero = fit.InteractionBlock(state, [], list(range(3)), [1.0] * 3)
        for name, options in grid.OPTIONS.items():
            np.testing.assert_array_equal(getattr(zero, name)(options), np.zeros((3, len(options))))

    def test_all_fits_ignore_held_out_y_and_one_held_out_profile_cannot_change_peers(self):
        s = self.s
        before = fit.evaluate_split(self.src, s)
        s.yv[:] = np.nan
        row = s.rows[0]
        for site in ("gut", "oral"):
            self.src.raw[site][row] = np.roll(self.src.raw[site][row], 4)
        s.dv[0] += 0.1
        s.gv[0] += 0.3
        after = fit.evaluate_split(self.src, s)
        for k in before:
            for x, y in zip(before[k], after[k], strict=True):
                np.testing.assert_allclose(x[1:], y[1:], atol=1e-9, rtol=0, err_msg=k)

    def test_empirical_ranks_and_partition_degeneracy(self):
        np.testing.assert_allclose(
            features.empirical_rank(np.array([1, 2, 2, 4]), [0, 1, 2, 4, 5]), [0, 0.125, 0.5, 0.875, 1]
        )
        a, b = features.indicators(np.ones(6), np.ones(2), 3)
        self.assertFalse(a.any() or b.any())
        bank, _, d = features.build(self.src, self.s)
        for m in (
            "rank_shannon",
            "bounded_fermentation",
            "oral_dominance",
            "gut_simpson",
            "dose_3",
            *grid.tree_labels("joint2"),
            *grid.region_labels("gut5"),
        ):
            self.assertAlmostEqual(bank[m][0].mean(), 0.0, places=11)
        np.testing.assert_array_equal(d[1][2], self.s.dv[:, 1])

    def test_sparse_float32_entropy_matches_definition_and_full_grid_remains_finite(self):
        # Float32 cannot represent the old log floor of 1e-300. Real zero
        # abundances must contribute exactly zero to entropy at either precision.
        raw = self.src.raw["oral"].astype(np.float32)
        raw[:, 2] += raw[:, :2].sum(axis=1)
        raw[:, :2] = 0
        raw[self.s.rows[0], :] = 0
        raw[self.s.rows[0], 3] = 1
        self.src.raw["oral"] = raw

        def entropy(row):
            positive = row[row > 0].astype(float)
            positive /= positive.sum()
            return -positive @ np.log(positive)

        with np.errstate(divide="raise", invalid="raise"):
            bank, _, _ = features.build(self.src, self.s)
            pairs = fit.evaluate_split(self.src, self.s)
        target = np.array([entropy(row) for row in raw])
        for side, rows in enumerate((self.s.train_rows, self.s.rows)):
            np.testing.assert_allclose(
                bank["oral_shannon"][side],
                target[rows] - target[self.s.train_rows].mean(),
                rtol=0,
                atol=1e-13,
            )
        self.assertTrue(all(np.isfinite(a).all() for arms in pairs.values() for a in arms))
