"""Independent adversarial checks: same penalty, numerical boundary, cohort, and factorial contrasts."""

import unittest
import numpy as np
import run_v10  # sets the checked historical module paths
import v10_audit as audit
import v10_cv as cv
import v10_grid as grid
import v10_report as report


def forecasts(y, r2):
    return y[:, None] * (1 - np.sqrt(1 - np.asarray(r2)))


class AuditTests(unittest.TestCase):
    @staticmethod
    def near_threshold():
        # Two settings of the SAME family take turns winning. Both must fail. The
        # physical baseline columns deliberately do not match setting order.
        names = ("family@0.01", "family@0.1", "flat@0.01", "above@0.03", "below@0.03", "adaptive")
        controls = ("adaptive", "below", "family", "above", "flat")
        indices = [2, 2, 4, 3, 1, 0]
        off = np.array([0.36, 0.40, 0.42, 0.39, 0.45])
        y = np.tile([-1.0, 1.0], 43)
        panels = []
        for r in (0, 1):
            ratios = np.array([(12, 8)[r], (8, 12)[r], 10, 10 + 1e-9, 10 - 1e-9, 15.0])
            full = off[indices] / (1 - ratios / 100.0)
            panels.append(
                dict(
                    rows=np.arange(len(y)),
                    y=y,
                    groups=np.repeat(np.arange(43), 2),
                    setting_off=forecasts(y, off),
                    setting_on=forecasts(y, full),
                    on=np.empty((len(y), 0)),
                    off=np.empty((len(y), 0)),
                )
            )
        return panels, names, controls, indices

    def test_exact_threshold_and_two_different_penalties_cannot_be_combined(self):
        p, n, c, i = self.near_threshold()
        rows, _, _ = audit.evaluate(p, n, i, c, 10.0, 199, 322, width=2)
        for level in ("point", "adjusted"):
            self.assertEqual(
                [k for k in n if rows[k]["above_target_both_" + level]], ["above@0.03", "adaptive"]
            )
        self.assertEqual([rows[k]["repeats"][0]["above_target_point"] for k in n[:2]], [True, False])
        self.assertEqual([rows[k]["repeats"][1]["above_target_point"] for k in n[:2]], [False, True])
        self.assertAlmostEqual(rows["above@0.03"]["minimum_ratio_pct"], 10 + 1e-9, places=10)
        self.assertEqual(rows["flat@0.01"]["off_family"], "flat")
        # Display precision cannot decide the true sign of the threshold margin.
        self.assertEqual({f'{rows[k]["minimum_ratio_pct"]:.3f}' for k in n[2:5]}, {"10.000"})

    def test_wiring_mixup_and_duplicate_registration_rejected(self):
        p, n, c, i = self.near_threshold()
        wrong = i.copy()
        wrong[1] = 4
        with self.assertRaisesRegex(ValueError, "own named off"):
            audit.evaluate(p, n, wrong, c, 10, 199, 322)
        p[0]["rows"][0] = p[0]["rows"][1]
        with self.assertRaisesRegex(ValueError, "Duplicate rows"):
            audit.evaluate(p, n, i, c, 10, 199, 322)

    def test_joint_lower_bound_vs_resampling_households_as_whole_rows(self):
        # One deliberately strong arm and several near-target arms; use different
        # concentration of errors in the two repeats, sharing every bootstrap draw.
        rng = np.random.default_rng(61083)
        h = 32
        groups = np.repeat(np.arange(h), 1 + np.arange(h) % 3)
        y = rng.normal(size=len(groups))
        rows = np.arange(len(y))
        sst = ((y - y.mean()) ** 2).sum()
        keys = ("f@0.01", "f@0.03", "g@0.01", "rule")
        controls = ("f", "g", "rule")
        idx = np.array([0, 0, 1, 2])

        def pred(r2):
            e = rng.normal(size=(len(y), len(r2))) + 0.4 * rng.normal(size=(h, len(r2)))[groups]
            return y[:, None] - e * np.sqrt(sst * (1 - np.asarray(r2)) / np.sum(e**2, axis=0))

        panels = [
            dict(
                rows=rows,
                groups=groups,
                y=y,
                setting_off=pred([0.45, 0.48, 0.45]),
                setting_on=pred([0.505 + i * 0.002, 0.50 - i * 0.01, 0.55, 0.88]),
                off=np.empty((len(y), 0)),
                on=np.empty((len(y), 0)),
            )
            for i in (0, 1)
        ]
        draws, seed = 399, 913
        result, critical, valid = audit.evaluate(panels, keys, idx, controls, 10.0, draws, seed, width=2)

        def margins(people):
            t = y[people]
            total = np.sum((t - t.mean()) ** 2)
            out = []
            for b in panels:
                off = 1 - np.sum((t[:, None] - b["setting_off"][people]) ** 2, axis=0) / total
                on = 1 - np.sum((t[:, None] - b["setting_on"][people]) ** 2, axis=0) / total
                out.append(100 * (on - off[idx]) - 10 * on)
            return np.array(out)

        point = margins(rows)
        rng = np.random.default_rng(seed)
        maxima = []
        members = [np.flatnonzero(groups == j) for j in range(h)]
        for _ in range(draws):
            people = np.concatenate([members[j] for j in rng.integers(0, h, h)])
            maxima.append(np.max(margins(people) - point))
        ref = max(0.0, np.quantile(maxima, 0.95))
        self.assertEqual(valid, draws)
        self.assertAlmostEqual(critical, ref)
        for j, k in enumerate(keys):
            row = result[k]
            np.testing.assert_allclose(
                [s["simultaneous_lower_margin_r2_pp"] for s in row["repeats"]],
                point[:, j] - ref,
                atol=2e-11,
            )
            if row["stable_both"]:
                self.assertEqual(row["above_target_both_adjusted"], bool(min(point[:, j]) > ref))
        self.assertTrue(result["f@0.01"]["above_target_both_point"])
        self.assertFalse(result["f@0.01"]["above_target_both_adjusted"])

    def test_inner_comparison_uses_only_shared_rows_and_keeps_its_own_targets(self):
        common = np.array([1, 2, 3])
        nb, nf = len(grid.FAMILIES), len(grid.KEYS)

        def piece(ids, y):
            y = np.array(y, dtype=float)
            ids = np.array(ids)
            return (ids, y, np.zeros((len(ids), nb)), np.zeros((len(ids), nf)))

        left = piece([3, 9, 2, 1], [3.0, 9e10, 2.0, 1.0])
        right = piece([20, 2, 3, 1], [-9e10, 2.0, 3.0, 1.0])
        for p in (left, right):
            p[2][:] = p[1][:, None] * 0.7
            p[3][:] = p[1][:, None] * 0.9
        # Also alter forecasts of a nonshared person in one role.
        left[2][1, :] = 8e18
        left[3][1, :] = 3e18
        a, b = cv.training_scores([left], common), cv.training_scores([right], common)
        self.assertEqual(a[2], 3)
        np.testing.assert_allclose(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1])
        self.assertTrue(np.isfinite(a[0]).all())
        with self.assertRaisesRegex(ValueError, "exactly once"):
            cv.training_scores([left], np.array([1, 2, 3, 10]))
        with self.assertRaisesRegex(ValueError, "Repeated"):
            cv.training_scores([left, left], common)
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            cv.training_scores([left], np.array([], dtype=int))

    def test_crossed_cells_share_all_mains_and_add_only_the_declared_blocks(self):
        for cells in grid.CROSS_CELLS.values():
            recipes = [grid.FITS[n] for n in cells]
            self.assertEqual(len({r.mains for r in recipes}), 1)
            base, dose, marker, both = [set(zip(r.terms, r.weights)) for r in recipes]
            self.assertEqual(base, set(zip(grid.CORE_TERMS, (1.0, 1.0, 3.0, 3.0))))
            self.assertEqual(dose & marker, base)
            self.assertEqual(dose | marker, both)
            self.assertTrue(base < dose and base < marker)

    def test_crossed_effects_use_matched_off_and_report_extra_gain_in_points(self):
        y = np.tile([-1.0, 1.0], 40)
        groups = np.repeat(np.arange(40), 2)
        b = dict(
            rows=np.arange(len(y)),
            y=y,
            groups=groups,
            setting_off=forecasts(y, np.full(len(grid.FAMILIES), 0.34)),
            setting_on=forecasts(y, np.full(len(grid.KEYS), 0.35)),
        )
        for cells in grid.CROSS_CELLS.values():
            idx = [grid.INDEX[grid.key(c, grid.CROSS_PENALTY)] for c in cells]
            b["setting_on"][:, idx] = forecasts(y, [0.4, 0.43, 0.45, 0.51])
        out = report.crossed_comparison([b, b], 199)
        for cols in out.values():
            for key, num in [
                ("dose_minus_base", 3.0),
                ("marker_minus_base", 5.0),
                ("both_minus_base", 11.0),
                ("excess_over_sum_of_single_gains", 3.0),
            ]:
                for score in cols[key]:
                    self.assertAlmostEqual(score["delta_r2_pp"], num)
                    np.testing.assert_allclose(score["simultaneous_ci95_pp"], [num, num], atol=2e-10)
        b["setting_off"][0, grid.FAMILY_INDEX[grid.CROSS_CELLS["composition"][1]]] += 0.1
        with self.assertRaisesRegex(ValueError, "different off"):
            report.crossed_comparison([b, b], 199)
