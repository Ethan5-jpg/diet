"""Tests of threshold arithmetic, family pairing, selection, and data isolation."""

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import run_v10 as runner
import v10_grid as grid
import v10_cv as cv
import v10_report as report
from v7_eval import _check_roles
from v8_selection import second_inner
from integration_fixture import Generated


def score_fixture(draws=199):
    # A has 12%/8%, B has 8%/12%, C has 11%/11%, D sits exactly on 10%.
    # Using each setting's distinct full R² prevents a single shared off baseline.
    y = np.tile([-1.0, 1.0], 80)
    groups = np.repeat(np.arange(80), 2)

    def predictions(r2):
        return y[:, None] * (1.0 - np.sqrt(1 - np.asarray(r2)))

    panels = []
    for ratios in ((0.12, 0.08, 0.11, 0.10), (0.08, 0.12, 0.11, 0.10)):
        full = np.array([0.5, 0.6, 0.7, 0.4])
        off = full * (1 - np.array(ratios))
        panels.append(
            dict(
                rows=np.arange(len(y)),
                groups=groups,
                y=y,
                setting_on=predictions(full),
                setting_off=predictions(off),
                off=np.empty((len(y), 0)),
                on=np.empty((len(y), 0)),
            )
        )
    return panels, report.evaluate(
        panels, ("a", "b", "c", "d"), np.arange(4), ("a", "b", "c", "d"), grid.TARGET, draws, 921, width=2
    )


class InferenceTests(unittest.TestCase):
    def test_strict_threshold_own_off_and_identical_setting(self):
        panels, (rows, _, n) = score_fixture()
        self.assertEqual(n, 199)
        self.assertEqual([k for k, v in rows.items() if v["above_target_both_point"]], ["c"])
        self.assertEqual([k for k, v in rows.items() if v["above_target_both_adjusted"]], ["c"])
        for j, p in enumerate(panels):
            for i, k in enumerate(rows):
                y = p["y"]
                base = 1 - ((y - p["setting_off"][:, i]) ** 2).sum() / ((y - y.mean()) ** 2).sum()
                full = 1 - ((y - p["setting_on"][:, i]) ** 2).sum() / ((y - y.mean()) ** 2).sum()
                s = rows[k]["repeats"][j]
                self.assertAlmostEqual(100 * (full - base) / full, s["ratio_pct"])
                self.assertAlmostEqual(100 * (full - base), s["delta_r2_pp"])

    def test_nonpositive_denominator_and_misaligned_families(self):
        panels, _ = score_fixture(99)
        for p in panels:
            p["setting_on"][:, 0] = -p["y"]
        rows, _, _ = report.evaluate(
            panels, ("a", "b", "c", "d"), np.arange(4), ("a", "b", "c", "d"), 10.0, 99, 0
        )
        self.assertIsNone(rows["a"]["minimum_ratio_pct"])
        self.assertFalse(rows["a"]["stable_both"])
        panels[1]["groups"] = panels[1]["groups"][::-1]
        with self.assertRaisesRegex(ValueError, "Unpaired"):
            report.evaluate(panels, ("a", "b", "c", "d"), np.arange(4), ("a", "b", "c", "d"), 10.0, 99, 0)

    def test_paired_bootstrap_vs_explicit_household_sampling(self):
        rng = np.random.default_rng(407)
        groups = np.repeat(np.arange(21), 1 + np.arange(21) % 4)
        y = rng.normal(size=len(groups))
        a = 0.55 * y + rng.normal(0, 0.12, len(y))
        b = 0.64 * y + rng.normal(0, 0.12, len(y))
        batch = dict(
            rows=np.arange(len(y)),
            groups=groups,
            y=y,
            setting_off=a[:, None],
            setting_on=b[:, None],
            off=np.empty((len(y), 0)),
            on=np.empty((len(y), 0)),
        )
        draws, seed = 199, 312
        result, _, _ = report.evaluate([batch, batch], ["only"], [0], ["only"], 10.0, draws, seed)
        reference = []
        rng = np.random.default_rng(seed)
        for _ in range(draws):
            indices = np.concatenate([np.flatnonzero(groups == h) for h in rng.integers(0, 21, size=21)])
            t = y[indices]
            sst = np.sum((t - t.mean()) ** 2)
            base = 1 - np.sum((t - a[indices]) ** 2) / sst
            full = 1 - np.sum((t - b[indices]) ** 2) / sst
            reference.append(100 * (full - base) / full)
        np.testing.assert_allclose(
            result["only"]["repeats"][0]["ratio_ci95_pct"],
            np.quantile(reference, [0.025, 0.975]),
            atol=1e-9,
        )

    def test_guard_and_inner_workflows_match_their_own_baselines(self):
        n = len(grid.KEYS)
        first, second = np.full(n, -1.0), np.full(n, -1.0)
        base = np.full(n, 0.12)
        candidates = [
            grid.INDEX[grid.key(k, 0.01)] for k in (grid.ANCHOR, "score_knot_35", "overlay_balances_6")
        ]
        for i, s in zip(candidates, [8.0, 11.0, 9.0]):
            first[i] = second[i] = s
        base[candidates[1]] -= 0.02
        choices = cv.choose((first, base, 100), (second, base, 100))
        self.assertEqual(choices["inner_maximin"]["keys"], [grid.KEYS[candidates[1]]])
        self.assertEqual(choices["inner_anchor_guard"]["keys"], [grid.KEYS[candidates[2]]])
        self.assertEqual(len(choices["inner_diverse_blend"]["keys"]), 2)
        rng = np.random.default_rng(4)
        x, z = rng.normal(size=(7, len(grid.FAMILIES))), rng.normal(size=(7, n))
        a, b = cv.adaptive_pairs(x, z, choices)
        ix = [candidates[2], candidates[0]]
        np.testing.assert_allclose(a[:, -1], x[:, np.array(grid.OFF_INDEX)[ix]].mean(axis=1))
        np.testing.assert_allclose(b[:, -1], z[:, ix].mean(axis=1))
        negative = (np.full(n, -1.0), base, 100)
        empty = cv.choose(negative, negative)
        a, b = cv.adaptive_pairs(x, z, empty)
        np.testing.assert_array_equal(a, b)

    def test_inner_roles_and_outer_outcome_isolation(self):
        src = Generated(n=180, seed=811)
        inner, outer = src.roles(0, 0)
        _check_roles(src.groups, src.reference["outer"][0], 0, inner, outer)
        alt = second_inner(src, 0, 0)
        _check_roles(src.groups, src.reference["outer"][0], 0, alt, outer)
        self.assertNotEqual({frozenset(s.rows) for s in inner}, {frozenset(s.rows) for s in alt})
        # Changing every outer outcome leaves inner choices and both forecast arms identical.
        target = src.y.copy()
        target[outer.rows] += 103.0
        modified = src.with_y(target)
        a, m = cv.fold_predictions(src, 0, 0)
        b, n = cv.fold_predictions(modified, 0, 0)
        self.assertEqual(m, n)
        self.assertFalse(np.array_equal(a["y"], b["y"]))
        for name in ("setting_off", "setting_on", "adaptive_off", "adaptive_on"):
            np.testing.assert_array_equal(a[name], b[name])

    def test_grid_coverage_and_integrity(self):
        self.assertEqual(len(grid.KEYS), len(set(grid.KEYS)))
        self.assertTrue({grid.PRIMARY, grid.REFERENCE}.issubset(grid.INDEX))
        self.assertEqual(len(grid.CELLS), 8)
        self.assertEqual(len(set(grid.CELLS)), 8)
        self.assertEqual(len(report.OFF), len(report.ALL))
        self.assertEqual(set(report.OFF), set(range(len(grid.FAMILIES) + len(grid.ADAPTIVE))))
        for n in grid.FAMILIES:
            for p in grid.SETTINGS[n]:
                self.assertEqual(report.OFF[grid.INDEX[grid.key(n, p)]], grid.FAMILY_INDEX[n])

    def test_equal_background_comparison_rejects_different_off_predictions(self):
        y = np.tile([-1.0, 1.0], 32)
        batch = dict(
            y=y,
            rows=np.arange(len(y)),
            groups=np.repeat(np.arange(32), 2),
            setting_off=np.tile(0.4 * y[:, None], (1, len(grid.FAMILIES))),
            setting_on=np.tile(0.5 * y[:, None], (1, len(grid.KEYS))),
        )
        rows = report.common_main_comparison([batch, batch], 99)
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(abs(row["delta_r2_difference_pp"]) < 1e-10 for row in rows))
        batch["setting_off"][0, grid.FAMILY_INDEX["controlled_dose"]] += 0.01
        with self.assertRaisesRegex(ValueError, "off predictions"):
            report.common_main_comparison([batch, batch], 99)

    def test_matching_same_people_with_changed_fold_membership_fails(self):
        src = Generated()
        arrays, meta = cv.fold_predictions(src, 0, 0)
        arrays["rows"] = np.roll(arrays["rows"], 1)
        with self.assertRaisesRegex(ValueError, "fold identity"):
            report.assemble(src, {(0, 0): (arrays, meta)})

    def test_replay_stops_on_changed_archived_predictions(self):
        import v9_cv as old_cv
        import v9_grid as old

        n = 3
        rows = np.arange(n)
        a = dict(rows=rows, groups=rows.astype(str), y=rows.astype(float))
        new = dict(
            a, setting_off=np.zeros((n, len(grid.FAMILIES))), setting_on=np.zeros((n, len(grid.KEYS)))
        )
        archived = dict(
            a, setting_off=np.zeros((n, len(old.FAMILIES))), setting_on=np.zeros((n, len(old.KEYS)))
        )
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            self.assertEqual(runner.replay({}, p)["folds_checked"], 0)
            (p / "COMPLETE.json").write_text("{}")
            with (
                patch.object(runner.prior, "completed"),
                patch.object(runner.store, "read_fold", return_value=(archived, {})),
            ):
                anchors = {(0, 0): (new, {})}
                self.assertEqual(runner.replay(anchors, p)["paired_settings"], 24)
                archived["setting_on"][0, old_cv.INDEX[grid.REFERENCE]] = 1e-5
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    runner.replay(anchors, p)
