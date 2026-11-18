"""Tests of adaptive selection, household pairing, ratio thresholds and resume."""

import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

import run_v9 as run
import run_v6 as store
import run_v8 as prior
import v8_models
import v8_scores
import v9_ablation
import v9_cv as cv
import v9_fixed
import v9_grid as grid
import v9_report as report
from integration_fixture import Generated


class Validation(unittest.TestCase):
    def setUp(self):
        limit = threadpool_limits(1)
        self.addCleanup(limit.restore_original_limits)

    def test_inner_minimum_uses_same_family_and_can_pick_non_gate(self):
        y = np.tile([-1.0, 1.0], 30)
        rows = np.arange(len(y))
        off = np.broadcast_to((y + np.sqrt(0.7))[:, None], (len(y), len(grid.KEYS)))
        a = grid.ANCHOR + "@0.1"
        b = "standalone_dose_curve_fermentation@0.3"
        c = "partition_gut2@1"

        def arrays(values):
            ratio = np.array([values.get(k, -1.0) for k in grid.KEYS])
            full = 0.3 / (1 - ratio / 100)
            return y, rows // 2, off, y[:, None] + np.sqrt(1.0 - full)

        p, q = arrays({a: 12.0, b: 7.7, c: 4.1}), arrays({a: 1.0, b: 7.3, c: 10.1})
        selected, meta = cv.select((rows, p), (rows[::-1], tuple(x[::-1] for x in q)))
        self.assertEqual(selected["selector_maximin"].describe()["settings"], [b])
        self.assertIn(a, selected[cv.PRIMARY].describe()["settings"])
        self.assertEqual(meta["positive_in_both"], 3)
        neg = arrays({})
        chosen, _ = cv.select((rows, neg), (rows, neg))
        self.assertFalse(chosen[cv.PRIMARY].describe()["enabled"])
        self.assertFalse(chosen["stack_unrestricted"].describe()["enabled"])
        self.assertTrue(chosen[cv.FIXED].describe()["enabled"])
        for rule in cv.WORKFLOWS:
            x, z = chosen[rule].predict(*neg[2:])
            np.testing.assert_array_equal(x, z)

    def test_weights_are_identical_and_mix_predictions_before_scoring(self):
        a = np.array([[0.1, 0.7], [0.4, 0.3]])
        b = np.array([[0.5, 0.6], [0.9, 0.5]])
        ch = cv.Choice((1, 0), (0.75, 0.25), (False, True))
        off, on = ch.predict(a, b)
        np.testing.assert_allclose(off, 0.75 * a[:, 1] + 0.25 * a[:, 0])
        np.testing.assert_allclose(on, 0.75 * a[:, 1] + 0.25 * b[:, 0])
        self.assertEqual(ch.describe()["active"], [False, True])

    def test_training_choices_ignore_outer_targets(self):
        src = Generated(n=180, seed=470)
        y = src.y.copy()
        y[src.reference["outer"][1] == 3] -= 1000.0
        a, ma = cv.fold_predictions(src, 1, 3)
        b, mb = cv.fold_predictions(src.with_y(y), 1, 3)
        self.assertEqual(ma, mb)
        self.assertFalse(np.array_equal(a["y"], b["y"]))
        for arm in ("off", "on"):
            np.testing.assert_array_equal(a[arm], b[arm])
        for arm in ("setting_off", "setting_on"):
            np.testing.assert_array_equal(a[arm], b[arm])

    def test_stored_setting_mapping_and_shared_off_are_checked(self):
        src = Generated(n=180, seed=814)
        a, m = cv.fold_predictions(src, 1, 2)
        cv.verify_fold_choices(a, m)
        bad_on = a["setting_on"].copy()
        bad_on[0, cv.INDEX[grid.FIXED_PC10]] += 0.05
        with self.assertRaisesRegex(ValueError, "differs from fixed settings"):
            cv.verify_fold_choices(dict(a, setting_on=bad_on), m)
        expanded_off = a["setting_off"][:, cv.OFF_INDEX].copy()
        np.testing.assert_array_equal(cv.common_family_off(expanded_off), a["setting_off"])
        expanded_off[0, cv.INDICES[grid.ANCHOR][-1]] -= 0.001
        with self.assertRaisesRegex(ValueError, "within model family"):
            cv.common_family_off(expanded_off)

    def test_exact_additive_outcome_produces_no_held_out_interaction_gain(self):
        src = Generated(n=180, seed=6611)
        additive = src.with_y(90.0 + 0.3 * src.fiber + 0.6 * src.age)
        a, meta = cv.fold_predictions(additive, 0, 0)
        self.assertTrue(all(not meta["choices"][n]["enabled"] for n in grid.FAMILIES))
        np.testing.assert_allclose(
            a["off"], np.broadcast_to(a["y"][:, None], a["off"].shape), atol=2e-9, rtol=0
        )
        np.testing.assert_allclose(a["setting_on"], a["setting_off"][:, cv.OFF_INDEX], atol=2e-9, rtol=0)

    def test_factorial_order_average_recovers_known_interaction_allocation(self):
        bits = np.array([[(mask >> j) & 1 for j in range(3)] for mask in range(8)]).T
        x, y, z = bits
        cells = 3.0 + 1.0 * x + 2.0 * y - 1.5 * z + 1.2 * x * y - 0.6 * x * y * z
        expected = np.array([1.0 + 0.6 - 0.2, 2.0 + 0.6 - 0.2, -1.5 - 0.2])
        got = v9_ablation.order_average(np.vstack((cells, -cells)), 3)
        np.testing.assert_allclose(got, [expected, -expected], atol=1e-13)

    def test_identical_predictions_have_zero_block_contributions_and_reject_a_changed_null(self):
        n = 80
        rows = np.arange(n)
        y = (-1.0) ** rows
        off = np.tile((y + 0.2)[:, None], (1, len(grid.FAMILIES)))
        on = np.tile((y + 0.2)[:, None], (1, len(grid.KEYS)))
        pane = dict(rows=rows, groups=rows // 2, y=y, setting_on=on, setting_off=off)
        summary = v9_ablation.summarize([pane, pane], 99, 197)
        for r in summary["repeats"]:
            self.assertAlmostEqual(r["full_minus_core_r2_pp"], 0.0)
            for block in r["blocks"].values():
                np.testing.assert_allclose(
                    block["order_average_simultaneous_ci95_pp"], [0.0, 0.0], atol=1e-11
                )
                self.assertAlmostEqual(block["order_average_r2_pp"], 0.0)
                self.assertAlmostEqual(block["full_minus_without_r2_pp"], 0.0)
        off[0, grid.FAMILIES.index(grid.ANCHOR)] += 0.01
        with self.assertRaisesRegex(ValueError, "nonidentical off"):
            v9_ablation.summarize([pane, pane], 99, 197)

    def test_household_threshold_requires_same_route_and_parent_ci_is_paired(self):
        rows = np.arange(80)
        y = (-1.0) ** rows * (1.0 + 0.05 * np.sin(rows // 2))
        sst = np.sum((y - y.mean()) ** 2)
        groups = rows // 2
        names = ["ref", "a", "b", "c"]
        cases = [[5.0, 8.0, 4.0, 7.0], [5.0, 4.0, 8.0, 7.0]]
        batches = []
        for r, vals in enumerate(cases):
            direction = np.sin(rows + r + 0.3)
            direction /= np.linalg.norm(direction)
            base = np.repeat(0.4, len(names))
            full = base / (1.0 - np.array(vals) / 100)
            off = y[:, None] + direction[:, None] * np.sqrt((1 - base) * sst)
            on = y[:, None] + direction[:, None] * np.sqrt((1 - full) * sst)
            batches.append(dict(rows=rows, y=y, groups=groups, off=off, on=on))
        got = v8_scores.compare_routes(batches, names, names, "ref", 199, 915)
        self.assertEqual(got["both_above_point"], ["c"])
        comps = report.parent_comparisons(batches, names, {"a": ("ref",)}, 199, 915)
        for r, delta in enumerate((3.0, -1.0)):
            self.assertAlmostEqual(comps[r]["ratio_difference_pp"], delta, places=10)
            self.assertAlmostEqual(
                comps[r]["delta_r2_difference_pp"],
                100 * (0.4 / (1 - cases[r][1] / 100) - 0.4 / 0.95),
                places=10,
            )
        # Independent naive household sampling, with the exact same random draws.
        rng = np.random.default_rng(915)
        draws = [[], []]
        for _ in range(199):
            ids = rng.integers(0, 40, 40)
            selected = np.column_stack((2 * ids, 2 * ids + 1)).ravel()
            for r, arr in enumerate(batches):
                yy = y[selected]
                ss = np.sum((yy - yy.mean()) ** 2)
                R = [
                    1.0 - np.sum((yy[:, None] - arr[arm][selected]) ** 2, axis=0) / ss
                    for arm in ("off", "on")
                ]
                ratio = 100 * (R[1] - R[0]) / R[1]
                draws[r].append(ratio[1] - ratio[0])
        for r in range(2):
            np.testing.assert_allclose(
                comps[r]["ratio_difference_ci95_pp"], np.quantile(draws[r], [0.025, 0.975]), atol=1e-9
            )
        bad = [batches[0], dict(batches[1], groups=groups[::-1])]
        with self.assertRaisesRegex(ValueError, "aligned"):
            v8_scores.compare_routes(bad, names, names, "ref", 99, 1)
        value, _, status = v8_scores.ratios(np.array([0.01, -0.01, -0.02]), np.empty((0, 3)), False)
        self.assertIsNone(value)
        self.assertIn("nonpositive", status)

    def test_fixed_settings_replay_joint_naive_threshold_and_pair_with_their_own_nulls(self):
        rng = np.random.default_rng(491)
        grouped = np.repeat(np.arange(50), 2)
        rows = np.arange(len(grouped))
        y = (-1.0) ** rows + 0.02 * rng.normal(size=len(rows))
        batches, combined = [], []
        names = ["adaptive", "fixed_a", "fixed_b", "fixed_c"]
        for _ in range(2):
            off = y[:, None] + rng.normal(size=(len(rows), 2)) * 0.22
            on = y[:, None] + rng.normal(size=(len(rows), 3)) * 0.19
            named = y[:, None] + rng.normal(size=(len(rows), 1)) * 0.25
            b = dict(
                rows=rows, groups=grouped, y=y, off=off[:, :1], on=named, setting_off=off, setting_on=on
            )
            batches.append(b)
            combined.append(
                dict(b, off=np.column_stack((b["off"], off[:, [1, 0, 1]])), on=np.column_stack((named, on)))
            )
        fixed, critical, count = v9_fixed.evaluate(batches, names[1:], [1, 0, 1], 6.46, 199, 718, width=2)
        oracle = v8_scores.compare_routes(combined, names, names, "adaptive", 199, 718)
        self.assertEqual(count, oracle["valid_draws"])
        self.assertAlmostEqual(critical, oracle["simultaneous_critical_r2_pp"], places=11)
        for name in names[1:]:
            self.assertEqual(
                fixed[name]["above_target_both_adjusted"],
                oracle["routes"][name]["above_target_both_adjusted"],
            )
            np.testing.assert_allclose(
                fixed[name]["minimum_ratio_ci95_pct"],
                oracle["routes"][name]["minimum_ratio_ci95_pct"],
                atol=1e-10,
            )
            for s, t in zip(fixed[name]["repeats"], oracle["routes"][name]["repeats"], strict=True):
                for field in (
                    "ratio_pct",
                    "delta_r2_pp",
                    "ratio_ci95_pct",
                    "delta_r2_ci95_pp",
                    "simultaneous_lower_margin_r2_pp",
                ):
                    np.testing.assert_allclose(s[field], t[field], atol=1e-10)

    def test_interrupted_batch_replays_all_v8_controls_and_reports_without_refitting(self):
        source = Generated(n=180, seed=56713)
        rel = {"kind": "generated"}
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            target, old = Path(td) / "v9", Path(td) / "v8"
            prior.execute(source, old, store.read_json(run.V8 / "upload_manifest.json"), 1, draws=99)
            fit = cv.fold_predictions

            def cut(src, r, f):
                if (r, f) == (0, 1):
                    raise InterruptedError("interrupted")
                return fit(src, r, f)

            with patch.object(cv, "fold_predictions", side_effect=cut), self.assertRaises(InterruptedError):
                run.execute(source, target, rel, 1, draws=99, previous=old)
            self.assertTrue((target / "r0_f0/receipt.json").is_file())
            self.assertFalse((target / "COMPLETE.json").exists())
            self.assertFalse((target / "RUNNING.lock").exists())
            with patch.object(cv, "fold_predictions", wraps=fit) as resume:
                summary = run.execute(source, target, rel, 1, draws=99, previous=old)
            self.assertEqual(resume.call_count, 9)
            self.assertEqual(summary["v8_replay"]["families_checked"], len(v8_models.FAMILIES))
            self.assertEqual(summary["v8_replay"]["folds_checked"], 10)
            self.assertEqual(summary["v8_replay"]["status"], "V8_predictions_match")
            self.assertEqual(set(summary["routes"]), set(cv.NAMES))
            self.assertEqual(set(summary["fixed_settings"]), set(grid.KEYS))
            self.assertFalse(summary["reserved_outcomes_read"])
            self.assertGreater(summary["routes"][cv.PRIMARY]["minimum_ratio_pct"], 0.0)
            self.assertTrue(summary["common_main_ablation"]["off_equal_all_rows"])
            for record in summary["common_main_ablation"]["repeats"]:
                self.assertAlmostEqual(
                    sum(b["order_average_r2_pp"] for b in record["blocks"].values()),
                    record["full_minus_core_r2_pp"],
                    places=10,
                )
            for filename, count in (
                ("all_routes.csv", 2 * len(cv.NAMES)),
                ("fixed_settings.csv", 2 * len(grid.KEYS)),
                ("parent_comparisons.csv", 2 * sum(map(len, grid.PARENTS.values()))),
            ):
                with (target / filename).open() as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), count)
            with patch.object(cv, "fold_predictions", side_effect=AssertionError("refit")):
                self.assertEqual(run.execute(source, target, rel, 1, draws=99, previous=old), summary)
                report.compact(run.completed(target, rel))
            (target / "r1_f4/metadata.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                run.completed(target, rel)
