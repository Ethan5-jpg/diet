"""Independent OOF, cluster and selection checks on generated observations."""

import copy
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch
import sys

import numpy as np

import v7_eval as ev


class RatioSelectionTests(unittest.TestCase):
    def test_different_denominators_and_weak_null_guard(self):
        # A weak null gives the highest percentage with smaller absolute gain.
        # Anchor: 0.60 -> 0.70; weak: 0.05 -> 0.11.
        y = np.tile([-1.0, 1.0], 150)
        g = np.repeat(np.arange(150), 2)

        def prediction(score):
            return (1 - np.sqrt(1 - score)) * y

        family = {"anchor": ["anchor@0.1"], "weak": ["weak@0.1"], "worse": ["bad"]}
        off = np.column_stack([prediction(0.60), prediction(0.05), prediction(0.61)])
        on = np.column_stack([prediction(0.70), prediction(0.11), prediction(0.55)])
        keys = ["anchor@0.1", "weak@0.1", "bad"]
        selections, _ = ev._select_inner(y, g, off, on, keys, family, "anchor", keys[0], keys[:1])
        self.assertEqual(selections[ev.PRIMARY], (keys[1], True))
        self.assertEqual(selections[ev.ORIGINAL], (keys[0], True))
        self.assertEqual(selections[ev.GUARDED], (keys[0], True))
        self.assertEqual(selections["worse"], ("bad", False))
        self.assertEqual(selections[ev.REFERENCE], (keys[0], True))
        null, _ = ev._select_inner(y, g, off, off, keys, family, "anchor", keys[0], keys[:1])
        self.assertEqual(null[ev.PRIMARY], (keys[0], False))
        self.assertEqual(null[ev.REFERENCE], (keys[0], True))

    def test_original_grid_is_frozen_including_penalties_and_tie_order(self):
        family = {"added_family": ["added_family@1"]}
        for name in reversed(ev.ORIGINAL_FAMILIES):
            family[name] = [f"{name}@{n}" for n in ("new", *reversed(ev.ORIGINAL_PENALTIES))]
        keys = ev._original_settings(family)
        self.assertEqual(len(keys), 175)
        self.assertEqual(keys[0], "gut_pc10_ridge@100")
        self.assertEqual(keys[-1], "blend_gut_ecology@0.01")
        self.assertFalse(any("new" in key or "added" in key for key in keys))
        family["gut_pc10_ridge"].remove("gut_pc10_ridge@0.1")
        with self.assertRaisesRegex(ValueError, "ORIGINAL35_COMPARISON_SETTING_MISSING"):
            ev._original_settings(family)

    def test_mutated_own_null_rejected(self):
        y = np.tile([-1.0, 1.0], 12)
        off = np.column_stack((0.5 * y, 0.6 * y))
        with self.assertRaisesRegex(ValueError, "FAMILY_NULL_CHANGED"):
            ev._select_inner(
                y,
                np.arange(len(y)),
                off,
                off,
                ["first", "second"],
                {"a": ["first", "second"]},
                "a",
                "first",
                ["first"],
            )


class BootstrapTests(unittest.TestCase):
    def test_against_person_expansion_and_paired_ratio_difference(self):
        rng = np.random.default_rng(19)
        groups = np.repeat(np.arange(24), 1 + np.arange(24) % 4)
        y = rng.normal(size=len(groups))
        off = np.column_stack((0.50 * y, 0.55 * y + 0.03 * rng.normal(size=len(y))))
        on = np.column_stack((0.73 * y + 0.04 * rng.normal(size=len(y)), 0.48 * y))
        seed, draws = 724, 207
        source = ev.paired_scores(
            y, groups, off, on, [ev.ORIGINAL, ev.PRIMARY], reference=ev.ORIGINAL, seed=seed, draws=draws
        )
        ratios, differences = [], []
        membership = [np.flatnonzero(groups == g) for g in range(24)]
        rng = np.random.default_rng(seed)
        for _ in range(draws):
            ids = np.concatenate([membership[g] for g in rng.integers(0, 24, size=24)])
            target = y[ids]
            denom = ((target - target.mean()) ** 2).sum()
            base_score = 1 - ((off[ids] - target[:, None]) ** 2).sum(axis=0) / denom
            full_score = 1 - ((on[ids] - target[:, None]) ** 2).sum(axis=0) / denom
            differences.append(full_score - base_score)
            ratios.append(100 * (full_score - base_score) / full_score)
        ratios = np.array(ratios)
        expected_diff = np.quantile(ratios[:, 1] - ratios[:, 0], [0.025, 0.975])
        b = source["routes"][ev.PRIMARY]
        self.assertLess(b["ratio_over_full_pct"], 0)
        np.testing.assert_allclose(
            b["delta_r2_ci95"], np.quantile(np.array(differences)[:, 1], [0.025, 0.975])
        )
        np.testing.assert_allclose(
            b["ratio_over_full_ci95_pct"], np.quantile(ratios[:, 1], [0.025, 0.975])
        )
        np.testing.assert_allclose(b["ratio_minus_reference_ci95_pp"], expected_diff)
        a = source["routes"][ev.ORIGINAL]
        self.assertTrue(a["beats_historical_point"])
        contrast = source["expanded_vs_original35"]
        self.assertAlmostEqual(contrast["ratio_difference_pp"], b["ratio_over_full_pct"] - a["ratio_over_full_pct"])
        self.assertAlmostEqual(contrast["delta_r2_difference_pp"], 100 * (b["delta_r2"] - a["delta_r2"]))
        np.testing.assert_allclose(contrast["ratio_difference_ci95_pp"], expected_diff)
        diffs = np.asarray(differences)
        expected_delta_ci = np.quantile(100 * (diffs[:, 1] - diffs[:, 0]), [0.025, 0.975])
        np.testing.assert_allclose(contrast["delta_r2_difference_ci95_pp"], expected_delta_ci)

    def test_unusable_ratio_does_not_suppress_paired_delta(self):
        y = np.tile([-1.0, 1.0], 15)
        groups = np.repeat(np.arange(15), 2)
        off = np.tile(0.25 * y[:, None], 2)
        on = np.column_stack((np.zeros(len(y)), 0.75 * y))
        result = ev.paired_scores(y, groups, off, on, [ev.ORIGINAL, ev.PRIMARY],
                                  reference=ev.ORIGINAL, draws=119)
        comparison = result["expanded_vs_original35"]
        self.assertIsNone(comparison["ratio_difference_pp"])
        self.assertIsNone(comparison["ratio_difference_ci95_pp"])
        self.assertGreater(comparison["delta_r2_difference_pp"], 0)
        self.assertEqual(len(comparison["delta_r2_difference_ci95_pp"]), 2)

    def test_exact_benchmark_replay_is_a_tie(self):
        from v7_reference import HISTORICAL_OFF, HISTORICAL_ON

        y = np.tile([-1.0, 1.0], 20)
        groups = np.repeat(np.arange(20), 2)
        off, on = [(1 - np.sqrt(1 - r)) * y[:, None] for r in (HISTORICAL_OFF, HISTORICAL_ON)]
        row = ev.paired_scores(y, groups, off, on, ["anchor"], reference="anchor", draws=99)[
            "routes"
        ]["anchor"]
        self.assertAlmostEqual(row["ratio_over_full_pct"], ev.BENCHMARK, places=10)
        self.assertFalse(row["beats_historical_point"])
        self.assertFalse(row["beats_historical_ci"])

    def test_nonpositive_or_uncertain_denominators(self):
        y = np.tile([-1.0, 1.0], 15)
        groups = np.arange(len(y))
        # Reference has zero full R2; its ratio and paired difference are undefined.
        off = np.tile(0.25 * y[:, None], 2)
        on = np.column_stack((np.zeros(len(y)), 0.75 * y))
        result = ev.paired_scores(y, groups, off, on, ["a", "b"], reference="a", draws=199)
        self.assertIsNone(result["routes"]["a"]["ratio_over_full_pct"])
        self.assertLess(result["routes"]["a"]["delta_r2"], 0)
        self.assertIsNone(result["routes"]["b"]["ratio_minus_reference_ci95_pp"])
        # Positive but weak FULL R2: zero-centred bootstrap draws cross zero.
        on[:, 0] = 0.01 * y
        result = ev.paired_scores(y, groups, off, on, ["a", "b"], reference="a", draws=199)
        self.assertIsNotNone(result["routes"]["a"]["ratio_over_full_pct"])
        self.assertIsNone(result["routes"]["a"]["ratio_over_full_ci95_pct"])


class NestedEvaluationTests(unittest.TestCase):
    @staticmethod
    def fixture():
        y = np.tile([-1.0, 1.0], 30)
        g = np.repeat(np.arange(30), 2)
        outer = np.stack([g % 5, (g + 1) % 5])

        def make(tr, va):
            return SimpleNamespace(
                train_rows=np.flatnonzero(tr), rows=np.flatnonzero(va), yt=y[tr], yv=y[va]
            )

        def roles(r, f):
            va = outer[r] == f
            tr = ~va
            return [make(tr & (g % 3 != j), tr & (g % 3 == j)) for j in range(3)], make(tr, va)

        src = SimpleNamespace(
            groups=np.array([f"hh{n}" for n in g], dtype=object),
            roles=roles,
            reference={"y": np.stack([y, y]), "outer": outer},
        )
        mod = ModuleType("v7_models")
        mod.families = lambda: {"anchor": ["anchor@0.1"], "weak": ["b1", "b2"]}
        mod.ANCHOR = "anchor"

        def fit(source, split, *, view_audit=None):
            if view_audit is not None:
                view_audit["training_people"] = len(split.train_rows)
            assert not set(source.groups[split.train_rows]).intersection(source.groups[split.rows])
            raw = np.where(split.rows % 2, 1.0, -1.0)
            return {
                "anchor@0.1": (0.55 * raw, 0.8 * raw),
                "b1": (0.1 * raw, 0.5 * raw),
                "b2": (0.1 * raw, 0.31 * raw),
            }

        mod.evaluate_split = fit
        return src, mod

    def test_pca_rows_include_unscored_people_but_exclude_held_households(self):
        source, _ = self.fixture()
        inner, outer = source.roles(0, 0)
        assignment = source.reference["outer"][0]
        for role in (outer, *inner):
            role.pca_training_rows = role.train_rows.copy()
        # Remove an entire household from clinical scoring, retain its original fold.
        absent_household = source.groups[outer.rows[0]]
        hidden = np.flatnonzero(source.groups == absent_household)
        outer.rows = np.setdiff1d(outer.rows, hidden)
        ev._check_roles(source.groups, assignment, 0, inner, outer)
        original = inner[0].pca_training_rows.copy()
        inner[0].pca_training_rows = np.r_[original, hidden]
        with self.assertRaisesRegex(ValueError, "PCA_TRAINING_ROLE_LEAK"):
            ev._check_roles(source.groups, assignment, 0, inner, outer)
        # A clinically unscored member of an inner validation household is also held.
        other = inner[0].rows[0]
        inner[0].rows = inner[0].rows[1:]
        inner[0].pca_training_rows = np.r_[original, other]
        with self.assertRaisesRegex(ValueError, "PCA_TRAINING_ROLE_LEAK|PCA_HOUSEHOLD"):
            ev._check_roles(source.groups, assignment, 0, inner, outer)

    def test_duplicate_validation_row_rejected_before_scoring(self):
        source, _ = self.fixture()
        inner, outer = source.roles(0, 0)
        inner[1].rows = np.r_[inner[1].rows, inner[1].rows[0]]
        with self.assertRaisesRegex(ValueError, "DUPLICATE_ROLE_ROWS"):
            ev._check_roles(source.groups, source.reference["outer"][0], 0, inner, outer)

    def test_outer_targets_and_exact_reference(self):
        source, model = self.fixture()
        with patch.dict(sys.modules, v7_models=model), patch.object(
                ev, "_original_settings", return_value=["anchor@0.1"]):
            folds = {(r, f): ev.run_fold(source, r, f) for r in range(2) for f in range(5)}
            result = ev.summarize(source, folds, draws=119)
            anchor = result["repeats"][0]["routes"][ev.REFERENCE]
            self.assertEqual(anchor["active_folds"], 5)
            self.assertGreater(anchor["ratio_over_full_pct"], 0)
            o = source.roles

            def perturbed(r, f):
                inn, out = o(r, f)
                out.yv = out.yv + 123
                return inn, out

            source.roles = perturbed
            altered, meta = ev.run_fold(source, 0, 1)
            original, oldmeta = folds[0, 1]
            self.assertEqual(original["groups"].dtype.kind, "U")
            for name in ("off", "on"):
                np.testing.assert_array_equal(original[name], altered[name])
            self.assertEqual(meta["choices"], oldmeta["choices"])
            self.assertEqual(meta["modifier_view_audit"], {"training_people": 48})
            self.assertEqual(meta["choices"][ev.ORIGINAL]["setting"], "anchor@0.1")
            self.assertIn("expanded_vs_original35", result["repeats"][0])
            self.assertIn("threshold_screen", result["repeats"][0])
            fake = copy.copy(folds)
            fake[0, 1] = altered, meta
            with self.assertRaisesRegex(ValueError, "OUTER_FOLD_ALIGNMENT_CHANGED"):
                ev.summarize(source, fake, draws=119)


if __name__ == "__main__":
    unittest.main()
