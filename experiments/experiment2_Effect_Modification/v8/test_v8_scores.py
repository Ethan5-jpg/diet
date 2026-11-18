"""Direct expansion of unequal-size resampled households is the score oracle."""
import unittest

import numpy as np

import run_v8
import v8_scores as scores


class ScoreTests(unittest.TestCase):
    def test_joint_bootstrap_matches_person_expansion_for_both_repeats(self):
        rng = np.random.default_rng(522)
        groups = np.repeat(np.arange(30), 1 + np.arange(30) % 4)
        y = rng.normal(size=len(groups))
        names = ["v7_reference", "peak_r0", "balanced"]
        batches = []
        for r in (0, 1):
            target = y + r * np.clip(y, -.7, .7)
            off = target[:, None] + rng.normal(0, .3, (len(groups), 3))
            on = target[:, None] + (off - target[:, None]) * [0.8, (0.45, 1.3)[r], .65]
            batches.append(dict(rows=np.arange(len(y)), groups=groups, y=target, off=off, on=on))
        def direct(indices):
            result = []
            for batch in batches:
                t = batch["y"][indices]
                sst = np.sum((t - t.mean())**2)
                a, b = [1 - np.sum((t[:, None] - batch[k][indices])**2, axis=0) / sst for k in ("off", "on")]
                result.append([a, b, b-a])
            return np.array(result)
        draws, seed = 199, 276
        group_rows = [np.flatnonzero(groups == g) for g in np.unique(groups)]
        sample_rng = np.random.default_rng(seed)
        expanded = []
        for _ in range(draws):
            sampled = sample_rng.integers(0, len(group_rows), size=len(group_rows))
            expanded.append(direct(np.concatenate([group_rows[j] for j in sampled])))
        truth, samples = direct(np.arange(len(y))), np.stack(expanded)
        result = scores.compare_routes(batches, names, names[1:], names[0], draws, seed)
        actual, boot = scores.paired_bootstrap(batches, draws, seed)
        np.testing.assert_allclose(actual, truth, atol=1e-14)
        np.testing.assert_allclose(boot, samples, atol=1e-14)
        margin = 100 * truth[:, 2] - scores.TARGET_PCT * truth[:, 1]
        margin_draw = 100 * samples[:, :, 2] - scores.TARGET_PCT * samples[:, :, 1]
        critical = max(0., np.quantile((margin_draw-margin).max(axis=(1, 2)), .95))
        self.assertAlmostEqual(result["simultaneous_critical_r2_pp"], critical)
        self.assertEqual(result["exploratory_ranking_by_minimum"][0], "balanced")
        self.assertFalse(result["routes"]["peak_r0"]["above_target_both_point"])
        for i, name in enumerate(names):
            row = result["routes"][name]
            self.assertAlmostEqual(row["minimum_ratio_pct"], min(100 * truth[:, 2, i] / truth[:, 1, i]))
            for r, score in enumerate(row["repeats"]):
                diff = 100 * (samples[:, r, 2, i]/samples[:, r, 1, i]
                              - samples[:, r, 2, 0]/samples[:, r, 1, 0])
                self.assertAlmostEqual(score["simultaneous_lower_margin_r2_pp"], margin[r, i]-critical)
                np.testing.assert_allclose(score["ratio_vs_v7_ci95_pp"], np.quantile(diff, [.025,.975]), atol=1e-11)
            self.assertEqual(row["above_target_both_adjusted"], bool(min(margin[:, i]) - critical > scores.EPS))

    def test_two_different_single_repeat_winners_do_not_clear_the_target(self):
        t = np.tile([-1., 1.], 50)
        b = .4
        names = ["old", "wins0", "wins1", "tie"]
        gains = [[5., 8., 1., scores.TARGET_PCT], [4., 2., 9., scores.TARGET_PCT]]
        batches = []
        for gain in gains:
            full = b / (1 - np.array(gain)/100)
            batches.append(dict(rows=np.arange(len(t)), groups=np.repeat(np.arange(50), 2), y=t,
                                off=t[:, None] + np.full((len(t), 4), np.sqrt(1-b)),
                                on=t[:, None] + np.sqrt(1-full)))
        result = scores.compare_routes(batches, names, names[1:], "old", 99)
        self.assertEqual(result["both_above_point"], [])
        self.assertEqual(result["both_above_adjusted"], [])
        self.assertEqual(result["exploratory_ranking_by_minimum"][0], "tie")
        batches[0]["on"][:, 1] = t + 20
        result = scores.compare_routes(batches, names, names[1:], "old", 99)
        invalid = result["routes"]["wins0"]
        self.assertIsNone(invalid["minimum_ratio_pct"])
        self.assertIsNone(invalid["above_target_both_adjusted"])
        self.assertLess(invalid["repeats"][0]["delta_r2"], 0)
        self.assertNotIn("wins0", result["exploratory_ranking_by_minimum"])
        batches[1]["groups"] = batches[1]["groups"][::-1]
        with self.assertRaisesRegex(ValueError, "aligned"):
            scores.compare_routes(batches, names, names[1:], "old", 99)

    def test_identical_arms_give_zero_and_a_weak_denominator_is_flagged(self):
        y = np.tile([-1., 1.], 40)
        predicted = np.column_stack((y*.5, y + np.linspace(0., 1.69, len(y))))
        batch = dict(rows=np.arange(len(y)), groups=np.repeat(np.arange(40), 2),
                     y=y, off=predicted, on=predicted.copy())
        out = scores.compare_routes([batch, batch], ["old", "weak"], ["weak"], "old", 199)
        self.assertEqual(out["routes"]["old"]["minimum_ratio_ci95_pct"], [0., 0.])
        weak = out["routes"]["weak"]
        self.assertEqual(weak["minimum_ratio_pct"], 0.)
        self.assertFalse(weak["stable_both"])
        self.assertIsNone(weak["above_target_both_adjusted"])
        self.assertEqual(out["both_above_point"], [])
