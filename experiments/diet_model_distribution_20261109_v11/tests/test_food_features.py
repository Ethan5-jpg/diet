"""Distribution identities, person weights, shrinkage and validation isolation."""

import copy
import pickle
import unittest

import numpy as np

from context import aggregate
from food_features import DietEncoder
from registry import CONTEXTS
from support import fixture, tables


class FoodChecks(unittest.TestCase):
    def test_all_food_encoders_replay_and_ignore_validation_days_and_values(self):
        data, cfg = fixture()
        extra, index = data.extra, data.index
        train, valid = np.arange(240), np.arange(240, len(index))
        changed = copy.deepcopy(extra)
        for a in changed["blocks"].values():
            a[valid] = 999
        changed["days"][np.isin(changed["owners"], valid)] = 999
        changed["joint_variance"][valid] = 999
        changed["covariance"][valid] = 999
        changed["alcohol"][valid] = "validation_only"
        other = index.copy()
        other.loc[valid, ["UPF_lower_equalday_fraction", "UPF_upper_equalday_fraction"]] = 0.9
        for name in CONTEXTS:
            with self.subTest(profile=name):
                a = DietEncoder(name, cfg["diet_parameters"]).fit(extra, index, train)
                b = DietEncoder(name, cfg["diet_parameters"]).fit(changed, other, train)
                np.testing.assert_array_equal(
                    a.transform(extra, index, train), b.transform(changed, other, train)
                )
                np.testing.assert_array_equal(a.medians, b.medians)
                self.assertNotIn("validation_only", a.categories)
                for rows in (train, valid):
                    value = a.transform(extra, index, rows)
                    self.assertTrue(np.isfinite(value).all())
                    np.testing.assert_array_equal(
                        value, pickle.loads(pickle.dumps(a)).transform(extra, index, rows)
                    )

    def test_bag_means_equal_explicit_transform_averages_and_projection_quantiles(self):
        data, cfg = fixture()
        extra, index, rows = data.extra, data.index, np.arange(10)
        for name in ("rff", "prototype", "sliced"):
            enc = DietEncoder(name, cfg["diet_parameters"]).fit(extra, index, rows)
            a = enc.bag(extra, rows)
            for i, row in enumerate(rows):
                z = enc.day_scaler.transform(extra["days"][extra["owners"] == row])
                if name == "rff":
                    expected = enc.basis.transform(z).mean(0)
                elif name == "prototype":
                    from scipy.special import softmax

                    expected = softmax(-enc.parameters["gamma"] * enc.basis.transform(z) ** 2, axis=1).mean(0)
                    self.assertAlmostEqual(a[i].sum(), 1)
                else:
                    expected = np.quantile(z @ enc.basis, enc.parameters["quantiles"], axis=0).ravel()
                np.testing.assert_allclose(a[i], expected, atol=1e-12)

    def test_person_weighted_basis_does_not_overweight_longer_logs(self):
        data, cfg = fixture()
        extra, index, rows = copy.deepcopy(data.extra), data.index, np.arange(4)
        # Repeat each day of one person, preserving that person's empirical distribution.
        select = extra["owners"] == 0
        extra["days"] = np.concatenate([extra["days"], extra["days"][select], extra["days"][select]])
        extra["owners"] = np.concatenate([extra["owners"], extra["owners"][select], extra["owners"][select]])
        extra["joint_count"][0] *= 3
        for name in ("rff", "prototype", "sliced"):
            enc = DietEncoder(name, cfg["diet_parameters"]).fit(extra, index, rows)
            means = [extra["days"][extra["owners"] == row].mean(0) for row in rows]
            np.testing.assert_allclose(enc.day_scaler.mean_, np.mean(means, axis=0), atol=1e-12)

    def test_shrinkage_formula_is_training_only_and_retains_sparse_people(self):
        data, _ = fixture()
        extra, index, rows = copy.deepcopy(data.extra), data.index, np.arange(4)
        extra["blocks"]["joint"][:4, :3] = np.array([1, 3, 5, 7])[:, None]
        extra["joint_variance"][:4] = 6.0
        extra["joint_count"][:4] = 6
        enc = DietEncoder("shrink_global").fit(extra, index, rows)
        expected_tau = np.var([1, 3, 5, 7], ddof=1) - 1
        np.testing.assert_allclose(enc.tau, expected_tau)
        expected = 4 + expected_tau / (expected_tau + 1) * (np.array([1, 3, 5, 7]) - 4)
        np.testing.assert_allclose(enc.numerical(extra, index, rows)[:, 0], expected)
        extra["joint_count"][5] = 2
        extra["blocks"]["joint"][5, :3] = np.nan
        for name in ("shrink_global", "shrink_local", "shrink_uncertainty", "quality"):
            fit = DietEncoder(name).fit(extra, index, rows)
            self.assertEqual(fit.transform(extra, index, np.array([5])).shape[0], 1)
            self.assertTrue(np.isfinite(fit.transform(extra, index, np.array([5]))).all())

    def test_covariance_distinguishes_equal_means_with_opposite_daily_pairing(self):
        data, _ = fixture()
        index = data.index.iloc[:2].copy()
        food, nutrient, life = tables(index)
        c = np.tile([90, 100, 110], 2)
        for person in range(2):
            pos = np.arange(person * 6, (person + 1) * 6)
            energy = food.loc[pos, "energy_kcal"].to_numpy()
            f = np.tile([30, 40, 50] if person == 0 else [50, 40, 30], 2)
            nutrient.loc[pos, "carbohydrate_g_complete_sum"] = c * energy / 1000
            nutrient.loc[pos, "lipid_g_complete_sum"] = f * energy / 1000
        extra = aggregate(food, nutrient, life, index)
        np.testing.assert_allclose(extra["blocks"]["joint"][:, :2], [[100, 40], [100, 40]])
        np.testing.assert_allclose(extra["covariance"][:, 3], [1, -1], atol=1e-12)
        covariance = np.cov(extra["days"][extra["owners"] == 0, :3], rowvar=False, ddof=1)
        np.testing.assert_allclose(extra["joint_variance"][0], np.diag(covariance))

    def test_empty_or_short_bags_remain_missing_without_transform_failures(self):
        data, cfg = fixture()
        extra, index = copy.deepcopy(data.extra), data.index
        extra["joint_count"][:3] = 2
        extra["days"] = extra["days"][extra["owners"] >= 3]
        extra["owners"] = extra["owners"][extra["owners"] >= 3]
        for name in ("rff", "prototype", "sliced"):
            enc = DietEncoder(name, cfg["diet_parameters"]).fit(extra, index, np.arange(3, 10))
            self.assertTrue(np.isnan(enc.bag(extra, np.arange(3))).all())
            self.assertTrue(np.isfinite(enc.transform(extra, index, np.arange(3))).all())
            empty = DietEncoder(name, cfg["diet_parameters"]).fit(extra, index, np.arange(3))
            self.assertTrue(np.isfinite(empty.transform(extra, index, np.arange(4))).all())

    def test_invalid_upf_bounds_are_rejected_and_zeros_are_not_missing(self):
        data, _ = fixture()
        enc = DietEncoder("joint_upf").fit(data.extra, data.index, np.arange(10))
        data.index.loc[0, ["UPF_lower_equalday_fraction", "UPF_upper_equalday_fraction"]] = 0
        values = enc.numerical(data.extra, data.index, np.arange(2))
        np.testing.assert_array_equal(values[0, -2:], [0, 0])
        data.index.loc[0, "UPF_lower_equalday_fraction"] = 0.9
        with self.assertRaises(ValueError):
            enc.transform(data.extra, data.index, np.arange(2))
