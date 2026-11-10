"""Closure, train-role isolation and frozen projection consistency."""

import copy
import pickle
import unittest

import numpy as np

from features import Role, ecology, gate_matrix, masked_scores
from support import fixture


class FeatureChecks(unittest.TestCase):
    def role(self, wide=False):
        data, cfg = fixture(wide)
        train = data.reference["outer"][0] != 0
        role = Role(
            data.split(train, ~train, cfg["contrast"]),
            data.raw,
            data.index,
            cfg["model_seed"],
            taxa=data.taxa,
            extra=data.extra,
        )
        return data, cfg, role

    def test_every_representation_replays_train_and_validation_after_serialization(self):
        _, _, role = self.role(wide=True)
        for method in (
            "clr",
            "hurdle",
            "log1p",
            "rank",
            "alr",
            "balance",
            "rclr",
            "eco",
            "ecohybrid",
            "compact",
        ):
            with self.subTest(method=method):
                design = role.design(
                    "balanced", method, dict(count=6, components=3, pairs=4), 2, joint=method != "clr"
                )
                state = pickle.loads(pickle.dumps(design["state"]))
                for train, key in ((True, "xt"), (False, "xv")):
                    np.testing.assert_allclose(role.replay(state, train)[0], design[key], atol=1e-10)

    def test_compositional_and_ecological_features_ignore_per_sample_scale(self):
        _, _, role = self.role(wide=True)
        other = Role(
            role.split,
            {k: v * np.linspace(0.1, 5, len(v))[:, None] for k, v in role.raw.items()},
            role.index,
            role.seed,
            taxa=role.taxa,
            extra=role.extra,
        )
        for method in ("hurdle", "log1p", "alr", "balance", "rclr", "eco", "ecohybrid", "compact"):
            a = role.design("balanced", method, dict(count=6, components=3, pairs=4), 2)
            np.testing.assert_allclose(role.replay(a["state"])[0], other.replay(a["state"])[0], atol=1e-8)

    def test_balance_weights_sum_to_zero_and_ecology_has_known_values(self):
        _, _, role = self.role()
        d = role.design("oral", "balance", dict(count=8, pairs=4), 2)
        np.testing.assert_allclose(d["state"]["blocks"][0]["weights"].sum(0), 0, atol=1e-14)
        values = ecology(np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]]))
        np.testing.assert_allclose(values[:, 1], [np.log(4), 0])
        np.testing.assert_allclose(values[:, 2], [0.75, 0])

    def test_masked_projection_agrees_with_independent_rowwise_ridge(self):
        rng = np.random.default_rng(77)
        x = rng.uniform(0.01, 1, (30, 8))
        x[rng.random(x.shape) < 0.4] = 0
        v = np.linalg.qr(rng.normal(size=(8, 3)))[0]
        expected = []
        for row in x:
            mask = row > 0
            values = np.log(row[mask])
            values -= values.mean()
            z = v[mask]
            expected.append(np.linalg.solve(z.T @ z + 0.1 * np.eye(3), z.T @ values))
        np.testing.assert_allclose(masked_scores(x, v, 0.1), expected, atol=1e-12)

    def test_validation_changes_do_not_change_learned_designs_or_imputation(self):
        _, _, role = self.role(wide=True)
        raw = copy.deepcopy(role.raw)
        for x in raw.values():
            x[role.split.rows] = x[role.split.rows, ::-1]
        extra = copy.deepcopy(role.extra)
        extra["values"][role.split.rows] = 999
        other = Role(role.split, raw, role.index, role.seed, taxa=role.taxa, extra=extra)
        for method in ("hurdle", "rank", "balance", "rclr", "eco", "compact"):
            case = dict(count=6, components=3, pairs=4)
            a, b = (
                role.design("balanced", method, case, 2, True),
                other.design("balanced", method, case, 2, True),
            )
            np.testing.assert_allclose(a["xt"], b["xt"], atol=1e-12)
        before = gate_matrix(role.index, role.raw, role.taxa, role.split.train_rows)
        after = gate_matrix(role.index, raw, role.taxa, role.split.train_rows)
        np.testing.assert_array_equal(before, after)

    def test_constant_covariate_host_contrast_remains_supported(self):
        data, cfg = fixture()
        data.index["sex_code"] = 0
        tr = data.reference["outer"][0] != 0
        role = Role(
            data.split(tr, ~tr, cfg["contrast"]),
            data.raw,
            data.index,
            cfg["model_seed"],
            taxa=data.taxa,
            extra=data.extra,
        )
        self.assertTrue(np.isfinite(role.host_at(role.split.delta)).all())
