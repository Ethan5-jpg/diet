"""Frozen transforms, numerical identities, known signal and every default case."""

import copy
import pickle
import unittest

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import rbf_kernel

from estimators import AdditiveKernel, BlockRidge
from features import ContextEncoder, Role
from fusion import blend_weights
from models import Fitted, atomic, interaction
from registry import INTERACTIONS, specifications
from support import fixture, reduced


def role_for(data, cfg):
    valid = data.reference["outer"][0] == 0
    return Role(
        data.split(~valid, valid, cfg["contrast"]),
        data.raw,
        data.index,
        cfg["model_seed"],
        taxa=data.taxa,
        extra=data.extra,
    )


class ZeroModel:
    def predict(self, x):
        return np.zeros(len(x))


class ModelChecks(unittest.TestCase):
    def test_every_default_candidate_fits_predicts_and_serializes_on_wide_sparse_profiles(self):
        data, cfg = fixture(wide=True)
        role, bases, count = role_for(data, cfg), {}, 0
        for spec in specifications(cfg):
            for j, case in enumerate(spec["grid"]):
                with self.subTest(model=spec["name"], candidate=j):
                    if spec["name"] in INTERACTIONS:
                        fit, note = interaction(role, spec, case, bases[spec["null"]], cfg)
                    else:
                        fit, note = atomic(role, spec, case, cfg, 1)
                    for training in (True, False):
                        p = fit.predict(role, training)
                        expected = len(role.split.yt) if training else len(role.split.yv)
                        self.assertEqual(p.shape, (expected,))
                        self.assertTrue(np.isfinite(p).all())
                        np.testing.assert_allclose(
                            pickle.loads(pickle.dumps(fit)).predict(role, training), p, atol=1e-10
                        )
                    if j == 0 and spec["name"] in INTERACTIONS.values():
                        bases[spec["name"]] = fit
                    self.assertIsInstance(note, dict)
                    count += 1
                    if count % 25 == 0 or count == 181:
                        print("CHECK_CANDIDATES %d/181" % count, flush=True)
            role.cache = {
                k: v for k, v in role.cache.items() if k[0] not in ("design", "block", "correction")
            }
        self.assertEqual(count, 181)

    def test_holdout_changes_cannot_fit_context_categories_screening_or_pca(self):
        data, cfg = fixture()
        cfg = reduced(cfg)
        a = role_for(data, cfg)
        changed = copy.copy(data)
        changed.extra, changed.raw = copy.deepcopy(data.extra), copy.deepcopy(data.raw)
        valid = data.reference["outer"][0] == 0
        for value in changed.extra["blocks"].values():
            value[valid] = 999
        changed.extra["alcohol"][valid] = "unseen_holdout_only"
        for value in changed.raw.values():
            value[valid] = value[valid, ::-1]
        b = role_for(changed, cfg)
        for method in ("clr", "pc", "hurdle", "ecohybrid"):
            case = dict(count=6, components=3)
            first = a.design("balanced", method, case, 1, "full")
            second = b.design("balanced", method, case, 1, "full")
            np.testing.assert_array_equal(first["xt"], second["xt"])
            enc = first["state"]["context"]
            self.assertNotIn("unseen_holdout_only", enc.categories)
            np.testing.assert_array_equal(enc.medians, second["state"]["context"].medians)

    def test_missing_unknown_categories_and_numeric_zero_are_distinct(self):
        extra = dict(
            blocks=dict(life=np.array([[0.0, 1, 7], [2, 3, 8], [np.nan, 2, 6], [3, 2, 7]])),
            alcohol=np.array(["never", "weekly", None, "new_label"], object),
        )
        encoder = ContextEncoder("life").fit(extra, np.array([0, 1]))
        raw = encoder.raw(extra, np.arange(4))
        self.assertEqual(raw[0, 0], 0)
        self.assertEqual(raw[2, 0], 1)
        self.assertEqual(raw[2, 3], 1)
        self.assertEqual(raw[2, -2], 1)
        self.assertEqual(raw[3, -1], 1)
        self.assertEqual(raw[0, -2:].sum(), 0)

    def test_block_ridge_matches_independent_weighted_normal_equations(self):
        rng = np.random.default_rng(11)
        x, y = rng.normal(size=(90, 7)), rng.normal(size=90)
        cfg = dict(penalty=0.03, micro_weight=0.25, gut_weight=2)
        fit = BlockRidge([2, 1, 2, 2], ["host", "context", "oral", "gut"], cfg).fit(x, y)
        z = (x - x.mean(0)) / x.std(0)
        weights = np.sqrt([1 / 2, 1 / 2, 1, 0.25 / 2, 0.25 / 2, 0.5 / 2, 0.5 / 2])
        a = z * weights
        beta = np.linalg.solve(a.T @ a + 90 * 0.03 * np.eye(7), a.T @ (y - y.mean()))
        np.testing.assert_allclose(fit.predict(x), y.mean() + a @ beta, rtol=1e-11, atol=1e-11)

    def test_additive_kernel_is_psd_and_matches_independent_dual_solution(self):
        rng = np.random.default_rng(12)
        x, v, y = rng.normal(size=(50, 7)), rng.normal(size=(10, 7)), rng.normal(size=50)
        fit = AdditiveKernel([2, 1, 2, 2], dict(penalty=0.02, bandwidth=0.5, micro_weight=0.25)).fit(x, y)
        a, b = (x - x.mean(0)) / x.std(0), (v - x.mean(0)) / x.std(0)

        def kernel(u, w):
            return (
                rbf_kernel(u[:, :3], w[:, :3], gamma=0.5 / 3)
                + 0.25 * (u[:, 3:5] @ w[:, 3:5].T + u[:, 5:] @ w[:, 5:].T) / 2
            )

        gram = kernel(a, a)
        self.assertGreater(np.linalg.eigvalsh(gram).min(), -1e-10)
        expected = kernel(b, a) @ np.linalg.solve(gram + np.eye(50), y)
        np.testing.assert_allclose(fit.predict(v), expected, atol=1e-11)

    def test_spline_ridge_recovers_nonlinearity_with_constant_and_binary_covariates(self):
        x = np.linspace(-2, 2, 150)
        matrix = np.column_stack([x, np.ones(150), x > 0])
        y = x * x
        train = np.arange(150) % 3 != 0
        fit = BlockRidge([3, 0], ["host", "context"], dict(penalty=0.00001), spline=True).fit(
            matrix[train], y[train]
        )
        linear = Ridge(alpha=0.01).fit(matrix[train], y[train])
        mse = np.mean((fit.predict(matrix[~train]) - y[~train]) ** 2)
        self.assertLess(mse, 0.05 * np.mean((linear.predict(matrix[~train]) - y[~train]) ** 2))

    def test_all_explicit_zero_options_and_native_contrasts_with_rank_dropped_sex(self):
        data, cfg = fixture()
        cfg = reduced(cfg)
        data.index["sex_code"] = 1
        role = role_for(data, cfg)
        specs = {s["name"]: s for s in specifications(cfg)}
        for name, null in INTERACTIONS.items():
            base, _ = atomic(role, specs[null], specs[null]["grid"][0], cfg, 1)
            zero, _ = interaction(role, specs[name], dict(weight=0), base, cfg)
            np.testing.assert_array_equal(zero.predict(role), base.predict(role))
            np.testing.assert_array_equal(zero.modifier(role), np.zeros(len(role.split.yv)))
            fit, _ = interaction(role, specs[name], dict(weight=1, components=3, penalty=0.01), base, cfg)
            additions = []
            for diet in ([0.0, 0.0], role.split.delta):
                changed = copy.copy(role)
                changed.split = copy.copy(role.split)
                changed.split.hv = role.host_at(diet)
                changed.split.dv = np.tile(diet, (len(role.split.yv), 1))
                additions.append(fit.predict(changed) - base.predict(changed))
            np.testing.assert_allclose(fit.modifier(role), additions[1] - additions[0], atol=1e-10)

    def test_pure_microbiome_modifier_can_be_learned(self):
        data, cfg = fixture()
        role = role_for(data, cfg)
        design = role.design("balanced", "pc", dict(components=5, count=0), 1, "full")
        z, zv, _ = role.modifier_basis("gut", 3)
        role.split.yt = 6 * z[:, 0] * role.split.dt[:, 0]
        target = 6 * zv[:, 0] * role.split.dv[:, 0]
        base = Fitted(design["state"], ZeroModel(), 0, 1)
        spec = dict(source="gut", nonlinear=False)
        fit, _ = interaction(role, spec, dict(weight=1, components=3, penalty=0.00001), base, cfg)
        self.assertLess(np.mean((target - fit.predict(role)) ** 2), 0.6 * np.mean(target**2))
        self.assertGreater(np.std(fit.modifier(role)), 1)

    def test_convex_fusion_recovers_known_weights(self):
        rng = np.random.default_rng(13)
        matrix = rng.normal(size=(100, 4))
        w = np.array([0.2, 0.5, 0.3, 0])
        fit = blend_weights(matrix, matrix @ w)
        np.testing.assert_allclose(fit, w, atol=1e-5)
