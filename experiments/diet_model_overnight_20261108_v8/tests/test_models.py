"""Independent numerical checks and actual configured backend smoke tests."""

import copy
import pickle
import unittest
import warnings

import numpy as np
from scipy.optimize import check_grad

from engine import blend_weights
from estimators import GroupLasso, fm_objective
from features import Role, host_at
from models import atomic, interaction
from registry import INTERACTIONS, specifications
from support import fixture, reduced


class ModelChecks(unittest.TestCase):
    def test_all_explicit_contrasts_work_after_constant_covariate_rank_drop(self):
        data, cfg = fixture()
        cfg = reduced(cfg)
        data.index["sex_code"] = 0
        train = data.reference["outer"][0] != 0
        role = Role(data.split(train, ~train, cfg["contrast"]), data.raw, data.index, cfg["model_seed"])
        specs = {s["name"]: s for s in specifications(cfg)}
        for name, null in INTERACTIONS.items():
            with self.subTest(model=name), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                base, _ = atomic(role, specs[null], specs[null]["grid"][0], cfg, 1)
                model, _ = interaction(role, specs[name], specs[name]["grid"][1], base, cfg)
                values = []
                for delta in ([0.0, 0.0], role.split.delta):
                    changed = copy.copy(role.split)
                    changed.hv = role.host_at(delta)
                    changed.dv = np.tile(delta, (len(changed.yv), 1))
                    alternate = Role(changed, role.raw, role.index, role.seed)
                    values.append(model.predict(alternate) - base.predict(alternate))
                np.testing.assert_allclose(values[1] - values[0], model.modifier(role), atol=1e-8)

    def test_fm_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(617)
        x, y = rng.normal(size=(31, 7)), rng.normal(size=31)
        theta = rng.normal(size=1 + 7 + 7 * 3)
        error = check_grad(
            lambda t: fm_objective(t, x, y, 3, 0.1)[0], lambda t: fm_objective(t, x, y, 3, 0.1)[1], theta
        )
        self.assertLess(error, 2e-5)

    def test_group_solver_matches_closed_form_orthogonal_groups(self):
        rng = np.random.default_rng(818)
        x = np.linalg.qr(rng.normal(size=(150, 10)))[0] * np.sqrt(150)
        y = x @ np.array([1, 2, 0, 0, -1, 3, 0, 0, 0.1, 0.2]) + rng.normal(0, 0.1, 150)
        model = GroupLasso(0.2, ridge=0.02).fit(x, y)
        rhs = (x.T @ y / len(x)).reshape(-1, 2)
        expected = GroupLasso.prox(rhs, model.penalty).ravel() / 1.02
        np.testing.assert_allclose(model.coef_, expected, atol=2e-6)
        self.assertTrue(model.diagnostic["converged"])

    def test_convex_fusion_recovers_known_optimum(self):
        rng = np.random.default_rng(923)
        x = rng.normal(size=(500, 3))
        weights = blend_weights(x, x @ np.array([0.2, 0.7, 0.1]))
        np.testing.assert_allclose(weights, [0.2, 0.7, 0.1], atol=2e-6)
        self.assertAlmostEqual(weights.sum(), 1.0)
        self.assertGreaterEqual(weights.min(), 0)

    def test_every_default_case_fits_predicts_and_serializes_wide_profiles(self):
        data, cfg = fixture(wide=True)
        train = data.reference["outer"][0] != 0
        role = Role(data.split(train, ~train, cfg["contrast"]), data.raw, data.index, cfg["model_seed"])
        bases = {}
        for spec in specifications(cfg):
            for j, case in enumerate(spec["grid"]):
                with self.subTest(model=spec["name"], candidate=j), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitted, _ = (
                        interaction(role, spec, case, bases[INTERACTIONS[spec["name"]]], cfg)
                        if spec["name"] in INTERACTIONS
                        else atomic(role, spec, case, cfg, 1)
                    )
                    prediction = fitted.predict(role)
                    self.assertEqual(prediction.shape, role.split.yv.shape)
                    self.assertTrue(np.isfinite(prediction).all())
                    clone = pickle.loads(pickle.dumps(fitted))
                    np.testing.assert_allclose(prediction, clone.predict(role), atol=1e-10)
                    if spec["name"] in set(INTERACTIONS.values()) and j == 0:
                        bases[spec["name"]] = fitted
            role.cache = {k: v for k, v in role.cache.items() if k[0] != "design"}

    def test_zero_interactions_equal_their_null_and_contrast_is_exact(self):
        data, cfg = fixture()
        cfg = reduced(cfg)
        train = data.reference["outer"][0] != 0
        role = Role(data.split(train, ~train, cfg["contrast"]), data.raw, data.index, cfg["model_seed"])
        specs = {s["name"]: s for s in specifications(cfg)}
        for name, null in INTERACTIONS.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                base, _ = atomic(role, specs[null], specs[null]["grid"][0], cfg, 1)
                zero, _ = interaction(role, specs[name], dict(weight=0.0), base, cfg)
                np.testing.assert_array_equal(zero.predict(role), base.predict(role))
                np.testing.assert_array_equal(zero.modifier(role), np.zeros(len(role.split.yv)))
                model, _ = interaction(role, specs[name], specs[name]["grid"][1], base, cfg)
                values = []
                for delta in ([0.0, 0.0], role.split.delta):
                    changed = copy.copy(role.split)
                    changed.hv = host_at(role.split, role.split.hv, delta)
                    changed.dv = np.tile(delta, (len(role.split.yv), 1))
                    new_role = Role(changed, role.raw, role.index, role.seed)
                    values.append(model.predict(new_role) - base.predict(new_role))
                np.testing.assert_allclose(values[1] - values[0], model.modifier(role), atol=1e-8)

    def test_sparse_correction_recovers_known_heterogeneity(self):
        data, cfg = fixture()
        train = data.reference["outer"][0] != 0
        split = data.split(train, ~train, cfg["contrast"])
        rng = np.random.default_rng(667)
        for host, z, d, name in ((split.ht, split.ot, split.dt, "yt"), (split.hv, split.ov, split.dv, "yv")):
            setattr(split, name, 90 + 3 * host[:, 1] + 8 * z[:, 2] * d[:, 0] + rng.normal(0, 0.5, len(z)))
        role = Role(split, data.raw, data.index, cfg["model_seed"])
        specs = {s["name"]: s for s in specifications(cfg)}
        base, _ = atomic(role, specs["linear_main_oral"], dict(count=0, penalty=0.1), cfg, 1)
        fitted, note = interaction(
            role, specs["sparse_diet_oral"], dict(weight=1.0, fraction=0.03), base, cfg
        )
        self.assertTrue(note["converged"])
        self.assertGreater(np.std(fitted.modifier(role)), 3.0)
        self.assertLess(
            np.mean((split.yv - fitted.predict(role)) ** 2),
            0.25 * np.mean((split.yv - base.predict(role)) ** 2),
        )
