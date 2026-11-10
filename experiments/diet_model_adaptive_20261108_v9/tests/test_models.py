"""Analytic gradients, known synthetic signal and all actual default backends."""

import copy
import pickle
import unittest
import warnings

import numpy as np
from scipy.optimize import check_grad
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from estimators import ConditionalGate, ResidualMLP, gate_value_gradient, neural_value_gradient
from features import Role
from fusion import blend_weights
from models import atomic, interaction
from registry import INTERACTIONS, specifications
from support import fixture, reduced


class ModelChecks(unittest.TestCase):
    def test_two_layer_neural_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(61)
        x, y, h = rng.normal(size=(17, 3)), rng.normal(size=17), 4
        theta = rng.normal(0, 0.1, 3 * h + h + h * 4 + 4 + 4 + 1)
        error = check_grad(
            lambda t: neural_value_gradient(t, x, y, h, 0.02)[0],
            lambda t: neural_value_gradient(t, x, y, h, 0.02)[1],
            theta,
        )
        self.assertLess(error, 2e-6)

    def test_softmax_gate_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(81)
        x = np.column_stack([np.ones(31), rng.normal(size=(31, 2))])
        p, y, prior = rng.normal(size=(31, 3)), rng.normal(size=31), np.array([0.2, 0.5, 0.3])
        theta = rng.normal(0, 0.2, 9)
        error = check_grad(
            lambda t: gate_value_gradient(t, x, p, y, prior, 0.1)[0],
            lambda t: gate_value_gradient(t, x, p, y, prior, 0.1)[1],
            theta,
        )
        self.assertLess(error, 2e-6)

    def test_conditional_gate_learns_which_expert_is_useful_in_a_subgroup(self):
        rng = np.random.default_rng(93)
        x, p = rng.normal(size=(900, 2)), rng.normal(size=(900, 3))
        y = np.where(x[:, 0] > 0, p[:, 0], p[:, 1]) + rng.normal(0, 0.05, 900)
        model = ConditionalGate(0.001).fit(x[:600], p[:600], y[:600], np.array([0.5, 0.5, 0]))
        prediction = model.predict(x[600:], p[600:])
        baseline = (p[600:, 0] + p[600:, 1]) / 2
        self.assertLess(np.mean((y[600:] - prediction) ** 2), 0.5 * np.mean((y[600:] - baseline) ** 2))
        np.testing.assert_allclose(model.weights(x[600:]).sum(1), 1, atol=1e-12)
        self.assertTrue((model.weights(x[600:]) >= 0).all())
        clone = pickle.loads(pickle.dumps(model))
        np.testing.assert_allclose(prediction, clone.predict(x[600:], p[600:]), atol=0)

    def test_residual_network_learns_known_nonlinearity(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(-2, 2, (600, 2))
        y = np.sin(2 * x[:, 0]) + 0.3 * x[:, 1] ** 2
        model = ResidualMLP(dict(hidden=24, epochs=180, decay=0.001, shrinkage=1), 1, 91).fit(
            x[:400], y[:400]
        )
        scale = StandardScaler().fit(x[:400])
        base = Ridge(alpha=40).fit(scale.transform(x[:400]), y[:400])
        self.assertLess(
            np.mean((y[400:] - model.predict(x[400:])) ** 2),
            0.6 * np.mean((y[400:] - base.predict(scale.transform(x[400:]))) ** 2),
        )

    def test_convex_fusion_recovers_a_known_solution(self):
        rng = np.random.default_rng(49)
        x = rng.normal(size=(500, 3))
        w = blend_weights(x, x @ np.array([0.2, 0.7, 0.1]))
        np.testing.assert_allclose(w, [0.2, 0.7, 0.1], atol=2e-6)

    def test_all_default_cases_fit_predict_and_serialize_on_sparse_wide_profiles(self):
        data, cfg = fixture(wide=True)
        train = data.reference["outer"][0] != 0
        role = Role(
            data.split(train, ~train, cfg["contrast"]),
            data.raw,
            data.index,
            cfg["model_seed"],
            taxa=data.taxa,
            extra=data.extra,
        )
        bases = {}
        for spec in specifications(cfg):
            for j, case in enumerate(spec["grid"]):
                with self.subTest(model=spec["name"], candidate=j), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model, _ = (
                        interaction(role, case, bases[INTERACTIONS[spec["name"]]])
                        if spec["name"] in INTERACTIONS
                        else atomic(role, spec, case, cfg, 1)
                    )
                    prediction = model.predict(role)
                    self.assertTrue(np.isfinite(prediction).all())
                    self.assertEqual(prediction.shape, role.split.yv.shape)
                    clone = pickle.loads(pickle.dumps(model))
                    np.testing.assert_allclose(prediction, clone.predict(role), atol=1e-9)
                    if spec["name"] in set(INTERACTIONS.values()) and j == 0:
                        bases[spec["name"]] = model
            role.cache = {k: v for k, v in role.cache.items() if k[0] != "design"}

    def test_matched_null_zero_and_native_contrasts_are_exact_after_rank_drop(self):
        data, cfg = fixture()
        cfg = reduced(cfg)
        data.index["sex_code"] = 0
        train = data.reference["outer"][0] != 0
        role = Role(
            data.split(train, ~train, cfg["contrast"]),
            data.raw,
            data.index,
            cfg["model_seed"],
            taxa=data.taxa,
            extra=data.extra,
        )
        specs = {s["name"]: s for s in specifications(cfg)}
        for n, base_name in INTERACTIONS.items():
            base, _ = atomic(role, specs[base_name], specs[base_name]["grid"][0], cfg, 1)
            zero, _ = interaction(role, dict(weight=0), base)
            np.testing.assert_array_equal(zero.predict(role), base.predict(role))
            model, _ = interaction(role, dict(weight=1, penalty=0.01), base)
            values = []
            for delta in ([0.0, 0.0], role.split.delta):
                s = copy.copy(role.split)
                s.hv, s.dv = role.host_at(delta), np.tile(delta, (len(s.yv), 1))
                changed = Role(s, role.raw, role.index, role.seed, taxa=role.taxa, extra=role.extra)
                values.append(model.predict(changed) - base.predict(changed))
            np.testing.assert_allclose(values[1] - values[0], model.modifier(role), atol=1e-8)

    def test_target_inverse_never_uses_validation_targets(self):
        data, cfg = fixture()
        cfg = reduced(cfg)
        train = data.reference["outer"][0] != 0
        role = Role(
            data.split(train, ~train, cfg["contrast"]),
            data.raw,
            data.index,
            cfg["model_seed"],
            taxa=data.taxa,
            extra=data.extra,
        )
        spec = next(s for s in specifications(cfg) if s["name"] == "gauss_target_cat_oral")
        model, _ = atomic(role, spec, spec["grid"][0], cfg, 1)
        prediction = model.predict(role)
        other = copy.copy(role.split)
        other.yv = other.yv + 1e6
        changed = Role(other, role.raw, role.index, role.seed, taxa=role.taxa, extra=role.extra)
        np.testing.assert_array_equal(prediction, model.predict(changed))
        self.assertTrue((prediction >= role.split.yt.min() - 1e-8).all())
        self.assertTrue((prediction <= role.split.yt.max() + 1e-8).all())
