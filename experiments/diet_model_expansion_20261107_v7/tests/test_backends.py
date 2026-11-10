import copy
import json
import pickle
import unittest

import numpy as np
from numpy.testing import assert_allclose
from support import ROOT, fixture

from features import Role
from models import atomic, interaction
from registry import INTERACTIONS, specifications


class BackendChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset, cls.config = fixture()
        assignment = cls.dataset.reference["outer"][0]
        cls.split = cls.dataset.split(assignment != 0, assignment == 0, cls.config["contrast"])

    def test_all_candidate_schemas_fit_predict_serialize(self):
        role = Role(self.split, self.dataset.raw, 7)
        bases, cache = {}, {}

        def remember(key, compute):
            if key not in cache:
                cache[key] = compute()
            return cache[key]

        for spec in specifications(self.config):
            for case in spec["grid"]:
                with self.subTest(name=spec["name"], case=case):
                    if spec["name"] in INTERACTIONS:
                        fit, _ = interaction(
                            role,
                            spec,
                            case,
                            bases[INTERACTIONS[spec["name"]]],
                            self.config,
                            1,
                            remember,
                        )
                        axis = fit.modifier(role)
                        self.assertEqual(axis.shape, self.split.yv.shape)
                        self.assertTrue(np.isfinite(axis).all())
                        if case["weight"] == 0:
                            assert_allclose(
                                fit.predict(role), fit.base.predict(role), atol=0, rtol=0
                            )
                    else:
                        fit, _ = atomic(role, spec, case, self.config, 1)
                        bases[spec["name"]] = fit
                    pred = fit.predict(role)
                    self.assertTrue(np.isfinite(pred).all())
                    self.assertEqual(pred.shape, self.split.yv.shape)
                    assert_allclose(pickle.loads(pickle.dumps(fit)).predict(role), pred, atol=1e-10)
        self.role, self.bases = role, bases

    def test_exact_default_grid_on_wide_synthetic_microbiomes(self):
        dataset = copy.copy(self.dataset)
        dataset.raw, dataset.taxa = {}, {}
        rng = np.random.default_rng(814)
        for site, width in (("oral", 144), ("gut", 176)):
            values = np.exp(rng.normal(size=(len(dataset.y), width)))
            values[rng.random(values.shape) < 0.15] = 0
            dataset.raw[site] = values / values.sum(1, keepdims=True)
            dataset.taxa[site] = ["synthetic_%d" % j for j in range(width)]
        config = json.loads((ROOT / "config.json").read_text())
        assignment = dataset.reference["outer"][0]
        split = dataset.split(assignment != 0, assignment == 0, config["contrast"])
        role, bases, cache = Role(split, dataset.raw, config["model_seed"]), {}, {}

        def remember(key, compute):
            if key not in cache:
                cache[key] = compute()
            return cache[key]

        for spec in specifications(config):
            for case in spec["grid"]:
                with self.subTest(name=spec["name"], exact_default_case=case):
                    if spec["name"] in INTERACTIONS:
                        fit, _ = interaction(
                            role, spec, case, bases[INTERACTIONS[spec["name"]]], config, 1, remember
                        )
                        self.assertTrue(np.isfinite(fit.modifier(role)).all())
                    else:
                        fit, _ = atomic(role, spec, case, config, 1)
                        bases[spec["name"]] = fit
                    self.assertTrue(np.isfinite(fit.predict(role)).all())
            role.cache = {k: v for k, v in role.cache.items() if k[0] == "pc"}

    def test_ebm_terms_sum_and_interaction_contrast(self):
        role = Role(self.split, self.dataset.raw, 7)
        spec = next(s for s in specifications(self.config) if s["name"] == "ebm_balanced")
        base, _ = atomic(role, spec, spec["grid"][0], self.config, 1)
        inter = next(s for s in specifications(self.config) if s["name"] == "ebm_diet_gut")
        fit, _ = interaction(
            role, inter, dict(weight=1.0), base, self.config, 1, lambda _, fn: fn()
        )
        full = fit.extra["full"]
        x, _ = role.replay(full.design_state)
        terms = full.model.eval_terms(x)
        assert_allclose(full.model.predict(x), terms.sum(1) + full.model.intercept_, atol=1e-12)
        low, high = role.low_high(x, self.config["contrast"])
        pairs = fit.extra["terms"]
        expected = (
            full.model.eval_terms(high)[:, pairs] - full.model.eval_terms(low)[:, pairs]
        ).sum(1) * full.scale
        assert_allclose(fit.modifier(role), expected, atol=1e-12)
        self.assertTrue(all(len(full.model.term_features_[j]) == 2 for j in pairs))

    def test_training_representation_and_selection_ignore_validation_features(self):
        original = Role(self.split, self.dataset.raw, 3)
        altered_split = copy.deepcopy(self.split)
        altered_raw = copy.deepcopy(self.dataset.raw)
        altered_split.ov += 123
        altered_split.gv -= 321
        for site in altered_raw:
            altered_raw[site][self.split.rows] = np.roll(
                altered_raw[site][self.split.rows], 2, axis=1
            )
        altered = Role(altered_split, altered_raw, 3)
        for representation in ("clr", "hellinger", "presence"):
            a = original.design("balanced", representation, 4)
            b = altered.design("balanced", representation, 4)
            assert_allclose(a.xt, b.xt, rtol=0, atol=0)
            for site in ("oral", "gut"):
                assert_allclose(a.state[site]["indices"], b.state[site]["indices"])
        assert_allclose(original.pcs()[2].components_, altered.pcs()[2].components_, rtol=0, atol=0)

    def test_additive_null_has_profile_independent_diet_contrast(self):
        role = Role(self.split, self.dataset.raw, 7)
        spec = next(s for s in specifications(self.config) if s["name"] == "additive_boost")
        base, _ = atomic(role, spec, spec["grid"][0], self.config, 1)
        _, blocks = role.replay(base.design_state)
        low = [b.copy() for b in blocks]
        high = [b.copy() for b in blocks]
        low[0][:] = low[0][0]
        high[0][:] = high[0][0]
        j = self.split.preprocessing["host"]["labels"].index("diet_linear")
        low[0][:, j], high[0][:, j] = -1.0, 1.0
        contrast = base.model.predict(high) - base.model.predict(low)
        assert_allclose(contrast, np.full(len(contrast), contrast[0]), atol=1e-12)

    def test_pc_interaction_modifier_matches_finite_difference(self):
        role = Role(self.split, self.dataset.raw, 7)
        spec = next(s for s in specifications(self.config) if s["name"] == "additive_boost")
        base, _ = atomic(role, spec, spec["grid"][0], self.config, 1)
        inter = next(
            s for s in specifications(self.config) if s["name"] == "additive_pc_interaction"
        )
        fit, _ = interaction(
            role, inter, dict(weight=1.0, penalty=0.1), base, self.config, 1, lambda _, fn: fn()
        )
        before = fit.extra["ridge"].predict(fit.matrix(role))
        changed = copy.deepcopy(role)
        changed.split.dv += role.split.delta
        after = fit.extra["ridge"].predict(fit.matrix(changed))
        assert_allclose(after - before, fit.modifier(role), atol=1e-12)
