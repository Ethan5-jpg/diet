"""Projection, covariate-context and screening behavior on synthetic data."""

import pickle
import unittest

import numpy as np

from features import Role, host_at, screen
from support import fixture


class FeatureChecks(unittest.TestCase):
    def test_constant_sex_can_be_rank_dropped_without_breaking_diet_contrast(self):
        data, cfg = fixture()
        data.index["sex_code"] = 0
        train = data.reference["outer"][0] != 0
        s = data.split(train, ~train, cfg["contrast"])
        self.assertNotIn("sex", s.preprocessing["host"]["labels"])
        changed = host_at(s, s.hv, s.delta)
        self.assertTrue(np.isfinite(changed).all())
        role = Role(s, data.raw, data.index, cfg["model_seed"])
        np.testing.assert_allclose(changed, role.host_at(s.delta), atol=1e-12)

    def test_retained_diet_interaction_uses_raw_covariate_when_main_column_is_dropped(self):
        import copy

        data, cfg = fixture()
        train = data.reference["outer"][0] != 0
        s = data.split(train, ~train, cfg["contrast"])
        s.preprocessing = copy.deepcopy(s.preprocessing)
        labels = s.preprocessing["host"]["labels"]
        position = labels.index("age")
        labels.pop(position)
        s.ht, s.hv = np.delete(s.ht, position, 1), np.delete(s.hv, position, 1)
        role = Role(s, data.raw, data.index, cfg["model_seed"])
        state = s.preprocessing["host"]
        for training, rows in ((True, s.train_rows), (False, s.rows)):
            value = role.host_at(s.delta, training)
            raw = data.index.iloc[rows].age_calendar_years.to_numpy()
            expected = s.delta[0] * (raw - state["numeric_means"][0]) / state["numeric_scales"][0]
            np.testing.assert_allclose(value[:, labels.index("diet_age")], expected, atol=1e-12)

    def test_cached_projection_warning_survives_other_learners_and_serialization(self):
        import warnings
        from unittest.mock import patch
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.exceptions import ConvergenceWarning
        from engine import Engine

        data, cfg = fixture()
        train = data.reference["outer"][0] != 0
        role = Role(data.split(train, ~train, cfg["contrast"]), data.raw, data.index, cfg["model_seed"])
        original = PLSRegression.fit

        def fit_with_warning(model, *args, **kwargs):
            fitted = original(model, *args, **kwargs)
            warnings.warn("Synthetic cached solver warning", ConvergenceWarning)
            return fitted

        with patch.object(PLSRegression, "fit", fit_with_warning):
            role.design("oral", "pls", dict(count=0, components=3), cfg["nmf_iterations"])
        engine = Engine(cfg, None, 1, None, lambda _: None)
        for learner, case in (
            ("hgb", dict(count=0, components=3, iterations=3, leaves=3)),
            ("cat", dict(count=0, components=3, iterations=3, depth=2)),
        ):
            spec = dict(name="cached_check", site="oral", representation="pls", learner=learner)
            result = engine.evaluate(role, spec, case, {}, keep=True)
            self.assertIn("Synthetic cached solver warning", result["diagnostic"]["warnings"])
            clone = pickle.loads(pickle.dumps(result["fitted"]))
            self.assertIn("Synthetic cached solver warning", clone.state["oral"]["factor"]["warnings"])

    def test_replay_matches_training_and_validation_for_every_representation(self):
        data, cfg = fixture()
        train = data.reference["outer"][0] != 0
        role = Role(data.split(train, ~train, cfg["contrast"]), data.raw, data.index, 13)
        for method in ("clr", "hellinger", "nmf", "hybrid", "pls", "pc", "screen"):
            case = dict(count=6, components=3)
            design = role.design("balanced", method, case, 150, joint=True)
            state = pickle.loads(pickle.dumps(design["state"]))
            for training, key in ((True, "xt"), (False, "xv")):
                actual, _ = role.replay(state, training)
                np.testing.assert_allclose(actual, design[key], atol=1e-10)

    def test_interaction_screen_recovers_a_pure_modifier(self):
        rng = np.random.default_rng(772)
        n = 3000
        d, z = rng.normal(size=(n, 1)), rng.normal(size=(n, 60))
        h = np.column_stack([np.ones(n), d])
        y = 3 * d[:, 0] * z[:, 29] + rng.normal(size=n)
        self.assertIn(29, screen(h, z, y, 3, d))

    def test_validation_features_do_not_change_factors_selectors_or_upf_imputation(self):
        import copy

        data, cfg = fixture()
        train = data.reference["outer"][0] != 0
        split = data.split(train, ~train, cfg["contrast"])
        role = Role(split, data.raw, data.index, 9)
        other_raw, other_index = copy.deepcopy(data.raw), data.index.copy()
        for a in other_raw.values():
            a[~train] = a[~train, ::-1]
        other_index.loc[~train, list(cfg["upf_columns"])] = 0.9
        other = Role(split, other_raw, other_index, 9)
        for method in ("nmf", "hybrid", "pls", "screen"):
            case = dict(count=4, components=3)
            a = role.design("balanced", method, case, 150, joint=True)
            b = other.design("balanced", method, case, 150, joint=True)
            np.testing.assert_allclose(a["xt"], b["xt"], atol=1e-12)
