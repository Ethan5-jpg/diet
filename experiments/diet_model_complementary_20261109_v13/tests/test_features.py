"""Projection identities, planted directions and frozen-CLR alignment."""

import copy
import pickle
import unittest

import numpy as np

from features import Directions, Role
from registry import specifications
from support import fixture


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


class FeatureChecks(unittest.TestCase):
    def test_sir_recovers_linear_and_save_recovers_symmetric_signal(self):
        rng = np.random.default_rng(1301)
        x = rng.normal(size=(3200, 8))
        nuisance = np.ones((len(x), 1))
        for method, y in (
            ("sir", 3 * x[:, 0] + 0.05 * rng.normal(size=len(x))),
            ("save", 3 * x[:, 0] ** 2 + 0.05 * rng.normal(size=len(x))),
        ):
            fitted = Directions(method, 1, 12).fit(x, nuisance, y)
            direction = fitted.projection[:, 0]
            self.assertGreater(abs(direction[0]) / np.linalg.norm(direction), 0.94)
            self.assertGreater(np.var(fitted.transform(x, nuisance)), 0.99)

    def test_constant_target_rank_drop_and_serialized_projection(self):
        rng = np.random.default_rng(1302)
        a = rng.normal(size=(100, 4))
        x = np.column_stack([a, a[:, 0], np.ones(100)])
        c = np.column_stack([np.ones(100), a[:, 1], a[:, 1]])
        for method in ("pc", "sir", "save", "graph"):
            f = Directions(method, 6, 8).fit(x, c, np.ones(100))
            expected = f.transform(x, c)
            self.assertTrue(np.isfinite(expected).all())
            np.testing.assert_array_equal(expected, pickle.loads(pickle.dumps(f)).transform(x, c))
            if method in ("sir", "save"):
                np.testing.assert_array_equal(expected, np.zeros_like(expected))

    def test_all_representations_replay_and_use_exact_raw_taxon_mapping(self):
        data, cfg = fixture(wide=True)
        for site in data.raw:
            data.raw[site][:, [2, 7, 12]] = 0
            data.raw[site] /= data.raw[site].sum(1, keepdims=True)
        role = role_for(data, cfg)
        for method in ("pc", "sir", "save", "graph"):
            case = dict(count=18, components=6, slices=8)
            d = role.design("balanced", method, case, 1)
            saved = pickle.loads(pickle.dumps(d["state"]))
            np.testing.assert_allclose(d["xt"], role.replay(saved, True)[0], atol=1e-12)
            np.testing.assert_allclose(d["xv"], role.replay(saved)[0], atol=1e-12)
            for state in saved["microbes"]:
                frozen = role.split.preprocessing["CLR"][state["site"]]
                columns = np.asarray(frozen["reference_indices"])[np.asarray(frozen["output_indices"])]
                np.testing.assert_array_equal(state["original_indices"], columns[state["indices"]])
                self.assertFalse(set(state["original_indices"]) & {2, 7, 12})
                self.assertTrue((columns != np.arange(len(columns))).any())

    def test_validation_values_do_not_fit_context_prevalence_or_slicing(self):
        data, cfg = fixture(wide=True)
        altered = copy.copy(data)
        altered.raw, altered.extra = copy.deepcopy(data.raw), copy.deepcopy(data.extra)
        valid = data.reference["outer"][0] == 0
        for a in altered.raw.values():
            a[valid] = a[valid, ::-1]
        for a in altered.extra["blocks"].values():
            a[valid] = 999
        altered.extra["alcohol"][valid] = "validation_only"
        roles = [role_for(d, cfg) for d in (data, altered)]
        for method in ("pc", "sir", "save", "graph"):
            designs = [r.design("balanced", method, dict(count=18, components=6, slices=8), 1) for r in roles]
            np.testing.assert_array_equal(designs[0]["xt"], designs[1]["xt"])
            for a, b in zip(designs[0]["state"]["microbes"], designs[1]["state"]["microbes"]):
                np.testing.assert_array_equal(a["indices"], b["indices"])
                np.testing.assert_array_equal(a["encoder"].projection, b["encoder"].projection)
        # Slice count belongs to the cache key even with equal dimensions.
        role = roles[0]
        role.design("oral", "save", dict(count=18, components=6, slices=5), 1)
        self.assertTrue(any(k[0] == "representation" and k[5] == 5 for k in role.cache))

    def test_joint_context_matches_the_original_v11_encoding(self):
        from bridge import V11, imported
        from food_features import DietEncoder

        data, cfg = fixture()
        old = imported("v13_original_food_encoder", V11 / "food_features.py").DietEncoder
        rows = np.flatnonzero(data.reference["outer"][0] != 0)
        valid = np.flatnonzero(data.reference["outer"][0] == 0)
        models = [cls("joint").fit(data.extra, data.index, rows) for cls in (old, DietEncoder)]
        for subset in (rows, valid):
            np.testing.assert_array_equal(
                models[0].transform(data.extra, data.index, subset),
                models[1].transform(data.extra, data.index, subset),
            )

    def test_candidate_registry_has_matched_controls_and_no_untried_claims(self):
        _, cfg = fixture()
        specs = specifications(cfg)
        self.assertEqual(len(specs), 38)
        self.assertEqual(len({v["name"] for v in specs}), 38)
        for spec in specs:
            self.assertEqual(spec["context"], "joint")
            self.assertNotIn(spec["learner"], ("fm", "mlp", "nystrom"))
