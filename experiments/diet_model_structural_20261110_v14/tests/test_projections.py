"""Independent direction checks, raw-column mapping and role-isolated replay."""

import copy
import pickle
import unittest

import numpy as np

from features import Role
from models import atomic
from projections import FourierMap, NonlinearDirections, PairedViews, StablePCA
from registry import specifications
from support import fixture


def role(data):
    valid = data.reference["outer"][0] == 0
    return Role(
        data.split(~valid, valid, [10, 15]), data.raw, data.index, 20261114, taxa=data.taxa, extra=data.extra
    )


class ProjectionChecks(unittest.TestCase):
    def test_rank_deficient_pca_zeros_unsupported_training_and_query_coordinates(self):
        rng = np.random.default_rng(1400)
        x = rng.normal(size=(100, 2))
        repeated = np.column_stack([x, x, 2 * x, x[:, [0]]])
        model = StablePCA(n_components=7, svd_solver="full").fit(repeated)
        self.assertEqual(int(model.active_.sum()), 2)
        query = rng.normal(size=(8, 7)) * 100
        np.testing.assert_array_equal(model.transform(query)[:, ~model.active_], 0)
        np.testing.assert_allclose(model.fit_transform(repeated), model.transform(repeated), atol=1e-12)

    def test_fourier_gradient_matches_independent_finite_differences(self):
        rng = np.random.default_rng(1401)
        x, beta = rng.normal(size=(12, 5)), rng.normal(size=40)
        encoder = FourierMap(40, 1.2, 3).fit(x)
        analytic = encoder.gradient(x, beta)
        numerical = np.zeros_like(analytic)
        for j in range(x.shape[1]):
            plus, minus = x.copy(), x.copy()
            plus[:, j] += 1e-6
            minus[:, j] -= 1e-6
            numerical[:, j] = (encoder.transform(plus) @ beta - encoder.transform(minus) @ beta) / 2e-6
        np.testing.assert_allclose(analytic, numerical, atol=1e-8, rtol=1e-6)

    def test_pilot_gradient_recovers_a_known_symmetric_direction(self):
        rng = np.random.default_rng(1402)
        x = rng.normal(size=(800, 4))
        nuisance = np.ones((len(x), 1))
        y = x[:, 0] ** 2 + 0.05 * rng.normal(size=len(x))
        case = dict(pre_components=4, rff_features=256, bandwidth=1.0, pilot_penalty=0.003, components=1)
        model = NonlinearDirections("pilot_gradient", case, 14).fit(x, nuisance, y)
        direction = model.pc.pca.components_.T @ (model.projection[:, 0] / model.pc.scaler.scale_)
        self.assertGreater(abs(direction[0]) / np.linalg.norm(direction), 0.9)

    def test_regularized_cca_recovers_shared_coordinates_and_matches_control_width(self):
        rng = np.random.default_rng(1403)
        shared = rng.normal(size=(400, 2))
        x = shared @ rng.normal(size=(2, 12)) + 0.2 * rng.normal(size=(400, 12))
        y = shared @ rng.normal(size=(2, 16)) + 0.2 * rng.normal(size=(400, 16))
        nuisance = np.ones((400, 1))
        c = dict(pre_components=8, components=2, cca_penalty=0.05)
        model = PairedViews("cca", c, 1).fit([x, y], nuisance)
        control = PairedViews("paired_pc", c, 1).fit([x, y], nuisance)
        a, b = [pc.transform(v, nuisance) for pc, v in zip(model.pcs, (x, y))]
        ca, cb = a @ model.axes[0], b @ model.axes[1]
        self.assertGreater(min(np.corrcoef(ca[:, j], cb[:, j])[0, 1] for j in range(2)), 0.95)
        self.assertEqual(model.transform([x, y], nuisance).shape, control.transform([x, y], nuisance).shape)

    def test_every_default_candidate_fits_predicts_and_serializes_wide_sparse_inputs(self):
        data, cfg = fixture(wide=True)
        r = role(data)
        count = 0
        for spec in specifications(cfg):
            for case in spec["grid"]:
                model, note = atomic(r, spec, case, cfg, 1)
                prediction = model.predict(r)
                self.assertTrue(np.isfinite(prediction).all(), spec["name"])
                self.assertEqual(prediction.shape, r.split.yv.shape)
                copied = pickle.loads(pickle.dumps(model))
                np.testing.assert_array_equal(prediction, copied.predict(r))
                design = r.design(spec["site"], spec["representation"], case, 1)
                np.testing.assert_array_equal(design["xt"], r.replay(model.state, True)[0])
                np.testing.assert_array_equal(design["xv"], r.replay(model.state)[0])
                for saved in model.state["microbes"]:
                    inputs = saved["inputs"] if saved["paired"] else [saved]
                    for state in inputs:
                        self.assertEqual(len(state["indices"]), case["count"])
                        original = data.raw[state["site"]][
                            np.ix_(r.split.train_rows, state["original_indices"])
                        ]
                        self.assertTrue((original > 0).any())
                self.assertIsInstance(note["warnings"], list)
                count += 1
                if count % 20 == 0:
                    print("CHECK_CANDIDATES %d/80" % count, flush=True)
        self.assertEqual(count, 80)

    def test_outer_features_do_not_change_training_representation_or_imputation(self):
        data, cfg = fixture(wide=True)
        changed = copy.copy(data)
        changed.raw, changed.extra = copy.deepcopy(data.raw), copy.deepcopy(data.extra)
        valid = data.reference["outer"][0] == 0
        for a in changed.raw.values():
            a[valid] = a[valid, ::-1]
        for a in changed.extra["blocks"].values():
            a[valid] = 999
        changed.extra["alcohol"][valid] = "outer_only"
        a, b = role(data), role(changed)
        for method in ("sir", "rff_sir", "pilot_gradient", "cca", "paired_pc"):
            case = cfg["families"]["hgb"]["grid"][0]
            da, db = [r.design("balanced", method, case, 1) for r in (a, b)]
            np.testing.assert_array_equal(da["xt"], db["xt"])
            self.assertNotIn("outer_only", db["state"]["context"].categories)
