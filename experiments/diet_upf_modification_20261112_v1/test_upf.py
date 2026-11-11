from hashlib import sha256
import json
import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from sources import Cohort, calibration, checked_import, integrate
from upf import BASE, ROOT, analyze, dose_design, pool


CFG = json.loads((ROOT / "config.json").read_bytes())
INPUTS, MODELS, IO = checked_import(BASE, CFG["base_code"], ("inputs", "models", "run"),
                                    lambda path: sha256(path.read_bytes()).hexdigest())


class SimpleCLR:
    """Synthetic-only full-rank preprocessing for exercising the model loop."""

    def fit(self, x, taxa, ids, *, site):
        self.center = np.log(x).mean(axis=0)
        return self

    def transform(self, x, taxa, *, site):
        return dict(values=np.log(x) - self.center, supported=np.ones(len(x), bool))


def fixture():
    rng = np.random.default_rng(728)
    n = 220
    people = pd.DataFrame(dict(participant_id=np.arange(n), split_unit_id=np.arange(n) // 2,
                               age_calendar_years=rng.uniform(35, 65, n), sex_code=rng.integers(0, 2, n),
                               mean_daily_energy_UPF=rng.uniform(1100, 2500, n),
                               smoking_status=np.where(np.arange(n) % 3 == 0, "yes", "no")))
    raw = {s: rng.dirichlet(np.ones(16), size=n) for s in ("gut", "oral")}
    g = raw["gut"]
    marker = -(g * np.log(g)).sum(axis=1)
    marker = (marker - marker.mean()) / marker.std()
    lower = rng.uniform(0.15, 0.58, n)
    weights = sparse.csr_matrix(np.column_stack([rng.uniform(0.02, 0.15, n), rng.uniform(0.02, 0.15, n)]))
    upper = lower + np.asarray(weights.sum(axis=1)).ravel()
    upf = lower + np.asarray(weights @ np.array([0.6, 0.2])).ravel()
    y = 92 + 2.5 * (upf - 0.4) / 0.1 * marker + rng.normal(0, 1.0, n)
    ref = pd.DataFrame(dict(food_id=np.arange(100), label_group=np.arange(100),
                            y=np.arange(100) % 2, category="same"))
    presence = pd.DataFrame(dict(food_id=np.arange(100), input_row=np.arange(100)))
    unknown = pd.DataFrame(dict(category=["one", "two"]))
    return Cohort(people, np.arange(n), y, np.empty((2, n)), raw, {s: list(range(16)) for s in raw},
                  lower, upper, weights, presence, ref, unknown, rng.normal(size=(n, 4)), {})


def group_draws(groups, categories, labels, weights, unknown, *, draws, seed, shrinkage):
    return np.tile([0.6, 0.2], (draws, 1)), dict(distinct_reference_groups=len(groups))


class Tests(unittest.TestCase):
    def test_shared_food_label(self):
        low, high = np.array([.1, .2]), np.array([.4, .5])
        weights = sparse.csr_matrix([[.3], [.3]])
        mu, draws = integrate(low, high, weights, np.full((128, 1), .5), seed=2)
        np.testing.assert_allclose(mu, [.25, .35])
        np.testing.assert_allclose(draws[:, 1] - draws[:, 0], .1)
        d = dose_design(mu, draws, low, high, .3)
        np.testing.assert_allclose(d["integrated"][:, 0], d["expectation"][:, 0])
        self.assertTrue(np.all(d["integrated"][:, 1] > d["expectation"][:, 1]))

    def test_training_reference_mask(self):
        data = fixture()
        seen = []

        def watch(groups, *args, **kwargs):
            seen.extend(groups)
            return group_draws(groups, *args, **kwargs)

        train = np.arange(len(data.index)) % 4 != 0
        calibration(data, train, watch, dict(draws=64), seed=17)
        self.assertEqual(set(seen), set(np.flatnonzero(train[:100])))

    def test_effect_and_prediction(self):
        cfg = {**CFG, "draws": 64, "scenario_fits": 4}
        data = fixture()
        train = data.index.split_unit_id.to_numpy() % 5 != 0
        part = analyze(data, train, cfg, group_draws, SimpleCLR, MODELS, seed=35)
        self.assertEqual(len(part["results"]), 128)
        pred = pool([dict(part, repeat=0)], float(np.sum((data.y[~train] - data.y.mean()) ** 2)))
        self.assertGreater(pred[CFG["primary"] + ":0.03"]["r0"]["delta_r2"], .7)
        direct = analyze(data, np.ones(len(data.y), bool), cfg, group_draws, SimpleCLR, MODELS, seed=35, direct=True)
        self.assertEqual(len(direct["results"]), 64)
        self.assertGreater(direct["results"][CFG["primary"]]["ci95"][0], 1.0)
        self.assertLess(direct["results"][CFG["primary"]]["holm_p"], .001)
        self.assertEqual(len(direct["label_scenario_contrast_quantiles"]), 3)


if __name__ == "__main__":
    unittest.main()
