"""Synthetic complete/partial food logs with the actual frozen preprocessor."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bridge import V6, module
from context import ALCOHOL, KEY, LIFE, NUTRIENTS, aggregate
from fusion import FusionFit
from registry import bank_names

ROOT = Path(__file__).resolve().parents[1]


def tables(index):
    rng = np.random.default_rng(1010)
    food, nutrients, lifestyle = [], [], []
    for person in index.itertuples():
        key = dict(zip(KEY, [person.participant_id, person.cohort, person.research_stage]))
        density = rng.uniform([70, 25, 30, 1000], [130, 65, 70, 2500])
        for day in range(6):
            energy = person.mean_daily_energy_fiber + 100 * (day - 2.5)
            fiber = (person.fiber_density + day - 2.5) * energy / 1000
            identity = dict(key, date=pd.Timestamp("2026-01-02") + pd.Timedelta(days=day))
            food.append(
                dict(identity, energy_kcal=energy, fiber_g=fiber, energy_eligible=True, fiber_eligible=True)
            )
            value = density * (1 + 0.03 * (day - 2.5)) * energy / 1000
            nutrients.append(
                dict(
                    identity,
                    calories_kcal_complete_sum=energy,
                    dietary_fiber_g_complete_sum=fiber,
                    **{n + "_complete_sum": v for n, v in zip(NUTRIENTS, value)},
                )
            )
        life = rng.uniform([0, 0, 5], [7, 7, 9])
        if person.participant_id % 7 == 0:
            life[0] = np.nan
        lifestyle.append(
            dict(
                key,
                **dict(zip(LIFE, life)),
                **{
                    ALCOHOL: None
                    if person.participant_id % 9 == 0
                    else ["never", "monthly", "weekly"][person.participant_id % 3]
                },
            )
        )
    return pd.DataFrame(food), pd.DataFrame(nutrients), pd.DataFrame(lifestyle)


def fixture(wide=False):
    helper = module("v10_synthetic_frozen", V6 / "tests/support.py")
    previous = sys.path[:]
    try:
        data, _ = helper.fixture()
    finally:
        sys.path[:] = previous
    data.index["cohort"], data.index["research_stage"] = "10k", "00_00_visit"
    data.index["n_fiber_days"] = 6
    if wide:
        rng, raw = np.random.default_rng(1011), {}
        for site, width in (("oral", 320), ("gut", 384)):
            x = rng.gamma(0.7, 1, (len(data.index), width))
            x[rng.random(x.shape) < 0.7] = 0
            x[:, 0] += 0.01
            raw[site] = x / x.sum(1, keepdims=True)
        data.raw = raw
        data.taxa = {site: ["taxon_%d" % j for j in range(a.shape[1])] for site, a in raw.items()}
    data.index["UPF_lower_equalday_fraction"] = 0.2
    data.index["UPF_upper_equalday_fraction"] = 0.4
    data.extra = aggregate(*tables(data.index), data.index)
    return data, json.loads((ROOT / "config.json").read_text())


def reduced(config):
    cfg = copy.deepcopy(config)
    for settings in cfg["families"].values():
        case = settings["grid"][0].copy()
        for name, value in dict(count=6, components=3, iterations=3).items():
            if name in case:
                case[name] = value
        settings["grid"] = [case]
    cfg["meta_penalties"] = [0.01, 0.1]
    cfg["sparse_penalties"] = [0.001, 0.01]
    for settings in cfg["families"].values():
        for case in settings["grid"]:
            if "max_terms" in case:
                case["max_terms"] = 5
            if "inducing" in case:
                case["inducing"] = 12
            if "slices" in case:
                case["slices"] = 4
            if "bins" in case:
                case["bins"] = 3
            if "quantiles" in case:
                case["quantiles"] = 3
    for family in cfg["families"].values():
        for case in family["grid"]:
            case.update(pre_components=6, rff_features=24)
            if "trees" in case:
                case["trees"] = 8
            if "max_iter" in case:
                case["max_iter"] = 40
    for cases in cfg["nonlinear_meta_grid"].values():
        case = cases[0].copy()
        case["meta_components"] = 4
        if "inducing" in case:
            case["inducing"] = 12
        if "max_iter" in case:
            case["max_iter"] = 40
        cases[:] = [case]
    return cfg


class FakeReference:
    paths = []

    def __init__(self, config):
        self.config, self.fits = config, {}

    def values(self, role):
        host, y = role.split.hv, role.split.yt
        return {
            n: y.mean() + 0.2 * host[:, j % host.shape[1]] + 0.1 * np.sin(host[:, (j + 1) % host.shape[1]])
            for j, n in enumerate(bank_names(self.config))
        }

    def inner(self, repeat, fold, roles):
        values = [self.values(r) for r in roles]
        predictions = {n: np.concatenate([v[n] for v in values]) for n in values[0]}
        old = [n for n in predictions if n.startswith("bank__")]
        all_names = list(predictions)
        replay = dict(
            v11_replay=FusionFit(old, "convex", weights=np.full(len(old), 1 / len(old))),
            v13_replay=FusionFit(all_names, "convex", weights=np.full(len(all_names), 1 / len(all_names))),
        )
        return dict(predictions=predictions, choices={}, replay=replay)

    def get(self, repeat, fold, rows, y, choices, role=None):
        predictions = self.values(role)
        first = np.mean(
            np.column_stack([p for n, p in predictions.items() if n.startswith("bank__")]), axis=1
        )
        second = np.mean(np.column_stack(list(predictions.values())), axis=1)
        predictions.update(
            v11_positive_combined=first, v11_sparse_combined=first + 0.1, v13_positive_combined=second
        )
        return predictions, {n: {} for n in bank_names(self.config)}
