"""Synthetic complete/partial food logs with the actual frozen preprocessor."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bridge import V6, module
from context import ALCOHOL, KEY, LIFE, NUTRIENTS, aggregate
from registry import REFERENCES

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
        for site, width in (("oral", 144), ("gut", 176)):
            x = rng.gamma(0.7, 1, (len(data.index), width))
            x[rng.random(x.shape) < 0.7] = 0
            x[:, 0] += 0.01
            raw[site] = x / x.sum(1, keepdims=True)
        data.raw = raw
        data.taxa = {site: ["taxon_%d" % j for j in range(a.shape[1])] for site, a in raw.items()}
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
    cfg["interaction_grid"] = [[3, 0.01]]
    return cfg


class FakeReference:
    paths = []

    def get(self, repeat, fold, rows, y):
        return {n: np.asarray(y) + 0.3 + 0.02 * j for j, n in enumerate(REFERENCES)}
