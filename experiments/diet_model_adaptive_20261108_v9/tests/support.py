"""Synthetic data and the actual frozen adapter; no server tables are opened."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bridge import V6, module
from daily import aggregate_food

ROOT = Path(__file__).resolve().parents[1]


def fixture(wide=False):
    helper = module("v9_synthetic_frozen", V6 / "tests/support.py")
    previous = sys.path[:]
    try:
        data, _ = helper.fixture()
    finally:
        sys.path[:] = previous
    data.index["cohort"], data.index["research_stage"] = "10k", "00_00_visit"
    rng = np.random.default_rng(916)
    if wide:
        raw = {}
        for site, p in (("oral", 144), ("gut", 176)):
            x = rng.gamma(0.7, 1, size=(len(data.index), p))
            x[rng.random(x.shape) < 0.7] = 0
            x[:, 0] += 0.01
            raw[site] = x / x.sum(1, keepdims=True)
        data.raw = raw
        data.taxa = {site: ["taxon_%d" % j for j in range(a.shape[1])] for site, a in raw.items()}
    rows = []
    for person in data.index.itertuples():
        for day in range(4):
            energy = 1600 + 100 * day
            density = person.fiber_density + day - 1.5
            rows.append(
                dict(
                    participant_id=person.participant_id,
                    cohort="10k",
                    research_stage="00_00_visit",
                    date=pd.Timestamp("2026-01-02") + pd.Timedelta(days=day),
                    energy_kcal=energy,
                    fiber_g=density * energy / 1000,
                    energy_eligible=True,
                    fiber_eligible=True,
                )
            )
    data.extra = aggregate_food(pd.DataFrame(rows), data.index)
    config = json.loads((ROOT / "config.json").read_text())
    return data, config


def reduced(config):
    cfg = copy.deepcopy(config)
    cfg.update(completion_iterations=2, neural_seeds=1, meta_tree_iterations=3)
    for settings in cfg["families"].values():
        case = settings["grid"][0].copy()
        for name, value in dict(
            count=6, components=3, iterations=3, pairs=4, hidden=8, epochs=3, layers=2
        ).items():
            if name in case:
                case[name] = value
        settings["grid"] = [case]
    cfg["interaction_penalties"] = [0.01]
    return cfg


class FakeReference:
    paths = []

    def get(self, repeat, fold, rows, y):
        return {
            n: np.asarray(y) + offset
            for n, offset in (
                ("v6_fusion", 0.4),
                ("v7_ridge_stack", 0.35),
                ("v8_ridge_wide", 0.3),
                ("v8_ridge_joint", 0.32),
            )
        }
