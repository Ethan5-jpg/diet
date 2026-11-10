"""Synthetic cohort; uses the original frozen preprocessing, never real data."""

import copy
import json
import sys
from pathlib import Path

import numpy as np

from bridge import V6, module
from features import UPF_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


def fixture(wide=False):
    helper = module("v8_frozen_synthetic_fixture", V6 / "tests/support.py")
    previous = sys.path[:]
    try:
        data, _ = helper.fixture()
    finally:
        sys.path[:] = previous
    rng = np.random.default_rng(117)
    lower = rng.uniform(0.05, 0.6, len(data.index))
    data.index[UPF_COLUMNS[0]] = lower
    data.index[UPF_COLUMNS[1]] = np.minimum(lower + 0.15, 1.0)
    data.index.loc[::9, list(UPF_COLUMNS)] = np.nan
    if wide:
        data.raw = {
            site: rng.dirichlet(np.ones(p), len(data.index)) for site, p in (("oral", 144), ("gut", 176))
        }
        data.taxa = {site: ["synthetic_%d" % i for i in range(a.shape[1])] for site, a in data.raw.items()}
    config = json.loads((ROOT / "config.json").read_text())
    return data, config


def reduced(config):
    cfg = copy.deepcopy(config)
    cfg.update(nmf_iterations=60, fm_iterations=40, sparse_iterations=100)
    for settings in cfg["families"].values():
        case = settings["grid"][0].copy()
        for key, value in dict(
            count=6, components=3, iterations=3, trees=6, rotations=2, cycles=1, neighbors=10
        ).items():
            if key in case:
                case[key] = value
        settings["grid"] = [case]
    cfg["sparse_fractions"], cfg["interaction_penalties"] = [0.1], [0.1]
    return cfg


class FakeReference:
    paths = []

    def get(self, repeat, fold, rows, y):
        return dict(v6_fusion=np.asarray(y) + 0.4, v7_ridge_stack=np.asarray(y) + 0.35)
