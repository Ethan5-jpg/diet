"""Synthetic data only, using the actual frozen preprocessing implementation."""

import copy
import json
from pathlib import Path

import numpy as np

from bridge import V6, module

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    helper = module("v7_frozen_test_fixture", V6 / "tests/support.py")
    dataset, _ = helper.fixture()
    config = json.loads((ROOT / "config.json").read_text())
    config.update(fm_iterations=40, residual_trees=8)
    for grid in config["grids"].values():
        for case in grid:
            if "iterations" in case:
                case["iterations"] = 4
            if "count" in case:
                case["count"] = min(case["count"], 6) if case["count"] else 0
            if "landmarks" in case:
                case["landmarks"] = 12
            if "components" in case:
                case["components"] = min(case["components"], 3)
    for case in config["mixture_grid"]:
        case["count"] = 6
    for case in config["additive_grid"]:
        case.update(iterations=3, count=6, cycles=1)
    return dataset, config


def reduced_config(config):
    value = copy.deepcopy(config)
    value["grids"] = {name: grid[:1] for name, grid in value["grids"].items()}
    value["mixture_grid"] = value["mixture_grid"][:1]
    value["additive_grid"] = value["additive_grid"][:1]
    value["interaction_penalties"] = [0.1]
    return value


class FakeReference:
    paths = []

    def get(self, repeat, fold, rows, y):
        return np.asarray(y) + 0.4
