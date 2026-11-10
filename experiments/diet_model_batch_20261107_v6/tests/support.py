"""Synthetic discovery analogue and the original preprocessing implementation."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data import Dataset, module
from storage import sha


def fixture():
    package = Path(__file__).resolve().parents[3]
    api = module(
        "batch_synthetic_old_api",
        package / "experiments/diet_nested_stability_20261103_v4/run_nested.py",
    )
    path = package / "preprocessing/code/preprocessing_v1_20261030.py"
    if not path.exists():
        path = package / "02_ALL_LOCAL_SOURCE/scripts/preprocessing_v1_20261030.py"
    if sha(path) != api.PREPROCESS_SHA:
        raise ValueError(
            "Synthetic fixture must use the same frozen preprocessing source"
        )
    preprocessing = module("batch_synthetic_preprocessor", path)
    pilot = (
        package / "experiments/diet_response_heterogeneity_20261103_v1_attempt02/code"
    )
    if not pilot.exists():
        pilot = package / "experiments/diet_response_heterogeneity_20261103_v1"
    sys.path.insert(0, str(pilot))
    host = module("batch_synthetic_host", pilot / "run_pilot.py")
    api.MicrobiomeCLR, api.PhenotypeOutliers = (
        preprocessing.MicrobiomeCLR,
        preprocessing.PhenotypeOutliers,
    )
    api.host_design, api.rcs = host.host_design, host.rcs
    rng, n = np.random.default_rng(392), 360
    x = pd.DataFrame(
        dict(
            participant_id=np.arange(n),
            split_unit_id=np.repeat(np.arange(n // 2), 2),
            n_fiber_days=4,
            fiber_density=rng.uniform(5, 25, n),
            age_calendar_years=rng.uniform(30, 70, n),
            sex_code=rng.integers(0, 2, n),
            mean_daily_energy_fiber=rng.uniform(1500, 2500, n),
            smoking_status="never",
        )
    )
    raw = {site: rng.dirichlet(np.ones(12), n) for site in ("oral", "gut")}
    for value in raw.values():
        value[::3, 0] = 0
        value /= value.sum(1)[:, None]
    taxa = {site: [str(i) for i in range(12)] for site in raw}
    y = (
        90
        + 0.1 * x.age_calendar_years.to_numpy()
        + 0.15 * x.fiber_density.to_numpy()
        + rng.normal(0, 4, n)
    )
    groups = x.split_unit_id.astype(str).to_numpy()
    outer = np.stack([api.folds(groups, 5, 41), api.folds(groups, 5, 20261104)])
    reference = dict(outer=outer, y=np.full((2, n), np.nan))
    dataset = Dataset(x, raw, taxa, y, api, reference, [])
    for repeat in range(2):
        for fold in range(5):
            split = dataset.split(
                outer[repeat] != fold, outer[repeat] == fold, [10.0, 15.0]
            )
            reference["y"][repeat, split.rows] = split.yv
    reference["v4_hgb_oral"] = reference["y"] + 0.2
    reference["v4_hgb_oral_gut"] = reference["y"] + 0.3
    config = json.loads(
        (Path(__file__).resolve().parents[1] / "config.json").read_text()
    )
    config.update(
        top_k=[4, 0],
        gut_weights=[0.1],
        main_penalties=[0.1, 0.001],
        interaction_weights=[0.0, 1.0],
        pls_components=[1, 2],
    )
    config["tree_grids"] = dict(
        catboost=[dict(depth=2, iterations=5, l2_leaf_reg=20.0)],
        xgboost=[
            dict(max_depth=2, n_estimators=5, min_child_weight=5, reg_lambda=10.0)
        ],
        hgb=[
            dict(
                max_leaf_nodes=7,
                max_iter=5,
                min_samples_leaf=15,
                l2_regularization=10.0,
            )
        ],
    )
    return dataset, config


def changed_outer_labels(dataset, repeat=0, fold=0):
    result = copy.copy(dataset)
    for name in ("index", "raw", "taxa", "y", "reference"):
        setattr(result, name, copy.deepcopy(getattr(dataset, name)))
    valid = result.reference["outer"][repeat] == fold
    result.y[valid] += 0.05
    split = result.split(~valid, valid, [10.0, 15.0])
    result.reference["y"][repeat, split.rows] = split.yv
    return result
