"""The only real-data adapter: discovery inputs and the existing v4 preprocessing API."""

import importlib.util
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from numerics import site_kernel


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@dataclass
class Split:
    ht: np.ndarray
    hv: np.ndarray
    ot: np.ndarray
    ov: np.ndarray
    gt: np.ndarray
    gv: np.ndarray
    yt: np.ndarray
    yv: np.ndarray
    dt: np.ndarray
    dv: np.ndarray
    rows: np.ndarray
    train_rows: np.ndarray
    delta: np.ndarray
    preprocessing: dict
    cache: dict = field(default_factory=dict)
    base: dict = field(default_factory=dict)

    def kernels(self, kind, seed):
        if kind not in self.cache:
            self.cache[kind] = (
                site_kernel(self.ot, self.ov, kind, seed),
                site_kernel(self.gt, self.gv, kind, seed),
            )
        return self.cache[kind]

    def microbes(self, site):
        if site == "oral":
            return self.ot, self.ov
        return np.column_stack([self.ot, self.gt]), np.column_stack([self.ov, self.gv])

    def snapshot(self):
        return dict(
            preprocessing=self.preprocessing,
            train_rows=self.train_rows,
            host=self.ht,
            oral=self.ot,
            gut=self.gt,
        )


@dataclass
class Dataset:
    index: object
    raw: dict
    taxa: dict
    y: np.ndarray
    api: object
    reference: dict
    paths: list

    def split(self, train, valid, contrast):
        self.api.separated(
            self.index.split_unit_id.astype(str).to_numpy(), train, valid
        )
        designs, y, tr, va, state = self.api.context(
            self.index, self.raw, self.taxa, self.y, train, valid
        )
        host = designs["host"]
        oral = designs["oral"][:, host.shape[1] :]
        gut = designs["oral_gut"][:, designs["oral"].shape[1] :]
        diet = state["diet"]
        u = (self.index.fiber_density.to_numpy(float) - diet["center"]) / diet["scale"]
        h = self.api.rcs(u, diet["knots"])
        low, high = (np.asarray(contrast) - diet["center"]) / diet["scale"]
        delta = np.array(
            [
                high - low,
                self.api.rcs(high, diet["knots"]) - self.api.rcs(low, diet["knots"]),
            ]
        )
        return Split(
            host[tr],
            host[va],
            oral[tr],
            oral[va],
            gut[tr],
            gut[va],
            y[tr],
            y[va],
            np.column_stack([u[tr], h[tr]]),
            np.column_stack([u[va], h[va]]),
            np.flatnonzero(va),
            np.flatnonzero(tr),
            delta,
            state,
        )


def load(package):
    package = Path(package)
    experiments, inputs = package / "experiments", package / "inputs/discovery_v2"
    v4_path = experiments / "diet_nested_stability_20261103_v4/run_nested.py"
    api = module("batch_frozen_preprocessing", v4_path)
    pilot = experiments / "diet_response_heterogeneity_20261103_v1_attempt02/code"
    v3 = experiments / "diet_representation_screen_20261103_v3"
    api.dependencies(package, pilot, v3)
    paths = [
        inputs / "discovery_covariates_food_TRE_ONLY.parquet",
        inputs / "D_Glu_raw_Y_TRE_ONLY.parquet",
    ]
    x = api.read_table(paths[0])
    if (
        len(x) != 4427
        or not x.final_role.eq("discovery").all()
        or not x.participant_id.is_unique
    ):
        raise ValueError("Expected the discovery-only cohort")
    if not np.array_equal(x.input_row, np.arange(len(x))):
        raise ValueError("Invalid discovery row order")
    if (
        x.split_unit_id.isna().any()
        or x.groupby("split_unit_id").cv_fold.nunique().max() != 1
    ):
        raise ValueError("Invalid household partition")
    outcomes = api.read_table(paths[1])
    rows = outcomes.input_row.to_numpy(int)
    if len(set(rows)) != len(rows) or not ((rows >= 0) & (rows < len(x))).all():
        raise ValueError("Invalid outcome row alignment")
    if outcomes.Y_unit.dropna().unique().tolist() != ["mg/dL"]:
        raise ValueError("Unexpected outcome unit")
    for key in ("participant_id", "cohort", "research_stage"):
        if key in outcomes:
            if not np.array_equal(
                outcomes[key].to_numpy(), x.iloc[rows][key].to_numpy()
            ):
                raise ValueError("Outcome alignment mismatch in " + key)
    y = np.full(len(x), np.nan)
    y[rows] = outcomes.Y_value.to_numpy(float)
    raw, taxa = {}, {}
    for site in ("oral", "gut"):
        path, dictionary = (
            inputs / (site + "_raw_TRE_ONLY.npz"),
            inputs / (site + "_taxa.json"),
        )
        paths.extend([path, dictionary])
        taxa[site] = json.loads(dictionary.read_text())
        with np.load(io.BytesIO(path.read_bytes()), allow_pickle=False) as archive:
            raw[site] = archive["abundance"]
        if raw[site].shape != (len(x), len(taxa[site])):
            raise ValueError("Microbial alignment mismatch")
    reference_path = (
        experiments
        / "diet_nested_stability_20261103_v4/results/TRE_ONLY/final_predictions.npz"
    )
    with np.load(
        io.BytesIO(reference_path.read_bytes()), allow_pickle=False
    ) as archive:
        reference = {
            name: archive[key]
            for name, key in [
                ("y", "y"),
                ("outer", "outer"),
                ("v4_hgb_oral", "hist_boost_oral"),
                ("v4_hgb_oral_gut", "hist_boost_oral_gut"),
            ]
        }
    outer = np.stack(
        [
            x.cv_fold.to_numpy(int),
            api.folds(x.split_unit_id.astype(str).to_numpy(), 5, 20261104),
        ]
    )
    if not np.array_equal(outer, reference["outer"]):
        raise ValueError("Reference household allocations differ")
    paths.extend(
        [
            reference_path,
            v4_path,
            pilot / "core.py",
            pilot / "run_pilot.py",
            v3 / "run_screen.py",
            package / "preprocessing/code/preprocessing_v1_20261030.py",
        ]
    )
    return Dataset(x, raw, taxa, y, api, reference, paths)
