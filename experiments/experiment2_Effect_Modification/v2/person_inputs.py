"""Read the V11 discovery sources and replay its V4 training-role transforms."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


V11 = "diet_model_distribution_20261109_v11"
V11_RUN = "distribution_v11_20261109"
V10 = "diet_model_context_20261108_v10"
V6 = "diet_model_batch_20261107_v6"
V4 = "diet_nested_stability_20261103_v4"
V1 = "diet_response_heterogeneity_20261103_v1_attempt02"
V3 = "diet_representation_screen_20261103_v3"
N_DISCOVERY = 4427
N_PEOPLE = 3343
N_HOUSEHOLDS = 3059
OUTER_SEED = 20261104
CONTRAST = (10, 15)


class SourceError(ValueError):
    """Historical source or validation contract did not match."""


def require(ok, detail):
    if not ok:
        raise SourceError(detail)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def registered_sources(package):
    """An exact allowlist of sources used by V11's original preparation."""
    exp = package / "experiments"
    table = package / "inputs/discovery_v2"
    raw = (
        table / "discovery_covariates_food_TRE_ONLY.parquet",
        table / "D_Glu_raw_Y_TRE_ONLY.parquet",
        table / "oral_raw_TRE_ONLY.npz",
        table / "oral_taxa.json",
        table / "gut_raw_TRE_ONLY.npz",
        table / "gut_taxa.json",
        exp / V4 / "results/TRE_ONLY/final_predictions.npz",
        exp / V4 / "run_nested.py",
        exp / V1 / "code/core.py",
        exp / V1 / "code/run_pilot.py",
        exp / V3 / "run_screen.py",
        package / "preprocessing/code/preprocessing_v1_20261030.py",
    )
    context = (
        package / "inputs/daily/food_days_discovery_TRE_ONLY.parquet",
        package.parent / "analysis/shared/food_day_qc_TRE_ONLY.parquet",
        package.parent / "analysis/shared/covariate_registry_TRE_ONLY.parquet",
    )
    helpers = (exp / V6 / "data.py", exp / V6 / "numerics.py", exp / V10 / "context.py")
    return raw, context, helpers


def read_anchor(package):
    run = package / "experiments" / V11 / "runs" / V11_RUN
    registration, summary = run / "registration.json", run / "summary.json"
    require(registration.is_file() and summary.is_file(), "Missing completed V11 registration or summary")
    reg = read_json(registration)
    config = reg["config"]
    expected = dict(version="v11_20261109", exposure="fiber_density", outcome="D_Glu",
                    confirmation_outcomes_read=False, outer_seed=OUTER_SEED,
                    outer_repeats=2, outer_folds=5, inner_folds=3, contrast=list(CONTRAST))
    require(all(config.get(key) == value for key, value in expected.items()), "V11 design identity changed")
    report = read_json(summary)
    require(report["status"] == "COMPLETE" and report["confirmation_outcomes_read"] is False,
            "V11 report is not a completed discovery report")
    old = sorted((row for row in report["scores"] if row["model"] == "positive_combined"),
                 key=lambda row: row["repeat"])
    require(len(old) == 2 and [row["repeat"] for row in old] == [0, 1]
            and all(row["N"] == N_PEOPLE and np.isfinite(row["R2"]) for row in old),
            "V11 historical summary evaluation contract changed")

    raw, context, helpers = registered_sources(package)
    hashes = {registration: digest(registration), summary: digest(summary)}
    for path in (*raw, *context, *helpers):
        expected_hash = reg["hashes"].get(str(path))
        require(isinstance(expected_hash, str) and path.is_file()
                and digest(path) == expected_hash, "V11 registered source mismatch: " + path.name)
        hashes[path] = expected_hash
    return raw, context, helpers, hashes, tuple(float(row["R2"]) for row in old)


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_frozen_data(package, helper):
    """Use the exact archived loader without bringing in the V11 tree libraries."""
    path_before = sys.path[:]
    bare_modules = ("numerics", "run_pilot", "core")
    cached = {name: sys.modules.pop(name, None) for name in bare_modules}
    try:
        sys.path.insert(0, str(helper.parent))
        data = imported("hpp_v3_v6_frozen_adapter", helper).load(package)
        expected = (
            helper.parent / "numerics.py",
            package / "experiments" / V1 / "code/run_pilot.py",
            package / "experiments" / V1 / "code/core.py",
        )
        require(all(name in sys.modules and Path(sys.modules[name].__file__).resolve() == path.resolve()
                    for name, path in zip(bare_modules, expected, strict=True)),
                "Historical adapter import origin mismatch")
    finally:
        sys.path[:] = path_before
        for name, previous in cached.items():
            sys.modules.pop(name, None)
            if previous is not None:
                sys.modules[name] = previous
    return data


class PersonSource:
    """Frozen index, context and fold-local role builder for matched evaluation."""

    def __init__(self, data, extra, hashes, historical_scores):
        self._data = data
        self.index = data.index
        self.raw = data.raw
        self.taxa = data.taxa
        self.y = data.y
        self.api = data.api
        self.groups = data.index.split_unit_id.astype(str).to_numpy()
        self.reference = {"y": data.reference["y"], "outer": data.reference["outer"]}
        self.extra = extra
        self.source_hashes = hashes
        self.historical_scores = historical_scores

    def verify_sources(self):
        for path, expected in self.source_hashes.items():
            require(path.is_file() and digest(path) == expected,
                    "A historical source changed during this run: " + path.name)

    def split(self, train, valid, contrast=CONTRAST):
        train, valid = np.asarray(train), np.asarray(valid)
        require(train.shape == valid.shape == (len(self.index),)
                and train.dtype == valid.dtype == np.dtype(bool)
                and train.any() and valid.any(), "Training/validation role is empty or malformed")
        require(not np.any(train & valid) and not set(self.groups[train]) & set(self.groups[valid]),
                "A household crossed training and validation")
        result = self._data.split(train, valid, contrast)
        require(train[result.train_rows].all() and valid[result.rows].all(),
                "Historical transform selected a row outside its role")
        return result

    def outer_split(self, repeat, fold):
        require(repeat in range(2) and fold in range(5), "Invalid historical fold")
        assignment = self.reference["outer"][repeat]
        valid = assignment == fold
        result = self.split(~valid, valid)
        expected = np.flatnonzero(valid & np.isfinite(self.reference["y"][repeat]))
        require(np.array_equal(result.rows, expected) and np.array_equal(
            result.yv, self.reference["y"][repeat, expected]),
            "Outer evaluated rows or transformed targets differ from the V11/V4 anchor")
        return result

    def roles(self, repeat, fold):
        require(repeat in range(2) and fold in range(5), "Invalid historical fold")
        train = self.reference["outer"][repeat] != fold
        inner = np.full(len(train), -1, dtype=int)
        inner[train] = self.api.folds(self.groups[train], 3, OUTER_SEED + 100 * repeat + fold + 1)
        roles = tuple(self.split(train & (inner != k), train & (inner == k)) for k in range(3))
        return roles, self.outer_split(repeat, fold)


def load_discovery(root):
    """Load only existing discovery inputs under the uploaded v3's handoff."""
    package = Path(root).resolve().parent.parent
    raw_paths, context_paths, helpers, hashes, historical = read_anchor(package)
    data = read_frozen_data(package, helpers[0])
    require(set(data.paths) == set(raw_paths), "Discovery adapter source list changed")
    groups = data.index.split_unit_id
    y, outer = data.reference["y"], data.reference["outer"]
    require(len(data.index) == N_DISCOVERY and y.shape == outer.shape == (2, N_DISCOVERY)
            and np.issubdtype(outer.dtype, np.integer) and not np.isinf(y).any(),
            "Frozen V11/V4 index or target shape changed")
    eligible = np.isfinite(y)
    require(np.array_equal(eligible[0], eligible[1]) and int(eligible[0].sum()) == N_PEOPLE
            and groups[eligible[0]].nunique() == N_HOUSEHOLDS
            and np.isfinite(data.y[eligible[0]]).all(),
            "Frozen people, households or original outcome availability changed")
    for assignment in outer:
        require(set(assignment) == set(range(5)), "Missing historical household fold")
        for fold in range(5):
            valid = assignment == fold
            require(np.any(valid & eligible[0]), "A historical outer fold has no eligible participants")
            data.api.separated(groups.astype(str).to_numpy(), ~valid, valid)

    context = imported("hpp_v3_v10_context", helpers[2])
    extra, loaded = context.read_context(package, data.index, eligible[0])
    require(set(loaded) == set(context_paths) and extra["audit"]["discovery_people"] == N_PEOPLE
            and extra["audit"]["daily_outcome_columns_read"] == []
            and extra["blocks"]["full"].shape == (N_DISCOVERY, 7)
            and len(extra["alcohol"]) == N_DISCOVERY,
            "V10/V11 discovery dietary context changed")
    for site in ("oral", "gut"):
        require(len(data.taxa[site]) == len(set(data.taxa[site])), "Duplicated species columns")
    return PersonSource(data, extra, hashes, historical)
