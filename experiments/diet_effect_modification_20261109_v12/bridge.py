"""Read-only adapters to established discovery sources. Never use V11 predictions."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
V6 = ROOT.parent / "diet_model_batch_20261107_v6"
V8 = ROOT.parent / "diet_model_overnight_20261108_v8"
V10 = ROOT.parent / "diet_model_context_20261108_v10"
V11 = ROOT.parent / "diet_model_distribution_20261109_v11"


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(V6))
try:
    Frozen = imported("hpp_v12_frozen_data", V6 / "data.py")
    Storage = imported("hpp_v12_storage", V6 / "storage.py")
finally:
    sys.path.pop(0)
BaseRole = imported("hpp_v12_base_role", V8 / "features.py").Role
Context = imported("hpp_v12_context_source", V10 / "context.py")
sha, write_json, Memo = Storage.sha, Storage.write_json, Storage.Memo


def load(package, config):
    before = sys.path[:]
    try:
        dataset = Frozen.load(package)
    finally:
        sys.path[:] = before
    eligible = np.isfinite(dataset.reference["y"])
    if eligible.shape != (2, len(dataset.index)) or not np.array_equal(eligible[0], eligible[1]):
        raise ValueError("Expected identical evaluation people in both repeats")
    if int(eligible[0].sum()) != config["expected_people"]:
        raise ValueError("Discovery evaluation population changed")
    dataset.extra, paths = Context.read_context(Path(package), dataset.index, eligible[0])
    dataset.paths.extend(paths)
    for site in ("oral", "gut"):
        if len(set(dataset.taxa[site])) != len(dataset.taxa[site]):
            raise ValueError("Duplicate within-site taxon identifiers")
    return dataset


def adapter_sources():
    return [V6 / n for n in ("data.py", "numerics.py", "storage.py")] + [
        V8 / "features.py", V10 / "context.py"
    ]
