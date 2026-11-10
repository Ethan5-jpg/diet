"""Frozen role utilities, discovery loader and evaluation-only prior predictions."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

V6 = Path(__file__).resolve().parent.parent / "diet_model_batch_20261107_v6"
V8 = V6.parent / "diet_model_overnight_20261108_v8"
sys.path.insert(0, str(V6))
try:
    from data import load as _frozen_load
    from data import module as module
    from numerics import metric as metric
    from storage import Memo as Memo
    from storage import sha as sha
    from storage import write_json as write_json
finally:
    sys.path.pop(0)

spec = importlib.util.spec_from_file_location("hpp_v8_role", V8 / "features.py")
role_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = role_module
spec.loader.exec_module(role_module)
BaseRole, screen = role_module.Role, role_module.screen


def load(package):
    previous = sys.path[:]
    try:
        data = _frozen_load(package)
    finally:
        sys.path[:] = previous
    from context import read_context

    masks = np.isfinite(data.reference["y"])
    if not np.array_equal(masks[0], masks[1]):
        raise ValueError("Expected the same fixed evaluation people in both repeats")
    data.extra, paths = read_context(Path(package), data.index, masks[0])
    data.paths.extend(paths)
    for site in ("oral", "gut"):
        if len(set(data.taxa[site])) != len(data.taxa[site]):
            raise ValueError("Taxon identifiers must be unique within site")
    return data


V10 = V6.parent / "diet_model_context_20261108_v10"


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


ContextCore = imported("hpp_v10_context", V10 / "context.py")
LegacyRole = imported("hpp_v10_features", V10 / "features.py").Role

from legacy import Reference as Reference
