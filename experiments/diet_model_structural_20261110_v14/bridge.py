"""Frozen role utilities, discovery loader and evaluation-only prior predictions."""

import importlib.util
import hashlib
import json
import pickle
import re
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
    from storage import Memo as _FrozenMemo
    from storage import sha as sha
    from storage import write_json as write_json
finally:
    sys.path.pop(0)


class Memo(_FrozenMemo):
    """Require one valid closed receipt; read-only misses create no unit folder."""

    def get(self, key, compute=None):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", key) or key in (".", ".."):
            raise ValueError("Invalid checkpoint key")
        folder = self.root / key
        if folder.is_symlink():
            raise ValueError("Checkpoint directory must not be a symlink")
        values = []
        for receipt in sorted(folder.glob("done_*.json")):
            try:
                record = json.loads(receipt.read_text())
            except (ValueError, UnicodeError):
                continue
            name = record.get("file") if isinstance(record, dict) else None
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError("Invalid checkpoint receipt: " + key)
            path = folder / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("Checkpoint payload missing: " + key)
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != record.get("sha256"):
                raise ValueError("Corrupted checkpoint: " + key)
            values.append(payload)
        if len(values) > 1:
            raise ValueError("Multiple closed checkpoint receipts: " + key)
        if values:
            return pickle.loads(values[0])
        if compute is None:
            raise FileNotFoundError(key)
        return super().get(key, compute)


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

V11 = V6.parent / "diet_model_distribution_20261109_v11"
V11Role = imported("hpp_v11_role", V11 / "features.py").Role
V11Bank = imported("hpp_v11_bank", V11 / "legacy.py").Reference

V13 = V6.parent / "diet_model_complementary_20261109_v13"
SIRDirections = imported("hpp_v13_directions", V13 / "features.py").Directions
V13Reference = imported("hpp_v13_reference", V13 / "legacy.py").Reference

from legacy import Reference as Reference
