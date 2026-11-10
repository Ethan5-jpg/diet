"""Frozen role utilities, discovery loader and evaluation-only prior predictions."""

import importlib.util
import json
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


class Reference:
    """Historical outer predictions never enter any training or gating features."""

    def __init__(self, package, v8_run, v9_run):
        self.paths, self.files = [], {}
        for folder, run_id, members in (
            (V8.name, v8_run, {"v8_ridge_wide": "ridge_wide", "v8_ridge_joint": "ridge_joint"}),
            ("diet_model_adaptive_20261108_v9", v9_run, {"v9_convex_wide": "convex_wide"}),
        ):
            root = Path(package) / "experiments" / folder / "runs" / run_id
            registration, summary = root / "registration.json", root / "summary.json"
            config = json.loads(registration.read_text())["config"]
            expected = dict(
                outcome="D_Glu",
                exposure="fiber_density",
                outer_seed=20261104,
                confirmation_outcomes_read=False,
                outer_repeats=2,
                outer_folds=5,
            )
            if (
                any(config.get(k) != v for k, v in expected.items())
                or json.loads(summary.read_text()).get("status") != "COMPLETE"
            ):
                raise ValueError("Historical comparator is incomplete or mismatched")
            self.paths.extend([registration, summary])
            for r in range(2):
                for f in range(5):
                    unit = root / "TRE_ONLY" / ("r%d_f%d_result" % (r, f))
                    if unit.is_symlink():
                        raise ValueError("Historical directory must not be a symlink")
                    valid = []
                    for receipt in sorted(unit.glob("done_*.json")):
                        try:
                            value = json.loads(receipt.read_text())
                        except json.JSONDecodeError:
                            continue
                        path = unit / value["file"]
                        if (
                            Path(value["file"]).name != value["file"]
                            or path.is_symlink()
                            or sha(path) != value["sha256"]
                        ):
                            raise ValueError("Invalid historical checkpoint receipt")
                        valid.append(path)
                        self.paths.extend([receipt, path])
                    if len(valid) != 1:
                        raise ValueError("Expected exactly one completed historical unit")
                    for label, member in members.items():
                        self.files[label, r, f] = valid[0], member

    def get(self, repeat, fold, rows, y):
        import pickle

        out, cache = {}, {}
        for label in ("v8_ridge_wide", "v8_ridge_joint", "v9_convex_wide"):
            path, member = self.files[label, repeat, fold]
            if path not in cache:
                cache[path] = pickle.loads(path.read_bytes())
            record = cache[path]
            if not np.array_equal(record["rows"], rows) or not np.array_equal(record["y"], y):
                raise ValueError("Historical rows or targets differ")
            p = record["predictions"][member]
            if p.shape != y.shape or not np.isfinite(p).all():
                raise ValueError("Invalid historical prediction")
            out[label] = p
        return out
