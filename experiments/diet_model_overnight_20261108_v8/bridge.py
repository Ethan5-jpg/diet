"""Frozen discovery adapter and read-only, matched historical comparators."""

import json
import sys
from pathlib import Path

import numpy as np

V6 = Path(__file__).resolve().parent.parent / "diet_model_batch_20261107_v6"
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


def load(package):
    """The legacy adapter imports pilot code; contain its search-path mutation."""
    previous = sys.path[:]
    try:
        return _frozen_load(package)
    finally:
        sys.path[:] = previous


class Reference:
    """Old outer predictions enter evaluation only, never model selection."""

    def __init__(self, package, v6_run, v7_run):
        self.paths, self.files = [], {}
        for label, folder, run_id, member in (
            ("v6_fusion", V6.name, v6_run, "fusion"),
            ("v7_ridge_stack", "diet_model_expansion_20261107_v7", v7_run, "ridge_stack"),
        ):
            root = Path(package) / "experiments" / folder / "runs" / run_id
            registration, summary = root / "registration.json", root / "summary.json"
            config = json.loads(registration.read_text())["config"]
            if (
                any(
                    config.get(k) != v
                    for k, v in dict(
                        outcome="D_Glu",
                        exposure="fiber_density",
                        outer_seed=20261104,
                        confirmation_outcomes_read=False,
                        outer_repeats=2,
                        outer_folds=5,
                    ).items()
                )
                or json.loads(summary.read_text()).get("status") != "COMPLETE"
            ):
                raise ValueError("Historical comparator is incomplete or mismatched")
            self.paths.extend([registration, summary])
            for r in range(2):
                for f in range(5):
                    unit = root / "TRE_ONLY" / ("r%d_f%d_result" % (r, f))
                    if unit.is_symlink():
                        raise ValueError("Historical checkpoint directory must not be a symlink")
                    valid = []
                    for receipt in sorted(unit.glob("done_*.json")):
                        try:
                            value = json.loads(receipt.read_text())
                        except json.JSONDecodeError:
                            continue
                        path = unit / value["file"]
                        if Path(value["file"]).name != value["file"] or path.is_symlink():
                            raise ValueError("Invalid historical checkpoint receipt")
                        if sha(path) != value["sha256"]:
                            raise ValueError("Historical checkpoint checksum mismatch")
                        valid.append(path)
                        self.paths.extend([receipt, path])
                    if len(valid) != 1:
                        raise ValueError("Expected exactly one completed historical outer unit")
                    self.files[label, r, f] = (valid[0], member)

    def get(self, repeat, fold, rows, y):
        import pickle

        out = {}
        for label in ("v6_fusion", "v7_ridge_stack"):
            path, member = self.files[label, repeat, fold]
            record = pickle.loads(path.read_bytes())
            if not np.array_equal(record["rows"], rows) or not np.array_equal(record["y"], y):
                raise ValueError("Historical comparator rows or targets differ")
            prediction = record["predictions"][member]
            if prediction.shape != y.shape or not np.isfinite(prediction).all():
                raise ValueError("Invalid historical prediction")
            out[label] = prediction
        return out
