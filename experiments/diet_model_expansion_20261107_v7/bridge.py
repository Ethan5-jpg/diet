"""Reuse frozen V6 preprocessing and read completed references without refitting."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

V6 = Path(__file__).resolve().parent.parent / "diet_model_batch_20261107_v6"
sys.path.insert(0, str(V6))
try:
    from data import load as load
    from data import module as module
    from numerics import metric as metric
    from storage import Memo as Memo
    from storage import sha
    from storage import write_json as write_json
finally:
    sys.path.pop(0)


class Reference:
    """V6 OOF predictions are evaluation comparators, never learner features."""

    def __init__(self, package, run_id):
        root = Path(package) / "experiments" / V6.name / "runs" / run_id
        registration = root / "registration.json"
        config = json.loads(registration.read_text())["config"]
        if (config["outcome"], config["exposure"], config["outer_seed"]) != (
            "D_Glu",
            "fiber_density",
            20261104,
        ) or config["confirmation_outcomes_read"]:
            raise ValueError("V6 reference does not match the discovery target")
        if (
            not (root / "summary.json").is_file()
            or json.loads((root / "summary.json").read_text()).get("status") != "COMPLETE"
        ):
            raise ValueError("A completed V6 reference is required")
        self.files, self.paths = {}, [registration, root / "summary.json"]
        for repeat in range(2):
            for fold in range(5):
                folder = root / "TRE_ONLY" / ("r%d_f%d_result" % (repeat, fold))
                valid = []
                for receipt in sorted(folder.glob("done_*.json")):
                    try:
                        record = json.loads(receipt.read_text())
                    except json.JSONDecodeError:
                        continue
                    path = folder / record["file"]
                    if Path(record["file"]).name != record["file"] or path.is_symlink():
                        raise ValueError("Invalid V6 reference receipt")
                    if sha(path) != record["sha256"]:
                        raise ValueError("V6 reference checksum mismatch")
                    valid.append(path)
                    self.paths.extend([receipt, path])
                if len(valid) != 1:
                    raise ValueError("Expected one completed V6 outer reference")
                self.files[(repeat, fold)] = valid[0]

    def get(self, repeat, fold, rows, y):
        record = pickle.loads(self.files[(repeat, fold)].read_bytes())
        if not np.array_equal(record["rows"], rows) or not np.array_equal(record["y"], y):
            raise ValueError("V6 reference rows or targets differ")
        prediction = record["predictions"]["fusion"]
        if prediction.shape != y.shape or not np.isfinite(prediction).all():
            raise ValueError("Invalid V6 fusion reference")
        return prediction
