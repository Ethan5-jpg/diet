"""Run the native feature sizes and full penalty grids on generated profiles only."""

import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from bridge import Memo
from engine import Engine
from report import aggregate, screen
from synthetic import Dataset

ROOT = Path(__file__).resolve().parent


def run():
    start = time.monotonic()
    config = json.loads((ROOT / "config.json").read_text())
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.py")}
    data = Dataset(n=1400, seed=20261112, taxa_count=64)
    with tempfile.TemporaryDirectory(prefix="hpp_v12_full_grid_") as temporary:
        memo = Memo(Path(temporary))
        engine = Engine(config, memo)
        completed = {(r, f): engine.checkpoint(data, r, f) for r in range(2) for f in range(5)}
        result = aggregate(completed, config, data.index.split_unit_id.to_numpy(), "SYNTHETIC_FULL_GRID_ONLY")
        json.dumps(result, allow_nan=False)
        replay_error = max(float(np.max(abs(model.predict() - part["prediction"][name])))
                           for part in completed.values() for name, model in part["models"].items())
        if replay_error > 1e-10 or any(v["status"] != "OK" for v in result["inference"]):
            raise AssertionError("Full-grid inference/replay check failed")
        if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest for name, digest in hashes.items()):
            raise RuntimeError("Full-grid source changed during verification")
        record = dict(status="PASS", evidence_source="SYNTHETIC_FULL_GRID_ONLY", n=1400,
                      taxa_per_site=64, actual_pc_per_site=30, outer_units=len(completed), workflows=13,
                      source_sha256=hashes, config=config, replay_max_error=replay_error,
                      main_columns=[p["main_columns"] for p in completed.values()],
                      inference_nuisance_ranks=[p["nuisance_rank"] for p in completed.values()],
                      seconds=time.monotonic() - start, native_inputs_read=False)
        folder = ROOT / "research_records/verification"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "full_grid.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        (folder / "synthetic_report_preview.txt").write_text("\n".join(screen(result, mode) for mode in (1, 2, 3)))
        print("SYNTHETIC_FULL_GRID_PASS outer=10 workflows=13 replay_error=%g seconds=%.2f" % (replay_error, record["seconds"]))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
