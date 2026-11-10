"""Synthetic checks only. No discovery tables or saved participant predictions read."""

import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import json
import sys
import tempfile
import unittest
from pathlib import Path

from threadpoolctl import threadpool_limits

from runtime import publish_json, publish_text


def storage_probe(root):
    """Exercise the experiment filesystem before any expensive real-data fit."""
    with tempfile.TemporaryDirectory(prefix=".v10_io_", dir=root) as folder:
        path = Path(folder) / "probe.json"
        publish_json(path, dict(probe=True))
        path.write_bytes(b'{"probe":')
        publish_json(path, dict(probe=True))
        if json.loads(path.read_text()) != dict(probe=True):
            raise RuntimeError("Experiment filesystem JSON readback failed")
        text = Path(folder) / "probe.txt"
        publish_text(text, "PROBE_OK\n")
        if text.read_text() != "PROBE_OK\n":
            raise RuntimeError("Experiment filesystem text readback failed")
    print("EXPERIMENT_STORAGE_OK", flush=True)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "tests"))
    suite = unittest.defaultTestLoader.discover(str(root / "tests"))
    with threadpool_limits(limits=1):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        storage_probe(root)
    raise SystemExit(0 if result.wasSuccessful() else 1)
