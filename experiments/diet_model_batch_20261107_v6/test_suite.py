"""Run synthetic tests only; never open participant input files."""

import unittest
from pathlib import Path

from threadpoolctl import threadpool_limits

if __name__ == "__main__":
    tests = unittest.defaultTestLoader.discover(
        str(Path(__file__).resolve().parent / "tests")
    )
    with threadpool_limits(limits=1):
        result = unittest.TextTestRunner(verbosity=2).run(tests)
    raise SystemExit(0 if result.wasSuccessful() else 1)
