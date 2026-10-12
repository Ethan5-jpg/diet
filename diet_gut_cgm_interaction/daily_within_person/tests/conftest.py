import json
from pathlib import Path
import sys

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))


@pytest.fixture
def config():
    return json.loads((PACKAGE / "config" / "analysis.json").read_text())
