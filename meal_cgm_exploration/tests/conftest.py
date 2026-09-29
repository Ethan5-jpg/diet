import json
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(PACKAGE / "scripts"))


@pytest.fixture
def config():
    return json.loads((PACKAGE / "config" / "default.json").read_text())
