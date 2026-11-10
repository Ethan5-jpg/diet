"""Read-only access to the frozen V12 background and matched ridge code."""
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

_CACHE = None


def modules():
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    root = Path(__file__).resolve().parent.parent
    before = sys.path[:]
    sys.path.insert(0, str(root))
    try:
        values = {}
        for name in ('upload_check', 'bridge', 'features', 'models', 'engine', 'runtime'):
            value = importlib.import_module(name)
            if Path(value.__file__).resolve().parent != root:
                raise RuntimeError('Frozen module name collision: ' + name)
            values[name] = value
    finally:
        sys.path[:] = before
    values.update(root=root, config=json.loads((root/'config.json').read_text()))
    _CACHE = SimpleNamespace(**values)
    return _CACHE
