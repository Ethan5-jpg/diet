#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_PYTHON="${PYTHON:-python3}"
mkdir -p "$TASK_DIR/logs"
TASK_LOG="$TASK_DIR/logs/run_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="$TASK_DIR/logs/matplotlib_cache"
{
  echo "STEP19B_PACKAGE=$TASK_DIR"
  echo "PYTHON=$TASK_PYTHON"
  "$TASK_PYTHON" - "$@" <<'PY'
import importlib
import sys
if sys.version_info < (3, 8):
    print('DEPENDENCY_CHECK=FAIL: Python 3.8+ required')
    raise SystemExit(2)
required = ['numpy', 'pandas']
if not any(x in sys.argv for x in ['--no-plots', '--check-only', '--inventory-only']):
    required.append('matplotlib')
missing = []
for name in required:
    try:
        mod = importlib.import_module(name)
        print('{}={}'.format(name, mod.__version__))
    except ImportError:
        missing.append(name)
if missing:
    print('DEPENDENCY_CHECK=FAIL: missing ' + ', '.join(missing))
    print('Use the Python environment used for previous HPP analyses.')
    print('Or install requirements.txt in a separate virtual environment; see README.md.')
    raise SystemExit(2)
print('DEPENDENCY_CHECK=PASS')
PY
  "$TASK_PYTHON" -u "$TASK_DIR/scripts/19b_core_taxa_direction_synthesis.py" "$@"
} 2>&1 | tee "$TASK_LOG"
