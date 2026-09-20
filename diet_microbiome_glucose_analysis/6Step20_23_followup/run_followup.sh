#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/hpp-followup-mpl-${UID}}"
"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 8):
    raise SystemExit('Python >= 3.8 required')
try:
    import numpy, pandas, matplotlib
except ImportError as e:
    raise SystemExit('Missing dependency: %s. Install requirements.txt using this Python.' % e)
print('PYTHON=' + sys.version.split()[0])
for m in [numpy, pandas, matplotlib]:
    print(m.__name__ + '=' + m.__version__)
print('DEPENDENCY_CHECK=PASS', flush=True)
PY
exec "$PYTHON_BIN" "$PACKAGE_DIR/scripts/run_followup.py" "$@"
