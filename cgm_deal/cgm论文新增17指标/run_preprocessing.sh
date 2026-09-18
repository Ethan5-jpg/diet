#!/usr/bin/env bash
set -euo pipefail
CGM17_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONIOENCODING=utf-8
exec "$PYTHON_BIN" "$CGM17_DIR/scripts/process_cgm17.py" "$@"
