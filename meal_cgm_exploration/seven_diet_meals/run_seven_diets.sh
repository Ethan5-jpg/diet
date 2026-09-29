#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${HPP_MEAL_PYTHON:-python3}"
exec "$PYTHON_BIN" "$SCRIPT_DIR/run_analysis.py" "$@"
