#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${HPP_MEAL_PYTHON:-python3}"
exec "$PYTHON_BIN" "$PACKAGE_DIR/scripts/run_pipeline.py" "$@"
