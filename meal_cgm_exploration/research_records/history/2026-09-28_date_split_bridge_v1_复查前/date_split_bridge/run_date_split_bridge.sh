#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONDONTWRITEBYTECODE=1
PYTHON_BIN="${HPP_MEAL_PYTHON:-python3}"
exec "$PYTHON_BIN" "$SCRIPT_DIR/run_analysis.py" "$@"
