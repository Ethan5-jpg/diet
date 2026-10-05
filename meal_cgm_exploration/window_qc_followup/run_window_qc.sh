#!/usr/bin/env bash
set -euo pipefail
MODULE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${HPP_MEAL_PYTHON:-python3}" "$MODULE_DIR/run_analysis.py" "$@"
