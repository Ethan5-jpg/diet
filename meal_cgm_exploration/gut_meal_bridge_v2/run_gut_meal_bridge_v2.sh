#!/usr/bin/env bash
set -euo pipefail
HPP_GUT_BRIDGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_BIN:-python3}" "$HPP_GUT_BRIDGE_DIR/run_analysis.py" "$@"
