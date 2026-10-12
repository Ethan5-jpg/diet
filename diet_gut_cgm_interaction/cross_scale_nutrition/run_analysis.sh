#!/usr/bin/env bash
set -euo pipefail
ANALYSIS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python}" -B "$ANALYSIS_DIR/run_analysis.py" "$@"
