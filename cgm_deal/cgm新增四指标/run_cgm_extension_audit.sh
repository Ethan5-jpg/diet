#!/usr/bin/env bash
set -euo pipefail
EXTENSION_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
WORKFLOW_DIR="$(CDPATH= cd -- "$EXTENSION_DIR/.." && pwd)"
HPP_ROOT="$(CDPATH= cd -- "$WORKFLOW_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONIOENCODING=utf-8
exec "$PYTHON_BIN" "$EXTENSION_DIR/scripts/06_audit_cgm_extension.py" \
  --core-csv "${CGM_CORE_CSV:-$WORKFLOW_DIR/outputs/data/05_cgm_core_phenotypes.csv}" \
  --iglu-csv "${CGM_IGLU_CSV:-$HPP_ROOT/Data/Transfer/cgm/iglu.csv}" \
  --output-dir "${CGM_EXTENSION_OUTPUT_DIR:-$WORKFLOW_DIR/outputs/cgm_extension}" \
  "$@"
