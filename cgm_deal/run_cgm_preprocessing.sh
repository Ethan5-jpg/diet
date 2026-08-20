#!/usr/bin/env bash

set -euo pipefail

WORKFLOW_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
HPP_ROOT="$(CDPATH= cd -- "$WORKFLOW_DIR/../.." && pwd)"
SOURCE_DIR="${CGM_SOURCE_DIR:-$HPP_ROOT/Data/Transfer/cgm}"
OUTPUT_DIR="${CGM_OUTPUT_DIR:-$WORKFLOW_DIR/outputs}"
DATA_DIR="${CGM_DATA_DIR:-$OUTPUT_DIR/data}"
REPORT_DIR="${CGM_REPORT_DIR:-$OUTPUT_DIR/reports}"
LOG_DIR="${CGM_LOG_DIR:-$OUTPUT_DIR/logs}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "$DATA_DIR" "$REPORT_DIR" "$LOG_DIR"
RUN_STAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$LOG_DIR/cgm_preprocessing_${RUN_STAMP}.log"

{
    echo "CGM preprocessing started"
    echo "SOURCE_DIR=$SOURCE_DIR"
    echo "OUTPUT_DIR=$OUTPUT_DIR"
    echo "DATA_DIR=$DATA_DIR"
    echo "REPORT_DIR=$REPORT_DIR"
    echo "PYTHON_BIN=$PYTHON_BIN"

    "$PYTHON_BIN" "$WORKFLOW_DIR/scripts/01_prepare_baseline_connections.py" \
        --cgm-csv "$SOURCE_DIR/cgm.csv" \
        --daily-csv "$SOURCE_DIR/iglu_daily.csv" \
        --data-dir "$DATA_DIR" \
        --report-dir "$REPORT_DIR"

    "$PYTHON_BIN" "$WORKFLOW_DIR/scripts/02_cgm_connection_qc.py" \
        --selected-csv "$DATA_DIR/01_baseline_cgm_connections.csv" \
        --iglu-csv "$SOURCE_DIR/iglu.csv" \
        --data-dir "$DATA_DIR" \
        --report-dir "$REPORT_DIR"

    "$PYTHON_BIN" "$WORKFLOW_DIR/scripts/03_extract_iglu_phenotypes.py" \
        --qc-pass-csv "$DATA_DIR/02_cgm_qc_pass_all.csv" \
        --data-dir "$DATA_DIR" \
        --report-dir "$REPORT_DIR"

    "$PYTHON_BIN" "$WORKFLOW_DIR/scripts/04_clean_cgm_phenotypes.py" \
        --raw-csv "$DATA_DIR/03_iglu_core_raw.csv" \
        --data-dir "$DATA_DIR" \
        --report-dir "$REPORT_DIR"

    "$PYTHON_BIN" "$WORKFLOW_DIR/scripts/05_standardize_cgm_phenotypes.py" \
        --clean-csv "$DATA_DIR/04_cgm_core_clean.csv" \
        --data-dir "$DATA_DIR" \
        --report-dir "$REPORT_DIR"

    echo "CGM preprocessing completed"
    echo "FINAL_FILE=$DATA_DIR/05_cgm_core_phenotypes.csv"
    echo "LOG_FILE=$LOG_FILE"
} 2>&1 | tee "$LOG_FILE"
