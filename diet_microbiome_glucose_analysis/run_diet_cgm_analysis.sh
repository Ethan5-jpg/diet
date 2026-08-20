#!/usr/bin/env bash

set -euo pipefail

ANALYSIS_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
DATA_ROOT="$(CDPATH= cd -- "$ANALYSIS_DIR/.." && pwd)"

AMED_CSV="${AMED_CSV:-$DATA_ROOT/diet_deal/outputs/05_diet_scores/amed/amed_participant_scores.csv}"
HPDI_CSV="${HPDI_CSV:-$DATA_ROOT/diet_deal/outputs/05_diet_scores/hpdi/hpdi_participant_scores.csv}"
CGM_CSV="${CGM_CSV:-$DATA_ROOT/cgm_deal/outputs/data/05_cgm_core_phenotypes.csv}"
GUT_CSV="${GUT_CSV:-}"

OUTPUT_DIR="${ANALYSIS_OUTPUT_DIR:-$ANALYSIS_DIR/outputs}"
DATA_DIR="${ANALYSIS_DATA_DIR:-$OUTPUT_DIR/data}"
REPORT_DIR="${ANALYSIS_REPORT_DIR:-$OUTPUT_DIR/reports}"
MODEL_DIR="${ANALYSIS_MODEL_DIR:-$OUTPUT_DIR/models}"
LOG_DIR="${ANALYSIS_LOG_DIR:-$OUTPUT_DIR/logs}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "$DATA_DIR" "$REPORT_DIR" "$MODEL_DIR" "$LOG_DIR"
RUN_STAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$LOG_DIR/diet_cgm_analysis_${RUN_STAMP}.log"

ALIGN_ARGS=(
    --amed-csv "$AMED_CSV"
    --hpdi-csv "$HPDI_CSV"
    --cgm-csv "$CGM_CSV"
    --data-dir "$DATA_DIR"
    --report-dir "$REPORT_DIR"
)
if [[ -n "$GUT_CSV" ]]; then
    ALIGN_ARGS+=(--gut-csv "$GUT_CSV")
fi

{
    echo "Diet-CGM analysis started"
    echo "AMED_CSV=$AMED_CSV"
    echo "HPDI_CSV=$HPDI_CSV"
    echo "CGM_CSV=$CGM_CSV"
    echo "GUT_MEMBERSHIP_EVALUATED=$([[ -n "$GUT_CSV" ]] && echo True || echo False)"
    echo "OUTPUT_DIR=$OUTPUT_DIR"
    echo "PYTHON_BIN=$PYTHON_BIN"

    "$PYTHON_BIN" "$ANALYSIS_DIR/scripts/00_align_diet_cgm.py" "${ALIGN_ARGS[@]}"

    "$PYTHON_BIN" "$ANALYSIS_DIR/scripts/01_unadjusted_diet_cgm_models.py" \
        --aligned-csv "$DATA_DIR/00_diet_cgm_aligned.csv" \
        --model-dir "$MODEL_DIR" \
        --report-dir "$REPORT_DIR"

    echo "Diet-CGM analysis completed"
    echo "ALIGNED_FILE=$DATA_DIR/00_diet_cgm_aligned.csv"
    echo "MODEL_FILE=$MODEL_DIR/01_unadjusted_diet_cgm_models.csv"
    echo "LOG_FILE=$LOG_FILE"
} 2>&1 | tee "$LOG_FILE"
