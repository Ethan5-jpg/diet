#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPP_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONDONTWRITEBYTECODE=1

"${PYTHON_BIN}" \
  "${PROJECT_ROOT}/scripts/02_build_covariate_master.py" \
  --project-root "${PROJECT_ROOT}" \
  --csv-dir "${PROJECT_ROOT}/csv" \
  --baseline-cgm-csv "${HPP_ROOT}/Data/cgm_deal/outputs/data/01_baseline_cgm_connections.csv" \
  --alcohol-csv "${HPP_ROOT}/Data/diet_deal/outputs/05_diet_scores/amed/amed_participant_scores.csv" \
  --output-dir "${PROJECT_ROOT}/outputs" \
  --cohort "10k" \
  --research-stage "00_00_visit" \
  "$@"
