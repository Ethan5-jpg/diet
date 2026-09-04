#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/scripts/00_build_formal_diet_cgm_cohort.py"
python "$SCRIPT_DIR/scripts/01_current4_diet_cgm_model0.py"
python "$SCRIPT_DIR/scripts/02_current4_diet_cgm_model2.py"
python "$SCRIPT_DIR/scripts/03_current4_diet_cgm_model3_bmi.py"
