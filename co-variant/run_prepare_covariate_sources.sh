#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" \
  "${PROJECT_ROOT}/scripts/01_prepare_covariate_sources.py" \
  --source-root "/home/ec2-user/studies/hpp_datasets" \
  --project-root "${PROJECT_ROOT}" \
  "$@"
