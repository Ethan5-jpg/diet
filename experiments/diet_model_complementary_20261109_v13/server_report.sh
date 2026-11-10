#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
hpp_python=${HPP_V13_PYTHON:-/home/ec2-user/.venvs/hpp_v8_20261108/bin/python}
if [[ -f .python_path ]]; then IFS= read -r hpp_python < .python_path; fi
exec "$hpp_python" report.py --run-id complementary_v13_20261109 "$@"
