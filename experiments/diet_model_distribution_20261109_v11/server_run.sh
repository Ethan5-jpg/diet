#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
bash setup.sh
IFS= read -r hpp_python < .python_path
exec "$hpp_python" launch.py --console --run-id distribution_v11_20261109 --threads 4 "$@"
