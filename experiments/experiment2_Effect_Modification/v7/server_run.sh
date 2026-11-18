#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
hpp_v7_python=${HPP_EXP2_PYTHON:-}
if [[ -z "$hpp_v7_python" ]]; then
    hpp_v7_python=$(cat ../../experiments/diet_effect_modification_20261109_v12/.python_path)
fi
exec "$hpp_v7_python" -B -u run_v7.py "$@"
