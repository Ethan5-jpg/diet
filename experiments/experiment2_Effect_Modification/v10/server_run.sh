#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
hpp_v10_python=${HPP_EXP2_PYTHON:-}
if [[ -z "$hpp_v10_python" ]]; then
    hpp_v10_python=$(cat ../../experiments/diet_effect_modification_20261109_v12/.python_path)
fi
exec "$hpp_v10_python" -B -u run_v10.py "$@"
