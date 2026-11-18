#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -n "${HPP_EXP2_PYTHON:-}" ]]; then
    hpp_exp2_python=$HPP_EXP2_PYTHON
else
    IFS= read -r hpp_exp2_python < ../../experiments/diet_effect_modification_20261109_v12/.python_path
fi
exec "$hpp_exp2_python" -B -u run_v6.py "$@"
