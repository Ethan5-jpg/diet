#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -n "${HPP_EXP2_PYTHON:-}" ]]; then
    hpp_moderation_python=$HPP_EXP2_PYTHON
else
    IFS= read -r hpp_moderation_python < ../../experiments/diet_effect_modification_20261109_v12/.python_path
fi
exec "$hpp_moderation_python" -B -X pycache_prefix=/dev/null -u run_v2.py "$@"
