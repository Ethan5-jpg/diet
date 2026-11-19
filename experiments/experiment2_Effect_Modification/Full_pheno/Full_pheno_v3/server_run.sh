#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
hpp_fp_python=${HPP_FULL_PHENO_PYTHON:-${HPP_EXP2_PYTHON:-}}
if [[ -z "$hpp_fp_python" ]]; then
    hpp_fp_pointer=../../../experiments/diet_effect_modification_20261109_v12/.python_path
    if [[ ! -r "$hpp_fp_pointer" ]]; then
        echo "FULL_PHENO_V3_STOP pinned Python path not found; set HPP_FULL_PHENO_PYTHON" >&2
        exit 1
    fi
    hpp_fp_python=$(< "$hpp_fp_pointer")
fi
exec "$hpp_fp_python" -B -u run_v3.py "$@"
