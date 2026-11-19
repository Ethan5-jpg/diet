#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
hpp_full_pheno_python=${HPP_FULL_PHENO_PYTHON:-${HPP_EXP2_PYTHON:-}}
if [[ -z "$hpp_full_pheno_python" ]]; then
    hpp_full_pheno_pointer=../../../experiments/diet_effect_modification_20261109_v12/.python_path
    if [[ ! -r "$hpp_full_pheno_pointer" ]]; then
        echo "FULL_PHENO_V1_STOP pinned Python path not found; set HPP_FULL_PHENO_PYTHON" >&2
        exit 1
    fi
    hpp_full_pheno_python=$(< "$hpp_full_pheno_pointer")
fi
exec "$hpp_full_pheno_python" -B -u run.py "$@"
