#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -n "${HPP_EXP2_PYTHON:-}" ]]; then
  hpp_experiment_python=$HPP_EXP2_PYTHON
elif [[ -x /home/ec2-user/miniconda/envs/pheno/bin/python ]]; then
  hpp_experiment_python=/home/ec2-user/miniconda/envs/pheno/bin/python
elif [[ -x /home/ec2-user/.venvs/hpp_v8_20261108/bin/python ]]; then
  hpp_experiment_python=/home/ec2-user/.venvs/hpp_v8_20261108/bin/python
else
  hpp_experiment_python=python
fi
umask 077
# -B alone still reads .pyc files. A non-directory cache prefix forces
# source imports so the executed adapters match the files we hash.
exec "$hpp_experiment_python" -B -X pycache_prefix=/dev/null -u exp2_run.py "$@"
