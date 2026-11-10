#!/usr/bin/env bash
set -euo pipefail
cd /home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_model_expansion_20261107_v7
mkdir -p /home/ec2-user/.venvs
/home/ec2-user/miniconda/envs/pheno/bin/python -m venv --system-site-packages /home/ec2-user/.venvs/hpp_v7_20261107
/home/ec2-user/.venvs/hpp_v7_20261107/bin/python -m pip install --no-deps --only-binary=:all: -r requirements.txt
/home/ec2-user/.venvs/hpp_v7_20261107/bin/python launch.py --console --run-id main_v7_20261107 --threads 4
