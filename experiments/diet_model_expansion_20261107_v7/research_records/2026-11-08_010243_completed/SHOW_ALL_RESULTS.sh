#!/usr/bin/env bash
set -euo pipefail
cd /home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_model_expansion_20261107_v7
/home/ec2-user/.venvs/hpp_v7_20261107/bin/python report.py --run-id main_v7_20261107 --all
