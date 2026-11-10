#!/usr/bin/env bash
set -euo pipefail
cd /home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_model_expansion_20261107_v7
/home/ec2-user/.venvs/hpp_v7_20261107/bin/python -m pip install --no-deps --only-binary=:all: \
  numpy==1.26.4 scipy==1.11.4 pandas==2.1.0 pyarrow==15.0.2 \
  scikit-learn==1.4.1.post1 lightgbm==3.3.5 interpret-core==0.6.5
/home/ec2-user/.venvs/hpp_v7_20261107/bin/python launch.py --console --run-id main_v7_20261107 --threads 4
