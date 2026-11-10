#!/usr/bin/env bash
set -euo pipefail
cd /home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_model_distribution_20261109_v11
bash server_report.sh
bash server_report.sh --diagnostics
