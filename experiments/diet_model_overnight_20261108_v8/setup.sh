#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
hpp_base_python=${HPP_BASE_PYTHON:-/home/ec2-user/miniconda/envs/pheno/bin/python}
hpp_env_dir=${HPP_VENV_DIR:-/home/ec2-user/.venvs/hpp_v8_20261108}
"$hpp_base_python" upload_check.py
mkdir -p -- "$(dirname -- "$hpp_env_dir")"
"$hpp_base_python" -m venv --system-site-packages "$hpp_env_dir"
hpp_env_python="$hpp_env_dir/bin/python"
"$hpp_env_python" -m pip install --no-deps --only-binary=:all: -r requirements.txt
"$hpp_env_python" - <<'PY'
import json
from pathlib import Path
from runtime import check_config, environment
config = json.loads(Path('config.json').read_text())
check_config(config)
versions = environment(config)
print('DEPENDENCIES_OK ' + ' '.join('%s=%s' % item for item in versions.items()))
PY
printf 'ENV_PYTHON=%s\n' "$hpp_env_python"
printf 'START: %s launch.py --console --run-id overnight_v8_20261108 --threads 4\n' "$hpp_env_python"
