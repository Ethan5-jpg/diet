#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
base_python=/home/ec2-user/miniconda/envs/pheno/bin/python
"$base_python" -m venv --system-site-packages .venv
.venv/bin/python -m pip install --no-deps --only-binary=:all: -r requirements.txt
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from runtime import check_config, environment
config = json.loads(Path('config.json').read_text())
check_config(config)
versions = environment(config)
print('DEPENDENCIES_OK ' + ' '.join('%s=%s' % item for item in versions.items()))
PY
if ! command -v tmux >/dev/null 2>&1; then
    printf '%s\n' 'TMUX_MISSING: install with sudo yum install -y tmux before launch.'
fi
