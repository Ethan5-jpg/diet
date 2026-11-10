#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
hpp_python=${HPP_V10_PYTHON:-/home/ec2-user/.venvs/hpp_v8_20261108/bin/python}
if [[ ! -x "$hpp_python" ]]; then
  printf 'Compatible native Python missing: %s\nSet HPP_V10_PYTHON to the verified V8 environment.\n' "$hpp_python" >&2
  exit 1
fi
"$hpp_python" upload_check.py
"$hpp_python" - <<'PY'
import json
from pathlib import Path
from runtime import check_config, environment
config = json.loads(Path('config.json').read_text())
check_config(config)
versions = environment(config)
print('DEPENDENCIES_OK ' + ' '.join('%s=%s' % v for v in versions.items()))
PY
printf '%s\n' "$hpp_python" > .python_path
printf 'ENV_READY=%s; no packages changed.\n' "$hpp_python"
