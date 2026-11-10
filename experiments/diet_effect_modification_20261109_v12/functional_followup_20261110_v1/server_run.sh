#!/usr/bin/env bash
set -euo pipefail
hpp_function_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
hpp_function_python=python
if [ -s "$hpp_function_dir/../.python_path" ]; then
    IFS= read -r hpp_function_python < "$hpp_function_dir/../.python_path"
fi
cd "$hpp_function_dir"
export PYTHONDONTWRITEBYTECODE=1
"$hpp_function_python" -B - <<'PY'
import hashlib,json
from pathlib import Path
root=Path.cwd()
manifest=json.loads((root/'manifest.json').read_text())
for name,expected in manifest['files'].items():
    path=root/name
    if Path(name).name != name or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit('FUNCTIONAL_UPLOAD_CHECK_FAILED')
print('FUNCTIONAL_UPLOAD_CHECK_OK files=%d' % len(manifest['files']))
PY
hpp_function_report_only=false
for hpp_function_arg in "$@"; do
    if [ "$hpp_function_arg" = --report ]; then
        hpp_function_report_only=true
    fi
done
if [ "$hpp_function_report_only" = false ]; then
    "$hpp_function_python" -B check_synthetic.py
fi
"$hpp_function_python" -B -u pilot.py "$@"
