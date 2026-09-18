#!/usr/bin/env bash
set -euo pipefail
CGM17_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONIOENCODING=utf-8
"$PYTHON_BIN" - "$CGM17_DIR" "$@" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
if len(sys.argv) > 2:
    report = Path(sys.argv[2]).expanduser() / 'reports/运行摘要.txt'
else:
    reports = sorted((root / 'outputs').glob('run_*/reports/运行摘要.txt'))
    if not reports:
        sys.exit('没有找到运行摘要；请先运行 bash run_preprocessing.sh')
    report = reports[-1]
print(report.read_text(encoding='utf-8'))
PY
