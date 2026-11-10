#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
hpp_python=${HPP_V14_PYTHON:-/home/ec2-user/.venvs/hpp_v8_20261108/bin/python}
exec "$hpp_python" resources.py
