#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
hpp_v9_log="v9_server_$(date +%Y%m%d_%H%M%S)_$$.log"
{
    bash server_run.sh --preflight --threads "${HPP_V9_THREADS:-2}" &&
    bash server_run.sh --threads "${HPP_V9_THREADS:-2}"
} 2>&1 | tee "$hpp_v9_log"
