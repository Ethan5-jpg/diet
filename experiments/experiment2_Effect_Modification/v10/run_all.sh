#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
hpp_v10_log="v10_server_$(date +%Y%m%d_%H%M%S)_$$.log"
{
    bash server_run.sh --preflight --threads "${HPP_V10_THREADS:-2}" &&
    bash server_run.sh --threads "${HPP_V10_THREADS:-2}"
} 2>&1 | tee "$hpp_v10_log"
