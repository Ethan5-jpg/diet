#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
hpp_v6_log="v6_server_$(date +%Y%m%d_%H%M%S).log"
{
    bash server_run.sh --preflight &&
    bash server_run.sh --audit &&
    bash server_run.sh --threads "${HPP_V6_THREADS:-2}"
} 2>&1 | tee "$hpp_v6_log"
