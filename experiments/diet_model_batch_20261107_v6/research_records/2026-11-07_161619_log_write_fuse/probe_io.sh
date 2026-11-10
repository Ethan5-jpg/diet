#!/usr/bin/env bash
timeout -k 2s 10s bash -c '
set -e
hpp_probe=$(mktemp "/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_model_batch_20261107_v6/io_probe.XXXXXX")
printf "CREATED %s\n" "$hpp_probe"
printf "HPP_V6_IO_PROBE\n" > "$hpp_probe"
printf "WRITE_CLOSED_OK\n"
cat "$hpp_probe"
printf "PROBE_DONE\n"
'
printf 'probe_exit=%s\n' "$?"
