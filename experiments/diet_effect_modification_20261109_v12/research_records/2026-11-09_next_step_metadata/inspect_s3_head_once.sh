#!/usr/bin/env bash
# One existing-AWS-CLI HeadObject request; no installs, listing or downloads.
hpp_meta_settings=/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_effect_modification_20261109_v12/.python_path
hpp_meta_python=python
if [ -s "$hpp_meta_settings" ]; then
    IFS= read -r hpp_meta_python < "$hpp_meta_settings"
fi
"$hpp_meta_python" -B - "$@" <<'PY'
import json, os, shutil, signal, subprocess, sys
from pathlib import Path
from urllib.parse import urlsplit, unquote
import pyarrow.parquet as pq

def expired(signum, frame):
    print('S3_HEAD_CHECK_INCOMPLETE; no retry.', flush=True)
    raise SystemExit(0)

signal.signal(signal.SIGALRM, expired)
signal.alarm(20)
print('S3_HEAD_ONCE: existing CLI/access; no object listing, body reads or writes.', flush=True)
aws = shutil.which('aws')
if aws is None:
    print('AWS_CLI_UNAVAILABLE; ask data provider for the TRE read method.', flush=True)
    raise SystemExit(0)
R = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/home/ec2-user/studies/hpp_datasets')
field = 'humann_gene_familites_gene_level_parquet'
try:
    with pq.ParquetFile(R/'oral_microbiome/oral_microbiome.parquet', pre_buffer=False) as pf:
        values = pf.read(columns=[field], use_threads=False).column(0).to_pylist()
    raw = next(value.strip() for value in values if isinstance(value, str) and value.strip().lower().startswith('s3://'))
    uri = urlsplit(raw)
    if uri.username or uri.password or uri.query or uri.fragment or not uri.netloc or not uri.path.lstrip('/'):
        raise ValueError('unsupported URI structure')
except (OSError, ValueError, TypeError, StopIteration):
    print('SOURCE_POINTER_UNAVAILABLE; no request made.', flush=True)
    raise SystemExit(0)

task_env = dict(os.environ)
task_env.update(AWS_MAX_ATTEMPTS='1', AWS_PAGER='')
args = [aws, '--cli-connect-timeout', '3', '--cli-read-timeout', '5',
        's3api', 'head-object', '--bucket', uri.netloc, '--key', unquote(uri.path.lstrip('/')),
        '--query', '{bytes:ContentLength,storage:StorageClass}', '--output', 'json']
try:
    result = subprocess.run(args, env=task_env, capture_output=True, text=True, timeout=12)
    if result.returncode == 0:
        metadata = json.loads(result.stdout)
        if not isinstance(metadata, dict) or not isinstance(metadata.get('bytes'), int) or metadata['bytes'] < 0:
            raise ValueError('unexpected object metadata')
        print('HEAD_OK bytes=' + str(metadata['bytes']), flush=True)
    else:
        message = result.stderr.lower()
        labels = [('accessdenied', 'ACCESS_DENIED_RESPONSE'), ('403', 'ACCESS_DENIED_RESPONSE'),
                  ('expiredtoken', 'EXPIRED_AUTH_RESPONSE'), ('unable to locate credentials', 'EXISTING_AUTH_UNAVAILABLE'),
                  ('nosuchkey', 'NOT_FOUND_RESPONSE'), ('404', 'NOT_FOUND_RESPONSE'),
                  ('redirect', 'REGION_OR_ENDPOINT_RESPONSE'), ('region', 'REGION_OR_ENDPOINT_RESPONSE'),
                  ('timed out', 'NETWORK_OR_TIMEOUT'), ('timeout', 'NETWORK_OR_TIMEOUT'),
                  ('could not connect', 'NETWORK_OR_ENDPOINT'), ('ssl', 'TLS_OR_ENDPOINT')]
        label = next((label for word, label in labels if word in message), 'OTHER_CLI_ERROR')
        print('HEAD_UNAVAILABLE', label, 'exit=' + str(result.returncode), flush=True)
except subprocess.TimeoutExpired:
    print('HEAD_PROCESS_TIMEOUT_12S; no retry.', flush=True)
except (OSError, ValueError, TypeError):
    print('HEAD_TOOL_UNAVAILABLE_OR_UNPARSEABLE; no retry.', flush=True)
print('S3_HEAD_ONCE_DONE', flush=True)
signal.alarm(0)
PY
