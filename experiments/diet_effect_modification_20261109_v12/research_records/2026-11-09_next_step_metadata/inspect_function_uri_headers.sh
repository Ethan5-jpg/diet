#!/usr/bin/env bash
# At most three exact-URI footer reads, using the server's existing reader/auth.
hpp_meta_settings=/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_effect_modification_20261109_v12/.python_path
hpp_meta_python=python
if [ -s "$hpp_meta_settings" ]; then
    IFS= read -r hpp_meta_python < "$hpp_meta_settings"
fi
"$hpp_meta_python" -B - "$@" <<'PY'
import os, re, signal, sys
from pathlib import Path
from urllib.parse import urlsplit
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq

def expired(signum, frame):
    print('TIME_LIMIT_REACHED; URI check incomplete.', flush=True)
    raise SystemExit(0)

def reason(error):
    message = str(error).lower()
    if any(word in message for word in ('accessdenied', 'access denied', 'permission', 'credential', 'unauthorized', 'forbidden', '403', 'expiredtoken')):
        return 'ACCESS_OR_AUTH_UNAVAILABLE'
    if any(word in message for word in ('nosuchkey', 'not found', 'no such file', '404')):
        return 'FILE_NOT_LOCATED'
    if any(word in message for word in ('region', 'redirect', '301')):
        return 'REGION_OR_ENDPOINT_UNRESOLVED'
    if any(word in message for word in ('timeout', 'timed out', 'resolve host', 'network', 'connect')):
        return 'NETWORK_OR_TIMEOUT'
    return type(error).__name__

signal.signal(signal.SIGALRM, expired)
signal.alarm(60)
R = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/home/ec2-user/studies/hpp_datasets')
allowed_root = Path(os.path.realpath(R))
sealed = re.compile(r'confirm|holdout|heldout|held_out|w22|w23', re.I)
bad = {'', 'undefined', 'none', 'null', 'nan', 'na', 'n/a', 'unknown', '<na>'}
print('FUNCTION_URI_HEADERS: exact source URIs; existing server access only.', flush=True)
print('At most 3 file footers; no pointer values, abundance rows, outcomes or writes.', flush=True)
index = R / 'oral_microbiome/oral_microbiome.parquet'
try:
    with pq.ParquetFile(index, pre_buffer=False) as pf:
        fields = [f.name for f in pf.schema_arrow if 'humann' in f.name.lower() and (pa.types.is_string(f.type) or pa.types.is_large_string(f.type))]
        data = pf.read(columns=fields, use_threads=False).to_pydict() if fields else {}
except (OSError, ValueError, TypeError) as error:
    print('INDEX_UNAVAILABLE', reason(error), flush=True)
    raise SystemExit(0)

for group in ('gene', 'pathway_abundance', 'pathway_coverage'):
    candidates = sorted([name for name in fields if group in name.lower() and name.lower().endswith('parquet')], key=lambda name: ('microbe' in name.lower(), name))
    selected = next(((name, value.strip()) for name in candidates for value in data[name] if isinstance(value, str) and value.strip().lower() not in bad), None)
    if selected is None:
        print(group, 'NO_NONEMPTY_PARQUET_POINTER', flush=True)
        continue
    name, raw = selected
    try:
        parts = urlsplit(raw)
        scheme = parts.scheme.lower() or 'local'
        print('FIELD', name, 'scheme=' + scheme, flush=True)
        if scheme not in ('file', 's3', 's3fs', 'gs', 'gcs', 'local'):
            print('  SCHEME_NOT_ATTEMPTED_BY_THIS_READER', flush=True)
            continue
        if parts.username or parts.password or parts.query or parts.fragment:
            print('  URI_OPTIONS_PRESENT_NOT_ATTEMPTED', flush=True)
            continue
        if sealed.search(parts.path):
            print('  SEALED_NAMED_PATH_NOT_ATTEMPTED', flush=True)
            continue
        fs, path = pafs.FileSystem.from_uri(raw)
        if isinstance(fs, pafs.LocalFileSystem):
            local = Path(os.path.realpath(path))
            if not local.is_relative_to(allowed_root) or sealed.search(str(local)):
                print('  LOCAL_PATH_OUTSIDE_ALLOWED_SCOPE', flush=True)
                continue
        info = fs.get_file_info(path)
        if info.type != pafs.FileType.File:
            print('  NOT_A_READABLE_FILE', flush=True)
            continue
        with pq.ParquetFile(path, filesystem=fs, pre_buffer=False) as probe:
            print('  HEADER_OK rows=%d columns=%d bytes=%d' % (probe.metadata.num_rows, len(probe.schema_arrow), info.size), flush=True)
    except Exception as error:
        label = reason(error)
        print('  HEADER_UNAVAILABLE', label, flush=True)
        if label == 'ACCESS_OR_AUTH_UNAVAILABLE':
            print('  REMAINING_PROBES_SKIPPED_AFTER_ACCESS_FAILURE', flush=True)
            break
print('Header access does not establish cohort overlap, sample identity, units or functional QC.', flush=True)
print('FUNCTION_URI_HEADERS_DONE', flush=True)
signal.alarm(0)
PY
