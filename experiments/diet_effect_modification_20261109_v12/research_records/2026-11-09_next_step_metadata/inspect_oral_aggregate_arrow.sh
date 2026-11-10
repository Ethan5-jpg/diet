#!/usr/bin/env bash
# Official aggregate fields; bounded local namespace/config candidates only.
hpp_meta_settings=/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_effect_modification_20261109_v12/.python_path
hpp_meta_python=python
if [ -s "$hpp_meta_settings" ]; then
    IFS= read -r hpp_meta_python < "$hpp_meta_settings"
fi
"$hpp_meta_python" -B - "$@" <<'PY'
import json, os, re, signal, sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit, unquote
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

def expired(signum, frame):
    print('AGGREGATE_CHECK_TIME_LIMIT; incomplete, no retry.', flush=True)
    raise SystemExit(0)

signal.signal(signal.SIGALRM, expired)
signal.alarm(30)
R = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/home/ec2-user/studies/hpp_datasets')
C = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.home()/'.pheno/config.json'
O, A = R/'oral_microbiome', R/'oral_microbiome/aggregated'
allowed = (Path(os.path.realpath(R.parent)), Path(os.path.realpath(R)))
sealed = re.compile(r'confirm|holdout|heldout|held_out|w22|w23', re.I)
bad = {'', 'undefined', 'none', 'null', 'nan', 'na', 'n/a', 'unknown', '<na>'}
fields = ['humann_aggregated_gene_familites_gene_level_arrow', 'humann_aggregated_pathway_abundance_pathway_level_arrow', 'humann_aggregated_pathway_coverage_pathway_level_arrow']
rules, roots = [], [R]
print('ORAL_AGGREGATE_CHECK: local metadata only; no S3 calls, rows, models or writes.', flush=True)
try:
    with C.open('rb') as stream:
        raw_config = stream.read(200001)
    if len(raw_config) > 200000:
        raise ValueError('configuration exceeds metadata budget')
    config = json.loads(raw_config)
    if not isinstance(config, dict):
        raise ValueError('configuration is not a mapping')
    keys = [key for key in config if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,79}', key) and re.search(r'path|root|dir|location|base|replace', key, re.I)]
    for key in keys:
        if isinstance(config[key], str) and re.search(r'data|bulk|base|root', key, re.I) and not re.search(r'dict|coding|config', key, re.I):
            path = Path(os.path.realpath(config[key]))
            if Path(config[key]).is_absolute() and any(path.is_relative_to(base) for base in allowed) and path not in roots:
                roots.append(path)
        if 'replace' in key.lower() and isinstance(config[key], dict):
            for source, target in config[key].items():
                if isinstance(source, str) and source and isinstance(target, str) and Path(target).is_absolute():
                    rules.append((source, target))
    print('PLATFORM_CONFIG path_keys=' + json.dumps(keys[:10]) + ' prefix_rules=%d root_candidates=%d' % (len(rules), len(roots)), flush=True)
except (OSError, ValueError, TypeError, AttributeError):
    print('PLATFORM_CONFIG unavailable; no configuration changed.', flush=True)
try:
    entries, limited = [], False
    with os.scandir(A) as listing:
        for number, entry in enumerate(listing, 1):
            if number > 60:
                limited = True
                break
            if not entry.is_symlink() and entry.is_dir(follow_symlinks=False) and not sealed.search(entry.name):
                if re.fullmatch(r'[A-Za-z_]{1,80}', entry.name):
                    entries.append(entry.name)
    print('EXACT_AGGREGATED_DIR exists=True limited=' + str(limited) + ' group_dirs=' + json.dumps(sorted(entries)[:10]), flush=True)
except OSError:
    print('EXACT_AGGREGATED_DIR unavailable.', flush=True)

def candidates(raw):
    uri = urlsplit(raw)
    if uri.username or uri.password or uri.query or uri.fragment:
        return []
    paths = []
    for source, target in rules:
        if raw.startswith(source) and (raw == source or source.endswith('/') or raw[len(source):].startswith('/')):
            suffix = Path(unquote(raw[len(source):].lstrip('/')))
            if '..' not in suffix.parts:
                paths.append(Path(target)/suffix)
    path = Path(unquote(uri.path))
    if '..' in path.parts:
        return []
    if uri.scheme in ('', 'file'):
        if uri.scheme == 'file' and uri.netloc not in ('', 'localhost'):
            return []
        paths.extend([path] if path.is_absolute() else [O/path, R/path])
    parts = path.parts
    for root in roots:
        for marker in ('oral_microbiome', 'aggregated'):
            if marker in parts:
                pos = parts.index(marker)
                paths.append((root if marker == 'oral_microbiome' else root/'oral_microbiome').joinpath(*parts[pos:]))
    result = []
    for path in paths:
        real = Path(os.path.realpath(path))
        if any(real.is_relative_to(base) for base in allowed) and not sealed.search(str(real)) and real not in result:
            result.append(real)
    return result[:4]

def scheme_of(value):
    try:
        scheme = urlsplit(value).scheme or 'relative_or_local'
        return scheme if scheme in ('file', 's3', 's3fs', 'gs', 'gcs', 'http', 'https', 'relative_or_local') else 'other'
    except ValueError:
        return 'unparseable'

try:
    with pq.ParquetFile(O/'oral_microbiome.parquet', pre_buffer=False) as pf:
        present = [name for name in fields if name in pf.schema_arrow.names]
        print('SOURCE_INDEX rows=%d requested_fields_present=%d/3' % (pf.metadata.num_rows, len(present)), flush=True)
        data = pf.read(columns=present, use_threads=False).to_pydict() if present else {}
except (OSError, ValueError, TypeError) as error:
    print('SOURCE_INDEX_UNAVAILABLE', type(error).__name__, flush=True)
    raise SystemExit(0)

for name in fields:
    if name not in data:
        print('FIELD_MISSING', name, flush=True)
        continue
    values = [v.strip() for v in data[name] if isinstance(v, str) and v.strip().lower() not in bad]
    unique = list(dict.fromkeys(values))
    schemes = Counter(scheme_of(v) for v in unique)
    print('FIELD', name, 'nonempty=%d unique=%d schemes=%s' % (len(values), len(unique), json.dumps(dict(schemes), sort_keys=True)), flush=True)
    if not unique:
        continue
    try:
        options = candidates(unique[0])
        found = next((path for path in options if path.is_file()), None)
        if found is None:
            print('  LOCAL_CANDIDATE_NOT_LOCATED tested=%d; remote not retried.' % len(options), flush=True)
            continue
        with pa.memory_map(str(found), 'r') as source:
            magic = source.read(6)
            source.seek(0)
            if magic == b'ARROW1':
                source.seek(-10, 2)
                if int.from_bytes(source.read(4), 'little') > 32*1024**2:
                    print('  IPC_FOOTER_BUDGET_EXCEEDED; not parsed.', flush=True)
                    continue
                source.seek(0)
                reader = ipc.open_file(source)
                print('  LOCAL_CANDIDATE_SCHEMA_OK format=IPC_FILE columns=%d batches=%d bytes=%d' % (len(reader.schema), reader.num_record_batches, source.size()), flush=True)
            elif magic[:4] == b'\xff\xff\xff\xff':
                reader = ipc.open_stream(source)
                print('  LOCAL_CANDIDATE_SCHEMA_OK format=IPC_STREAM columns=%d bytes=%d' % (len(reader.schema), source.size()), flush=True)
            elif magic[:4] == b'PAR1':
                reader = pq.ParquetFile(source, pre_buffer=False)
                print('  LOCAL_CANDIDATE_SCHEMA_OK format=PARQUET columns=%d rows=%d' % (len(reader.schema_arrow), reader.metadata.num_rows), flush=True)
            else:
                print('  LOCAL_CANDIDATE_FORMAT_UNRECOGNIZED; body not read.', flush=True)
    except (OSError, ValueError, TypeError) as error:
        print('  LOCAL_CANDIDATE_UNAVAILABLE', type(error).__name__, flush=True)
print('Local candidates require source/version and locked-sample matching; 3343-person coverage unverified.', flush=True)
print('ORAL_AGGREGATE_CHECK_DONE', flush=True)
signal.alarm(0)
PY
