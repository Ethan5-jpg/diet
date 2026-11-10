#!/usr/bin/env bash
# Read-only metadata inspection; do not source or import project analysis code.
hpp_meta_settings=/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_effect_modification_20261109_v12/.python_path
hpp_meta_python=python
if [ -s "$hpp_meta_settings" ]; then
    IFS= read -r hpp_meta_python < "$hpp_meta_settings"
fi
"$hpp_meta_python" -B - "$@" <<'PY'
import json, os, re, signal, sys, time
from pathlib import Path
import pyarrow.parquet as pq

def expired(signum, frame):
    print('TIME_LIMIT_REACHED; inspection incomplete; no files changed.', flush=True)
    raise SystemExit(0)

signal.signal(signal.SIGALRM, expired)
signal.alarm(60)
P = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
R = P.parent.parent / 'hpp_datasets'
sealed = re.compile(r'confirm|holdout|heldout|held_out|w22|w23', re.I)
date_fields = re.compile(r'date|timestamp|sampling_time|collection_time', re.I)
functional = re.compile(r'humann|pathabundance|pathcoverage|pathway.?abundance|gene.?famil|gene.?abundance|uniref|metacyc|kegg|cazy|dbcan|eggnog|ko_abundance|functional|scfa', re.I)
allowed = ('.parquet', '.tsv', '.tsv.gz', '.csv', '.csv.gz', '.txt', '.txt.gz', '.biom', '.h5', '.hdf5', '.npz')
print('METADATA_ONLY: no participant rows, outcomes, fitting or writes.', flush=True)

def small_json(path):
    with path.open('rb') as stream:
        raw = stream.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError('metadata file exceeds 2 MB')
    return json.loads(raw)

rule = P / 'config/sample_selection_rules.yaml'
try:
    value = small_json(rule).get('date_policy')
    print('DATE_POLICY:', json.dumps(value, ensure_ascii=True)[:380], flush=True)
except (OSError, ValueError, AttributeError):
    print('DATE_POLICY: unavailable; biological date meaning remains unverified.', flush=True)

sources = {}
for path in (P / 'data_qc/w11_w12_covariates_samples_20261028/w12_source_manifest_TRE_ONLY.json', P / 'manifests/w02_sources_20261026T130908Z/source_manifest.json'):
    if all(role in sources for role in ('oral_index', 'gut_index')):
        break
    try:
        obj = small_json(path)
        if isinstance(obj, dict) and isinstance(obj.get('sources'), list):
            entries = [(item.get('role'), item) for item in obj['sources'] if isinstance(item, dict)]
        elif isinstance(obj, dict):
            entries = obj.items()
        else:
            raise ValueError('unknown manifest structure')
        for role, item in entries:
            if role in ('oral_index', 'gut_index') and isinstance(item, dict) and isinstance(item.get('path'), str):
                candidate = Path(item['path'])
                if candidate.is_absolute() and not sealed.search(str(candidate)):
                    sources.setdefault(role, candidate)
    except (OSError, ValueError, TypeError):
        print('MANIFEST_UNAVAILABLE:', path.name, flush=True)

tables = [('LOCKED_SAMPLE_REGISTRY', P / 'analysis/shared/sample_registry_TRE_ONLY.parquet')] + list(sources.items())
for role in ('oral_index', 'gut_index'):
    if role not in sources:
        print('SOURCE_PATH_UNAVAILABLE:', role, flush=True)
for label, path in tables:
    try:
        if label != 'LOCKED_SAMPLE_REGISTRY':
            print('SOURCE:', label, json.dumps(str(path), ensure_ascii=True), flush=True)
        with pq.ParquetFile(path, pre_buffer=False) as pf:
            fields = [str(field.name) + ':' + str(field.type) for field in pf.schema_arrow if date_fields.search(field.name)]
            print(label, 'source_rows=' + str(pf.metadata.num_rows), 'date_fields=' + str(len(fields)), flush=True)
            print('  ' + ', '.join(fields[:7]), flush=True)
            if len(fields) > 7:
                print('  MORE_DATE_FIELDS=' + str(len(fields) - 7), flush=True)
    except (OSError, ValueError, TypeError):
        print(label, 'HEADER_UNAVAILABLE', flush=True)

roots = [P / 'analysis/shared'] + sorted({path.parent for path in sources.values()}) + [R]
seen, hits, errors = set(), [], 0
entries_seen, limited = 0, False
deadline = time.monotonic() + 20
for root in roots:
    print('SCAN_ROOT:', json.dumps(str(root), ensure_ascii=True), flush=True)
    stack = [(root, 0)]
    while stack:
        if time.monotonic() > deadline or entries_seen >= 30000:
            limited = True
            break
        folder, depth = stack.pop()
        if str(folder) in seen or sealed.search(str(folder)):
            continue
        seen.add(str(folder))
        try:
            with os.scandir(folder) as listing:
                for item in listing:
                    if time.monotonic() > deadline or entries_seen >= 30000:
                        limited = True
                        break
                    entries_seen += 1
                    if sealed.search(item.name) or item.name.startswith('.') or item.is_symlink():
                        continue
                    if item.is_dir(follow_symlinks=False):
                        if depth < 2 and item.name not in ('runs', 'TRE_ONLY', '__pycache__', 'node_modules'):
                            stack.append((Path(item.path), depth + 1))
                    elif item.is_file(follow_symlinks=False) and item.name.lower().endswith(allowed) and functional.search(item.name):
                        hits.append(item.path)
        except OSError:
            errors += 1
    if limited:
        break
print('FUNCTION_FILENAME_SCAN: dirs=%d entries=%d errors=%d limited=%s max_depth=2' % (len(seen), entries_seen, errors, limited), flush=True)
for path in sorted(set(hits))[:10]:
    print('CANDIDATE:', json.dumps(path, ensure_ascii=True), flush=True)
if len(set(hits)) > 10:
    print('MORE_CANDIDATES=' + str(len(set(hits)) - 10), flush=True)
if not hits:
    print('NOT_LOCATED_IN_LIMITED_SCAN; this does not establish absence.', flush=True)
print('Date meaning, functional validity and overlap with 3343 people remain UNVERIFIED.', flush=True)
print('METADATA_CHECK_DONE', flush=True)
signal.alarm(0)
PY
