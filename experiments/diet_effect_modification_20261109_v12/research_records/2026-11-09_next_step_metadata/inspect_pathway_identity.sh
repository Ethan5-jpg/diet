#!/usr/bin/env bash
# Read-only, projected identity columns; no functional values or outcomes.
hpp_identity_python=python
if [ -s .python_path ]; then
    IFS= read -r hpp_identity_python < .python_path
fi
"$hpp_identity_python" -B - "$@" <<'PY'
import hashlib, json, os, re, signal, sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

def expired(signum, frame):
    print('PATHWAY_IDENTITY_TIME_LIMIT; incomplete.', flush=True)
    raise SystemExit(0)
signal.signal(signal.SIGALRM, expired)
signal.alarm(30)
R = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/home/ec2-user/studies/hpp_datasets')
P = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
O = R/'oral_microbiome'
allowed = (Path(os.path.realpath(R)), Path(os.path.realpath(R.parent)))
K = ['participant_id', 'cohort', 'research_stage']
IDS = K + ['array_index', 'sample_name', 'sample_id', 'uuid', 'sample', '__index_level_0__']
F = ['humann_aggregated_pathway_abundance_pathway_level_arrow', 'humann_aggregated_pathway_coverage_pathway_level_arrow']
bad = {'', 'undefined', 'none', 'null', 'nan', 'na', 'n/a', 'unknown', '<na>'}
print('PATHWAY_IDENTITY: local file hashes, schemas and projected identifiers; no outcomes, models, row output or writes.', flush=True)

def local_candidates(raw):
    u = urlsplit(raw)
    if u.scheme not in ('', 'file') or u.netloc not in ('', 'localhost') or u.query or u.fragment:
        return []
    p = Path(unquote(u.path))
    if '..' in p.parts:
        return []
    paths = [p] if p.is_absolute() else [O/p, R/p]
    for marker in ('oral_microbiome', 'aggregated'):
        if marker in p.parts:
            paths.append((R if marker == 'oral_microbiome' else O).joinpath(*p.parts[p.parts.index(marker):]))
    found = []
    for p in paths:
        q = Path(os.path.realpath(p))
        if not any(q == b or b in q.parents for b in allowed) or re.search(r'confirm|holdout|heldout|held_out|w22|w23', str(q), re.I):
            continue
        if q not in found and q.is_file():
            found.append(q)
    return found

def valid(v):
    return v is not None and not (isinstance(v, float) and v != v) and not (isinstance(v, str) and v.strip().lower() in bad)

def keys(data, names):
    return [tuple(row) if all(valid(v) for v in row) else None for row in zip(*(data[n] for n in names))]

def link(label, data, raw, names):
    a, b = keys(data, names), keys(raw, names)
    counts = Counter(v for v in b if v is not None)
    own = Counter(v for v in a if v is not None)
    print('  KEY_LINK', label, 'found=%d missing=%d null_key=%d source_ambiguous=%d table_duplicate_rows=%d' % (
        sum(v in counts for v in a if v is not None), sum(v not in counts for v in a if v is not None),
        a.count(None), sum(counts[v] > 1 for v in a if v is not None), sum(n for n in own.values() if n > 1)), flush=True)

try:
    with pq.ParquetFile(O/'oral_microbiome.parquet', pre_buffer=False) as pf:
        names = pf.schema_arrow.names
        raw = pf.read(columns=[n for n in F + IDS if n in names], use_threads=False).to_pydict()
        print('SOURCE_INDEX rows=%d identifier_fields=%s' % (pf.metadata.num_rows, json.dumps([n for n in IDS if n in raw])), flush=True)
    reg = pq.ParquetFile(P/'analysis/shared/sample_registry_TRE_ONLY.parquet', pre_buffer=False)
    print('LOCKED_REGISTRY rows=%d linkage_fields=%s (schema only)' % (reg.metadata.num_rows, json.dumps([n for n in K + ['site', 'linked_index_row_ordinal', 'index_sample_name', 'primary_index_source_path'] if n in reg.schema_arrow.names])), flush=True)
    reg.close()
except (OSError, ValueError, TypeError) as e:
    print('INDEX_OR_REGISTRY_UNAVAILABLE', type(e).__name__, flush=True)
    raise SystemExit(0)

resolved, pointers, digests = {}, {}, {}
for label, field in zip(('abundance', 'coverage'), F):
    values = list(dict.fromkeys(v.strip() for v in raw.get(field, []) if isinstance(v, str) and v.strip().lower() not in bad))
    print('TABLE', label, 'unique_pointers=%d' % len(values), flush=True)
    if len(values) != 1:
        print('  POINTER_NOT_UNIQUE; not selected.', flush=True)
        continue
    pointers[label] = values[0]
    try:
        options = local_candidates(values[0])
        if len(options) != 1:
            print('  LOCAL_CANDIDATE_COUNT=%d; not selected.' % len(options), flush=True)
            continue
        path = options[0]
        state = path.stat()
        resolved[label] = path
        if state.st_size > 64*1024**2:
            print('  FILE_BUDGET_EXCEEDED; not parsed.', flush=True)
            continue
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024**2), b''):
                h.update(chunk)
        with pa.memory_map(str(path), 'r') as source:
            if source.read(6) != b'ARROW1' or state.st_size < 10:
                print('  NOT_IPC_FILE; not parsed.', flush=True)
                continue
            source.seek(-10, 2)
            if int.from_bytes(source.read(4), 'little') > 32*1024**2:
                print('  IPC_FOOTER_BUDGET_EXCEEDED; not parsed.', flush=True)
                continue
            source.seek(0)
            reader = ipc.open_file(source)
            schema = reader.schema
            keep = [n for n in IDS if n in schema.names]
            numeric = sum(pa.types.is_floating(f.type) or pa.types.is_integer(f.type) for f in schema if f.name not in IDS)
            codes = [n for n in schema.names if re.fullmatch(r'PWY-[0-9]+[A-Z]?|[A-Z][A-Z0-9-]{0,39}-PWY', n)]
            print('  STRUCTURE columns=%d batches=%d numeric_nonidentifier_columns=%d known_identifiers=%s' % (len(schema), reader.num_record_batches, numeric, json.dumps(keep)), flush=True)
            print('  PUBLIC_PATHWAY_CODE_CANDIDATES', json.dumps(codes[:3]), flush=True)
            if keep:
                source.seek(0)
                reader = ipc.open_file(source, options=ipc.IpcReadOptions(use_threads=False, included_fields=[schema.get_field_index(n) for n in keep]))
                ids = reader.read_all()
                print('  IDENTIFIER_PROJECTION rows=%d columns=%d' % (ids.num_rows, ids.num_columns), flush=True)
                data = ids.to_pydict()
                if all(n in data and n in raw for n in K):
                    link('visit', data, raw, K)
                    for extra in ('array_index', 'sample_name'):
                        if extra in data and extra in raw:
                            link('visit+' + extra, data, raw, K + [extra])
            else:
                print('  NO_KNOWN_IDENTIFIER; record batches not read.', flush=True)
        after = path.stat()
        if (state.st_size, state.st_mtime_ns) == (after.st_size, after.st_mtime_ns):
            digests[label] = h.digest()
        else:
            print('  FILE_CHANGED_DURING_CHECK; identity unresolved.', flush=True)
    except (OSError, ValueError, TypeError, KeyError) as e:
        print('  TABLE_CHECK_UNAVAILABLE', type(e).__name__, flush=True)

if len(pointers) == 2:
    print('PAIR pointer_equal=%s' % (pointers['abundance'] == pointers['coverage']), flush=True)
if len(resolved) == 2:
    try:
        print('PAIR canonical_path_equal=%s same_file=%s' % (resolved['abundance'] == resolved['coverage'], os.path.samefile(resolved['abundance'], resolved['coverage'])), flush=True)
    except OSError as e:
        print('PAIR_FILE_IDENTITY_UNAVAILABLE', type(e).__name__, flush=True)
if len(digests) == 2:
    print('PAIR full_file_bytes_equal=%s' % (digests['abundance'] == digests['coverage']), flush=True)
print('Key matches are linkage diagnostics; locked-specimen identity, units and fixed 3343-person coverage remain unverified.', flush=True)
print('PATHWAY_IDENTITY_DONE', flush=True)
signal.alarm(0)
PY
