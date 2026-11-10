#!/usr/bin/env bash
# Diagnose missing identifier components without reading numerical features.
hpp_keys_python=python
if [ -s .python_path ]; then
    IFS= read -r hpp_keys_python < .python_path
fi
"$hpp_keys_python" -B - "$@" <<'PY'
import json, math, os, re, signal, sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

def expired(signum, frame):
    print('KEY_DIAGNOSTIC_TIME_LIMIT; incomplete.', flush=True)
    raise SystemExit(0)
signal.signal(signal.SIGALRM, expired)
signal.alarm(30)
R = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/home/ec2-user/studies/hpp_datasets')
O = R/'oral_microbiome'
K = ['participant_id', 'cohort', 'research_stage', 'array_index']
IDS = K + ['sample_name', 'sample_id', 'uuid', 'sample', 'index', 'collection_date', 'collection_timestamp', 'run_name', 'run_id', 'registration_code', 'person_id', 'individual_id', 'subject_id', 'specimen_id', 'biosample_id', 'visit_id', '__index_level_0__', '__index_level_1__', '__index_level_2__']
F = ['humann_aggregated_pathway_abundance_pathway_level_arrow', 'humann_aggregated_pathway_coverage_pathway_level_arrow']
bad = {'', 'undefined', 'none', 'null', 'nan', 'na', 'n/a', 'unknown', '<na>'}
print('PATHWAY_KEYS: projected metadata only; no individual values, features, outcomes, fitting or writes.', flush=True)

def category(v):
    if v is None:
        return 'null'
    if isinstance(v, float) and math.isnan(v):
        return 'nan'
    if isinstance(v, str) and v.strip().lower() in bad:
        return 'placeholder'
    try:
        hash(v)
        return 'valid'
    except TypeError:
        return 'nonscalar'

def typename(t):
    if pa.types.is_dictionary(t):
        return 'dictionary<' + typename(t.value_type) + '>'
    for name in ('null', 'string', 'large_string', 'integer', 'floating', 'timestamp', 'date', 'boolean', 'binary'):
        if getattr(pa.types, 'is_' + name)(t):
            return name
    return 'other'

def describe(label, table):
    data = table.to_pydict()
    for n in [n for n in IDS if n in data]:
        c = Counter(category(v) for v in data[n])
        unique = len(set(v for v in data[n] if category(v) == 'valid'))
        print('  COLUMN', label, n, 'type=%s null=%d nan=%d placeholder=%d valid=%d unique_valid=%d nonscalar=%d' % (
            typename(table.schema.field(n).type), c['null'], c['nan'], c['placeholder'], c['valid'], unique, c['nonscalar']), flush=True)
    return data

def candidates(raw):
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
    allowed = Path(os.path.realpath(R.parent))
    result = []
    for p in paths:
        q = Path(os.path.realpath(p))
        if allowed not in q.parents or re.search(r'confirm|holdout|heldout|held_out|w22|w23', str(q), re.I):
            continue
        if q not in result and q.is_file():
            result.append(q)
    return result

try:
    with pq.ParquetFile(O/'oral_microbiome.parquet', pre_buffer=False) as pf:
        names = pf.schema_arrow.names
        raw = pf.read(columns=[n for n in F + IDS if n in names], use_threads=False)
    source = describe('source', raw.select([n for n in IDS if n in names]))
except (OSError, ValueError, TypeError) as e:
    print('SOURCE_UNAVAILABLE', type(e).__name__, flush=True)
    raise SystemExit(0)

for label, field in zip(('abundance', 'coverage'), F):
    print('TABLE', label, flush=True)
    try:
        values = list(dict.fromkeys(v.strip() for v in raw[field].to_pylist() if isinstance(v, str) and category(v) == 'valid')) if field in names else []
        options = candidates(values[0]) if len(values) == 1 else []
        if len(options) != 1:
            print('  POINTER_OR_PATH_AMBIGUOUS_OR_UNAVAILABLE; not selected.', flush=True)
            continue
        path = options[0]
        with pa.memory_map(str(path), 'r') as stream:
            if stream.size() > 64*1024**2 or stream.size() < 10 or stream.read(6) != b'ARROW1':
                print('  FILE_BUDGET_OR_FORMAT_UNSUPPORTED.', flush=True)
                continue
            stream.seek(-10, 2)
            if int.from_bytes(stream.read(4), 'little') > 32*1024**2:
                print('  IPC_FOOTER_BUDGET_EXCEEDED.', flush=True)
                continue
            stream.seek(0)
            schema = ipc.open_file(stream).schema
            meta = json.loads((schema.metadata or {}).get(b'pandas', b'{}'))
            ix = meta.get('index_columns', [])
            print('  PANDAS_INDEX known_fields=%s unknown_named_fields=%d range_descriptors=%d' % (
                json.dumps([v for v in ix if isinstance(v, str) and v in IDS][:24]),
                sum(isinstance(v, str) and v not in IDS for v in ix),
                sum(isinstance(v, dict) and v.get('kind') == 'range' for v in ix)), flush=True)
            labels = [f.name for f in schema if f.name not in IDS]
            other = [(i, typename(f.type)) for i, f in enumerate(schema) if f.name not in IDS and not (pa.types.is_integer(f.type) or pa.types.is_floating(f.type))]
            tokens = sorted(set(m.group(0) for n in labels for m in re.finditer(r'PWY-[0-9]+[A-Z]?|[A-Z][A-Z0-9-]{0,39}-PWY', n)))
            print('  LABELS numeric_names=%d contains_PWY=%d other=%d public_code_tokens=%s unknown_nonnumeric_positions_types=%s' % (
                sum(n.isdigit() for n in labels), sum('PWY' in n.upper() for n in labels),
                sum(not n.isdigit() and 'PWY' not in n.upper() for n in labels), json.dumps(tokens[:3]), json.dumps(other[:8])), flush=True)
            keep = [n for n in IDS if n in schema.names]
            if not keep:
                print('  NO_RECOGNIZED_METADATA_FIELDS; batches not read.', flush=True)
                continue
            stream.seek(0)
            reader = ipc.open_file(stream, options=ipc.IpcReadOptions(use_threads=False, included_fields=[schema.get_field_index(n) for n in keep]))
            table = reader.read_all()
            print('  PROJECTED rows=%d metadata_columns=%d' % (table.num_rows, table.num_columns), flush=True)
            data = describe(label, table)
            if 'participant_id' in data and 'participant_id' in source:
                a = Counter(v for v in data['participant_id'] if category(v) == 'valid')
                b = Counter(v for v in source['participant_id'] if category(v) == 'valid')
                if a and b:
                    print('  PID_ONLY_DIAGNOSTIC shared_unique_ids=%d table_unique_ids=%d source_unique_ids=%d table_repeated_ids=%d source_repeated_ids=%d' % (
                        len(a.keys() & b.keys()), len(a), len(b), sum(v > 1 for v in a.values()), sum(v > 1 for v in b.values())), flush=True)
                else:
                    print('  PID_ONLY_DIAGNOSTIC unavailable_valid_ids; not a zero-overlap result.', flush=True)
            else:
                print('  PID_ONLY_DIAGNOSTIC missing_identifier_column; no comparison.', flush=True)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as e:
        print('  TABLE_DIAGNOSTIC_UNAVAILABLE', type(e).__name__, flush=True)
print('Participant overlap is not locked-sample identity or fixed 3343-person coverage; no key filling or row-order matching.', flush=True)
print('PATHWAY_KEYS_DONE', flush=True)
signal.alarm(0)
PY
