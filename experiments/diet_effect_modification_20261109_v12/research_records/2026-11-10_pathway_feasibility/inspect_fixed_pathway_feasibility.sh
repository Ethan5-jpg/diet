#!/usr/bin/env bash
# Fixed-population provenance, then conditional descriptive feature QC.
hpp_feasibility_python=python
if [ -s .python_path ]; then
    IFS= read -r hpp_feasibility_python < .python_path
fi
"$hpp_feasibility_python" -B - "$@" <<'PY'
import hashlib, json, math, os, re, signal, sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote, urlsplit
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

def expired(signum, frame):
    print('FEASIBILITY_TIME_LIMIT; incomplete, no retry.', flush=True)
    raise SystemExit(0)
signal.signal(signal.SIGALRM, expired)
signal.alarm(60)
R = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/home/ec2-user/studies/hpp_datasets')
P = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
expected = int(sys.argv[3]) if len(sys.argv) > 3 else 3343
input_count = int(sys.argv[4]) if len(sys.argv) > 4 else 4427
expected_households = int(sys.argv[5]) if len(sys.argv) > 5 else 3059
B = P/'collaborator_handoff_20261102_v1'
O = R/'oral_microbiome'
K = ['participant_id', 'cohort', 'research_stage']
META = K + ['array_index', 'collection_date']
F = ['humann_aggregated_pathway_abundance_pathway_level_arrow', 'humann_aggregated_pathway_coverage_pathway_level_arrow']
BAD = {'', 'undefined', 'none', 'null', 'nan', 'na', 'n/a', 'unknown', '<na>'}
print('FIXED_PATHWAY_FEASIBILITY: TRE-only linkage and descriptive QC; old outcome availability only for fixed membership; no outcome summaries, models or writes.', flush=True)

def require(condition, label):
    if not condition:
        raise ValueError(label)

def digest(path):
    require(path.stat().st_size <= 128*1024**2, 'CHECKSUM_FILE_BUDGET')
    h = hashlib.sha256()
    before = path.stat()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'CHECKSUM_FILE_CHANGED')
    return h.hexdigest()

def json_file(path):
    with path.open('rb') as stream:
        value = stream.read(2*1024**2 + 1)
    require(len(value) <= 2*1024**2, 'JSON_BUDGET')
    result = json.loads(value)
    require(isinstance(result, dict), 'JSON_NOT_MAPPING')
    return result

def valid(value):
    if value is None or isinstance(value, str) and value.strip().lower() in BAD:
        return False
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return False

def key(row):
    values = tuple(row.get(n) for n in K)
    return values if all(valid(v) for v in values) else None

def day(value):
    if not valid(value):
        return None
    try:
        return pd.Timestamp(value).date()
    except (TypeError, ValueError, OverflowError):
        return None

def same(a, b):
    return valid(a) and valid(b) and a == b

def matches(a, b):
    stage_compatible = not valid(b.get('research_stage')) or same(a.get('research_stage'), b.get('research_stage'))
    return stage_compatible and same(a.get('cohort'), b.get('cohort')) and same(a.get('array_index'), b.get('array_index')) and day(a.get('collection_date')) is not None and day(a.get('collection_date')) == day(b.get('collection_date'))

def signature(path):
    state = path.stat()
    return state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns

def unique_pid(rows):
    counts = Counter(r.get('participant_id') for r in rows if valid(r.get('participant_id')))
    return {r['participant_id']:i for i,r in enumerate(rows) if valid(r.get('participant_id')) and counts[r['participant_id']] == 1}

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
    allowed, result = Path(os.path.realpath(R.parent)), []
    for p in paths:
        q = Path(os.path.realpath(p))
        if allowed in q.parents and not re.search(r'confirm|holdout|heldout|held_out|w22|w23', str(q), re.I) and q not in result and q.is_file():
            result.append(q)
    return result

def ipc_schema(stream):
    require(10 <= stream.size() <= 64*1024**2, 'IPC_FILE_BUDGET')
    require(stream.read(6) == b'ARROW1', 'NOT_IPC_FILE')
    stream.seek(-10, 2)
    require(int.from_bytes(stream.read(4), 'little') <= 32*1024**2, 'IPC_FOOTER_BUDGET')
    stream.seek(0)
    return ipc.open_file(stream).schema

def ipc_columns(path, columns, expected_signature=None):
    before = signature(path)
    require(expected_signature is None or before == expected_signature, 'FUNCTION_CHANGED_SINCE_METADATA')
    with pa.memory_map(str(path), 'r') as stream:
        schema = ipc_schema(stream)
        require(all(n in schema.names for n in columns) and bool(columns), 'IPC_COLUMNS_MISSING')
        stream.seek(0)
        options = ipc.IpcReadOptions(use_threads=False, included_fields=[schema.get_field_index(n) for n in columns])
        result = ipc.open_file(stream, options=options).read_all()
        require(result.nbytes <= 256*1024**2, 'DECODED_TABLE_BUDGET')
        require(before == signature(path), 'FUNCTION_CHANGED_DURING_READ')
        return result

def qc(label, path, rows, columns, expected_signature):
    selected = ipc_columns(path, columns, expected_signature).take(pa.array(rows, type=pa.int64()))
    nulls = sum(selected.column(n).null_count for n in columns)
    matrix = np.column_stack([selected.column(n).to_numpy(zero_copy_only=False) for n in columns]).astype(float)
    finite = np.isfinite(matrix)
    observations = finite.sum(axis=0)
    low = np.where(finite, matrix, np.inf).min(axis=0)
    high = np.where(finite, matrix, -np.inf).max(axis=0)
    values = matrix[finite]
    quantiles = np.quantile(values, [0, .5, .99, 1]).tolist() if len(values) else []
    positive = (finite & (matrix > 0)).sum(axis=1)
    print('  QC_CELLS', label, 'cells=%d arrow_null=%d nonnull_nan=%d inf=%d negative=%d gt1=%d min_med_p99_max=%s units_unverified=True' % (
        matrix.size, nulls, int(np.isnan(matrix).sum())-nulls, int(np.isinf(matrix).sum()),
        int((finite & (matrix < 0)).sum()), int((finite & (matrix > 1)).sum()), json.dumps(quantiles)), flush=True)
    print('  QC_PROFILES', label, 'fixed_people=%d no_finite=%d no_positive_observed=%d all_observed_zero=%d constant_observed_features=%d all_missing_features=%d positive_fields_q10_50_90=%s' % (
        len(rows), int((finite.sum(axis=1) == 0).sum()), int((positive == 0).sum()),
        int((finite.all(axis=1) & (matrix == 0).all(axis=1)).sum()), int(((observations > 0) & (low == high)).sum()),
        int((observations == 0).sum()), json.dumps(np.quantile(positive, [.1, .5, .9]).tolist())), flush=True)
    return matrix

try:
    cov_path = B/'inputs/discovery_v2/discovery_covariates_food_TRE_ONLY.parquet'
    v4 = B/'experiments/diet_nested_stability_20261103_v4/results'
    receipt = json_file(v4/'registration.json')['source_and_input_sha256']
    require(receipt.get(str(cov_path)) == digest(cov_path), 'DISCOVERY_INPUT_CHECKSUM_MISMATCH')
    with pq.ParquetFile(cov_path, pre_buffer=False) as pf:
        x = pf.read(columns=K+['input_row', 'final_role', 'split_unit_id'], use_threads=False).to_pandas(ignore_metadata=True)
    require(len(x) == input_count and x.final_role.eq('discovery').all() and x.participant_id.is_unique, 'DISCOVERY_INDEX_CHANGED')
    require(np.array_equal(x.input_row.to_numpy(), np.arange(len(x))), 'DISCOVERY_ROW_ORDER_CHANGED')
    with np.load(str(v4/'TRE_ONLY/final_predictions.npz'), allow_pickle=False) as archive:
        eligible = np.isfinite(archive['y'])
    require(eligible.shape == (2, input_count) and np.array_equal(eligible[0], eligible[1]), 'FIXED_MEMBERSHIP_CHANGED')
    target = x.loc[eligible[0]].copy()
    require(len(target) == expected and target.split_unit_id.notna().all() and target.split_unit_id.nunique() == expected_households, 'FIXED_COUNTS_CHANGED')
    people = target[K].to_dict('records')
    require(all(key(r) is not None for r in people), 'FIXED_KEYS_INVALID')
    print('FROZEN_MEMBERSHIP discovery_rows=%d fixed_people=%d households=%d same_two_repeat_mask=True discovery_input_sha_verified=True' % (len(x), len(target), target.split_unit_id.nunique()), flush=True)

    reg_path = P/'analysis/shared/sample_registry_TRE_ONLY.parquet'
    lock = json_file(P/'config/w13_sample_selection_lock.json')
    require(lock.get('sample_registry_sha256') == digest(reg_path), 'LOCKED_REGISTRY_CHECKSUM_MISMATCH')
    reg_fields = K+['site', 'sample_lock_status', 'linked_index_row_ordinal', 'primary_index_source_path', 'primary_index_source_sha256']
    with pq.ParquetFile(reg_path, pre_buffer=False) as pf:
        registry = pf.read(columns=reg_fields, use_threads=False).to_pylist()
    reg_map = defaultdict(list)
    for r in registry:
        if r['site'] == 'oral' and key(r) is not None:
            reg_map[key(r)].append(r)
    src_path = O/'oral_microbiome.parquet'
    src_hash = digest(src_path)
    with pq.ParquetFile(src_path, pre_buffer=False) as pf:
        src_table = pf.read(columns=META+F, use_threads=False)
    source = src_table.select(META).to_pylist()
    src_map = unique_pid(source)
    reasons, approved = Counter(), []
    for person in people:
        candidates_reg = reg_map.get(key(person), [])
        if len(candidates_reg) != 1:
            reasons['registry_missing_or_ambiguous'] += 1
            approved.append(None)
            continue
        r = candidates_reg[0]
        reason = None
        if r['sample_lock_status'] != 'locked_W13_independent_full_source_verification_passed':
            reason = 'registry_not_locked'
        elif not isinstance(r['primary_index_source_path'], str) or os.path.realpath(r['primary_index_source_path']) != os.path.realpath(src_path):
            reason = 'source_path_mismatch'
        elif r['primary_index_source_sha256'] != src_hash:
            reason = 'source_checksum_mismatch'
        else:
            ordinal = r['linked_index_row_ordinal']
            if not valid(ordinal) or not isinstance(ordinal, (int, float)) or not math.isfinite(ordinal) or int(ordinal) != ordinal or not 1 <= ordinal <= len(source):
                reason = 'source_ordinal_invalid'
            else:
                index = int(ordinal)-1
                if key(source[index]) != key(person) or src_map.get(person['participant_id']) != index:
                    reason = 'source_key_or_uniqueness_mismatch'
        reasons[reason or 'source_record_verified'] += 1
        approved.append(index if reason is None else None)
    print('LOCKED_SOURCE_RECORDS registry_sha_verified=True reasons=' + json.dumps(dict(reasons), sort_keys=True), flush=True)

    records = {}
    for label, field in zip(('abundance', 'coverage'), F):
        print('TABLE', label, flush=True)
        try:
            values = list(dict.fromkeys(v.strip() for v in src_table[field].to_pylist() if isinstance(v, str) and valid(v)))
            paths = candidates(values[0]) if len(values) == 1 else []
            require(len(paths) == 1, 'FUNCTION_POINTER_OR_PATH_AMBIGUOUS')
            path = paths[0]
            stamp = signature(path)
            with pa.memory_map(str(path), 'r') as stream:
                schema = ipc_schema(stream)
            numeric = [f.name for f in schema if f.name not in META and (pa.types.is_floating(f.type) or pa.types.is_integer(f.type))]
            special = [n for n in numeric if n.split(':', 1)[0].strip().upper() in ('UNMAPPED', 'UNINTEGRATED')]
            columns = sorted(n for n in numeric if n not in special)
            require(bool(columns), 'NO_NUMERIC_CANDIDATE_FEATURES')
            metadata = ipc_columns(path, META, stamp).to_pylist()
            mapping = unique_pid(metadata)
            shared = [(r, metadata[mapping[r['participant_id']]]) for r in source if r['participant_id'] in src_map and r['participant_id'] in mapping]
            print('  SOURCE_METADATA shared_unique_people=%d cohort_equal=%d array_equal=%d calendar_date_equal=%d timestamp_equal=%d source_record_metadata_compatible=%d' % (
                len(shared), sum(same(a['cohort'], b['cohort']) for a,b in shared), sum(same(a['array_index'], b['array_index']) for a,b in shared),
                sum(day(a['collection_date']) is not None and day(a['collection_date']) == day(b['collection_date']) for a,b in shared),
                sum(same(a['collection_date'], b['collection_date']) for a,b in shared), sum(matches(a,b) for a,b in shared)), flush=True)
            linked, fixed_reasons = [], Counter()
            for person, source_index in zip(people, approved):
                if source_index is None:
                    fixed_reasons['locked_source_unresolved'] += 1
                    continue
                position = mapping.get(person['participant_id'])
                if position is None:
                    fixed_reasons['function_person_missing_or_nonunique'] += 1
                    continue
                if not matches(source[source_index], metadata[position]):
                    fixed_reasons['function_metadata_conflict_or_date_missing'] += 1
                    continue
                fixed_reasons['source_record_metadata_agree'] += 1
                linked.append(position)
            print('  FIXED_LINK target=%d linked=%d reasons=%s' % (expected, len(linked), json.dumps(dict(fixed_reasons), sort_keys=True)), flush=True)
            print('  FEATURE_SCHEMA numeric_total=%d special_totals_excluded=%d candidate_columns=%d annotation_and_units_pending=True' % (len(numeric), len(special), len(columns)), flush=True)
            records[label] = dict(path=path, signature=stamp, metadata=metadata, mapping=mapping, linked=linked, columns=columns, complete=len(linked)==expected)
        except (OSError, ValueError, TypeError, KeyError) as e:
            detail = str(e) if isinstance(e, ValueError) and re.fullmatch(r'[A-Z_]{1,80}', str(e)) else type(e).__name__
            print('  TABLE_UNRESOLVED', detail, flush=True)

    complete = len(records) == 2 and all(v['complete'] for v in records.values())
    if len(records) == 2:
        a, b = records['abundance'], records['coverage']
        print('PAIR_FEATURE_LABEL_SETS_EQUAL=%s' % (a['columns'] == b['columns']), flush=True)
    if complete:
        print('DESCRIPTIVE_QC: all fixed people passed source-record metadata checks; missing cells are retained.', flush=True)
        matrices = {label:qc(label, r['path'], r['linked'], r['columns'], r['signature']) for label,r in records.items()}
        if records['abundance']['columns'] == records['coverage']['columns']:
            a, b = matrices['abundance'], matrices['coverage']
            common = np.isfinite(a) & np.isfinite(b)
            print('PAIR_NUMERIC fixed_people=%d common_finite_cells=%d exact_equal_common_finite_cells=%d' % (expected, int(common.sum()), int((a[common] == b[common]).sum())), flush=True)
    else:
        print('QC_SKIPPED: complete fixed-population source-record linkage not established; no subset replacement.', flush=True)
    print('Source-record agreement is not independent native-file specimen validation, verified units, causal timing or a new scientific effect.', flush=True)
    print('FIXED_PATHWAY_FEASIBILITY_DONE', flush=True)
except (OSError, ValueError, TypeError, KeyError, AttributeError) as e:
    detail = str(e) if isinstance(e, ValueError) and re.fullmatch(r'[A-Z_]{1,80}', str(e)) else type(e).__name__
    print('FEASIBILITY_STOP', detail, '; no model or subset replacement.', flush=True)
signal.alarm(0)
PY
