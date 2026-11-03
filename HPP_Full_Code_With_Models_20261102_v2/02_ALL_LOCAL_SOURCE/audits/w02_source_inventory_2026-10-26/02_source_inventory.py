"""W02: bounded TRE-only source metadata inventory; never reads participant rows.

Deploy through the HPP browser UI. All source schemas/dictionaries/results stay
inside TRE. The local copy is researcher-authored code only.
"""
import csv
import hashlib
import json
import os
import re
import resource
import socket
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

R = Path('/home/ec2-user/studies/hpp_datasets')
V = Path('/home/ec2-user/studies/diet_gut_cgm_v2')
P = V / 'multisystem_project_20261021'
L = Path('/home/ec2-user/multisystem_runtime_20261021')
PY = '/home/ec2-user/miniconda/envs/pheno/bin/python'
HISTORY = V / 'phenotype_food_dates_20261018/source_manifest.json'
CORE = {
    'food_index': 'diet_logging/diet_logging.parquet',
    'food_events': 'diet_logging/diet_logging_events.parquet',
    'food_raw_events': 'diet_logging/raw_diet_logging_events.parquet',
    'cgm_connections': 'cgm/cgm.parquet',
    'cgm_window_legacy': 'cgm/iglu.parquet',
    'cgm_daily_legacy': 'cgm/iglu_daily.parquet',
    'anthropometrics': 'anthropometrics/anthropometrics.parquet',
    'body_composition': 'body_composition/body_composition.parquet',
    'liver_detail': 'liver_ultrasound/liver_ultrasound.parquet',
    'liver_aggregated': 'liver_ultrasound/liver_ultrasound_aggregated.parquet',
    'oral_index': 'oral_microbiome/oral_microbiome.parquet',
    'oral_index_alternate': 'oral_microbiome/oral_microbiome_2025_04_UUID.parquet',
    'gut_index': 'gut_microbiome/gut_microbiome.parquet',
    'oral_species': 'oral_microbiome/aggregated/metaphlan_abundances_aggregated/species.parquet',
    'gut_species': 'gut_microbiome/abundance/metaphlan4/aggregated/abundance_species.parquet',
}
IDENTITY = {'participant_id', 'cohort', 'research_stage', 'sample_id', 'uuid',
            'connection_id', 'collection_date', 'collection_timestamp', 'local_timestamp', 'food_id'}
FOOD = {'short_food_name', 'product_name', 'food_category', 'weight_g', 'calories_kcal',
        'carbohydrate_g', 'protein_g', 'lipid_g', 'dietary_fiber_g', 'logging_day'}
TERMS = re.compile(r'age|sex|gender|birth|smok|alcohol|medicat|drug|antibio|diabet|family|relation|batch|oral|dental|periodont|fat|visceral|liver|steato|stiff|attenuat|calori|fiber|fibre|weight|nutrient|glucose', re.I)

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, ensure_ascii=False, indent=2, default=str) + '\n').encode()
    path.write_bytes(data)
    if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(data).digest():
        raise RuntimeError('readback mismatch: ' + str(path))

def csv_save(path, rows, fields):
    import io
    b = io.StringIO()
    w = csv.DictWriter(b, fieldnames=fields, extrasaction='ignore')
    w.writeheader()
    for row in rows:
        w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    data = b.getvalue().encode()
    path.write_bytes(data)
    assert path.read_bytes() == data

def inspect_one(path):
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[k] = '1'
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (35, 35))
    p = Path(path)
    st = p.stat()
    with p.open('rb') as f:
        f.seek(-8, 2)
        last = f.read(8)
        if last[4:] != b'PAR1':
            raise ValueError('unsupported parquet trailer')
        n = struct.unpack('<I', last[:4])[0]
        if n > 8 * 1024**2:
            raise ValueError('footer exceeds 8 MiB metadata budget')
        f.seek(-8-n, 2)
        footer_hash = hashlib.sha256(f.read(n)).hexdigest()
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(p, pre_buffer=False, thrift_string_size_limit=16*1024**2,
                        thrift_container_size_limit=2000000)
    fields = [{'name': x.name, 'type': str(x.type), 'nullable': x.nullable} for x in pf.schema_arrow]
    selected = [x['name'] for x in fields if x['name'] in IDENTITY | FOOD or TERMS.search(x['name'])]
    budget = 0
    for g in range(pf.metadata.num_row_groups):
        rg = pf.metadata.row_group(g)
        for j in range(rg.num_columns):
            col = rg.column(j)
            if col.path_in_schema in selected:
                budget += col.total_uncompressed_size
    return {'path': str(p), 'status': 'metadata_read', 'bytes': st.st_size,
            'mtime_ns': st.st_mtime_ns, 'rows_metadata': pf.metadata.num_rows,
            'row_groups': pf.metadata.num_row_groups, 'columns': fields,
            'footer_bytes': n, 'footer_sha256': footer_hash,
            'schema_sha256': hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest(),
            'candidate_selected_columns': selected, 'selected_uncompressed_bytes': budget,
            'participant_rows_read': 0,
            'scope': 'footer/schema only; candidate fields and units are not semantically validated',
            'worker_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}

def guarded_meta(path, work, number):
    if not path.is_file():
        return {'path': str(path), 'status': 'missing', 'participant_rows_read': 0}
    out, err = work / ('worker_%03d.json' % number), work / ('worker_%03d.err' % number)
    with out.open('w') as stdout, err.open('w') as stderr:
        proc = subprocess.Popen([sys.executable, __file__, '--one', str(path)], stdout=stdout, stderr=stderr)
        start = time.monotonic()
        stopped = None
        while proc.poll() is None:
            try:
                status = Path('/proc/%s/status' % proc.pid).read_text()
                match = re.search(r'^VmRSS:\s+(\d+)', status, re.M)
                rss = int(match.group(1)) if match else 0
                mem = Path('/proc/meminfo').read_text()
                available = int(re.search(r'^MemAvailable:\s+(\d+)', mem, re.M).group(1))
                if rss > 512*1024 or available < 1536*1024:
                    stopped = 'memory_guard'
            except (FileNotFoundError, AttributeError):
                pass
            if time.monotonic()-start > 60:
                stopped = 'metadata_timeout'
            if stopped:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                break
            time.sleep(.15)
    if proc.returncode != 0 or stopped:
        return {'path': str(path), 'status': stopped or 'metadata_error', 'exit_code': proc.returncode,
                'error_tail': err.read_text()[-1500:], 'participant_rows_read': 0}
    return json.loads(out.read_text())

def dictionary_inventory(modules):
    manifests, candidates = [], []
    for module in modules:
        meta = R / 'metadata' if module == 'metadata' else R / module / 'metadata'
        if not meta.is_dir():
            continue
        files = []
        for child in sorted(meta.iterdir())[:40]:
            if child.is_file():
                files.append(child)
            elif child.is_dir():
                files.extend(p for p in sorted(child.iterdir())[:40] if p.is_file())
        files.sort()
        for p in files[:40]:
            st = p.stat()
            record = {'path': str(p), 'module': module, 'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns}
            if p.suffix.lower() == '.csv' and st.st_size <= 4*1024**2:
                raw = p.read_bytes()
                record['sha256'] = hashlib.sha256(raw).hexdigest()
                try:
                    text = raw.decode('utf-8-sig')
                    reader = csv.DictReader(text.splitlines())
                    record['headers'] = reader.fieldnames
                    row_count = 0
                    for row in reader:
                        row_count += 1
                        if row_count > 10000:
                            record['rows_truncated'] = True
                            break
                        if TERMS.search(' '.join(str(v) for v in row.values())):
                            candidates.append({'dictionary_path': str(p), 'module': module, 'row_number': row_count+1,
                                               'definition_candidate': row,
                                               'interpretation': 'candidate dictionary evidence; not an instruction or frozen variable selection'})
                    record['dictionary_rows_examined'] = min(row_count, 10000)
                    record['status'] = 'csv_dictionary_read'
                except (UnicodeError, csv.Error) as e:
                    record['status'], record['error'] = 'dictionary_parse_error', str(e)
            else:
                record['status'] = 'listed_only_non_csv_or_size_guard'
            manifests.append(record)
    return manifests, candidates

def main():
    if sys.executable != PY:
        raise RuntimeError('Use the verified pheno Python: ' + PY)
    run = 'w02_sources_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    work = L / run
    work.mkdir(parents=True, exist_ok=False)
    for name in ('scripts', 'manifests', 'config', 'data_qc', 'memos', 'logs'):
        (P/name/run).mkdir(parents=True, exist_ok=True)
    source_bytes = Path(__file__).read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    deployed = P/'scripts'/run/'02_source_inventory.py'
    deployed.write_bytes(source_bytes)
    assert hashlib.sha256(deployed.read_bytes()).hexdigest() == source_hash
    historical = json.loads(HISTORY.read_text()) if HISTORY.is_file() else []
    historical = historical if isinstance(historical, list) else []
    by_path = {x['path']: x for x in historical if isinstance(x, dict) and 'path' in x}
    modules = sorted(p.name for p in R.iterdir() if p.is_dir())
    wanted = set(x.split('/')[0] for x in CORE.values()) | {
        'population', 'medications', 'medical_conditions', 'health_and_medical_history',
        'lifestyle_and_environment', 'sociodemographics', 'family_history', 'metadata'}
    wanted.update(x for x in modules if 'relation' in x.lower())
    covariate_modules = sorted(wanted - set(x.split('/')[0] for x in CORE.values()))
    paths = dict(CORE)
    for module in covariate_modules:
        if (R/module).is_dir():
            for p in sorted((R/module).glob('*.parquet'))[:8]:
                paths['candidate_' + module + '_' + p.stem] = str(p.relative_to(R))
    entries, deltas = [], []
    for number, (role, rel) in enumerate(paths.items()):
        entry = guarded_meta(R/rel, work, number)
        entry.update(role=role, required=role in CORE)
        old = by_path.get(entry['path'])
        compare = {}
        if old and entry['status'] == 'metadata_read':
            for key in ('bytes', 'mtime_ns', 'rows_metadata'):
                compare[key] = {'old': old.get(key), 'current': entry.get(key), 'equal': old.get(key) == entry.get(key)}
            compare['previous_projected_columns_present'] = all(c in [f['name'] for f in entry['columns']] for c in old.get('projected_columns', []))
        status = ('historical_metadata_matches_only' if compare and all(compare[k]['equal'] for k in ('bytes','mtime_ns','rows_metadata'))
                  and compare['previous_projected_columns_present'] else 'metadata_changed_review_required' if old else 'no_historical_comparator')
        entry['historical_comparison'] = compare
        entry['delta_status'] = status
        entry['quality_state'] = 'source_schema_available_semantics_and_participant_QC_pending'
        entries.append(entry)
        deltas.append({'role': role, 'path': entry['path'], 'status': entry['status'], 'delta_status': status,
                       'previous_scan_evidence_available': bool(old),
                       'inheritance': 'reference_prior_selected_column/date_checks_conditionally; metadata equality is not proof of unchanged participant values' if old else 'new_source_requires_its_own_followup_QC'})
        print('SOURCE_METADATA', role, entry['status'], 'rows', entry.get('rows_metadata'), 'delta', status, flush=True)
    ts = R/'cgm/timeseries'
    fragments = sorted(p for p in ts.iterdir() if p.is_file() and p.suffix == '.parquet') if ts.is_dir() else []
    samples = []
    for j, p in enumerate(fragments[:3]):
        samples.append(guarded_meta(p, work, 900+j))
    dictionaries, candidates = dictionary_inventory(sorted(wanted & set(modules)))
    env = {'host': socket.gethostname(), 'python': sys.executable, 'run_id': run,
           'W01': 'full_reexecution_waived_by_user; current_host_paths_and_successful_W02_output_readback_only',
           'data_root': str(R), 'project_root': str(P), 'runtime': str(work)}
    bundle = {'run_id': run, 'scope': 'W02_source_metadata_only', 'participant_rows_read': 0,
              'sources': entries, 'cgm_timeseries': {'directory': str(ts), 'parquet_file_count': len(fragments),
                  'representatives': samples, 'scope': 'directory enumeration and first 3 sorted-file footers; not full point/schema QC'},
              'dictionary_inventory': dictionaries, 'dictionary_candidate_count': len(candidates),
              'history': {'path': str(HISTORY), 'records': len(historical), 'sha256': hashlib.sha256(HISTORY.read_bytes()).hexdigest() if HISTORY.exists() else None},
              'fingerprint_limit': 'new footer/schema hashes and old metadata compare structure only; do not prove unchanged data-page values or freeze clinical semantics',
              'all_results_TRE_ONLY': True, 'raw_data_modified': False}
    manifest = P/'manifests'/run/'source_manifest.json'
    delta = P/'data_qc'/run/'source_delta.csv'
    plan = P/'config'/run/'read_plan.yaml'
    report = P/'data_qc'/run/'REPORT.md'
    memo = P/'memos'/run/'MEMO.md'
    dump(manifest, bundle)
    dump(P/'manifests'/run/'environment.json', env)
    dump(P/'data_qc'/run/'dictionary_candidates_TRE_ONLY.json', candidates)
    csv_save(delta, deltas, ['role','path','status','delta_status','previous_scan_evidence_available','inheritance'])
    read_plan = {'format': 'JSON syntax valid as YAML 1.2', 'run_id': run,
                 'next_step_only_not_executed': True, 'max_concurrent_workers': 1,
                 'metadata_worker_limits': {'timeout_s': 60, 'rss_mib': 512, 'address_space_gib': 2, 'footer_mib': 8},
                 'future_data_reads': [{'role': e['role'], 'path': e['path'],
                      'candidate_columns': e.get('candidate_selected_columns', []),
                      'projected_uncompressed_bytes': e.get('selected_uncompressed_bytes'),
                      'batch_size_candidate': 512, 'use_threads': False,
                      'gate': 'validate exact meaning, grain, identity and source-specific QC before scientific use; wide abundance tables require sample/feature-specific bounded plan'} for e in entries],
                 'covariate_selection': 'dictionary/schema candidates only; family_history is not evidence of a family clustering ID',
                 'outcome_access': 'no participant outcomes or associations read in W02',
                 'diet_feature_tables': 'local user definitions reviewed; no participant FoodID mapping or NOVA coverage verified in W02'}
    dump(plan, read_plan)
    required_errors = [e['role'] for e in entries if e['required'] and e['status'] != 'metadata_read']
    summary = {'run_id': run, 'host': env['host'], 'source_files': len(entries),
               'metadata_read': sum(e['status']=='metadata_read' for e in entries),
               'historical_compared': sum(bool(e['historical_comparison']) for e in entries),
               'historical_metadata_matches': sum(e['delta_status']=='historical_metadata_matches_only' for e in entries),
               'required_errors': required_errors, 'dictionary_files': len(dictionaries),
               'dictionary_candidates': len(candidates), 'cgm_fragment_files': len(fragments),
               'cgm_representative_metadata_read': sum(e['status']=='metadata_read' for e in samples),
               'participant_rows_read': 0, 'source_manifest': str(manifest), 'read_plan': str(plan),
               'source_delta': str(delta), 'report': str(report), 'memo': str(memo),
               'code_path': str(deployed), 'code_sha256': source_hash,
               'status': 'W02_SOURCE_INVENTORY_COMPLETE' if not required_errors else 'W02_SOURCE_INVENTORY_PARTIAL'}
    report.write_text('# W02 source inventory\n\n' + json.dumps(summary, indent=2) +
                      '\n\nMetadata only; no participant rows, final N, source semantics, microbiome technical QC, CGM point QC, model, or split has been validated.\n'
                      'Old checks are referenced conditionally. Matching bytes/mtime/rows is not proof of unchanged data values.\n')
    memo.write_text('# W02 run memo\n\nCode: '+str(deployed)+'\n\nFunction: metadata/schema inventory, history comparison, dictionary candidates and bounded future read plan.\n\n'
                    'Inputs: read-only '+str(R)+' and historical '+str(HISTORY)+'\n\n'
                    'Results: '+str(manifest.parent)+', '+str(delta.parent)+', '+str(plan)+'\n\n'
                    'Command: '+sys.executable+' -u '+str(deployed)+'\n\n'
                    'No raw participant rows or scientific models read/run. Source definitions and actual cohort counts remain pending.\n\n'
                    'Code SHA256: '+source_hash+'\n\nRun: '+run+'\nHost: '+env['host']+'\n\n'
                    'Exit code: '+('0' if not required_errors else '2')+'; inventory success is separate from data quality and scientific calibration.\n'
                    'For failures inspect TRE-local worker stderr; resume with a new run, preserving this one. No direct V2 append.\n')
    dump(P/'manifests'/run/'summary.json', summary)
    artifacts = [manifest, delta, plan, report, memo, deployed, P/'manifests'/run/'summary.json']
    evidence = [{'path': str(f), 'bytes': f.stat().st_size, 'sha256': hashlib.sha256(f.read_bytes()).hexdigest()} for f in artifacts]
    dump(P/'manifests'/run/'artifact_manifest.json', evidence)
    for f in artifacts:
        assert f.exists() and f.stat().st_size > 0
    print('W02_SUMMARY ' + json.dumps(summary, sort_keys=True), flush=True)
    print('W02_READBACK_VERIFIED',len(artifacts),'CODE_SHA256',source_hash,flush=True)
    return 0 if not required_errors else 2

if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--one':
        print(json.dumps(inspect_one(sys.argv[2]), ensure_ascii=False))
    else:
        raise SystemExit(main())
