"""Inspect current TRE contracts and role metadata, without scientific Y reads."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
from collections import Counter
import datetime, hashlib, json, resource, shutil, sys
import numpy as np
import pyarrow.parquet as pq

P = Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
L = Path('/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1')
Q = P/'data_qc/preprocessing_20261030_v1'
resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (240, 240))
assert sys.executable == '/home/ec2-user/miniconda/envs/pheno/bin/python'
assert P.is_dir() and shutil.disk_usage(L).free > 2*1024**3
Q.mkdir(exist_ok=False)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()

active_path = P/'config/ACTIVE_PROTOCOL.json'
active = json.loads(active_path.read_text())
register_path = Path(active['analysis_register'])
register = json.loads(register_path.read_text())
snapshots = []
for p in [active_path, register_path, P/'config/access_policy.yaml',
          P/'config/w13_sample_selection_lock.json', P/'config/w15_split_access_lock.json']:
    snapshots.append({'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)})

verified = 0
for run in ['w13_microbiome_qc_20261029', 'w14_eligibility_20261029', 'w15_split_access_20261029']:
    p = P/'data_qc'/run/'final_artifact_manifest.json'
    manifest = json.loads(p.read_text())
    assert isinstance(manifest, list), 'unexpected prior manifest structure'
    snapshots.append({'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)})
    for entry in manifest:
        f = Path(entry['path'])
        assert sha(f) == entry['sha256'], 'prior published artifact changed: '+str(f)
        verified += 1

order_path = P/'freeze/split_order_TRE_ONLY.parquet'
order_file = pq.ParquetFile(order_path, pre_buffer=False)
assert {'initial_role','final_role'} <= set(order_file.schema_arrow.names)
initial, final = Counter(), Counter()
for batch in order_file.iter_batches(columns=['initial_role','final_role'], batch_size=1024, use_threads=False):
    d = batch.to_pydict()
    initial.update(d['initial_role'])
    final.update('UNASSIGNED' if x is None else str(x) for x in d['final_role'])
assert sum(final.values()) == order_file.metadata.num_rows

inputs = {}
for name in ['oral_selected_abundance_TRE_ONLY.parquet', 'gut_selected_abundance_TRE_ONLY.parquet',
             'sample_registry_TRE_ONLY.parquet', 'outcome_registry_TRE_ONLY.parquet',
             'cgm_visit_phenotypes_TRE_ONLY.parquet']:
    p = P/'analysis/shared'/name
    if not p.exists():
        inputs[name] = {'exists': False}
        continue
    f = pq.ParquetFile(p, pre_buffer=False)
    inputs[name] = {'path': str(p), 'exists': True, 'rows': f.metadata.num_rows,
                    'columns': f.schema_arrow.names, 'bytes': p.stat().st_size,
                    'sha256': sha(p)}
    print('INPUT_SCHEMA', name, f.metadata.num_rows, json.dumps(f.schema_arrow.names[:24]), flush=True)

summary = {'status': 'metadata_preflight_complete', 'time_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'python': sys.version, 'numpy': np.__version__, 'protocol_version': active['version'],
           'candidate_people': order_file.metadata.num_rows, 'initial_roles': dict(initial), 'final_roles': dict(final),
           'split_fraction': register.get('split_policy',{}).get('current_split_fraction'),
           'prior_artifacts_verified': verified, 'new_scientific_Y_reads': 0,
           'real_preprocessing_fitted': False, 'roles_frozen': 'UNASSIGNED' not in final,
           'inputs': inputs, 'snapshots': snapshots}
for name, value in [('preflight_TRE_ONLY.json', summary), ('before_artifact_hashes.json', snapshots)]:
    path = Q/name
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
print('PREPROCESS_PREFLIGHT', json.dumps({k:v for k,v in summary.items() if k not in ['inputs','snapshots','python']}), flush=True)
