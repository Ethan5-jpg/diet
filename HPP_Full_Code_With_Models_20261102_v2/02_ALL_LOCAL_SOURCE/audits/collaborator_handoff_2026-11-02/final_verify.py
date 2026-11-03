from pathlib import Path
import json,hashlib,subprocess,sys
import numpy as np
import pyarrow.parquet as pq
O=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1')
L=Path('/home/ec2-user/multisystem_runtime_20261021/collaborator_handoff_20261102_v1')
for name in ['actual_exit_code.txt','actual_exit_code_processed.txt']:
 p=L/name
 if not p.exists():
  print('PENDING',name);print((L/'processed.log').read_text()[-2500:]);raise SystemExit(2)
 if p.read_text().strip()!='0':
  print('FAILED',name,p.read_text());print((L/'processed.log').read_text()[-5000:]);raise SystemExit(1)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
payload=json.loads((O/'manifests/payload_sha256_final.json').read_text())
for entry in payload:
 p=O/entry['relative_path'];assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256']
copies=json.loads((O/'manifests/source_copies.json').read_text())
assert all(sha(Path(i['source']))==i['sha256'] for i in copies)
assert not (O/'results').exists() and not (O/'code').exists()
pyfiles=sorted(str(p.relative_to(O)) for p in O.rglob('*.py'))
assert len(pyfiles)==4,pyfiles
assert not any('model' in p.name.lower() or 'predictor' in p.name.lower() for p in O.rglob('*') if p.is_file())
parquets=list(O.rglob('*.parquet'))
for p in parquets:
 names=pq.ParquetFile(p,pre_buffer=False).schema_arrow.names
 assert not any('score' in c.lower() or 'predict' in c.lower() for c in names),str(p)
tables=list((O/'inputs/analysis_ready/full').rglob('*.parquet'));assert len(tables)==18
for p in tables:
 d=pq.ParquetFile(p,pre_buffer=False).read(columns=['participant_id','Y_processed','input_row']).to_pandas(ignore_metadata=True)
 assert d.participant_id.is_unique and d.input_row.is_unique and np.isfinite(d.Y_processed).all()
status=dict(status='PASS',base_actual_exit=0,processed_actual_exit=0,
 payload_files=len(payload),payload_bytes=sum(i['bytes'] for i in payload),source_hashes_unchanged=len(copies),
 analysis_ready_tables=18,fold_outcome_tables=54,microbiome_parquet_views=12,
 no_predictive_models=True,no_scores_or_predictions=True,python_files=pyfiles,
 output=str(O),first_read='00_START_HERE_数据说明.md',python=sys.executable)
with (L/'final_verification.json').open('x') as f:f.write(json.dumps(status,indent=2,ensure_ascii=False)+'\n')
print('FINAL_VERIFICATION',json.dumps(status,ensure_ascii=False),flush=True)
print('READY_COUNTS\n'+(O/'manifests/analysis_ready_counts.csv').read_text(),flush=True)
print('PREPROCESSING_LOG\n'+(L/'processed.log').read_text()[:1000],flush=True)
for n in ['runner_pid.txt','worker_pid.txt','processed_runner_pid.txt','processed_worker_pid.txt']:
 print(n,(L/n).read_text().strip())
print('TOP_LEVEL',', '.join(sorted(p.name for p in O.iterdir())))
print('FIRST_READ',str(O/'00_START_HERE_数据说明.md'))
print('DATA_ONLY_HANDOFF_VERIFIED_EXIT_0')
