# -*- coding: utf-8 -*-
"""W06 source contract: metadata/dictionaries and runtime, no participant rows."""
from pathlib import Path
import os,sys,json,csv,hashlib,platform,shutil,time
import pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');R=Path('/home/ec2-user/studies/hpp_datasets/diet_logging')
L=Path('/home/ec2-user/multisystem_runtime_20261021/w06_food_cleaning_20261028');L.mkdir(parents=True,exist_ok=True)
Q=P/'data_qc/w06_food_cleaning_20261028';Q.mkdir(parents=True,exist_ok=True)
assert sys.executable=='/home/ec2-user/miniconda/envs/pheno/bin/python'
rows=[]
for p in sorted(R.glob('*.parquet')):
 f=pq.ParquetFile(p,pre_buffer=False);st=p.stat()
 x={'path':str(p),'rows':f.metadata.num_rows,'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'schema':str(f.schema_arrow),'pandas_metadata':json.loads(f.schema_arrow.metadata.get(b'pandas',b'{}')) if f.schema_arrow.metadata else {},'row_groups':f.num_row_groups}
 rows.append(x)
 print('SOURCE',p.name,'ROWS',x['rows'],flush=True)
 for field in f.schema_arrow:print('FIELD',field.name,str(field.type),flush=True)
d=[]
for p in sorted((R/'metadata').glob('*.csv')):
 allrows=list(csv.DictReader(p.open()))
 hits=[x for x in allrows if any(s in str(x).lower() for s in ['timestamp','collection_date','diet_logging_events','raw_diet','logging_day','food_items','edit','delet','event_id','backfill'])]
 d.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows':allrows})
 print('DICTIONARY',p.name,flush=True)
 for x in hits:print('DICT_FIELD',json.dumps({k:v for k,v in x.items() if k in ['tabular_field_name','field_string','description_string','data_coding','field_type','units']}),flush=True)
free=shutil.disk_usage(L);mem=[s.strip() for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith(('MemTotal','MemAvailable'))]
print('RUNTIME',platform.node(),sys.executable,'DISK_FREE_GiB',round(free.free/1024**3,2),'MEM',mem,flush=True)
assert free.free>4*1024**3,'Insufficient local working disk'
prior={}
for rel in ['data_qc/w04_mapping_review_20261027/verification.json','config/w04_followup_20261027/diagnostic_policy.json','config/w03_keys_20261026T134938Z/key_contract.yaml']:
 p=P/rel;prior[rel]={'exists':p.exists(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None}
result={'date':'2026-10-28','host':platform.node(),'python':sys.executable,'sources':rows,'dictionaries':d,'prior_artifacts':prior,'runtime_free_bytes':free.free,'memory':mem,'no_participant_rows_read':True}
x=Q/'source_contract_TRE_ONLY.json';x.write_text(json.dumps(result,ensure_ascii=False,indent=2));assert json.loads(x.read_text())==result
print('W06_SOURCE_CONTRACT_DONE','RESULT',str(Q),flush=True)
