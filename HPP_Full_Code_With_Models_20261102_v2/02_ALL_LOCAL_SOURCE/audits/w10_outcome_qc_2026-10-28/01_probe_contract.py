# -*- coding: utf-8 -*-
"""W10 bounded source schema and dictionary preflight; no participant rows."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,csv,re,sys,hashlib,resource
import pyarrow.parquet as pq
R=Path('/home/ec2-user/studies/hpp_datasets');P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');Q=P/'data_qc/w10_outcome_qc_20261028';L=Path('/home/ec2-user/multisystem_runtime_20261021/w10_outcome_qc_20261028')
Q.mkdir(parents=True,exist_ok=True);L.mkdir(parents=True,exist_ok=True)
print('ENV',sys.executable,flush=True)
paths=['body_composition/body_composition.parquet','anthropometrics/anthropometrics.parquet','liver_ultrasound/liver_ultrasound.parquet','liver_ultrasound/liver_ultrasound_aggregated.parquet','cgm/iglu.parquet']
term=re.compile(r'fat|visceral|adipose|bmi|height|weight|mass|quality|valid|qc|status|device|elastic|velocity|viscos|dispersion|attenuat|speed_of_sound',re.I)
report=[]
for rel in paths:
 p=R/rel;pf=pq.ParquetFile(p,pre_buffer=False);assert pf.metadata.num_rows<1000000
 schema=[{'name':f.name,'type':str(f.type),'metadata':str(f.metadata)} for f in pf.schema_arrow];print('SOURCE',rel,'rows',pf.metadata.num_rows,'cols',len(schema),flush=True)
 chosen=[x for x in schema if term.search(x['name']) or x['name'] in ['participant_id','cohort','research_stage','collection_date','collection_timestamp']]
 print('FIELDS',json.dumps(chosen),flush=True)
 rows=[];files=[]
 for d in sorted((p.parent/'metadata').glob('*')):
  files.append({'file':str(d),'bytes':d.stat().st_size})
  if d.suffix.lower()!='.csv' or d.stat().st_size>8*1024**2:continue
  with d.open(encoding='utf-8-sig',newline='') as f:
   for i,row in enumerate(csv.DictReader(f)):
    assert i<20000
    if term.search(str(row)):rows.append({'file':str(d),'row':row})
 print('META_FILES',json.dumps(files),flush=True)
 # Print relevant short dictionary rows, preserving all original evidence inside TRE.
 for row in rows:
  if term.search(str(row)):print('DICT',json.dumps(row,ensure_ascii=False),flush=True)
 report.append({'path':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'rows':pf.metadata.num_rows,'schema':schema,'dictionary_rows':rows,'metadata_files':files})
(Q/'preflight_TRE_ONLY.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
print('W10_PREFLIGHT_COMPLETE',len(report),'NO_PARTICIPANT_ROWS',flush=True)
