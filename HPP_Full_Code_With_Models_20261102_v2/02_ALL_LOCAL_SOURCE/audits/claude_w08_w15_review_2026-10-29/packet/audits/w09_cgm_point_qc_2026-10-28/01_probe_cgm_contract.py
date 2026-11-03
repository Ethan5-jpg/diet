# -*- coding: utf-8 -*-
"""W09 inspect schema, units, refs, bounded representative cadence. No individual export."""
import sys,os,json,csv,hashlib,time,shutil,resource
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');R=Path('/home/ec2-user/studies/hpp_datasets/cgm');Q=P/'data_qc/w09_cgm_point_qc_20261028';L=Path('/home/ec2-user/multisystem_runtime_20261021/w09_cgm_point_qc_20261028');Q.mkdir(parents=True,exist_ok=True);L.mkdir(parents=True,exist_ok=True)
assert sys.executable=='/home/ec2-user/miniconda/envs/pheno/bin/python';assert shutil.disk_usage(L).free>4*1024**3
metadata=[]
for p in sorted(R.glob('*.parquet')):
 f=pq.ParquetFile(p,pre_buffer=False);metadata.append({'file':str(p),'rows':f.metadata.num_rows,'schema':str(f.schema_arrow),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns})
 print('SOURCE',p.name,'ROWS',f.metadata.num_rows,flush=True)
 for c in f.schema_arrow:print('FIELD',c.name,str(c.type),flush=True)
dictionaries=[]
for p in sorted((R/'metadata').rglob('*.csv')):
 rows=list(csv.DictReader(p.open()));dictionaries.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows':rows})
 print('DICTIONARY',p.name,flush=True)
 for row in rows:
  if any(x in str(row).lower() for x in ['glucose','connection','time','unit','sensor','device','libre']):print('DICT',json.dumps(row,ensure_ascii=False),flush=True)
parts=sorted((R/'timeseries').glob('*.parquet'));assert 0<len(parts)<20000
(Q/'fragment_paths_TRE_ONLY.json').write_text(json.dumps([str(p) for p in parts]))
print('FRAGMENTS',len(parts),'BYTES',sum(p.stat().st_size for p in parts),flush=True)
samples=[]
for i in sorted({0,len(parts)//2,len(parts)-1}):
 p=parts[i];f=pq.ParquetFile(p,pre_buffer=False);assert f.metadata.num_rows<10000;d=f.read().to_pandas(ignore_metadata=True);x={'ordinal':i,'rows':len(d),'schema':str(f.schema_arrow)}
 print('REPRESENTATIVE',i,'ROWS',len(d),'SCHEMA',str(f.schema_arrow).split('-- schema metadata --')[0],flush=True)
 if 'collection_timestamp' in d:
  ts=pd.to_datetime(d.collection_timestamp,errors='coerce');diff=ts.sort_values().diff().dt.total_seconds();x['interval_seconds_top']={str(k):int(v) for k,v in diff.value_counts().head(6).items()};print('INTERVAL_SECONDS_TOP',json.dumps(x['interval_seconds_top']),flush=True)
 if 'glucose' in d:
  g=pd.to_numeric(d.glucose,errors='coerce');x['glucose_aggregate']={'n':int(g.notna().sum()),'min':float(g.min()),'median':float(g.median()),'max':float(g.max())};print('GLUCOSE_AGGREGATE',json.dumps(x['glucose_aggregate']),flush=True)
 samples.append(x)
result={'source_metadata':metadata,'dictionaries':dictionaries,'fragment_count':len(parts),'bounded_representatives':samples,'full_point_QC_performed':False,'python':sys.executable,'free_GiB':shutil.disk_usage(L).free/1024**3}
(Q/'source_contract_TRE_ONLY.json').write_text(json.dumps(result,indent=2));print('W09_CONTRACT_PROBE_DONE','FREE_GIB',result['free_GiB'],flush=True)
