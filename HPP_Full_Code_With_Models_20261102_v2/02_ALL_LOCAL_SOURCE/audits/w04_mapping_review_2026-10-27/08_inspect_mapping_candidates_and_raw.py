# -*- coding: utf-8 -*-
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,ast,time,resource
from pathlib import Path
from collections import defaultdict
import pandas as pd
import pyarrow.parquet as pq
resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');Q=P/'data_qc/w04_mapping_review_20261027'
j=json.loads((Q/'existing_food_dictionary_sources_TRE_ONLY.json').read_text())
print('SEARCH_ENTRIES',j['entries'],'COMPLETE',j['complete_within_depth'],flush=True)
for b in j['builders']:
 if 'diet_indicator' not in b['path']:continue
 for x in b['arguments']:print('DIET_BUILDER_ARG',json.dumps(x),flush=True)
 for x in b['IO_lines']:print('DIET_BUILDER_IO',x['line'],x['text'],flush=True)
candidates=[]
for r in j['files']:
 cols=r.get('columns',[])
 if any('food_id' in c.lower() for c in cols) and (any('nova' in c.lower() for c in cols) or any('canonical_short' in c.lower() for c in cols)):
  candidates.append(r)
  print('FOODID_MAPPING_CANDIDATE',r['path'],'BYTES',r['bytes'],'ROWS',r.get('rows'),'COLS',json.dumps(cols),flush=True)
(Q/'specific_foodid_mapping_candidates_TRE_ONLY.json').write_text(json.dumps(candidates,indent=2))
# Food-label metadata can be audited without assigning raw events to any visit.
tree=ast.parse((P/'scripts/w04_mapping_review_20261027/03_audit_event_mapping_days.py').read_text())
cat=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CATALOG' for t in n.targets))
norm=lambda x:str(x).strip().casefold() if pd.notna(x) else ''
mp={norm(n):v for n,v in cat};raw={};n=0;start=time.monotonic()
r=Path('/home/ec2-user/studies/hpp_datasets/diet_logging/raw_diet_logging_events.parquet');f=pq.ParquetFile(r,pre_buffer=False)
before=(r.stat().st_size,r.stat().st_mtime_ns)
budget=sum(f.metadata.row_group(g).column(c).total_uncompressed_size for g in range(f.num_row_groups) for c in range(f.metadata.num_columns) if f.metadata.row_group(g).column(c).path_in_schema in ['food_id','short_food_name','food_category'])
assert budget<1536*1024**2
for b in f.iter_batches(batch_size=8192,columns=['food_id','short_food_name','food_category'],use_threads=False):
 d=b.to_pandas(ignore_metadata=True);n+=len(d)
 for fid,name,c in d.drop_duplicates().itertuples(index=False,name=None):
  if pd.isna(fid):continue
  z=raw.setdefault(str(int(fid)),{'names':set(),'categories':set()})
  if pd.notna(name):z['names'].add(str(name))
  if pd.notna(c):z['categories'].add(str(c))
 assert len(raw)<100000
 if time.monotonic()-start>600:raise RuntimeError('Raw food-label scan time bound')
assert n==6485022 and before==(r.stat().st_size,r.stat().st_mtime_ns)
valid={k:next(iter(v)) for k,z in raw.items() if len(v:={mp.get(norm(x)) for x in z['names']})==1 and None not in v}
processed=json.loads((Q/'food_id_name_category_candidates_TRE_ONLY.json').read_text())
oldknown={str(int(float(x['food_id']))) for x in processed if x['all_names_have_valid_NOVA']}
newkeys=set(valid)-oldknown
summary={'raw_rows_scanned':n,'raw_food_IDs':len(raw),'valid_NOVA_raw_IDs':len(valid),'additional_raw_NOVA_IDs_vs_processed':len(newkeys),'raw_IDs_with_multiple_nonmissing_names':sum(len(z['names'])>1 for z in raw.values()),'raw_stage_not_imputed':True,'no_raw_intake_pooled':True,'mapping_candidates':len(candidates)}
(Q/'raw_food_label_audit_summary.json').write_text(json.dumps(summary,indent=2))
(Q/'raw_food_label_dictionary_TRE_ONLY.json').write_text(json.dumps({k:{'names':sorted(z['names']),'categories':sorted(z['categories'])} for k,z in raw.items()},ensure_ascii=False))
print('RAW_FOOD_LABEL_AUDIT',json.dumps(summary),flush=True)
print('CANDIDATE_AND_RAW_AUDIT_DONE',flush=True)
