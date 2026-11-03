# -*- coding: utf-8 -*-
"""Bounded projected univariate diagnostics. All source/individual content stays TRE."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,re,sys
from pathlib import Path
import numpy as np,pandas as pd,pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');Q=P/'data_qc/w10_outcome_qc_20261028';J=json.loads((Q/'preflight_TRE_ONLY.json').read_text());out=[]
for s in J[:2]:
 print('SOURCE',Path(s['path']).name,'rows',s['rows'])
 names=[x['name'] for x in s['schema']]
 print('ALL_FIELDS',','.join(names))
 for e in s['dictionary_rows']:
  r=e['row'];n=r.get('tabular_field_name','')
  if re.search(r'total|visceral|vat|bmi|height|weight|quality|qc|valid',n,re.I):
   print('DICT_SHORT',n,r.get('units'),r.get('field_string'),r.get('description_string','')[:250])
 chosen=[n for n in names if re.search(r'total.*fat|fat.*total|visceral|vat|bmi|^height|^weight|quality|qc|valid|status|^device$',n,re.I)]
 cols=list(dict.fromkeys([n for n in ['participant_id','cohort','research_stage']+chosen if n in names]));pf=pq.ParquetFile(s['path'],pre_buffer=False);d=pd.concat([b.to_pandas() for b in pf.iter_batches(columns=cols,batch_size=2048,use_threads=False)],ignore_index=True)
 print('STAGES',d.groupby(['cohort','research_stage'],dropna=False).size().to_dict())
 for c in chosen:
  v=pd.to_numeric(d[c],errors='coerce');a=v[np.isfinite(v)];r={'field':c,'n_nonmissing':int(d[c].notna().sum()),'n_numeric':len(a),'negative':int(a.lt(0).sum()),'zero':int(a.eq(0).sum()),'quantiles':a.quantile([0,.01,.5,.99,1]).to_dict() if len(a) else {},'non_numeric_levels':d[c].value_counts(dropna=False).astype(int).to_dict() if len(a)==0 else {}}
  print('VALUE_DIAG',json.dumps(r,default=str));out.append(r)
 print('KEY_DUPLICATE_ROWS',int(d.duplicated(['participant_id','cohort','research_stage'],keep=False).sum()))
(Q/'value_preflight_TRE_ONLY.json').write_text(json.dumps(out,indent=2,default=str))
print('W10_VALUE_PREFLIGHT_DONE')
