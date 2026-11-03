# -*- coding: utf-8 -*-
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,re,sys
import numpy as np,pandas as pd,pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');Q=P/'data_qc/w10_outcome_qc_20261028';J=json.loads((Q/'preflight_TRE_ONLY.json').read_text());K=['participant_id','cohort','research_stage']
select={'body_composition.parquet':['body_comp_total_fat_mass','body_comp_total_region_percent_fat','body_comp_total_tissue_percent_fat','total_scan_vat_volume'],'liver_ultrasound.parquet':['elasticity_qbox_mean','velocity_qbox_mean','viscosity_qbox_mean','dispersion_qbox_mean','attenuation_coefficient_qbox','speed_of_sound_qbox']}
allresults=[]
for s in J:
 name=Path(s['path']).name
 if name not in select:continue
 cols=select[name];names=[x['name'] for x in s['schema']];print('SOURCE',name,'rows',s['rows'],'SELECTED',cols);assert set(K+cols)<=set(names)
 print('QC_FIELDS',[(x['name'],x['type']) for x in s['schema'] if re.search('qc|quality|valid|status|flag|device|protocol|reject|success',x['name'],re.I)])
 print('TIME_FIELDS',[(x['name'],x['type']) for x in s['schema'] if re.search('date|time',x['name'],re.I)])
 for e in s['dictionary_rows']:
  r=e['row'];n=r.get('tabular_field_name','')
  if n in cols:print('UNIT',n,r.get('units'),r.get('description_string','')[:180])
 d=pd.concat([b.to_pandas(ignore_metadata=True) for b in pq.ParquetFile(s['path']).iter_batches(columns=K+cols,batch_size=2048,use_threads=False)],ignore_index=True);base=d.loc[d.cohort.eq('10k')&d.research_stage.eq('00_00_visit')]
 print('BASE_SOURCE',len(base),base.participant_id.nunique(),'GROUP_SIZE',base.groupby(K).size().value_counts().sort_index().to_dict())
 for c in cols:
  v=pd.to_numeric(d[c],errors='coerce');a=v[np.isfinite(v)];b=pd.to_numeric(base[c],errors='coerce');r={'field':c,'all_nonmissing':int(d[c].notna().sum()),'all_nonfinite_notmissing':int((v.notna()&~np.isfinite(v)).sum()),'negative':int(a.lt(0).sum()),'zero':int(a.eq(0).sum()),'q':{str(k):float(v) for k,v in a.quantile([0,.01,.5,.99,1]).items()},'baseline_present_rows':int(b.notna().sum()),'baseline_present_people':int(base.loc[b.notna(),'participant_id'].nunique()),'baseline_positive_people':int(base.loc[np.isfinite(b)&b.gt(0),'participant_id'].nunique())};allresults.append(r);print('FINAL_DIAG',json.dumps(r))
(Q/'final_field_preflight_TRE_ONLY.json').write_text(json.dumps(allresults,indent=2));print('FINAL_FIELD_PROBE_DONE')
