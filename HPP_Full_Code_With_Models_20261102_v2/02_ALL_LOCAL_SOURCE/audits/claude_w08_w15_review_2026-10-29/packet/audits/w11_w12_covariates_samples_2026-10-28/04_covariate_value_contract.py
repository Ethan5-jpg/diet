# -*- coding: utf-8 -*-
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,re,csv
import pandas as pd,numpy as np,pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');R=Path('/home/ec2-user/studies/hpp_datasets');Q=P/'data_qc/w11_w12_covariates_samples_20261028';J=json.loads((Q/'preflight_sources_TRE_ONLY.json').read_text());E=json.loads((Q/'preflight_dictionary_TRE_ONLY.json').read_text());out=[]
for rel in ['population/population.parquet','medications/medications.parquet','sociodemographics/ukbb.parquet','sociodemographics/initial_medical.parquet','health_and_medical_history/initial_medical.parquet','health_and_medical_history/ukbb.parquet','lifestyle_and_environment/lifestyle_and_environment.parquet']:
 entries=[s for s in J if str(s['path']).endswith('/'+rel)]
 if not entries:print('SOURCE_ABSENT',rel);continue
 s=entries[0];names=[x['name'] for x in s['schema']];print('COV_SOURCE',rel,s['rows'])
 if rel.startswith(('population/','medications/','health_and_medical_history/')):print('COV_ALL_FIELDS',[(x['name'],x['type']) for x in s['schema']])
 selected=[n for n in names if re.search(r'^(age.*|sex|gender|birth.*|year_of_birth|smoking_current_status|smoking_past_frequency|study_family_member|study_household_member|.*antibiotic.*|.*taking.*medic.*|.*medic.*current.*|.*physical_activity.*|.*alcohol.*)$',n)]
 cols=list(dict.fromkeys([c for c in ['participant_id','cohort','research_stage']+selected if c in names]));pf=pq.ParquetFile(s['path']);assert pf.metadata.num_rows<300000;d=pd.concat([b.to_pandas(ignore_metadata=True) for b in pf.iter_batches(columns=cols,batch_size=1024,use_threads=False)],ignore_index=True)
 for n in selected:
  v=d[n];vnum=pd.to_numeric(v,errors='coerce');levels=v.value_counts(dropna=False).to_dict() if v.nunique(dropna=True)<30 else None;r={'source':rel,'field':n,'nonmissing':int(v.notna().sum()),'levels':{str(k):int(v) for k,v in levels.items()} if levels is not None else None,'numeric_min':float(vnum.min()) if vnum.notna().any() else None,'numeric_max':float(vnum.max()) if vnum.notna().any() else None};out.append(r);print('COV_VALUE',json.dumps(r))
 for e in E:
  r=e['row']
  if r.get('tabular_field_name') in selected:print('COV_DEF',r.get('tabular_field_name'),r.get('units'),r.get('data_coding'),r.get('description_string','')[:180])
 print('KEYS_UNIQUE',len(d),int(d.duplicated([c for c in ['participant_id','cohort','research_stage'] if c in d]).sum()))
(Q/'covariate_value_contract_TRE_ONLY.json').write_text(json.dumps(out,indent=2));print('COV_VALUE_CONTRACT_DONE')
