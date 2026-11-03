# -*- coding: utf-8 -*-
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,csv,re
from collections import Counter
import pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');R=Path('/home/ec2-user/studies/hpp_datasets');Q=P/'data_qc/w11_w12_covariates_samples_20261028';J=json.loads((Q/'preflight_sources_TRE_ONLY.json').read_text());E=json.loads((Q/'preflight_dictionary_TRE_ONLY.json').read_text());V=json.loads((Q/'covariate_value_contract_TRE_ONLY.json').read_text());coding=list(csv.DictReader((R/'metadata/coding_mapping.csv').open(encoding='utf-8-sig',newline='')))
for s in J:
 rel=str(Path(s['path']).relative_to(R));names=[x['name'] for x in s['schema']]
 if rel.startswith(('population/','medications/')):print('CORE_SCHEMA',rel,s['rows'],json.dumps(s['schema']))
 if rel.startswith('lifestyle_and_environment'):print('LIFE_FEATURES',','.join(n for n in names if re.search('smoking_current_status|physical|exercise|activity|sleep|alcohol_frequency|alcohol_current|dental|teeth|gum',n)))
 if rel.startswith('health_and_medical_history'):print('HEALTH_RELEVANT_FIELDS',rel,','.join(n for n in names if re.search('medic|drug|antibio|diabet|treat|insulin',n)))
 if rel=='sociodemographics/ukbb.parquet':
  cols=[n for n in names if n.startswith(('study_family','study_household'))];print('FAMILY_SCHEMA',[(x['name'],x['type']) for x in s['schema'] if x['name'] in cols]);d=pq.ParquetFile(s['path']).read(columns=cols,use_threads=False).to_pylist();print('FAMILY_CELL_TYPES',{c:dict(Counter(type(r[c]).__name__ for r in d)) for c in cols})
for v in V:
 if v['field'] in ['sex','gender','smoking_current_status','study_family_member','study_household_member'] or v['source'].startswith('population/'):print('CORE_VALUE',json.dumps(v))
for e in E:
 r=e['row'];n=r.get('tabular_field_name','')
 if n in ['smoking_current_status','smoking_past_frequency','alcohol_current','alcohol_frequency'] or (e['file'].split('/')[-3]=='population'):
  code=r.get('data_coding','');print('CORE_DEF',n,r.get('units'),code,r.get('description_string','')[:100]);print('CORE_CODING',code,[(c['coding'],c['english']) for c in coding if c['code_number']==str(code)])

S=json.loads((Q/'sample_identity_probe_TRE_ONLY.json').read_text())
for r in S['sources']:print('W12_SHORT',r['role'],r['rows'],'valid',r['valid_key_rows'],'unique',r['unique_valid_person_visits'],'multi',r['multirow_person_visits'],'base',r['baseline_people'],'identitycols',','.join(c for c in r['identity_columns'] if not c.startswith(('k__','d__'))))
print('FOCUSED_DONE')
