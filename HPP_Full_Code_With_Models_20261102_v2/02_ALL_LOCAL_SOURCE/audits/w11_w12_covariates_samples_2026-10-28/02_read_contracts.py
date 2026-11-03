# -*- coding: utf-8 -*-
from pathlib import Path
import json,re,csv
import pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');R=Path('/home/ec2-user/studies/hpp_datasets');Q=P/'data_qc/w11_w12_covariates_samples_20261028';J=json.loads((Q/'preflight_sources_TRE_ONLY.json').read_text());E=json.loads((Q/'preflight_dictionary_TRE_ONLY.json').read_text())
print('SOURCE_SUMMARY')
for s in J:print('SRC',str(Path(s['path']).relative_to(R)),s['rows'],len(s['schema']))
print('W11_CONTRACTS')
for s in J:
 if any(t in s['path'] for t in ['oral_microbiome','gut_microbiome','sleep/']):continue
 cols=[x for x in s['schema'] if re.search(r'participant|cohort|research_stage|array_index|date|timestamp|(?:^|_)(age|sex|gender|birth|smok|alcohol|medic|drug|atc|family|household|relative|batch|activ)',x['name'],re.I)]
 print('FIELDS',Path(s['path']).name,json.dumps(cols))
print('W12_CONTRACTS')
for s in J:
 if not any(t in s['path'] for t in ['oral_microbiome','gut_microbiome']):continue
 cols=[x for x in s['schema'] if re.search(r'participant|cohort|stage|sample|uuid|connection|date|timestamp|array_index|quality|valid|qc|flag|reads|depth|batch',x['name'],re.I)]
 print('SAMPLE_FIELDS',str(Path(s['path']).relative_to(R)),json.dumps(cols))
print('FOCUSED_DICTIONARY')
for e in E:
 row=e['row'];n=row.get('tabular_field_name','')
 if re.search(r'^(age|sex|gender|birth_date|birth_year|date_of_birth|smoking_status|smoking|study_family_member.*|study_household_member.*)$|^medication|^drug|^atc|^smoking_',n,re.I):print('DEF',n,row.get('units'),row.get('data_coding'),row.get('description_string','')[:160])
print('METADATA_FILES')
for mod in ['metadata','population','medications','medical_conditions','health_and_medical_history','lifestyle_and_environment','sociodemographics','oral_microbiome','gut_microbiome']:
 d=R/mod if mod=='metadata' else R/mod/'metadata'
 if d.is_dir():print('META',mod,[(p.name,p.stat().st_size) for p in d.iterdir() if p.is_file()])
print('CONTRACT_VIEW_DONE')
