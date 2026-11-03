# -*- coding: utf-8 -*-
from pathlib import Path
import json,csv,re
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');R=Path('/home/ec2-user/studies/hpp_datasets');Q=P/'data_qc/w11_w12_covariates_samples_20261028';J=json.loads((Q/'preflight_sources_TRE_ONLY.json').read_text());E=json.loads((Q/'preflight_dictionary_TRE_ONLY.json').read_text())
for s in J:
 rel=str(Path(s['path']).relative_to(R))
 if any(t in rel for t in ['sleep/','gut_microbiome','oral_microbiome']):continue
 names=[f['name'] for f in s['schema']];cols=[n for n in names if re.search(r'^(age|sex|gender|birth|year_of_birth)|smoking|medication|drug|atc|family_member|household_member|^collection|participant_id|cohort|stage|array_index',n,re.I)]
 print('W11SRC',rel,s['rows'],len(names),','.join(cols))
for s in J:
 rel=str(Path(s['path']).relative_to(R))
 if any(t in rel for t in ['gut_microbiome','oral_microbiome']):print('W12SRC',rel,s['rows'],json.dumps([x for x in s['schema'] if re.search('participant|cohort|stage|sample|uuid|date|timestamp|array_index|quality|reads|batch|depth|file|path',x['name'],re.I)]))
for e in E:
 r=e['row'];n=r.get('tabular_field_name','')
 if re.search(r'^(age|sex|gender|birth_date|birth_year|smoking|smoking_status|smoking_previous|smoking_current|medication|atc3|atc4|atc5|study_family_member|study_family_member_participant_id|study_household_member|study_household_member_participant_id)$',n):print('DEFINITION',n,r.get('data_coding'),r.get('relative_location'),r.get('description_string','')[:150])
f=R/'metadata/coding_mapping.csv'
with f.open(encoding='utf-8-sig',newline='') as h:
 rd=csv.DictReader(h);print('CODING_HEADERS',rd.fieldnames);a=list(rd)
print('CODING_9_OR_7',[r for r in a if any(str(v) in ['9','7'] for v in list(r.values())[:1])]);print('COMPACT_CONTRACT_DONE')
