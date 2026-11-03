# -*- coding: utf-8 -*-
from pathlib import Path
import json,hashlib
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');Q=P/'data_qc/w04_mapping_review_20261027';L=Path('/home/ec2-user/multisystem_runtime_20261021/w04_mapping_review_20261027')
assert (L/'09_actual_exit_code').read_text()=='0'
p=P/'config/w04_followup_20261027/diagnostic_policy.json';old=p.read_bytes();backup=p.with_name('diagnostic_policy_before_user_source_scope.json');assert not backup.exists();backup.write_bytes(old)
v=json.loads(old);v['additional_external_mapping_sources_allowed']=False;v['allowed_sources']='current attached HPP data plus existing user-provided classification material';v['usda_candidate_status']='not found in current diet_logging module; no further external search or retrieval';p.write_text(json.dumps(v,indent=2));assert json.loads(p.read_text())==v
note='\n\n## 2026-10-27 USER_INTERNAL_SOURCES_ONLY\n\nUser confirmed the candidate was thought to be inside diet_logging. If absent, do not seek additional external sources. Use current attached HPP data and existing user-provided classification material only; unresolved classifications remain unknown. This closes the external-source search, not the NOVA coverage gap. Script 10_record_internal_source_scope.py records this decision only; no new data scan, score, threshold, exposure selection or final cohort N.\n'
for dest in [Q/'MEMO.md',P/'memos/w04_mapping_review_20261027/MEMO.md']:
 t=dest.read_text();assert 'USER_INTERNAL_SOURCES_ONLY' not in t;dest.write_text(t+note);assert dest.read_text()==t+note
x=Q/'verification.json';v=json.loads(x.read_text());v['additional_external_mapping_sources_allowed']=False;x.write_text(json.dumps(v,indent=2))
h=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
records=[{'path':str(f),'bytes':f.stat().st_size,'sha256':h(f)} for f in sorted(Q.iterdir()) if f.is_file() and f.name!='artifact_manifest.json']
(Q/'artifact_manifest.json').write_text(json.dumps(records,indent=2))
print('INTERNAL_SOURCE_SCOPE_RECORDED','external_mapping_allowed',False,'artifacts',len(records),flush=True)
print('POLICY_SHA',h(p),'MEMO_SHA',h(Q/'MEMO.md'),flush=True)
print('NO_NEW_DATA_OR_THRESHOLD_DECISION',flush=True)
