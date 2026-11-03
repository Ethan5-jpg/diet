"""Compact schemas and historical execution provenance; no individual values printed."""
from pathlib import Path
import json,hashlib
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');Q=P/'data_qc/claude_readonly_followup_20261030_v1';L=Path('/home/ec2-user/multisystem_runtime_20261021')
M=json.loads((Q/'preflight.json').read_text());code=json.loads((Q/'remote_review_code_comparison.json').read_text());print('CODE_NOT_MATCHED',json.dumps([x for x in code if not x.get('matches')]))
for role in ['raw_liver','profile_numeric_QC','sample_flags','relatedness_edges','population_source','medication_timing','medication_window_status','CGM_visit','CGM_visit_days','CGM_person_connections','cgm_all_daily','food_windows']:
 m=M['inputs'][role];cols=list(m['schema']);print('COLUMNS',role,m['rows'],','.join(cols),flush=True)
r=L/'w09_cgm_point_qc_20261028';out=[]
for p in sorted(r.glob('*exit_code*')):
 s=p.read_text().strip();out.append({'name':p.name,'content':s if len(s)<30 and all(c in '-0123456789 \n' for c in s) else 'noninteger_not_exported','bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
for row in out:print('W09_EXIT_FILE',json.dumps(row),flush=True)
assert not (Q/'W09_exit_file_provenance.json').exists();(Q/'W09_exit_file_provenance.json').write_text(json.dumps(out,indent=2)+'\n');print('SCHEMA_PROVENANCE_DONE')
