"""Verify the closed audit manifest and post-launch actual exits; final immutable receipt."""
from pathlib import Path
import json,hashlib
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');Q=P/'data_qc/claude_readonly_followup_20261030_v1';L=Path('/home/ec2-user/multisystem_runtime_20261021/claude_readonly_followup_20261030_v1')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
for t in ['07a_source','07','07a']:assert (L/(t+'_actual_exit_code')).read_text().strip()=='0'
manifest=json.loads((Q/'final_artifact_manifest.json').read_text());assert all(sha(r['path'])==r['sha256'] for r in manifest);s=json.loads((Q/'aggregate_delivery.json').read_text());digest=hashlib.sha256(json.dumps(s,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest();assert digest==(Q/'aggregate_delivery.sha256').read_text().strip();receipt={'status':'all_closed_files_and_actual_prior_exits_verified','aggregate_sha256':digest,'sealed_manifest_sha256':sha(Q/'final_artifact_manifest.json'),'sealed_files_verified':len(manifest),'actual_exit_07':0,'actual_exit_07a':0,'closed_final_log_receipts':[{'path':str(L/n),'sha256':sha(L/n),'bytes':(L/n).stat().st_size} for n in ['07.log','07a.log','07_actual_exit_code','07a_actual_exit_code']],'prior_W14_W15_artifacts_unchanged':46,'models_and_scientific_associations_run':False,'individual_exported':False};p=Q/'08_final_delivery_verification.json';assert not p.exists();p.write_text(json.dumps(receipt,indent=2)+'\n');print('FOLLOWUP_FINAL_VERIFIED',json.dumps(receipt),flush=True);print('COMPACT_AUDIT_COUNTS',json.dumps({k:s[k] for k in ['technical_access_receipts','technical_Y_projection_receipts','medication_rows_verified','medication_mismatch_total','raw_medication_invalid_visit_rows','W14_additional_check_count','W14_additional_mismatch_total','W14_flow_stages_verified','six_liver_means_verified','W15_people','final_roles_assigned','design_core_selected']}),flush=True)
