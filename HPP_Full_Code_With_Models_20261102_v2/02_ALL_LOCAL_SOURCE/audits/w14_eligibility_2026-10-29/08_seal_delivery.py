from pathlib import Path
import json,hashlib,shutil
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');Q=P/'data_qc/w14_eligibility_20261029';L=Path('/home/ec2-user/multisystem_runtime_20261021/w14_eligibility_20261029');assert (L/'07_actual_exit_code').read_text().strip()=='0';V=json.loads((Q/'final_delivery_verification.json').read_text());assert V['passed']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for suffix in ['.log','_actual_exit_code']:
 src=L/('07'+suffix);dst=P/'logs/w14_eligibility_20261029'/src.name;assert not dst.exists();shutil.copyfile(src,dst);assert sha(src)==sha(dst)
codes=[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted((P/'scripts/w14_eligibility_20261029').glob('*.py'))];(Q/'final_code_manifest.json').write_text(json.dumps(codes,indent=2)+'\n')
for t in ['01','02','03','03b','04','05','06','07']:assert (L/(t+'_actual_exit_code')).read_text().strip()=='0'
status={'status':'W14_complete_technical_candidate_delivery','run_id':'w14_eligibility_20261029','code_scripts':len(codes),'production_artifacts':V['production_artifacts_verified'],'registry_rows':V['canonical_registry_rows'],'candidate_union_people':V['W15_union_people'],'aggregate_sha256':V['aggregate_sha256'],'prior_actual_exit_codes_all_zero':True,'next':'W15 global person/family roles and historical access ledger; W16 synthetic interfaces may prepare; no W15 role assignment performed','pending_W18_W22':['formal_exposures_and_UPF_measurement_interface','minimum_days_and_aggregation','formal_covariates_and_medication_window_choice','family_role_proportions','score_support_q_rank_and_final_N'],'human_team_review_complete':False,'models_or_associations_run':False,'current_seal_actual_exit':'check L/08_actual_exit_code produced by external runner'}
(Q/'completion_status.json').write_text(json.dumps(status,indent=2)+'\n');note='\n\nW14 FINAL STATUS\n'+json.dumps(status,indent=2)+'\n'
for p in [P/'W14_HANDOFF.md',P/'memos/w14_eligibility_20261029/MEMO.md']:
 with p.open('a') as f:f.write(note)
print('W14_SEAL_PASS',json.dumps(status),flush=True)
