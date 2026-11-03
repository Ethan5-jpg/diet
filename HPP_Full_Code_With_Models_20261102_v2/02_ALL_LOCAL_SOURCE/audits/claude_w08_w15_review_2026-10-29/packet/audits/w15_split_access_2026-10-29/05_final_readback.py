"""Independent published-file readback and actual metadata reader smoke check."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib,shutil,importlib.util,resource
import pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');L=Path('/home/ec2-user/multisystem_runtime_20261021/w15_split_access_20261029');Q=P/'data_qc/w15_split_access_20261029';checks=[]
resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(240,240))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ok(n,x):
 assert x,n
 checks.append(n)
for t in ['01','02','03','04']:ok('actual_exit_'+t,(L/(t+'_actual_exit_code')).read_text().strip()=='0')
manifest=json.loads((Q/'final_artifact_manifest.json').read_text());ok('ten_artifacts',len(manifest)==10)
for m in manifest:ok('published_'+m['name'],sha(m['path'])==m['sha256'] and Path(m['path']).stat().st_size==m['bytes'])
lock=json.loads((P/'config/w15_split_access_lock.json').read_text());pre=json.loads((Q/'preflight.json').read_text());ok('manifest_lock',sha(Q/'final_artifact_manifest.json')==lock['production_manifest_sha256'])
for name,m in pre['inputs'].items():ok('unchanged_source_'+name,sha(m['path'])==m['sha256'])
active=P/'config/ACTIVE_PROTOCOL.json';a=json.loads(active.read_text());ok('active_protocol_unchanged',sha(active)==pre['active_protocol_sha256']);ok('analysis_register_unchanged',sha(a['analysis_register'])==pre['analysis_register_sha256']);ok('W14_manifest_unchanged',sha(P/'data_qc/w14_eligibility_20261029/final_artifact_manifest.json')==pre['W14_production_manifest_sha256'])
for m in json.loads((P/'data_qc/w14_eligibility_20261029/final_artifact_manifest.json').read_text()):ok('W14_artifact_'+Path(m['path']).name,sha(m['path'])==m['sha256'])
ok('seed_precommit_unchanged',sha(Q/'precommit_split_policy.json')==pre['precommit_sha256']);ok('history_declaration_unchanged',sha(Q/'user_history_declaration.json')==pre['history_sha256'])
spec=importlib.util.spec_from_file_location('published_w15_guard',P/'scripts/w15_split_access_20261029/w15_access_guard.py');guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
ids=pq.ParquetFile(P/'freeze/module_membership_TRE_ONLY.parquet',pre_buffer=False).read(columns=['analysis_id'],use_threads=False).to_pandas(ignore_metadata=True).analysis_id.value_counts().to_dict();ok('module_count',len(ids)==11 and sum(ids.values())==54151)
for aid,count in sorted(ids.items()):
 d=guard.load_blind_membership(aid,['participant_id','cohort','research_stage','split_unit_id','unit_order','person_order','initial_role']);ok('blind_reader_'+aid,len(d)==count and d.initial_role.eq('unopened_pending_design_freeze').all() and not d.duplicated(['cohort','participant_id','research_stage']).any())
ok('D_Glu_count',ids['D_Glu']==5876)
try:guard.load_blind_membership('D_Glu',['participant_id','Y_value'])
except PermissionError:checks.append('real_reader_denies_Y')
else:raise AssertionError('real reader allowed Y')
try:guard.load_blind_membership('UNREGISTERED_MODULE',['participant_id'])
except KeyError:checks.append('real_reader_denies_unknown_module')
else:raise AssertionError('real reader allowed unknown module')
s=pq.ParquetFile(P/'freeze/split_order_TRE_ONLY.parquet',pre_buffer=False).read(columns=['split_unit_id','person_order','initial_role','final_role','design_core_selected','new_scientific_Y_access_opened'],use_threads=False).to_pandas(ignore_metadata=True);ok('published_union_and_units',len(s)==6305 and s.split_unit_id.nunique()==5699);ok('all_roles_pending',s.initial_role.eq('unopened_pending_design_freeze').all() and s.final_role.isna().all());ok('no_new_access_or_core',not s.design_core_selected.any() and not s.new_scientific_Y_access_opened.any())
ag=json.loads((Q/'aggregate_delivery.json').read_text());digest=hashlib.sha256(json.dumps(ag,sort_keys=True,separators=(',',':')).encode()).hexdigest();ok('aggregate_digest',digest==(Q/'aggregate_delivery_sha256.txt').read_text().strip()==lock['aggregate_sha256']);ok('independent_BFS',json.loads((Q/'independent_verification.json').read_text())['checks']==61)
code=[]
for p in sorted((P/'scripts/w15_split_access_20261029').glob('*.py')):compile(p.read_bytes(),str(p),'exec');code.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
ok('all_five_execution_scripts_and_guard_present',len(code)==6)
for suffix in ['.log','_actual_exit_code']:
 p=Q/'logs'/('04'+suffix);assert not p.exists();shutil.copyfile(L/('04'+suffix),p)
completion={'status':'W15_complete_global_order_history_ledger_workflow_guard','actual_prior_exits':{t:0 for t in ['01','02','03','04']},'final_step':'05 actual exit recorded by launcher after this script closes','production_artifacts_verified':10,'independent_BFS_checks':61,'published_readback_checks':len(checks),'candidate_people':6305,'candidate_units':5699,'module_membership_rows':54151,'modules':11,'history_event_rows':8,'new_scientific_Y_values_read':0,'design_core_selected_people':0,'final_fraction':None,'final_roles':'unassigned_until_W18','history_exact_ID_overlap':'unknown','access_enforcement':'workflow_API_guard_not_filesystem_ACL','aggregate_sha256':digest,'check_names':checks,'module_membership_counts':{k:int(v) for k,v in ids.items()},'code_manifest':code,'no_individual_data_exported':True,'next_workpackage':'W16'}
for name,data in [('final_delivery_verification.json',completion),('completion_status.json',{k:v for k,v in completion.items() if k not in ['check_names','code_manifest','module_membership_counts']})]:
 p=Q/name;assert not p.exists();p.write_text(json.dumps(data,indent=2)+'\n');assert json.loads(p.read_text())==data
done=P/'W15_COMPLETED.md';assert not done.exists();done.write_text('# W15 complete\n\nGlobal person/family order, history ledger and workflow access guard are verified. Final ratio and roles remain W18; optional design core unactivated. Exact historical ID overlap is unknown. No newly opened scientific Y or association.\n\nVerification: data_qc/w15_split_access_20261029/final_delivery_verification.json\n\nRequire step 05 actual exit 0 in runtime before accepting this completion.\n\nAggregate SHA256: '+digest+'\n')
print('W15_FINAL_READBACK_PASS',json.dumps({k:v for k,v in completion.items() if k not in ['check_names','code_manifest']}),flush=True)
