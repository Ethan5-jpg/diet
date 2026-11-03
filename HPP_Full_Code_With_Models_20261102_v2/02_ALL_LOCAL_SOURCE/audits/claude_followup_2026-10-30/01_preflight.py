"""TRE read-only follow-up preflight. Only new audit files; no existing table mutation."""
import os
for n in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
from pathlib import Path
import json,hashlib,resource,sys,socket,platform,shutil
import pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');R=Path('/home/ec2-user/studies/hpp_datasets');RUN='claude_readonly_followup_20261030_v1';Q=P/'data_qc'/RUN;L=Path('/home/ec2-user/multisystem_runtime_20261021')/RUN
assert sys.executable=='/home/ec2-user/miniconda/envs/pheno/bin/python';resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(240,240));mem=int(dict(s.split(':',1) for s in Path('/proc/meminfo').read_text().splitlines())['MemAvailable'].split()[0]);assert mem>3*1024**2;assert shutil.disk_usage(L.parent).free>2*1024**3
Q.mkdir(exist_ok=False);L.mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(name,obj):
 p=Q/name;assert not p.exists();p.write_text(json.dumps(obj,indent=2,default=str)+'\n')
A=json.loads((P/'config/ACTIVE_PROTOCOL.json').read_text());assert A['version']=='multisystem_v3_user_amendment_20261029';lock=A['W13_sample_selection_lock'];assert sha(lock['path'])==lock['sha256'];base={};verified=[]
for wk,filename in [('w14_eligibility_20261029','final_artifact_manifest.json'),('w15_split_access_20261029','final_artifact_manifest.json')]:
 m=P/'data_qc'/wk/filename;base[str(m)]=sha(m)
 for row in json.loads(m.read_text()):assert sha(row['path'])==row['sha256'];verified.append({'path':row['path'],'sha256':row['sha256']})
assert len(verified)==46
J=json.loads((P/'data_qc/w14_eligibility_20261029/preflight_TRE_ONLY.json').read_text());inputs={k:Path(v['path']) for k,v in J['inputs'].items()};inputs.update({'raw_liver':R/'liver_ultrasound/liver_ultrasound.parquet','family_source':R/'sociodemographics/ukbb.parquet','population_source':R/'population/population.parquet','relatedness_edges':P/'data_qc/w11_w12_covariates_samples_20261028/relatedness_edges_TRE_ONLY.parquet','sample_flags':P/'analysis/shared/microbiome_sample_flags_TRE_ONLY.parquet','sample_registry':P/'analysis/shared/sample_registry_TRE_ONLY.parquet','sample_candidates':P/'analysis/shared/sample_candidates_TRE_ONLY.parquet','profile_numeric_QC':P/'data_qc/w13_microbiome_qc_20261029/profile_numeric_qc_TRE_ONLY.parquet','oral_index_projection':P/'data_qc/w13_microbiome_qc_20261029/oral_index_projection_TRE_ONLY.parquet','gut_index_projection':P/'data_qc/w13_microbiome_qc_20261029/gut_index_projection_TRE_ONLY.parquet','eligibility_registry':P/'analysis/eligibility_registry_TRE_ONLY.parquet','medication_window_status':P/'analysis/shared/medication_window_status_TRE_ONLY.parquet','W15_order':P/'freeze/split_order_TRE_ONLY.parquet','W15_related_membership':P/'freeze/known_related_membership_TRE_ONLY.parquet','CGM_visit':P/'analysis/shared/cgm_visit_phenotypes_TRE_ONLY.parquet','CGM_visit_days':P/'analysis/shared/cgm_visit_qualified_daily_TRE_ONLY.parquet','CGM_person_connections':P/'analysis/shared/cgm_person_connection_selection_TRE_ONLY.parquet','Daily_UPF':P/'analysis/Daily_CGM/UPF_bounds/base_table_TRE_ONLY.parquet','Daily_fiber':P/'analysis/Daily_CGM/fiber_density/base_table_TRE_ONLY.parquet'})
metadata={}
for role,p in inputs.items():
 assert p.is_file(),(role,str(p));f=pq.ParquetFile(p,pre_buffer=False);metadata[role]={'path':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'rows':f.metadata.num_rows,'schema':{x.name:str(x.type) for x in f.schema_arrow},'sha256':sha(p)}
 if role not in J['inputs'] or role in ['medication_timing','food_daily','cgm_all_daily','cgm_connections']:print('SCHEMA',role,f.metadata.num_rows,json.dumps(metadata[role]['schema']),flush=True)
expected_file=P/'scripts'/RUN/'expected_review_code_hashes.json';expected=json.loads(expected_file.read_text());code=[]
for x in expected:
 p=P/'scripts'/x['run']/x['name'];r={'run':x['run'],'script':x['name'],'exists':p.is_file(),'expected_sha256':x['sha256']}
 if p.is_file():r['actual_sha256']=sha(p);r['matches']=r['actual_sha256']==x['sha256']
 code.append(r)
save('remote_review_code_comparison.json',code)
request={'version':RUN,'purpose':'technical_QC_only_no_association_or_training','cases':['X-02_liver_duplicates','X-03_identical_profiles','X-04_family_cohort_identity','X-05_medication_evidence_and_independent_merge','X-06_W14_missing_verifier_coverage','X-07_depth_metadata','X-09_food_window_joint_description','X-10_W09_exit_provenance','X-12_recovered_lags'],'scope':'inspect current published data; no source/candidate/role/sample exclusion or protocol changes; all individual diagnostics stay TRE','access':'each subsequent projection records a new immutable technical-access receipt before reading; may read Y-source values only for predeclared numerical QC; no scientific association results opened','model_or_association_run':False,'individual_exported':False,'existing_artifacts_verified':len(verified),'source_state':'preflight schema and integrity only; no Parquet value projection in this step'}
save('audit_scope.json',request);save('preflight.json',{'inputs':metadata,'active_protocol_sha256':sha(P/'config/ACTIVE_PROTOCOL.json'),'analysis_register_sha256':sha(A['analysis_register']),'sample_lock_sha256':lock['sha256'],'prior_manifests':base,'verified_existing_artifacts':verified,'python':sys.version,'host':socket.gethostname(),'platform':platform.platform(),'MemAvailable_KiB':mem,'remote_review_scripts':len(code),'matching_remote_review_scripts':sum(x.get('matches',False) for x in code)})
print('FOLLOWUP_PREFLIGHT_PASS','host',socket.gethostname(),'verified_existing_artifacts',len(verified),'sources',len(metadata),'remote_review_code_counts',json.dumps({'expected':len(code),'exists':sum(x['exists'] for x in code),'matches':sum(x.get('matches',False) for x in code)}),'MemAvailable_KiB',mem,'scientific_model_or_association',False,flush=True)
