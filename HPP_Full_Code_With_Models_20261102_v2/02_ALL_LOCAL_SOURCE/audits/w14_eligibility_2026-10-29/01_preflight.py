# -*- coding: utf-8 -*-
"""Read current W13 lock, amended config, and bounded W08-W11 input schemas for W14."""
import sys,json,hashlib,resource,shutil,platform
from pathlib import Path
import pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');L=Path('/home/ec2-user/multisystem_runtime_20261021/w14_eligibility_20261029');Q=P/'data_qc/w14_eligibility_20261029';L.mkdir(parents=True,exist_ok=True);Q.mkdir(parents=True,exist_ok=False);resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(120,120));assert sys.executable=='/home/ec2-user/miniconda/envs/pheno/bin/python';assert shutil.disk_usage(L).free>3*1024**3
mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024;assert mem>2*1024**3

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
v=json.loads((P/'data_qc/w13_microbiome_qc_20261029/final_delivery_verification.json').read_text());assert v['passed'] and (Path('/home/ec2-user/multisystem_runtime_20261021/w13_microbiome_qc_20261029/08_actual_exit_code')).read_text().strip()=='0'
for x in json.loads((P/'data_qc/w13_microbiome_qc_20261029/final_artifact_manifest.json').read_text()):assert sha(x['path'])==x['sha256']
A=json.loads((P/'config/ACTIVE_PROTOCOL.json').read_text());assert A['version']=='multisystem_v3_user_amendment_20261029';lock=json.loads(Path(A['W13_sample_selection_lock']['path']).read_text());assert sha(A['W13_sample_selection_lock']['path'])==A['W13_sample_selection_lock']['sha256'];assert lock['apply_same_lock_across_outcome_modules'];register=json.loads(Path(A['analysis_register']).read_text());print('CURRENT_PROTOCOL',A['version'],'W13_gate_pass',v['final_manifest_artifacts_verified'],flush=True)
files={'food_daily':P/'analysis/shared/food_daily_TRE_ONLY.parquet','food_windows':P/'analysis/shared/food_windows_TRE_ONLY.parquet','food_cgm_daily':P/'analysis/shared/food_cgm_common_daily_TRE_ONLY.parquet','food_cgm_windows':P/'analysis/shared/food_cgm_common_windows_TRE_ONLY.parquet','outcomes':P/'analysis/shared/outcome_registry_TRE_ONLY.parquet','covariates':P/'analysis/shared/covariate_registry_TRE_ONLY.parquet','samples':P/'analysis/shared/sample_registry_TRE_ONLY.parquet','cgm_daily_selected_legacy':P/'analysis/shared/cgm_daily_TRE_ONLY.parquet','cgm_all_daily':P/'data_qc/w09_cgm_point_qc_20261028/cgm_daily_all_connections_TRE_ONLY.parquet','cgm_connections':P/'data_qc/cgm_point_connection_QC.parquet','medication_timing':P/'data_qc/w11_w12_covariates_samples_20261028/medication_timing_TRE_ONLY.parquet'}
meta={}
for role,p in files.items():
 f=pq.ParquetFile(p,pre_buffer=False);budget=sum(f.metadata.row_group(i).column(j).total_uncompressed_size for i in range(f.metadata.num_row_groups) for j in range(f.metadata.row_group(i).num_columns));assert budget<512*1024**2;meta[role]={'path':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':sha(p),'rows':f.metadata.num_rows,'uncompressed_bytes':budget,'schema':[{'name':z.name,'type':str(z.type)} for z in f.schema_arrow]};print('INPUT_SCHEMA',role,f.metadata.num_rows,json.dumps(meta[role]['schema']),flush=True)
for rel in ['config/covariate_roles.yaml','config/cgm_point_daily_rules_W09_20261028.json']:
 d=json.loads((P/rel).read_text());print('CONFIG_CONTENT',rel,json.dumps(d),flush=True)
print('REGISTER_MODULES',json.dumps([{'id':x['analysis_id'],'role':x['role'],'diet_view':x['diet_view']} for x in register['modules']]),flush=True)
out={'protocol':A,'inputs':meta,'W13_verified':True,'available_memory_bytes':mem,'python':sys.version,'platform':platform.platform()};(Q/'preflight_TRE_ONLY.json').write_text(json.dumps(out,indent=2)+'\n');print('W14_PREFLIGHT_PASS',flush=True)
