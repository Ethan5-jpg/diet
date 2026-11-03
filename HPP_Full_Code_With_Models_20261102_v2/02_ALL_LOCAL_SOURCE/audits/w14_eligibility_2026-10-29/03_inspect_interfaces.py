from pathlib import Path
import json,sys,hashlib,resource
import pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');Q=P/'data_qc/w14_eligibility_20261029';L=Path('/home/ec2-user/multisystem_runtime_20261021/w14_eligibility_20261029')
resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));assert sys.executable=='/home/ec2-user/miniconda/envs/pheno/bin/python'
for tag in ['01','02']:assert (L/(tag+'_actual_exit_code')).read_text().strip()=='0'
A=json.loads((P/'config/ACTIVE_PROTOCOL.json').read_text());R=json.loads(Path(A['analysis_register']).read_text())
print('MODULES',json.dumps(R['modules']),flush=True)
for role in ['outcomes','food_daily','food_cgm_daily','food_cgm_windows','covariates','medication_timing']:
 m=json.loads((Q/'preflight_TRE_ONLY.json').read_text())['inputs'][role];f=pq.ParquetFile(m['path'],pre_buffer=False);print('SCHEMA',role,f.metadata.num_rows,json.dumps(f.schema_arrow.names),flush=True)
for rel in ['config/covariate_roles.yaml','config/outcome_dictionary.yaml']:
 print('CONTRACT',rel,(P/rel).read_text(),flush=True)
print('W14_FILES',json.dumps(sorted(x.name for x in Q.iterdir())),flush=True)
print('INSPECT_PASS',flush=True)
