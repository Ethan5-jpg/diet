from pathlib import Path
import json,hashlib,datetime,sys,os
import pyarrow.parquet as pq
from targeted_core import checks
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');S=Path(__file__).parent;B=P/'data_qc/w20_targeted_20261101_v3'
assert sys.executable=='/home/ec2-user/miniconda/envs/pheno/bin/python'
assert json.loads((S/'47_preflight_report.json').read_text())['status']=='preflight_metadata_and_provenance_verified'
for path,h in json.loads((S/'expected_dependency_hashes.json').read_text()).items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
B.mkdir(exist_ok=False);(B/'results').mkdir();(B/'targeted_spec.md').write_bytes((S/'targeted_spec.md').read_bytes())
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in S.iterdir() if p.suffix in ['.py','.md','.json']}
report={'registered_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'python':sys.executable,'pid':os.getpid(),'source_sha256':manifest,'checks':checks(),'confirmation_Y_materialized':0,'status':'registered_before_new_fits'}
(B/'registration.json').write_text(json.dumps(report,indent=2)+'\n')
print('REGISTERED',json.dumps(report),flush=True)
r=json.loads((S/'47_preflight_report.json').read_text())
for s in r['schemas']:
 print('SCHEMA',Path(s['path']).name,'parent',str(Path(s['path']).parent.relative_to(P)),'rows',s.get('rows'),'fields',','.join(v['name'] for v in s.get('columns',[])),flush=True)
