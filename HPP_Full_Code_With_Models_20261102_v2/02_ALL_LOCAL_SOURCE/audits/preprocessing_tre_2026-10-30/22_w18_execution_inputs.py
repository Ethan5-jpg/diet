"""Inspect protocol/API/schema for W18 implementation; no scientific Y values."""
from pathlib import Path
import hashlib,json,ast
import pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
Q=P/'data_qc/preprocessing_20261030_v1'
active=json.loads((P/'config/ACTIVE_PROTOCOL.json').read_text())
register=json.loads(Path(active['analysis_register']).read_text())
print('CURRENT_PROTOCOL',json.dumps(active))
print('MODULE_REGISTER',json.dumps(register))
for base in [P/'data_qc/w15_split_access_20261029',P/'analysis/shared']:
    for p in sorted(base.glob('*.parquet')):
        if base.name=='shared' and not any(t in p.name for t in ['covariate','sample_registry','medication','cgm_visit']):continue
        f=pq.ParquetFile(p,pre_buffer=False)
        print('INPUT_SCHEMA',p.name,f.metadata.num_rows,json.dumps({v.name:str(v.type) for v in f.schema_arrow}))
manifest=json.loads((P/'data_qc/w15_split_access_20261029/final_artifact_manifest.json').read_text())
for item in manifest:
    p=Path(item['path'])
    if p.name=='w15_access_guard.py':
        s=p.read_text()
        print('GUARD_SHA256',hashlib.sha256(s.encode()).hexdigest())
        print('GUARD_SOURCE_BEGIN\n'+s+'\nGUARD_SOURCE_END')
for name in ['blind_design_inventory.json','food_qc_v1/summary.json']:
    p=Q/name
    if p.exists():
        v=json.loads(p.read_text())
        print('BLIND_DESIGN_METADATA',name,json.dumps(v))
print('W18_EXECUTION_INPUTS_INSPECTED_NO_Y_VALUES')
