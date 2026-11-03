"""Current schemas and hypothetical whole-family splits; no outcome-value reads."""
from pathlib import Path
from collections import defaultdict
import hashlib, importlib.util, json
import numpy as np
import pyarrow.parquet as pq

P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
Q=P/'data_qc/preprocessing_20261030_v1'
L=Path('/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1')
assert (L/'02_actual_exit_code').read_text().strip()=='0'
active=json.loads((P/'config/ACTIVE_PROTOCOL.json').read_text())
register=json.loads(Path(active['analysis_register']).read_text())
manifest=json.loads((P/'data_qc/w15_split_access_20261029/final_artifact_manifest.json').read_text())
guards=[Path(x['path']) for x in manifest if x['path'].endswith('/w15_access_guard.py')]
assert len(guards)==1
spec=importlib.util.spec_from_file_location('current_W15_guard',guards[0])
guard=importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
columns=['participant_id','cohort','research_stage','analysis_id','split_unit_id','unit_order','person_order']
receipt={'purpose':'technical_membership_only','columns':columns,'scientific_Y':False,
         'guard_path':str(guards[0]),'guard_sha256':hashlib.sha256(guards[0].read_bytes()).hexdigest(),
         'roles_are_hypothetical_only':True,'fractions':[0.5,0.6,0.7,0.8]}
(Q/'03_preaccess_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
out=[]
for module in register['modules']:
    name=module['analysis_id']
    if name.startswith('P_') or name=='Common4': continue
    d=guard.load_blind_membership(name,columns)
    for fraction in receipt['fractions']:
        cutoff=int(np.floor(5699*fraction))
        discovery=d.unit_order.le(cutoff)
        out.append({'analysis_id':name,'discovery_fraction':fraction,'unit_cutoff':cutoff,
                    'base_discovery_people':int(discovery.sum()),'base_confirmation_people':int((~discovery).sum()),
                    'discovery_units':int(d.loc[discovery,'split_unit_id'].nunique()),
                    'confirmation_units':int(d.loc[~discovery,'split_unit_id'].nunique())})

schemas={}
names=['sample_registry_TRE_ONLY.parquet','food_daily_TRE_ONLY.parquet','food_windows_TRE_ONLY.parquet',
       'food_events_clean_TRE_ONLY.parquet','covariate_registry_TRE_ONLY.parquet',
       'cgm_visit_phenotypes_TRE_ONLY.parquet']
for name in names:
    p=P/'analysis/shared'/name
    if not p.exists(): continue
    f=pq.ParquetFile(p,pre_buffer=False)
    schemas[name]={'path':str(p),'rows':f.metadata.num_rows,'columns':{v.name:str(v.type) for v in f.schema_arrow}}
    print('DESIGN_SCHEMA',name,json.dumps(schemas[name]['columns']),flush=True)
(Q/'blind_design_inventory.json').write_text(json.dumps({'role_scenarios':out,'schemas':schemas,
                                                          'new_scientific_Y_reads':0},indent=2)+'\n')
print('BLIND_ROLE_SCENARIOS',json.dumps(out),flush=True)
print('BLIND_INVENTORY_PASS','modules',len(out)//4,'new_scientific_Y_reads',0,flush=True)
