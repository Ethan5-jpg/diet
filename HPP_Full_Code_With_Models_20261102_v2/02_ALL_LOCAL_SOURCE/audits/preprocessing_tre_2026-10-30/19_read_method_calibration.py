"""Read aggregate method checks and our job status; no participant rows/Y."""
from pathlib import Path
import json
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
Q=P/'data_qc/preprocessing_20261030_v1'
L=Path('/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1')
for tag in ['13','14','14b','15','16','17','18']:
    p=L/(tag+'_actual_exit_code')
    print('RECORDED_EXIT',tag,p.read_text().strip() if p.exists() else 'NOT_YET_RECORDED')
for name in ['estimated_probability_calibration_v1.json','learned_score_full55_calibration_v1.json']:
    p=Q/name
    if not p.exists():
        print('RESULT_PENDING',name)
        continue
    v=json.loads(p.read_text())
    for k in ['null_scenarios','scenarios','power']:
        if k in v:print('CALIBRATION',name,k,json.dumps(v[k],allow_nan=False))
    print('GATES',name,json.dumps({k:t for k,t in v.items() if 'gate' in k or 'ready' in k or 'activation' in k or 'activated' in k}))
print('LEARNED_JOB_RECENT_PROGRESS')
print('\n'.join((L/'18.log').read_text().splitlines()[-8:]))
print('READ_ONLY_AGGREGATE_STATUS_COMPLETE')
