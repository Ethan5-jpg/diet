"""Record completed QC and failed method candidates without activating W18."""
from pathlib import Path
from collections import Counter
import datetime,hashlib,json
import pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
Q=P/'data_qc/preprocessing_20261030_v1'
S=P/'scripts/preprocessing_20261030_v1'
L=Path('/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1')
assert (L/'09_actual_exit_code').read_text().strip()=='0'
assert (L/'10_actual_exit_code').read_text().strip()=='0'
verify=json.loads((Q/'food_qc_v1/independent_verification.json').read_text())
assert verify['status']=='food_QC_independent_readback_passed'
stages={tag:int((L/(tag+'_actual_exit_code')).read_text().strip()) for tag in ['01','02','03','04','05','06','06b','07','08','09','10']}
calibrations={}
for name in ['W17_calibration_v1b.json','W17_calibration_v2_RCS.json']:
    d=json.loads((Q/name).read_text())
    calibrations[name]={k:d[k] for k in ['all_development_gates_passed','interval_power_validated','interval_power_stress_checks']}
    calibrations[name]['point_calibration']=d['point_calibration']
    calibrations[name]['interval_calibration']=d['interval_calibration']
roles=Counter()
f=pq.ParquetFile(P/'freeze/split_order_TRE_ONLY.parquet',pre_buffer=False)
for rb in f.iter_batches(columns=['final_role'],batch_size=1024,use_threads=False):
    roles.update('UNASSIGNED' if x is None else str(x) for x in rb.column(0).to_pylist())
assert roles==Counter({'UNASSIGNED':6305})
state={'time_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'version':'preprocessing_20261030_v1',
       'requested_scope':'W16-W18 then train-only formal transformations; UPF and fiber formal questions retained',
       'stages_actual_exits':stages,'W16':'synthetic interfaces passed; production fitting/activation pending W18',
       'W17':'two conservative interval candidates statistically not approved; power stress checks unsuccessful',
       'W18':'not frozen; UPF method assumptions and design unresolved',
       'food_QC':'built and independently source/day/window verified',
       'microbiome_and_phenotype_formal_transform':'not fitted; train-only synthetic code verified',
       'roles':dict(roles),'new_scientific_Y_reads':0,'real_supervised_training_runs':0,
       'old_W13_W14_W15_artifacts_verified':verify['original_W13_W14_W15_artifact_hashes_verified'],
       'calibration_candidates':calibrations,
       'next_step':'resolve explicit model assumptions for total UPF, or obtain food classification evidence; rerun method calibration before W18',
       'not_claimed':['Claude independent acceptance','real scientific negative result','complete W16-W18','independent validation cohort','formal preprocessing completed']}
status=Q/'execution_status.json';assert not status.exists()
status.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
artifacts=[]
for root in [Q,S]:
    for p in sorted(root.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts:continue
        artifacts.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
for p in sorted(L.glob('*')):
    if p.is_file() and (p.suffix=='.log' or p.name.endswith('_actual_exit_code')):
        artifacts.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
manifest=Q/'staged_artifact_manifest.json';assert not manifest.exists()
manifest.write_text(json.dumps(artifacts,indent=2)+'\n')
assert json.loads(status.read_text())==state
M=P/'memos/preprocessing_20261030_v1';M.mkdir(exist_ok=False)
(M/'MEMO.md').write_text('# Preprocessing/W16-W18 progress, 2026-10-30\n\n'
    'Food QC built and independently verified. Formal W18 remains unresolved because both conservative interval-UPF candidates failed practical power checks. '
    'No midpoint substitution, unknown=non-UPF, formal role assignment, real preprocessing parameter fitting, supervised score fitting or scientific associations were performed. '
    'The two exposure questions remain requested. Next choose explicit assumptions/evidence and calibrate before freezing.\n\n'
    'Status: '+str(status)+'\n\nManifest: '+str(manifest)+'\n\nLogs: '+str(L)+'\n\n'
    'The manifest covers staged outputs and completed logs through stage10; stage11 exit is recorded by its launcher afterward.\n')
compact={k:v for k,v in state.items() if k!='calibration_candidates'}
compact['staged_artifacts']=len(artifacts);compact['manifest_sha256']=sha(manifest)
print('VERIFIED_PROGRESS_SEALED',json.dumps(compact),flush=True)
print('RCS_POINT_CALIBRATION_SUMMARY',json.dumps(calibrations['W17_calibration_v2_RCS.json']['point_calibration']),flush=True)
