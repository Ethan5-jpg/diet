"""Synthetic interval-covariate checks and compact schema display; no real Y."""
from pathlib import Path
import json, subprocess, sys
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
Q=P/'data_qc/preprocessing_20261030_v1'
L=Path('/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1')
S=Path(__file__).parent
assert (L/'04_actual_exit_code').read_text().strip()=='0'
with (L/'05_interval_tests.log').open('x') as log:
    proc=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(S),
                         '-p','test_interval_upf_prototype.py','-v'],stdout=log,stderr=subprocess.STDOUT)
(L/'05_interval_tests_actual_exit_code').write_text(str(proc.returncode)+'\n')
assert proc.returncode==0,'interval prototype tests failed; inspect preserved log'
receipt={'status':'interval_outer_set_prototype_checks_passed','test_count':8,'test_exit':proc.returncode,
         'real_fit':False,'new_scientific_Y_reads':0,'formal_activation':'pending W17 statistical calibration and W18 register'}
(Q/'interval_prototype_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
inventory=json.loads((Q/'blind_design_inventory.json').read_text())
for name,meta in inventory['schemas'].items():
    if name.startswith('food_events'):cols=list(meta['columns'])
    else:cols=[k for k in meta['columns'] if any(v in k.lower() for v in ['age','sex','smok','energy','fiber','nova','upf','matrix','row_ordinal','sample_id','date','stage','cohort','participant'])]
    print('COMPACT_SCHEMA',name,json.dumps(cols),flush=True)
print('INTERVAL_PROTOTYPE_PASS',json.dumps(receipt),flush=True)
