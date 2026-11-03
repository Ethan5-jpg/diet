"""Verify W16 model and inference interfaces on synthetic data in pheno."""
from pathlib import Path
import json, subprocess, sys
import scipy
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
Q=P/'data_qc/preprocessing_20261030_v1'
L=Path('/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1')
S=Path(__file__).parent
assert (L/'03_actual_exit_code').read_text().strip()=='0'
with (L/'04_interface_tests.log').open('x') as log:
    proc=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(S),
                         '-p','test_w16_interfaces.py','-v'],stdout=log,stderr=subprocess.STDOUT)
(L/'04_interface_tests_actual_exit_code').write_text(str(proc.returncode)+'\n')
assert proc.returncode==0,'interface tests failed; see preserved log'
receipt={'status':'W16_synthetic_transform_and_estimator_interfaces_passed',
         'transform_tests':11,'estimator_tests':9,'scipy':scipy.__version__,
         'actual_test_exit':proc.returncode,'real_fits':0,'new_scientific_Y_reads':0,
         'W17_status':'pending calibration including interval-UPF branch',
         'W18_status':'not frozen'}
(Q/'W16_interface_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('W16_INTERFACE_PASS',json.dumps(receipt),flush=True)
