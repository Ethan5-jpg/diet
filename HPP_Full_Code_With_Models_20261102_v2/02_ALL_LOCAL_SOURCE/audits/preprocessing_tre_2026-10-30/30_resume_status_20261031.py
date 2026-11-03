from pathlib import Path
import json,subprocess
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
Q=P/'data_qc/preprocessing_20261030_v1';L=Path('/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1')
for tag in ['25','26','27','27b','28','29']:
 p=L/(tag+'_actual_exit_code');print('EXIT',tag,p.read_text().strip() if p.exists() else 'not_recorded')
for tag in ['25','26']:
 p=L/(tag+'.log')
 lines=p.read_text().splitlines() if p.exists() else []
 if tag=='25':lines=[s for s in lines if s.startswith('W18_EXECUTION_DESIGN_REGISTERED')]
 else:lines=lines[-10:]
 print('LOG',tag,'\n'.join(lines))
print('RUNNING_26_PID',subprocess.run(['pgrep','-f','[2]6_apply_microbiome_preprocessing.py'],capture_output=True,text=True).stdout.strip())
for site in ['oral','gut']:
 root=Q/'microbiome_transformed_v1'/site
 print('MATRIX_FILES',site,[(p.name,p.stat().st_size) for p in sorted(root.glob('*')) if p.is_file()])
print('RESUME_STATUS_READ_NO_Y')
