import sys,subprocess
from pathlib import Path
R=Path('/home/ec2-user/multisystem_runtime_20261021/w09_cgm_point_qc_20261028');D=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/scripts/w09_cgm_point_qc_20261028')
for x in ['03b','09','04b','05','06','07']:
 p=R/(x+'_actual_exit_code');print('STAGE',x,p.read_text() if p.exists() else 'pending')
p=R/'07_actual_exit_code'
if p.exists() and p.read_text().strip()=='0':
 with (R/'08.log').open('w') as f:c=subprocess.run([sys.executable,'-u',str(D/'08_display_aggregate.py')],stdout=f,stderr=subprocess.STDOUT)
 (R/'08_actual_exit_code').write_text(str(c.returncode));print('READ_AGGREGATE_EXIT',c.returncode);print('\n'.join((R/'08.log').read_text().splitlines()[:18]))
else:
 for x in ['05','06','07']:
  p=R/(x+'.log')
  if p.exists():print('TAIL',x,p.read_text()[-1200:])
