from pathlib import Path
import json
L=Path('/home/ec2-user/multisystem_runtime_20261021/w20_simplified_20261101_v2')
B=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/data_qc/w20_simplified_20261101_v2')
print('V2_EXITS',{t:(L/(t+'_actual_exit_code')).read_text().strip() if (L/(t+'_actual_exit_code')).exists() else 'pending' for t in ['41','42','43_launch','43']})
for name in ['43_launcher.log','43.log']:
 p=L/name
 if p.exists():print('LOG',name,p.read_text()[-12000:])
if (B/'results/summary.json').exists():print('V2_SUMMARY',(B/'results/summary.json').read_text())
