from pathlib import Path
import json,subprocess
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
L=Path('/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1');B=P/'data_qc/w20_discovery_20261101_v1/score_results_v1'
p=L/'38_actual_exit_code';print('W20_ACTUAL_EXIT',p.read_text().strip() if p.exists() else 'not_finished')
print('W20_LOG_TAIL','\n'.join((L/'38.log').read_text().splitlines()[-18:]) if (L/'38.log').exists() else 'no log yet')
for n in ['checkpoint.json','summary.json']:
 if (B/n).exists():print(n,(B/n).read_text())
print('W20_RUNNING',subprocess.run(['pgrep','-af','[3]8_run_W20_scores.py'],capture_output=True,text=True).stdout.strip())
