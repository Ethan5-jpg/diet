from pathlib import Path
import subprocess,sys,os,json,time
L=Path('/home/ec2-user/multisystem_runtime_20261021/w20_simplified_20261101_v2')
S=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/scripts/w20_simplified_20261101_v2')
assert not (L/'43_actual_exit_code').exists()
(L/'43_started.json').write_text(json.dumps({'launcher_pid':os.getpid(),'time':time.time(),'script':str(S/'43_run_v2.py')})+'\n')
with (L/'43.log').open('x') as f:r=subprocess.run([sys.executable,'-u',str(S/'43_run_v2.py')],stdout=f,stderr=subprocess.STDOUT)
(L/'43_actual_exit_code').write_text(str(r.returncode))
