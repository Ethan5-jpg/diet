from pathlib import Path
import subprocess,sys
S=Path(__file__).resolve().parent
L=Path('/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1')
assert not (L/'34_actual_exit_code').exists()
with (L/'34.log').open('x') as f:
 r=subprocess.run([sys.executable,'-u',str(S/'34_run_W19_baselines.py')],stdout=f,stderr=subprocess.STDOUT)
(L/'34_actual_exit_code').write_text(str(r.returncode))
print('ACTUAL_EXIT 34',r.returncode,flush=True)
