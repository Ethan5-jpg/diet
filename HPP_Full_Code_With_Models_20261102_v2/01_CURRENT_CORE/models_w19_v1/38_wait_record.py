from pathlib import Path
import subprocess,sys
S=Path(__file__).resolve().parent
L=Path('/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1')
assert not (L/'38_actual_exit_code').exists()
with (L/'38.log').open('x') as f:
 r=subprocess.run([sys.executable,'-u',str(S/'38_run_W20_scores.py')],stdout=f,stderr=subprocess.STDOUT)
(L/'38_actual_exit_code').write_text(str(r.returncode))
print('ACTUAL_EXIT 38',r.returncode,flush=True)
