"""Run independently logged read-only audits sequentially; retain every actual exit."""
from pathlib import Path
import subprocess,sys,json
D=Path(__file__).parent;L=Path('/home/ec2-user/multisystem_runtime_20261021/claude_readonly_followup_20261030_v1');statuses={}
for tag,name in [('04c_medication','04c_medication_and_diet.py'),('05','05_W14_independent_coverage.py'),('06','06_profile_lag_provenance.py')]:
 p=L/(tag+'.log');assert not p.exists()
 with p.open('w') as f:r=subprocess.run([sys.executable,'-u',str(D/name)],stdout=f,stderr=subprocess.STDOUT)
 (L/(tag+'_actual_exit_code')).write_text(str(r.returncode));statuses[tag]=r.returncode;print('AUDIT_STEP',tag,flush=True);print(p.read_text()[-17000:],flush=True);print('ACTUAL_EXIT',tag,r.returncode,flush=True)
print('BATCH_ACTUAL_EXITS',json.dumps(statuses),flush=True);sys.exit(int(any(v!=0 for v in statuses.values())))
