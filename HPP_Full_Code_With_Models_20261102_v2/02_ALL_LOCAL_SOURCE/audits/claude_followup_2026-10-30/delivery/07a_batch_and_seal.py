from pathlib import Path
import subprocess,sys
D=Path(__file__).parent;L=Path('/home/ec2-user/multisystem_runtime_20261021/claude_readonly_followup_20261030_v1')
for tag,name in [('07a_source','07a_medication_source_reconciliation.py'),('07','07_seal_audit.py')]:
 p=L/(tag+'.log');assert not p.exists()
 with p.open('w') as f:r=subprocess.run([sys.executable,'-u',str(D/name)],stdout=f,stderr=subprocess.STDOUT)
 (L/(tag+'_actual_exit_code')).write_text(str(r.returncode));print(p.read_text()[-16000:],flush=True);print('ACTUAL_EXIT',tag,r.returncode,flush=True)
 if r.returncode:sys.exit(r.returncode)
