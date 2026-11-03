from pathlib import Path
import subprocess,sys,os,json,hashlib
L=Path('/home/ec2-user/multisystem_runtime_20261021/w20_simplified_20261101_v2')
S=Path(__file__).parent
B=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/data_qc/w20_simplified_20261101_v2')
assert (L/'42_actual_exit_code').read_text().strip()=='0'
assert not (L/'43_started.json').exists()
sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in S.glob('*.py')}
(B/'registered_source_hashes_before_real_fits.json').write_text(json.dumps(sources,indent=2)+'\n')
with (L/'43_launcher.log').open('x') as f:
 p=subprocess.Popen([sys.executable,'-u',str(S/'43_wait_record.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(L/'43_launcher_pid').write_text(str(p.pid));print('V2_BACKGROUND_LAUNCHED',p.pid,'LOG',str(L/'43.log'),flush=True)
