from pathlib import Path
import base64,zlib,json,hashlib,shlex
D=Path(__file__).resolve().parent
names=['38_run_W20_scores.py','38_wait_record.py'];bundle={}
for n in names:
 b=(D/n).read_bytes();compile(b,n,'exec');bundle[n]={'base64':base64.b64encode(b).decode(),'sha256':hashlib.sha256(b).hexdigest()}
a=base64.b64encode(zlib.compress(json.dumps(bundle).encode(),9)).decode()
remote='/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/scripts/preprocessing_20261030_v1';runtime='/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1'
s="import pathlib,json,zlib,base64,hashlib,subprocess,sys; a=json.loads(zlib.decompress(base64.b64decode("+repr(a)+"))); d=pathlib.Path("+repr(remote)+"); [(lambda p,b,h:(not p.exists() or (_ for _ in ()).throw(AssertionError('exists')),hashlib.sha256(b).hexdigest()==h or (_ for _ in ()).throw(AssertionError('hash')),p.write_bytes(b),p.read_bytes()==b or (_ for _ in ()).throw(AssertionError('readback'))))(d/n,base64.b64decode(e['base64']),e['sha256']) for n,e in a.items()]; l=pathlib.Path("+repr(runtime)+"); f=(l/'38_launcher.log').open('x'); r=subprocess.Popen([sys.executable,'-u',str(d/'38_wait_record.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True); (l/'38_launcher_pid').write_text(str(r.pid)); print('W20_BACKGROUND_LAUNCHED',r.pid,'completion tracked in 38_actual_exit_code and score_results_v1/summary.json')"
compile(s,'loader','exec');(D/'38_background_command.txt').write_text('/home/ec2-user/miniconda/envs/pheno/bin/python -c '+shlex.quote(s));(D/'source_manifest_38_background.json').write_text(json.dumps({n:v['sha256'] for n,v in bundle.items()},indent=2)+'\n');print('PREPARED',len(s))
