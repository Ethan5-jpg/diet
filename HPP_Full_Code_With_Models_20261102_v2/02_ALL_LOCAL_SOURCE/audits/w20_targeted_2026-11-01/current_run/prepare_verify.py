from pathlib import Path
import json,zlib,base64,hashlib,shlex
S=Path(__file__).parent;names=['targeted_spec.md','targeted_core.py','48_register.py','49_models.py','49b_models.py','50_daily.py','50b_daily.py','51_verify.py']
expected={n:hashlib.sha256((S/n).read_bytes()).hexdigest() for n in names};(S/'expected_v3_sources.json').write_text(json.dumps(expected,indent=2)+'\n')
files=['51_verify.py','expected_v3_sources.json'];entries={n:{'body':(S/n).read_text(),'sha256':hashlib.sha256((S/n).read_bytes()).hexdigest()} for n in files};pack=base64.b64encode(zlib.compress(json.dumps(entries).encode(),9)).decode();remote='/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/scripts/w20_targeted_20261101_v3';runtime='/home/ec2-user/multisystem_runtime_20261021/w20_targeted_20261101_v3'
wrapper=f'''from pathlib import Path
import json,zlib,base64,hashlib,subprocess,sys,os
s=Path({remote!r});l=Path({runtime!r})
print('50b_EXIT',(l/'50b_actual_exit_code').read_text());print('50b_PIDs',[(p.name,p.read_text()) for p in l.glob('50b*pid')]);print((l/'50b.log').read_text()[-9500:]);assert (l/'50b_actual_exit_code').read_text().strip()=='0'
e=json.loads(zlib.decompress(base64.b64decode({pack!r})))
for name,v in e.items():
 b=v['body'].encode();assert hashlib.sha256(b).hexdigest()==v['sha256']
 with (s/name).open('xb') as f:f.write(b)
 assert hashlib.sha256((s/name).read_bytes()).hexdigest()==v['sha256']
 if name.endswith('.py'):compile(b,name,'exec')
 print('SOURCE_VERIFIED',name,v['sha256'])
env=os.environ.copy();env['PYTHONPATH']=str(s)+':'+str(s.parent/'preprocessing_20261030_v1')
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:env[k]='1'
with (l/'51.log').open('x') as f:
 p=subprocess.Popen([sys.executable,'-u',str(s/'51_verify.py')],env=env,stdout=f,stderr=subprocess.STDOUT);(l/'51_pid').write_text(str(p.pid));rc=p.wait()
(l/'51_actual_exit_code').write_text(str(rc));print((l/'51.log').read_text()[-13500:]);print('ACTUAL_EXIT 51',rc)
'''
(S/'51_command.txt').write_text('/home/ec2-user/miniconda/envs/pheno/bin/python -c '+shlex.quote('exec('+repr(wrapper)+')'));print('command_bytes',len((S/'51_command.txt').read_bytes()),'source_hashes',expected)
