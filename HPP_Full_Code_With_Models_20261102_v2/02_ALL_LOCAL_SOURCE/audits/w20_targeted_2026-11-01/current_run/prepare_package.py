from pathlib import Path
import json,hashlib,zlib,base64,shlex,ast
S=Path(__file__).parent
remote='/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/scripts/w20_targeted_20261101_v3'
runtime='/home/ec2-user/multisystem_runtime_20261021/w20_targeted_20261101_v3'
for stage,files in [('48',['targeted_spec.md','targeted_core.py','48_register.py']),('49',['49_models.py']),('50',['50_daily.py'])]:
 entries={n:{'body':(S/n).read_text(),'sha256':hashlib.sha256((S/n).read_bytes()).hexdigest()} for n in files}
 package=base64.b64encode(zlib.compress(json.dumps(entries).encode(),9)).decode()
 wrapper=f'''import base64,zlib,json,hashlib,pathlib,subprocess,sys,os
s=pathlib.Path({remote!r});l=pathlib.Path({runtime!r})
entries=json.loads(zlib.decompress(base64.b64decode({package!r})))
for name,e in entries.items():
 b=e['body'].encode();assert hashlib.sha256(b).hexdigest()==e['sha256'];p=s/name
 with p.open('xb') as f:f.write(b)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==e['sha256']
 if name.endswith('.py'):compile(b,name,'exec')
 print('SOURCE_VERIFIED',name,e['sha256'])
'''
 if stage=='48':
  wrapper+=f'''env=os.environ.copy();env['PYTHONPATH']=str(s)+':'+{remote.rsplit('/scripts/',1)[0]!r}+'/scripts/preprocessing_20261030_v1'
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:env[k]='1'
with (l/'48.log').open('x') as log:r=subprocess.run([sys.executable,'-u',str(s/'48_register.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
(l/'48_actual_exit_code').write_text(str(r.returncode));print((l/'48.log').read_text());print('ACTUAL_EXIT 48',r.returncode)
'''
 else:
  wrapper+=f"print('TRANSFER_ONLY {stage}; no model launched')\n"
 cmd='/home/ec2-user/miniconda/envs/pheno/bin/python -c '+shlex.quote('exec('+repr(wrapper)+')')
 ast.parse(wrapper);assert shlex.split(cmd)[-1]=='exec('+repr(wrapper)+')'
 (S/(stage+'_command.txt')).write_text(cmd)
 (S/(stage+'_source_manifest.json')).write_text(json.dumps({n:e['sha256'] for n,e in entries.items()},indent=2)+'\n')
 print(stage,len(cmd),len(package),{n:e['sha256'] for n,e in entries.items()})
