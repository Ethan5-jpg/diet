from pathlib import Path
import sys,hashlib,json,base64,zlib,shlex
D=Path(__file__).parent
tag,name=sys.argv[1:3]
names=sys.argv[3:]+[name]
files={n:(D/n).read_bytes() for n in names}
for n,b in files.items():compile(b,n,'exec')
entries={n:{'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()} for n,b in files.items()}
packed=base64.b64encode(zlib.compress(json.dumps(entries).encode(),9)).decode()
s='/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/scripts/w20_simplified_20261101_v2'
l='/home/ec2-user/multisystem_runtime_20261021/w20_simplified_20261101_v2'
old='/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/scripts/preprocessing_20261030_v1'
loader=f"""import json,zlib,base64,hashlib,pathlib,subprocess,sys,os
a=json.loads(zlib.decompress(base64.b64decode({packed!r})))
d=pathlib.Path({s!r});d.mkdir(parents=True,exist_ok=True)
for n,e in a.items():
 p=d/n;b=base64.b64decode(e['base64']);assert hashlib.sha256(b).hexdigest()==e['sha256'];assert not p.exists(),str(p);p.write_bytes(b);assert p.read_bytes()==b;compile(b,n,'exec')
l=pathlib.Path({l!r});l.mkdir(parents=True,exist_ok=True)
env=os.environ.copy();env['PYTHONPATH']={old!r};env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
f=(l/{tag+'.log'!r}).open('x');r=subprocess.run([sys.executable,'-u',str(d/{name!r})],stdout=f,stderr=subprocess.STDOUT,env=env);f.close();(l/{tag+'_actual_exit_code'!r}).write_text(str(r.returncode));print((l/{tag+'.log'!r}).read_text()[-14000:]);print('ACTUAL_EXIT',{tag!r},r.returncode)
"""
compile(loader,'loader','exec')
command='/home/ec2-user/miniconda/envs/pheno/bin/python -c '+shlex.quote('exec('+repr(loader)+')')
(D/(tag+'_command.txt')).write_text(command)
(D/(tag+'_source_manifest.json')).write_text(json.dumps({n:e['sha256'] for n,e in entries.items()},indent=2)+'\n')
print('PREPARED',tag,len(command))
