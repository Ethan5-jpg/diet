from pathlib import Path
import json,zlib,hashlib
A=Path(__file__).resolve().parent
files={p.name:{'source':p.read_text(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [A/'01_build_w07_w08.py',A/'02_verify_w07_w08.py']}
for n,x in files.items():compile(x['source'],n,'exec')
s=json.dumps(files,separators=(',',':')).encode();packed=zlib.compress(s);sh=hashlib.sha256(s).hexdigest()
runner="""import sys,subprocess,pathlib,os
r=pathlib.Path('/home/ec2-user/multisystem_runtime_20261021/w07_w08_diet_views_20261028')
d=pathlib.Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/scripts/w07_w08_diet_views_20261028')
for prefix,name in [('01','01_build_w07_w08.py'),('02','02_verify_w07_w08.py')]:
 with (r/(prefix+'.log')).open('w') as f:
  c=subprocess.run([sys.executable,'-u',str(d/name)],stdout=f,stderr=subprocess.STDOUT)
 (r/(prefix+'_actual_exit_code')).write_text(str(c.returncode))
 if c.returncode:break
"""
bootstrap="""import pathlib,hashlib,json,zlib,subprocess,sys,os
s=zlib.decompress(bytes.fromhex(%r))
assert hashlib.sha256(s).hexdigest()==%r
items=json.loads(s)
d=pathlib.Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/scripts/w07_w08_diet_views_20261028');d.mkdir(parents=True,exist_ok=False)
for n,x in items.items():
 b=x['source'].encode();assert hashlib.sha256(b).hexdigest()==x['sha256'];compile(b,n,'exec');p=d/n;p.write_bytes(b);assert p.read_bytes()==b
r=pathlib.Path('/home/ec2-user/multisystem_runtime_20261021/w07_w08_diet_views_20261028');r.mkdir(parents=True,exist_ok=False)
p=r/'runner.py';p.write_text(%r);compile(p.read_bytes(),str(p),'exec')
e=os.environ.copy();e.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
f=(r/'launcher.log').open('w');c=subprocess.Popen([sys.executable,'-u',str(p)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=e);f.close();(r/'pid').write_text(str(c.pid))
print('W07_W08_LAUNCHED',c.pid,'SOURCE_PACKAGE_SHA',%r)
"""%(packed.hex(),sh,runner,sh)
(A/'launch.py').write_text(bootstrap);compile(bootstrap,'launch','exec')
# Compress the bootstrap itself; lower-case hex input avoids unreliable native shifted characters.
outer=zlib.compress(bootstrap.encode());(A/'transfer_hex.txt').write_text(outer.hex())
(A/'source_manifest.json').write_text(json.dumps({n:{'sha256':x['sha256'],'bytes':len(x['source'].encode())} for n,x in files.items()},indent=2))
print('package_sha',sh,'outer_hex_chars',len(outer.hex()))
