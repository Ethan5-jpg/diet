"""Package only our generic preflight; use the authorized browser clipboard route."""
from pathlib import Path
import base64
import hashlib
import json
import shlex
import zlib

D = Path(__file__).resolve().parent
P = '/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021'
dependencies = {
    P + '/scripts/preprocessing_20261030_v1/w19_baseline_core.py': Path('/private/tmp/hpp_w19_active_20261101/w19_baseline_core.py'),
    P + '/scripts/preprocessing_20261030_v1/preprocessing_v1_20261030.py': Path('/private/tmp/hpp_preprocessing_active_20261030/preprocessing_v1_20261030.py'),
    P + '/scripts/w20_simplified_20261101_v2/v2_ridge_core.py': Path('/private/tmp/hpp_w20_v2_active_20261101/v2_ridge_core.py'),
}
hashes = {remote: hashlib.sha256(local.read_bytes()).hexdigest() for remote, local in dependencies.items()}
(D / 'expected_dependency_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
files = {name: (D / name).read_bytes() for name in ['47_preflight.py', 'expected_dependency_hashes.json']}
compile(files['47_preflight.py'], '47_preflight.py', 'exec')
entries = {n: {'sha256': hashlib.sha256(b).hexdigest(), 'base64': base64.b64encode(b).decode()} for n, b in files.items()}
packed = base64.b64encode(zlib.compress(json.dumps(entries).encode(), 9)).decode()
loader = f'''import base64,hashlib,json,pathlib,subprocess,sys,zlib
entries=json.loads(zlib.decompress(base64.b64decode({packed!r})))
d=pathlib.Path({(P + '/scripts/w20_targeted_20261101_v3')!r})
d.mkdir(parents=True,exist_ok=False)
for name,entry in entries.items():
 b=base64.b64decode(entry['base64']);assert hashlib.sha256(b).hexdigest()==entry['sha256']
 p=d/name
 with p.open('xb') as f:f.write(b)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256']
l=pathlib.Path('/home/ec2-user/multisystem_runtime_20261021/w20_targeted_20261101_v3');l.mkdir(parents=True,exist_ok=False)
with (l/'47.log').open('x') as f:r=subprocess.run([sys.executable,'-u',str(d/'47_preflight.py')],stdout=f,stderr=subprocess.STDOUT)
(l/'47_actual_exit_code').write_text(str(r.returncode))
print((l/'47.log').read_text()[-24000:]);print('ACTUAL_EXIT 47',r.returncode)
'''
compile(loader, 'transfer_loader', 'exec')
command = '/home/ec2-user/miniconda/envs/pheno/bin/python -c ' + shlex.quote('exec(' + repr(loader) + ')')
(D / '47_command.txt').write_text(command)
(D / '47_source_manifest.json').write_text(json.dumps({n: e['sha256'] for n, e in entries.items()}, indent=2) + '\n')
print('Prepared browser-only preflight command:', len(command), 'characters; NOT transmitted or executed.')
