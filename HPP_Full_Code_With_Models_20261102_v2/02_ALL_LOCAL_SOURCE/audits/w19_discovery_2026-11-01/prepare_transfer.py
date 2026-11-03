from pathlib import Path
import base64, hashlib, json, shlex, sys, zlib

D = Path(__file__).resolve().parent
ROOT = D.parents[1]
name, tag = sys.argv[1:3]
files = {name:(D/name).read_bytes()}
if tag == '01':
    for rel in ['scripts/preprocessing_v1_20261030.py', 'scripts/test_preprocessing_v1_20261030.py',
                'config/preprocessing_v1_20261030.json']:
        p = ROOT/rel
        files[p.name] = p.read_bytes()
if tag == '04':
    for name2 in ['w16_estimators.py','test_w16_interfaces.py']:
        files[name2] = (D/name2).read_bytes()
if tag == '05':
    for name2 in ['w17_interval_upf_prototype.py','test_interval_upf_prototype.py']:
        files[name2] = (D/name2).read_bytes()
if tag == '08':
    files['w17_interval_upf_rcs.py'] = (D/'w17_interval_upf_rcs.py').read_bytes()
    files['09_w17_rcs_calibration.py'] = (D/'09_w17_rcs_calibration.py').read_bytes()
if tag == '12':
    files['upf_catalogue_calibration.py'] = (D/'upf_catalogue_calibration.py').read_bytes()
if tag == '16':
    files['upf_catalogue_calibration_v2.py'] = (D/'upf_catalogue_calibration_v2.py').read_bytes()
if tag == '26':
    files['w18_preprocessing_reader.py'] = (D/'w18_preprocessing_reader.py').read_bytes()
if tag == '33':
    files['w19_baseline_core.py'] = (D/'w19_baseline_core.py').read_bytes()
for n, b in files.items():
    if n.endswith('.py'):
        compile(b, n, 'exec')
bundle = {n:{'base64':base64.b64encode(b).decode(),'sha256':hashlib.sha256(b).hexdigest()} for n,b in files.items()}
packed = base64.b64encode(zlib.compress(json.dumps(bundle).encode(),9)).decode()
remote = '/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/scripts/preprocessing_20261030_v1'
runtime = '/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1'
loader = (
    "import json,zlib,base64,hashlib,pathlib,subprocess,sys; "
    "a=json.loads(zlib.decompress(base64.b64decode("+repr(packed)+"))); "
    "d=pathlib.Path("+repr(remote)+"); d.mkdir(parents=True,exist_ok=True); "
    "[(lambda p,b,h: (not p.exists() or (_ for _ in ()).throw(AssertionError('exists:'+str(p))), "
    "hashlib.sha256(b).hexdigest()==h or (_ for _ in ()).throw(AssertionError('hash')), "
    "p.write_bytes(b),p.read_bytes()==b or (_ for _ in ()).throw(AssertionError('readback'))))"
    "(d/n,base64.b64decode(e['base64']),e['sha256']) for n,e in a.items()]; "
    "l=pathlib.Path("+repr(runtime)+"); l.mkdir(parents=True,exist_ok=True); "
    "f=(l/"+repr(tag+'.log')+").open('x'); "
    "r=subprocess.run([sys.executable,'-u',str(d/"+repr(name)+")],stdout=f,stderr=subprocess.STDOUT); "
    "f.close(); (l/"+repr(tag+'_actual_exit_code')+").write_text(str(r.returncode)); "
    "print((l/"+repr(tag+'.log')+").read_text()[-15000:]); print('ACTUAL_EXIT',"+repr(tag)+",r.returncode)"
)
compile(loader,'loader','exec')
command = '/home/ec2-user/miniconda/envs/pheno/bin/python -c '+shlex.quote(loader)
(D/(tag+'_command.txt')).write_text(command)
(D/('source_manifest_'+tag+'.json')).write_text(json.dumps({'files':[{'name':n,'sha256':e['sha256']} for n,e in bundle.items()], 'command_characters':len(command)},indent=2)+'\n')
print('PREPARED',tag,len(files),len(command))
