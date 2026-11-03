"""Run checks and screen detached; prepare only audited aggregate outputs for egress."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

parser=argparse.ArgumentParser()
for key in ('package','pilot-code','breadth-code','code-commit'):
    parser.add_argument('--'+key,required=True)
parser.add_argument('--egress-store',required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parent
logs=root/'logs'
logs.mkdir(exist_ok=False)
runtime=Path(tempfile.mkdtemp(prefix='hpp_representation_'))
live=runtime/'run.log'
command=[sys.executable,'-u',str(root/'run_screen.py')]
for key in ('package','pilot_code','breadth_code','code_commit'):
    command += ['--'+key.replace('_','-'),getattr(args,key)]
(logs/'execution.json').write_text(json.dumps(dict(command=command,live_log=str(live),
    code_commit=args.code_commit,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
with live.open('x') as stream:
    rc=subprocess.run([sys.executable,str(root/'verify_screen.py')],stdout=stream,stderr=subprocess.STDOUT).returncode
    if rc==0:
        rc=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT).returncode
(logs/'run.log').write_bytes(live.read_bytes())
(logs/'actual_exit_code.txt').write_text(str(rc)+'\n')
if rc==0:
    audit=json.loads((root/'results/audit.json').read_text())
    assert audit['status']=='PASS'
    names=['results/'+n for n in ('summary.json','model_comparison.csv','grid_scores.json',
        'diagnostics.json','registration.json','audit.json','ANALYSIS.md')]
    names += [f'results/fold{f}_encoding.json' for f in range(5)]
    names += ['logs/execution.json','logs/run.log','logs/actual_exit_code.txt']
    public=root/'public_export'
    public.mkdir(exist_ok=False)
    bundle=dict(format='hpp_aggregate_export_v1',includes_individual_health_records=False,
        files={n:(root/n).read_text() for n in names},
        excluded='All TRE_ONLY arrays, participant tables, fitted estimators and PCA objects.')
    (public/'aggregate_results.json').write_text(json.dumps(bundle,ensure_ascii=False,allow_nan=False)+'\n')
    for name in ('model_comparison.csv','ANALYSIS.md'):
        shutil.copyfile(root/'results'/name,public/name)
    manifest={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
              for p in public.iterdir() if p.is_file()}
    (public/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Staging is not approval. The platform's formal egress request remains separate.
    egress=Path(args.egress_store)
    assert egress.is_dir() and egress.name.startswith('egress-store-')
    destination=egress/(root.name+'_aggregate')
    destination.mkdir(exist_ok=False)
    for path in public.iterdir():
        shutil.copyfile(path,destination/path.name)
    (logs/'egress_staging.json').write_text(json.dumps(dict(status='staged_not_approved',
        destination=str(destination),manifest=manifest),indent=2)+'\n')
print('SCREEN_EXIT',rc,flush=True)
sys.exit(rc)
