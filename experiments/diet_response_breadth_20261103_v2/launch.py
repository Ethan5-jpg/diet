"""Detached runner; live log is local, completed log archived to TRE experiment."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--package', required=True)
parser.add_argument('--pilot-code', required=True)
parser.add_argument('--code-commit', required=True)
args = parser.parse_args()
source = Path(__file__).resolve().parent
logs = source/'logs'
logs.mkdir(exist_ok=False)
runtime = Path(tempfile.mkdtemp(prefix='hpp_breadth_'))
live = runtime/'run.log'
command = [sys.executable, '-u', str(source/'run_breadth.py'), '--package', args.package,
           '--pilot-code', args.pilot_code, '--output', str(source/'results'),
           '--code-commit', args.code_commit]
(logs/'execution.json').write_text(json.dumps(dict(command=command, live_log=str(live),
    code_commit=args.code_commit, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()), indent=2)+'\n')
with live.open('x') as stream:
    result = subprocess.run([sys.executable, str(source/'verify_breadth.py'), '--pilot-code',
                             args.pilot_code], stdout=stream, stderr=subprocess.STDOUT)
    rc = result.returncode
    if rc == 0:
        rc = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT).returncode
with (logs/'run.log').open('xb') as stream:
    stream.write(live.read_bytes())
(logs/'actual_exit_code.txt').write_text(str(rc)+'\n')
print('BREADTH_EXIT', rc, flush=True)
sys.exit(rc)
