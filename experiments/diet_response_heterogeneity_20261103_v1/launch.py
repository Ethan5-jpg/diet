"""Run verified pilot in a detached process and retain the actual exit status."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import datetime
import json
from pathlib import Path
import resource
import subprocess
import sys

source = Path(__file__).resolve().parent
package = source.parents[2]
commit = sys.argv[1]
logs = source.parent / "logs"
logs.mkdir(exist_ok=False)
resource.setrlimit(resource.RLIMIT_AS, (5*1024**3, 5*1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (10800, 10800))
command = [sys.executable, "-u", str(source/"run_pilot.py"), "--package", str(package),
           "--output", str(source.parent/"results"), "--code-commit", commit]
(logs/"execution.json").write_text(json.dumps(dict(code_commit=commit, command=command,
    cwd=str(source), pid=os.getpid(), started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    address_space_limit_bytes=5*1024**3, cpu_limit_seconds=10800), indent=2)+"\n")
with (logs/"pilot.log").open("x") as log:
    check = subprocess.run([sys.executable, str(source/"verify_synthetic.py")], stdout=log, stderr=subprocess.STDOUT)
    rc = check.returncode
    if rc == 0:
        rc = subprocess.run(command, cwd=source, stdout=log, stderr=subprocess.STDOUT).returncode
(logs/"actual_exit_code.txt").write_text(str(rc)+"\n")
print("PILOT_EXIT", rc, flush=True)
sys.exit(rc)
