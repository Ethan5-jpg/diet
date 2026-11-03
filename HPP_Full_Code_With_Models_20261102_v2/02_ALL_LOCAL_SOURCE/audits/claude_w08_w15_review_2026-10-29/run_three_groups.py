"""Start three authorized read-only Claude reviews after verified cache is ready."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,subprocess,sys,time,datetime
D=Path(__file__).resolve().parent;T=Path('/private/tmp/hpp_claude_w08_w15_20261029');ready=D/'cached_packet_verification.json';print('WAITING_FOR_VERIFIED_CACHE',flush=True)
while not ready.exists():time.sleep(1)
m=json.loads(ready.read_text());assert m['status']=='passed' and m['files']==153
print('STARTING_THREE_CLAUDE_REVIEWS',datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
def run(tag):
 log=T/(tag+'_runner.log');assert not log.exists(),str(log)
 with log.open('w') as f:
  p=subprocess.Popen([sys.executable,str(D/'run_claude_review.py'),tag],cwd=D.parent.parent,stdout=f,stderr=subprocess.STDOUT,text=True);(T/(tag+'_runner_pid.txt')).write_text(str(p.pid));code=p.wait()
 return {'tag':tag,'runner_exit':code,'log':str(log)}
results=[]
with ThreadPoolExecutor(max_workers=3) as pool:
 for f in as_completed([pool.submit(run,t) for t in ['A_W08_W10','B_W11_W13','C_W14_W15']]):
  r=f.result();results.append(r);print('GROUP_RUNNER_FINISHED',json.dumps(r),flush=True)
(T/'three_group_run_status.json').write_text(json.dumps(results,indent=2)+'\n');(D/'three_group_run_status.json').write_text(json.dumps(results,indent=2)+'\n');print('THREE_GROUP_RUNNERS_FINISHED',flush=True)
