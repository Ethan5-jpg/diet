"""Invoke installed Claude Code from project root, read-only; retain exact output/exit."""
from pathlib import Path
import subprocess,json,sys,datetime,hashlib,time
D=Path(__file__).resolve().parent;W=D.parent.parent;T=Path('/private/tmp/hpp_claude_w08_w15_20261029');tag=sys.argv[1];prompt=(T/(tag+'_prompt.txt')).read_text();stream=T/(tag+'_transcript.jsonl');err=T/(tag+'_stderr.log');assert not stream.exists(),str(stream)
cmd=['/Users/haochenxue/.local/bin/claude','-p','--output-format','stream-json','--verbose','--no-session-persistence','--safe-mode','--strict-mcp-config','--no-chrome','--disable-slash-commands','--add-dir',str(T/'packet'),'--tools','Read,Glob,Grep','--allowedTools','Read,Glob,Grep','--permission-mode','plan','--permission-prompts','none']
start=datetime.datetime.now(datetime.timezone.utc).isoformat();meta={'tag':tag,'started_utc':start,'cwd':str(W),'command':cmd,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'tools':'Read,Glob,Grep only; plan permission mode; no server or browser access','prompt_path':str(D/(tag+'_prompt.txt'))};(D/(tag+'_invocation.json')).write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n');result=None;event_count=0;reads=[];last_progress=0
with stream.open('w') as out,err.open('w') as ef:
 p=subprocess.Popen(cmd,cwd=W,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=ef,text=True,bufsize=1);(D/(tag+'_pid.txt')).write_text(str(p.pid));p.stdin.write(prompt);p.stdin.close();print('CLAUDE_STARTED',tag,'pid',p.pid,flush=True)
 for line in p.stdout:
  out.write(line);out.flush();event_count+=1
  try:event=json.loads(line)
  except json.JSONDecodeError:continue
  if event.get('type')=='system' and event.get('subtype')=='init':
   print('CLAUDE_INIT',tag,json.dumps({k:event.get(k) for k in ['model','permissionMode','claude_code_version','tools']}),flush=True)
  if event.get('type')=='assistant':
   for c in event.get('message',{}).get('content',[]):
    if c.get('type')=='tool_use':
     x=c.get('input',{});path=x.get('file_path') or x.get('path') or x.get('pattern','');reads.append({'tool':c.get('name'),'path':path})
     if time.monotonic()-last_progress>15:print('CLAUDE_PROGRESS',tag,c.get('name'),path,flush=True);last_progress=time.monotonic()
  if event.get('type')=='result':result=event
 code=p.wait()
meta.update({'actual_exit_code':code,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event_count':event_count,'tool_call_count':len(reads),'tool_reads':reads,'result_received':result is not None})
if result:
 meta.update({k:result.get(k) for k in ['is_error','subtype','duration_ms','num_turns','total_cost_usd','usage','modelUsage','permission_denials']});(D/(tag+'_result.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');body=result.get('result') or ''
 if isinstance(body,str) and body:(D/(tag+'_review.md')).write_text(body+'\n')
(D/(tag+'_invocation.json')).write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n');(D/(tag+'_actual_exit_code.txt')).write_text(str(code)+'\n')
import shutil
for p in [stream,err]:shutil.copyfile(p,D/p.name)
for p in D.glob(tag+'_*'):
 if p.is_file() and p.suffix in ['.json','.md','.txt'] and p.name!=tag+'_prompt.txt':shutil.copyfile(p,T/p.name)
print('CLAUDE_FINISHED',tag,'exit',code,'result',result is not None,'is_error',None if not result else result.get('is_error'),'tool_calls',len(reads),flush=True)
sys.exit(0 if code==0 and result and not result.get('is_error') else 1)
