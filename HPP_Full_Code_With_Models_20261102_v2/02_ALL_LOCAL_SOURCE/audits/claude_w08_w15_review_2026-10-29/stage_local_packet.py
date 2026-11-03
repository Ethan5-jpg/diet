"""Verify and cache the frozen packet outside iCloud to keep review reads stable."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,hashlib
D=Path(__file__).resolve().parent;T=Path('/private/tmp/hpp_claude_w08_w15_20261029');T.mkdir(exist_ok=True);B=T/'packet';B.mkdir(exist_ok=True)
manifest=json.loads((D/'input_manifest.json').read_text());print('MANIFEST_READ',len(manifest),flush=True)
def cp(m):
 p=D/'packet'/m['path'];data=p.read_bytes();assert len(data)==m['bytes'] and hashlib.sha256(data).hexdigest()==m['sha256'],m['path'];q=B/m['path'];q.parent.mkdir(parents=True,exist_ok=True)
 if q.exists():assert q.read_bytes()==data
 else:q.write_bytes(data)
 return m['path']
count=0
with ThreadPoolExecutor(max_workers=12) as pool:
 for f in as_completed([pool.submit(cp,m) for m in manifest]):
  f.result();count+=1
  if count%15==0 or count==len(manifest):print('CACHED_VERIFIED',count,flush=True)
(T/'input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
for tag in ['A_W08_W10','B_W11_W13','C_W14_W15']:
 prompt=(D/(tag+'_prompt.txt')).read_text().replace(str(D/'packet'),str(B));(T/(tag+'_prompt.txt')).write_text(prompt)
(D/'cached_packet_verification.json').write_text(json.dumps({'status':'passed','files':count,'bytes':sum(m['bytes'] for m in manifest),'cache_path':str(B),'input_manifest_sha256':hashlib.sha256((D/'input_manifest.json').read_bytes()).hexdigest(),'original_project_root':str(D.parent.parent)},ensure_ascii=False,indent=2)+'\n')
print('CACHE_READY',T,flush=True)
