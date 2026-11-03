# -*- coding: utf-8 -*-
"""Read-only bounded diet/food auxiliary inventory. No participant rows."""
from pathlib import Path
import os, sys, json, time, hashlib, platform, traceback
import pyarrow.parquet as pq
ROOT=Path('/home/ec2-user/studies/hpp_datasets')
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
OUT=P/'data_qc/w04_mapping_review_20261027'
OUT.mkdir(parents=True,exist_ok=True)
def save(name,obj):
 p=OUT/name
 p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str))
 assert json.loads(p.read_text()) is not None
 return p
start=time.monotonic()
root_dirs=sorted(x.name for x in ROOT.iterdir() if x.is_dir())
mods=[ROOT/x for x in root_dirs if any(k in x.lower() for k in ['diet','food','nutrition'])]
print('PYTHON',sys.executable,'HOST',platform.node(),flush=True)
print('MODULES',json.dumps([str(x) for x in mods]),flush=True)
records=[]; visited=0; complete=True; errors=[]
for module in mods:
 stack=[module]
 while stack:
  folder=stack.pop()
  try:
   for e in os.scandir(folder):
    visited+=1
    if visited>15000 or time.monotonic()-start>120:
     complete=False; stack=[]; break
    if e.is_dir(follow_symlinks=False): stack.append(Path(e.path)); continue
    if not e.is_file(follow_symlinks=False): continue
    p=Path(e.path); st=e.stat()
    rec={'path':str(p),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
    if p.suffix.lower() in ['.parquet','.pq']:
     try:
      f=pq.ParquetFile(p,pre_buffer=False)
      rec.update(rows=f.metadata.num_rows,cols=f.schema_arrow.names,schema=str(f.schema_arrow),row_groups=f.num_row_groups)
      print('PARQUET',p.relative_to(ROOT),'ROWS',f.metadata.num_rows,'COLS',json.dumps(f.schema_arrow.names),flush=True)
     except Exception as ex: rec['error']=repr(ex)
    records.append(rec)
  except Exception as ex: errors.append({'path':str(folder),'error':repr(ex)})
  if not complete: break
 if not complete: break
save('inventory_TRE_ONLY.json',{'root_dirs':root_dirs,'modules':[str(x) for x in mods],'entries':visited,'complete':complete,'errors':errors,'records':records})
codehash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
save('run_metadata.json',{'python':sys.executable,'host':platform.node(),'code_sha256':codehash,'elapsed_sec':time.monotonic()-start,'no_participant_values_read':True})
print('INVENTORY_COMPLETE',complete,'ENTRIES',visited,'FILES',len(records),'ERRORS',len(errors),flush=True)
print('NONPARQUET',json.dumps([r['path'] for r in records if not r['path'].endswith(('.parquet','.pq'))]),flush=True)
print('CODE_SHA',codehash,'RESULT',str(OUT),flush=True)
if not complete or errors: raise RuntimeError('Incomplete inventory')
