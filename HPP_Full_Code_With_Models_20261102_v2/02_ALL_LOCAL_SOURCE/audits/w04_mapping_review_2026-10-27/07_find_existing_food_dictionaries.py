# -*- coding: utf-8 -*-
from pathlib import Path
import json,os,time,ast,csv
import pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');Q=P/'data_qc/w04_mapping_review_20261027'
old=json.loads((Q/'old_mapping_script_evidence_TRE_ONLY.json').read_text())
args=[]
for x in old:
 t=Path(x['path']).read_text();a=[]
 for n in ast.walk(ast.parse(t)):
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument':
   flags=[ast.literal_eval(z) for z in n.args if isinstance(z,ast.Constant)]
   defaults={z.arg:ast.unparse(z.value) for z in n.keywords if z.arg in ['default','required','type']}
   a.append({'flags':flags,**defaults})
 hits=[{'line':i,'text':s.strip()[:300]} for i,s in enumerate(t.splitlines(),1) if any(k in s.lower() for k in ['read_csv','read_excel','read_parquet','to_csv','to_parquet','.xlsx','.csv','.parquet','output_dir','dictionary_path'])]
 args.append({'path':x['path'],'arguments':a,'IO_lines':hits})
 print('BUILDER',x['path'],flush=True)
 for z in a:print('ARGUMENT',json.dumps(z),flush=True)
 for z in hits[:20]:print('IO_LINE',z['line'],z['text'],flush=True)
root=Path('/home/ec2-user/studies/diet_gut_cgm');stack=[(root,0)];found=[];count=0;start=time.monotonic();complete=True
while stack:
 p,depth=stack.pop()
 if depth>8:continue
 for e in os.scandir(p):
  count+=1
  if count>10000 or time.monotonic()-start>60:complete=False;stack=[];break
  if e.is_dir(follow_symlinks=False):
   if e.name not in ['.git','__pycache__']:stack.append((Path(e.path),depth+1))
   continue
  low=e.name.lower()
  if not any(k in low for k in ['dictionary','catalog','classif','override','nova','food','indicator']):continue
  f=Path(e.path);r={'path':str(f),'bytes':e.stat().st_size}
  if f.suffix=='.parquet' and r['bytes']>0:
   pf=pq.ParquetFile(f,pre_buffer=False);r.update(rows=pf.metadata.num_rows,columns=pf.schema_arrow.names)
  elif f.suffix in ['.csv','.tsv'] and r['bytes']>0:
   with f.open() as h:r['columns']=next(csv.reader(h,delimiter='\t' if f.suffix=='.tsv' else ','),[])
  found.append(r)
print('FOOD_DICTIONARY_SEARCH',complete,'ENTRIES',count,'FILES',len(found),flush=True)
for r in found:
 if Path(r['path']).suffix in ['.csv','.tsv','.parquet','.xlsx','.json']:
  print('FOOD_CANDIDATE',json.dumps(r),flush=True)
(Q/'existing_food_dictionary_sources_TRE_ONLY.json').write_text(json.dumps({'builders':args,'search_root':str(root),'complete_within_depth':complete,'entries':count,'files':found},indent=2))
print('EXISTING_MAPPING_SEARCH_DONE',flush=True)
