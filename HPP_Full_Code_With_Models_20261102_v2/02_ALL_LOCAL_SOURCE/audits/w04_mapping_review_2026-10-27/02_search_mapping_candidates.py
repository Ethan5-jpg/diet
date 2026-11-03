# -*- coding: utf-8 -*-
from pathlib import Path
import os,time,json,csv,sys
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
Q=P/'data_qc/w04_mapping_review_20261027'
inv=json.loads((Q/'inventory_TRE_ONLY.json').read_text())
print('MOUNTED_DATA_MODULES',','.join(inv['root_dirs']),flush=True)
D=[]
for r in inv['records']:
 if r['path'].endswith('.csv'):
  rows=list(csv.DictReader(open(r['path'])))
  hits=[x for x in rows if any(t in str(x).lower() for t in ['food_id','food_category','short_food_name','product_name'])]
  D.append({'path':r['path'],'fields':hits})
  print('DICT',Path(r['path']).name,json.dumps(hits),flush=True)
start=time.monotonic();n=0;found=[];complete=True
roots=[Path('/home/ec2-user/studies/diet_gut_cgm'),Path('/home/ec2-user/studies/diet_gut_cgm_v2')]
for root in roots:
 stack=[(root,0)]
 while stack:
  folder,depth=stack.pop()
  if depth>7: continue
  try:
   for e in os.scandir(folder):
    n+=1
    if n>40000 or time.monotonic()-start>100:
     complete=False; stack=[]; break
    low=e.name.lower()
    if any(x in low for x in ['mapping','nova','food_class','food_categor','supplementary_table_2','supplementary2']):
     found.append({'path':e.path,'directory':e.is_dir(follow_symlinks=False)})
    if e.is_dir(follow_symlinks=False) and low not in ['.git','node_modules','__pycache__','.ipynb_checkpoints']:
     stack.append((Path(e.path),depth+1))
  except (PermissionError,FileNotFoundError): pass
  if not complete: break
 if not complete: break
(Q/'search_mapping_candidates_TRE_ONLY.json').write_text(json.dumps({'roots':list(map(str,roots)),'max_depth':7,'entries':n,'complete_within_depth':complete,'found':found,'diet_dictionary':D},indent=2))
print('SEARCH_COMPLETE_WITHIN_DEPTH',complete,'ENTRIES',n,'CANDIDATES',len(found),flush=True)
for x in found: print('CANDIDATE',x['path'],flush=True)
print('NO_USDA_MODULE_IN_CURRENT_DATASET_MOUNT',not any('auxiliary' in x or 'usda' in x for x in inv['root_dirs']),flush=True)
