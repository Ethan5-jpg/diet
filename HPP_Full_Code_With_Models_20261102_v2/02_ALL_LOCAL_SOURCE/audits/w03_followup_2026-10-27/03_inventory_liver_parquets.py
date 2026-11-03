"""TRE-only recursive liver Parquet inventory and schema comparison, no rows read."""
import os,time,json,hashlib,sys
from pathlib import Path
import pyarrow.parquet as pq
R=Path('/home/ec2-user/studies/hpp_datasets/liver_ultrasound')
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
O=P/'data_qc/w03_liver_schema_20261027';O.mkdir(parents=True,exist_ok=True)
S=P/'scripts/w03_liver_schema_20261027';S.mkdir(parents=True,exist_ok=True)
families=['elasticity','velocity','viscosity','dispersion','attenuation','speed_of_sound']
files=[];others=[];seen=0;t=time.monotonic();stack=[R]
while stack:
 root=stack.pop()
 for e in os.scandir(root):
  seen+=1;assert seen<=10000 and time.monotonic()-t<120,'Bounded module traversal incomplete'
  if e.is_symlink():others.append({'path':e.path,'kind':'symlink_not_followed'});continue
  if e.is_dir(follow_symlinks=False):stack.append(Path(e.path));continue
  if not e.is_file(follow_symlinks=False):continue
  if e.name.lower().endswith(('.parquet','.pq')):files.append(Path(e.path))
  else:others.append({'path':e.path,'kind':'other_file','bytes':e.stat().st_size})
rows=[]
for p in sorted(files):
 st=p.stat();pf=pq.ParquetFile(p,pre_buffer=False);cols=[{'name':f.name,'type':str(f.type)} for f in pf.schema_arrow];meta=pf.schema_arrow.metadata or {};pandas=json.loads(meta.get(b'pandas',b'{}'))
 indexed=pandas.get('index_columns',[]);measurement=[f for f in cols if any(k in f['name'] for k in families)]
 rows.append({'path':str(p),'relative':str(p.relative_to(R)),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'rows_metadata':pf.metadata.num_rows,'row_groups':pf.metadata.num_row_groups,'physical_columns':len(cols),'pandas_index_columns':indexed,'known_liver_measurement_columns':len(measurement),'family_counts':{k:sum(k in f['name'] for f in cols) for k in families},'columns':cols,'source_unchanged':p.stat().st_size==st.st_size and p.stat().st_mtime_ns==st.st_mtime_ns})
 assert rows[-1]['source_unchanged']
result={'scope':str(R),'traversal_complete_within_scope':True,'directories_recursive':True,'symlinks_not_followed':True,'parquet_or_pq_files':len(rows),'directory_entries_seen':seen,'sources':rows,'other_files':others,'no_data_rows_read':True,'elapsed_s':round(time.monotonic()-t,3)}
b=(json.dumps(result,indent=2)+'\n').encode();(O/'all_liver_parquet_inventory_TRE_ONLY.json').write_bytes(b);assert (O/'all_liver_parquet_inventory_TRE_ONLY.json').read_bytes()==b
code=S/'03_inventory_liver_parquets.py';code.write_bytes(Path(__file__).read_bytes())
(O/'PARQUET_INVENTORY_MEMO.md').write_text('03_inventory_liver_parquets.py recursively enumerates liver_ultrasound only (10000 entries/120s guard), reads every discovered Parquet footer/schema and pandas index metadata. No clinical rows. Full source field lists and non-Parquet listing remain TRE in all_liver_parquet_inventory_TRE_ONLY.json. A small column count is not inferred from file name or size. Does not search the whole data root or unrelated derived projects.\n')
print('\n'*30+'ALL_LIVER_PARQUETS_INVENTORY_PASS')
print('ROOT',R,'FILES',len(rows),'ENTRIES',seen,'COMPLETE',True,'NO_DATA_ROWS_READ',True)
for r in rows:
 print('FILE',r['relative'],'ROWS',r['rows_metadata'],'COLUMNS',r['physical_columns'],'MEASURE_COLUMNS',r['known_liver_measurement_columns'],'INDEX_COLUMNS',json.dumps(r['pandas_index_columns']))
 print('FAMILY_COUNTS',json.dumps(r['family_counts']))
 for f in r['columns']:print('COLUMN',f['name'],f['type'])
print('OTHER_FILES',[(str(Path(x['path']).relative_to(R)),x['kind']) for x in others]);print('CODE_SHA256',hashlib.sha256(Path(__file__).read_bytes()).hexdigest());print('RESULT',O/'all_liver_parquet_inventory_TRE_ONLY.json');print('LIVER_INVENTORY_DONE')
