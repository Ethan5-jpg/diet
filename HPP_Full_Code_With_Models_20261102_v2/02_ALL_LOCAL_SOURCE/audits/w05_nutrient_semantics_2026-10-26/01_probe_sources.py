"""TRE-only W05 source/schema/dictionary probe; no participant values printed."""
import csv,hashlib,json,re,socket,sys
from pathlib import Path
R=Path('/home/ec2-user/studies/hpp_datasets/diet_logging')
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
import pyarrow.parquet as pq
print('\n'*30+'W05_SOURCE_PROBE',socket.gethostname(),sys.executable)
for n in ['diet_logging.parquet','diet_logging_events.parquet','raw_diet_logging_events.parquet']:
 p=R/n;pf=pq.ParquetFile(p,pre_buffer=False)
 names=pf.schema_arrow.names
 cols=[c for c in names if re.search('weight|calori|fiber|carbo|lipid|protein|alcohol|serv|portion|unit|raw|cook|delet|event|food|sugar|energy',c,re.I)]
 print('SCHEMA',n,'rows',pf.metadata.num_rows,json.dumps([(c,str(pf.schema_arrow.field(c).type)) for c in cols],separators=(',',':')))
 print('SCHEMA_METADATA_KEYS',n,[k.decode(errors='replace') for k in (pf.schema_arrow.metadata or {})])
print('MODULE_CONTENTS',[(p.name,'dir' if p.is_dir() else p.stat().st_size) for p in sorted(R.iterdir())])
meta=R/'metadata'
for p in sorted(meta.rglob('*')):
 if not p.is_file() or len(p.relative_to(meta).parts)>2:continue
 print('META_FILE',str(p.relative_to(R)),p.stat().st_size)
 if p.suffix.lower()=='.csv' and p.stat().st_size<4*1024**2:
  with p.open(encoding='utf-8-sig',newline='') as f:
   reader=csv.DictReader(f);print('DICT_HEADERS',reader.fieldnames)
   for i,row in enumerate(reader,2):
    if i>10001:break
    if re.search('weight|calori|fiber|carbo|lipid|protein|alcohol|nutri|usda|serving|portion',str(row),re.I):print('DICT_MATCH',p.name,i,json.dumps(row,separators=(',',':'))[:1700])
print('W05_PROBE_DONE')
