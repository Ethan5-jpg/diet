"""Bounded schema/dictionary inspection only; no participant values printed."""
from pathlib import Path
import json, sys, hashlib, csv, datetime
import pyarrow.parquet as pq

ROOT=Path('/home/ec2-user/studies/hpp_datasets')
OUT=Path('/home/ec2-user/studies/diet_gut_cgm_v2/phenotype_food_dates_20261018')
assert '/envs/pheno/' in sys.executable
OUT.mkdir(exist_ok=False)
source=Path(__file__).read_bytes()
(OUT/'10_discover.py').write_bytes(source)
mods=sorted(p.name for p in ROOT.iterdir() if p.is_dir())
print('MODULES',','.join(mods),flush=True)
selected=[m for m in mods if any(t in m.lower() for t in ['body','anthrop','liver','diet','food'])]
report={'python':sys.executable,'source_sha256':hashlib.sha256(source).hexdigest(),'modules':mods,'selected':{}}
for m in selected:
    d=ROOT/m
    item={'files':[],'dictionaries':[]}
    for p in sorted(d.glob('*.parquet')):
        f=pq.ParquetFile(p,pre_buffer=False)
        s=[(x.name,str(x.type)) for x in f.schema_arrow]
        item['files'].append({'path':str(p),'rows':f.metadata.num_rows,'schema':s,'bytes':p.stat().st_size})
        print('TABLE',m,p.name,'ROWS',f.metadata.num_rows,'COLS',len(s),flush=True)
        print('SCHEMA',json.dumps(s if len(s)<75 else s[:15]+[x for x in s[15:] if any(t in x[0].lower() for t in ['date','time','food','fat','liver','bmi','weight','waist'])],ensure_ascii=False),flush=True)
    for p in sorted((d/'metadata').glob('*data_dictionary.csv')):
        with p.open(newline='') as h: rows=list(csv.DictReader(h))
        hits=[r for r in rows if any(t in str(r).lower() for t in ['date','timestamp','logging_day','event_id','food_id','fat_percentage','visceral','liver','cap','stiffness'])]
        item['dictionaries'].append({'path':str(p),'matching_rows':hits})
        print('DICTIONARY',p.name,json.dumps(hits[:18],ensure_ascii=False),flush=True)
    report['selected'][m]=item
(OUT/'discovery.json').write_text(json.dumps(report,indent=2,default=str))
(OUT/'MEMO.md').write_text('# Phenotype and food dates audit\n\n10_discover.py: bounded module/schema/date-field dictionary inspection; no participant values printed. Root and OUT are modification points. Discovery is not timing analysis.\n')
(OUT/'10_exit_code').write_text('0\n')
print('DISCOVERY_COMPLETE',OUT,'SOURCE_SHA256',report['source_sha256'],flush=True)
