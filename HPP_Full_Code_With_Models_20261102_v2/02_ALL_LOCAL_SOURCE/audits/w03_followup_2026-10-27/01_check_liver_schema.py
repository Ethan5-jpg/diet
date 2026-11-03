"""TRE metadata-only liver source check; no outcome or participant values read."""
from pathlib import Path
import csv,json,hashlib,sys
import pyarrow.parquet as pq
R=Path('/home/ec2-user/studies/hpp_datasets/liver_ultrasound')
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
O=P/'data_qc/w03_liver_schema_20261027';O.mkdir(parents=True,exist_ok=True)
sources=[]
for name in ['liver_ultrasound.parquet','liver_ultrasound_aggregated.parquet']:
 p=R/name;f=pq.ParquetFile(p,pre_buffer=False);s=p.stat();sources.append({'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'rows':f.metadata.num_rows,'columns':[{'name':x.name,'type':str(x.type)} for x in f.schema_arrow]})
evidence=[]
for p in (R/'metadata').glob('*.csv'):
 assert p.stat().st_size<4*1024**2
 with p.open(encoding='utf-8-sig',newline='') as h:
  for i,r in enumerate(csv.DictReader(h)):
   assert i<10000
   if any(x in str(r).lower() for x in ['elasticity','velocity','viscosity','dispersion','attenuation','speed_of_sound']):evidence.append({'file':str(p),'row':r})
(O/'source_metadata_TRE_ONLY.json').write_text(json.dumps(sources,indent=2)+'\n');(O/'dictionary_evidence_TRE_ONLY.json').write_text(json.dumps(evidence,indent=2)+'\n')
S=P/'scripts/w03_liver_schema_20261027';S.mkdir(parents=True,exist_ok=True);(S/'01_check_liver_schema.py').write_bytes(Path(__file__).read_bytes())
memo='''# W03 liver source metadata check\n01_check_liver_schema.py reads only both liver source footers/schemas and local dictionary CSVs. No individual keys, phenotype values, associations or quality selection. Confirms current source column inventory; six intended study fields require explicit selection, not choosing arbitrary mean/median variants.\nSource metadata and dictionary evidence remain TRE in data_qc/w03_liver_schema_20261027.\n''';(O/'MEMO.md').write_text(memo)
print('\n'*30+'W03_LIVER_SCHEMA_PROBE')
for s in sources:
 print('SOURCE',Path(s['path']).name,'ROWS',s['rows'],'COLUMNS',len(s['columns']))
 cols=[x['name'] for x in s['columns']]
 for family in ['elasticity','velocity','viscosity','dispersion','attenuation','speed_of_sound']:
  print('FAMILY',family,json.dumps([c for c in cols if family in c]))
print('DICTIONARY_MATCHES',len(evidence));print('RESULT',O);print('CODE_SHA256',hashlib.sha256(Path(__file__).read_bytes()).hexdigest());print('NO_PHENOTYPE_VALUES_READ');print('W03_LIVER_PROBE_DONE')
