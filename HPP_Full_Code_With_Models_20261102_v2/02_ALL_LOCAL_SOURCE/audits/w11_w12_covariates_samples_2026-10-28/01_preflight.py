# -*- coding: utf-8 -*-
"""W10 gate + bounded W11/W12 metadata preflight; no individual row output."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,csv,re,hashlib
import pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');R=Path('/home/ec2-user/studies/hpp_datasets');Q=P/'data_qc/w11_w12_covariates_samples_20261028';Q.mkdir(parents=True,exist_ok=True)
L=Path('/home/ec2-user/multisystem_runtime_20261021');W10=P/'data_qc/w10_outcome_qc_20261028'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
assert sys.executable=='/home/ec2-user/miniconda/envs/pheno/bin/python';assert (L/'w10_outcome_qc_20261028/07_actual_exit_code').read_text().strip()=='0'
d=json.loads((W10/'delivery_TRE_ONLY.json').read_text());assert d['passed'];am=json.loads((W10/'build_artifact_manifest_TRE_ONLY.json').read_text())
for a in am:assert sha(Path(a['path']))==a['sha256']
print('W10_GATE_PASS',len(am),'hashes; documented limits retained, no blocking structural failure')
mods=sorted(x.name for x in R.iterdir() if x.is_dir());print('ROOT_MODULES',','.join(mods))
m=json.loads((P/'manifests/w02_sources_20261026T130908Z/source_manifest.json').read_text());paths={x['path'] for x in m['sources'] if x['role'].startswith('candidate_')}
selectedmods=[x for x in mods if re.search(r'population|medic|lifestyle|sociodem|family|relation|oral|gut|question|sleep|activ|demo',x,re.I)]
for mod in selectedmods:
 top=sorted((R/mod).glob('*.parquet'));print('MODULE_FILES',mod,[x.name for x in top])
 if not re.search('oral|gut',mod):paths.update(str(p) for p in top)
for role in ['oral_index','oral_index_alternate','oral_species','gut_index','gut_species']:paths.add(next(x['path'] for x in m['sources'] if x['role']==role))
term=re.compile(r'age|sex|gender|birth|smok|alcohol|medicat|drug|antibio|diabet|insulin|metformin|family|relativ|household|batch|oral|dental|periodont|activity|sleep',re.I);sources=[];evidence=[]
for path in sorted(paths):
 p=Path(path);pf=pq.ParquetFile(p,pre_buffer=False);sch=[{'name':x.name,'type':str(x.type)} for x in pf.schema_arrow];keycols=[x for x in sch if re.search('participant|cohort|research_stage|sample|uuid|connection|date|timestamp|array_index|quality|valid|status|flag|reads|depth|batch|sex|gender|birth|age',x['name'],re.I)]
 print('SOURCE',str(p.relative_to(R)),'rows',pf.metadata.num_rows,'cols',len(sch));print('SOURCE_FIELDS',json.dumps(sch if len(sch)<100 else keycols));sources.append({'path':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'rows':pf.metadata.num_rows,'schema':sch})
for mod in selectedmods:
 for p in sorted((R/mod/'metadata').glob('*.csv')):
  if p.stat().st_size>8*1024**2:continue
  with p.open(encoding='utf-8-sig',newline='') as f:
   for i,row in enumerate(csv.DictReader(f)):
    assert i<30000
    if term.search(str(row)):
     evidence.append({'file':str(p),'row':row})
     # Save all evidence in TRE; print a short field/label and encoding pointer only.
     n=row.get('tabular_field_name','')
     if term.search(n) or mod=='population':print('DICT',mod,n,row.get('field_string'),row.get('units'),row.get('data_coding'),row.get('description_string','')[:180])
(Q/'preflight_sources_TRE_ONLY.json').write_text(json.dumps(sources,indent=2));(Q/'preflight_dictionary_TRE_ONLY.json').write_text(json.dumps(evidence,indent=2));print('PREFLIGHT_COMPLETE',len(sources),'sources',len(evidence),'dictionary rows',flush=True)
