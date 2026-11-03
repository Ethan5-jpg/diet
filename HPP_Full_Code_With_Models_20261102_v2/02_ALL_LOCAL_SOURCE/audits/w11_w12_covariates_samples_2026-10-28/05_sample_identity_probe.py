# -*- coding: utf-8 -*-
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,re,hashlib,pickle
from collections import Counter,defaultdict
from datetime import datetime,date
import pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');R=Path('/home/ec2-user/studies/hpp_datasets');Q=P/'data_qc/w11_w12_covariates_samples_20261028';L=Path('/home/ec2-user/multisystem_runtime_20261021/w11_w12_covariates_samples_20261028');K=['participant_id','cohort','research_stage'];BAD={'','undefined','null','none','nan','na','n/a','unknown','<na>'}
M=json.loads((P/'manifests/w02_sources_20261026T130908Z/source_manifest.json').read_text());out={};info=[]
for role in ['oral_index','oral_index_alternate','oral_species','gut_index','gut_species']:
 e=next(s for s in M['sources'] if s['role']==role);p=Path(e['path']);pf=pq.ParquetFile(p,pre_buffer=False);names=pf.schema_arrow.names;assert pf.metadata.num_rows<100000
 cols=[n for n in names if n in K or re.search(r'(^collection_|^array_index$|sample|uuid|batch|reads|quality|qc|status|kit|tube|barcode)',n,re.I)];rows=[r for b in pf.iter_batches(columns=cols,batch_size=1024,use_threads=False) for r in b.to_pylist()];out[role]=rows
 def valid(r):
  if any(r[k] is None or str(r[k]).strip().lower() in BAD or str(r[k])!=str(r[k]).strip() for k in K):return False
  try:return int(r[K[0]])>0
  except (TypeError,ValueError):return False
 good=[r for r in rows if valid(r)];gc=Counter(tuple(str(r[k]) for k in K) for r in good);base=[r for r in good if str(r['cohort'])=='10k' and str(r['research_stage'])=='00_00_visit'];summary={'role':role,'rows':len(rows),'physical_columns':len(names),'identity_columns':cols,'non_human_columns':[n for n in names if not n.startswith('human_')][:45] if role.endswith('index') or role.endswith('alternate') else [n for n in names if not n.startswith(('k__','d__','s__')) and n in cols],'valid_key_rows':len(good),'invalid_key_rows':len(rows)-len(good),'unique_valid_person_visits':len(gc),'multirow_person_visits':sum(v>1 for v in gc.values()),'baseline_rows':len(base),'baseline_people':len({r['participant_id'] for r in base}),'column_nonmissing':{n:sum(r[n] is not None for r in rows) for n in cols if n not in K},'source_path':str(p),'source_bytes':p.stat().st_size,'source_mtime_ns':p.stat().st_mtime_ns};info.append(summary);print('SAMPLE_IDENTITY',json.dumps(summary),flush=True)
# Short timing-based ambiguity diagnostics for gut; never expands both many sides.
def day(r):
 x=r.get('collection_date');return x.date().isoformat() if isinstance(x,datetime) else x.isoformat() if isinstance(x,date) else None
ix=defaultdict(list);ab=defaultdict(list)
for r in out['gut_index']:
 if valid(r):ix[tuple(str(r[k]) for k in K)].append(r)
for r in out['gut_species']:
 if valid(r):ab[tuple(str(r[k]) for k in K)].append(r)
classes=Counter();dateclass=Counter()
for k,ar in ab.items():
 ir=ix.get(k,[]);classes['both_many' if len(ir)>1 and len(ar)>1 else 'one_to_one' if len(ir)==len(ar)==1 else 'index_missing' if not ir else 'other_cardinality']+=1
 for r in ar:
  matches=[z for z in ir if day(r) is not None and day(z)==day(r)];dateclass['exact_day_unique' if len(matches)==1 else 'exact_day_ambiguous' if len(matches)>1 else 'no_exact_day']+=1
print('GUT_CARDINALITY',dict(classes),'ABUNDANCE_DATE_MATCHES',dict(dateclass),flush=True)
(Q/'sample_identity_probe_TRE_ONLY.json').write_text(json.dumps({'sources':info,'gut_key_cardinality':dict(classes),'gut_abundance_date_matches':dict(dateclass)},indent=2,default=str));checkpoint=L/'sample_identity_projections_TRE_ONLY.pkl';assert not checkpoint.exists();checkpoint.write_bytes(pickle.dumps(out,protocol=4));print('SAMPLE_IDENTITY_PROBE_DONE',flush=True)
