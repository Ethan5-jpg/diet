"""Evidence-only oral visit bridge; exact person/cohort/calendar date, never nearest date or baseline imputation."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import hashlib,json,resource
from collections import Counter,defaultdict
from pathlib import Path
from datetime import datetime,date
import pyarrow as pa
import pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
RUN='w03_keys_20261026T134938Z'
BAD={'','undefined','null','none','nan','na','n/a','unknown','<na>'}
def valid(v):return v is not None and str(v)==str(v).strip() and str(v).lower() not in BAD

def save(p,v):
 b=(json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode();p.write_bytes(b);assert p.read_bytes()==b

def main():
 resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
 manifest=json.loads((P/'manifests/w02_sources_20261026T130908Z/source_manifest.json').read_text());e={r['role']:r for r in manifest['sources']}
 rows={}
 for role in ('oral_index','oral_index_alternate','oral_species'):
  p=Path(e[role]['path']);assert p.stat().st_size==e[role]['bytes'] and p.stat().st_mtime_ns==e[role]['mtime_ns']
  pf=pq.ParquetFile(p,pre_buffer=False); cols=['participant_id','cohort','research_stage','collection_date']; assert all(c in pf.schema_arrow.names for c in cols)
  rows[role]=[r for b in pf.iter_batches(batch_size=1024,columns=cols,use_threads=False) for r in b.to_pylist()]
  assert len(rows[role])==pf.metadata.num_rows
 indices=defaultdict(list)
 for ordinal,r in enumerate(rows['oral_index'],1):
  dt=r['collection_date'];day=dt.date().isoformat() if isinstance(dt,datetime) else dt.isoformat() if isinstance(dt,date) else None
  if all(valid(r[c]) for c in ('participant_id','cohort','research_stage')) and day:
   indices[(str(r['participant_id']),day)].append((ordinal,str(r['cohort']),str(r['research_stage'])))
 mapping=[];stats=Counter();invalid_fields=Counter()
 for ordinal,r in enumerate(rows['oral_species'],1):
  for c in ('participant_id','cohort','research_stage'):
   if not valid(r[c]): invalid_fields[c]+=1
  dt=r['collection_date'];day=dt.date().isoformat() if isinstance(dt,datetime) else dt.isoformat() if isinstance(dt,date) else None
  candidates=indices.get((str(r['participant_id']),day),[]) if valid(r['participant_id']) and day else []
  # Respect any already-valid original cohort/stage, even if it prevents a match.
  candidates=[x for x in candidates if (not valid(r['cohort']) or x[1]==str(r['cohort'])) and (not valid(r['research_stage']) or x[2]==str(r['research_stage']))]
  candidates=list(dict.fromkeys(candidates))
  status='resolved_exact_person_calendar_index_row' if len(candidates)==1 else 'unresolved_no_exact_match' if not candidates else 'unresolved_ambiguous_index_rows'
  stats[status]+=1;chosen=candidates[0] if len(candidates)==1 else None
  if chosen and chosen[1:] == ('10k','00_00_visit'):stats['resolved_baseline_rows']+=1
  mapping.append({'abundance_row_ordinal':ordinal,'participant_id':str(r['participant_id']),'source_cohort':None if r['cohort'] is None else str(r['cohort']),'source_stage':None if r['research_stage'] is None else str(r['research_stage']),'calendar_date':day,'index_row_ordinal':chosen[0] if chosen else None,'evidence_cohort':chosen[1] if chosen else None,'evidence_stage':chosen[2] if chosen else None,'status':status})
 counts=Counter((r['participant_id'],r['evidence_cohort'],r['evidence_stage']) for r in mapping if r['index_row_ordinal'])
 stats['resolved_person_visits']=len(counts);stats['resolved_person_visits_with_multiple_abundance_rows']=sum(n>1 for n in counts.values())
 stats['resolved_baseline_person_visits']=sum(k[1:]==('10k','00_00_visit') for k in counts)
 q=P/'data_qc'/RUN;cfg=P/'config'/RUN;m=P/'manifests'/RUN
 out=q/'oral_visit_evidence_map_TRE_ONLY.parquet';tab=pa.Table.from_pylist(mapping);pq.write_table(tab,out);assert pq.ParquetFile(out).metadata.num_rows==len(mapping)
 result={'run_id':RUN,'source_rows':len(mapping),'original_invalid_key_field_counts':dict(invalid_fields),'mapping_counts':dict(stats),'rule':'Unique valid oral_index row matching participant_id and exact source collection_date calendar date, respecting any valid original cohort/stage. Does not infer baseline from participant alone; unresolved rows held.','not_final_biological_sample_registry':True,'index_version':'oral_index only; alternate not pooled','map':str(out)}
 save(q/'oral_visit_bridge_summary.json',result)
 contract=json.loads((cfg/'key_contract.yaml').read_text());contract['oral_visit_evidence_bridge']=result;contract['microbiome']['sample_mapping']+=' Oral stage bridge only uses exact calendar and unique explicit index evidence; not sample-QC or version equivalence. Repeated resolved abundance rows remain W11 gate.';save(cfg/'key_contract.yaml',contract)
 memo=P/'memos'/RUN/'MEMO.md';memo.write_text(memo.read_text()+'\n04_resolve_oral_visit_evidence.py: follow-up full 3-source key/date scan after raw composite-key failures. Unique exact-date evidence only; no outcome or abundance feature read; original columns never overwritten.\n'+json.dumps(result,indent=2)+'\n')
 code=P/'scripts'/RUN/'04_resolve_oral_visit_evidence.py';code.write_bytes(Path(__file__).read_bytes());assert code.read_bytes()==Path(__file__).read_bytes()
 receipts=json.loads((m/'artifact_manifest.json').read_text());files={x['path'] for x in receipts}|{str(out),str(q/'oral_visit_bridge_summary.json'),str(code)}
 receipts=[{'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in map(Path,sorted(files))];save(m/'artifact_manifest.json',receipts);assert all(hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256'] for x in receipts)
 print('\n'*20+'W03_ORAL_BRIDGE_SUMMARY',json.dumps(result,separators=(',',':')),flush=True)
 print('ORAL_BRIDGE_HASHES_VERIFIED',len(receipts),'CODE_SHA256',hashlib.sha256(code.read_bytes()).hexdigest(),flush=True)
 return 0
if __name__=='__main__':raise SystemExit(main())
