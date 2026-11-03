# -*- coding: utf-8 -*-
"""Audit actual oral-health field spelling; complete optional source integration with preserved prior artifacts."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,csv,hashlib,shutil
from collections import defaultdict
import pandas as pd,numpy as np,pyarrow as pa,pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');R=Path('/home/ec2-user/studies/hpp_datasets');Q=P/'data_qc/w11_w12_covariates_samples_20261028';L=Path('/home/ec2-user/multisystem_runtime_20261021/w11_w12_covariates_samples_20261028');K=['participant_id','cohort','research_stage'];BAD={'','undefined','none','null','nan','na','n/a','unknown','<na>'}
assert (L/'07_actual_exit_code').read_text().strip()=='0';M=json.loads((Q/'w11_artifact_manifest_TRE_ONLY.json').read_text());revisions=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for e in M:assert sha(Path(e['path']))==e['sha256']
def archive(p):
 target=Q/'w11_before_oral_health_completion'/p.relative_to(P);assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target);assert sha(target)==sha(p);revisions.append({'path':str(p),'archived_path':str(target),'before_sha256':sha(p)})
regpath=P/'analysis/shared/covariate_registry_TRE_ONLY.parquet';config=P/'config/covariate_roles.yaml';summarypath=Q/'w11_summary_TRE_ONLY.json';csvpath=P/'data_qc/covariate_missingness.csv';d=pq.ParquetFile(regpath).read(use_threads=False).to_pandas(ignore_metadata=True);C=json.loads(config.read_text());S=json.loads(summarypath.read_text());source=R/'health_and_medical_history/initial_medical.parquet';pf=pq.ParquetFile(source);names=[n for n in pf.schema_arrow.names if 'antibiot' in n.lower() and ('oral' in n.lower() or 'ging' in n.lower())];E=json.loads((Q/'preflight_dictionary_TRE_ONLY.json').read_text());coding=list(csv.DictReader((R/'metadata/coding_mapping.csv').open(encoding='utf-8-sig',newline='')));status={'source_path':str(source),'source_sha256':sha(source),'source_rows':pf.metadata.num_rows,'source_mtime_ns':source.stat().st_mtime_ns,'coding_source_path':str(R/'metadata/coding_mapping.csv'),'coding_source_sha256':sha(R/'metadata/coding_mapping.csv'),'actual_source_fields':names,'completed':False,'main_registry_rows_unchanged':True,'biological_meaning_not_expanded_to_general_antibiotic_use':True}
if len(names)==1:
 field=names[0];label=field+'_label';status['source_field']=field
 if label in d:status['completed']=True;status['already_present']=True
 else:
  e=[e['row'] for e in E if e['row'].get('tabular_field_name')==field and 'health_and_medical_history' in e['file']];assert len(e)==1;e=e[0];code=e.get('data_coding');cm={r['coding']:r['english'] for r in coding if r['code_number']==code};assert cm;valid={k:v for k,v in cm.items() if not k.startswith('-') and not any(t in v.lower() for t in ['prefer not','do not know','refus'])};raw=pf.read(columns=K+['collection_date',field],use_threads=False).to_pylist();groups=defaultdict(list)
  for row in raw:
   if any(row[k] is None or str(row[k]).lower() in BAD for k in K):continue
   if int(row['participant_id'])<=0:continue
   groups[(int(row['participant_id']),str(row['cohort']),str(row['research_stage']))].append(row)
  records=[]
  for key,rs in groups.items():
   vals=set()
   for r in rs:
    v=r[field]
    if v is not None:
     try:f=float(v);v=str(int(f)) if f.is_integer() else str(v)
     except (ValueError,TypeError):v=str(v)
     if v.lower() not in BAD:vals.add(v)
   record=dict(zip(K,key));record[field+'_source_status']='consensus' if len(vals)==1 else 'missing' if not vals else 'conflicting_repeat';record[field+'_raw_code']=next(iter(vals)) if len(vals)==1 else None;record[label]=valid.get(record[field+'_raw_code']);dates=[r['collection_date'] for r in rs if r['collection_date'] is not None];record['oralhealth_date_min']=min(dates) if dates else None;record['oralhealth_date_max']=max(dates) if dates else None;record['oralhealth_source_rows']=len(rs);records.append(record)
  oh=pd.DataFrame(records);original_keys=d[K].copy();d=d.merge(oh,on=K,how='left',validate='one_to_one');assert d[K].equals(original_keys)
  def rel(r):
   lo,hi,first,last=r.oralhealth_date_min,r.oralhealth_date_max,r.food_first,r.food_last
   if any(pd.isna(x) for x in [lo,hi,first,last]):return 'unknown'
   if hi<first:return 'recorded_before_food_window'
   if lo>last:return 'recorded_after_food_window'
   return 'recorded_within_or_spanning_food_window'
  d['oralhealth_recording_time_vs_food']=[rel(r) for r in d.itertuples()];C['roles'].append({'variable':field,'role':'sensitivity_candidate_self_reported_oral_health','source':str(source),'coding_id':code,'coding':cm,'source_description':e.get('description_string'),'not_general_recent_antibiotic_use':True,'source_spelling_preserved':True,'current_actual_use_not_inferred':True});C['oral_health_source_completion']='Actual provided source spelling used after explicit schema comparison; no approximate name match across datasets.'
  b=d.loc[d.cohort.eq('10k')&d.research_stage.eq('00_00_visit')];n=int(b[label].notna().sum());S['baseline_counts'][label]={'available':n,'missing_or_unavailable':len(b)-n};S['oral_health_source_field']=field;S['oral_health_baseline_labels']=b[label].fillna('unknown').value_counts().astype(int).to_dict();S['oral_health_recording_time_vs_food']=b.oralhealth_recording_time_vs_food.value_counts().astype(int).to_dict()
  for p in [regpath,config,summarypath,csvpath]:archive(p)
  runtime=L/'covariate_registry_with_oral_health_v2_TRE_ONLY.parquet';assert not runtime.exists();pq.write_table(pa.Table.from_pandas(d,preserve_index=False),runtime,compression='zstd');shutil.copyfile(runtime,regpath);assert sha(runtime)==sha(regpath) and pq.ParquetFile(regpath).metadata.num_rows==len(d);config.write_text(json.dumps(C,indent=2,ensure_ascii=False));summarypath.write_text(json.dumps(S,indent=2,ensure_ascii=False));m=pd.read_csv(csvpath);m=pd.concat([m,pd.DataFrame([{'variable':label,'baseline_denominator':len(b),'available':n,'missing_or_unavailable':len(b)-n}])],ignore_index=True);m.to_csv(csvpath,index=False)
  for entry in M:
   p=Path(entry['path'])
   if str(p) in {r['path'] for r in revisions}:entry.update(bytes=p.stat().st_size,sha256=sha(p));entry['runtime_source']=str(runtime) if p==regpath else entry.get('runtime_source')
  status.update(completed=True,already_present=False,baseline_available=n,baseline_unknown=len(b)-n,baseline_labels=S['oral_health_baseline_labels'],source_field=field,source_dictionary_code=code)
for e in M:
 p=Path(e['path']);assert sha(p)==e['sha256']
 if e.get('runtime_source'):assert sha(Path(e['runtime_source']))==sha(p)
(Q/'w11_current_artifact_manifest_TRE_ONLY.json').write_text(json.dumps(M,indent=2));(Q/'w11_artifact_revisions_TRE_ONLY.json').write_text(json.dumps(revisions,indent=2));(Q/'oral_health_completion_TRE_ONLY.json').write_text(json.dumps(status,indent=2));print('W11_ORAL_HEALTH_AUDIT',json.dumps(status),flush=True);print('CURRENT_W11_MANIFEST_VERIFIED',len(M),'PRESERVED_PRIOR_REVISIONS',len(revisions),flush=True)
