# -*- coding: utf-8 -*-
"""W12 all-source candidate ledger and immutable selection contract, no final QC or selection."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
from collections import defaultdict,Counter
from datetime import date,datetime
import json,hashlib,shutil,bisect
import pandas as pd,pyarrow as pa,pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');R=Path('/home/ec2-user/studies/hpp_datasets');Q=P/'data_qc/w11_w12_covariates_samples_20261028';L=Path('/home/ec2-user/multisystem_runtime_20261021/w11_w12_covariates_samples_20261028');K=['participant_id','cohort','research_stage'];BAD={'','undefined','none','null','nan','na','n/a','unknown','<na>'};SEED=20261027
M=json.loads((P/'manifests/w02_sources_20261026T130908Z/source_manifest.json').read_text());J=json.loads((Q/'preflight_sources_TRE_ONLY.json').read_text());roles=['oral_index','oral_index_alternate','oral_species','gut_index','gut_species'];sources={};rows={};artifacts=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def good(x):return x is not None and str(x).lower() not in BAD and str(x)==str(x).strip()
def key(r):
 if not all(good(r.get(k)) for k in K):return None
 try:
  p=int(r['participant_id']);assert p>0 and p==float(r['participant_id']);return p,str(r['cohort']),str(r['research_stage'])
 except (ValueError,TypeError,AssertionError):return None
def day(x):
 if x is None or pd.isna(x):return None
 if isinstance(x,datetime):return x.date().isoformat()
 if isinstance(x,date):return x.isoformat()
 return None

def dump(p,v):
 assert not p.exists(),str(p);p.parent.mkdir(parents=True,exist_ok=True);b=(json.dumps(v,ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n').encode();p.write_bytes(b);assert p.read_bytes()==b;artifacts.append({'path':str(p),'bytes':len(b),'sha256':sha(p)})
def parquet(d,name,dst):
 src=L/name;assert not src.exists() and not dst.exists();pq.write_table(pa.Table.from_pandas(d,preserve_index=False),src,compression='zstd');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);assert sha(src)==sha(dst) and pq.ParquetFile(dst).metadata.num_rows==len(d);artifacts.append({'path':str(dst),'bytes':dst.stat().st_size,'rows':len(d),'sha256':sha(dst),'runtime_source':str(src)})
for role in roles:
 e=next(s for s in M['sources'] if s['role']==role);p=Path(e['path']);pre=next(s for s in J if s['path']==str(p));assert p.stat().st_size==pre['bytes'] and p.stat().st_mtime_ns==pre['mtime_ns'];pf=pq.ParquetFile(p,pre_buffer=False);assert pf.metadata.num_rows<100000
 # Exact identity whitelist: taxon strings containing e.g. 'kit' are never selected.
 identity=K+['array_index','collection_date','collection_timestamp','sample_name','sample_id','uuid','raw_read_count','trimmed_read_count','duplicate_reads_count','excessive_duplicate_reads','run_name','wgs_dna_code','metaphlan4_results_txt','metaphlan_abundance_species_parquet']
 cols=[c for c in identity if c in pf.schema_arrow.names];data=[r for b in pf.iter_batches(columns=cols,batch_size=1024,use_threads=False) for r in b.to_pylist()];assert len(data)==pf.metadata.num_rows;rows[role]=data;sources[role]={'path':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'rows':len(data),'sha256':sha(p),'physical_fields':len(pf.schema_arrow.names),'projected_fields':cols};print('SOURCE_IDENTITY_SCANNED',role,len(data),flush=True)
# Descriptive lags use nearest actually observed day, never just an interval's apparent span.
observed={}
for label,p,filtercol in [('food',P/'analysis/shared/food_daily_TRE_ONLY.parquet',None),('CGM',P/'analysis/shared/cgm_daily_TRE_ONLY.parquet','cgm_day_eligible')]:
 pf=pq.ParquetFile(p);d=pf.read(columns=K+['date']+([filtercol] if filtercol else []),use_threads=False).to_pylist();m=defaultdict(set)
 for r in d:
  k=key(r)
  if k and (filtercol is None or r[filtercol] is True):m[k].add(date.fromisoformat(r['date']).toordinal())
 observed[label]={k:sorted(v) for k,v in m.items()}
def lag(k,s,label):
 days=observed[label].get(k)
 if not days or s is None:return None
 v=date.fromisoformat(s).toordinal();i=bisect.bisect_left(days,v);choices=days[max(0,i-1):min(len(days),i+1)];near=min(choices,key=lambda t:(abs(v-t),t));return int(v-near)
# Index maps keep multiplicity as lists; no Cartesian merge.
indexes={};referenced={};name_counts={}
for site in ['oral','gut']:
 role=site+'_index';byK=defaultdict(list);byPD=defaultdict(list);pid=defaultdict(list);names=Counter()
 for i,r in enumerate(rows[role],1):
  k=key(r)
  if k:byK[k].append(i);pid[k[0]].append(i)
  if good(r.get('participant_id')) and day(r.get('collection_date')):byPD[(str(r['participant_id']),day(r['collection_date']))].append(i)
  if good(r.get('sample_name')):names[str(r['sample_name'])]+=1
 indexes[site]={'K':byK,'PD':byPD,'PID':pid};referenced[site]=set();name_counts[site]=names
abundance_byK={site:Counter(key(r) for r in rows[site+'_species'] if key(r)) for site in ['oral','gut']};candidates=[];reasoncounts=Counter();bridgecheck=[]
for site in ['oral','gut']:
 ar=site+'_species';ir=site+'_index'
 for ordinal,r in enumerate(rows[ar],1):
  rawK=key(r);k=rawK;ad=day(r.get('collection_date'));possible=[];proposed=[];linked=None;status='';evidence='';identity_ready=False
  if site=='oral':
   pool=indexes[site]['PD'].get((str(r.get('participant_id')),ad),[]) if ad else []
   possible=[i for i in pool if key(rows[ir][i-1]) and all(not good(r.get(c)) or str(r[c])==str(rows[ir][i-1][c]) for c in ['cohort','research_stage'])]
   if len(possible)==1:
    linked=possible[0];k=key(rows[ir][linked-1]);status='resolved_exact_person_calendar_unique_primary_index';evidence='exact source person/calendar evidence restores missing visit; not a lag inclusion filter';identity_ready=True
   else:status='held_oral_no_exact_visit_evidence' if not possible else 'held_oral_ambiguous_index_evidence'
   bridgecheck.append({'abundance_row_ordinal':ordinal,'index_row_ordinal':linked,'resolved_key':k})
  elif k:
   possible=indexes[site]['K'].get(k,[])
   if len(possible)==1 and abundance_byK[site][k]==1:linked=possible[0];status='resolved_unique_person_visit_aggregate_and_index';evidence='unique same-person/cohort/stage aggregate profile and index; no exact date required';identity_ready=True
   elif not possible:
    other=indexes[site]['PID'].get(k[0],[]);samecohort=[i for i in other if str(rows[ir][i-1].get('cohort'))==k[1]];status='held_gut_index_same_person_cohort_other_stage_only' if samecohort else 'held_gut_index_same_person_other_cohort_only' if other else 'held_gut_person_absent_from_primary_index'
   else:
    # Equal array_index is recorded as a proposal only; not a validated biological-ID join.
    proposed=[i for i in possible if good(r.get('array_index')) and r.get('array_index')==rows[ir][i-1].get('array_index')];status='held_repeated_visit_specimen_mapping_required';evidence='array equality is diagnostic until shared specimen namespace is verified; no arbitrary pair or random many-to-many join'
  else:status='held_invalid_source_person_cohort_stage'
  referenced[site].update(possible);ix=rows[ir][linked-1] if linked else None;sample=ix.get('sample_name') if ix else None;collision=bool(good(sample) and name_counts[site][str(sample)]>1)
  if collision:identity_ready=False;status='held_source_sample_name_repeated_identity_requires_review'
  cd=day(ix.get('collection_date')) if ix else ad if site=='oral' else None;cid=hashlib.sha256((sources[ar]['sha256']+'|'+str(ordinal)+'|'+site).encode()).hexdigest();priority=hashlib.sha256((str(SEED)+'|'+site+'|'+cid).encode()).hexdigest();tech={c:ix.get(c) for c in ['raw_read_count','trimmed_read_count','duplicate_reads_count','excessive_duplicate_reads','run_name','wgs_dna_code','metaphlan4_results_txt','metaphlan_abundance_species_parquet'] if ix is not None and c in ix}
  z={'site':site,'candidate_id':cid,'candidate_kind':'abundance_row','participant_id':k[0] if k else None,'cohort':k[1] if k else None,'research_stage':k[2] if k else None,'source_participant_id':str(r.get('participant_id')),'source_cohort':str(r.get('cohort')),'source_research_stage':str(r.get('research_stage')),'abundance_row_ordinal':ordinal,'source_array_index':r.get('array_index'),'abundance_source_path':sources[ar]['path'],'abundance_source_sha256':sources[ar]['sha256'],'primary_index_source_path':sources[ir]['path'],'primary_index_source_sha256':sources[ir]['sha256'],'linked_index_row_ordinal':linked,'candidate_index_ordinals_json':json.dumps(possible),'proposed_equal_array_index_ordinals_json':json.dumps(proposed),'index_sample_name':str(sample) if good(sample) else None,'sample_id_kind':'source_sample_name_biological_vs_library_not_proven' if good(sample) else 'versioned_aggregate_row_locator_not_biological_sample_ID','identity_status':status,'identity_evidence':evidence,'identity_ready_for_W13':identity_ready,'abundance_calendar_date':ad,'linked_index_calendar_date':day(ix.get('collection_date')) if ix else None,'descriptive_sample_calendar_date':cd,'sample_date_missing':cd is None,'date_conflict_between_linked_sources':bool(ix and ad and day(ix.get('collection_date')) and ad!=day(ix.get('collection_date'))),'sample_minus_nearest_observed_food_day':lag(k,cd,'food') if k else None,'sample_minus_nearest_qualified_CGM_day':lag(k,cd,'CGM') if k else None,'technical_metadata_json':json.dumps(tech,default=str),'technical_QC_status':'not_evaluated_W13','selected_primary_sample':False,'random_priority_after_required_identity_and_technical_QC':priority};candidates.append(z);reasoncounts[(site,status)]+=1
 # Index rows without ANY candidate profile relation remain explicit, including invalid-stage rows.
 for i,ix in enumerate(rows[ir],1):
  if i in referenced[site]:continue
  k=key(ix);cd=day(ix.get('collection_date'));cid=hashlib.sha256((sources[ir]['sha256']+'|index_only|'+str(i)+'|'+site).encode()).hexdigest();candidates.append({'site':site,'candidate_id':cid,'candidate_kind':'index_only','participant_id':k[0] if k else None,'cohort':k[1] if k else None,'research_stage':k[2] if k else None,'source_participant_id':str(ix.get('participant_id')),'source_cohort':str(ix.get('cohort')),'source_research_stage':str(ix.get('research_stage')),'abundance_row_ordinal':None,'source_array_index':ix.get('array_index'),'abundance_source_path':sources[ar]['path'],'abundance_source_sha256':sources[ar]['sha256'],'primary_index_source_path':sources[ir]['path'],'primary_index_source_sha256':sources[ir]['sha256'],'linked_index_row_ordinal':i,'candidate_index_ordinals_json':json.dumps([i]),'proposed_equal_array_index_ordinals_json':'[]','index_sample_name':str(ix['sample_name']) if good(ix.get('sample_name')) else None,'sample_id_kind':'index_record_without_matched_abundance','identity_status':'held_index_without_abundance' if k else 'held_index_invalid_visit_without_abundance','identity_evidence':'all primary index rows represented; no abundance invented','identity_ready_for_W13':False,'abundance_calendar_date':None,'linked_index_calendar_date':cd,'descriptive_sample_calendar_date':cd,'sample_date_missing':cd is None,'date_conflict_between_linked_sources':False,'sample_minus_nearest_observed_food_day':lag(k,cd,'food') if k else None,'sample_minus_nearest_qualified_CGM_day':lag(k,cd,'CGM') if k else None,'technical_metadata_json':'{}','technical_QC_status':'not_evaluated_W13','selected_primary_sample':False,'random_priority_after_required_identity_and_technical_QC':hashlib.sha256((str(SEED)+'|'+site+'|'+cid).encode()).hexdigest()});reasoncounts[(site,candidates[-1]['identity_status'])]+=1
c=pd.DataFrame(candidates);c['participant_id']=pd.to_numeric(c.participant_id,errors='coerce').astype('Int64');c['abundance_row_ordinal']=pd.to_numeric(c.abundance_row_ordinal,errors='coerce').astype('Int64');c['linked_index_row_ordinal']=pd.to_numeric(c.linked_index_row_ordinal,errors='coerce').astype('Int64');assert c.candidate_id.is_unique and not c.selected_primary_sample.any()
# Verify every abundance row appears once, every primary index row is represented without expansion.
for site in ['oral','gut']:
 a=c.loc[c.site.eq(site)&c.candidate_kind.eq('abundance_row')];assert set(a.abundance_row_ordinal.astype(int))==set(range(1,len(rows[site+'_species'])+1));refs=set(i for v in c.loc[c.site.eq(site),'candidate_index_ordinals_json'] for i in json.loads(v));assert refs==set(range(1,len(rows[site+'_index'])+1))
# Independent historical oral bridge is a comparator, never a replacement for current source reconstruction.
old=P/'data_qc/w03_keys_20261026T134938Z/oral_visit_evidence_map_TRE_ONLY.parquet';oldrows=pq.ParquetFile(old).read(columns=['abundance_row_ordinal','index_row_ordinal'],use_threads=False).to_pylist();assert len(oldrows)==len(bridgecheck)
for a,b in zip(oldrows,bridgecheck):assert a['abundance_row_ordinal']==b['abundance_row_ordinal'] and a['index_row_ordinal']==b['index_row_ordinal']
# Alternate oral processing version is inventoried, not pooled or silently substituted.
primary={key(r):r for r in rows['oral_index'] if key(r)};alt={key(r):r for r in rows['oral_index_alternate'] if key(r)};same=set(primary)&set(alt);alternate={'primary_valid_keys':len(primary),'alternate_valid_keys':len(alt),'same_valid_keys':len(same),'same_key_collection_date_agreements':sum(day(primary[k].get('collection_date'))==day(alt[k].get('collection_date')) for k in same),'versions_pooled':False,'primary_role':'oral_index','raw_pointer_equivalence_not_assumed':True}
summary={'candidate_rows':len(c),'all_source_reason_counts':[{'site':site,'reason':reason,'rows':n} for (site,reason),n in sorted(reasoncounts.items())],'per_site':{},'oral_bridge_current_matches_prior_all_rows':len(oldrows),'selected_primary_samples':0,'W13_technical_QC_done':False,'individual_data_exported':False,'no_fixed_sampling_lag_gate':True}
for site in ['oral','gut']:
 s=c.loc[c.site.eq(site)];b=s.loc[s.cohort.eq('10k')&s.research_stage.eq('00_00_visit')];ready=s.loc[s.identity_ready_for_W13];rb=b.loc[b.identity_ready_for_W13];summary['per_site'][site]={'candidate_rows':len(s),'abundance_rows':int(s.candidate_kind.eq('abundance_row').sum()),'index_only_rows':int(s.candidate_kind.eq('index_only').sum()),'all_stage_identity_ready_rows':len(ready),'all_stage_identity_ready_person_visits':len(ready[K].drop_duplicates()),'known_baseline_abundance_people':int(b.loc[b.candidate_kind.eq('abundance_row'),'participant_id'].nunique()),'known_baseline_identity_ready_people':int(rb.participant_id.nunique()),'known_baseline_held_rows':int((~b.identity_ready_for_W13).sum()),'date_missing_all_candidates':int(s.sample_date_missing.sum()),'baseline_lag_to_food_computable_rows':int(rb.sample_minus_nearest_observed_food_day.notna().sum()),'baseline_lag_to_food_median_signed_days':float(rb.sample_minus_nearest_observed_food_day.median()) if rb.sample_minus_nearest_observed_food_day.notna().any() else None,'source_sample_name_duplicate_values':sum(n>1 for n in name_counts[site].values())};print('W12_SITE_SUMMARY',site,json.dumps(summary['per_site'][site]),flush=True)
o=set(c.loc[c.site.eq('oral')&c.identity_ready_for_W13&c.cohort.eq('10k')&c.research_stage.eq('00_00_visit'),'participant_id'].dropna().astype(int));g=set(c.loc[c.site.eq('gut')&c.identity_ready_for_W13&c.cohort.eq('10k')&c.research_stage.eq('00_00_visit'),'participant_id'].dropna().astype(int));summary['baseline_joint_identity_ready_people_not_final_QC']=len(o&g)
rules={'version':'W12_started20261028_finished20261029','source_versions_fixed':sources,'candidate_grain':'one row per versioned aggregate abundance row; additional index-only records retained; candidate locator is not claimed biological ID','join_key':K,'array_index_namespace':'equal values recorded as proposal only in repeated groups, not accepted specimen evidence without validation','user_override_of_original_quality_ranking':'QC first, then one eligible biological sample at random per same-person/cohort/stage/site; do not rank by phenotype, date distance or effect','random_seed':SEED,'random_implementation':'lexical minimum SHA256(str(seed)|site|candidate_id) among identity-verified, technically eligible, distinct biological candidates; stable stored priority; no seed search','technical_remeasurements_vs_biological_repeats':'unresolved repeats held for specimen identity evidence, not automatically interpreted as different biological samples; exact duplicate collapse requires W13 matrix evidence and retained lineage','technical_failure_handling':'W13 applies documented source failure/quality rules before random ordering; alternate eligible biological candidate may replace failed candidate before final lock only','stage_repair':'oral exact person and source calendar date plus unique valid primary index, respecting valid original cohort/stage; no baseline imputation from person alone','date_policy':'dates and signed nearest observed food/qualified CGM day gaps descriptive only; no 30/60/90 day, same-day or pre-sampling gate; missing date alone does not reject a known valid same-visit sample','unmatched_or_ambiguous_identity':'held in candidate ledger; W13 resolves provenance before eligibility; no many-to-many Cartesian join or arbitrary dedup','final_selection_executed':False,'technical_QC_executed':False,'next_task':'W13 verifies matrix scales, numerical QC, source technical metadata and uncertain specimen linkage, then locks one sample reused across all outcomes','no_outcome_values_or_effects_used':True}
parquet(c,'sample_candidates_TRE_ONLY.parquet',P/'analysis/shared/sample_candidates_TRE_ONLY.parquet');dump(P/'config/sample_selection_rules.yaml',rules);dump(Q/'w12_summary_TRE_ONLY.json',summary);dump(Q/'oral_alternate_version_diagnostic.json',alternate);dump(Q/'w12_source_manifest_TRE_ONLY.json',sources)
rr=pd.DataFrame([{'site':site,'identity_reason':reason,'candidate_rows':n} for (site,reason),n in sorted(reasoncounts.items())]);p=P/'data_qc/sample_identity_reasons.csv';assert not p.exists();rr.to_csv(p,index=False);artifacts.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)});dump(Q/'w12_artifact_manifest_TRE_ONLY.json',artifacts);print('W12_CANDIDATES_COMPLETE',len(c),'JOINT_BASELINE_IDENTITY_READY_NOT_FINAL',len(o&g),flush=True)
for r in summary['all_source_reason_counts']:print('W12_REASON',r['site'],r['reason'],r['rows'],flush=True)
