"""Predeclared technical QC only. No associations, selection changes, or raw export."""
import os
for n in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone,date
import json,hashlib,resource,uuid,bisect,sys
import numpy as np,pandas as pd,pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');Q=P/'data_qc/claude_readonly_followup_20261030_v1';K=['participant_id','cohort','research_stage'];PK=['cohort','participant_id'];M=json.loads((Q/'preflight.json').read_text());resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(600,600));(Q/'access_receipts').mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(n,x):
 p=Q/n;assert not p.exists(),str(p);p.write_text(json.dumps(x,indent=2,default=str,allow_nan=False)+'\n');return p

def readpath(p,cols=None,purpose='technical_QC',y=False):
 p=Path(p);f=pq.ParquetFile(p,pre_buffer=False);cols=list(f.schema_arrow.names) if cols is None else list(cols);assert set(cols)<=set(f.schema_arrow.names);assert f.metadata.num_rows<=2500000
 budget=sum(f.metadata.row_group(i).column(j).total_uncompressed_size for i in range(f.metadata.num_row_groups) for j in range(f.metadata.row_group(i).num_columns) if f.metadata.row_group(i).column(j).path_in_schema in cols);assert budget<512*1024**2,(p.name,budget)
 receipt={'time_UTC':datetime.now(timezone.utc).isoformat(),'script':str(Path(sys.argv[0])),'script_sha256':sha(sys.argv[0]),'path':str(p),'source_sha256':sha(p),'columns':cols,'source_rows':f.metadata.num_rows,'projected_uncompressed_bytes':budget,'purpose':purpose,'technical_Y_value_read':y,'scientific_association_or_training':False,'individual_data_exported':False};save('access_receipts/'+uuid.uuid4().hex+'.json',receipt)
 return pd.concat([b.to_pandas(ignore_metadata=True) for b in f.iter_batches(columns=cols,batch_size=4096,use_threads=False)],ignore_index=True)
def read(role,cols=None,purpose='technical_QC',y=False):
 m=M['inputs'][role];p=Path(m['path']);assert sha(p)==m['sha256'];return readpath(p,cols,purpose,y)
def keys(d,cols=K):return set(d[cols].itertuples(index=False,name=None))
def dist(s):return {str(k):int(v) for k,v in s.value_counts(dropna=False).sort_index().items()}
def equal(a,b):return (pd.isna(a) and pd.isna(b)) or (not pd.isna(a) and not pd.isna(b) and a==b)
def liver():
 d=read('raw_liver',purpose='X02 exact row/index/date duplication and independent six means',y=True);assert len(d)<200000 and len(d.columns)<128
 norm=d.copy()
 for c in norm:
  if norm[c].dtype=='object':norm[c]=norm[c].map(lambda v:json.dumps(v.tolist() if isinstance(v,np.ndarray) else v,sort_keys=True,default=str) if isinstance(v,(np.ndarray,list,dict,tuple)) else v)
 out={'source_rows':len(d),'source_columns':len(d.columns),'exact_full_row_duplicate_excess':int(norm.duplicated().sum()),'exact_full_row_duplicate_members':int(norm.duplicated(keep=False).sum()),'duplicate_K_array_index_excess':int(d.duplicated(K+['array_index']).sum()),'person_visit_measurement_count_distribution':dist(d.groupby(K).size())}
 for col in ['collection_date','collection_timestamp']:
  if col in d:out[col+'_distinct_per_visit_distribution']=dist(d.groupby(K)[col].nunique(dropna=True))
 specs=[s for s in json.loads((P/'config/outcome_dictionary.yaml').read_text())['static_outcomes'] if s.get('old_aggregate')];o=read('outcomes',K+['outcome_id','value','n_valid_measurements'],purpose='X02 verify stored liver arithmetic mean',y=True);rr=[]
 for s in specs:
  v=pd.to_numeric(d[s['field']],errors='coerce');t=d[K].copy();t['v']=v.where(np.isfinite(v)&v.gt(0));g=t.groupby(K).v.agg(['mean','count']).reset_index();z=g.merge(o.loc[o.outcome_id.eq(s['id'])],on=K,how='outer',indicator=True,validate='one_to_one');both=z._merge.eq('both');missing=z['mean'].isna()!=z.value.isna();err=(z['mean']-z.value).abs();rr.append({'outcome':s['id'],'keys':len(g),'keyset_difference':int((~both).sum()),'mean_mismatch_gt_1e_minus_9':int((missing|err.gt(1e-9)).sum()),'valid_count_mismatch':int(z['count'].ne(z.n_valid_measurements).sum()),'measurement_counts':dist(g['count'])})
 out['independent_mean_check']=rr;old=json.loads((P/'data_qc/w10_outcome_qc_20261028/liver_aggregate_diagnostic.json').read_text());out['official_summary_comparison']=[{k:r[k] for k in ['outcome_id','baseline_comparable','baseline_mean_difference_gt_1e_minus_6','mean_mismatch_by_valid_measurement_count']} for r in old['diagnostics']];out['rules_changed']=False;save('X02_liver.json',out);print('X02_LIVER',json.dumps(out),flush=True)
def microbiome():
 F=read('sample_flags',purpose='X03 X07 X12 full sample lineage technical flags');N=read('profile_numeric_QC');a=F.loc[F.candidate_kind.eq('abundance_row')].copy();assert not a.duplicated(['site','abundance_row_ordinal']).any();assert keys(a,['site','abundance_row_ordinal'])==keys(N,['site','abundance_row_ordinal']);out={};details=[]
 for site in ['oral','gut']:
  z=a.loc[a.site.eq(site)];groups=Counter();members=Counter();selected=Counter();selgroups=Counter()
  for h,g in z.groupby('profile_sha256_float64'):
   if len(g)<2:continue
   typ='unknown_identity' if g[K].isna().any().any() else 'cross_person' if len(g[PK].drop_duplicates())>1 else 'same_person_cross_visit' if len(g[K].drop_duplicates())>1 else 'same_person_same_visit'
   groups[typ]+=1;members[typ]+=len(g);selected[typ]+=int(g.selected_primary_sample.sum());selgroups[typ]+=int(g.selected_primary_sample.sum()>1)
  out[site]={'identical_groups':dict(groups),'member_rows':dict(members),'selected_member_rows':dict(selected),'groups_with_multiple_selected_members':dict(selgroups),'hash_definition':'raw float64 profile including source-null pattern; source hashes fixed; equality alone does not prove specimen duplication'}
  idx=read(site+'_index_projection');srcmeta=json.loads((P/'data_qc/w13_microbiome_qc_20261029/source_metadata_TRE_ONLY.json').read_text());sourcecols=set(pq.ParquetFile(srcmeta[site+'_index']['path']).schema_arrow.names);s=z.loc[z.selected_primary_sample]
  depth={}
  for field in ['raw_read_count','trimmed_read_count']:
   vals=s.technical_metadata_json.map(lambda x:json.loads(x).get(field));nums=pd.to_numeric(vals,errors='coerce');depth[field]={'field_in_raw_source':field in sourcecols,'selected_rows':len(s),'finite_positive':int((np.isfinite(nums)&nums.gt(0)).sum()),'finite_nonpositive':int((np.isfinite(nums)&nums.le(0)).sum()),'missing_or_nonfinite':int((~np.isfinite(nums)).sum()),'metadata_key_absent':int(s.technical_metadata_json.map(lambda x:field not in json.loads(x)).sum())}
  out[site]['depth_metadata']=depth;out[site]['selected_QC_reasons']=s.available_QC_reason.value_counts().astype(int).to_dict()
 rec=F.loc[F.identity_status.eq('resolved_unique_native_profile_same_visit_distinct_dated_biological_repeats')].copy();lagout={'recovered_rows':len(rec),'selected_recovered_rows':int(rec.selected_primary_sample.sum()),'admission_or_choice_uses_lags':False};maps={}
 for label,role,flag in [('food','food_daily',None),('CGM','cgm_daily_selected_legacy','cgm_day_eligible')]:
  t=read(role,K+['date']+([flag] if flag else []));t=t if flag is None else t.loc[t[flag].eq(True)];t['ordinal']=pd.to_datetime(t.date).map(lambda x:x.date().toordinal());maps[label]={k:sorted(set(g.ordinal)) for k,g in t.groupby(K)}
  col='sample_minus_nearest_observed_food_day' if label=='food' else 'sample_minus_nearest_qualified_CGM_day';computed=[]
  for r in rec.to_dict('records'):
   ds=maps[label].get(tuple(r[k] for k in K));sd=r['descriptive_sample_calendar_date'];v=None
   if ds and not pd.isna(sd):
    n=pd.Timestamp(sd).date().toordinal();i=bisect.bisect_left(ds,n);near=min(ds[max(0,i-1):min(len(ds),i+1)],key=lambda x:(abs(n-x),x));v=n-near
   computed.append(v)
  lagout[label]={'computable_after_recovery':sum(v is not None for v in computed),'published_missing':int(rec[col].isna().sum()),'stale_or_mismatched_rows':sum(not equal(x,y) for x,y in zip(rec[col],computed)),'selected_stale_rows':sum(bool(s) and not equal(x,y) for x,y,s in zip(rec[col],computed,rec.selected_primary_sample))}
  for r,v in zip(rec.to_dict('records'),computed):details.append({'candidate_id':r['candidate_id'],'label':label,'published_lag':r[col],'recomputed_lag':v})
 save('X03_X07_microbiome.json',out);save('X12_recovered_lags.json',lagout);save('X12_lag_details_TRE_ONLY.json',details);print('X03_X07_MICROBIOME',json.dumps(out),flush=True);print('X12_LAGS',json.dumps(lagout),flush=True)
def family():
 e=read('relatedness_edges');cv=read('covariates',K);pop=read('population_source',['cohort','participant_id']);s=read('W15_order');known=read('W15_related_membership');universe=keys(cv,PK)|keys(pop,PK);pids=defaultdict(set)
 for c,p in universe:pids[p].add(c)
 same=other=nowhere=0;nodes=set();candidate_nodes=keys(s,PK);missing_nodes=set();crosscandidate=set()
 for r in e.itertuples(index=False):
  node=(r.cohort,r.related_participant_id);nodes.add(node)
  if node in universe:same+=1
  else:
   missing_nodes.add(node)
   if r.related_participant_id in pids:
    other+=1
    if (r.cohort,r.participant_id) in candidate_nodes:crosscandidate.add((r.cohort,r.participant_id))
   else:nowhere+=1
 out={'edges':len(e),'source_cohort_edge_counts':e.cohort.value_counts().astype(int).to_dict(),'distinct_target_nodes':len(nodes),'target_same_cohort_known_edges':same,'target_only_other_cohort_edges':other,'target_not_in_covariate_or_population_universe_edges':nowhere,'distinct_unmatched_target_nodes':len(missing_nodes),'candidate_source_people_with_other_cohort_target_only':len(crosscandidate),'pid_values_seen_in_multiple_cohorts':sum(len(v)>1 for v in pids.values()),'candidate_people':len(s),'known_membership_rows':len(known),'cross_cohort_identity_assumed':False,'graph_or_split_changed':False};save('X04_family.json',out);print('X04_FAMILY',json.dumps(out),flush=True)
if __name__=='__main__':
 errors=[]
 for name,fn in [('liver',liver),('microbiome',microbiome),('family',family)]:
  try:fn()
  except Exception as ex:
   errors.append({'section':name,'error_type':type(ex).__name__,'message':str(ex)});print('SECTION_ERROR',name,type(ex).__name__,str(ex),flush=True)
 save('03_execution_summary.json',{'errors':errors,'scientific_associations':False,'original_artifacts_mutated':False});print('SOURCE_QC_FINISHED','errors',len(errors),flush=True);sys.exit(bool(errors))
