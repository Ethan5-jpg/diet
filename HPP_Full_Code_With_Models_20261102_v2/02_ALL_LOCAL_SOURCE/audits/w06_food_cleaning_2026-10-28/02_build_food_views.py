# -*- coding: utf-8 -*-
"""Outcome-blind W06: source-preserving event/day views, signature audit, no dedup."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,hashlib,sqlite3,time,math,resource,shutil,platform
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
assert sys.executable=='/home/ec2-user/miniconda/envs/pheno/bin/python'
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
R=Path('/home/ec2-user/studies/hpp_datasets/diet_logging')
L=Path('/home/ec2-user/multisystem_runtime_20261021/w06_food_cleaning_20261028')
Q=P/'data_qc/w06_food_cleaning_20261028'; B=L/'02_artifacts';B.mkdir(parents=True,exist_ok=False)
for rel in ['analysis/shared/food_events_clean_TRE_ONLY.parquet','analysis/shared/food_event_lineage_TRE_ONLY.parquet','analysis/shared/food_day_qc_TRE_ONLY.parquet','config/food_day_rules.yaml','data_qc/food_exclusions.csv']:
 assert not (P/rel).exists(),'Existing destination requires review: '+rel
start=time.monotonic(); K=['participant_id','cohort','research_stage','date']
N=['calories_kcal','carbohydrate_g','lipid_g','protein_g','sodium_mg','alcohol_g','dietary_fiber_g']
def guard():
 if time.monotonic()-start>1500:raise RuntimeError('W06 time budget exhausted')
 if shutil.disk_usage(L).free<4*1024**3:raise RuntimeError('W06 disk budget')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(name,cols=None):
 return pq.ParquetFile(R/name,pre_buffer=False).iter_batches(batch_size=8192,columns=cols,use_threads=False)
def dates(d):return pd.to_datetime(d,errors='coerce').dt.strftime('%Y-%m-%d').fillna('')
def canonical(v):
 if v is None or pd.isna(v):return None
 if isinstance(v,(pd.Timestamp,np.datetime64)):return pd.Timestamp(v).isoformat()
 if isinstance(v,(np.integer,int)):return int(v)
 if isinstance(v,(np.floating,float)):return float(v) if math.isfinite(v) else str(v)
 if isinstance(v,(np.bool_,bool)):return bool(v)
 return str(v)
def sigs(d,cols):
 return [hashlib.sha256(json.dumps([canonical(v) for v in row],separators=(',',':'),ensure_ascii=False).encode()).digest() for row in d[cols].itertuples(index=False,name=None)]
def validkey(k):return k[0] is not None and k[1] not in ('','undefined') and k[2] not in ('','undefined') and k[3]!=''
def keys(d):
 return [(int(a) if pd.notna(a) else None,str(b) if pd.notna(b) else '',str(c) if pd.notna(c) else '',s) for a,b,c,s in d[K].itertuples(index=False,name=None)]
def newday():return {'n':0,'idx_n':0,'idx_food_min':None,'idx_food_max':None,'bad':[0]*len(N),'sum':[0.]*len(N),'local_conflict':0,'aware_conflict':0,'missing_local':0,'missing_aware':0,'tmin':None,'tmax':None,'exact_flagged':0,'candidate_flagged':0}
inputs={}
for p in sorted(R.glob('*.parquet')):
 f=pq.ParquetFile(p,pre_buffer=False);s=p.stat();inputs[p.name]={'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'rows':f.metadata.num_rows,'schema':str(f.schema_arrow)}
assert inputs['diet_logging_events.parquet']['rows']==2640959
assert inputs['raw_diet_logging_events.parquet']['rows']==6485022
assert inputs['diet_logging.parquet']['rows']==149912
source_contract=json.loads((Q/'source_contract_TRE_ONLY.json').read_text())
assert len(source_contract['sources'])==3
allcols=pq.ParquetFile(R/'diet_logging_events.parquet').schema_arrow.names
rawcols=pq.ParquetFile(R/'raw_diet_logging_events.parquet').schema_arrow.names
lifecycle=[c for c in sorted(set(allcols+rawcols)) if any(x in c.lower() for x in ['delete','edit','updated','created','event_id','backfill'])]
shared=[c for c in allcols if c in rawcols and c not in ['research_stage','logging_day']]
candidate=['participant_id','cohort','research_stage','collection_date','collection_timestamp','food_id']
assert all(c in allcols for c in candidate+N)
D={};index_rows=0
for rb in load('diet_logging.parquet', ['participant_id','cohort','research_stage','collection_date','food_items']):
 d=rb.to_pandas(ignore_metadata=True);d['date']=dates(d.collection_date)
 for k,food in zip(keys(d),d.food_items):
  z=D.setdefault(k,newday());z['idx_n']+=1
  if pd.notna(food):
   n=float(food);z['idx_food_min']=n if z['idx_food_min'] is None else min(n,z['idx_food_min']);z['idx_food_max']=n if z['idx_food_max'] is None else max(n,z['idx_food_max'])
 index_rows+=len(d)
assert index_rows==149912
print('INDEX_LOADED',index_rows,'KEYS',len(D),flush=True)
con=sqlite3.connect(L/'signatures_TRE_ONLY.sqlite');con.execute('pragma journal_mode=off');con.execute('pragma synchronous=off');con.execute('pragma cache_size=-65536')
for t in ['exact','candidate']:con.execute('create table '+t+' (s blob primary key,n integer not null) without rowid')
con.execute('create table shared (s blob primary key,n integer not null,raw_n integer not null) without rowid')
con.execute('create table event (i integer primary key,e blob,c blob,s blob)')
# Independent synthetic checks: identical records are flagged, never collapsed; timezone calendar boundaries.
a=pd.DataFrame({'x':[1,1,2],'y':[np.nan,np.nan,0.]})
assert sigs(a,['x','y'])[0]==sigs(a,['x','y'])[1] and sigs(a,['x','y'])[0]!=sigs(a,['x','y'])[2]
t=pd.Series(pd.to_datetime(['2026-01-01T22:30:00Z'],utc=True)).dt.tz_convert('Asia/Jerusalem')
assert t.dt.strftime('%Y-%m-%d').iloc[0]=='2026-01-02' and t.dt.tz_convert('UTC').dt.strftime('%Y-%m-%d').iloc[0]=='2026-01-01'
assert pd.Series([1.,np.nan]).sum(min_count=2)!=1.
assert not validkey((1,'10k','undefined','2026-01-01'))
clock=Counter();delta=Counter();timezones=Counter();stages=Counter();offsets=Counter();n=0
for rb in load('diet_logging_events.parquet'):
 d=rb.to_pandas(ignore_metadata=True);m=len(d);e=sigs(d,allcols);c=sigs(d,candidate);s=sigs(d,shared)
 for table,values in [('exact',e),('candidate',c)]:
  cnt=Counter(values);con.executemany('insert into '+table+' values (?,?) on conflict(s) do update set n=n+excluded.n',cnt.items())
 cnt=Counter(s);con.executemany('insert into shared values (?,?,0) on conflict(s) do update set n=n+excluded.n',cnt.items())
 con.executemany('insert into event values (?,?,?,?)',((n+i,e[i],c[i],s[i]) for i in range(m)));con.commit()
 d['date']=dates(d.collection_date);kk=keys(d)
 aw=pd.to_datetime(d.collection_timestamp,errors='coerce');assert isinstance(aw.dtype,pd.DatetimeTZDtype)
 wall=aw.dt.tz_convert('Asia/Jerusalem').dt.tz_localize(None);utc=aw.dt.tz_convert('UTC').dt.tz_localize(None)
 loc=pd.to_datetime(d.local_timestamp,errors='coerce');assert not isinstance(loc.dtype,pd.DatetimeTZDtype)
 wc=dates(wall);lc=dates(loc);both=aw.notna()&loc.notna();lconf=loc.notna()&(lc!=d.date);aconf=aw.notna()&(wc!=d.date)
 flags={'processed_rows':np.ones(m,bool),'missing_local':loc.isna(),'missing_aware':aw.isna(),'date_vs_local_conflict':lconf,'date_vs_aware_conflict':aconf,'local_equals_aware_wall':both&(loc==wall),'local_equals_UTC_naive':both&(loc==utc),'local_other_than_both':both&(loc!=wall)&(loc!=utc),'date_conflict_UTC_equal':lconf&both&(loc==utc),'date_conflict_wall_equal':lconf&both&(loc==wall)}
 baseline=(d.cohort=='10k')&(d.research_stage=='00_00_visit')
 for label,mask in flags.items():clock['all__'+label]+=int(np.asarray(mask).sum());clock['baseline__'+label]+=int((mask&baseline).sum())
 for v,z in Counter(((loc-wall).dt.total_seconds()[lconf&both]).dropna().astype(float)).items():delta[v]+=z
 for v,z in Counter(((loc.dt.normalize()-wall.dt.normalize()).dt.days[lconf&both]).dropna().astype(int)).items():offsets[v]+=z
 timezones.update(d.timezone.astype(str));stages.update(d.research_stage.astype(str))
 vals=np.column_stack([pd.to_numeric(d[x],errors='coerce').to_numpy(dtype=float) for x in N]);bad=~np.isfinite(vals)|(vals<0)
 times=aw.dt.tz_convert('UTC').dt.tz_localize(None).astype('datetime64[ns]').astype('int64').to_numpy();present=aw.notna().to_numpy()
 for i,k in enumerate(kk):
  z=D.setdefault(k,newday());z['n']+=1
  for j in range(len(N)):
   z['bad'][j]+=int(bad[i,j]);z['sum'][j]+=float(vals[i,j]) if not bad[i,j] else 0.
  z['local_conflict']+=int(lconf.iloc[i]);z['aware_conflict']+=int(aconf.iloc[i]);z['missing_local']+=int(loc.isna().iloc[i]);z['missing_aware']+=int(aw.isna().iloc[i])
  if present[i]:
   tt=int(times[i]);z['tmin']=tt if z['tmin'] is None else min(tt,z['tmin']);z['tmax']=tt if z['tmax'] is None else max(tt,z['tmax'])
 n+=m
 if n//250000!=(n-m)//250000:print('PROCESSED_PASS1',n,'DAY_KEYS',len(D),flush=True)
 guard()
assert n==inputs['diet_logging_events.parquet']['rows']
raw_n=0;raw_stages=Counter()
for rb in load('raw_diet_logging_events.parquet',shared+['research_stage']):
 d=rb.to_pandas(ignore_metadata=True);counts=Counter(sigs(d,shared));con.executemany('update shared set raw_n=raw_n+? where s=?',((v,k) for k,v in counts.items()));con.commit();raw_stages.update(d.research_stage.astype(str));raw_n+=len(d)
 if raw_n//1000000!=(raw_n-len(d))//1000000:print('RAW_SIGNATURE_AUDIT',raw_n,flush=True)
 guard()
assert raw_n==6485022
sigsummary={}
for t in ['exact','candidate']:
 v=con.execute('select count(*),coalesce(sum(n),0),coalesce(sum(case when n>1 then 1 else 0 end),0),coalesce(sum(case when n>1 then n else 0 end),0),coalesce(sum(n-1),0) from '+t).fetchone()
 sigsummary[t]=dict(zip(['groups','rows','repeated_groups','rows_in_repeated_groups','excess_rows_if_collapsed_NOT_applied'],v))
sigsummary['raw_shared_match']=dict(zip(['unique_processed_signatures','processed_rows_no_raw_match','processed_rows_raw_match','processed_rows_raw_multimatch'],con.execute('select count(*),coalesce(sum(case when raw_n=0 then n else 0 end),0),coalesce(sum(case when raw_n>0 then n else 0 end),0),coalesce(sum(case when raw_n>1 then n else 0 end),0) from shared').fetchone()))
print('SIGNATURE_AUDIT',json.dumps(sigsummary),flush=True)
# Ordered signatures provide flags, never an expanding join of the raw and processed event sources.
cur=con.execute('select v.i,e.n,c.n,s.raw_n from event v join exact e on e.s=v.e join candidate c on c.s=v.c join shared s on s.s=v.s order by v.i')
cleanpath=B/'food_events_clean_TRE_ONLY.parquet';linepath=B/'food_event_lineage_TRE_ONLY.parquet';cw=None;lw=None;n=0;kept=0;reason_counts=Counter();clean_energy=0.;clean_bad_energy=0
for rb in load('diet_logging_events.parquet'):
 d=rb.to_pandas(ignore_metadata=True);m=len(d);d['date']=dates(d.collection_date);kk=keys(d);flags=cur.fetchmany(m);assert len(flags)==m
 assert [x[0] for x in flags]==list(range(n,n+m))
 ex=np.array([x[1] for x in flags],dtype=np.int64);ca=np.array([x[2] for x in flags],dtype=np.int64);rawmatch=np.array([x[3] for x in flags],dtype=np.int64)
 reason=np.array(['retained' if validkey(k) and D[k]['idx_n']>0 else ('invalid_identity_visit_or_date' if not validkey(k) else 'event_day_without_index') for k in kk]);keep=reason=='retained';reason_counts.update(reason)
 d['source_row_ordinal']=np.arange(n,n+m,dtype=np.int64);d['qc_exact_signature_n']=ex;d['qc_same_food_time_n']=ca;d['qc_raw_shared_signature_n']=rawmatch;d['qc_index_matched']=[D[k]['idx_n']>0 for k in kk];d['qc_primary_eligible']=keep;d['qc_row_disposition']=reason
 aw=pd.to_datetime(d.collection_timestamp,errors='coerce');wall=aw.dt.tz_convert('Asia/Jerusalem').dt.tz_localize(None);loc=pd.to_datetime(d.local_timestamp,errors='coerce')
 d['qc_date_local_conflict']=loc.notna()&(dates(loc)!=d.date);d['qc_date_aware_conflict']=aw.notna()&(dates(wall)!=d.date)
 for x in N:
  v=pd.to_numeric(d[x],errors='coerce').astype(float);ok=np.isfinite(v)&(v>=0);d[x+'_clean']=v.where(ok,np.nan)
 for i,k in enumerate(kk):D[k]['exact_flagged']+=int(ex[i]>1);D[k]['candidate_flagged']+=int(ca[i]>1)
 lineage=d[['source_row_ordinal']+K+['qc_primary_eligible','qc_row_disposition','qc_exact_signature_n','qc_same_food_time_n','qc_raw_shared_signature_n','qc_date_local_conflict','qc_date_aware_conflict']]
 lt=pa.Table.from_pandas(lineage,preserve_index=False)
 if lw is None:lw=pq.ParquetWriter(linepath,lt.schema,compression='zstd')
 lw.write_table(lt)
 ct=pa.Table.from_pandas(d.loc[keep],preserve_index=False)
 if cw is None:cw=pq.ParquetWriter(cleanpath,ct.schema,compression='zstd')
 cw.write_table(ct)
 clean_energy+=float(d.loc[keep,'calories_kcal_clean'].sum());clean_bad_energy+=int(d.loc[keep,'calories_kcal_clean'].isna().sum());kept+=int(keep.sum());n+=m;guard()
cw.close();lw.close();assert cur.fetchone() is None;assert n==2640959;assert sum(reason_counts.values())==n
out=[]
for k,z in D.items():
 valid=validkey(k);has=z['n']>0;indexed=z['idx_n']>0;primary=valid and has and indexed
 a=dict(zip(K,k));a.update(n_events=z['n'],n_index_rows=z['idx_n'],index_matched=indexed,identity_visit_date_valid=valid,has_events=has,primary_day=primary,n_exact_duplicate_flagged=z['exact_flagged'],n_same_food_time_flagged=z['candidate_flagged'],date_local_conflict_rows=z['local_conflict'],date_aware_conflict_rows=z['aware_conflict'],missing_local_rows=z['missing_local'],missing_aware_rows=z['missing_aware'],recorded_span_hours=(z['tmax']-z['tmin'])/3.6e12 if z['tmin'] is not None else np.nan,index_food_items_min=z['idx_food_min'],index_food_items_max=z['idx_food_max'])
 a['index_food_count_consistent']=bool(indexed and has and z['idx_food_min']==z['idx_food_max']==z['n'])
 for j,x in enumerate(N):
  a[x+'_valid_partial_sum']=z['sum'][j] if has else np.nan;a[x+'_invalid_rows']=z['bad'][j];a[x+'_complete_sum']=z['sum'][j] if has and z['bad'][j]==0 else np.nan
 a['energy_eligible']=bool(primary and z['bad'][0]==0 and z['sum'][0]>0)
 a['fiber_eligible']=bool(a['energy_eligible'] and z['bad'][N.index('dietary_fiber_g')]==0)
 a['clock_calendar_consistent']=bool(primary and z['aware_conflict']==0 and z['missing_aware']==0)
 a['meal_ingestion_time_validated']=False;a['complete_daily_intake_established']=False
 a['day_disposition']='retained_indexed_event_day' if primary else ('index_without_events' if indexed and not has else ('invalid_identity_visit_or_date' if not valid else 'event_day_without_index'))
 out.append(a)
day=pd.DataFrame(out);daypath=B/'food_day_qc_TRE_ONLY.parquet';pq.write_table(pa.Table.from_pandas(day,preserve_index=False),daypath,compression='zstd')
assert int(day.loc[day.primary_day,'n_events'].sum())==kept
assert math.isclose(float(day.loc[day.primary_day,'calories_kcal_valid_partial_sum'].sum()),clean_energy,rel_tol=1e-10,abs_tol=1e-6)
assert int(day.loc[day.primary_day,'calories_kcal_invalid_rows'].sum())==clean_bad_energy
assert not day.duplicated(K).any()
summary={}
for scope,mask in [('all_source',pd.Series(True,index=day.index)),('baseline_10k',(day.cohort=='10k')&(day.research_stage=='00_00_visit'))]:
 z=day.loc[mask];v={}
 for label,ok in [('event_days',z.has_events),('indexed_event_days',z.primary_day),('energy_eligible',z.energy_eligible),('fiber_eligible',z.fiber_eligible),('calendar_consistent',z.clock_calendar_consistent)]:
  g=z.loc[ok].groupby(['participant_id','cohort','research_stage'],dropna=False).size();v[label]={'days':int(ok.sum()),'participant_visits':len(g),'participants':int(z.loc[ok,'participant_id'].nunique()),'ge7_days_DIAGNOSTIC_ONLY':int((g>=7).sum())}
 v['index_only_days']=int((z.index_matched&~z.has_events).sum());v['event_days_without_index']=int((z.has_events&~z.index_matched).sum());v['index_food_count_mismatch_days']=int((z.primary_day&~z.index_food_count_consistent).sum());summary[scope]=v
print('DAY_FLOW',json.dumps(summary),flush=True);print('CLOCK_COUNTS',json.dumps(dict(clock)),flush=True)
print('CLOCK_CONFLICT_DELTAS_SECONDS_TOP',json.dumps(delta.most_common(15)),'DAY_OFFSETS',json.dumps(offsets.most_common()),flush=True)
# Independent readback of new output, not only in-memory assertions.
readn=0;reads=0.;readbad=0
for rb in pq.ParquetFile(cleanpath).iter_batches(batch_size=8192,columns=['qc_primary_eligible','calories_kcal_clean']):
 d=rb.to_pandas();assert d.qc_primary_eligible.all();readn+=len(d);reads+=float(d.calories_kcal_clean.sum());readbad+=int(d.calories_kcal_clean.isna().sum())
assert readn==kept and math.isclose(reads,clean_energy,rel_tol=1e-10,abs_tol=1e-6) and readbad==clean_bad_energy
assert pq.ParquetFile(linepath).metadata.num_rows==2640959
assert pq.ParquetFile(daypath).metadata.num_rows==len(day)
rules={'version':'W06_2026-10-28','keys':K,'source_event_table':'diet_logging_events.parquet','raw_intake_pooled':False,'raw_stage_imputed':False,'primary_day':'valid identity/cohort/defined stage/date AND >=1 processed event AND matching diet_logging daily index','missing_day_is_zero':False,'recorded_day_implies_complete_intake':False,'automatic_event_deduplication':False,'duplicate_policy':'flag exact-row and same-food-time candidates; preserve all processed rows; no event lifecycle identity exists in exposed scalar columns','primary_date':'source collection_date','local_timestamp_required':False,'aware_timezone':'Asia/Jerusalem','meal_ingestion_time_validated':False,'energy_multiplier':1,'clean_nutrients':'preserve original; append *_clean; negative/nonfinite/missing -> NA, never zero','energy_day':'primary_day AND all event calories finite/nonnegative AND total>0','fiber_day':'energy_day AND all event fiber finite/nonnegative','global_minimum_days':None,'ge7_counts':'diagnostic only','energy_plausibility_cutoff':None,'span_or_backfill_cutoff':None,'NOVA_mapping':'deferred to W07; no additional external sources','outcomes_read':False,'merged_rows':0}
rulelocal=B/'food_day_rules.yaml';rulelocal.write_text(json.dumps(rules,indent=2))
exclusions=pd.DataFrame([{'unit':'processed_event','reason':k,'n':v} for k,v in sorted(reason_counts.items())]+[{'unit':'day_key','reason':str(k),'n':int(v)} for k,v in day.day_disposition.value_counts().items()]);exclusions.to_csv(B/'food_exclusions.csv',index=False)
result={'source_inputs':inputs,'processed_rows':n,'raw_rows_scanned':raw_n,'raw_stage_counts':dict(raw_stages),'processed_stage_counts':dict(stages),'lifecycle_fields_exposed':lifecycle,'shared_signature_columns_ignoring_stage':shared,'candidate_signature_columns':candidate,'signature_summary':sigsummary,'clock_counts':dict(clock),'conflict_seconds_top':delta.most_common(30),'conflict_date_offsets':offsets.most_common(),'timezone_labels':dict(timezones),'day_flow':summary,'kept_event_rows':kept,'event_disposition':dict(reason_counts),'day_rows':len(day),'no_rows_merged':True,'source_nutrient_multiplier':1,'source_missing_not_zero':True,'synthetic_checks_passed':4,'event_day_conservation_passed':True,'independent_binary_readback_passed':True,'outcomes_read':False,'runtime_seconds':time.monotonic()-start,'peak_rss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'packages':{'python':platform.python_version(),'pandas':pd.__version__,'pyarrow':pa.__version__,'numpy':np.__version__},'limitations':['Exposed tables lack event lifecycle ID/edit/delete/backfill fields; raw-only signatures cannot establish deletion.','Dictionary describes food logging time; actual ingestion time and recording delay are not independently established.','Same food and timestamp and exact duplicate signatures are flags, not proof of duplicate consumption.','No global minimum days or complete-day calorie/span threshold is frozen in W06.','NOVA and dietary exposure construction remain W07/W08.']}
for name,old in inputs.items():
 p=R/name;st=p.stat();assert st.st_size==old['bytes'] and st.st_mtime_ns==old['mtime_ns'];assert pq.ParquetFile(p).metadata.num_rows==old['rows']
mirrors=[]
for source,target in [(cleanpath,P/'analysis/shared'/cleanpath.name),(linepath,P/'analysis/shared'/linepath.name),(daypath,P/'analysis/shared'/daypath.name),(rulelocal,P/'config/food_day_rules.yaml'),(B/'food_exclusions.csv',P/'data_qc/food_exclusions.csv')]:
 target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists(),str(target)
 with source.open('rb') as src,target.open('wb') as dst:shutil.copyfileobj(src,dst,1024*1024)
 time.sleep(1);assert source.stat().st_size==target.stat().st_size and sha(source)==sha(target)
 if source.suffix=='.parquet':assert pq.ParquetFile(source).metadata.num_rows==pq.ParquetFile(target).metadata.num_rows
 mirrors.append({'runtime':str(source),'cloud':str(target),'bytes':source.stat().st_size,'sha256':sha(source),'verified':True})
result['mirrors']=mirrors;result['source_unchanged_checks_passed']=True
result=json.loads(json.dumps(result))
(Q/'w06_summary_TRE_ONLY.json').write_text(json.dumps(result,indent=2))
(Q/'binary_mirror_verification.json').write_text(json.dumps(mirrors,indent=2))
assert json.loads((Q/'w06_summary_TRE_ONLY.json').read_text())==result
M=P/'memos/w06_food_cleaning_20261028';M.mkdir(parents=True,exist_ok=True)
(M/'MEMO.md').write_text('# W06 food cleaning, 2026-10-28\n\nCode: '+str(Path(__file__))+'\n\nFunction: outcome-blind full source scan, raw/processed signature audit, timestamp diagnostics, primary indexed-event-day views; no event deduplication or raw stage imputation.\n\nResult: '+str(Q)+'\n\nRuntime/log: '+str(L)+'\n\nShared outputs: '+str(P/'analysis/shared')+'\n\nRules: '+str(P/'config/food_day_rules.yaml')+'\n\nCounts and validation:\n```json\n'+json.dumps({k:result[k] for k in ['processed_rows','raw_rows_scanned','kept_event_rows','day_rows','event_disposition','day_flow','clock_counts','signature_summary','limitations']},indent=2)+'\n```\n\nActual exit code must be checked in 02_actual_exit_code. No outcome analysis performed; exposure construction remains W07/W08.\n')
assert (M/'MEMO.md').read_text().startswith('# W06')
print('W06_BUILD_VERIFIED','KEPT_EVENTS',kept,'DAY_ROWS',len(day),'MIRRORS',len(mirrors),'PEAK_RSS_KiB',result['peak_rss_KiB'],flush=True)
print('RESULT',Q,'MEMO',M/'MEMO.md',flush=True)
con.close()
