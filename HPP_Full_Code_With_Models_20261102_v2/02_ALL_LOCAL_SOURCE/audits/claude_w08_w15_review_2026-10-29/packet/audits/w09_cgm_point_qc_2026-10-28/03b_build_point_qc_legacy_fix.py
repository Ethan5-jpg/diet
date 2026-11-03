# -*- coding: utf-8 -*-
"""All CGM fragments, connection QC, calendar-day coverage; never interpolates glucose."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,hashlib,shutil,resource,platform
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');R=Path('/home/ec2-user/studies/hpp_datasets/cgm');Q=P/'data_qc/w09_cgm_point_qc_20261028';L=Path('/home/ec2-user/multisystem_runtime_20261021/w09_cgm_point_qc_20261028');B=L/'artifacts_v2';B.mkdir(exist_ok=False)
K=['participant_id','cohort','research_stage'];KC=K+['connection_id'];KD=K+['date'];TZ='Asia/Jerusalem';start=time.monotonic()
assert sys.executable=='/home/ec2-user/miniconda/envs/pheno/bin/python'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p,columns=None):return pq.ParquetFile(p,pre_buffer=False).read(columns=columns,use_threads=False).to_pandas(ignore_metadata=True)
def cid(v):
 assert pd.notna(v)
 return str(int(v)) if isinstance(v,(int,np.integer)) else str(v)
def epochs(t):return t.dt.tz_convert('UTC').dt.tz_localize(None).to_numpy(dtype='datetime64[ns]').astype('int64')/1e9
def support(t,a,b):
 # Each recorded value represents at most one nominal 15-minute acquisition slot.
 # Union measures observed support, never invents glucose values in gaps.
 lo=np.maximum(t-450,a);hi=np.minimum(t+450,b);ok=hi>lo;lo=lo[ok];hi=hi[ok]
 if not len(lo):return 0.,float(b-a)
 total=0.;end=a;gap=0.
 for x,y in zip(lo,hi):
  if x>end:gap=max(gap,float(x-end));total+=float(y-x);end=y
  elif y>end:total+=float(y-end);end=y
 gap=max(gap,float(b-end));return total,gap
# Fixed synthetic coverage, boundary and DST checks.
for date,hours in [('2026-01-04',24),('2026-03-27',23),('2026-10-25',25)]:
 a=pd.Timestamp(date,tz=TZ);b=a+pd.DateOffset(days=1);a=a.timestamp();b=b.timestamp();assert (b-a)/3600==hours
 t=np.arange(a+450,b,900);c,g=support(t,a,b);assert c==b-a and g==0
 t=t[(t<a+3600)|(t>=a+10800)];c,g=support(t,a,b);assert g==7200
assert support(np.array([]),0,86400)==(0.,86400.)
contract=json.loads((Q/'source_contract_TRE_ONLY.json').read_text());units=[]
for d in contract['dictionaries']:
 for row in d['rows']:
  if row.get('tabular_field_name')=='glucose':units.append(row)
assert len(units)==1 and units[0]['units']=='mg/dL' and float(units[0]['min_plausible_value'])==40 and float(units[0]['max_plausible_value'])==500 and units[0]['sampling_rate']=='15 minutes'
I=load(R/'cgm.parquet');assert set(KC+['cgm_time_series','cgm_device_type','timezone'])<=set(I)
I['connection_id']=I.connection_id.map(cid);assert I.connection_id.is_unique and len(I)==9497 and not I[KC].isna().any().any()
assert set(I.timezone.str.lower())=={'asia/jerusalem'}
assert set(I.cgm_device_type)<= {'abbott_freestyle_libre_pro','abbott_freestyle_libre_pro_iq'}
index={row['connection_id']:row for row in I.to_dict('records')}
# Preflight all legacy fields before the costly full scan; source uses cgm_daily_mean.
old=load(R/'iglu_daily.parquet');assert set(KC+['collection_date','cgm_daily_mean'])<=set(old),list(old.columns)
old=old.rename(columns={'cgm_daily_mean':'cgm_mean'});old['connection_id']=old.connection_id.map(cid)
print('LEGACY_PREFLIGHT_PASS',len(old),'mean_field=cgm_daily_mean',flush=True)
parts=[Path(p) for p in json.loads((Q/'fragment_paths_TRE_ONLY.json').read_text())];assert len(parts)==len(I)
manifest=[];conn=[];daily=[];seen=set();pointcount=Counter();intervals=Counter();masked=[];curve_samples=[]
# Load only food-day eligibility, no dietary exposure or disease outcomes; used later for predeclared connection choice.
food=load(P/'analysis/shared/food_daily_TRE_ONLY.parquet',KD+['energy_eligible']);assert not food.duplicated(KD).any()
foodsets={key:set(z.loc[z.energy_eligible,'date']) for key,z in food.groupby(K,sort=False)}
for ordinal,p in enumerate(parts):
 assert time.monotonic()-start<3600 and shutil.disk_usage(L).free>3*1024**3
 st=p.stat();f=pq.ParquetFile(p,pre_buffer=False);assert f.metadata.num_rows<100000
 assert set(KC+['collection_timestamp','glucose'])<=set(f.schema_arrow.names)
 assert str(f.schema_arrow.field('collection_timestamp').type).endswith('tz=Asia/Jerusalem]')
 d=f.read(use_threads=False).to_pandas(ignore_metadata=True);assert len(d)==f.metadata.num_rows and len(d)>0
 assert len(d[KC].drop_duplicates())==1 and not d[KC].isna().any().any()
 row=d.iloc[0];c=cid(row.connection_id);assert c in index and c not in seen;seen.add(c);ir=index[c]
 for k in K:assert row[k]==ir[k]
 key=tuple(ir[k] for k in K);n=len(d);t=pd.to_datetime(d.collection_timestamp,utc=True,errors='coerce').dt.tz_convert(TZ);d['t']=t;d['g']=pd.to_numeric(d.glucose,errors='coerce')
 badts=int(t.isna().sum());z=d.loc[t.notna(),['t','g']].copy();exact=int(z.duplicated(['t','g']).sum());z=z.drop_duplicates(['t','g'])
 conflicttimes=set(z.loc[z.duplicated('t',keep=False),'t']);conflictdates={x.date() for x in conflicttimes};conflictrows=int(z.t.isin(conflicttimes).sum());z=z.loc[~z.t.isin(conflicttimes)].copy()
 numericbad=~np.isfinite(z.g.to_numpy(dtype=float));outside=(z.g<40)|(z.g>500);invalid=int((numericbad|outside).sum());good=z.loc[~(numericbad|outside)].sort_values('t');assert not good.t.duplicated().any()
 counts={'input_points':n,'missing_timestamp_points':badts,'exact_duplicate_rows_removed':exact,'conflicting_unique_rows_removed':conflictrows,'invalid_numeric_or_range_points':invalid,'valid_unique_points':len(good)};assert n==sum(counts[x] for x in counts if x!='input_points');pointcount.update(counts)
 allts=t.dropna();assert len(allts)>0
 tt=epochs(good.t).to_numpy() if False else epochs(good.t)
 dif=np.diff(tt);rounded=np.round(dif,6);intervals.update(rounded.tolist());offgrid=int((np.abs(dif/900-np.round(dif/900))>1e-6).sum());cadenceok=offgrid==0
 first=allts.min();last=allts.max();dates=pd.date_range(pd.Timestamp(first.date()),pd.Timestamp(last.date()),freq='D');assert len(dates)<90
 gd=good.t.dt.date.to_numpy();gg=good.g.to_numpy(dtype=float);rawdates=d.t.dt.date
 cr=dict(zip(KC,[ir[k] for k in K]+[c]));cr.update(counts,fragment_ordinal=ordinal,first_observed=first,last_observed=last,observed_calendar_span=len(dates),span_elapsed_days=(last-first).total_seconds()/86400,device=ir['cgm_device_type'],timezone=TZ,unit='mg/dL',nominal_interval_seconds=900,offgrid_intervals=offgrid,cadence_verified=cadenceok,conflict_timestamps=len(conflicttimes),source_prior_QC_loss_pct=ir.get('percentage_of_cgm_datapoints_lost_in_qc'),source_globally_postprocessed=True)
 rows=[]
 for date in dates:
  date=date.date();a=pd.Timestamp(date,tz=TZ);b=a+pd.DateOffset(days=1);startsec=a.timestamp();endsec=b.timestamp();daysec=endsec-startsec
  keep=gd==date;vals=gg[keep];num=len(vals);cover,gap=support(tt[(tt>=startsec-450)&(tt<=endsec+450)],startsec,endsec);cov=cover/daysec
  source_day=d.loc[rawdates==date,'g'];mean=float(vals.mean()) if num else np.nan;sd=float(vals.std(ddof=1)) if num>1 else np.nan
  r=dict(zip(KC,[ir[k] for k in K]+[c]));r.update(date=date,n_points=num,n_input_points=int((rawdates==date).sum()),day_length_seconds=int(daysec),covered_seconds=cover,coverage_fraction=cov,max_uncovered_gap_seconds=gap,has_timestamp_conflict=date in conflictdates,cadence_verified=cadenceok,glucose_mean_mg_dL=mean,glucose_sd_mg_dL=sd,glucose_CV_pct=100*sd/mean if num>1 else np.nan,source_all_points_mean=float(source_day.mean()) if source_day.notna().any() else np.nan,unit='mg/dL',timezone=TZ,point_qc_verified=True,clock_basis='CGM Asia/Jerusalem calendar dates; food logging date is not validated ingestion clock')
  for threshold in [.7,.8,.9]:
   for gap_h in [1,2,4]:r['eligible_cov'+str(int(threshold*100))+'_gap'+str(gap_h)+'h']=bool(num>0 and cov>=threshold and gap<=gap_h*3600 and cadenceok and date not in conflictdates and badts==0)
  r['cgm_day_eligible']=r['eligible_cov80_gap2h'];r['qc_rule_status']='operational_candidate_80pct_2h; sensitivity grid retained; not final model protocol freeze';rows.append(r)
  # Fixed first 100 dense, regular days; mask random slots only for measurement precision diagnostic.
  if len(masked)<100 and cov>=.99 and num>=92 and not conflictdates and cadenceok:
   rng=np.random.default_rng(20261028+len(masked));nkeep=int(np.ceil(.8*num));sample=vals[np.sort(rng.choice(num,nkeep,replace=False))];masked.append(float(sample.mean()-mean))
 conn.append(cr);daily.extend(rows)
 if ordinal in {0,len(parts)//2,len(parts)-1}:curve_samples.append({'ordinal':ordinal,'identity':{k:ir[k] for k in KC},'source_path':str(p),'points':[{'timestamp':x.isoformat(),'glucose':float(y)} for x,y in zip(good.t,good.g)]})
 manifest.append({'ordinal':ordinal,'source_path':str(p),'source_bytes':st.st_size,'source_mtime_ns':st.st_mtime_ns,'source_sha256':sha(p),'rows':n,'connection_id':c,'status':'processed','unit':'mg/dL','timestamp_type':str(f.schema_arrow.field('collection_timestamp').type),'counts':counts})
 assert p.stat().st_size==st.st_size and p.stat().st_mtime_ns==st.st_mtime_ns
 if (ordinal+1)%1000==0:print('W09_PROGRESS',ordinal+1,'OF',len(parts),'POINTS',pointcount['input_points'],flush=True)
assert seen==set(index)
C=pd.DataFrame(conn);D=pd.DataFrame(daily);assert not D.duplicated(KC+['date']).any()
D['food_energy_day_present']=[row.date in foodsets.get(tuple(getattr(row,k) for k in K),set()) for row in D.itertuples()]
D['qualified_food_energy_common']=D.food_energy_day_present&D.cgm_day_eligible
cg=D.groupby(KC).agg(qualified_days=('cgm_day_eligible','sum'),qualified_food_common_days=('qualified_food_energy_common','sum')).reset_index();C=C.merge(cg,on=KC,validate='one_to_one')
C=C.sort_values(K+['qualified_food_common_days','first_observed','connection_id'],ascending=[True]*len(K)+[False,True,True],kind='stable');C['selected_connection']=~C.duplicated(K);selected=set(C.loc[C.selected_connection,'connection_id']);D['selected_connection']=D.connection_id.isin(selected)
assert C.loc[C.selected_connection].groupby(K).size().eq(1).all()
multis=[]
for key,z in C.groupby(K,sort=False):
 if len(z)<2:continue
 candidates=z.to_dict('records');overlap=max(0,(min(z.last_observed)-max(z.first_observed)).total_seconds());multis.append({'key':dict(zip(K,key)),'connections':candidates,'overlap_seconds':overlap,'interpretation':'multiple documented connection IDs; sensor replacement reason not inferred'})
selecteddaily=D.loc[D.selected_connection].copy();assert not selecteddaily.duplicated(KD).any()
# TRE-only checkpoint before reconciliation, to recover our own completed scan without source reread.
import pickle
with (L/'completed_point_scan_v2_TRE_ONLY.pkl').open('wb') as cache:pickle.dump({'C':C,'D':D,'selecteddaily':selecteddaily,'manifest':manifest,'multis':multis,'counts':pointcount,'intervals':intervals,'masked':masked,'curve_samples':curve_samples},cache)
# Source legacy daily reconciliation at exact person/visit/connection/date grain.
assert set(KC+['collection_date','cgm_mean'])<=set(old)
old['date']=pd.to_datetime(old.collection_date,utc=True).dt.tz_convert(TZ).dt.date;assert not old.duplicated(KC+['date']).any()
rec=D.merge(old[KC+['date','cgm_mean']],on=KC+['date'],how='outer',indicator=True,validate='one_to_one');both=rec.loc[rec._merge=='both'];delta=both.glucose_mean_mg_dL-both.cgm_mean
rec['reconstructed_minus_legacy_mean_mg_dL']=rec.glucose_mean_mg_dL-rec.cgm_mean
recon={'legacy_rows':len(old),'reconstructed_rows':len(D),'matched':len(both),'reconstructed_only':int((rec._merge=='left_only').sum()),'legacy_only':int((rec._merge=='right_only').sum()),'mean_abs_difference_mg_dL':float(delta.abs().mean()),'max_abs_difference_mg_dL':float(delta.abs().max()),'diff_over_1e_6':int((delta.abs()>1e-6).sum()),'qualified_matched':int(both.cgm_day_eligible.sum())}
def sc(d):
 g=d.groupby(K).size();return {'days':len(d),'people':int(d.participant_id.nunique()),'people_visits':len(g),'visits_ge7_DIAGNOSTIC':int((g>=7).sum()),'median_days':float(g.median()) if len(g) else None}
baseline=(selecteddaily.cohort=='10k')&(selecteddaily.research_stage=='00_00_visit');Z=selecteddaily.loc[baseline];counts={}
for name,d in [('all_source_selected',selecteddaily),('baseline_10k_selected',Z)]:
 counts[name]={'observed_calendar':sc(d),'qualified_CGM':sc(d.loc[d.cgm_day_eligible]),'qualified_CGM_food_energy':sc(d.loc[d.qualified_food_energy_common])}
grid=[]
for threshold in [70,80,90]:
 for gap in [1,2,4]:
  flag='eligible_cov'+str(threshold)+'_gap'+str(gap)+'h';grid.append({'coverage_pct':threshold,'max_gap_h':gap,'baseline_qualified':sc(Z.loc[Z[flag]]),'baseline_food_energy_common':sc(Z.loc[Z[flag]&Z.food_energy_day_present])})
summary={'full_point_scan_performed':True,'fragments':len(parts),'index_rows':len(I),'unique_people':int(I.participant_id.nunique()),'index_people_visits':len(C.loc[C.selected_connection]),'point_accounting':dict(pointcount),'devices':I.cgm_device_type.value_counts().to_dict(),'research_stages':I.research_stage.value_counts().to_dict(),'unit_dictionary':units[0],'nominal_interval_seconds':900,'interval_top_seconds':{str(a):b for a,b in intervals.most_common(8)},'offgrid_connections':int((C.offgrid_intervals>0).sum()),'connection_span_over14_calendar_days':int((C.observed_calendar_span>14).sum()),'span_calendar_quantiles':C.observed_calendar_span.quantile([0,.5,.99,1]).to_dict(),'multi_connection_visits':len(multis),'multi_connection_overlap_seconds':[m['overlap_seconds'] for m in multis],'counts':counts,'all_day_lengths':D.day_length_seconds.value_counts().to_dict(),'selected_baseline_day_lengths':Z.day_length_seconds.value_counts().to_dict(),'legacy_reconciliation':recon,'sensitivity_grid':grid,'precision_random20pct_mask_dense_days':{'n':len(masked),'mean_absolute_mean_error_mg_dL':float(np.mean(np.abs(masked))) if masked else None,'p95_absolute_error_mg_dL':float(np.quantile(np.abs(masked),.95)) if masked else None,'limitation':'fixed-seed MCAR masking; does not establish real missingness at random or validate causal assumptions'},'point_QC_rule':'dictionary glucose 40..500 mg/dL; finite; exact duplicates collapsed with counts; conflicting timestamps excluded and day quarantined; no interpolation','daily_QC_rule':'candidate coverage>=80%, max unsupported gap<=2h, cadence verified, no timestamp conflict or missing timestamps; grid70/80/90 x1/2/4h','coverage_definition':'union of observed +/-7.5-minute acquisition-slot supports, clipped to actual Israel calendar day; coverage is not glucose interpolation','connection_selection_rule':'most qualified common days with W06 energy-complete indexed food days; earliest observed start; lexical stable connection ID; never by glucose value/diet effect','not_yet_frozen':['primary CGM endpoint','final minimum window days','final model QC sensitivity choice','diet ingestion-time semantics'],'synthetic_checks_passed':7,'elapsed_seconds':time.monotonic()-start,'peak_rss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'packages':{'python':platform.python_version(),'pandas':pd.__version__,'pyarrow':pa.__version__}}
# Write on local runtime disk first, then cloud byte copy and independent footer/hash.
artifacts=[]
def save(d,name,dst):
 src=B/name;pq.write_table(pa.Table.from_pandas(d,preserve_index=False),src,compression='zstd');assert pq.ParquetFile(src).metadata.num_rows==len(d);dst.parent.mkdir(parents=True,exist_ok=True);assert not dst.exists();shutil.copyfile(src,dst);time.sleep(.3);assert sha(src)==sha(dst) and pq.ParquetFile(dst).metadata.num_rows==len(d);artifacts.append({'runtime':str(src),'cloud':str(dst),'rows':len(d),'bytes':src.stat().st_size,'sha256':sha(src)})
save(C,'cgm_point_connection_QC.parquet',P/'data_qc/cgm_point_connection_QC.parquet');save(D,'cgm_daily_all_connections_TRE_ONLY.parquet',Q/'cgm_daily_all_connections_TRE_ONLY.parquet');save(selecteddaily,'cgm_daily_TRE_ONLY.parquet',P/'analysis/shared/cgm_daily_TRE_ONLY.parquet');save(rec,'cgm_daily_reconciliation_TRE_ONLY.parquet',Q/'cgm_daily_reconciliation_TRE_ONLY.parquet')
(Q/'fragment_processing_manifest_TRE_ONLY.json').write_text(json.dumps(manifest,indent=2));(Q/'multi_connection_decisions_TRE_ONLY.json').write_text(json.dumps(multis,default=str,indent=2));(Q/'fixed_curve_samples_TRE_ONLY.json').write_text(json.dumps(curve_samples,default=str))
pd.DataFrame([recon]).to_csv(P/'data_qc/cgm_daily_reconciliation.csv',index=False);(P/'config/cgm_point_daily_rules_W09_20261028.json').write_text(json.dumps({k:summary[k] for k in ['point_QC_rule','daily_QC_rule','coverage_definition','connection_selection_rule','not_yet_frozen']},indent=2))
summary['artifacts']=artifacts;(Q/'summary_TRE_ONLY.json').write_text(json.dumps(summary,default=str,indent=2));assert len(json.loads((Q/'fragment_processing_manifest_TRE_ONLY.json').read_text()))==9497
print('W09_BUILD_DONE',json.dumps({k:summary[k] for k in ['fragments','point_accounting','offgrid_connections','multi_connection_visits','multi_connection_overlap_seconds','counts','all_day_lengths','legacy_reconciliation','precision_random20pct_mask_dense_days','peak_rss_KiB']}),flush=True)
