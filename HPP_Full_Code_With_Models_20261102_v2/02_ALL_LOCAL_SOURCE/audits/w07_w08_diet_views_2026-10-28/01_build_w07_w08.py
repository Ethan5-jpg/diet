# -*- coding: utf-8 -*-
"""W07 FoodID crosswalk, W08 diet-only daily/window candidates. No outcomes."""
import os
for x in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[x]='1'
import sys,json,ast,hashlib,time,resource,shutil,platform
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');R=Path('/home/ec2-user/studies/hpp_datasets/diet_logging')
L=Path('/home/ec2-user/multisystem_runtime_20261021/w07_w08_diet_views_20261028');Q=P/'data_qc/w07_w08_diet_views_20261028';B=L/'artifacts'
assert sys.executable=='/home/ec2-user/miniconda/envs/pheno/bin/python'
assert shutil.disk_usage(L).free>4*1024**3
B.mkdir(exist_ok=False);Q.mkdir(parents=True,exist_ok=True)
K=['participant_id','cohort','research_stage'];KD=K+['date'];start=time.monotonic()
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def norm(x):return str(x).strip().casefold() if pd.notna(x) else ''
def fid(x):return int(x) if pd.notna(x) else None
def batches(p,cols):
 f=pq.ParquetFile(p,pre_buffer=False);assert set(cols)<=set(f.schema_arrow.names)
 for b in f.iter_batches(batch_size=8192,columns=cols,use_threads=False):yield b.to_pandas(ignore_metadata=True)
def guard():
 assert time.monotonic()-start<1200,'W07/W08 time budget'
 assert shutil.disk_usage(L).free>3*1024**3
v06=json.loads((P/'data_qc/w06_food_cleaning_20261028/verification.json').read_text());assert v06['verified'] and v06['build_actual_exit']==0
assert (Path('/home/ec2-user/multisystem_runtime_20261021/w06_food_cleaning_20261028')/'03_actual_exit_code').read_text().strip()=='0'
legacy=P/'scripts/w04_mapping_review_20261027/03_audit_event_mapping_days.py';tree=ast.parse(legacy.read_text());catalog=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CATALOG' for t in n.targets));mp={norm(a):b for a,b in catalog};assert len(mp)==len(catalog)==643
assert {a for a,b in catalog if b is None}=={'Stevia Sweetner','Spirulina'}
cv=hashlib.sha256(json.dumps(catalog,separators=(',',':')).encode()).hexdigest();version='legacy_NCDiet_NOVA_643_'+cv[:12]+'_FoodID_W07_20261028'
source=R/'diet_logging_events.parquet';st=source.stat();inputfp={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'rows':pq.ParquetFile(source).metadata.num_rows};assert inputfp['rows']==2640959
foods={};categories=Counter();n=0
for d in batches(source,['food_id','short_food_name','product_name','food_category']):
 n+=len(d);categories.update(d.food_category.dropna().astype(str))
 for a,b,c,e in d.drop_duplicates().itertuples(index=False,name=None):
  a=fid(a)
  if a is None:continue
  z=foods.setdefault(a,{'names':set(),'products':set(),'categories':set()})
  for key,val in [('names',b),('products',c),('categories',e)]:
   if pd.notna(val):z[key].add(str(val))
 guard()
assert n==2640959
records=[];lookup={}
for a,z in sorted(foods.items()):
 names={norm(x) for x in z['names']};vals={mp.get(x) for x in names};known=bool(names) and len(vals)==1 and next(iter(vals)) in [1,2,3,4]
 conflict=len(vals-{None})>1
 status='exact_legacy_short_name' if known else ('conflicting_legacy_class' if conflict else ('legacy_blank' if names and names<=mp.keys() else ('missing_short_name' if not names else 'not_in_legacy_catalog')))
 nova=next(iter(vals)) if known else None
 if known:lookup[a]=nova
 records.append({'food_id':a,'short_names_json':json.dumps(sorted(z['names'])),'products_json':json.dumps(sorted(z['products'])),'source_categories_json':json.dumps(sorted(z['categories'])),'n_short_names':len(names),'n_products':len(z['products']),'n_categories':len(z['categories']),'NOVA':nova,'classification_status':status,'mapping_version':version,'classification_basis':'user-provided legacy NC-Diet short-name catalogue; trim/casefold exact; no fuzzy/external mapping','manual_current_product_NOVA_review_completed':False})
cross=pd.DataFrame(records);cross.NOVA=cross.NOVA.astype('Int8');assert cross.food_id.is_unique;assert not (cross.classification_status=='conflicting_legacy_class').any()
print('W07_CROSSWALK','FOOD_IDS',len(cross),'NOVA_KNOWN_IDS',len(lookup),'MULTI_NAMES',int((cross.n_short_names>1).sum()),'MULTI_CATEGORIES',int((cross.n_categories>1).sum()),flush=True)
# Process only W06 admitted events; FoodID joins are dictionary lookups, cannot expand rows.
clean=P/'analysis/shared/food_events_clean_TRE_ONLY.parquet';days={};unknown=defaultdict(lambda:[0,0.]);foodcounts=Counter();n=0;missingid=0;nameconflict=0
cols=KD+['source_row_ordinal','food_id','short_food_name','product_name','calories_kcal_clean','dietary_fiber_g_clean']
for d in batches(clean,cols):
 n+=len(d);v=d.food_id.map(fid);nova=v.map(lookup);namevals=d.short_food_name.map(norm).map(mp)
 mismatch=nova.notna()&namevals.notna()&(nova!=namevals);nameconflict+=int(mismatch.sum());assert not mismatch.any()
 E=d.calories_kcal_clean.to_numpy(dtype=float);F=d.dietary_fiber_g_clean.to_numpy(dtype=float);eok=np.isfinite(E)&(E>=0);fok=np.isfinite(F)&(F>=0);known=nova.isin([1,2,3,4]).to_numpy()
 d['_n']=1;d['_E']=np.where(eok,E,0.);d['_badE']=(~eok).astype(int);d['_F']=np.where(fok,F,0.);d['_badF']=(~fok).astype(int);d['_K']=np.where(eok&(nova.to_numpy()==4),E,0.);d['_U']=np.where(eok&~known,E,0.);d['_unknownN']=(~known).astype(int)
 g=d.groupby(KD,sort=False)[['_n','_E','_badE','_F','_badF','_K','_U','_unknownN']].sum()
 for key,arr in zip(g.index,g.to_numpy()):
  if key in days:days[key]+=arr
  else:days[key]=arr
 missingid+=int(v.isna().sum());foodcounts.update(v.dropna().astype(int))
 u=d.loc[~known].copy();u['_validE']=u._E
 for (a,b,c),q in u.groupby(['food_id','short_food_name','product_name'],dropna=False,sort=False):
  key=(fid(a),str(b),str(c));z=unknown[key];z[0]+=len(q);z[1]+=float(q._validE.sum())
 guard()
assert n==v06['clean_rows']==2420902 and nameconflict==0
D=pd.DataFrame([dict(zip(KD,key),event_rows=int(a[0]),energy_kcal_partial=a[1],energy_bad_rows=int(a[2]),fiber_g_partial=a[3],fiber_bad_rows=int(a[4]),known_UPF_kcal=a[5],unknown_NOVA_kcal=a[6],unknown_NOVA_rows=int(a[7])) for key,a in days.items()])
assert not D.duplicated(KD).any();assert len(D)==149912
qc=pq.ParquetFile(P/'analysis/shared/food_day_qc_TRE_ONLY.parquet').read(columns=KD+['primary_day','n_events','energy_eligible','fiber_eligible']).to_pandas();qc=qc.loc[qc.primary_day]
D=D.merge(qc,on=KD,validate='one_to_one');assert len(D)==149912 and (D.event_rows==D.n_events).all()
eok=(D.energy_bad_rows==0)&(D.energy_kcal_partial>0);fok=eok&(D.fiber_bad_rows==0);assert D.energy_eligible.equals(eok) and D.fiber_eligible.equals(fok)
D['energy_kcal']=D.energy_kcal_partial.where(eok);D['fiber_g']=D.fiber_g_partial.where(fok);D['mapping_version']=version
D['NOVA_energy_coverage']=np.where(eok,1-D.unknown_NOVA_kcal/D.energy_kcal,np.nan)
D['UPF_lower_pp']=100*D.known_UPF_kcal/D.energy_kcal;D['UPF_upper_pp']=100*(D.known_UPF_kcal+D.unknown_NOVA_kcal)/D.energy_kcal
D['UPF_bound_width_pp']=D.UPF_upper_pp-D.UPF_lower_pp
D['UPF_point_zero_unknown_energy_pp']=D.UPF_lower_pp.where(eok&(D.unknown_NOVA_kcal==0))
D['UPF_point_all_rows_classified_pp']=D.UPF_lower_pp.where(eok&(D.unknown_NOVA_rows==0))
D['fiber_density_g_per_1000kcal']=1000*D.fiber_g/D.energy_kcal
D['clock_basis']='W06 source logging calendar date; actual ingestion date not independently validated'
D=D.sort_values(KD).reset_index(drop=True);D['record_day_order']=D.groupby(K).cumcount()+1
assert ((D.UPF_upper_pp.dropna()-D.UPF_lower_pp.dropna())>=-1e-10).all();assert D.loc[~eok,'UPF_lower_pp'].isna().all()
assert ((D.loc[eok,'UPF_lower_pp']>=-1e-10)&(D.loc[eok,'UPF_upper_pp']<=100+1e-8)).all();assert D.loc[D.unknown_NOVA_kcal>0,'UPF_point_zero_unknown_energy_pp'].isna().all()
# W08 window builder preserves numerator, denominator, dates and number of eligible days separately per candidate.
def window(d):
 version=str(d.mapping_version.iloc[0]) if len(d) else 'empty'
 rows=[]
 for key,z in d.groupby(K,sort=False):
  a=dict(zip(K,key));a.update(window_id='food_window',n_recorded_days=len(z),first_recorded_date=z.date.min(),last_recorded_date=z.date.max(),mapping_version=version)
  e=z.loc[z.energy_eligible];a['n_energy_days']=len(e);a['n_unknown_NOVA_days']=int((e.unknown_NOVA_kcal>0).sum())
  total=float(e.energy_kcal.sum());k=float(e.known_UPF_kcal.sum());u=float(e.unknown_NOVA_kcal.sum())
  a.update(energy_kcal_sum_energy_days=total if len(e) else np.nan,UPF_known_kcal_sum_energy_days=k if len(e) else np.nan,NOVA_unknown_kcal_sum_energy_days=u if len(e) else np.nan,NOVA_coverage_energyweighted=1-u/total if total>0 else np.nan)
  for label,col in [('UPF_zeroU','UPF_point_zero_unknown_energy_pp'),('UPF_strict','UPF_point_all_rows_classified_pp'),('fiber','fiber_density_g_per_1000kcal')]:
   t=z.loc[z[col].notna()];a['n_'+label+'_days']=len(t);a[label+'_equal_day_mean']=float(t[col].mean()) if len(t) else np.nan
   a[label+'_numerator_sum']=float(t.fiber_g.sum() if label=='fiber' else t.known_UPF_kcal.sum()) if len(t) else np.nan
   a[label+'_energy_denominator_sum']=float(t.energy_kcal.sum()) if len(t) else np.nan
   a[label+'_cumulative_ratio']=(1000 if label=='fiber' else 100)*a[label+'_numerator_sum']/a[label+'_energy_denominator_sum'] if len(t) else np.nan
   a[label+'_first_valid_date']=t.date.min() if len(t) else None;a[label+'_last_valid_date']=t.date.max() if len(t) else None
   a[label+'_flag_ge7_DIAGNOSTIC']=len(t)>=7
  a['UPF_lower_equal_day_mean_all_energy_days']=float(e.UPF_lower_pp.mean()) if len(e) else np.nan;a['UPF_upper_equal_day_mean_all_energy_days']=float(e.UPF_upper_pp.mean()) if len(e) else np.nan
  a['UPF_lower_cumulative_all_energy_days']=100*k/total if total>0 else np.nan;a['UPF_upper_cumulative_all_energy_days']=100*(k+u)/total if total>0 else np.nan
  a['UPF_exact_full_energy_window_equal_day_mean']=a['UPF_lower_equal_day_mean_all_energy_days'] if len(e)>0 and u==0 else np.nan
  a['UPF_exact_full_energy_window_cumulative_ratio']=100*k/total if total>0 and u==0 else np.nan
  rows.append(a)
 return pd.DataFrame(rows)
W=window(D);assert not W.duplicated(K).any();assert int(W.n_recorded_days.sum())==len(D)
# Reusable CGM adapter requires unique, explicitly W09-qualified days; never trusts old index presence as point QC.
def common_view(d,cgm):
 required=KD+['cgm_day_eligible','point_qc_verified'];assert set(required)<=set(cgm)
 assert not cgm.duplicated(KD).any(),'CGM daily keys must be unique';assert cgm.point_qc_verified.eq(True).all(),'W09 verification required'
 out=d.merge(cgm.loc[cgm.cgm_day_eligible,required],on=KD,how='inner',validate='one_to_one');return out
# Worked numerical / missingness / join checks independent of real outcomes.
assert 100*600/2000==30 and 100*(600+200)/2000==40
assert float(np.mean([80.,10.]))==45 and 100*(800+300)/(1000+3000)==27.5
assert pd.isna(pd.Series([np.nan]).sum(min_count=1))
base={K[0]:1,K[1]:'synthetic',K[2]:'visit','date':'2026-01-01'};s=pd.DataFrame([base]);c=pd.DataFrame([dict(base,cgm_day_eligible=True,point_qc_verified=True)]);assert len(common_view(s,c))==1
for bad in [pd.concat([c,c]),c.assign(point_qc_verified=False)]:
 try:common_view(s,bad)
 except AssertionError:pass
 else:raise AssertionError('CGM adapter accepted invalid input')
# Baseline descriptive exposure QC, no Y. Observed odd/even days are not discovery/confirmation groups.
Z=D.loc[(D.cohort=='10k')&(D.research_stage=='00_00_visit')];ZW=W.loc[(W.cohort=='10k')&(W.research_stage=='00_00_visit')]
def describe(v):
 x=pd.to_numeric(v,errors='coerce').dropna();return {'n':len(x),'zero_n':int((x==0).sum()),'min':float(x.min()) if len(x) else None,'p01':float(x.quantile(.01)) if len(x) else None,'p50':float(x.median()) if len(x) else None,'p99':float(x.quantile(.99)) if len(x) else None,'max':float(x.max()) if len(x) else None}
def split_qc(z,col):
 x=z.loc[z[col].notna()].sort_values(KD).copy();x['_order']=x.groupby(K).cumcount();x['_half']=np.where(x._order%2==0,'odd','even');g=x.groupby(K+['_half'])[col].agg(['mean','count']).unstack('_half');p=g.loc[(g['count'].sum(axis=1)>=7)&g['mean'].notna().all(axis=1)]
 rho=float(p['mean']['odd'].rank().corr(p['mean']['even'].rank())) if len(p)>2 else None
 if rho is not None and not np.isfinite(rho):rho=None
 means=x.groupby(K)[col].mean();within=x[col]-x.groupby(K)[col].transform('mean')
 return {'n_visits_ge7_both_halves':len(p),'odd_even_spearman':rho,'pooled_within_person_SD':float(np.sqrt((within**2).sum()/(len(x)-len(means)))) if len(x)>len(means) else None,'SD_person_visit_means':float(means.std()) if len(means)>1 else None,'interpretation':'descriptive observed-period repeatability; not long-term trait stability, not test-set evaluation'}
counts={}
for scope,d,w in [('all_source',D,W),('baseline_10k',Z,ZW)]:
 values={}
 for label,col in [('indexed','date'),('energy','energy_kcal'),('UPF_zeroU','UPF_point_zero_unknown_energy_pp'),('UPF_strict','UPF_point_all_rows_classified_pp'),('fiber','fiber_density_g_per_1000kcal')]:
  m=d[col].notna();g=d.loc[m].groupby(K).size();values[label]={'days':int(m.sum()),'participant_visits_ge1':len(g),'participants_ge1':int(d.loc[m,'participant_id'].nunique()),'participant_visits_ge7_DIAGNOSTIC':int((g>=7).sum())}
 en=d.loc[d.energy_eligible];values['NOVA_coverage_energyweighted']=float(1-en.unknown_NOVA_kcal.sum()/en.energy_kcal.sum());values['bound_width_pp']=describe(en.UPF_bound_width_pp);values['full_energy_window_exact_point_visits']=int(w.UPF_exact_full_energy_window_cumulative_ratio.notna().sum());counts[scope]=values
 assert len(d)==int(w.n_recorded_days.sum())
summary={'mapping_version':version,'catalog_sha256':cv,'legacy_code_sha256':digest(legacy),'source_input':inputfp,'all_FoodIDs':len(cross),'known_NOVA_FoodIDs':len(lookup),'unknown_NOVA_FoodIDs':len(cross)-len(lookup),'multiple_short_name_IDs':int((cross.n_short_names>1).sum()),'multiple_category_IDs':int((cross.n_categories>1).sum()),'event_rows_scanned_for_dictionary':n if False else 2640959,'admitted_events_scanned':n,'missing_foodID_events':missingid,'event_name_ID_NOVA_conflicts':nameconflict,'counts':counts,'daily_distribution_baseline':{col:describe(Z[col]) for col in ['energy_kcal','UPF_point_zero_unknown_energy_pp','UPF_point_all_rows_classified_pp','UPF_bound_width_pp','fiber_density_g_per_1000kcal']},'window_distribution_baseline':{col:describe(ZW[col]) for col in ['UPF_zeroU_equal_day_mean','UPF_zeroU_cumulative_ratio','fiber_equal_day_mean','fiber_cumulative_ratio']},'observed_repeatability_baseline':{col:split_qc(Z,col) for col in ['UPF_point_zero_unknown_energy_pp','UPF_point_all_rows_classified_pp','fiber_density_g_per_1000kcal']},'CGM_common_status':'waiting_for_W09_point_QC; real shared-day data not created','W08_status':'diet_only_views_built_common_branch_pending_W09','minimum_days_frozen':False,'primary_aggregation_frozen':False,'formal_exposures_frozen':False,'new_external_mapping_used':False,'manual_current_product_NOVA_review_completed':False,'I_MEDAS_production_built':False,'source_explanation_categories_not_automatically_mapped':True,'outcomes_read':False,'synthetic_checks_passed':6,'elapsed_sec':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'packages':{'python':platform.python_version(),'pandas':pd.__version__,'pyarrow':pa.__version__}}
own_tree=ast.parse(Path(__file__).read_text());lib_functions=[x for x in own_tree.body if isinstance(x,ast.FunctionDef) and x.name in ['window','common_view']]
lib='import numpy as np\nimport pandas as pd\nK='+repr(K)+'\nKD=K+["date"]\n'+ '\n\n'.join(ast.unparse(x) for x in lib_functions)+'\n'
libpath=Path(__file__).parent/'common_day_adapter.py';assert not libpath.exists();libpath.write_text(lib);compile(lib,str(libpath),'exec')
print('W07_W08_COUNTS',json.dumps(counts),flush=True);print('W08_REPEATABILITY',json.dumps(summary['observed_repeatability_baseline']),flush=True)
# Save unknown product priority inside TRE for human adjudication, never fill from product-name guesses.
unknownrows=[{'food_id':k[0],'short_name':k[1],'product_name':k[2],'events':v[0],'known_energy_kcal':v[1]} for k,v in unknown.items()];unknownrows.sort(key=lambda x:x['known_energy_kcal'],reverse=True)
(Q/'unknown_products_for_review_TRE_ONLY.json').write_text(json.dumps(unknownrows,indent=2))
(Q/'source_category_inventory_TRE_ONLY.json').write_text(json.dumps(categories,indent=2))
versioned=B/'food_id_crosswalk_versioned.parquet';coverage=B/'nova_coverage_TRE_ONLY.parquet';daily=B/'food_daily_TRE_ONLY.parquet';windows=B/'food_windows_TRE_ONLY.parquet'
for p,d in [(versioned,cross),(coverage,D[KD+['event_rows','energy_kcal','known_UPF_kcal','unknown_NOVA_kcal','unknown_NOVA_rows','NOVA_energy_coverage','UPF_lower_pp','UPF_upper_pp','UPF_bound_width_pp','mapping_version']]),(daily,D),(windows,W)]:
 pq.write_table(pa.Table.from_pandas(d,preserve_index=False),p,compression='zstd');assert pq.ParquetFile(p).metadata.num_rows==len(d)
audit=cross.groupby('classification_status',dropna=False).agg(FoodIDs=('food_id','size')).reset_index();audit.to_csv(B/'nova_mapping_audit.csv',index=False)
config={'version':version,'catalog_size':643,'catalog_sha256':cv,'source_legacy_script_sha256':digest(legacy),'mapping':'FoodID -> all observed nonmissing short names must share one valid exact inherited legacy classification; trim/casefold only; no fuzzy match','NOVA_blank_names':['Stevia Sweetner','Spirulina'],'unknown':'missing/unmatched/conflicting -> NA; never non-UPF','denominator':'all retained event energy; complete finite/nonnegative, daily total>0','point_zero_unknown_energy':'100*K/E only if U==0; measured-zero energy unknown rows allowed algebraically, named separately','point_strict':'additionally require zero unknown event rows','bounds':'[100*K/E,100*(K+U)/E]','units_UPF':'percentage points 0..100','fiber':'1000*complete fiber_g/complete energy_kcal','window':'feature-specific valid days; equal-day mean and same-day cumulative ratio separate; full-energy-window exact point only if all energy-day U sums zero','minimum_days':None,'outcome_free':True,'CGM_common':'requires W09 verified unique eligible day table; no CGM dependency in food_window','I_MEDAS':'not manufactured from unresolved proxies','source_categories':'requires separately verified legacy category crosswalk; not guessed from NOVA','formal_primary_exposure_choice':None}
(B/'diet_views_rules.json').write_text(json.dumps(config,indent=2))
# Keep statistical diagnostics readable in TRE without exporting individual features.
html='<html><meta charset="utf-8"><title>W07/W08 diet QC</title><h1>W07/W08 diet-only QC</h1><p>Outcome blind; CGM common days wait for W09. Candidate definitions, not formal exposure freeze.</p><pre>'+json.dumps(summary,indent=2)+'</pre>'+audit.to_html(index=False)+'</html>'
(B/'diet_candidate_QC.html').write_text(html)
artifacts=[(versioned,P/'config'/versioned.name),(coverage,P/'analysis/shared'/coverage.name),(daily,P/'analysis/shared'/daily.name),(windows,P/'analysis/shared'/windows.name),(B/'nova_mapping_audit.csv',P/'data_qc/nova_mapping_audit.csv'),(B/'diet_views_rules.json',P/'config/diet_views_rules_W07_W08_20261028.json'),(B/'diet_candidate_QC.html',P/'data_qc/diet_candidate_QC.html')]
mirrors=[]
for src,dst in artifacts:
 assert not dst.exists(),str(dst);dst.parent.mkdir(parents=True,exist_ok=True)
 with src.open('rb') as f,dst.open('wb') as g:shutil.copyfileobj(f,g,1048576)
 time.sleep(1);assert src.stat().st_size==dst.stat().st_size and digest(src)==digest(dst)
 rows=pq.ParquetFile(src).metadata.num_rows if src.suffix=='.parquet' else None
 if rows is not None:assert pq.ParquetFile(dst).metadata.num_rows==rows
 mirrors.append({'runtime':str(src),'cloud':str(dst),'bytes':src.stat().st_size,'sha256':digest(src),'rows':rows,'verified':True})
assert source.stat().st_size==inputfp['bytes'] and source.stat().st_mtime_ns==inputfp['mtime_ns'];summary['artifacts']=mirrors
summary=json.loads(json.dumps(summary));(Q/'summary_TRE_ONLY.json').write_text(json.dumps(summary,indent=2));assert json.loads((Q/'summary_TRE_ONLY.json').read_text())==summary
M=P/'memos/w07_w08_diet_views_20261028';M.mkdir(parents=True,exist_ok=True)
(M/'MEMO.md').write_text('# W07 and W08 diet views, 2026-10-28\n\nCode: '+str(Path(__file__))+'\n\nResult: '+str(Q)+'\n\nFunction: versioned inherited FoodID/NOVA crosswalk with unknowns; food-only daily and feature-specific windows; named day means/cumulative ratios/bounds; descriptive observed-period repeatability. No outcome reading.\n\nW08 CGM shared-day branch waits for W09 verified point QC. No formal aggregation, minimum-day or primary exposure choice frozen. Legacy category transfer does not constitute new product-level NOVA adjudication. Optional category sources and I-MEDAS not guessed.\n\nSummary:\n```json\n'+json.dumps(summary,indent=2)+'\n```\n\nActual exit and independent verification must be checked before acceptance.\n')
print('W07_W08_DIET_VIEWS_BUILT','CROSSWALK',len(cross),'DAILY',len(D),'WINDOWS',len(W),'MIRRORS',len(mirrors),'PEAK_RSS_KiB',summary['peak_rss_kib'],flush=True)
