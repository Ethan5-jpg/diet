# -*- coding: utf-8 -*-
"""Real W09-qualified calendar-day W08 views; recompute dietary windows on that subset."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,hashlib,shutil,time,importlib.util,resource
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');Q=P/'data_qc/w09_cgm_point_qc_20261028';L=Path('/home/ec2-user/multisystem_runtime_20261021/w09_cgm_point_qc_20261028');B=L/'w08_common_artifacts';B.mkdir(exist_ok=False);K=['participant_id','cohort','research_stage'];KD=K+['date']
assert (L/'03_actual_exit_code').read_text().strip()=='0' and (L/'04_actual_exit_code').read_text().strip()=='0' and json.loads((Q/'verification.json').read_text())['verified']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return pq.ParquetFile(p).read(use_threads=False).to_pandas(ignore_metadata=True)
old=P/'analysis/shared/food_windows_TRE_ONLY.parquet';oldsha=sha(old)
D=load(P/'analysis/shared/food_daily_TRE_ONLY.parquet');C=load(P/'analysis/shared/cgm_daily_TRE_ONLY.parquet');assert not D.duplicated(KD).any() and not C.duplicated(KD).any()
lib=P/'scripts/w07_w08_diet_views_20261028/common_day_adapter.py';spec=importlib.util.spec_from_file_location('w08common',lib);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
J=module.common_view(D,C);assert len(J)>0
fields=['connection_id','n_points','day_length_seconds','coverage_fraction','max_uncovered_gap_seconds','glucose_mean_mg_dL','glucose_sd_mg_dL','glucose_CV_pct','unit','timezone','qc_rule_status']
J=J.merge(C[KD+fields],on=KD,validate='one_to_one');J=J.sort_values(KD).reset_index(drop=True);W=module.window(J);W['window_id']='food_cgm_common';W['cgm_rule']='W09 operational candidate coverage80pct/maxgap2h';W['alignment_scope']='true local calendar day; actual food ingestion time not independently verified'
assert int(W.n_recorded_days.sum())==len(J) and not W.duplicated(K).any()
def sc(d):
 g=d.groupby(K).size();return {'days':len(d),'people':int(d.participant_id.nunique()),'people_visits':len(g),'visits_ge7_DIAGNOSTIC':int((g>=7).sum()),'median_days':float(g.median()) if len(g) else None}
counts={}
for name,z,w in [('all_source',J,W),('baseline_10k',J.loc[(J.cohort=='10k')&(J.research_stage=='00_00_visit')],W.loc[(W.cohort=='10k')&(W.research_stage=='00_00_visit')])]:
 counts[name]={label:sc(z.loc[z[col].notna()]) for label,col in [('indexed_common','date'),('energy_common','energy_kcal'),('fiber_common','fiber_density_g_per_1000kcal'),('UPF_zeroU_common','UPF_point_zero_unknown_energy_pp'),('UPF_strict_common','UPF_point_all_rows_classified_pp')]};counts[name]['full_energy_common_window_exact_UPF_people_visits']=int(w.UPF_exact_full_energy_window_equal_day_mean.notna().sum())
# Denominator changes are provenance, never reuse all-food-window numerators on common subset.
oldw=load(old);cmp=W.merge(oldw[K+['n_recorded_days','fiber_equal_day_mean','UPF_zeroU_equal_day_mean']],on=K,validate='one_to_one',suffixes=('_common','_food'))
changed={'windows_compared':len(cmp),'different_calendar_day_counts':int((cmp.n_recorded_days_common!=cmp.n_recorded_days_food).sum()),'different_fiber_means':int((cmp.fiber_equal_day_mean_common.sub(cmp.fiber_equal_day_mean_food).abs()>1e-10).sum()),'different_UPF_zeroU_means':int((cmp.UPF_zeroU_equal_day_mean_common.sub(cmp.UPF_zeroU_equal_day_mean_food).abs()>1e-10).sum())}
art=[]
for name,d in [('food_cgm_common_daily_TRE_ONLY.parquet',J),('food_cgm_common_windows_TRE_ONLY.parquet',W)]:
 src=B/name;dst=P/'analysis/shared'/name;assert not dst.exists();pq.write_table(pa.Table.from_pandas(d,preserve_index=False),src,compression='zstd');shutil.copyfile(src,dst);time.sleep(.3);assert sha(src)==sha(dst) and pq.ParquetFile(dst).metadata.num_rows==len(d);art.append({'runtime':str(src),'cloud':str(dst),'rows':len(d),'sha256':sha(src),'bytes':src.stat().st_size})
assert sha(old)==oldsha
# Exposure-only joint distribution and observed-period repeatability on actual common dates.
Z=J.loc[(J.cohort=='10k')&(J.research_stage=='00_00_visit')].copy();repeat={}
for col in ['fiber_density_g_per_1000kcal','UPF_point_zero_unknown_energy_pp','UPF_point_all_rows_classified_pp']:
 x=Z.loc[Z[col].notna()].sort_values(KD).copy();x['_n']=x.groupby(K)[col].transform('size');x['_order']=x.groupby(K).cumcount();x['_half']=x._order%2;g=x.groupby(K+['_half'])[col].agg(['mean','count']).unstack('_half');valid=g.loc[(g['count'].sum(axis=1)>=7)&g['mean'].notna().all(axis=1)];rho=float(valid['mean'][0].rank().corr(valid['mean'][1].rank())) if len(valid)>2 else None
 means=x.groupby(K)[col].mean();within=x[col]-x.groupby(K)[col].transform('mean');repeat[col]={'n_visits_ge7':len(valid),'odd_even_spearman':rho,'within_person_SD':float(np.sqrt((within**2).sum()/(len(x)-len(means)))) if len(x)>len(means) else None,'between_person_SD':float(means.std()),'valid_day_count_bins_visits':pd.cut(x.groupby(K).size(),[0,3,6,9,float('inf')],labels=['1-3','4-6','7-9','10+']).value_counts(sort=False).to_dict(),'limitation':'observed-period descriptive; not long-term trait stability or discovery/confirmation validation'}
joint=Z[['fiber_density_g_per_1000kcal','UPF_point_zero_unknown_energy_pp']].dropna();jointqc={'both_feature_days':len(joint),'spearman_daily':float(joint.iloc[:,0].rank().corr(joint.iloc[:,1].rank())) if len(joint)>2 else None,'not_diet_outcome_association':True}
s={'W08_real_common_built':True,'W09_verified':True,'counts':counts,'common_exposure_repeatability':repeat,'common_joint_exposure_QC':jointqc,'window_recalculation_check':changed,'artifacts':art,'food_only_window_sha_unchanged':oldsha,'source_adapter_sha':sha(lib),'CGM_candidate_QC_rule':'80% observed slot support, max unsupported gap2h; sensitivity grid in W09','minimum_days_frozen':False,'primary_aggregation_frozen':False,'actual_food_ingestion_time_validated':False,'microbiome_or_body_liver_joined':False,'not_final_model_N':True};(Q/'w08_common_summary_TRE_ONLY.json').write_text(json.dumps(s,indent=2));print('W08_REAL_COMMON_BUILT',json.dumps(s),flush=True)
