# -*- coding: utf-8 -*-
"""Independent readback and algebra/source checks; no outcome association."""
import os
for x in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:os.environ[x]='1'
import sys,json,hashlib,time,resource,ast,math,importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');L=Path('/home/ec2-user/multisystem_runtime_20261021/w07_w08_diet_views_20261028');Q=P/'data_qc/w07_w08_diet_views_20261028'
assert sys.executable=='/home/ec2-user/miniconda/envs/pheno/bin/python'
assert (L/'01_actual_exit_code').read_text().strip()=='0'
s=json.loads((Q/'summary_TRE_ONLY.json').read_text());K=['participant_id','cohort','research_stage'];KD=K+['date']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
for x in s['artifacts']:
 a=Path(x['runtime']);b=Path(x['cloud']);assert a.stat().st_size==b.stat().st_size==x['bytes'];assert sha(a)==sha(b)==x['sha256']
 if x['rows'] is not None:assert pq.ParquetFile(a).metadata.num_rows==pq.ParquetFile(b).metadata.num_rows==x['rows']
C=pq.ParquetFile(P/'config/food_id_crosswalk_versioned.parquet').read().to_pandas();D=pq.ParquetFile(P/'analysis/shared/food_daily_TRE_ONLY.parquet').read().to_pandas();W=pq.ParquetFile(P/'analysis/shared/food_windows_TRE_ONLY.parquet').read().to_pandas();N=pq.ParquetFile(P/'analysis/shared/nova_coverage_TRE_ONLY.parquet').read().to_pandas()
assert len(C)==s['all_FoodIDs'] and C.food_id.is_unique and not D.duplicated(KD).any() and not W.duplicated(K).any() and not N.duplicated(KD).any()
old=P/'scripts/w04_mapping_review_20261027/03_audit_event_mapping_days.py';tree=ast.parse(old.read_text());cat=next(ast.literal_eval(x.value) for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CATALOG' for t in x.targets));norm=lambda x:str(x).strip().casefold();mp={norm(a):b for a,b in cat}
for row in C.itertuples():
 names=json.loads(row.short_names_json);vals={mp.get(norm(a)) for a in names};valid=bool(names) and len(vals)==1 and next(iter(vals)) in [1,2,3,4]
 assert valid==pd.notna(row.NOVA)
 if valid:assert row.NOVA==next(iter(vals))
assert not C.manual_current_product_NOVA_review_completed.any()
E=D.energy_kcal;lo=100*D.known_UPF_kcal/E;hi=100*(D.known_UPF_kcal+D.unknown_NOVA_kcal)/E
assert np.allclose(lo,D.UPF_lower_pp,equal_nan=True) and np.allclose(hi,D.UPF_upper_pp,equal_nan=True)
assert np.allclose(D.fiber_density_g_per_1000kcal,1000*D.fiber_g/E,equal_nan=True)
assert D.UPF_point_zero_unknown_energy_pp.notna().equals(D.energy_eligible&(D.unknown_NOVA_kcal==0));assert D.UPF_point_all_rows_classified_pp.notna().equals(D.energy_eligible&(D.unknown_NOVA_rows==0))
assert np.allclose(D.NOVA_energy_coverage,1-D.unknown_NOVA_kcal/E,equal_nan=True)
assert np.allclose(N.UPF_lower_pp,D.UPF_lower_pp,equal_nan=True);assert N[KD].equals(D[KD])
Q6=pq.ParquetFile(P/'analysis/shared/food_day_qc_TRE_ONLY.parquet').read(columns=KD+['primary_day','n_events','energy_eligible','fiber_eligible','calories_kcal_complete_sum','dietary_fiber_g_complete_sum']).to_pandas();Q6=Q6.loc[Q6.primary_day]
T=D.merge(Q6,on=KD,validate='one_to_one',suffixes=('','_W06'));assert len(T)==len(Q6)==149912
assert (T.event_rows==T.n_events_W06).all() if 'n_events_W06' in T else (T.event_rows==T.n_events).all()
assert np.allclose(T.energy_kcal,T.calories_kcal_complete_sum.where(T.energy_eligible),equal_nan=True)
assert np.allclose(T.fiber_g,T.dietary_fiber_g_complete_sum.where(T.fiber_eligible),equal_nan=True)
WW=W.set_index(K)
for label,col,num,scale in [('UPF_zeroU','UPF_point_zero_unknown_energy_pp','known_UPF_kcal',100),('UPF_strict','UPF_point_all_rows_classified_pp','known_UPF_kcal',100),('fiber','fiber_density_g_per_1000kcal','fiber_g',1000)]:
 z=D.loc[D[col].notna()];g=z.groupby(K).agg(n=(col,'size'),avg=(col,'mean'),numerator=(num,'sum'),energy=('energy_kcal','sum'));v=WW.loc[g.index]
 assert np.array_equal(g.n,v['n_'+label+'_days']);assert np.allclose(g.avg,v[label+'_equal_day_mean']);assert np.allclose(scale*g.numerator/g.energy,v[label+'_cumulative_ratio'])
 assert int(WW['n_'+label+'_days'].sum())==len(z)
full=D.loc[D.energy_eligible].groupby(K).agg(E=('energy_kcal','sum'),U=('unknown_NOVA_kcal','sum'),K=('known_UPF_kcal','sum'));v=WW.loc[full.index]
assert np.allclose(v.UPF_lower_cumulative_all_energy_days,100*full.K/full.E);assert np.allclose(v.UPF_upper_cumulative_all_energy_days,100*(full.K+full.U)/full.E)
assert np.array_equal(v.UPF_exact_full_energy_window_cumulative_ratio.notna(),full.U.eq(0))
# Source reread: collect 3 deterministic baseline days, then independently sum exact FoodID labels in original admitted events.
picks=D.loc[(D.cohort=='10k')&(D.research_stage=='00_00_visit')&D.energy_eligible].copy();targets=[]
for condition in [picks.unknown_NOVA_kcal>0,picks.UPF_point_all_rows_classified_pp.notna(),picks.fiber_eligible]:
 a=picks.loc[condition].iloc[0];targets.append(tuple(a[k] for k in KD))
lookup=dict(zip(C.food_id,C.NOVA));vals={k:[0,0.,0.,0] for k in targets};scanned=0
for rb in pq.ParquetFile(P/'analysis/shared/food_events_clean_TRE_ONLY.parquet').iter_batches(batch_size=8192,columns=KD+['food_id','calories_kcal_clean']):
 d=rb.to_pandas();scanned+=len(d)
 for k in vals:
  mask=np.ones(len(d),bool)
  for name,value in zip(KD,k):mask &= d[name].eq(value).to_numpy()
  z=d.loc[mask];nov=z.food_id.map(lookup);e=z.calories_kcal_clean
  a=vals[k];a[0]+=len(z);a[1]+=float(e[nov.eq(4).fillna(False)].sum());a[2]+=float(e[nov.isna()].sum());a[3]+=int(nov.isna().sum())
assert scanned==2420902
for key,a in vals.items():
 row=D.set_index(KD).loc[key];assert a[0]==row.event_rows and a[3]==row.unknown_NOVA_rows;assert math.isclose(a[1],row.known_UPF_kcal,rel_tol=1e-10,abs_tol=1e-7) and math.isclose(a[2],row.unknown_NOVA_kcal,rel_tol=1e-10,abs_tol=1e-7)
# Confirm generated common-day helper imports without running the build and genuinely recomputes common windows.
p=P/'scripts/w07_w08_diet_views_20261028/common_day_adapter.py';spec=importlib.util.spec_from_file_location('common_adapter',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
base={K[0]:-1,K[1]:'synthetic',K[2]:'visit','mapping_version':'synthetic','energy_eligible':True,'known_UPF_kcal':800.,'unknown_NOVA_kcal':0.,'fiber_g':10.,'UPF_lower_pp':80.,'UPF_upper_pp':80.,'UPF_point_zero_unknown_energy_pp':80.,'UPF_point_all_rows_classified_pp':80.,'fiber_density_g_per_1000kcal':10.,'energy_kcal':1000.}
a=dict(base,date='2026-01-01');b=dict(base,date='2026-01-02',known_UPF_kcal=300.,energy_kcal=3000.,UPF_lower_pp=10.,UPF_upper_pp=10.,UPF_point_zero_unknown_energy_pp=10.,UPF_point_all_rows_classified_pp=10.,fiber_density_g_per_1000kcal=10/3)
synthetic=pd.DataFrame([a,b]);fullwin=m.window(synthetic).iloc[0];assert fullwin.UPF_zeroU_equal_day_mean==45 and fullwin.UPF_zeroU_cumulative_ratio==27.5
cg=pd.DataFrame([dict({k:a[k] for k in KD},cgm_day_eligible=True,point_qc_verified=True)]);subset=m.common_view(synthetic,cg);single=m.window(subset).iloc[0];assert len(subset)==1 and single.UPF_zeroU_equal_day_mean==80 and single.UPF_zeroU_cumulative_ratio==80
assert s['CGM_common_status'].startswith('waiting_for_W09') and s['outcomes_read'] is False
result={'verified':True,'W07_crosswalk_with_unknowns_built':True,'W08_diet_only_views_verified':True,'W08_CGM_common_real_data_completed':False,'W09_dependency':'point QC pending','crosswalk_rows':len(C),'daily_rows':len(D),'window_rows':len(W),'mirrors_verified':len(s['artifacts']),'source_days_independently_verified':len(vals),'admitted_source_rows_reread':scanned,'synthetic_common_window_recomputed':True,'primary_exposures_frozen':False,'outcomes_read':False,'human_review_done':False,'code_sha256':{p.name:sha(p) for p in sorted((P/'scripts/w07_w08_diet_views_20261028').glob('*.py'))}}
(Q/'verification.json').write_text(json.dumps(result,indent=2));assert json.loads((Q/'verification.json').read_text())==result
M=P/'memos/w07_w08_diet_views_20261028/MEMO.md';M.write_text(M.read_text()+'\n## Independent verification\n'+json.dumps(result,indent=2)+'\nVerifier actual exit must be checked in 02_actual_exit_code.\n')
print('W07_W08_INDEPENDENT_PASS',json.dumps(result),flush=True)
