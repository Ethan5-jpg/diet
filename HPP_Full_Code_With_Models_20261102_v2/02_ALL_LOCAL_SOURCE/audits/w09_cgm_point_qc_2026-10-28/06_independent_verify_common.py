# -*- coding: utf-8 -*-
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');Q=P/'data_qc/w09_cgm_point_qc_20261028';L=Path('/home/ec2-user/multisystem_runtime_20261021/w09_cgm_point_qc_20261028');K=['participant_id','cohort','research_stage'];KD=K+['date']
assert (L/'05_actual_exit_code').read_text().strip()=='0'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return pq.ParquetFile(p).read(use_threads=False).to_pandas(ignore_metadata=True)
s=json.loads((Q/'w08_common_summary_TRE_ONLY.json').read_text());D=load(P/'analysis/shared/food_daily_TRE_ONLY.parquet');C=load(P/'analysis/shared/cgm_daily_TRE_ONLY.parquet');J=load(P/'analysis/shared/food_cgm_common_daily_TRE_ONLY.parquet');W=load(P/'analysis/shared/food_cgm_common_windows_TRE_ONLY.parquet')
assert not J.duplicated(KD).any() and not W.duplicated(K).any() and C.point_qc_verified.all();expected=D.merge(C.loc[C.cgm_day_eligible,KD],on=KD,validate='one_to_one');assert set(expected[KD].itertuples(index=False,name=None))==set(J[KD].itertuples(index=False,name=None))
assert len(J)==int(W.n_recorded_days.sum()) and set(W.window_id)=={'food_cgm_common'}
for col in ['energy_kcal','fiber_density_g_per_1000kcal','UPF_point_zero_unknown_energy_pp','UPF_lower_pp','UPF_upper_pp']:
 t=expected[KD+[col]].merge(J[KD+[col]],on=KD,validate='one_to_one',suffixes=('_a','_b'));assert np.allclose(t[col+'_a'],t[col+'_b'],equal_nan=True,rtol=0,atol=1e-10)
for label,col,num,mult in [('fiber','fiber_density_g_per_1000kcal','fiber_g',1000),('UPF_zeroU','UPF_point_zero_unknown_energy_pp','known_UPF_kcal',100),('UPF_strict','UPF_point_all_rows_classified_pp','known_UPF_kcal',100)]:
 z=J.loc[J[col].notna()];g=z.groupby(K).agg(mean=(col,'mean'),n=(col,'size'),num=(num,'sum'),den=('energy_kcal','sum'));t=W.merge(g,on=K,how='left',validate='one_to_one');assert np.allclose(t[label+'_equal_day_mean'],t['mean'],equal_nan=True,rtol=0,atol=1e-10);assert np.allclose(t[label+'_cumulative_ratio'],mult*t.num/t.den,equal_nan=True,rtol=0,atol=1e-10);assert np.array_equal(t['n_'+label+'_days'],t.n.fillna(0))
for x in s['artifacts']:assert sha(x['runtime'])==sha(x['cloud'])==x['sha256'] and pq.ParquetFile(x['cloud']).metadata.num_rows==x['rows']
assert sha(P/'analysis/shared/food_windows_TRE_ONLY.parquet')==s['food_only_window_sha_unchanged']
v={'verified':True,'W09_point_QC_verified':True,'W08_true_common_day_branch_verified':True,'exact_unique_common_keyset_verified':True,'all_common_feature_values_and_window_formulas_verified':True,'food_only_window_unchanged':True,'code_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},'build_05_actual_exit':0,'minimum_days_or_primary_exposure_frozen':False};(Q/'w08_common_verification.json').write_text(json.dumps(v,indent=2));print('W08_COMMON_INDEPENDENT_PASS',json.dumps(v),flush=True)
