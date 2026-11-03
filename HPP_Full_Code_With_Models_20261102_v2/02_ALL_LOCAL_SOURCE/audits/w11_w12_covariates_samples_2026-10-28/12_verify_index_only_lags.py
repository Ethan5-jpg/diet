# -*- coding: utf-8 -*-
"""Complete independent lag coverage for index-only records (not model candidates)."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
from collections import defaultdict
from datetime import date
import json,hashlib
import pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');Q=P/'data_qc/w11_w12_covariates_samples_20261028';L=Path('/home/ec2-user/multisystem_runtime_20261021/w11_w12_covariates_samples_20261028');K=['participant_id','cohort','research_stage'];assert (L/'11_actual_exit_code').read_text().strip()=='0'
def read(p,cols=None):return pq.ParquetFile(p).read(columns=cols,use_threads=False).to_pylist()
def key(r):return tuple(r[k] for k in K) if all(r[k] is not None for k in K) else None
C=[r for r in read(P/'analysis/shared/sample_candidates_TRE_ONLY.parquet') if r['candidate_kind']=='index_only'];assert len(C)==418;maps={}
for label,path,flag in [('food',P/'analysis/shared/food_daily_TRE_ONLY.parquet',None),('CGM',P/'analysis/shared/cgm_daily_TRE_ONLY.parquet','cgm_day_eligible')]:
 d=defaultdict(set)
 for r in read(path,K+['date']+([flag] if flag else [])):
  if key(r) and (flag is None or r[flag] is True):d[key(r)].add(date.fromisoformat(r['date']))
 maps[label]=d
checked=0
for c in C:
 s=date.fromisoformat(c['descriptive_sample_calendar_date']) if c['descriptive_sample_calendar_date'] else None
 for label,column in [('food','sample_minus_nearest_observed_food_day'),('CGM','sample_minus_nearest_qualified_CGM_day')]:
  days=maps[label].get(key(c),set());expected=(s-min(days,key=lambda x:(abs((s-x).days),x))).days if s and days else None;assert c[column]==expected;checked+=1
v={'passed':True,'index_only_records':len(C),'additional_lag_fields_checked':checked,'abundance_candidate_lags_verified_in_script10':20129,'all_20547_candidate_rows_now_covered_by_independent_lag_checks':True,'source_rows_or_selection_changed':False,'finalizer11_actual_exit_verified':0};p=Q/'index_only_lag_verification.json';assert not p.exists();p.write_text(json.dumps(v,indent=2));memo=P/'memos/w11_w12_covariates_samples_20261028/VALIDATION_SUPPLEMENT.md';assert not memo.exists();memo.write_text('# Script12 independent index-only lag completion\n\nScript10 independently checked lags for every abundance-row candidate; this script also checks the418 index-only records (836 lag fields). Together all20547 candidate rows have been covered. No dataset, rule or selection changed. Script11 actual exit0 verified.\n\n'+json.dumps(v,indent=2)+'\n');assert json.loads(p.read_text())['passed'];print('W11_W12_FULL_LAG_COVERAGE',json.dumps(v),flush=True)
