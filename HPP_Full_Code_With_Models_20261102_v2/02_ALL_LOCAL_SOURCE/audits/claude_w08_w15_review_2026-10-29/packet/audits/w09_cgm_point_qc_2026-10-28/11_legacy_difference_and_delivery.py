# -*- coding: utf-8 -*-
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,hashlib,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');Q=P/'data_qc/w09_cgm_point_qc_20261028';L=Path('/home/ec2-user/multisystem_runtime_20261021/w09_cgm_point_qc_20261028');K=['participant_id','cohort','research_stage'];KC=K+['connection_id'];KD=K+['date']
assert (L/'07_actual_exit_code').read_text().strip()=='0'
def load(p):return pq.ParquetFile(p).read(use_threads=False).to_pandas(ignore_metadata=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
R=load(Q/'cgm_daily_reconciliation_TRE_ONLY.parquet');C=load(P/'data_qc/cgm_point_connection_QC.parquet');both=R.loc[R['_merge']=='both'];finite=both.glucose_mean_mg_dL.notna()&both.cgm_mean.notna();assert np.allclose(both.loc[finite,'glucose_mean_mg_dL'],both.loc[finite,'cgm_mean'],atol=1e-9,rtol=0)
E=R.loc[R['_merge']=='left_only'].merge(C[KC+['first_observed','last_observed','device']],on=KC,validate='many_to_one');E['first_day']=E.date.eq(E.first_observed.dt.strftime('%Y-%m-%d'));E['last_day']=E.date.eq(E.last_observed.dt.strftime('%Y-%m-%d'))
S=R.loc[R.selected_connection.eq(True)].copy();S['legacy_daily_present']=S['_merge'].eq('both');assert not S.duplicated(KD).any();Z=S.loc[(S.cohort=='10k')&(S.research_stage=='00_00_visit')&S.cgm_day_eligible.eq(True)]
flag=S[KD+['connection_id','legacy_daily_present','cgm_day_eligible','n_points']];src=L/'legacy_daily_presence_selected_TRE_ONLY.parquet';dst=Q/src.name;assert not src.exists() and not dst.exists();pq.write_table(pa.Table.from_pandas(flag,preserve_index=False),src,compression='zstd');shutil.copyfile(src,dst);assert sha(src)==sha(dst) and pq.ParquetFile(dst).metadata.num_rows==len(flag)
def sc(d):
 g=d.groupby(K).size();return {'people':int(d.participant_id.nunique()),'days':len(d),'ge7_DIAGNOSTIC':int(g.ge(7).sum())}
J=load(P/'analysis/shared/food_cgm_common_daily_TRE_ONLY.parquet');J=J.merge(flag[KD+['legacy_daily_present']],on=KD,validate='one_to_one');J=J.loc[(J.cohort=='10k')&(J.research_stage=='00_00_visit')]
s={'verified':True,'all_legacy_days':len(both),'both_mean_available':int(finite.sum()),'legacy_reconstructed_mean_missing':int((~finite).sum()),'extra_reconstructed_days':len(E),'extra_no_points':int(E.n_points.eq(0).sum()),'extra_positive_points':int(E.n_points.gt(0).sum()),'extra_qualified':int(E.cgm_day_eligible.eq(True).sum()),'extra_first_days':int(E.first_day.sum()),'extra_last_days':int(E.last_day.sum()),'extra_interior_days':int((~E.first_day&~E.last_day).sum()),'extra_positive_point_counts':E.loc[E.n_points.gt(0),'n_points'].value_counts().sort_index().to_dict(),'extra_device_counts':E.device.value_counts().to_dict(),'baseline_CGM_qualified_legacy_present':sc(Z.loc[Z.legacy_daily_present]),'baseline_common_indexed_only_legacy_days':sc(J.loc[J.legacy_daily_present]),'baseline_common_energy_only_legacy_days':sc(J.loc[J.legacy_daily_present&J.energy_eligible]),'baseline_common_fiber_only_legacy_days':sc(J.loc[J.legacy_daily_present&J.fiber_density_g_per_1000kcal.notna()]),'upstream_omission_reason':'not supplied in mounted metadata; do not infer that no glucose was observed','auxiliary_presence_table':str(dst),'auxiliary_rows':len(flag),'auxiliary_sha256':sha(dst),'primary_shared_tables_unchanged':True};(Q/'legacy_difference_audit.json').write_text(json.dumps(s,indent=2))
M=P/'memos/w09_cgm_point_qc_20261028/MEMO.md';text='\n\n## Additional legacy-daily difference audit (script11)\n\nNo original or primary shared table changed. The exact source-presence flags permit sensitivity restricted to legacy-listed days. Upstream omission reason remains unreported, not invented.\n```json\n'+json.dumps(s,indent=2)+'\n```\n';M.write_text(M.read_text()+text);assert text in M.read_text()
print('LEGACY_DIFF_VERIFIED',json.dumps(s),flush=True)
for line in (L/'08.log').read_text().splitlines():
 if line.startswith(('FLOW','VERIFY','SHARED_ROWS','CONNECTION_QC')):print(line)
summary=json.loads((Q/'summary_TRE_ONLY.json').read_text())
for m in summary['multi_connection_aggregate']:
 print('MULTI_SHORT',m['case_ordinal'],'overlap_h',m['overlap_hours'],[(x['selected'],x['qualified_food_common_days'],x['qualified_CGM_days'],x['observed_calendar_days']) for x in m['candidates']])
print('FINAL_DELIVERY_AUDIT_DONE',flush=True)
