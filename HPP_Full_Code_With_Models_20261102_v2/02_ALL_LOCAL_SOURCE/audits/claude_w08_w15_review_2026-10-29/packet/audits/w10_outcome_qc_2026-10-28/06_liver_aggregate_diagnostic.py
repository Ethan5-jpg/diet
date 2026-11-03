# -*- coding: utf-8 -*-
"""Diagnostic only: quantify source-summary mismatch; selected mean unchanged."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,math,hashlib
import pandas as pd,numpy as np,pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');R=Path('/home/ec2-user/studies/hpp_datasets/liver_ultrasound');Q=P/'data_qc/w10_outcome_qc_20261028';K=['participant_id','cohort','research_stage']
D=json.loads((P/'config/outcome_dictionary.yaml').read_text());spec=[s for s in D['static_outcomes'] if s.get('old_aggregate')]
def read(p,cols):return pq.ParquetFile(p).read(columns=cols,use_threads=False).to_pandas(ignore_metadata=True)
d=read(R/'liver_ultrasound.parquet',K+[s['field'] for s in spec]);a=read(R/'liver_ultrasound_aggregated.parquet',K+[s['old_aggregate'] for s in spec]);out=[]
for s in spec:
 t=d[K+[s['field']]].copy();v=pd.to_numeric(t[s['field']],errors='coerce');t[s['field']]=v.where(np.isfinite(v)&v.gt(0));g=t.groupby(K)[s['field']].agg(['mean','median','min','max','count']).reset_index();z=g.merge(a[K+[s['old_aggregate']]],on=K,how='inner',validate='one_to_one');v=pd.to_numeric(z[s['old_aggregate']],errors='coerce');m=z['mean'].notna()&v.notna();z=z.loc[m].copy();v=v[m];err=(z['mean']-v).abs();base=z.cohort.eq('10k')&z.research_stage.eq('00_00_visit');match={f:int((z[f]-v).abs().le(1e-6).sum()) for f in ['mean','median','min','max']};row={'outcome_id':s['id'],'comparable_all_visits':len(z),'match_at_1e_minus_6':match,'absolute_mean_error_quantiles':{str(k):float(v) for k,v in err.quantile([0,.5,.9,.95,.99,1]).items()},'baseline_comparable':int(base.sum()),'baseline_mean_difference_gt_1e_minus_6':int(err[base].gt(1e-6).sum()),'baseline_max_abs_mean_difference':float(err[base].max()),'mean_mismatch_by_valid_measurement_count':{str(n):{'n':int(m.sum()),'mean_differences':int(err[m].gt(1e-6).sum())} for n in sorted(z['count'].unique()) for m in [z['count'].eq(n)]},'matching_median_does_not_prove_upstream_production_algorithm':True};out.append(row);print('LIVER_AGG_DIAG',json.dumps(row),flush=True)
J=json.loads((Q/'preflight_TRE_ONLY.json').read_text());e=next(x for x in J if x['path'].endswith('liver_ultrasound_aggregated.parquet'))
for r in e['dictionary_rows']:
 v=r['row']
 if v.get('tabular_field_name') in [s['old_aggregate'] for s in spec]:print('AGG_DICTIONARY',v.get('tabular_field_name'),v.get('description_string'))
p=Q/'liver_aggregate_diagnostic.json';assert not p.exists();p.write_text(json.dumps({'diagnostics':out,'selected_outcome_aggregation_unchanged':'arithmetic_mean_of_valid_detail_rows_per_user','no_automatically_interchangeable_official_aggregate':True},indent=2))
m=P/'memos/w10_outcome_qc_20261028/MEMO.md';note='\n\n## Script06 official-liver-summary mismatch\n\nCurrent source aggregate is not interchangeable with raw-detail arithmetic mean. Diagnostic comparisons below do not establish the unpublished upstream production algorithm. Selected six raw-detail means follow the user decision and remain unchanged.\n```json\n'+json.dumps(out,indent=2)+'\n```\n';m.write_text(m.read_text()+note);assert note in m.read_text();print('LIVER_DIAGNOSTIC_DONE',flush=True)
