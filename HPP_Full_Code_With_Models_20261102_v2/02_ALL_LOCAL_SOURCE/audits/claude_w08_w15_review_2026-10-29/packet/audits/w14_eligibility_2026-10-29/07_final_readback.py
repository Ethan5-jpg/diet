# -*- coding: utf-8 -*-
"""Independent final readback of published W14 interfaces and unchanged sources."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib,resource,shutil
import pandas as pd,pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');Q=P/'data_qc/w14_eligibility_20261029';L=Path('/home/ec2-user/multisystem_runtime_20261021/w14_eligibility_20261029');K=['participant_id','cohort','research_stage'];resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(300,300));assert (L/'06_actual_exit_code').read_text().strip()=='0';manifest=json.loads((Q/'final_artifact_manifest.json').read_text());pre=json.loads((Q/'preflight_TRE_ONLY.json').read_text());contract=json.loads((P/'config/eligibility_rules_W14_20261029.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p,cols=None):
 f=pq.ParquetFile(p,pre_buffer=False);return pd.concat([b.to_pandas(ignore_metadata=True) for b in f.iter_batches(batch_size=257,columns=cols,use_threads=False)],ignore_index=True)
def ks(d):return set(d[K].itertuples(index=False,name=None))
assert len(manifest)==36 and len({x['path'] for x in manifest})==36
for x in manifest:
 assert sha(x['path'])==x['sha256']==sha(x['runtime_source']);assert Path(x['path']).stat().st_size==x['bytes']
 if x['rows'] is not None:assert pq.ParquetFile(x['path']).metadata.num_rows==x['rows']
for x in pre['inputs'].values():assert sha(x['path'])==x['sha256']
assert sha(P/'config/ACTIVE_PROTOCOL.json')==contract['protocol_sha256'];A=json.loads((P/'config/ACTIVE_PROTOCOL.json').read_text());assert sha(A['W13_sample_selection_lock']['path'])==A['W13_sample_selection_lock']['sha256']
E=read(P/'analysis/eligibility_registry_TRE_ONLY.parquet');assert len(E)==224477 and E.analysis_id.nunique()==11 and not E.duplicated(K+['analysis_id']).any();assert E[['window_eligible','covariate_eligible','support_eligible','final_eligible','discovery_role','confirmation_role']].isna().all().all();old=read(L/'candidate_build/eligibility_registry_TRE_ONLY.parquet');pd.testing.assert_frame_equal(E.loc[~E.analysis_id.eq('Daily_CGM'),old.columns].reset_index(drop=True),old,check_dtype=False)
glu=E.loc[E.analysis_id.eq('D_Glu')&E.base_eligible];daily=E.loc[E.analysis_id.eq('Daily_CGM')&E.base_eligible];assert ks(daily)<=ks(glu);both=read(P/'analysis/Daily_CGM/UPF_bounds/base_table_TRE_ONLY.parquet',K);assert ks(daily)==ks(both);union=read(P/'analysis/shared/W15_candidate_union_TRE_ONLY.parquet');assert ks(E.loc[E.base_eligible])==ks(union) and len(union)==6305
flow=pd.read_csv(P/'results/cohort_flow_blind.csv');assert flow[['support_n','discovery_n','confirmation_n','final_n','actual_q','actual_rank','family_n']].isna().all().all();assert set(flow.analysis_id)==set(E.analysis_id)
for analysis in set(E.analysis_id)-{'Daily_CGM'}:
 rows=flow.loc[flow.analysis_id.eq(analysis)&flow.exposure_id.eq('ALL')];n=rows.participants.tolist();assert all(n[i]>=n[i+1] for i in range(len(n)-1));assert int(rows.iloc[-1].participants)==int((E.analysis_id.eq(analysis)&E.base_eligible).sum());assert int(rows.excluded_from_previous.fillna(0).sum())+n[-1]==n[0]
med=pq.ParquetFile(P/'analysis/shared/medication_window_status_TRE_ONLY.parquet');states=set();nulls=0;medrows=0
for b in med.iter_batches(columns=['state','window_type'],batch_size=8192,use_threads=False):
 d=b.to_pandas();states.update(d.state);nulls+=int(d.state.isna().sum());medrows+=len(d)
assert states<=set(['exposed_in_window','confirmed_no_use_in_window','unknown']) and 'confirmed_no_use_in_window' not in states and nulls==0
agg=json.loads((Q/'aggregate_delivery.json').read_text());h=hashlib.sha256(json.dumps(agg,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest();assert h==(Q/'aggregate_delivery.sha256').read_text().strip() and agg['production_artifacts']==len(manifest)
code=json.loads((Q/'code_manifest.json').read_text())
for x in code:assert sha(x['path'])==x['sha256']
for tag in ['01','02','03','03b','04','05','06']:assert (L/(tag+'_actual_exit_code')).read_text().strip()=='0'
for suffix in ['.log','_actual_exit_code']:
 src=L/('06'+suffix);dst=P/'logs/w14_eligibility_20261029'/src.name;assert not dst.exists();shutil.copyfile(src,dst);assert sha(src)==sha(dst)
result={'passed':True,'production_artifacts_verified':len(manifest),'unchanged_preflight_inputs':len(pre['inputs']),'canonical_registry_rows':len(E),'canonical_registry_modules':int(E.analysis_id.nunique()),'W15_union_people':len(union),'Daily_base_people':len(daily),'medication_ledger_rows':medrows,'medication_states':sorted(states),'no_inferred_confirmed_no_use':True,'all_pending_fields_null':True,'actual_prior_exit_codes':[0]*7,'aggregate_sha256':h,'models_run':False,'scientific_associations_opened':False}
(Q/'final_delivery_verification.json').write_text(json.dumps(result,indent=2)+'\n');p=P/'memos/w14_eligibility_20261029/FINAL_READBACK.md';p.write_text('# W14 final publication readback\n\n'+json.dumps(result,indent=2)+'\n\n07 actual exit is recorded by its external runner; do not infer exit from this file alone.\n');print('W14_FINAL_READBACK_PASS',json.dumps(result),flush=True)
print('COUNTS_HEADER module exposure E window_ge1 ge7_diagnostic age_sex_smoking_candidate')
for r in agg['counts']:print(r['analysis_id'],r['exposure_id'],r['base_E'],r['window_ge1'],r['ge7_diagnostic'],r['age_sex_smoking_candidate'])
print('DAILY',json.dumps(agg['daily']));print('MEDICATION_WINDOW_COUNTS',json.dumps(agg['medication_window_counts']));print('CHECKS',json.dumps(agg['independent_checks']));print('AGGREGATE_SHA256',h)
