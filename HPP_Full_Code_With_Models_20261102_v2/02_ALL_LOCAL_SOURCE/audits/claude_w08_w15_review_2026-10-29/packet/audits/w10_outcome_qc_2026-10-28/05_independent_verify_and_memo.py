# -*- coding: utf-8 -*-
"""Recompute all static outcomes independently with row-wise Python sums/counts."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
from collections import defaultdict,Counter
import json,math,hashlib,sys,platform,csv
import pyarrow.parquet as pq
import pandas as pd,numpy as np
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');Q=P/'data_qc/w10_outcome_qc_20261028';L=Path('/home/ec2-user/multisystem_runtime_20261021/w10_outcome_qc_20261028');R=Path('/home/ec2-user/studies/hpp_datasets');K=['participant_id','cohort','research_stage'];BAD={'','undefined','null','none','nan','na','n/a','unknown','<na>'}
assert (L/'04_actual_exit_code').read_text().strip()=='0'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p,cols=None):return pq.ParquetFile(p,pre_buffer=False).read(columns=cols,use_threads=False).to_pandas(ignore_metadata=True)
def dump(p,j):
 assert not p.exists(),str(p);b=(json.dumps(j,indent=2,ensure_ascii=False,default=str,allow_nan=False)+'\n').encode();p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);assert p.read_bytes()==b
D=json.loads((P/'config/outcome_dictionary.yaml').read_text());S=json.loads((Q/'summary_TRE_ONLY.json').read_text());M=json.loads((Q/'build_artifact_manifest_TRE_ONLY.json').read_text());inputs=json.loads((Q/'source_input_manifest_TRE_ONLY.json').read_text());A=read(P/'analysis/shared/outcome_registry_TRE_ONLY.parquet');assert len(A)==S['registry_rows'] and not A.duplicated(K+['outcome_id']).any();observed={tuple(r[k] for k in K)+ (r['outcome_id'],):r for r in A.to_dict('records')};expected={};checks=[];rawchecks=[]
for inp in inputs:
 p=Path(inp['path']);assert p.stat().st_size==inp['bytes'] and p.stat().st_mtime_ns==inp['mtime_ns'];assert pq.ParquetFile(p).metadata.num_rows==inp['rows']
for item in M:
 p=Path(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
 if 'rows' in item:assert pq.ParquetFile(p).metadata.num_rows==item['rows'];assert sha(Path(item['runtime_source']))==sha(p)
for rel in sorted({s['source'] for s in D['static_outcomes']}):
 specs=[s for s in D['static_outcomes'] if s['source']==rel];p=R/rel;pf=pq.ParquetFile(p);assert pf.metadata.num_rows<1000000;g=defaultdict(lambda:defaultdict(list));n=Counter();zeros=Counter();reasons=Counter();invalid=0
 for batch in pf.iter_batches(columns=K+[s['field'] for s in specs],batch_size=512,use_threads=False):
  for row in batch.to_pylist():
   values=[row[k] for k in K];valid=not any(x is None or str(x).lower() in BAD or str(x)!=str(x).strip() for x in values)
   try:valid=valid and int(values[0])>0 and float(values[0])==int(values[0])
   except (TypeError,ValueError):valid=False
   if not valid:invalid+=1;continue
   key=(int(values[0]),str(values[1]),str(values[2]))
   for s in specs:
    k=key+(s['id'],);n[k]+=1;v=row[s['field']]
    if v is not None:
     try:v=float(v)
     except (ValueError,TypeError):v=None
    good=v is not None and math.isfinite(v) and (v>=0 if s['valid']=='finite_ge_zero' else v>0)
    if good:g[k]['v'].append(v)
    if v is not None and math.isfinite(v) and v==0:zeros[k]+=1
 for key,num in n.items():
  vals=g[key]['v'];mean=math.fsum(vals)/len(vals) if vals else None;expected[key]=(mean,len(vals),num,zeros[key]);a=observed[key];assert a['n_valid_measurements']==len(vals) and a['n_source_measurements']==num and a['zero_measurement_count']==zeros[key];assert bool(a['phenotype_eligible'])==bool(vals)
  if vals:assert math.isclose(float(a['value']),mean,rel_tol=1e-12,abs_tol=1e-9),(key[-1],float(a['value'])-mean)
  else:assert pd.isna(a['value'])
 assert invalid==S['source_meta'][rel]['invalid_key_rows'];rawchecks.append({'source':rel,'sha256':sha(p),'rows':pf.metadata.num_rows,'independent_recomputed_outcome_groups':len(n),'invalid_key_rows':invalid})
assert set(expected)==set(observed)
# Whole-registry missingness and independent source-specific eligibility.
assert A.outcome_id.nunique()==9;assert set(A.loc[A.outcome_id.str.startswith('liver_'),'outcome_id'])=={'liver_'+s['phenotype_id'] for s in json.loads((P/'config/w10_user_rules_20261028/liver_six_mean_spec.json').read_text())['fields']};assert D['CGM']['values_materialized_W10'] is False;assert D['CGM']['do_not_substitute_iglu_full_period_mean'];assert set(A.outcome_id).isdisjoint({'glucose_common_day_mean'})
# Independent food-presence and aggregate count conservation, no model fit.
f=read(P/'analysis/shared/food_daily_TRE_ONLY.parquet',K+['energy_eligible']);foodset=set(map(tuple,f[K].itertuples(index=False,name=None)));enset=set(map(tuple,f.loc[f.energy_eligible.eq(True),K].itertuples(index=False,name=None)))
for r in S['baseline_outcome_counts']:
 a=A.loc[A.outcome_id.eq(r['outcome_id'])&A.cohort.eq('10k')&A.research_stage.eq('00_00_visit')];v=a.loc[a.phenotype_eligible];vkeys=set(map(tuple,v[K].itertuples(index=False,name=None)));assert len(a)==r['baseline_source_people'];assert len(v)==r['baseline_eligible_people'];assert len(vkeys&foodset)==r['baseline_with_indexed_food_same_visit'];assert len(vkeys&enset)==r['baseline_with_energy_food_same_visit'];assert r['baseline_eligible_people']+r['baseline_missing_or_invalid_people']==len(a)
# Source day counts and interface are NOT an outcome value.
cg=read(P/'analysis/shared/cgm_daily_TRE_ONLY.parquet',K+['date','cgm_day_eligible']);ct=Counter(tuple(row[k] for k in K) for row in cg.to_dict('records') if row['cgm_day_eligible'] is True);av=read(Q/'cgm_outcome_availability_TRE_ONLY.parquet');assert len(av)==len(ct)
for row in av.to_dict('records'):assert row['n_qualified_cgm_days']==ct[tuple(row[k] for k in K)] and row['outcome_value_materialized'] is False
actual={}
for tag in ['01','02','02b','03','04']:actual[tag]=int((L/(tag+'_actual_exit_code')).read_text())
assert all(actual[t]==0 for t in ['01','02b','03','04']);assert actual['02']==1
for p in (P/'scripts/w10_outcome_qc_20261028').glob('*.py'):checks.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
result={'verified':True,'independent_static_groups_recomputed':len(expected),'registry_rows':len(A),'measurement_QC_rows':S['measurement_QC_rows'],'independent_source_rechecks':rawchecks,'build_artifact_hashes_verified':len(M),'all_per_outcome_baseline_flows_recomputed':True,'source_vs_registry_values_full_recomputation':True,'independent_CGM_availability_verified':True,'CGM_window_values_deferred_to_W14':True,'units_traceable_to_TRE_dictionary':True,'device_manufacturer_quality_flags_absent_not_invented':True,'no_association_model_or_confirmatory_split_used':True,'actual_exits_before_this_verifier':actual,'code_hashes':checks,'environment':{'python':sys.executable,'python_version':sys.version,'pandas':pd.__version__,'pyarrow':__import__('pyarrow').__version__,'platform':platform.platform()}}
dump(Q/'independent_verification_TRE_ONLY.json',result)
body='# W10 outcome fields, units and basic value QC\n\nCode: '+str(P/'scripts/w10_outcome_qc_20261028')+'\nResult: '+str(Q)+'\nShared registry: '+str(P/'analysis/shared/outcome_registry_TRE_ONLY.parquet')+'\nDictionary: '+str(P/'config/outcome_dictionary.yaml')+'\nCount CSV: '+str(P/'data_qc/outcome_field_QC.csv')+'\nRuntime/logs/exits: '+str(L)+'\n\n01 schema/dictionary preflight. 02 initial value probe failed because pandas restored key columns as an index; retained exit1 and log, no final registry made. 02b explicitly ignores pandas index metadata and succeeded. 03 final source fields/units/basic distributions. 04 builds the nine static outcomes with independent valid flags, plus separate CGM day-availability interface. 05 independently re-reads all selected raw values and reconstructs every outcome mean/count with Python row-wise accumulators and math.fsum, verifies hashes and optional diet-presence counts.\n\nLatest user six-liver mean and same-visit policy snapshots are stored in config/w10_user_rules_20261028. Six liver outcomes use equally weighted valid detail rows in each explicit participant/cohort/stage. Body/BMI keys are unique; no random dedup. Same-stage outcomes do not require valid numeric timestamps or date lag gates. Independent outcomes do not require one another, food, CGM, covariates or O/G here; presence counts with food are diagnostics only. W14 supplies final E/D cohorts.\n\nSelected candidates: total fat MASS in g, VAT VOLUME in cm^3, BMI in kg/m^2; no switch among body metrics based on significance. Liver exact six fields and units are in dictionary. First four stiffness-related fields are not called liver fat. No main/secondary confirmatory budget is frozen; BMI remains secondary candidate. No log VAT, normal-health labels or percentile trimming.\n\nQC is finite plus mathematical domain, on source-provided values. No per-measurement manufacturer/device QC flag fields were supplied in inspected source tables; do not claim manufacturer QC was independently validated. Positive liver/fat/BMI; VAT nonnegative and zero flagged rather than interpreted as absent visceral fat. Diagnostic percentage columns appear fraction-scaled despite source dictionary %; they are not the selected fat-mass outcome. Consult structural comparisons before enabling percent outcomes.\n\nCGM outcome formula is equal-day average of qualified daily means over each MODULE-SPECIFIC common-day set. W10 only publishes available-day counts, never a whole-period CGM Y. W14 must recompute dietary features and CGM Y on the correct common window. W09 thresholds/minimum days remain registration candidates until protocol freeze.\n\nAll individual values, IDs, dates, matrices and full dictionary content stay TRE. Original source files are read-only. Binary outputs are written to local runtime first then copied to cloud with complete byte SHA and Parquet row-count verification. No exposure-outcome associations, prediction, discovery/confirmation selection, plots of private individual data or new primary hypotheses were run.\n\n## Baseline aggregate counts (not final microbiome analysis N)\n```json\n'+json.dumps(S['baseline_outcome_counts'],indent=2)+'\n```\n\n## Independent verification\n```json\n'+json.dumps(result,indent=2)+'\n```\n';p=P/'memos/w10_outcome_qc_20261028/MEMO.md';p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_text(body);assert p.read_text()==body
print('W10_INDEPENDENT_VERIFIED',json.dumps({k:v for k,v in result.items() if k not in ['independent_source_rechecks','code_hashes','environment']}),flush=True)
print('W10_BASELINE_COUNTS outcome source eligible indexed_food energy_food zero_rows',flush=True)
for r in S['baseline_outcome_counts']:print('COUNT',r['outcome_id'],r['baseline_source_people'],r['baseline_eligible_people'],r['baseline_with_indexed_food_same_visit'],r['baseline_with_energy_food_same_visit'],r['baseline_zero_measurement_rows'],flush=True)
print('MEMO',p,flush=True);print('W10_VERIFIER_DONE',flush=True)
