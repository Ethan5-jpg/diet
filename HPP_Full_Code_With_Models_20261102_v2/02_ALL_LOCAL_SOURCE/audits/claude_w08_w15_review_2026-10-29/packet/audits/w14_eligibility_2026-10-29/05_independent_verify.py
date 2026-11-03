# -*- coding: utf-8 -*-
"""Read inputs independently, rebuild sets/formulas/medication evidence, verify all W14 candidate rows."""
import os
for n in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
from pathlib import Path
from collections import defaultdict
import json,hashlib,resource,ast,re,sys
import numpy as np,pandas as pd,pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');Q=P/'data_qc/w14_eligibility_20261029';L=Path('/home/ec2-user/multisystem_runtime_20261021/w14_eligibility_20261029');K=['participant_id','cohort','research_stage'];KD=K+['date'];resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(600,600));assert sys.executable=='/home/ec2-user/miniconda/envs/pheno/bin/python';assert (L/'04_actual_exit_code').read_text().strip()=='0'
J=json.loads((Q/'preflight_TRE_ONLY.json').read_text());art=json.loads((Q/'candidate_artifact_manifest.json').read_text());contract=json.loads((Q/'W14_candidate_contract.json').read_text());B=L/'candidate_build';checks=defaultdict(int)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p,cols=None):
 f=pq.ParquetFile(p,pre_buffer=False);return pd.concat([b.to_pandas(ignore_metadata=True) for b in f.iter_batches(columns=cols,batch_size=127,use_threads=False)],ignore_index=True)
def load(role,cols=None):
 m=J['inputs'][role];assert Path(m['path']).stat().st_size==m['bytes'] and sha(m['path'])==m['sha256'];return read(m['path'],cols)
def ks(d):return set(d[K].itertuples(index=False,name=None))
def baseline(d):return d.loc[d.cohort.eq('10k')&d.research_stage.eq('00_00_visit')].copy()
def day(v):return None if v is None or pd.isna(v) else pd.Timestamp(v).date().isoformat()
for a in art:assert sha(a['runtime'])==a['sha256'] and pq.ParquetFile(a['runtime']).metadata.num_rows==a['rows'];checks['candidate_hashes']+=1
for a in json.loads((Q/'cgm_visit_artifact_manifest.json').read_text()):assert sha(a['path'])==a['sha256'] and sha(a['runtime'])==a['sha256'];checks['CGM_hashes']+=1
F=load('food_windows');FD=load('food_daily');S=load('samples',K+['site','sample_id','selected_primary_sample']);O=load('outcomes');C=load('covariates');CD=load('food_cgm_daily');DA=load('cgm_all_daily');CO=load('cgm_connections');Y=read(Q/'cgm_visit_phenotypes_TRE_ONLY.parquet');MT=load('medication_timing')
for d,k in [(F,K),(FD,KD),(S,K+['site']),(O,K+['outcome_id']),(C,K),(CD,KD),(DA,K+['connection_id','date'])]:assert not d.duplicated(k).any()
# Different implementation: sort Python tuples for each candidate sensor, independently of diet.
selected={}
for k,g in CO.groupby(K,observed=True):
 candidates=[]
 for r in g.itertuples(index=False):
  nd=int(DA.loc[DA.connection_id.eq(r.connection_id),'cgm_day_eligible'].sum());assert nd==r.qualified_days;candidates.append((-nd,r.first_observed,str(r.connection_id)))
 selected[k]=min(candidates)[2]
for r in Y.itertuples(index=False):
 k=tuple(getattr(r,c) for c in K);assert str(r.connection_id)==selected[k];z=DA.loc[DA.connection_id.eq(r.connection_id)&DA.cgm_day_eligible];assert len(z)==r.n_qualified_CGM_days;assert np.isclose(sum(z.glucose_mean_mg_dL)/len(z),r.value,rtol=0,atol=1e-10);checks['CGM_visit_values']+=1
# Reconstruct food windows from original daily numerators and denominators, not stored window means.
food={}
for k,g in FD.groupby(K,observed=True):
 z=g.loc[g.energy_eligible];t=g.loc[g.fiber_eligible];v={}
 if len(z):v.update(UPF_lower_equal_day_mean_all_energy_days=float(np.mean(100*z.known_UPF_kcal/z.energy_kcal)),UPF_upper_equal_day_mean_all_energy_days=float(np.mean(100*(z.known_UPF_kcal+z.unknown_NOVA_kcal)/z.energy_kcal)),UPF_lower_cumulative_all_energy_days=float(100*z.known_UPF_kcal.sum()/z.energy_kcal.sum()),UPF_upper_cumulative_all_energy_days=float(100*(z.known_UPF_kcal.sum()+z.unknown_NOVA_kcal.sum())/z.energy_kcal.sum()))
 if len(t):v.update(fiber_equal_day_mean=float(np.mean(1000*t.fiber_g/t.energy_kcal)),fiber_cumulative_ratio=float(1000*t.fiber_g.sum()/t.energy_kcal.sum()))
 v['UPF_bounds_dates']=sorted(z.date.tolist());v['fiber_density_dates']=sorted(t.date.tolist());food[k]=v
oral={tuple(r[:3]):r[3] for r in S.loc[S.site.eq('oral'),K+['sample_id']].itertuples(index=False,name=None)};gut={tuple(r[:3]):r[3] for r in S.loc[S.site.eq('gut'),K+['sample_id']].itertuples(index=False,name=None)};foodset=ks(F.loc[F.n_recorded_days.gt(0)]);cv={tuple(r[k] for k in K):r for r in C.to_dict('records')};ER=read(B/'eligibility_registry_TRE_ONLY.parquet');assert not ER.duplicated(K+['analysis_id']).any();mapping={'D_Glu':'glucose_visit_equal_day_mean','D_TotalFat':'total_fat_mass','D_VAT':'vat_volume','D_BMI':'bmi'};mapping.update({'D_Liver_'+p:'liver_'+p for p in ['elasticity','velocity','viscosity','dispersion','attenuation','speed_of_sound']})
# Independent medication parsing and Cartesian-free evidence lookup.
def parse(v):
 if isinstance(v,np.ndarray):v=v.tolist()
 if isinstance(v,(list,tuple,set)):return set(q for x in v for q in parse(x))
 if isinstance(v,str):
  if re.fullmatch(r'[A-Z][0-9]{2}[A-Z]{2}[0-9]{2}',v.strip()):return {v.strip()}
  try:
   x=ast.literal_eval(v)
   if isinstance(x,(list,tuple)):return parse(x)
  except (SyntaxError,ValueError):pass
 return set()
med=defaultdict(list)
for r in MT.to_dict('records'):med[tuple(r[k] for k in K)].append((parse(r['atc5']),day(r['start_interval_lower']),day(r['start_interval_upper'])))
mc={}
def expected_med(k,lo,hi,cl):
 ck=(k,lo,hi,cl)
 if ck not in mc:
  hit=False
  if lo and hi:
   for codes,a,b in med[k]:
    matched=any((c==p if cl=='metformin' else c.startswith(p)) for c in codes for p in contract['medication_classes'][cl])
    if matched and a and b and lo<=a and b<=hi:hit=True;break
  mc[ck]='exposed_in_window' if hit else 'unknown'
 return mc[ck]
flow=pd.read_csv(Q/'cohort_flow_blind_W14.csv');allE=set();medcounts=defaultdict(int)
for analysis,oid in mapping.items():
 src=Y if analysis=='D_Glu' else O.loc[O.outcome_id.eq(oid)];src=src.loc[src.phenotype_eligible&np.isfinite(src.value)];ys={tuple(r[k] for k in K):r for r in src.to_dict('records')};expected={k for k in foodset&set(oral)&set(gut)&set(ys) if k[1:] == ('10k','00_00_visit')};allE|=expected
 er=ER.loc[ER.analysis_id.eq(analysis)];assert ks(er.loc[er.base_eligible])==expected;assert er.groupby('first_exclusion_reason').size().sum()==len(er)
 for exp in ['UPF_bounds','fiber_density']:
  d=read(B/'analysis'/analysis/exp/'base_table_TRE_ONLY.parquet');assert not d.duplicated(K).any() and ks(d)==expected
  assert d[['window_eligible','covariate_eligible','support_eligible','final_eligible','discovery_role','confirmation_role']].isna().all().all()
  for r in d.to_dict('records'):
   k=tuple(r[c] for c in K);assert r['O_sample_id']==oral[k] and r['G_sample_id']==gut[k];assert np.isclose(r['Y_value'],ys[k]['value'],atol=0,rtol=0);v=food[k];dates=v[exp+'_dates'];assert r['n_exposure_days']==len(dates) and r['window_available_candidate']==bool(dates)
   assert day(r['food_first_date'])==(dates[0] if dates else None) and day(r['food_last_date'])==(dates[-1] if dates else None)
   for col in (['UPF_lower_equal_day_mean_all_energy_days','UPF_upper_equal_day_mean_all_energy_days','UPF_lower_cumulative_all_energy_days','UPF_upper_cumulative_all_energy_days'] if exp=='UPF_bounds' else ['fiber_equal_day_mean','fiber_cumulative_ratio']):assert np.isclose(r[col],v.get(col,np.nan),rtol=1e-12,atol=1e-10,equal_nan=True)
   if exp=='UPF_bounds':assert pd.isna(r['exposure_value'])
   cr=cv.get(k,{});assert r['age_sex_available_candidate']==bool(cr.get('core_age_sex_available',False)) and r['age_sex_smoking_available_candidate']==bool(cr.get('core_age_sex_smoking_available',False))
   for wn in ['food','outcome','envelope']:
    lo=day(r['med_'+wn+'_first']);hi=day(r['med_'+wn+'_last'])
    for cl in contract['medication_classes']:
     st=r['med_'+wn+'_'+cl];assert st==expected_med(k,lo,hi,cl);medcounts[(wn,cl,st)]+=1;checks['medication_states_rebuilt']+=1
   checks['person_candidate_rows_rebuilt']+=1
  ff=flow.loc[flow.analysis_id.eq(analysis)&flow.exposure_id.eq(exp)].set_index('stage');assert int(ff.loc['base_E_before_exposure','participants'])==len(d);assert int(ff.loc['window_ge1_candidate','participants'])==int(d.window_available_candidate.sum());assert int(ff.loc['window_ge7_DIAGNOSTIC','participants'])==int(d.food_days_ge7_DIAGNOSTIC.sum());assert int(ff.loc['window_ge1_age_sex_smoking_candidate','participants'])==int((d.window_available_candidate&d.age_sex_smoking_available_candidate).sum())
 print('VERIFIED_MODULE',analysis,len(expected),flush=True)
# Daily verification preserves source dates, CGM connection and Y; checks person weights and exposure-only information.
eglu=ks(ER.loc[ER.analysis_id.eq('D_Glu')&ER.base_eligible]);common={tuple(r[k] for k in KD):r for r in CD.to_dict('records') if tuple(r[k] for k in K) in eglu}
assert not (MT.start_interval_lower.notna()&MT.start_interval_lower.eq(MT.start_interval_upper)).any(),'single-day medication evidence requires additional verification'
for exp in ['UPF_bounds','fiber_density']:
 d=read(B/'analysis/Daily_CGM'/exp/'base_table_TRE_ONLY.parquet');assert set(d[KD].itertuples(index=False,name=None))==set(common) and not d.duplicated(KD).any();assert d[['window_eligible','covariate_eligible','support_eligible','final_eligible']].isna().all().all()
 for r in d.to_dict('records'):
  k=tuple(r[c] for c in KD);s=common[k];assert r['connection_id']==s['connection_id'] and r['Y_value']==s['glucose_mean_mg_dL'];assert r['O_sample_id']==oral[k[:3]] and r['G_sample_id']==gut[k[:3]]
  assert all(r['med_actual_day_'+cl]=='unknown' for cl in contract['medication_classes']);checks['daily_rows_source_matched']+=1
 v=d.loc[d.exposure_available_candidate];assert np.allclose(v.groupby(K).weight_equal_person_candidate.sum(),1);info=read(B/'analysis/Daily_CGM'/exp/'within_person_information_TRE_ONLY.parquet');assert ks(info)==ks(v)
 for r in info.to_dict('records'):
  k=tuple(r[c] for c in K);g=v.loc[v.participant_id.eq(k[0])&v.cohort.eq(k[1])&v.research_stage.eq(k[2])];assert len(g)==r['n_exposure_days']
  if exp=='fiber_density':x=g.exposure_value.to_numpy();ss=sum((x-x.mean())**2);assert np.isclose(ss,r['raw_within_person_SS'],atol=1e-8,rtol=1e-10)
  else:assert pd.isna(r['raw_within_person_SS']) and pd.isna(r['raw_positive_variation_DIAGNOSTIC'])
 checks['daily_person_weights']+=len(info)
assert ks(read(B/'W15_candidate_union_TRE_ONLY.parquet'))==allE
assert len(art)==28 and len(mapping)==10
result={'passed':True,'independent_batch_size':127,'checks':dict(checks),'W15_union_people':len(allE),'candidate_artifacts':len(art),'final_support_q_rank_roles_and_model_N_unassigned':True,'no_model_or_association_run':True,'medication_state_counts':[{'window':k[0],'class':k[1],'state':k[2],'rows':v} for k,v in sorted(medcounts.items())],'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
(Q/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n');print('W14_INDEPENDENT_PASS',json.dumps({k:v for k,v in result.items() if k!='medication_state_counts'}),flush=True)
