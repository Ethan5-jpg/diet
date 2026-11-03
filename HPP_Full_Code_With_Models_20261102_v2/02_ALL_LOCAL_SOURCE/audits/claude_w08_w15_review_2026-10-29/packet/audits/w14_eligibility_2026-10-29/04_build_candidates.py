# -*- coding: utf-8 -*-
"""W14: blind, baseline, source-specific candidate cohorts. No model/role/final support selection."""
import os
for n in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
from pathlib import Path
from collections import defaultdict
import json,hashlib,resource,shutil,ast,re,sys
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');Q=P/'data_qc/w14_eligibility_20261029';L=Path('/home/ec2-user/multisystem_runtime_20261021/w14_eligibility_20261029');B=L/'candidate_build';B.mkdir(exist_ok=False)
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(600,600));assert sys.executable=='/home/ec2-user/miniconda/envs/pheno/bin/python';assert shutil.disk_usage(L).free>3*1024**3
for tag in ['01','02','03','03b']:assert (L/(tag+'_actual_exit_code')).read_text().strip()=='0'
J=json.loads((Q/'preflight_TRE_ONLY.json').read_text());A=json.loads((P/'config/ACTIVE_PROTOCOL.json').read_text());R=json.loads(Path(A['analysis_register']).read_text());K=['participant_id','cohort','research_stage'];KD=K+['date'];art=[];flow=[];elig=[];medrows=[];summary=[]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p,cols=None):
 f=pq.ParquetFile(p,pre_buffer=False);assert f.metadata.num_rows<500000;budget=sum(f.metadata.row_group(i).column(j).total_uncompressed_size for i in range(f.metadata.num_row_groups) for j in range(f.metadata.row_group(i).num_columns));assert budget<512*1024**2
 return pd.concat([b.to_pandas(ignore_metadata=True) for b in f.iter_batches(columns=cols,batch_size=1024,use_threads=False)],ignore_index=True)
def load(role,cols=None):
 m=J['inputs'][role];p=Path(m['path']);assert p.stat().st_size==m['bytes'] and p.stat().st_mtime_ns==m['mtime_ns'] and sha(p)==m['sha256'];return read(p,cols)
def save(d,rel):
 p=B/rel;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();pq.write_table(pa.Table.from_pandas(d,preserve_index=False),p,compression='zstd');assert pq.ParquetFile(p).metadata.num_rows==len(d);art.append({'runtime':str(p),'relative_path':rel,'rows':len(d),'bytes':p.stat().st_size,'sha256':sha(p)})
def day(x):
 if x is None or pd.isna(x):return None
 return pd.Timestamp(x).date().isoformat()
def dated(d,c):return d[c].map(day)
def keyset(d):return set(d[K].itertuples(index=False,name=None))
def flags(d):
 for c in ['window_eligible','covariate_eligible','support_eligible','final_eligible']:d[c]=pd.array([None]*len(d),dtype='boolean')
 for c in ['discovery_role','confirmation_role']:d[c]=pd.array([None]*len(d),dtype='string')
 return d
contract={'version':'W14_candidate_contract_v1_20261029','medication_contract_precedence':'three-window user confirmation supersedes the single-envelope medication fields in earlier CGM operational contract','scope':{'cohort':'10k','research_stage':'00_00_visit','one_person_per_module':True,'other_visits':'retained in union source availability registry; outside current baseline scope, no pooled repeated observations'},'E':'baseline same-visit indexed actual food day present + W13 locked O/G + this outcome basic QC; no other outcome or covariate/window/score-support requirement','D':'final D not created; retain exposure-specific window availability and diagnostic minimum days separately from unassigned final eligibility','exposures':['UPF_bounds','fiber_density'],'UPF':'equal-day and cumulative lower/upper bounds on ALL energy-qualified days; no midpoint or lower bound relabeled as true scalar; legacy zero-unknown and strict subsets named sensitivity candidates only','CGM_person':str(Q/'W14_operational_contract.json'),'covariates':'W11 carried unmodified; age-sex and age-sex-smoking availability are candidate masks, final condition set/medication handling waits W18; no dropna or imputation fit','medication_window_user_confirmed':'2026-10-29 retain food period, outcome period, and their envelope separately; select formal model window at W18','medication_evidence':'exposed_in_window only if a valid ATC-coded start interval is wholly within window, meaning at least one initiation event in this period; no continuity inferred; confirmed_no_use_in_window requires explicit absence evidence unavailable in current source; otherwise unknown','medication_classes':{'metformin':['A10BA02'],'insulin_prefix':['A10A'],'glucose_lowering_prefix':['A10'],'statin_prefix':['C10AA'],'lipid_lowering_prefix':['C10'],'antihypertensive_prefixes':['C02','C03','C07','C08','C09'],'systemic_antibiotic_prefix':['J01']},'daily':'original W09 chosen connection and actual unique shared dates; retain different person-level connection selection explicitly; 1/n exposure-available days weights; diet-only raw within-person diagnostics, no scoring/OLS/association','UPF_daily_information':'bounds endpoint variation does not prove true scalar variation; true within-person information null pending measurement-model choice; exact zero-unknown subset diagnostic separately','no_learning':True,'support_rank_q_roles_and_final_N':None,'deferred':['P_Glu_Liver','P_VAT_TotalFat','Common4'],'protocol_sha256':sha(P/'config/ACTIVE_PROTOCOL.json'),'source_manifest':str(Q/'preflight_TRE_ONLY.json')}
cp=Q/'W14_candidate_contract.json';assert not cp.exists();cp.write_text(json.dumps(contract,indent=2)+'\n')
# Preserve all sources and validate canonical one-to-one keys before joins.
F=load('food_windows');FD=load('food_daily');C=load('covariates');S=load('samples',K+['site','sample_id','selected_primary_sample','sample_lock_status','descriptive_sample_calendar_date']);O=load('outcomes');CG=read(Q/'cgm_visit_phenotypes_TRE_ONLY.parquet');CD=load('food_cgm_daily');MT=load('medication_timing')
for d,keys in [(F,K),(FD,KD),(C,K),(S,K+['site']),(O,K+['outcome_id']),(CG,K),(CD,KD)]:assert not d.duplicated(keys).any()
assert S.selected_primary_sample.eq(True).all();assert np.isfinite(CG.value).all();assert CD.cgm_day_eligible.eq(True).all() and CD.point_qc_verified.eq(True).all();assert FD.primary_day.eq(True).all()
# Every source key in registry, while the intended analysis scope remains explicit.
U=pd.concat([d[K] for d in [F,C,S,O,CG]],ignore_index=True).drop_duplicates().sort_values(K).reset_index(drop=True);assert U[K].notna().all().all();U['scope_eligible']=U.cohort.eq('10k')&U.research_stage.eq('00_00_visit')
U=U.merge(F[K+['n_recorded_days']],on=K,how='left',validate='one_to_one');U['diet_source_available']=U.n_recorded_days.fillna(0).gt(0)
for site,prefix in [('oral','O'),('gut','G')]:
 z=S.loc[S.site.eq(site),K+['sample_id','descriptive_sample_calendar_date']].rename(columns={'sample_id':prefix+'_sample_id','descriptive_sample_calendar_date':prefix+'_sample_date'});U=U.merge(z,on=K,how='left',validate='one_to_one');U[prefix+'_source_available']=U[prefix+'_sample_id'].notna()
# Keep W11 values, labels, dates, technical and observed-family provenance; avoid collision with outcome/exposure columns.
cvcols=[c for c in C.columns if c not in K];Cv=C.rename(columns={c:'C_'+c for c in cvcols});U=U.merge(Cv,on=K,how='left',validate='one_to_one');U['age_sex_available_candidate']=U.C_core_age_sex_available.fillna(False).astype(bool);U['age_sex_smoking_available_candidate']=U.C_core_age_sex_smoking_available.fillna(False).astype(bool)
# Use actual exposure-qualified dates, independently of outcome; useful for medication windows and diagnostic day counts.
for exposure,mask in [('UPF_bounds',FD.energy_eligible),('fiber_density',FD.fiber_eligible)]:
 t=FD.loc[mask];g=t.groupby(K,observed=True).agg(first=('date','min'),last=('date','max'),n=('date','size')).reset_index();names={x:exposure+'_'+x for x in ['first','last','n']};F=F.merge(g.rename(columns=names),on=K,how='left',validate='one_to_one')
# Index medication rows by explicit visit, normalize only supplied ATC5 codes, never infer absence.
def codes(v):
 if isinstance(v,np.ndarray):v=v.tolist()
 if isinstance(v,(tuple,list,set)):return sorted(set(q for x in v for q in codes(x)))
 if isinstance(v,str):
  v=v.strip()
  if re.fullmatch(r'[A-Z][0-9]{2}[A-Z]{2}[0-9]{2}',v):return [v]
  if v.startswith(('[','(')):
   try:return codes(ast.literal_eval(v))
   except (ValueError,SyntaxError):pass
 return []
medmap=defaultdict(list)
for row in MT.to_dict('records'):medmap[tuple(row[k] for k in K)].append((codes(row['atc5']),day(row['start_interval_lower']),day(row['start_interval_upper'])))
medcache={}
def medication(k,lo,hi):
 cachekey=(k,lo,hi)
 if cachekey in medcache:return medcache[cachekey]
 result={};rows=medmap.get(k,[])
 for cl,patterns in contract['medication_classes'].items():
  matched=[(a,b) for c,a,b in rows if any(code==p if cl=='metformin' else code.startswith(p) for code in c for p in patterns)];proved=sum(bool(lo and hi and a and b and lo<=a<=b<=hi) for a,b in matched)
  result[cl]={'state':'exposed_in_window' if proved else 'unknown','listed_rows':len(matched),'initiation_in_window_rows':proved,'evidence':'start_interval_wholly_inside_window' if proved else 'no_window_dates' if not (lo and hi) else 'no_medication_records' if not rows else 'listed_but_continuation_unobserved_or_start_uncertain' if matched else 'class_not_listed_is_not_confirmed_no_use'}
 medcache[cachekey]=result;return result

def countflow(a,analysis,exposure,stage,previous=None):
 # actual family count is unknown; observational connected-component proxies are reported distinctly.
 ids=a.C_observed_related_group_id if 'C_observed_related_group_id' in a else pd.Series(dtype='string');known=ids.dropna().astype(str);unknown=int(ids.isna().sum());proxy=len(set(known))+unknown
 flow.append({'analysis_id':analysis,'exposure_id':exposure,'scope':'baseline_10k','stage':stage,'participant_visits':len(a),'participants':int(a.participant_id.nunique()),'excluded_from_previous':None if previous is None else previous-len(a),'observed_related_groups':int(known.nunique()),'unlinked_or_unobserved_people':unknown,'working_cluster_proxy_n':proxy,'family_n':None,'person_days':int(a.n_exposure_days.sum()) if 'n_exposure_days' in a else None,'support_n':None,'discovery_n':None,'confirmation_n':None,'final_n':None,'actual_q':None,'actual_rank':None})
# Explicit outcomes; no six-liver composite or unrelated outcome gate.
mapping={'D_Glu':'glucose_visit_equal_day_mean','D_TotalFat':'total_fat_mass','D_VAT':'vat_volume','D_BMI':'bmi'};mapping.update({'D_Liver_'+v:'liver_'+v for v in ['elasticity','velocity','viscosity','dispersion','attenuation','speed_of_sound']})
Ycols=['value','outcome_id','unit','phenotype_eligible','first_collection_date','last_collection_date','n_valid_measurements']
for analysis,oid in mapping.items():
 if analysis=='D_Glu':
  y=CG.copy();y['first_collection_date']=y.first_cgm_day;y['last_collection_date']=y.last_cgm_day;y['n_valid_measurements']=y.n_qualified_CGM_days
 else:y=O.loc[O.outcome_id.eq(oid)].copy()
 y=y[K+Ycols].rename(columns={'value':'Y_value','unit':'Y_unit','first_collection_date':'Y_first_date','last_collection_date':'Y_last_date','n_valid_measurements':'Y_valid_measurements'});assert not y.duplicated(K).any();b=U.merge(y,on=K,how='left',validate='one_to_one');b['outcome_available']=b.phenotype_eligible.fillna(False).astype(bool)&np.isfinite(b.Y_value);b['analysis_id']=analysis
 b['base_eligible']=b.scope_eligible&b.diet_source_available&b.O_source_available&b.G_source_available&b.outcome_available
 b['first_exclusion_reason']=np.select([~b.scope_eligible,~b.diet_source_available,~b.O_source_available,~b.G_source_available,~b.outcome_available],['outside_baseline_scope','no_indexed_actual_food_day','no_locked_oral_sample','no_locked_gut_sample','this_outcome_unavailable_or_basic_QC_failed'],default='base_eligible')
 regcols=K+['analysis_id','scope_eligible','diet_source_available','O_source_available','G_source_available','outcome_available','base_eligible','first_exclusion_reason','age_sex_available_candidate','age_sex_smoking_available_candidate']
 e=b[regcols].copy();e=flags(e);elig.append(e)
 s=b.loc[b.scope_eligible].copy();countflow(s,analysis,'ALL','scope_universe')
 prev=len(s)
 for col,stage in [('diet_source_available','indexed_food_source'),('O_source_available','locked_oral'),('G_source_available','locked_gut'),('outcome_available','base_E')]:
  s=s.loc[s[col]].copy();countflow(s,analysis,'ALL',stage,prev);prev=len(s)
 assert keyset(s)==keyset(b.loc[b.base_eligible])
 for exp in contract['exposures']:
  d=s.merge(F.drop(columns=['n_recorded_days']),on=K,how='left',validate='one_to_one');d['exposure_id']=exp;d['n_exposure_days']=d[exp+'_n'].fillna(0).astype(int);d['food_first_date']=d[exp+'_first'];d['food_last_date']=d[exp+'_last'];d['window_available_candidate']=d.n_exposure_days.ge(1)
  for n in [3,7,10,14]:d['food_days_ge'+str(n)+'_DIAGNOSTIC']=d.n_exposure_days.ge(n)
  if analysis=='D_Glu':
   d=d.merge(CG[K+['n_qualified_CGM_days','connection_id']],on=K,validate='one_to_one')
   for n in [3,7,10,14]:d['CGM_days_ge'+str(n)+'_DIAGNOSTIC']=d.n_qualified_CGM_days.ge(n)
  if exp=='UPF_bounds':
   d['exposure_value']=np.nan;d['exposure_lower_equal_day']=d.UPF_lower_equal_day_mean_all_energy_days;d['exposure_upper_equal_day']=d.UPF_upper_equal_day_mean_all_energy_days;d['exposure_lower_cumulative']=d.UPF_lower_cumulative_all_energy_days;d['exposure_upper_cumulative']=d.UPF_upper_cumulative_all_energy_days;d['exposure_measurement']='interval_bounds_no_scalar_imputation';d['exposure_unit']='energy_percentage_points'
  else:d['exposure_value']=d.fiber_equal_day_mean;d['exposure_measurement']='fiber_density_equal_day_and_cumulative_candidates';d['exposure_unit']='g_per_1000kcal'
  d=flags(d);d['table_status']='W14_candidate_not_final_D';d['support_status']='not_evaluated_until_W22';d['protocol_version']=A['version'];d['sample_lock_sha256']=A['W13_sample_selection_lock']['sha256']
  # Three explicitly named medication windows, user confirmed for W14; no final choice.
  medcols=[]
  for r in d.itertuples(index=False):
   k=tuple(getattr(r,c) for c in K);fl=day(r.food_first_date);fh=day(r.food_last_date);yl=day(r.Y_first_date);yh=day(r.Y_last_date);complete=all([fl,fh,yl,yh]);el=min(fl,yl) if complete else None;eh=max(fh,yh) if complete else None;mc={}
   for wn,lo,hi in [('food',fl,fh),('outcome',yl,yh),('envelope',el,eh)]:
    mc['med_'+wn+'_first']=lo;mc['med_'+wn+'_last']=hi
    for cl,st in medication(k,lo,hi).items():
     mc['med_'+wn+'_'+cl]=st['state'];medrows.append(dict(zip(K,k),analysis_id=analysis,exposure_id=exp,window_type=wn,window_first=lo,window_last=hi,medication_class=cl,**st))
   medcols.append(mc)
  d=pd.concat([d.reset_index(drop=True),pd.DataFrame(medcols)],axis=1)
  available=d.loc[d.window_available_candidate];countflow(d,analysis,exp,'base_E_before_exposure');countflow(available,analysis,exp,'window_ge1_candidate',len(d));c7=available.loc[available.food_days_ge7_DIAGNOSTIC];countflow(c7,analysis,exp,'window_ge7_DIAGNOSTIC',len(available));age=available.loc[available.age_sex_available_candidate];countflow(age,analysis,exp,'window_ge1_age_sex_candidate',len(available));sm=age.loc[age.age_sex_smoking_available_candidate];countflow(sm,analysis,exp,'window_ge1_age_sex_smoking_candidate',len(age))
  save(d,'analysis/'+analysis+'/'+exp+'/base_table_TRE_ONLY.parquet');summary.append({'analysis_id':analysis,'exposure_id':exp,'base_E':len(d),'window_ge1':len(available),'ge7_diagnostic':len(c7),'age_sex_candidate':len(age),'age_sex_smoking_candidate':len(sm)})
 print('MODULE_BUILT',analysis,'BASE_E',len(s),flush=True)
# Daily has its own calendar interface, drawn from E_Glu before any scalar window/covariate/support gate.
eglu=elig[0];assert eglu.analysis_id.eq('D_Glu').all();dailyE=U.merge(eglu.loc[eglu.base_eligible,K],on=K,how='inner',validate='one_to_one');daily=CD.merge(dailyE,on=K,how='inner',validate='many_to_one');assert not daily.duplicated(KD).any();assert np.isfinite(daily.glucose_mean_mg_dL).all();daily['Y_value']=daily.glucose_mean_mg_dL;daily['analysis_id']='Daily_CGM';daily['base_eligible']=True
for exp in contract['exposures']:
 d=daily.copy();d['exposure_id']=exp
 if exp=='fiber_density':d['exposure_value']=d.fiber_density_g_per_1000kcal;d['exposure_available_candidate']=d.fiber_eligible;d['exposure_measurement']='scalar_fiber_density'
 else:d['exposure_value']=np.nan;d['exposure_available_candidate']=d.energy_eligible;d['exposure_lower']=d.UPF_lower_pp;d['exposure_upper']=d.UPF_upper_pp;d['exposure_measurement']='interval_bounds_no_midpoint'
 t=d.loc[d.exposure_available_candidate].copy();g=t.groupby(K,observed=True).agg(n_exposure_days=('date','size'),food_first_date=('date','min'),food_last_date=('date','max')).reset_index();d=d.merge(g,on=K,how='left',validate='many_to_one');d.n_exposure_days=d.n_exposure_days.fillna(0).astype(int);d['weight_equal_person_candidate']=np.where(d.exposure_available_candidate,1/d.n_exposure_days.replace(0,np.nan),np.nan)
 assert np.allclose(d.loc[d.exposure_available_candidate].groupby(K,observed=True).weight_equal_person_candidate.sum(),1)
 info=g.copy();info['within_person_scalar_information_status']='pending_UPF_measurement_model' if exp=='UPF_bounds' else 'raw_exposure_only_diagnostic_not_residual_score_information'
 if exp=='fiber_density':
  t['raw_centered_D']=t.exposure_value-t.groupby(K).exposure_value.transform('mean');t['raw_centered_D_squared']=t.raw_centered_D**2;ss=t.groupby(K).raw_centered_D_squared.sum().rename('raw_within_person_SS').reset_index();info=info.merge(ss,on=K,validate='one_to_one');info['raw_positive_variation_DIAGNOSTIC']=info.n_exposure_days.ge(2)&info.raw_within_person_SS.gt(0)
 else:info['raw_within_person_SS']=np.nan;info['raw_positive_variation_DIAGNOSTIC']=pd.array([None]*len(info),dtype='boolean')
 # Exact-unknown-energy-zero UPF days are a distinct sensitivity subset; never substitute for full window.
 if exp=='UPF_bounds':
  q=t.loc[t.UPF_point_zero_unknown_energy_pp.notna()].copy();ex=q.groupby(K).UPF_point_zero_unknown_energy_pp.agg(['size','nunique']).reset_index().rename(columns={'size':'exact_subset_days','nunique':'exact_subset_distinct_D'});info=info.merge(ex,on=K,how='left',validate='one_to_one')
 for n in [3,7,10,14]:info['days_ge'+str(n)+'_DIAGNOSTIC']=info.n_exposure_days.ge(n)
 d=flags(d);d['score_source']='D_Glu_frozen_scores_pending';d['support_status']='within_person_joint_design_rank_pending_W22';d['table_status']='W14_daily_candidate_not_final';d['medication_window_date']=d.date
 # For single-day windows, coarse source start intervals cannot certify use unless exactly covered.
 daymed=[]
 for r in d.itertuples(index=False):
  k=tuple(getattr(r,c) for c in K);daymed.append({'med_actual_day_'+cl:st['state'] for cl,st in medication(k,r.date,r.date).items()})
 d=pd.concat([d.reset_index(drop=True),pd.DataFrame(daymed)],axis=1)
 save(d,'analysis/Daily_CGM/'+exp+'/base_table_TRE_ONLY.parquet');save(info,'analysis/Daily_CGM/'+exp+'/within_person_information_TRE_ONLY.parquet');countflow(d.drop_duplicates(K),'Daily_CGM',exp,'common_day_people_before_exposure');countflow(d.loc[d.exposure_available_candidate].drop_duplicates(K),'Daily_CGM',exp,'window_ge1_candidate');countflow(d.loc[d.exposure_available_candidate&d.age_sex_smoking_available_candidate].drop_duplicates(K),'Daily_CGM',exp,'window_ge1_age_sex_smoking_candidate');summary.append({'analysis_id':'Daily_CGM','exposure_id':exp,'base_E':len(d.drop_duplicates(K)),'window_ge1':len(info),'ge7_diagnostic':int(info.days_ge7_DIAGNOSTIC.sum()),'age_sex_candidate':int(d.loc[d.exposure_available_candidate&d.age_sex_available_candidate].participant_id.nunique()),'age_sex_smoking_candidate':int(d.loc[d.exposure_available_candidate&d.age_sex_smoking_available_candidate].participant_id.nunique()),'available_person_days':int(d.exposure_available_candidate.sum()),'raw_positive_variation_people_DIAGNOSTIC':int(info.raw_positive_variation_DIAGNOSTIC.sum()) if exp=='fiber_density' else None})
# Registry keeps failure reasons and all source keys, without outcome values; union supports W15 without assigning roles.
ER=pd.concat(elig,ignore_index=True);save(ER,'eligibility_registry_TRE_ONLY.parquet');save(U.drop(columns=[c for c in U if c.startswith('C_')]),'source_availability_union_TRE_ONLY.parquet');save(pd.DataFrame(medrows),'medication_window_status_TRE_ONLY.parquet')
union=ER.loc[ER.base_eligible,K].drop_duplicates();union=union.merge(U[K+['C_observed_related_group_id','C_relatedness_fully_ascertained']],on=K,validate='one_to_one');save(union,'W15_candidate_union_TRE_ONLY.parquet')
DF=pd.DataFrame(flow);DF.to_csv(Q/'cohort_flow_blind_W14.csv',index=False);(Q/'candidate_summary.json').write_text(json.dumps(summary,indent=2)+'\n');(Q/'candidate_artifact_manifest.json').write_text(json.dumps(art,indent=2)+'\n');print('W14_BUILD_PASS','artifacts',len(art),'registry_rows',len(ER),'W15_union',len(union),'peak_RSS_KiB',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
print('CANDIDATE_COUNTS',json.dumps(summary),flush=True)
