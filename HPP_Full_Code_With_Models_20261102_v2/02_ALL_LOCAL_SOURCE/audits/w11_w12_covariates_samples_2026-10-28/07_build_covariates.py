# -*- coding: utf-8 -*-
"""W11 source-coding, time-aware covariate registry. No imputation/model fit."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
from collections import defaultdict,Counter
import json,csv,re,hashlib,shutil,ast,math,bisect,calendar
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');R=Path('/home/ec2-user/studies/hpp_datasets');Q=P/'data_qc/w11_w12_covariates_samples_20261028';L=Path('/home/ec2-user/multisystem_runtime_20261021/w11_w12_covariates_samples_20261028');K=['participant_id','cohort','research_stage'];BAD={'','undefined','none','null','nan','na','n/a','unknown','<na>'}
assert (L/'06_actual_exit_code').read_text().strip()=='0'
E=json.loads((Q/'preflight_dictionary_TRE_ONLY.json').read_text());J=json.loads((Q/'preflight_sources_TRE_ONLY.json').read_text());inputs=[];artifacts=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):
 assert not p.exists(),str(p);p.parent.mkdir(parents=True,exist_ok=True);b=(json.dumps(v,ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n').encode();p.write_bytes(b);assert p.read_bytes()==b;artifacts.append({'path':str(p),'bytes':len(b),'sha256':sha(p)})
def load(p,cols):
 pf=pq.ParquetFile(p,pre_buffer=False);assert set(cols)<=set(pf.schema_arrow.names);assert pf.metadata.num_rows<400000;budget=sum(pf.metadata.row_group(i).column(j).total_uncompressed_size for i in range(pf.metadata.num_row_groups) for j in range(pf.metadata.row_group(i).num_columns) if pf.metadata.row_group(i).column(j).path_in_schema in cols);assert budget<128*1024**2
 inputs.append({'path':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'rows':pf.metadata.num_rows,'projection':cols,'projected_uncompressed_bytes':budget});return pd.concat([b.to_pandas(ignore_metadata=True) for b in pf.iter_batches(columns=cols,batch_size=1024,use_threads=False)],ignore_index=True)
def keyvalid(row,cols=K):
 for k in cols:
  x=row.get(k)
  if x is None or pd.isna(x) or str(x).lower() in BAD or str(x)!=str(x).strip():return False
 try:return float(row['participant_id'])==int(row['participant_id']) and int(row['participant_id'])>0
 except (TypeError,ValueError):return False

def canon(d,cols=K):
 m=pd.Series([keyvalid(r,cols) for r in d.to_dict('records')],index=d.index);d=d.loc[m].copy();d['participant_id']=pd.to_numeric(d.participant_id).astype('int64')
 for c in cols[1:]:d[c]=d[c].astype(str)
 return d

def parquet(d,n,dest):
 p=L/n;assert not p.exists() and not dest.exists();pq.write_table(pa.Table.from_pandas(d,preserve_index=False),p,compression='zstd');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);assert sha(p)==sha(dest) and pq.ParquetFile(dest).metadata.num_rows==len(d);artifacts.append({'path':str(dest),'runtime_source':str(p),'rows':len(d),'bytes':dest.stat().st_size,'sha256':sha(dest)})
# Source names/types/units were inspected before selecting fields.
core={rel:next(s for s in J if s['path']==str(R/rel)) for rel in ['population/population.parquet','medications/medications.parquet','lifestyle_and_environment/lifestyle_and_environment.parquet','sociodemographics/ukbb.parquet']}
for rel,s in core.items():p=R/rel;assert p.stat().st_size==s['bytes'] and p.stat().st_mtime_ns==s['mtime_ns']
# Canonical registry universe is a UNION of necessary-source explicit visit keys, not a cross-domain intersection.
food=canon(load(P/'analysis/shared/food_daily_TRE_ONLY.parquet',K+['date']));food['d']=pd.to_datetime(food.date,format='%Y-%m-%d',errors='raise');fw=food.groupby(K).agg(food_first=('d','min'),food_last=('d','max'),food_days=('date','size')).reset_index();fw['anchor_date']=fw.food_first+(fw.food_last-fw.food_first)/2;fw['anchor_source']='midpoint_of_actual_indexed_food_window'
stat=canon(load(P/'analysis/shared/outcome_registry_TRE_ONLY.parquet',K));cg=canon(load(P/'data_qc/w10_outcome_qc_20261028/cgm_outcome_availability_TRE_ONLY.parquet',K+['first_cgm_day']));anthpath=R/'anthropometrics/anthropometrics.parquet';anth=canon(load(anthpath,K+['collection_date']));anth['d']=pd.to_datetime(anth.collection_date,errors='coerce');aa=anth.groupby(K).d.min().reset_index().rename(columns={'d':'anthropometric_date'})
life_names=[f['name'] for f in core['lifestyle_and_environment/lifestyle_and_environment.parquet']['schema']];smoke='smoking_current_status';assert smoke in life_names
# Optional alcohol item only if unambiguous dictionary wording and actual source field agree.
alcohol_candidates=sorted(set(e['row']['tabular_field_name'] for e in E if 'lifestyle_and_environment' in e['file'] and e['row'].get('tabular_field_name') in life_names and re.search(r'how often do you drink alcohol|alcohol intake frequency',e['row'].get('description_string','')+' '+e['row'].get('field_string',''),re.I)))
alcohol=alcohol_candidates[0] if len(alcohol_candidates)==1 else None
alcohol='alcohol_current_frequency' if 'alcohol_current_frequency' in life_names else alcohol
numeric_life=[n for n in ['activity_walking_10min_days_weekly','activity_moderate_days_weekly','activity_vigorous_days_weekly','sleep_hours_daily'] if n in life_names]
sel=[smoke]+([alcohol] if alcohol else [])+numeric_life;life=canon(load(R/'lifestyle_and_environment/lifestyle_and_environment.parquet',K+['collection_date']+sel))
fn=['study_family_member','study_family_member_participant_id','study_household_member','study_household_member_participant_id'];fam=canon(load(R/'sociodemographics/ukbb.parquet',K+['collection_date']+fn))
mednames=[f['name'] for f in core['medications/medications.parquet']['schema']];assert all(c in mednames for c in K+['medication','atc5','start_year','start_month','start_date']);assert not any(re.search(r'^(stop|end)_',n) for n in mednames);med=canon(load(R/'medications/medications.parquet',K+[n for n in ['collection_date','collection_timestamp','medication','atc3','atc4','atc5','start_year','start_month','start_date'] if n in mednames]))
keys=pd.concat([f[K] for f in [food,stat,cg,anth,life,fam,med]],ignore_index=True)
for role in ['gut_index','gut_species','oral_index']:
 m=json.loads((P/'manifests/w02_sources_20261026T130908Z/source_manifest.json').read_text());src=Path(next(s['path'] for s in m['sources'] if s['role']==role));keys=pd.concat([keys,canon(load(src,K))[K]],ignore_index=True)
base=keys.drop_duplicates().sort_values(K).reset_index(drop=True);assert not base.duplicated(K).any();base=base.merge(fw,on=K,how='left',validate='one_to_one').merge(aa,on=K,how='left',validate='one_to_one');base.loc[base.anchor_date.isna()&base.anthropometric_date.notna(),'anchor_source']='anthropometrics_same_visit_date';base['anchor_date']=base.anchor_date.fillna(base.anthropometric_date);cg['cg_date']=pd.to_datetime(cg.first_cgm_day,errors='coerce');base=base.merge(cg[K+['cg_date']],on=K,how='left',validate='one_to_one');m=base.anchor_date.isna()&base.cg_date.notna();base.loc[m,'anchor_date']=base.loc[m,'cg_date'];base.loc[m,'anchor_source']='first_qualified_CGM_day_same_visit';base=base.drop(columns=['cg_date'])
# Population birth year / sex are invariant source attributes, keyed by person+cohort when supplied.
popnames=[f['name'] for f in core['population/population.parquet']['schema']];assert 'participant_id' in popnames and 'sex' in popnames and 'year_of_birth' in popnames
PK=['participant_id','cohort'] if 'cohort' in popnames else ['participant_id'];pop=canon(load(R/'population/population.parquet',PK+['year_of_birth','sex']),PK)
# Collapsing duplicates uses nonmissing consensus, not random choice or outcome-aware choice.
def consensus(d,fields,key=K,datecol='collection_date'):
 rows=[]
 for kv,g in d.groupby(key,sort=True,dropna=False,observed=True):
  kv=kv if isinstance(kv,tuple) else (kv,);z=dict(zip(key,kv));z['source_rows']=len(g)
  if datecol in g:
   ds=pd.to_datetime(g[datecol],errors='coerce');z['source_date_min']=ds.min();z['source_date_max']=ds.max()
  for c in fields:
   vals=g[c].dropna();vals=vals.loc[~vals.astype(str).str.strip().str.lower().isin(BAD)];unique=pd.unique(vals.astype(str));z[c]=unique[0] if len(unique)==1 else None;z[c+'_source_status']='consensus' if len(unique)==1 else 'missing' if len(unique)==0 else 'conflicting_repeat';z[c+'_distinct_nonmissing']=len(unique)
  rows.append(z)
 return pd.DataFrame(rows)
pv=consensus(pop,['year_of_birth','sex'],PK,None).rename(columns={'source_rows':'population_source_rows'});base=base.merge(pv,on=PK,how='left',validate='many_to_one');birth=pd.to_numeric(base.year_of_birth,errors='coerce');good=np.isfinite(birth)&birth.eq(np.floor(birth))&base.anchor_date.notna()&birth.le(base.anchor_date.dt.year);base['age_calendar_years']=((base.anchor_date.dt.year-birth).where(good));base['age_definition']='reference_calendar_year_minus_reported_birth_year_not_exact_birthday_age';base['age_status']=np.where(base.age_calendar_years.notna(),'available_calendar_year_age',np.where(base.anchor_date.isna(),'reference_date_missing','birth_year_missing_or_conflict'));base['sex_code']=pd.to_numeric(base.sex,errors='coerce').where(pd.to_numeric(base.sex,errors='coerce').isin([0,1])).astype('Int64');base['sex_label']=base.sex_code.map({0:'Female',1:'Male'}).astype('string')
code_rows=list(csv.DictReader((R/'metadata/coding_mapping.csv').open(encoding='utf-8-sig',newline='')));coding={}
for c in code_rows:coding.setdefault(c['code_number'],{})[c['coding']]=c['english']
assert coding['9']=={'0':'Female','1':'Male'}
roles=[]
def register(name,role,**extra):roles.append(dict(variable=name,role=role,**extra))
register('age_calendar_years','core_adjustment_candidate',unit='years',definition='year(anchor_date)-year_of_birth; calendar-year age, uncertainty up to one year; no exact DOB invented',source=str(R/'population/population.parquet'),reference='same-visit food midpoint; fallback anthropometrics then first qualified CGM date; missing reference stays missing')
register('sex_code','core_adjustment_candidate',coding={'0':'Female','1':'Male'},source=str(R/'population/population.parquet'),gender_not_silently_substituted_for_sex=True)
questionnaire=consensus(life,sel).rename(columns={'source_rows':'lifestyle_source_rows','source_date_min':'lifestyle_date_min','source_date_max':'lifestyle_date_max'});base=base.merge(questionnaire,on=K,how='left',validate='one_to_one')
for c in [smoke]+([alcohol] if alcohol else []):
 e=[e['row'] for e in E if e['row'].get('tabular_field_name')==c and 'lifestyle_and_environment' in e['file']];assert len(e)==1;(defn,)=e;code=defn.get('data_coding');assert code in coding
 def norm(v):
  if v is None or pd.isna(v):return None
  try:f=float(v);return str(int(f)) if f.is_integer() else str(v)
  except (ValueError,TypeError):return str(v)
 base[c+'_raw_code']=base[c].map(norm).astype('string');validcodes={k:v for k,v in coding[code].items() if not re.search(r'prefer not|do not know|don.t know|refus',v,re.I) and not k.startswith('-')};base[c+'_label']=base[c+'_raw_code'].map(validcodes).astype('string');base[c+'_available']=base[c+'_label'].notna();register(c,'core_adjustment_candidate_same_visit' if c==smoke else 'sensitivity_candidate',source=str(R/'lifestyle_and_environment/lifestyle_and_environment.parquet'),coding_id=code,coding=coding[code],negative_or_refused_codes_become_unknown_not_No=True,source_date_fields=['lifestyle_date_min','lifestyle_date_max'],conflicting_repeat_rule='missing with conflict flag')
for c in numeric_life:
 raw=pd.to_numeric(base[c],errors='coerce');upper=24 if c=='sleep_hours_daily' else 7;base[c+'_numeric']=raw.where(np.isfinite(raw)&raw.ge(0)&raw.le(upper));base[c+'_available']=base[c+'_numeric'].notna();register(c,'sensitivity_candidate',source=str(R/'lifestyle_and_environment/lifestyle_and_environment.parquet'),unit='hours/day' if c=='sleep_hours_daily' else 'days/week',source_date_fields=['lifestyle_date_min','lifestyle_date_max'],not_automatically_added_to_all_models=True,invalid_or_refused_values_not_zero=True)

def time_relation(lo,hi,first,last):
 if pd.isna(lo) or pd.isna(hi) or pd.isna(first) or pd.isna(last):return 'unknown'
 if hi<first:return 'recorded_before_food_window'
 if lo>last:return 'recorded_after_food_window'
 if lo>=first and hi<=last:return 'recorded_within_food_window'
 return 'records_span_food_window_boundaries'
base['lifestyle_recording_time_vs_food']=[time_relation(r.lifestyle_date_min,r.lifestyle_date_max,r.food_first,r.food_last) for r in base.itertuples()]
# Preserve the actually available oral-health questionnaire item; do not relabel it recent systemic antibiotic use.
oral_item='oralhealth_antibiotics_for_gingivitis';oh_source=R/'health_and_medical_history/initial_medical.parquet';oh_names=pq.ParquetFile(oh_source).schema_arrow.names
if oral_item in oh_names:
 oh=canon(load(oh_source,K+['collection_date',oral_item]));ov=consensus(oh,[oral_item]).rename(columns={'source_rows':'oralhealth_source_rows','source_date_min':'oralhealth_date_min','source_date_max':'oralhealth_date_max'});base=base.merge(ov,on=K,how='left',validate='one_to_one');oe=[e['row'] for e in E if e['row'].get('tabular_field_name')==oral_item and 'health_and_medical_history' in e['file']];assert len(oe)==1;oc=oe[0].get('data_coding');assert oc in coding;validcodes={k:v for k,v in coding[oc].items() if not k.startswith('-') and not re.search(r'prefer not|do not know|don.t know|refus',v,re.I)};base[oral_item+'_label']=base[oral_item].map(norm).map(validcodes).astype('string');base['oralhealth_recording_time_vs_food']=[time_relation(r.oralhealth_date_min,r.oralhealth_date_max,r.food_first,r.food_last) for r in base.itertuples()];register(oral_item,'sensitivity_candidate_self_reported_oral_health',source=str(oh_source),coding_id=oc,coding=coding[oc],not_general_recent_antibiotic_use=True,no_unreported_time_window_invented=True)

# ATC flags indicate what supplied records list, not active exposure or absence at a target date.
patterns={'ATC5_A10BA02_listed':('exact','A10BA02'),'ATC5_A10A_prefix_listed':('prefix','A10A'),'ATC5_A10_prefix_listed':('prefix','A10'),'ATC5_C10AA_prefix_listed':('prefix','C10AA'),'ATC5_C10_prefix_listed':('prefix','C10'),'ATC5_C02_C03_C07_C08_C09_prefix_listed':('prefixes',['C02','C03','C07','C08','C09']),'ATC5_J01_prefix_listed':('prefix','J01')}
def atcs(x):
 if x is None:return []
 if isinstance(x,np.ndarray):x=x.tolist()
 if isinstance(x,(tuple,list,set)):
  return sorted(set(v for z in x for v in atcs(z)))
 if isinstance(x,str):
  s=x.strip()
  if re.fullmatch('[A-Z][0-9]{2}[A-Z]{2}[0-9]{2}',s):return [s]
  if s.startswith(('[','(')):
   try:return atcs(ast.literal_eval(s))
   except (ValueError,SyntaxError):return []
 return []
# Retain observed treatment starts with year/month uncertainty; no stop dates supplied.
med=med.merge(base[K+['food_first','food_last']],on=K,how='left',validate='many_to_one')
def start_interval(row):
 y=row.get('start_year');month=row.get('start_month');dt=pd.to_datetime(row.get('start_date'),errors='coerce');sy=int(y) if y is not None and not pd.isna(y) and float(y).is_integer() and 1<=int(y)<=9999 else None;sm=int(month) if month is not None and not pd.isna(month) and float(month).is_integer() and 1<=int(month)<=12 else None
 if sy is not None and pd.notna(dt) and sy!=dt.year:return pd.NaT,pd.NaT,'source_start_year_date_conflict'
 if sm is not None and pd.notna(dt) and sm!=dt.month:return pd.NaT,pd.NaT,'source_start_month_date_conflict'
 if sy is None:
  if pd.isna(dt):return pd.NaT,pd.NaT,'start_unavailable'
  sy=dt.year;sm=dt.month;precision='provider_date_month_used_day_precision_unverified'
 else:precision='reported_year_month_interval' if sm is not None else 'reported_year_interval'
 try:
  if sm is not None:return pd.Timestamp(year=sy,month=sm,day=1),pd.Timestamp(year=sy,month=sm,day=calendar.monthrange(sy,sm)[1]),precision
  return pd.Timestamp(year=sy,month=1,day=1),pd.Timestamp(year=sy,month=12,day=31),precision
 except (ValueError,OverflowError):return pd.NaT,pd.NaT,'unusable_calendar_start'
times=[start_interval(r) for r in med.to_dict('records')];med['start_interval_lower']=[r[0] for r in times];med['start_interval_upper']=[r[1] for r in times];med['start_time_precision_status']=[r[2] for r in times];med['start_interval_vs_food']=[time_relation(r.start_interval_lower,r.start_interval_upper,r.food_first,r.food_last).replace('recorded_','start_').replace('records_span','start_interval_spans') for r in med.itertuples()]
parquet(med,'medication_timing_TRE_ONLY.parquet',Q/'medication_timing_TRE_ONLY.parquet')
medrows=[]
for kv,g in med.groupby(K,sort=True,observed=True):
 allcodes=[atcs(v) for v in g.atc5];codes=sorted(set(v for row in allcodes for v in row));z=dict(zip(K,kv));z['medication_source_rows']=len(g);z['medication_rows_without_valid_ATC5']=sum(not v for v in allcodes);z['medication_ATC5_codes_json']=json.dumps(codes);ds=pd.to_datetime(g.collection_date,errors='coerce') if 'collection_date' in g else pd.Series(pd.NaT,index=g.index);z['medication_record_date_min']=ds.min();z['medication_record_date_max']=ds.max();z['medication_active_interval_status']='start_information_available_stop_and_continuation_unobserved';z['medication_start_precision_counts_json']=json.dumps(g.start_time_precision_status.value_counts().astype(int).to_dict());z['medication_start_vs_food_counts_json']=json.dumps(g.start_interval_vs_food.value_counts().astype(int).to_dict());z['class_start_timing_json']=json.dumps({n:Counter(g.iloc[i].start_interval_vs_food for i,codes in enumerate(allcodes) if any(code==rule if mode=='exact' else code.startswith(rule) if mode=='prefix' else any(code.startswith(p) for p in rule) for code in codes)) for n,(mode,rule) in patterns.items()})
 for n,(mode,rule) in patterns.items():
  hit=any(code==rule if mode=='exact' else code.startswith(rule) if mode=='prefix' else any(code.startswith(p) for p in rule) for code in codes);z[n]=1 if hit else 0 if codes and z['medication_rows_without_valid_ATC5']==0 else None
 medrows.append(z)
medview=pd.DataFrame(medrows);base=base.merge(medview,on=K,how='left',validate='one_to_one')
for n in patterns:base[n]=pd.to_numeric(base[n],errors='coerce').astype('Int64');register(n,'sensitivity_only_until_medication_timing_protocol_frozen',source=str(R/'medications/medications.parquet'),coding='1=listed valid ATC5; 0=not listed among fully coded supplied records, NOT no use; NA=no source or incomplete code mapping',actual_use_at_food_window='start_year/start_month/start_date are retained as uncertain start intervals; no stop/continuation information, so active use remains unproven',timing_table=str(Q/'medication_timing_TRE_ONLY.parquet'),pattern=patterns[n])
base['medication_recording_time_vs_food']=[time_relation(r.medication_record_date_min,r.medication_record_date_max,r.food_first,r.food_last) for r in base.itertuples()]
# Self-reported participant links support conservative group blocking; absence does not prove unrelatedness.
def ids(x):
 if x is None or (isinstance(x,float) and math.isnan(x)):return [],'missing'
 if isinstance(x,np.ndarray):x=x.tolist()
 if isinstance(x,(tuple,list,set)):
  result=[]
  for z in x:
   got,status=ids(z)
   if status not in ['valid','missing']:return [],'unparsed'
   result.extend(got)
  return sorted(set(result)),'valid' if result else 'missing'
 if isinstance(x,(int,float,np.integer,np.floating)):
  return ([int(x)],'valid') if np.isfinite(x) and float(x).is_integer() and x>0 else ([], 'invalid_numeric')
 if isinstance(x,str):
  s=x.strip()
  if s.lower() in BAD:return [],'missing'
  if re.fullmatch(r'[1-9][0-9]*(?:\.0)?',s):return [int(float(s))],'valid'
  if s.startswith(('[','(')):
   try:return ids(ast.literal_eval(s))
   except (ValueError,SyntaxError):return [],'unparsed'
  if re.fullmatch(r'[0-9.,; ]+',s):
   parts=re.split(r'[,;]',s);result=[]
   for z in parts:
    got,st=ids(z)
    if st!='valid':return [],'unparsed'
    result+=got
   return sorted(set(result)),'valid' if result else 'missing'
 return [],'unparsed'
parent={}
def find(x):
 parent.setdefault(x,x)
 while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
 return x
edges=[];parsestatus=Counter();person_flags=defaultdict(lambda:defaultdict(set))
for row in fam.to_dict('records'):
 p=(row['cohort'],int(row['participant_id']));find(p)
 for flag,col,kind in [('study_family_member','study_family_member_participant_id','family'),('study_household_member','study_household_member_participant_id','household')]:
  value=row.get(flag)
  if value is not None and not pd.isna(value):person_flags[p][kind].add(str(value))
  others,st=ids(row.get(col));parsestatus[(kind,st)]+=1
  for other in others:
   if other==p[1]:parsestatus[(kind,'self_link')]+=1;continue
   q=(p[0],other);a,b=find(p),find(q)
   if a!=b:parent[max(a,b)]=min(a,b)
   edges.append({'cohort':p[0],'participant_id':p[1],'related_participant_id':other,'link_kind':kind,'evidence':'self_reported_study_participant_ID_not_genetic_validation'})
edge=pd.DataFrame(edges,columns=['cohort','participant_id','related_participant_id','link_kind','evidence']).drop_duplicates();membership=defaultdict(set)
for p in list(parent):membership[find(p)].add(p)
base['observed_related_group_id']=[hashlib.sha256(json.dumps(find((r.cohort,int(r.participant_id)))).encode()).hexdigest()[:24] for r in base.itertuples()];base['observed_related_component_size']=[len(membership.get(find((r.cohort,int(r.participant_id))),{(r.cohort,int(r.participant_id))})) for r in base.itertuples()];base['relatedness_fully_ascertained']=False;base['family_link_scope']='self-reported parseable participant links only; unlinked does not mean genetically unrelated'
# Preserve source-coverage distinctions and no-data rows without global complete-case deletion.
fv=consensus(fam,['study_family_member','study_household_member']).rename(columns={'source_rows':'family_questionnaire_source_rows','source_date_min':'family_questionnaire_date_min','source_date_max':'family_questionnaire_date_max'});base=base.merge(fv,on=K,how='left',validate='one_to_one');register('observed_related_group_id','split_grouping_not_regression_adjustment',source=str(R/'sociodemographics/ukbb.parquet'),includes='connected components of self-reported family and household participant links across visits within cohort',not_complete_genetic_relatedness=True)
for n in ['total_energy','diet_window_calendar','CGM_device','microbiome_sequencing_depth_batch']:
 register(n,'module_specific_interface_not_materialized_here',source='W08/W09/W12/W13',rule='derive only within each module window/selected sample; no whole-window substitution; no learned transform fit in W11')
optional=[]
for concept,pattern in [('physical_activity','physical|exercise|activity'),('sleep','sleep|insomnia'),('oral_health','dental|periodont|gingiv|gum|teeth'),('antibiotic_timing','antibiot')]:
 matches=[{'field':e['row'].get('tabular_field_name'),'source_dictionary':e['file'],'description':e['row'].get('description_string','')} for e in E if not any(t in e['file'] for t in ['oral_microbiome','gut_microbiome']) and re.search(pattern,str(e['row']),re.I)];optional.append({'concept':concept,'candidate_dictionary_rows':matches,'status':'available_dictionary_candidates_not_selected_or_time_validated' if matches else 'not_found_in_inspected_dictionary_scope','not_automatically_missing_individuals':True})
register('other_metabolic_outcomes_BMI_weight','not_automatic_mutual_adjustment',rule='No concurrent outcome is silently added to all modules; weight-derived energy plausibility ratio is not a body-fat confounder by default')
base['core_age_sex_available']=base.age_calendar_years.notna()&base.sex_code.notna();base['core_age_sex_smoking_available']=base.core_age_sex_available&base[smoke+'_available'];assert not base.duplicated(K).any()
parquet(base,'covariate_registry_TRE_ONLY.parquet',P/'analysis/shared/covariate_registry_TRE_ONLY.parquet');parquet(edge,'relatedness_edges_TRE_ONLY.parquet',Q/'relatedness_edges_TRE_ONLY.parquet')
# Standalone all-source ingredient projections remain inside TRE for exact reproducibility of W11.
summary={'registry_person_visits':len(base),'registry_people':int(base.participant_id.nunique()),'baseline_registry_people':int(base.loc[base.cohort.eq('10k')&base.research_stage.eq('00_00_visit'),'participant_id'].nunique()),'baseline_counts':{},'medication_start_precision_counts':med.start_time_precision_status.value_counts().astype(int).to_dict(),'medication_start_vs_food_counts':med.start_interval_vs_food.value_counts().astype(int).to_dict(),'source_duplicate_person_visits':{'lifestyle':int(life.groupby(K).size().gt(1).sum()),'family_questionnaire':int(fam.groupby(K).size().gt(1).sum()),'population':int(pop.groupby(PK).size().gt(1).sum())},'family_edges':len(edge),'family_edge_types':edge.link_kind.value_counts().astype(int).to_dict(),'family_parse_status':{str(k):v for k,v in parsestatus.items()},'known_connected_components_with_multiple_members':sum(len(v)>1 for v in membership.values()),'unlinked_not_assumed_unrelated':True,'alcohol_selected':alcohol,'no_imputation_or_spline_PCA_fit':True,'no_final_analysis_N':True}
b=base.loc[base.cohort.eq('10k')&base.research_stage.eq('00_00_visit')];variables=['age_calendar_years','sex_code',smoke+'_label','core_age_sex_available','core_age_sex_smoking_available']+list(patterns)+[c+'_numeric' for c in numeric_life]+([alcohol+'_label'] if alcohol else [])+([oral_item+'_label'] if oral_item+'_label' in base else [])
missing=[]
for n in variables:
 valid=b[n].eq(True).fillna(False) if n.startswith('core_') else b[n].notna();summary['baseline_counts'][n]={'available':int(valid.sum()),'missing_or_unavailable':int((~valid).sum())};missing.append({'variable':n,'baseline_denominator':len(b),'available':int(valid.sum()),'missing_or_unavailable':int((~valid).sum())})
summary['baseline_medication_record_time_vs_food']=b.medication_recording_time_vs_food.value_counts(dropna=False).astype(int).to_dict();summary['baseline_lifestyle_record_time_vs_food']=b.lifestyle_recording_time_vs_food.value_counts(dropna=False).astype(int).to_dict();summary['baseline_smoking_status_counts']=b[smoke+'_label'].fillna('unknown').value_counts().astype(int).to_dict();summary['baseline_medication_source_visits']=int(b.medication_source_rows.notna().sum());summary['baseline_in_observed_related_group_size_gt_one']=int(b.observed_related_component_size.gt(1).sum())
rolesdoc={'version':'W11_started20261028_finished20261029','key':K,'universe':'union of explicit visits across necessary sources, not cross-outcome complete cases','roles':roles,'optional_availability_inventory':optional,'user_same_visit_policy_preserved':True,'age_is_calendar_year_approximation':True,'unknown_not_no':True,'host_interaction_candidates_before_protocol_freeze':['D×age_calendar_years','D×sex_code','D×key_medication_class_only_if_active_timing_semantics_validated'],'training_only_fit_rules':{'continuous_splines':'candidate 3df; knots fitted only within corresponding training fold','scaling_imputation_encoding':'fit in training fold only; W11 does not fit parameters','medication_time_missing':'observed start intervals retained; imputation cannot reconstruct unobserved stopping or continuation'},'main_adjustment_protocol_not_frozen':True};dump(P/'config/covariate_roles.yaml',rolesdoc);dump(Q/'w11_summary_TRE_ONLY.json',summary);dump(Q/'w11_input_manifest_TRE_ONLY.json',inputs)
path=P/'data_qc/covariate_missingness.csv';assert not path.exists();pd.DataFrame(missing).to_csv(path,index=False);artifacts.append({'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)});dump(Q/'w11_artifact_manifest_TRE_ONLY.json',artifacts)
print('W11_BUILD_SUCCESS',json.dumps(summary),flush=True)
