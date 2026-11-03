# -*- coding: utf-8 -*-
"""Independent source re-reads, key/value reconstruction and specimen graph/lag verification."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
from collections import defaultdict,Counter,deque
from datetime import date,datetime
import hashlib,json,math,re,ast,calendar
import pandas as pd,numpy as np,pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');R=Path('/home/ec2-user/studies/hpp_datasets');Q=P/'data_qc/w11_w12_covariates_samples_20261028';L=Path('/home/ec2-user/multisystem_runtime_20261021/w11_w12_covariates_samples_20261028');K=['participant_id','cohort','research_stage'];BAD={'','undefined','none','null','nan','na','n/a','unknown','<na>'}
actual={t:int((L/(t+'_actual_exit_code')).read_text()) for t in ['01','02','03','04','05','06','07','08','09']};assert all(v==0 for v in actual.values())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def rows(p,cols):
 pf=pq.ParquetFile(p,pre_buffer=False);assert pf.metadata.num_rows<400000;assert set(cols)<=set(pf.schema_arrow.names);return [r for b in pf.iter_batches(columns=cols,batch_size=512,use_threads=False) for r in b.to_pylist()]
def key(r,fields=K):
 if any(r.get(c) is None or str(r[c]).lower() in BAD or str(r[c])!=str(r[c]).strip() for c in fields):return None
 try:
  pid=int(r['participant_id']);assert pid>0 and pid==float(r['participant_id']);return (pid,)+tuple(str(r[c]) for c in fields[1:])
 except (ValueError,TypeError,AssertionError):return None

def ismissing(v):return v is None or (isinstance(v,(float,np.floating)) and math.isnan(v)) or v is pd.NA

def equal(x,y):return ismissing(x) and ismissing(y) or not ismissing(x) and not ismissing(y) and x==y

def ds(v):return v.date().isoformat() if isinstance(v,datetime) else v.isoformat() if isinstance(v,date) else None

def norm(v):
 if v is None:return None
 try:f=float(v);return str(int(f)) if f.is_integer() else str(v)
 except (ValueError,TypeError):return str(v)

M11=json.loads((Q/'w11_current_artifact_manifest_TRE_ONLY.json').read_text());M12=json.loads((Q/'w12_artifact_manifest_TRE_ONLY.json').read_text());rev=json.loads((Q/'w11_artifact_revisions_TRE_ONLY.json').read_text());mirror=0
for e in M11+M12:
 p=Path(e['path']);assert p.stat().st_size==e['bytes'] and sha(p)==e['sha256']
 if e.get('runtime_source'):assert sha(Path(e['runtime_source']))==sha(p);assert pq.ParquetFile(p).metadata.num_rows==e['rows'];mirror+=1
for e in rev:assert sha(Path(e['archived_path']))==e['before_sha256']
for e in json.loads((Q/'w11_input_manifest_TRE_ONLY.json').read_text()):
 p=Path(e['path']);assert p.stat().st_size==e['bytes'] and p.stat().st_mtime_ns==e['mtime_ns'];assert pq.ParquetFile(p).metadata.num_rows==e['rows']
D=pq.ParquetFile(P/'analysis/shared/covariate_registry_TRE_ONLY.parquet').read(use_threads=False).to_pandas(ignore_metadata=True);assert not D.duplicated(K).any();dr={tuple(r[c] for c in K):r for r in D.to_dict('records')};S11=json.loads((Q/'w11_summary_TRE_ONLY.json').read_text());roles=json.loads((P/'config/covariate_roles.yaml').read_text());assert len(D)==S11['registry_person_visits']
# Reconstruct the UNION universe independently from each declared source, not from output eligibility flags.
source_universe=[P/'analysis/shared/food_daily_TRE_ONLY.parquet',P/'analysis/shared/outcome_registry_TRE_ONLY.parquet',P/'data_qc/w10_outcome_qc_20261028/cgm_outcome_availability_TRE_ONLY.parquet',R/'anthropometrics/anthropometrics.parquet',R/'lifestyle_and_environment/lifestyle_and_environment.parquet',R/'sociodemographics/ukbb.parquet',R/'medications/medications.parquet']
manifest=json.loads((P/'manifests/w02_sources_20261026T130908Z/source_manifest.json').read_text());mp={s['role']:Path(s['path']) for s in manifest['sources']}
source_universe += [mp[t] for t in ['gut_index','gut_species','oral_index']];universe=set();rawkeycounts={}
for p in source_universe:
 rr=rows(p,K);kk=[key(r) for r in rr];good={k for k in kk if k is not None};universe|=good;rawkeycounts[str(p)]={'raw_rows':len(rr),'valid_key_rows':sum(k is not None for k in kk),'unique_person_visits':len(good)}
assert universe==set(dr)
foodrows=rows(P/'analysis/shared/food_daily_TRE_ONLY.parquet',K+['date']);food=defaultdict(set)
for r in foodrows:
 k=key(r)
 if k:food[k].add(date.fromisoformat(r['date']))
a_dates=defaultdict(list)
for r in rows(R/'anthropometrics/anthropometrics.parquet',K+['collection_date']):
 if key(r) and r['collection_date'] is not None:a_dates[key(r)].append(r['collection_date'])
cg_dates={key(r):date.fromisoformat(r['first_cgm_day']) for r in rows(P/'data_qc/w10_outcome_qc_20261028/cgm_outcome_availability_TRE_ONLY.parquet',K+['first_cgm_day']) if key(r)}
pop=defaultdict(lambda:defaultdict(set));popraw=rows(R/'population/population.parquet',['participant_id','cohort','year_of_birth','sex'])
for r in popraw:
 k=key(r,['participant_id','cohort'])
 if k:
  for f in ['year_of_birth','sex']:
   if r[f] is not None and str(r[f]).lower() not in BAD:pop[k][f].add(str(r[f]))
for k,r in dr.items():
 anchor=None
 if k in food:anchor=pd.Timestamp(min(food[k]))+(pd.Timestamp(max(food[k]))-pd.Timestamp(min(food[k])))/2
 elif k in a_dates:anchor=pd.Timestamp(min(a_dates[k]))
 elif k in cg_dates:anchor=pd.Timestamp(cg_dates[k])
 assert (anchor is None and pd.isna(r['anchor_date'])) or anchor==r['anchor_date']
 sy=pop[k[:2]]['sex'];sx=float(next(iter(sy))) if len(sy)==1 else None;sx=int(sx) if sx in [0,1] else None;assert equal(r['sex_code'],sx)
 by=pop[k[:2]]['year_of_birth'];yy=float(next(iter(by))) if len(by)==1 else None;age=anchor.year-yy if anchor is not None and yy is not None and math.isfinite(yy) and yy.is_integer() and yy<=anchor.year else None;assert equal(r['age_calendar_years'],age)
 assert r['core_age_sex_available']==(age is not None and sx is not None)
# Verify every retained questionnaire item against raw per-visit consensus and the source codebook.
lifefields=['smoking_current_status','alcohol_current_frequency','activity_walking_10min_days_weekly','activity_moderate_days_weekly','activity_vigorous_days_weekly','sleep_hours_daily'];lifefields=[f for f in lifefields if f in D];rawlife=rows(R/'lifestyle_and_environment/lifestyle_and_environment.parquet',K+lifefields);lg=defaultdict(lambda:defaultdict(set))
for r in rawlife:
 k=key(r)
 if not k:continue
 for f in lifefields:
  if r[f] is not None and str(r[f]).lower() not in BAD:lg[k][f].add(str(r[f]))
byrole={r['variable']:r for r in roles['roles']};numcheck=0
for k,r in dr.items():
 for f in lifefields:
  values=lg[k][f];value=next(iter(values)) if len(values)==1 else None
  if f in ['smoking_current_status','alcohol_current_frequency']:
   code=norm(value);cm=byrole[f]['coding'];label=cm.get(code) if code is not None and not code.startswith('-') and code in cm and not re.search('prefer not|do not know|refus',cm[code],re.I) else None;assert equal(r[f+'_label'],label)
  else:
   v=float(value) if value is not None else None;upper=24 if f=='sleep_hours_daily' else 7;v=v if v is not None and math.isfinite(v) and 0<=v<=upper else None;assert equal(r[f+'_numeric'],v)
  numcheck+=1
 assert bool(r['core_age_sex_smoking_available'])==(bool(r['core_age_sex_available']) and not ismissing(r['smoking_current_status_label']))
# Full source medication code re-read, independently reconstructed positive/not-listed/unknown flags.
def flatten(v):
 if v is None:return []
 if isinstance(v,(list,tuple)):
  return sorted({c for x in v for c in flatten(x)})
 if isinstance(v,str) and re.fullmatch(r'[A-Z][0-9]{2}[A-Z]{2}[0-9]{2}',v):return [v]
 if isinstance(v,str) and v.startswith(('[','(')):
  try:return flatten(ast.literal_eval(v))
  except (ValueError,SyntaxError):return []
 return []
medcols=K+['collection_date','medication','atc5','start_year','start_month','start_date'];rawmed=[r for r in rows(R/'medications/medications.parquet',medcols) if key(r)];mg=defaultdict(list)
for r in rawmed:mg[key(r)].append(r)
patterns={r['variable']:r['pattern'] for r in roles['roles'] if 'pattern' in r};medchecks=0
for k,r in dr.items():
 group=mg[k];cc=[flatten(x['atc5']) for x in group];codes={c for row in cc for c in row}
 for f,(mode,rule) in patterns.items():
  yes=any(c==rule if mode=='exact' else c.startswith(rule) if mode=='prefix' else any(c.startswith(p) for p in rule) for c in codes);expected=1 if yes else 0 if codes and all(bool(x) for x in cc) else None;assert equal(r[f],expected);medchecks+=1
  if not group:assert ismissing(r[f])
 if group:assert r['medication_source_rows']==len(group)
 else:assert pd.isna(r['medication_source_rows'])
T=rows(Q/'medication_timing_TRE_ONLY.parquet',medcols+['start_interval_lower','start_interval_upper','start_time_precision_status','start_interval_vs_food','food_first','food_last']);assert len(T)==len(rawmed)
def simple(v):
 if isinstance(v,(datetime,date)):return v.isoformat()
 if isinstance(v,list):return [simple(x) for x in v]
 if isinstance(v,float) and math.isfinite(v) and v.is_integer():return int(v)
 return v
sig=lambda r:json.dumps([simple(r.get(c)) for c in medcols],default=str)
assert Counter(sig(r) for r in T)==Counter(sig(r) for r in rawmed)
for t in T:
 lo,hi,st=t['start_interval_lower'],t['start_interval_upper'],t['start_time_precision_status']
 if st in ['reported_year_month_interval','provider_date_month_used_day_precision_unverified']:
  assert lo is not None and hi is not None and lo.day==1 and hi.day==calendar.monthrange(lo.year,lo.month)[1] and lo.year==hi.year and lo.month==hi.month
 elif st=='reported_year_interval':assert lo.month==1 and lo.day==1 and hi.month==12 and hi.day==31 and lo.year==hi.year
 else:assert lo is None and hi is None
 first,last=t['food_first'],t['food_last'];expected='unknown' if any(x is None for x in [lo,hi,first,last]) else 'start_before_food_window' if hi<first else 'start_after_food_window' if lo>last else 'start_within_food_window' if lo>=first and hi<=last else 'start_interval_spans_food_window_boundaries';assert expected==t['start_interval_vs_food']
# Source self-reported family/household edges, full graph check incl. external referenced IDs.
fcols=K+['study_family_member_participant_id','study_household_member_participant_id'];rawfamily=rows(R/'sociodemographics/ukbb.parquet',fcols);expected_edges=set();nodes=set()
for r in rawfamily:
 k=key(r)
 if not k:continue
 nodes.add(k[:2][::-1]) # (cohort, participant_id)
 for f,kind in [('study_family_member_participant_id','family'),('study_household_member_participant_id','household')]:
  q=r[f]
  if q is not None and isinstance(q,(int,float)) and q>0 and float(q).is_integer() and int(q)!=k[0]:expected_edges.add((k[1],k[0],int(q),kind))
edges=rows(Q/'relatedness_edges_TRE_ONLY.parquet',['cohort','participant_id','related_participant_id','link_kind']);assert {(e['cohort'],e['participant_id'],e['related_participant_id'],e['link_kind']) for e in edges}==expected_edges;adj=defaultdict(set)
for cohort,a,b,kind in expected_edges:adj[(cohort,a)].add((cohort,b));adj[(cohort,b)].add((cohort,a))
for k in dr:nodes.add((k[1],k[0]))
components={};seen=set()
for p in sorted(nodes|set(adj)):
 if p in seen:continue
 todo=[p];component=set()
 while todo:
  v=todo.pop()
  if v in component:continue
  component.add(v);todo.extend(adj[v]-component)
 seen|=component;root=min(component)
 for v in component:components[v]=(root,len(component))
for k,r in dr.items():
 root,size=components[(k[1],k[0])];expected=hashlib.sha256(json.dumps(root).encode()).hexdigest()[:24];assert r['observed_related_group_id']==expected and r['observed_related_component_size']==size and not r['relatedness_fully_ascertained']
# Optional actual-spelling oral-health field, full decoded source consensus.
OH=json.loads((Q/'oral_health_completion_TRE_ONLY.json').read_text());oralchecks=0
if OH['completed']:
 f=OH['source_field'];rr=rows(Path(OH['source_path']),K+[f]);gg=defaultdict(set)
 for r in rr:
  if key(r) and r[f] is not None:gg[key(r)].add(norm(r[f]))
 cm=byrole[f]['coding'];valid={k:v for k,v in cm.items() if not k.startswith('-') and not any(t in v.lower() for t in ['prefer not','do not know','refus'])}
 for k,r in dr.items():
  vals=gg[k];label=valid.get(next(iter(vals))) if len(vals)==1 else None;assert equal(r[f+'_label'],label);oralchecks+=1
# W12 source versions, complete row conservation, independent oral/gut linkage decisions.
C=pq.ParquetFile(P/'analysis/shared/sample_candidates_TRE_ONLY.parquet').read(use_threads=False).to_pylist();rules=json.loads((P/'config/sample_selection_rules.yaml').read_text());sources=rules['source_versions_fixed'];src={}
for role,s in sources.items():
 p=Path(s['path']);assert sha(p)==s['sha256'] and p.stat().st_mtime_ns==s['mtime_ns'];fields=K+['array_index']+(['collection_date'] if 'collection_date' in pq.ParquetFile(p).schema_arrow.names else [])+(['sample_name'] if 'sample_name' in pq.ParquetFile(p).schema_arrow.names else []);src[role]=rows(p,fields)
assert len({r['candidate_id'] for r in C})==len(C);assert not any(r['selected_primary_sample'] for r in C);assert not rules['technical_QC_executed'] and rules['random_seed']==20261027
cg=defaultdict(set)
for r in rows(P/'analysis/shared/cgm_daily_TRE_ONLY.parquet',K+['date','cgm_day_eligible']):
 if key(r) and r['cgm_day_eligible'] is True:cg[key(r)].add(date.fromisoformat(r['date']))
for site in ['oral','gut']:
 ix=src[site+'_index'];ab=src[site+'_species'];ik=defaultdict(list);ipd=defaultdict(list);names=Counter()
 for i,r in enumerate(ix,1):
  if key(r):ik[key(r)].append(i)
  if r.get('collection_date') is not None:ipd[(str(r['participant_id']),ds(r['collection_date']))].append(i)
  if r.get('sample_name') is not None:names[r['sample_name']]+=1
 counts=Counter(key(r) for r in ab if key(r));sc=[r for r in C if r['site']==site];ac={r['abundance_row_ordinal']:r for r in sc if r['candidate_kind']=='abundance_row'};assert set(ac)==set(range(1,len(ab)+1));allrefs=set()
 for c in sc:
  allrefs.update(json.loads(c['candidate_index_ordinals_json']));assert c['technical_QC_status']=='not_evaluated_W13';assert c['random_priority_after_required_identity_and_technical_QC']==hashlib.sha256((str(20261027)+'|'+site+'|'+c['candidate_id']).encode()).hexdigest()
  if c['candidate_kind']!='abundance_row':
   assert not c['identity_ready_for_W13'];origin=ix[c['linked_index_row_ordinal']-1];assert key(c)==key(origin) and c['linked_index_calendar_date']==ds(origin.get('collection_date'));continue
  a=ab[c['abundance_row_ordinal']-1];possible=[]
  if site=='oral':possible=[i for i in ipd.get((str(a['participant_id']),ds(a.get('collection_date'))),[]) if key(ix[i-1]) and all(a.get(f) is None or str(a[f]).lower() in BAD or str(a[f])==str(ix[i-1][f]) for f in ['cohort','research_stage'])];link=possible[0] if len(possible)==1 else None
  else:
   ka=key(a);possible=ik.get(ka,[]) if ka else [];link=possible[0] if len(possible)==1 and counts[ka]==1 else None
  assert json.loads(c['candidate_index_ordinals_json'])==possible
  if link is not None:
   assert c['linked_index_row_ordinal']==link and key(c)==key(ix[link-1]);assert c['linked_index_calendar_date']==ds(ix[link-1].get('collection_date')) and c['descriptive_sample_calendar_date']==ds(ix[link-1].get('collection_date'));expected_ready=names.get(ix[link-1].get('sample_name'),0)<=1;assert c['identity_ready_for_W13']==expected_ready
  else:assert c['linked_index_row_ordinal'] is None and not c['identity_ready_for_W13']
  # Brute-force nearest day recomputation, independent of producer bisect implementation.
  stamp=c['descriptive_sample_calendar_date'];sample=date.fromisoformat(stamp) if stamp else None;k=key(c)
  for label,days in [('food',food),('CGM',cg)]:
   field='sample_minus_nearest_observed_food_day' if label=='food' else 'sample_minus_nearest_qualified_CGM_day';available=days.get(k,set());expected=(sample-min(available,key=lambda t:(abs((sample-t).days),t))).days if sample and available else None;assert equal(c[field],expected)
 assert allrefs==set(range(1,len(ix)+1))
# Recompute all reported missingness counts and candidate counts from final individual tables.
b=D.loc[D.cohort.eq('10k')&D.research_stage.eq('00_00_visit')]
for n,v in S11['baseline_counts'].items():
 m=b[n].eq(True).fillna(False) if n.startswith('core_') else b[n].notna();assert int(m.sum())==v['available'] and int((~m).sum())==v['missing_or_unavailable']
S12=json.loads((Q/'w12_summary_TRE_ONLY.json').read_text());assert len(C)==S12['candidate_rows']
result={'passed':True,'scope':'W11 covariates and W12 candidate identities, not W13 matrix QC/final samples','actual_exit_codes_before_verifier':actual,'covariate_person_visits_reverified':len(D),'questionnaire_values_reverified':numcheck,'medication_class_cells_reverified':medchecks,'medication_start_rows_reverified':len(T),'family_edges_independently_rebuilt':len(edges),'oral_health_values_reverified':oralchecks,'sample_candidates_reverified':len(C),'oral_source_abundance_rows_reverified':len(src['oral_species']),'gut_source_abundance_rows_reverified':len(src['gut_species']),'candidate_lags_full_bruteforce_verified':True,'current_artifact_hashes_verified':len(M11)+len(M12),'preserved_prior_artifact_revisions_verified':len(rev),'binary_mirrors_verified':mirror,'union_universe_not_complete_case_intersection_verified':True,'source_key_counts':rawkeycounts,'no_model_fit_or_imputation':True,'final_primary_samples_selected':0,'W13_technical_QC_done':False,'raw_data_exported':False};out=Q/'independent_verification_TRE_ONLY.json';assert not out.exists();out.write_text(json.dumps(result,indent=2));print('INDEPENDENT_W11_W12_PASS',json.dumps({k:v for k,v in result.items() if k!='source_key_counts'}),flush=True)
print('W11_BASELINE_DENOMINATOR',len(b),flush=True)
for n,v in S11['baseline_counts'].items():print('COV_COUNT',n,v['available'],v['missing_or_unavailable'],flush=True)
print('ORAL_HEALTH_SOURCE',json.dumps(OH),flush=True)
print('W12_IDENTITY_COUNTS',json.dumps(S12['per_site']),flush=True);print('W11_W12_VERIFIER_DONE',flush=True)
