"""Independent medication class joins and diet-only joint description. No association with Y."""
import runpy
from pathlib import Path
B=runpy.run_path(str(Path(__file__).with_name('03_source_QC.py')),run_name='auditlib');globals().update({k:v for k,v in B.items() if not k.startswith('__')})
import re
T=read('medication_timing',purpose='X05 independently classify and join medication evidence');rawpath=Path('/home/ec2-user/studies/hpp_datasets/medications/medications.parquet');raw=readpath(rawpath,K+['atc5'],purpose='X05 raw versus timing class/key provenance');rules=json.loads((P/'config/medication_rules_W14_20261029.json').read_text())['class_patterns']
def atcset(x):
 if isinstance(x,(np.ndarray,list,tuple,set)):
  return set().union(*(atcset(y) for y in x)) if len(x) else set()
 return set(re.findall(r'(?<![A-Z0-9])[A-Z][0-9]{2}[A-Z]{2}[0-9]{2}(?![A-Z0-9])',str(x)))
def classify(d,dates):
 rows=[]
 for i,r in enumerate(d.to_dict('records')):
  cs=atcset(r['atc5'])
  for cl,pat in rules.items():
   if any(c==p if cl=='metformin' else c.startswith(p) for c in cs for p in pat):rows.append({**{k:r[k] for k in K},'medication_class':cl,'source_row':i,**({'lo':r['start_interval_lower'],'hi':r['start_interval_upper']} if dates else {})})
 return pd.DataFrame(rows)
C=classify(T,True);RC=classify(raw,False);cmp=C.groupby(K+['medication_class']).size().rename('timing_rows').to_frame().join(RC.groupby(K+['medication_class']).size().rename('raw_rows'),how='outer');source_mismatch=int(cmp.fillna(0).timing_rows.ne(cmp.fillna(0).raw_rows).sum());C['lo']=pd.to_datetime(C.lo,errors='coerce').dt.normalize();C['hi']=pd.to_datetime(C.hi,errors='coerce').dt.normalize();medkey=keys(T);cache={};errors=Counter();counts=Counter();evidence=Counter();windowkeys=set();nonemptykeys=set();classkeys=set();checked=0
mp=Path(M['inputs']['medication_window_status']['path']);assert sha(mp)==M['inputs']['medication_window_status']['sha256'];f=pq.ParquetFile(mp,pre_buffer=False);cols=K+['window_type','window_first','window_last','medication_class','state','listed_rows','initiation_in_window_rows','evidence'];save('access_receipts/'+uuid.uuid4().hex+'.json',{'time_UTC':datetime.now(timezone.utc).isoformat(),'script':str(Path(__file__)),'script_sha256':sha(__file__),'path':str(mp),'source_sha256':sha(mp),'columns':cols,'source_rows':f.metadata.num_rows,'purpose':'X05 independent projected batch join with all published window states','technical_Y_value_read':False,'scientific_association_or_training':False,'individual_data_exported':False})
for batch in f.iter_batches(columns=cols,batch_size=16384,use_threads=False):
 d=batch.to_pandas(ignore_metadata=True);ck=K+['window_first','window_last','medication_class'];u=d[ck].drop_duplicates();new=[]
 for v in u.itertuples(index=False,name=None):
  key=tuple(None if pd.isna(x) else x for x in v)
  if key not in cache:new.append(key)
 if new:
  w=pd.DataFrame(new,columns=ck);w['request_id']=np.arange(len(w));w['wl']=pd.to_datetime(w.window_first,errors='coerce');w['wh']=pd.to_datetime(w.window_last,errors='coerce');z=w.merge(C,on=K+['medication_class'],how='left',validate='many_to_many');z['listed']=z.source_row.notna();z['proved']=z.listed&z.wl.notna()&z.wh.notna()&z.lo.notna()&z.hi.notna()&z.wl.le(z.lo)&z.lo.le(z.hi)&z.hi.le(z.wh);g=z.groupby('request_id')[['listed','proved']].sum()
  for i,v in enumerate(new):
   nr=int(g.loc[i,'listed']);pr=int(g.loc[i,'proved']);hasdates=v[3] is not None and v[4] is not None;hasmed=tuple(v[:3]) in medkey;ev='start_interval_wholly_inside_window' if pr else 'no_window_dates' if not hasdates else 'no_medication_records' if not hasmed else 'listed_but_continuation_unobserved_or_start_uncertain' if nr else 'class_not_listed_is_not_confirmed_no_use';cache[v]=(nr,pr,'exposed_in_window' if pr else 'unknown',ev)
 for r in d.itertuples(index=False):
  pk=tuple(getattr(r,k) for k in K);key=pk+tuple(None if pd.isna(getattr(r,k)) else getattr(r,k) for k in ['window_first','window_last','medication_class']);nr,pr,st,ev=cache[key];windowkeys.add(pk)
  if pk in medkey:nonemptykeys.add(pk)
  if nr:classkeys.add(pk)
  for name,a,b in [('listed_rows',r.listed_rows,nr),('initiation_rows',r.initiation_in_window_rows,pr),('state',r.state,st),('evidence',r.evidence,ev)]:errors[name]+=int(a!=b)
  counts[(r.window_type,r.state)]+=1;evidence[(r.window_type,r.medication_class,r.evidence)]+=1;checked+=1
out={'published_rows_checked':checked,'independent_classification_source_rows':len(T),'raw_source_rows':len(raw),'raw_vs_timing_class_key_count_mismatches':source_mismatch,'raw_ATC_class_row_counts':RC.medication_class.value_counts().astype(int).to_dict(),'source_start_precision_counts':T.start_time_precision_status.value_counts(dropna=False).astype(int).to_dict(),'mismatches':dict(errors),'state_counts':[{'window':w,'state':s,'rows':n} for (w,s),n in sorted(counts.items())],'window_person_visits':len(windowkeys),'window_person_visits_with_any_medication_rows':len(nonemptykeys),'window_person_visits_with_listed_target_class':len(classkeys),'unique_window_class_requests_independently_joined':len(cache),'evidence_counts':[{'window':w,'class':c,'evidence':e,'rows':n} for (w,c,e),n in sorted(evidence.items())],'confirmed_no_use_requires_unavailable_explicit_absence_evidence':True,'unknown_states_changed':False}
save('X05_medication_independent_merge.json',out);print('X05_MEDICATION',json.dumps({k:v for k,v in out.items() if k!='evidence_counts'}),flush=True)
F=read('food_windows');FD=read('food_daily',K+['date','energy_eligible','fiber_eligible','UPF_lower_pp','UPF_upper_pp','fiber_density_g_per_1000kcal']);S=read('sample_registry',K+['site']);og=keys(S.loc[S.site.eq('oral')])&keys(S.loc[S.site.eq('gut')]);baseline=F.cohort.eq('10k')&F.research_stage.eq('00_00_visit');joint=F.fiber_equal_day_mean.notna()&F.UPF_lower_equal_day_mean_all_energy_days.notna()&F.UPF_upper_equal_day_mean_all_energy_days.notna();rows=[]
for label,mask in [('all_baseline_food_windows',baseline),('baseline_with_locked_oral_and_gut',baseline&pd.Series([k in og for k in F[K].itertuples(index=False,name=None)]))]:
 z=F.loc[mask&joint];r={'scope':label,'joint_window_people':len(z),'UPF_bound_vs_fiber_Spearman':{}}
 for c in ['UPF_lower_equal_day_mean_all_energy_days','UPF_upper_equal_day_mean_all_energy_days']:r['UPF_bound_vs_fiber_Spearman'][c]=float(z[c].rank().corr(z.fiber_equal_day_mean.rank())) if len(z)>1 else None
 rows.append(r)
d=FD.loc[FD.cohort.eq('10k')&FD.research_stage.eq('00_00_visit')&FD.energy_eligible&FD.fiber_eligible];diet={'joint_window_descriptions':rows,'baseline_actual_days_jointly_energy_and_fiber_qualified':len(d),'baseline_people_with_jointly_qualified_days':len(d[K].drop_duplicates()),'no_midpoint_UPF_used':True,'CGM_or_other_outcome_columns_read_for_joint_description':False,'descriptive_only_no_hypothesis_test_or_model':True};save('X09_food_window_joint_description.json',diet);print('X09_DIET_JOINT',json.dumps(diet),flush=True);print('MEDICATION_DIET_FINISHED','mismatches',sum(errors.values())+source_mismatch,flush=True)
