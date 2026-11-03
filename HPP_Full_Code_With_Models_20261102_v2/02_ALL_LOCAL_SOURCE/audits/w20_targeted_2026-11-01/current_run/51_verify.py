"""Independent formula, source, role, date and regression readback; TRE only."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import io,json,hashlib,datetime,sys,subprocess,platform
import numpy as np,pandas as pd,pyarrow.parquet as pq,pyarrow.dataset as ds,scipy
from scipy import stats
from w19_baseline_core import HostEncoder
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');B=P/'data_qc/w20_simplified_20261101_v2';V=B/'results';N=P/'data_qc/w20_targeted_20261101_v3';O=N/'results/attempt49b';S=Path(__file__).parent;L=Path('/home/ec2-user/multisystem_runtime_20261021/w20_targeted_20261101_v3')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def rd(p):return pq.ParquetFile(p,pre_buffer=False).read().to_pandas(ignore_metadata=True)
def npz(p):
 with np.load(io.BytesIO(p.read_bytes()),allow_pickle=False) as f:return {k:f[k] for k in f.files}
exits={t:int((L/(t+'_actual_exit_code')).read_text()) for t in ['47','48','49','49b','50b']};assert all(exits[t]==0 for t in ['47','48','49b','50b']) and exits['49']==1
expected=json.loads((S/'expected_v3_sources.json').read_text())
for name,h in expected.items():assert sha(S/name)==h,name
reg=json.loads((N/'registration.json').read_text());assert sha(N/'targeted_spec.md')==reg['source_sha256']['targeted_spec.md']
protected=json.loads((B/'protected_v1_hashes.json').read_text())
for path,h in protected.items():assert sha(path)==h
oldmanifest=json.loads((V/'delivery/artifact_manifest_TRE_ONLY.json').read_text())
for e in oldmanifest:
 p=Path(e['path']);assert p.is_relative_to(P);assert sha(p)==e['sha256'],'old v2 changed: '+str(p)
R=json.loads((O/'model_summary.json').read_text());D=json.loads((O/'daily_summary.json').read_text());X=rd(B/'discovery_covariates_food_TRE_ONLY.parquet');Q=rd(O/'development_predictions_and_scores_TRE_ONLY.parquet');roles=rd(B/'roles_v2_full_TRE_ONLY.parquet');K=['participant_id','cohort','research_stage'];KD=K+['date'];assert not Q.duplicated(K).any();pd.testing.assert_frame_equal(X[K+['split_unit_id','cv_fold']].reset_index(drop=True),Q[K+['split_unit_id','cv_fold']].reset_index(drop=True))
assert X.final_role.eq('discovery').all() and roles.groupby('split_unit_id').final_role.nunique().max()==1
assert not set(X.participant_id)&set(roles.loc[roles.final_role.eq('confirmation_reserved'),'participant_id'])
model_errors=[];daily_errors=[]
def design(key,tr):
 a=npz(V/'fold_preprocessing'/key/'transformed_discovery_TRE_ONLY.npz');diet=json.loads((V/'fold_preprocessing'/key/'diet_basis.json').read_text())['fiber_density'];diet.update(u=a['fiber_u'],h=a['fiber_h']);enc=HostEncoder().fit(X,tr,'fiber_density',diet);host=enc.transform(X);pc=lambda k:np.column_stack([a['oral_PC'][:,:k],a['gut_PC'][:,:k]]);base=np.column_stack([host,pc(R['background_PC'])]);nd=[j for j,l in enumerate(enc.labels) if not l.startswith('diet_')];non=np.column_stack([host[:,nd],pc(R['background_PC'])]);z=np.column_stack([a['oral_z'],a['gut_z']]);return a,diet,base,non,z,pc
for name in ['residual_CLR','residual_PC5','residual_PC10']:
 for key in ['full']+['fold'+str(o) for o in range(5)]:
  f=npz(O/(name+'_full_predictor_TRE_ONLY.npz' if key=='full' else key+'_'+name+'_fit_TRE_ONLY.npz'))
  if key=='full':tr=f['training'];g=f['coef'];valid=tr
  else:
   o=int(key[-1]);tr=npz(O/('daily_'+key+'_oral_formula_TRE_ONLY.npz'))['training'];g=f['coef_all'][[.01,.1,1.,10.].index(R['selection'][name]['lambda'])];valid=X.cv_fold.eq(o).to_numpy()&np.isfinite(Q.Y)
   assert not set(X.loc[tr,'split_unit_id'])&set(X.loc[X.cv_fold.eq(o),'split_unit_id'])
  a,diet,base,non,z,pc=design(key,tr);mod=z if name=='residual_CLR' else pc(5 if name=='residual_PC5' else 10)
  e=non@f['exposure_base']+z@f['exposure_CLR'];W=(diet['u']-e)[:,None]*(mod-f['modifier_center']);pp=base@f['nuisance_base']+z@f['nuisance_CLR']+(W-base@f['W_base_projection']-z@f['W_CLR_projection'])@g
  target=f['prediction'] if key=='full' else Q[name+'_prediction'].to_numpy();err=float(np.max(np.abs(pp[valid]-target[valid])));assert err<2e-8;model_errors.append(err)
  if key=='full':
   # Fitted slope-heterogeneity amplitudes are in observed Y units per fiber+5, not a causal benefit.
   p=a['oral_z'].shape[1] if name=='residual_CLR' else (5 if name=='residual_PC5' else 10)
   for si,site in enumerate(['oral','gut']):
    axis=(mod[:,:p] if si==0 else mod[:,p:])@(g[:p] if si==0 else g[p:]);R.setdefault('full_slope_SD',[]).append({'model':name,'site':site,'SD_per_fiber5':float(axis[tr].std()),'Y_unit':R['Y_unit']})
for o in range(5):
 for site in ['oral','gut']:
  f=npz(O/('daily_fold'+str(o)+'_'+site+'_formula_TRE_ONLY.npz'));tr=f['training'];held=f['held_supported'];a=npz(V/'fold_preprocessing'/('fold'+str(o))/'transformed_discovery_TRE_ONLY.npz');axis=a[site+'_z']@f['weights']-f['center'];assert abs(float(axis[tr].mean()))<1e-9;assert abs(float(axis[tr].std())-1)<1e-9;assert not set(X.loc[tr,'split_unit_id'])&set(X.loc[held,'split_unit_id']);err=float(np.max(np.abs(axis[held]-Q.loc[held,'daily_'+site+'_score'])));assert err<1e-9;daily_errors.append(err)
J=rd(O/'daily_analysis_TRE_ONLY.parquet');J['date']=pd.to_datetime(J.date).dt.normalize();assert not J.duplicated(KD).any();assert not set(J.participant_id)&set(roles.loc[roles.final_role.eq('confirmation_reserved'),'participant_id']);assert J.groupby('participant_id').size().min()>=3
np.testing.assert_allclose(J.groupby('participant_id').person_weight.sum(),1.,atol=1e-12);np.testing.assert_allclose(J.person_weight,1/J.groupby('participant_id').date.transform('size'))
# Independent source value/date check with explicit discovery dataset filter.
filt=(ds.field('cohort')=='10k')&(ds.field('research_stage')=='00_00_visit')&ds.field('participant_id').isin(X.participant_id.tolist())
for p,cols,mapping in [(P/'analysis/Daily_CGM/fiber_density/base_table_TRE_ONLY.parquet',KD+['Y_value'],{'Y_value':'Y_value'}),(P/'data_qc/preprocessing_20261030_v1/food_qc_v1/food_daily_primary_TRE_ONLY.parquet',KD+['fiber_density_g_per_1000kcal','energy_kcal','fiber_eligible','energy_eligible'],{'fiber_density_g_per_1000kcal':'fiber_density_g_per_1000kcal','energy_kcal':'energy_kcal'})]:
 raw=ds.dataset(p,format='parquet').scanner(columns=cols,filter=filt,use_threads=False).to_table().to_pandas(ignore_metadata=True);raw['date']=pd.to_datetime(raw.date).dt.normalize();assert not raw.duplicated(KD).any();m=J.merge(raw,on=KD,how='left',validate='one_to_one',suffixes=('','_source'));assert len(m)==len(J)
 for col in mapping:np.testing.assert_allclose(m[col],m[col+'_source'],atol=1e-10)
 if 'fiber_eligible_source' in m:assert m.fiber_eligible_source.all() and m.energy_eligible_source.all()
# Independent weighted least-squares and household sandwich, no call to fitting implementation.
f=npz(O/'daily_regression_TRE_ONLY.npz');y=f['y'];x=f['x'];w=f['weights'];g=f['groups'];b=np.linalg.lstsq(x*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0];res=y-x@b;bread=np.linalg.inv(x.T@(w[:,None]*x));ug,inv=np.unique(g,return_inverse=True);meat=np.zeros_like(bread)
for j in range(len(ug)):
 ss=(x[inv==j]*(w[inv==j]*res[inv==j])[:,None]).sum(0);meat+=np.outer(ss,ss)
G=len(ug);nn=len(y);pp=x.shape[1];people=J.participant_id.nunique();cov=bread@meat@bread*(G/(G-1))*((nn-1)/(nn-people-pp));np.testing.assert_allclose(b,f['beta'],atol=1e-10);np.testing.assert_allclose(cov,f['cov'],atol=1e-10)
for name in ['v2_ridge','residual_CLR','residual_PC5','residual_PC10']:
 pp=Q['v2_prediction' if name=='v2_ridge' else name+'_prediction'];ok=np.isfinite(Q.Y)&np.isfinite(pp);r2=1-np.sum((Q.Y[ok]-pp[ok])**2)/np.sum((Q.Y[ok]-Q.Y[ok].mean())**2);record=next(a for a in R['models'] if a['model']==name);assert abs(r2-record['R2'])<1e-12
assert R['confirmation_Y_used'] is False and D['confirmation_Y_materialized']==0 and not R['formal55_tests_started'] and not D['formal55_tests_started']
checks={'predictor_formula_readbacks':len(model_errors),'max_prediction_error':max(model_errors),'household_excluded_daily_score_readbacks':len(daily_errors),'max_daily_score_error':max(daily_errors),'old_v1_hashes_preserved':len(protected),'old_v2_artifact_hashes_preserved':len(oldmanifest),'source_hashes_verified':len(expected),'exact_source_calendar_and_values':True,'independent_daily_WLS_sandwich':True,'same_person_total_weight':True,'actual_exits':exits,'versions':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__},'verification_pid':os.getpid(),'time_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'verified'}
summary={'verification':checks,'models':R,'daily':D}
(O/'verified_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
manifest=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(O.iterdir()) if p.is_file()]
(O/'artifact_manifest_TRE_ONLY.json').write_text(json.dumps(manifest,indent=2)+'\n')
lines=['# HPP targeted v3 actual run and verification','', 'Exploratory observational association modification. Formal 55 Holm confirmation has not started.','',f'Python {sys.executable}; actual exits {exits}.',f'Discovery {R["n_discovery"]} people/{R["discovery_families"]} families; CV {R["CV_n"]}/{R["CV_families"]}; full eligible training {R["full_training_n"]}/{R["full_training_units"]}.','', '|Model|CV R2|Delta R2 vs matched null|Conditional 95% CI delta R2|','|---|---:|---:|---|']
for r in R['models']:lines.append(f'|{r["model"]}|{r["R2"]:.7f}|{r["delta_R2"]:+.7f}|{r["delta_R2_CI_conditional"]}|')
lines+=['',f'Daily exact-date cohort: {D["flow"]}.','', '|Daily term|Estimate|95% CI|Y unit|','|---|---:|---|---|']
for r in D['effects']:lines.append(f'|{r["term"]}|{r["beta"]:.7f}|{r["CI95"]}|{r["Y_unit"]}|')
lines+=['','CIs condition on learned scores and do not include direction-learning and development adaptation. They are not independent confirmation. Prediction metrics were used in development selection. Residual CLR includes the tied nonlinear nuisance term -e(C,Z)*Z; prediction change alone cannot establish slope modification. Slopes and daily within-person estimates remain observational.','',f'Verification: {checks}','', 'UPF and all nine endpoints remain in the formal scope. This run covers fiber/D_Glu only, with no change to roles or 55 Holm budget. Do not auto-open confirmation Y.']
(O/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('VERIFIED',json.dumps(checks),flush=True)
print('MODEL people families train_min/max fulltrain',R['CV_n'],R['CV_families'],R['models'][0]['training_min'],R['models'][0]['training_max'],R['full_training_n'],R['Y_unit'])
for r in R['models']:print('MODEL',r['model'],'lambda',r['lambda'],'R2',round(r['R2'],8),'delta',round(r['delta_R2'],8),'CI',[round(v,8) for v in r['delta_R2_CI_conditional']])
print('DAILY_FLOW',json.dumps(D['flow']))
for r in D['effects'][:3]:print('DAILY',r['term'],r['beta'],r['CI95'],r['Y_unit'])
print('DAILY_JOINT',D['joint'],'within deltaR2',D['delta_within_R2_descriptive'])
for r in R['slope_calibration']:print('SLOPE',r['model'],r['site'],r['beta_outcome_per_fiber5_per_scoreSD'],r['CI95'])
print('DONE: verified formulas/date/weights/cluster; confirmation Y unused; 55 Holm unchanged.')
