"""Authorized bounded discovery comparison and household-excluded daily scores."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,io,time,hashlib,resource
import numpy as np,pandas as pd,pyarrow.parquet as pq,pyarrow as pa
from scipy import stats
from w19_baseline_core import HostEncoder,phenotype,metrics
from v2_ridge_core import BlockDesign
from targeted_core import ResidualVC,cluster_fit
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');B=P/'data_qc/w20_simplified_20261101_v2';V=B/'results';N=P/'data_qc/w20_targeted_20261101_v3';O=N/'results';S=Path(__file__).parent
assert json.loads((N/'registration.json').read_text())['status']=='registered_before_new_fits'
resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(7200,7200));start=time.monotonic()
def rd(p):return pq.ParquetFile(p,pre_buffer=False).read().to_pandas(ignore_metadata=True)
def save(d,p):
 assert not p.exists();pq.write_table(pa.Table.from_pandas(d.reset_index(drop=True),preserve_index=False),p,compression='zstd');pd.testing.assert_frame_equal(d.reset_index(drop=True),rd(p),check_dtype=False)
def dump(p,**a):
 assert not p.exists();np.savez_compressed(p,**a)
 with np.load(p) as f:
  for k,v in a.items():np.testing.assert_array_equal(v,f[k])
X=rd(B/'discovery_covariates_food_TRE_ONLY.parquet');n=len(X);K=['participant_id','cohort','research_stage'];assert X.final_role.eq('discovery').all() and X.participant_id.is_unique
assert X.groupby('split_unit_id').cv_fold.nunique().max()==1
Yd=rd(B/'D_Glu_raw_Y_TRE_ONLY.parquet');rows=Yd.input_row.to_numpy(int);units=Yd.Y_unit.dropna().unique().tolist();assert len(units)==1
old=json.loads((V/'models/D_Glu/fiber_density/scores_and_predictor.json').read_text());cfg=json.loads((V/'configuration_order.json').read_text());jn=old['selection']['main_only_selected_index'];ji=old['selection']['interaction_selected_index'];k,ml,_=cfg[jn]
with np.load(V/'models/D_Glu/fiber_density/selected_CV_losses_TRE_ONLY.npz') as f:oldpred=f['all_predictions'];oldy=f['fold_Y']
LAMS=[.01,.1,1.,10.];names=['residual_CLR','residual_PC5','residual_PC10'];loss={s:np.zeros((4,5)) for s in names};pred={s:np.full((n,4),np.nan) for s in names};qheld=np.full((n,2),np.nan);counts=[];folddata={};daily_scales=[];stability=[];yheld=np.full(n,np.nan);slope={s:np.full((n,2),np.nan) for s in names};foldcoefs={}

def prep(key):
 folder=V/'fold_preprocessing'/key
 with np.load(folder/'transformed_discovery_TRE_ONLY.npz') as f:a={x:f[x] for x in f.files}
 diet=json.loads((folder/'diet_basis.json').read_text())['fiber_density'];diet.update(u=a['fiber_u'],h=a['fiber_h']);training=a['training'].astype(bool)
 yy,ps=phenotype(Yd.Y_value.to_numpy(float),rows,training,'D_Glu');y=np.full(n,np.nan);y[rows]=yy
 E=X.mean_daily_energy_fiber.to_numpy(float);elig=X.n_fiber_days.ge(3).to_numpy()&a['supported']&np.isfinite(X.age_calendar_years)&X.sex_code.isin([0,1])&np.isfinite(E)&np.isfinite(diet['u'])&np.isfinite(diet['h'])&np.isfinite(y);elig=np.asarray(elig);tr=training&elig
 enc=HostEncoder().fit(X,tr,'fiber_density',diet);host=enc.transform(X);z=np.column_stack([a['oral_z'],a['gut_z']]);pc=lambda kk:np.column_stack([a['oral_PC'][:,:kk],a['gut_PC'][:,:kk]])
 base=np.column_stack([host,pc(k)]);labels=enc.labels+['PC']*(2*k);nd=[j for j,l in enumerate(enc.labels) if not l.startswith('diet_')];non=np.column_stack([host[:,nd],pc(k)])
 return dict(a=a,y=y,elig=elig,tr=tr,host=host,z=z,pc=pc,base=base,non=non,diet=diet,enc=enc,phenotype=ps)
for o in range(5):
 d=prep('fold'+str(o));tr=d['tr'];va=X.cv_fold.eq(o).to_numpy()&d['elig'];assert not set(X.loc[tr,'split_unit_id'])&set(X.loc[X.cv_fold.eq(o),'split_unit_id'])
 null=BlockDesign(d['base'],d['z'],d['diet']['u'],d['y'],tr).fit(ml,None)
 np.testing.assert_allclose(null['prediction'][va],oldpred[va,jn],atol=2e-9);np.testing.assert_allclose(d['y'][va],oldy[va],atol=1e-12);yheld[va]=d['y'][va]
 mods={'residual_CLR':d['z'],'residual_PC5':d['pc'](5),'residual_PC10':d['pc'](10)}
 co={}
 for name,mod in mods.items():
  vc=ResidualVC(d['base'],d['z'],d['diet']['u'],d['y'],tr,ml,d['non'],mod);np.testing.assert_allclose(vc.nullpred[va],oldpred[va,jn],atol=2e-9)
  co[name]=[]
  for j,lam in enumerate(LAMS):
   fit=vc.fit(lam);pred[name][va,j]=fit['prediction'][va];loss[name][j,o]=np.mean((d['y'][va]-fit['prediction'][va])**2);co[name].append(fit['coef'])
  dump(O/('fold'+str(o)+'_'+name+'_fit_TRE_ONLY.npz'),**{'coef_all':np.array(co[name]),'modifier_center':vc.center,'exposure_base':vc.eb,'exposure_CLR':vc.ez,'W_base_projection':vc.wb,'W_CLR_projection':vc.wz,'nuisance_base':vc.nb,'nuisance_CLR':vc.nz,'R_scales':vc.sd})
 foldcoefs[o]=co
 # Daily hyperparameters fixed in the specification, no held Y tuning.
 dailybase=np.column_stack([d['host'],d['pc'](5)]);dailyfit=BlockDesign(dailybase,d['z'],d['diet']['u'],d['y'],tr).fit(1.,10.)
 p=d['a']['oral_z'].shape[1]
 for si,site in enumerate(['oral','gut']):
  v=dailyfit['interaction_coef'][:p] if si==0 else dailyfit['interaction_coef'][p:];z=d['a'][site+'_z'];ss=z@v;mu=float(ss[tr].mean());sd=float(ss[tr].std());assert sd>1e-12
  held=X.cv_fold.eq(o).to_numpy()&d['a']['supported'];qheld[held,si]=(ss[held]-mu)/sd
  daily_scales.append({'fold':o,'site':site,'training_n':int(tr.sum()),'training_units':int(X.loc[tr,'split_unit_id'].nunique()),'mean':mu,'SD_outcome_per_fiber5':sd,'positive_means':'higher learned fiber association slope','k':5,'main_lambda':1.,'interaction_lambda':10.})
  dump(O/('daily_fold'+str(o)+'_'+site+'_formula_TRE_ONLY.npz'),weights=v/sd,center=np.array(mu/sd),training=tr,held_supported=held)
 counts.append({'fold':o,'training_n':int(tr.sum()),'training_units':int(X.loc[tr,'split_unit_id'].nunique()),'validation_n':int(va.sum()),'validation_units':int(X.loc[va,'split_unit_id'].nunique())});folddata[o]=d
 print('TARGETED_FOLD_DONE',json.dumps(counts[-1]),flush=True)
common=np.isfinite(yheld);Ncv=int(common.sum());assert Ncv==3343
summary=[];selected={};nullm=metrics(yheld[common],oldpred[common,jn]);v2m=metrics(yheld[common],oldpred[common,ji]);family=X.loc[common,'split_unit_id'].to_numpy()
for name in ['v2_ridge']+names:
 if name=='v2_ridge':pp=oldpred[:,ji];chosen=cfg[ji][2]
 else:
  j=min(range(4),key=lambda x:(loss[name][x].mean(),-LAMS[x]));selected[name]=j;pp=pred[name][:,j];chosen=LAMS[j]
 mm=metrics(yheld[common],pp[common]);dl=(yheld[common]-oldpred[common,jn])**2-(yheld[common]-pp[common])**2
 beta,cov,ci,G,res=cluster_fit(dl,np.ones((Ncv,1)),np.ones(Ncv),family)
 var=float(np.var(yheld[common]));row={'model':name,'n':Ncv,'families':G,'training_min':min(x['training_n'] for x in counts),'training_max':max(x['training_n'] for x in counts),'lambda':chosen,'null_R2':nullm['R2'],'R2':mm['R2'],'delta_R2':mm['R2']-nullm['R2'],'delta_R2_CI_conditional':(ci[0]/var).tolist(),'MSE':mm['MSE'],'null_MSE':nullm['MSE'],'development_tuned':True}
 summary.append(row);print('MODEL_RESULT',json.dumps(row),flush=True)
# Full discovery models and original-unit slope contrasts; fitted uncertainty below is out-of-fold diagnostic only.
full=prep('full');fullpreds={};fullgamma={}
for name in names:
 mod=full['z'] if name=='residual_CLR' else full['pc'](5 if name=='residual_PC5' else 10)
 vc=ResidualVC(full['base'],full['z'],full['diet']['u'],full['y'],full['tr'],ml,full['non'],mod);j=selected[name];fit=vc.fit(LAMS[j]);g=fit['coef'];fullpreds[name]=fit['prediction'];fullgamma[name]=g
 dump(O/(name+'_full_predictor_TRE_ONLY.npz'),coef=g,modifier_center=vc.center,exposure_base=vc.eb,exposure_CLR=vc.ez,W_base_projection=vc.wb,W_CLR_projection=vc.wz,nuisance_base=vc.nb,nuisance_CLR=vc.nz,training=full['tr'],prediction=fit['prediction'])
 for o,d in folddata.items():
  modfold=d['z'] if name=='residual_CLR' else d['pc'](5 if name=='residual_PC5' else 10);gfold=foldcoefs[o][name][j];p=d['a']['oral_z'].shape[1] if name=='residual_CLR' else (5 if name=='residual_PC5' else 10);held=X.cv_fold.eq(o).to_numpy()&common
  for si in range(2):
   zz=modfold[:,:p] if si==0 else modfold[:,p:];gg=gfold[:p] if si==0 else gfold[p:];axis=zz@gg;sd=float(axis[d['tr']].std());mu=float(axis[d['tr']].mean());assert sd>1e-12;slope[name][held,si]=(axis[held]-mu)/sd
   ff=(mod[:,:p] if si==0 else mod[:,p:])@(g[:p] if si==0 else g[p:]) if name!='residual_CLR' else None
   # CLR feature dictionaries can differ across folds: compare full-derived raw scores, not coefficient positions.
   if name=='residual_CLR':
    fp=full['a']['oral_z'].shape[1];ff=full['a']['oral_z' if si==0 else 'gut_z']@(g[:fp] if si==0 else g[fp:])
   corr=float(np.corrcoef(axis[held],ff[held])[0,1]);stability.append({'model':name,'site':['oral','gut'][si],'fold':o,'SD_slope_per5':sd,'signed_full_reference_correlation':corr,'reference_contains_overlap':True})
# Save actual held directions and daily score sources before any daily Y is projected.
out=X[K+['split_unit_id','cv_fold']].copy();out['Y']=yheld;out['null_prediction']=oldpred[:,jn];out['v2_prediction']=oldpred[:,ji]
for name in names:
 out[name+'_prediction']=pred[name][:,selected[name]]
 for si,site in enumerate(['oral','gut']):out[name+'_'+site+'_score']=slope[name][:,si]
for si,site in enumerate(['oral','gut']):out['daily_'+site+'_score']=qheld[:,si]
save(out,O/'development_predictions_and_scores_TRE_ONLY.parquet')
# Slopes: regress held residual on u*held score, retaining base nuisances and fold indicators.
# CIs condition on learned direction, chosen lambda and fitted nuisance; this is development calibration.
effects=[]
for name in names:
 q=slope[name];ok=common&np.isfinite(q).all(1);dd=None;rr=None
 for o,d in folddata.items():
  va=X.cv_fold.eq(o).to_numpy()&ok;u=d['diet']['u'];aa=np.column_stack([np.ones(n),u,d['diet']['h'],u*q[:,0],u*q[:,1],np.eye(5)[X.cv_fold][:,1:]])
  if dd is None:dd=np.zeros((n,aa.shape[1]));rr=np.zeros(n)
  dd[va]=aa[va];rr[va]=d['y'][va]-oldpred[va,jn]
 beta,cov,ci,G,res=cluster_fit(rr[ok],dd[ok],np.ones(ok.sum()),X.loc[ok,'split_unit_id'])
 for si,site in enumerate(['oral','gut']):
  row={'model':name,'site':site,'beta_outcome_per_fiber5_per_scoreSD':float(beta[3+si]),'CI95':ci[3+si].tolist(),'Y_unit':units[0],'diagnostic_not_confirmation':True};effects.append(row);print('SLOPE_CALIBRATION',json.dumps(row),flush=True)
pd.DataFrame(summary).to_csv(O/'model_comparison.csv',index=False);pd.DataFrame(stability).to_csv(O/'score_stability.csv',index=False)
result={'status':'fiber_Glu_targeted_complete','python':os.sys.executable,'pid':os.getpid(),'seconds':time.monotonic()-start,'n_discovery':n,'discovery_families':int(X.split_unit_id.nunique()),'CV_n':Ncv,'CV_families':int(X.loc[common,'split_unit_id'].nunique()),'full_training_n':int(full['tr'].sum()),'full_training_units':int(X.loc[full['tr'],'split_unit_id'].nunique()),'background_PC':k,'main_lambda':ml,'Y_unit':units[0],'models':summary,'counts':counts,'selection':{name:{'lambda':LAMS[j],'fold_losses':loss[name].tolist()} for name,j in selected.items()},'daily_score_scales':daily_scales,'slope_calibration':effects,'confirmation_Y_used':False,'formal55_tests_started':False}
(O/'model_summary.json').write_text(json.dumps(result,indent=2)+'\n');print('TARGETED_MODELS_COMPLETE',json.dumps({k:v for k,v in result.items() if k not in ['selection','models','daily_score_scales','slope_calibration','counts']}),flush=True)
