"""User-authorized simplified discovery rerun; no reserved-confirmation Y access."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,io,time,hashlib,resource
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq
from preprocessing_v1_20261030 import MicrobiomeCLR
from w19_baseline_core import FoldContext,sparse_npz,dump_npz,phenotype,HostEncoder,main_columns,metrics
from v2_ridge_core import BlockDesign,direction,score,select_minimum
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
L=Path('/home/ec2-user/multisystem_runtime_20261021/w20_simplified_20261101_v2')
B=P/'data_qc/w20_simplified_20261101_v2';O=B/'results';O.mkdir(exist_ok=False)
assert (L/'42_actual_exit_code').read_text().strip()=='0'
resource.setrlimit(resource.RLIMIT_AS,(5*1024**3,5*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(10800,10800))
spec=json.loads((B/'execution_spec.json').read_text());cfg=[(k,m,i) for k in spec['background_PC_per_site'] for m in spec['main_lambda'] for i in [None]+spec['interaction_lambda']]
(O/'configuration_order.json').write_text(json.dumps(cfg)+'\n')
receipts=[]
def read(name):
 path=B/name;receipts.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'role':'new_discovery_only'})
 (O/'input_receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
 return pq.ParquetFile(path,pre_buffer=False).read().to_pandas(ignore_metadata=True)
def save(d,path):
 d=d.reset_index(drop=True);pq.write_table(pa.Table.from_pandas(d,preserve_index=False),path,compression='zstd')
 pd.testing.assert_frame_equal(d,pq.ParquetFile(path,pre_buffer=False).read().to_pandas(ignore_metadata=True),check_dtype=False)
X=read('discovery_covariates_food_TRE_ONLY.parquet');assert X.final_role.eq('discovery').all();n=len(X);alltrain=np.ones(n,bool)
assert X.groupby('split_unit_id').cv_fold.nunique().max()==1
raw={}
for site in ['oral','gut']:
 with np.load(io.BytesIO((B/(site+'_raw_TRE_ONLY.npz')).read_bytes())) as f:raw[site]=f['abundance']
taxa={s:json.loads((B/(s+'_taxa.json')).read_text()) for s in raw};A=sparse_npz(B/'unknown_weights_TRE_ONLY.npz')
reference=read('reference_food_labels_TRE_ONLY.parquet');unknown=read('unknown_food_categories_TRE_ONLY.parquet');presence=read('reference_food_presence_TRE_ONLY.parquet')
contexts={};context_root=O/'fold_preprocessing';context_root.mkdir();start=time.monotonic()
for o in list(range(5))+['full']:
 key='fold'+str(o) if o!='full' else 'full';training=X.cv_fold.ne(o).to_numpy() if o!='full' else alltrain
 seed=spec['seed']+(int(o)+1 if o!='full' else 99)
 c=FoldContext(X,raw['oral'],raw['gut'],taxa,A,reference,unknown,presence,training,seed);c.z={};c.names={};c.scales={}
 for site in ['oral','gut']:
  m=MicrobiomeCLR();m.state=c.transforms[site]['CLR'];r=m.transform(raw[site],taxa[site],site=site)
  c.z[site]=r['values'];c.z[site][~r['supported']]=0;c.names[site]=r['feature_names'];c.scales[site]=m.state['scales']
 c.save(context_root/key)
 dump_npz(context_root/key/'transformed_discovery_TRE_ONLY.npz',oral_z=c.z['oral'],gut_z=c.z['gut'],oral_PC=c.pc['oral'],gut_PC=c.pc['gut'],supported=c.supported,training=training,UPF_u=c.diets['UPF']['u'],UPF_h=c.diets['UPF']['h'],fiber_u=c.diets['fiber_density']['u'],fiber_h=c.diets['fiber_density']['h'])
 contexts[key]=c;print('V2_CONTEXT_READY',key,int(training.sum()),'features',c.z['oral'].shape[1],c.z['gut'].shape[1],flush=True)
results=[];stability=[];controls=[]

def prepare(yd,endpoint,exposure,key):
 c=contexts[key];training=c.training;rows=yd.input_row.to_numpy(int)
 values,state=phenotype(yd.Y_value.to_numpy(float),rows,training,endpoint);y=np.full(n,np.nan);y[rows]=values
 days=(X.n_energy_days if exposure=='UPF' else X.n_fiber_days).ge(3).to_numpy()
 E=X['mean_daily_energy_UPF' if exposure=='UPF' else 'mean_daily_energy_fiber'].to_numpy(float)
 eligible=days&c.supported&np.isfinite(X.age_calendar_years.to_numpy(float))&X.sex_code.isin([0,1]).to_numpy()&np.isfinite(E)&np.isfinite(c.diets[exposure]['u'])&np.isfinite(c.diets[exposure]['h'])&np.isfinite(y)
 tr=training&eligible;encoder=HostEncoder().fit(X,tr,exposure,c.diets[exposure]);host=encoder.transform(X)
 assert tr.sum()>0
 bases={};base_names={}
 for k in spec['background_PC_per_site']:
  main,names=main_columns(c,'oral_gut_main',(k,False,0));bases[k]=np.column_stack([host,main]);base_names[k]=encoder.labels+names
 return {'c':c,'y':y,'eligible':eligible,'training':tr,'encoder':encoder,'bases':bases,'base_names':base_names,'phenotype':state,'u':c.diets[exposure]['u'],'z':np.column_stack([c.z['oral'],c.z['gut']])}

def fit_all(data):
 models={};loss_design={}
 for k in spec['background_PC_per_site']:
  design=BlockDesign(data['bases'][k],data['z'],data['u'],data['y'],data['training'])
  for j,(kk,m,i) in enumerate(cfg):
   if kk==k:models[j]=design.fit(m,i)
 return models

def run_module(yd,endpoint,exposure,control_tag=None,oracle_axis=None):
 elapsed=time.monotonic();losses=np.zeros((len(cfg),5));preds=np.full((n,len(cfg)),np.nan);ys=np.full(n,np.nan)
 folddata={};foldfits={};foldcounts=[];oracle=np.full(n,np.nan)
 for o in range(5):
  key='fold'+str(o);d=prepare(yd,endpoint,exposure,key);va=X.cv_fold.eq(o).to_numpy()&d['eligible'];fits=fit_all(d)
  assert va.any() and not set(X.loc[d['training'],'split_unit_id'])&set(X.loc[va,'split_unit_id'])
  for j in range(len(cfg)):
   pp=fits[j]['prediction'];assert np.isfinite(pp[va]).all();preds[va,j]=pp[va];losses[j,o]=np.mean((d['y'][va]-pp[va])**2)
  ys[va]=d['y'][va];foldcounts.append({'fold':o,'training_n':int(d['training'].sum()),'validation_n':int(va.sum()),'oral_features':d['c'].z['oral'].shape[1],'gut_features':d['c'].z['gut'].shape[1]})
  if control_tag is None:folddata[o]=d;foldfits[o]=fits
  if oracle_axis is not None:
   # Oracle knows the pre-fixed synthetic axis, but coefficients still use training Y only.
   b=d['bases'][5];xx=np.column_stack([b,oracle_axis,d['u']*oracle_axis]);co=np.linalg.lstsq(xx[d['training']],d['y'][d['training']],rcond=None)[0];oracle[va]=xx[va]@co
 jnull,jint,means=select_minimum(losses,cfg);common=np.isfinite(ys)
 nm=metrics(ys[common],preds[common,jnull]);im=metrics(ys[common],preds[common,jint])
 row={'analysis_id':endpoint,'exposure_id':exposure,'n':int(common.sum()),'null_R2':nm['R2'],'interaction_R2':im['R2'],'delta_R2':im['R2']-nm['R2'],'null_MSE':nm['MSE'],'interaction_MSE':im['MSE'],'relative_MSE_change':im['MSE']/nm['MSE']-1,'prediction_winner':'interaction' if means[jint]<means[jnull] else 'null','background_PCs_per_site':cfg[jnull][0],'main_lambda':cfg[jnull][1],'interaction_lambda':cfg[jint][2],'training_n_min':min(r['training_n'] for r in foldcounts),'training_n_max':max(r['training_n'] for r in foldcounts),'CV_metrics_are_tuned_development_only':True,'confirmation_Y_reads':0}
 if control_tag is not None:
  row['control']=control_tag;row['oracle_R2']=metrics(ys[common],oracle[common])['R2'];row['oracle_delta_R2']=row['oracle_R2']-row['null_R2']
  controls.append(row);pd.DataFrame(controls).to_csv(O/'synthetic_controls.csv',index=False)
  print('V2_SYNTHETIC_CONTROL',json.dumps(row),flush=True);return row
 full=prepare(yd,endpoint,exposure,'full');c=full['c'];k,m,inter=cfg[jint]
 design=BlockDesign(full['bases'][k],full['z'],full['u'],full['y'],full['training']);finalfit=design.fit(m,inter);nullfit=design.fit(m,None)
 po=c.z['oral'].shape[1];vv={'oral':finalfit['interaction_coef'][:po],'gut':finalfit['interaction_coef'][po:]}
 directions={s:direction(c.z[s],vv[s],full['training'],c.scales[s]) for s in ['oral','gut']}
 folder=O/'models'/endpoint/exposure;folder.mkdir(parents=True,exist_ok=False)
 ds={};held_scores={s:np.full(n,np.nan) for s in ['oral','gut']}
 for site in ['oral','gut']:
  vals=score(c.z[site],directions[site]);correlations=[];fold_available=0
  for o in range(5):
   d=folddata[o];fit=foldfits[o][jint];p=d['c'].z['oral'].shape[1];v=fit['interaction_coef'][:p] if site=='oral' else fit['interaction_coef'][p:]
   di=direction(d['c'].z[site],v,d['training'],d['c'].scales[site]);held=X.cv_fold.eq(o).to_numpy()&common
   if di is not None:
    fold_available+=1;ss=score(d['c'].z[site],di);held_scores[site][held]=ss[held]
    if vals is not None and held.sum()>1:correlations.append(float(np.corrcoef(vals[held],ss[held])[0,1]))
  srow={'analysis_id':endpoint,'exposure_id':exposure,'site':site,'score_available':directions[site] is not None,'available_folds':fold_available,'median_absolute_full_reference_correlation':float(np.median(np.abs(correlations))) if correlations else None,'signed_correlations':json.dumps(correlations),'hard_stability_gate':False,'reference_correlation_is_not_independent_validation':True}
  stability.append(srow);ds[site]=srow
 row.update({'full_training_n':int(full['training'].sum()),'oral_q':int(directions['oral'] is not None),'gut_q':int(directions['gut'] is not None),'oral_stability':ds['oral']['median_absolute_full_reference_correlation'],'gut_stability':ds['gut']['median_absolute_full_reference_correlation']})
 desc={'analysis_id':endpoint,'exposure_id':exposure,'selection':{'configuration_order':cfg,'mean_validation_MSE':means.tolist(),'main_only_selected_index':jnull,'interaction_selected_index':jint,'null_predictive_veto':False,'one_SE':False},'directions':directions,'feature_names':c.names,'preprocessing_path':str(context_root/'full'),'host_transform':full['encoder'].state(),'phenotype_transform':full['phenotype'],'background_columns':full['base_names'][k],'background_coef':finalfit['base_coef'].tolist(),'CLR_main_coef':finalfit['main_coef'].tolist(),'CLR_interaction_coef':finalfit['interaction_coef'].tolist(),'matched_null_background_coef':nullfit['base_coef'].tolist(),'matched_null_CLR_main_coef':nullfit['main_coef'].tolist(),'no_unpenalized_compressed_refit':True,'score_feature_input':'microbiome only; fixed preprocessing and weights','confirmation_used':False,'inference_pending':'W22 global support/rank/dictionary and W23 association tests','q_by_site':{s:int(directions[s] is not None) for s in directions},'training_n':row['full_training_n'],'base_rank':finalfit['base_rank']}
 (folder/'scores_and_predictor.json').write_text(json.dumps(desc,indent=2,allow_nan=False)+'\n')
 out=X.loc[common,['participant_id','cohort','research_stage','split_unit_id','cv_fold']].copy();out['Y_fold_processed']=ys[common];out['null_prediction']=preds[common,jnull];out['interaction_prediction']=preds[common,jint]
 for site in ['oral','gut']:out[site+'_heldfold_score']=held_scores[site][common]
 save(out,folder/'development_CV_predictions_TRE_ONLY.parquet')
 scores=X.loc[full['training'],['participant_id','cohort','research_stage','split_unit_id']].copy()
 for site in ['oral','gut']:
  if directions[site] is not None:scores[site+'_score']=score(c.z[site],directions[site])[full['training']]
 save(scores,folder/'discovery_frozen_scores_TRE_ONLY.parquet')
 dump_npz(folder/'selected_CV_losses_TRE_ONLY.npz',losses=losses,all_predictions=preds,fold_Y=ys)
 dump_npz(folder/'full_fit_readback_TRE_ONLY.npz',training=full['training'],background=full['bases'][k],prediction=finalfit['prediction'],null_prediction=nullfit['prediction'],Y=full['y'])
 (folder/'fold_training_counts.json').write_text(json.dumps(foldcounts,indent=2)+'\n')
 results.append(row);pd.DataFrame(results).to_csv(O/'discovery_comparison.csv',index=False);pd.DataFrame(stability).to_csv(O/'score_stability.csv',index=False)
 (O/'checkpoint.json').write_text(json.dumps({'modules':len(results),'last':endpoint+'/'+exposure,'seconds':time.monotonic()-start})+'\n')
 print('V2_MODULE_COMPLETE',json.dumps(row),'seconds',round(time.monotonic()-elapsed,1),flush=True)
 return row

# The small control diagnoses capability; it does not add a success prerequisite or tune this recipe.
yd=read('D_Glu_raw_Y_TRE_ONLY.parquet');full_control=prepare(yd,'D_Glu','fiber_density','full');eligible=full_control['eligible'];rows=np.flatnonzero(eligible)
c=contexts['full'];axis=c.pc['gut'][:,0].copy();axis=(axis-np.mean(axis[eligible]))/np.std(axis[eligible]);u=c.diets['fiber_density']['u']
signal=u*axis;bb=full_control['bases'][5];scoef=np.linalg.lstsq(bb[eligible],signal[eligible],rcond=None)[0];signal=signal-bb@scoef;signal/=np.std(signal[eligible])
rng=np.random.default_rng(spec['seed']+773);unit_noise={v:rng.normal() for v in sorted(X.split_unit_id.unique())};noise=rng.normal(size=n)+.2*X.split_unit_id.map(unit_noise).to_numpy(float);noise/=np.std(noise[eligible]);background=2+.3*(X.age_calendar_years.to_numpy(float)-50)/10
for f in [0.,.01,.05]:
 yy=background+noise+np.sqrt(f/(1-f))*signal;synthetic=pd.DataFrame({'input_row':rows,'Y_value':yy[rows]})
 run_module(synthetic,'D_Glu','fiber_density',control_tag='signal_residual_fraction_'+str(f),oracle_axis=axis)
for endpoint in spec['endpoints']:
 yd=read(endpoint+'_raw_Y_TRE_ONLY.parquet')
 for exposure in spec['exposures']:run_module(yd,endpoint,exposure)
assert len(results)==18 and len(stability)==36 and len(controls)==3
summary={'status':'simplified_discovery_v2_complete','modules':18,'discovery_people':n,'prediction_interaction_wins':sum(r['prediction_winner']=='interaction' for r in results),'frozen_score_dimensions':sum(r['oral_q']+r['gut_q'] for r in results),'confirmation_Y_reads':0,'formal_tests_started':False,'CV_performance':'tuned development metrics, not unbiased final validation','seconds':time.monotonic()-start,'next':'readback and comparison; W22/W23 not started'}
(O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('V2_COMPLETE',json.dumps(summary),flush=True)
