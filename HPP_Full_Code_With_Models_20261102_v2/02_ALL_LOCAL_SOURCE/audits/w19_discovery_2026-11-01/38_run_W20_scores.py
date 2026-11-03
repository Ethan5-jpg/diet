"""Registered discovery-only block ridge and per-site compressed scores."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,io,time,hashlib,resource
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq
from preprocessing_v1_20261030 import MicrobiomeCLR
from w19_baseline_core import phenotype,HostEncoder,main_columns,metrics
from w20_score_core import fit_directions_grid,score_values,low_fit,one_se,direction_json
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');L=Path('/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1')
B=P/'data_qc/w20_discovery_20261101_v1';W=P/'data_qc/w19_discovery_20261101_v1';C=W/'baseline_results_v1/fold_preprocessing';O=B/'score_results_v1';O.mkdir(exist_ok=False)
assert (L/'37_actual_exit_code').read_text().strip()=='0'
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(10800,10800))
spec=json.loads((B/'execution_spec.json').read_text());cfg=[tuple(x) for x in spec['ridge_grid']]
receipts=[]
def read(n):
 p=W/n;receipts.append({'path':str(p),'purpose':'W20_supervised_discovery_learning_only','sha256':hashlib.sha256(p.read_bytes()).hexdigest()});(O/'input_receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
 return pq.ParquetFile(p,pre_buffer=False).read().to_pandas(ignore_metadata=True)
X=read('discovery_covariates_food_TRE_ONLY.parquet');assert X.final_role.eq('discovery').all();n=len(X);alltrain=np.ones(n,bool);raw={}
for s in ['oral','gut']:
 with np.load(io.BytesIO((W/(s+'_raw_TRE_ONLY.npz')).read_bytes())) as f:raw[s]=f['abundance']
taxa={s:json.loads((W/(s+'_taxa.json')).read_text()) for s in raw};contexts={}
class Context:
 def __init__(self,key):
  folder=C/key;self.pc={};self.z={};self.scales=[];self.names={}
  with np.load(io.BytesIO((folder/'transformed_discovery_TRE_ONLY.npz').read_bytes())) as f:
   self.pc={'oral':f['oral_PC'],'gut':f['gut_PC']};self.supported=f['supported'];self.training=f['training'];arr={e:(f[e+'_u'],f[e+'_h']) for e in ['UPF','fiber']}
  ds=json.loads((folder/'diet_basis.json').read_text());self.diets={}
  for e,prefix in [('UPF','UPF'),('fiber_density','fiber')]:self.diets[e]={**ds[e],'u':arr[prefix][0],'h':arr[prefix][1]}
  for site in ['oral','gut']:
   state=json.loads((folder/(site+'_transform.json')).read_text())['CLR'];m=MicrobiomeCLR();m.state=state;r=m.transform(raw[site],taxa[site],site=site);z=r['values'];z[~r['supported']]=0;self.z[site]=z;self.scales.append(state['scales']);self.names[site]=r['feature_names']
 def save_reference(self):return str(C)
def context(key):
 if key not in contexts:contexts[key]=Context(key)
 return contexts[key]
def save(d,path):
 d=d.reset_index(drop=True);pq.write_table(pa.Table.from_pandas(d,preserve_index=False),path,compression='zstd');pd.testing.assert_frame_equal(d,pq.ParquetFile(path).read().to_pandas(ignore_metadata=True),check_dtype=False)
results=[];stability=[];selection=[];start=time.monotonic()
for endpoint in spec['endpoints']:
 yd=read(endpoint+'_raw_Y_TRE_ONLY.parquet');rows=yd.input_row.to_numpy(int)
 for exposure in spec['exposures']:
  cache={};elapsed=time.monotonic()
  def prepared(key,training):
   if key in cache:return cache[key]
   c=context(key);assert np.array_equal(c.training,training)
   values,ystate=phenotype(yd.Y_value.to_numpy(float),rows,training,endpoint);y=np.full(n,np.nan);y[rows]=values
   days=X.n_energy_days.ge(3).to_numpy() if exposure=='UPF' else X.n_fiber_days.ge(3).to_numpy();E=X['mean_daily_energy_UPF' if exposure=='UPF' else 'mean_daily_energy_fiber'].to_numpy(float)
   eligible=days&c.supported&np.isfinite(X.age_calendar_years.to_numpy(float))&X.sex_code.isin([0,1]).to_numpy()&np.isfinite(E)&np.isfinite(c.diets[exposure]['u'])&np.isfinite(c.diets[exposure]['h'])&np.isfinite(y)
   tr=training&eligible;encoder=HostEncoder().fit(X,tr,exposure,c.diets[exposure]);host=encoder.transform(X)
   main,names=main_columns(c,'oral_gut_main',(20,True,100.));base=np.column_stack([host,main]);u=c.diets[exposure]['u'];oral=c.z['oral'];gut=c.z['gut']
   ds=fit_directions_grid(base,oral,gut,u,y,tr,cfg,feature_scales=c.scales)
   fits=[low_fit(base,oral,gut,u,y,tr,d) for d in ds]
   data={'context':c,'y':y,'eligible':eligible,'training':tr,'encoder':encoder,'base':base,'base_names':encoder.labels+names,'phenotype':ystate,'directions':ds,'fits':fits,'u':u}
   cache[key]=data;return data
  oof_y=np.full(n,np.nan);oof=np.full(n,np.nan);null=np.full(n,np.nan);outerdirs=[];losses=np.zeros((len(cfg),5))
  for o in range(5):
   tr=X.outer_fold.ne(o).to_numpy();outer=prepared('outer'+str(o),tr);va=~tr&outer['eligible'];innerloss=np.zeros((len(cfg),3))
   for i in range(3):
    inn=tr&X['inner_for_outer_'+str(o)].ne(i).to_numpy();d=prepared('outer'+str(o)+'_inner'+str(i),inn);iv=tr&~inn&d['eligible'];assert iv.sum()>30
    for j,f in enumerate(d['fits']):innerloss[j,i]=np.mean((d['y'][iv]-f['prediction'][iv])**2)
   j,meta=one_se(innerloss,cfg);oof_y[va]=outer['y'][va];oof[va]=outer['fits'][j]['prediction'][va];null[va]=outer['fits'][0]['prediction'][va]
   for k,f in enumerate(outer['fits']):losses[k,o]=np.mean((outer['y'][va]-f['prediction'][va])**2)
   outerdirs.append(outer['directions'][j]);selection.append({'analysis_id':endpoint,'exposure_id':exposure,'outer_fold':o,'selected_config':cfg[j],'inner_oneSE':meta,'q':sum(d is not None for d in outer['directions'][j])})
   assert not set(X.loc[outer['training'],'split_unit_id'])&set(X.loc[va,'split_unit_id'])
   print('W20_OUTER_FOLD_COMPLETE',endpoint,exposure,o,'q',sum(d is not None for d in outer['directions'][j]),flush=True)
  common=np.isfinite(oof_y);assert np.isfinite(oof[common]).all() and np.isfinite(null[common]).all()
  full=prepared('full',alltrain);selected,meta=one_se(losses,cfg);directions=full['directions'][selected];gated=[]
  for si,site in enumerate(['oral','gut']):
   candidate=directions[si];active=sum(ds[si] is not None for ds in outerdirs);correlations=[]
   if candidate is not None:
    fulls=score_values(full['context'].z[site],candidate)
    for o,ds in enumerate(outerdirs):
     if ds[si] is None:continue
     held=X.outer_fold.eq(o).to_numpy()&common;s=score_values(context('outer'+str(o)).z[site],ds[si]);cc=float(np.corrcoef(fulls[held],s[held])[0,1]);correlations.append(abs(cc))
   median=float(np.median(correlations)) if correlations else None
   stable=candidate is not None and active>=4 and median is not None and median>=.5
   gated.append(candidate if stable else None)
   status={'analysis_id':endpoint,'exposure_id':exposure,'site':site,'full_configuration':json.dumps(cfg[selected]),'full_direction_available':candidate is not None,'active_outer_folds':active,'median_absolute_heldfold_correlation':median,'stable_direction':stable,'q':int(stable),'reason':'stable' if stable else ('null_selected' if candidate is None else 'insufficient_discovery_reproducibility')};stability.append(status)
  finalfit=low_fit(full['base'],full['context'].z['oral'],full['context'].z['gut'],full['u'],full['y'],full['training'],gated)
  dest=P/'models'/endpoint/exposure/'ridge_scores/w20_20261101_v1';dest.mkdir(parents=True,exist_ok=False)
  desc={'analysis_id':endpoint,'exposure_id':exposure,'selected_config':cfg[selected],'selection':meta,'q':sum(d is not None for d in gated),'q_by_site':{s:int(d is not None) for s,d in zip(['oral','gut'],gated)},'directions':{s:direction_json(d) for s,d in zip(['oral','gut'],gated)},'candidate_directions_before_stability':{s:direction_json(d) for s,d in zip(['oral','gut'],directions)},'score_feature_names':full['context'].names,'score_inputs':'microbiome_only_with_fixed_CLR_reference','preprocessing_reference':str(C/'full'),'conditions_transform':full['encoder'].state(),'phenotype_transform':full['phenotype'],'base_columns':full['base_names'],'low_model_columns':finalfit['columns'].tolist(),'low_model_labels':finalfit['labels'],'low_model_coefficients':finalfit['coefficients'].tolist(),'low_model_rank':finalfit['rank'],'final_fit_people':int(full['training'].sum()),'confirmation_used':False,'W22_support_freeze_pending':True}
  (dest/'scores_and_model.json').write_text(json.dumps(desc,indent=2,allow_nan=False)+'\n')
  out=X.loc[common,['participant_id','cohort','research_stage','split_unit_id','final_role','outer_fold']].copy();out['Y_fold_processed']=oof_y[common];out['fold_selected_prediction']=oof[common];out['same_dictionary_null_prediction']=null[common];save(out,dest/'nested_OOF_predictions_TRE_ONLY.parquet')
  scores=X.loc[full['training'],['participant_id','cohort','research_stage','split_unit_id','final_role']].copy()
  for s,d in zip(['oral','gut'],gated):
   if d is not None:
    vals=score_values(full['context'].z[s],d);scores[s+'_score']=vals[full['training']];np.testing.assert_allclose(scores[s+'_score'].mean(),0,atol=1e-10);np.testing.assert_allclose(scores[s+'_score'].std(ddof=0),1,atol=1e-10)
  save(scores,dest/'discovery_stable_scores_TRE_ONLY.parquet')
  met=metrics(oof_y[common],oof[common]);nm=metrics(oof_y[common],null[common]);r={'analysis_id':endpoint,'exposure_id':exposure,**met,'null_R2':nm['R2'],'paired_delta_MSE_vs_same_dictionary_null':met['MSE']-nm['MSE'],'final_q':desc['q'],'oral_q':desc['q_by_site']['oral'],'gut_q':desc['q_by_site']['gut'],'selected_config':json.dumps(cfg[selected]),'final_fit_people':int(full['training'].sum()),'path':str(dest),'OOF_before_global_stability_gate':True};results.append(r)
  pd.DataFrame(results).to_csv(O/'discovery_score_comparison.csv',index=False);pd.DataFrame(stability).to_csv(O/'discovery_score_stability.csv',index=False);(O/'nested_selection_log.json').write_text(json.dumps(selection,indent=2)+'\n')
  (O/'checkpoint.json').write_text(json.dumps({'modules':len(results),'last_endpoint':endpoint,'last_exposure':exposure,'seconds':time.monotonic()-start})+'\n')
  print('W20_SCORE_MODULE_COMPLETE',endpoint,exposure,'q',desc['q'],'seconds',round(time.monotonic()-elapsed,2),flush=True)
assert len(results)==18 and len(stability)==36
summary={'status':'W20_discovery_scores_complete','modules':18,'site_stability_records':36,'active_score_dimensions':sum(r['final_q'] for r in results),'modules_with_stable_score':sum(r['final_q']>0 for r in results),'q0_modules':sum(r['final_q']==0 for r in results),'confirmation_Y_reads':0,'formal_p_values_computed':False,'seconds':time.monotonic()-start,'next':'W22 global support/rank/contrast freeze; optional W21 has not been enabled'}
(O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('W20_SCORES_COMPLETE',json.dumps(summary),flush=True)
for r in results:print('W20_Q_RESULT',r['analysis_id'],r['exposure_id'],r['oral_q'],r['gut_q'],r['R2'],r['null_R2'],flush=True)
