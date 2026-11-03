"""Execute registered W19 nested family CV; discovery data only."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,io,hashlib,time,resource
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq
from w19_baseline_core import FoldContext,sparse_npz,dump_npz,phenotype,HostEncoder,configs,main_columns,ridge_fit,predict,choose_one_se,metrics
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');L=Path('/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1')
B=P/'data_qc/w19_discovery_20261101_v1';O=B/'baseline_results_v1';O.mkdir(exist_ok=False)
for t in ['32','33']:assert (L/(t+'_actual_exit_code')).read_text().strip()=='0',t
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(10800,10800))
spec=json.loads((B/'execution_spec.json').read_text());nreads=[]
def read(name):
 p=B/name;nreads.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 (O/'input_receipts.json').write_text(json.dumps(nreads,indent=2)+'\n')
 return pq.ParquetFile(p,pre_buffer=False).read().to_pandas(ignore_metadata=True)
X=read('discovery_covariates_food_TRE_ONLY.parquet');assert X.final_role.eq('discovery').all();n=len(X);alltrain=np.ones(n,bool)
roles=read('discovery_roles_folds_TRE_ONLY.parquet');assert X.participant_id.equals(roles.participant_id)
raw={}
for site in ['oral','gut']:
 with np.load(io.BytesIO((B/(site+'_raw_TRE_ONLY.npz')).read_bytes())) as f:raw[site]=f['abundance']
taxa={s:json.loads((B/(s+'_taxa.json')).read_text()) for s in raw};A=sparse_npz(B/'unknown_weights_TRE_ONLY.npz')
reference=read('reference_food_labels_TRE_ONLY.parquet');unknown=read('unknown_food_categories_TRE_ONLY.parquet');presence=read('reference_food_presence_TRE_ONLY.parquet')
context_cache={};context_root=O/'fold_preprocessing';context_root.mkdir()
def context(key,training):
 if key not in context_cache:
  seed=spec['seed']+int.from_bytes(hashlib.sha256(key.encode()).digest()[:3],'big')
  c=FoldContext(X,raw['oral'],raw['gut'],taxa,A,reference,unknown,presence,training,seed)
  c.save(context_root/key);dump_npz(context_root/key/'transformed_discovery_TRE_ONLY.npz',oral_PC=c.pc['oral'],gut_PC=c.pc['gut'],supported=c.supported,training=training,UPF_u=c.diets['UPF']['u'],UPF_h=c.diets['UPF']['h'],fiber_u=c.diets['fiber_density']['u'],fiber_h=c.diets['fiber_density']['h'])
  context_cache[key]=c
  print('W19_FOLD_PREPROCESSING_COMPLETE',key,int(training.sum()),flush=True)
 return context_cache[key]
def save(d,p):
 d=d.reset_index(drop=True);pq.write_table(pa.Table.from_pandas(d,preserve_index=False),p,compression='zstd')
 pd.testing.assert_frame_equal(d,pq.ParquetFile(p,pre_buffer=False).read().to_pandas(ignore_metadata=True),check_dtype=False)
models=['host_only','oral_main','gut_main','oral_gut_main'];results=[];selections=[];foldreports=[];start=time.monotonic()
# Shared fold preprocessing is outcome-independent and fit only on the respective training families.
for o in range(5):
 tr=X.outer_fold.ne(o).to_numpy();context('outer'+str(o),tr)
 for i in range(3):
  inn=tr&X['inner_for_outer_'+str(o)].ne(i).to_numpy();assert not set(X.loc[inn,'split_unit_id'])&set(X.loc[~tr,'split_unit_id'])
  context('outer'+str(o)+'_inner'+str(i),inn)
context('full',alltrain)
for endpoint in spec['endpoints']:
 yd=read(endpoint+'_raw_Y_TRE_ONLY.parquet');yr=np.full(n,np.nan);rows=yd.input_row.to_numpy(int);yr[rows]=yd.Y_value.to_numpy(float)
 for exposure in spec['exposures']:
  elapsed=time.monotonic();cache={}
  def prepared(key,training):
   if key in cache:return cache[key]
   c=context(key,training);values,ystate=phenotype(yd.Y_value.to_numpy(float),rows,training,endpoint)
   y=np.full(n,np.nan);y[rows]=values
   days=X.n_energy_days.ge(3).to_numpy() if exposure=='UPF' else X.n_fiber_days.ge(3).to_numpy()
   E=X['mean_daily_energy_UPF' if exposure=='UPF' else 'mean_daily_energy_fiber'].to_numpy(float)
   eligible=days&c.supported&np.isfinite(X.age_calendar_years.to_numpy(float))&X.sex_code.isin([0,1]).to_numpy()&np.isfinite(E)&np.isfinite(c.diets[exposure]['u'])&np.isfinite(c.diets[exposure]['h'])&np.isfinite(y)
   tr=training&eligible;assert tr.sum()>100
   enc=HostEncoder().fit(X,tr,exposure,c.diets[exposure]);base=enc.transform(X)
   cache[key]={'context':c,'Y':y,'eligible':eligible,'training':tr,'base':base,'encoder':enc,'phenotype_state':ystate}
   return cache[key]
  pred={m:np.full(n,np.nan) for m in models};oof_y=np.full(n,np.nan);all_config_losses={m:np.zeros((len(configs(m)),5)) for m in models}
  for o in range(5):
   train=X.outer_fold.ne(o).to_numpy();held=~train;outer=prepared('outer'+str(o),train);va=held&outer['eligible'];assert va.sum()>30
   assert not set(X.loc[outer['training'],'split_unit_id'])&set(X.loc[va,'split_unit_id'])
   oof_y[va]=outer['Y'][va]
   for model in models:
    cfg=configs(model);loss=np.zeros((len(cfg),3))
    for i in range(3):
     inntrain=train&X['inner_for_outer_'+str(o)].ne(i).to_numpy();innheld=train&~inntrain
     data=prepared('outer'+str(o)+'_inner'+str(i),inntrain);iv=innheld&data['eligible'];assert iv.sum()>30
     for j,config in enumerate(cfg):
      main,names=main_columns(data['context'],model,config);fit=ridge_fit(data['base'],main,data['Y'],data['training'],config[2])
      pp=predict(data['base'][iv],main[iv],fit);loss[j,i]=np.mean((data['Y'][iv]-pp)**2)
    selected,meta=choose_one_se(loss,cfg)
    for j,config in enumerate(cfg):
     main,names=main_columns(outer['context'],model,config);fit=ridge_fit(outer['base'],main,outer['Y'],outer['training'],config[2]);pp=predict(outer['base'][va],main[va],fit)
     all_config_losses[model][j,o]=np.mean((outer['Y'][va]-pp)**2)
     if j==selected:pred[model][va]=pp
    selection={'analysis_id':endpoint,'exposure_id':exposure,'model':model,'outer_fold':o,'configurations':cfg,'selected_config':cfg[selected],'inner_selection':meta};selections.append(selection)
    fr={'analysis_id':endpoint,'exposure_id':exposure,'model':model,'outer_fold':o,'training_people':int(outer['training'].sum()),'validation_people':int(va.sum()),'MSE':float(all_config_losses[model][selected,o])};foldreports.append(fr)
   print('W19_OUTER_FOLD_COMPLETE',endpoint,exposure,o,int(va.sum()),flush=True)
  common=np.isfinite(oof_y)
  for model in models:assert np.array_equal(np.isfinite(pred[model]),common)
  out=X.loc[common,['participant_id','cohort','research_stage','split_unit_id','final_role','outer_fold']].copy();out['Y_fold_processed']=oof_y[common]
  full=prepared('full',alltrain)
  dest=P/'models'/endpoint/exposure/'baseline_bundle/w19_20261101_v1';dest.mkdir(parents=True,exist_ok=False)
  (dest/'phenotype_transform.json').write_text(json.dumps(full['phenotype_state'],indent=2)+'\n')
  (dest/'conditions_dictionary.json').write_text(json.dumps(full['encoder'].state(),indent=2)+'\n')
  (dest/'shared_preprocessing_reference.json').write_text(json.dumps({'path':str(context_root/'full'),'spec_sha256':hashlib.sha256((B/'execution_spec.json').read_bytes()).hexdigest()},indent=2)+'\n')
  for model in models:
   cfg=configs(model);chosen,meta=choose_one_se(all_config_losses[model],cfg);config=cfg[chosen]
   main,names=main_columns(full['context'],model,config);fit=ridge_fit(full['base'],main,full['Y'],full['training'],config[2])
   desc={'model':model,'configuration':config,'host_columns':full['encoder'].labels,'microbiome_main_columns':names,'host_coefficients':fit[0].tolist(),'microbiome_main_coefficients':fit[1].tolist(),'selection':meta,'final_fit_people':int(full['training'].sum()),'confirmation_data_used':False,'contains_microbiome_interactions':False}
   (dest/(model+'.json')).write_text(json.dumps(desc,indent=2,allow_nan=False)+'\n')
   pp=predict(full['base'][full['training']],main[full['training']],fit);assert np.isfinite(pp).all()
   out[model+'_prediction']=pred[model][common];met=metrics(oof_y[common],pred[model][common]);hostloss=(oof_y[common]-pred['host_only'][common])**2;thisloss=(oof_y[common]-pred[model][common])**2
   res={'analysis_id':endpoint,'exposure_id':exposure,'model':model,**met,'paired_delta_MSE_vs_host':float(np.mean(thisloss-hostloss)),'final_config':json.dumps(config),'final_fit_people':int(full['training'].sum()),'baseline_bundle':str(dest),'confirmation_Y_reads':0};results.append(res)
  save(out,dest/'nested_OOF_predictions_TRE_ONLY.parquet')
  print('W19_BASELINE_MODULE_COMPLETE',endpoint,exposure,'OOF_N',int(common.sum()),'seconds',round(time.monotonic()-elapsed,2),flush=True)
  pd.DataFrame(results).to_csv(O/'discovery_baseline_comparison.csv',index=False);pd.DataFrame(foldreports).to_csv(O/'outer_fold_metrics.csv',index=False)
  (O/'nested_selection_log.json').write_text(json.dumps(selections,indent=2)+'\n')
  (O/'checkpoint.json').write_text(json.dumps({'completed_modules':len(results)//4,'last_endpoint':endpoint,'last_exposure':exposure,'seconds':time.monotonic()-start})+'\n')
assert len(results)==72 and len(foldreports)==360
summary={'status':'W19_discovery_baselines_complete','modules':18,'candidate_comparisons':72,'outer_model_evaluations':360,'fold_preprocessing_bundles':len(context_cache),'outer_folds':5,'inner_folds':3,'confirmation_Y_reads':0,'real_discovery_models_fitted':True,'real_interaction_scores_fitted':False,'confirmation_inference_started':False,'seconds':time.monotonic()-start,'next':'W20 block ridge score learning using sealed discovery recipe'}
(O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('W19_BASELINES_COMPLETE',json.dumps(summary),flush=True)
for exposure in spec['exposures']:
 for model in models:
  z=[x for x in results if x['exposure_id']==exposure and x['model']==model]
  print('W19_AGGREGATE',exposure,model,'median_R2',float(np.median([x['R2'] for x in z])),'median_delta_MSE',float(np.median([x['paired_delta_MSE_vs_host'] for x in z])),flush=True)
