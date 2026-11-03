"""Independent readback of saved discovery predictions, scores and frozen artifacts."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import io,json,hashlib,datetime
import numpy as np,pandas as pd,pyarrow.parquet as pq
from preprocessing_v1_20261030 import MicrobiomeCLR
from w19_baseline_core import metrics
from v2_ridge_core import select_minimum,score
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');B=P/'data_qc/w20_simplified_20261101_v2';O=B/'results'
L=Path('/home/ec2-user/multisystem_runtime_20261021/w20_simplified_20261101_v2')
assert (L/'43_actual_exit_code').read_text().strip()=='0'
read=lambda p:pq.ParquetFile(p,pre_buffer=False).read().to_pandas(ignore_metadata=True)
X=read(B/'discovery_covariates_food_TRE_ONLY.parquet');roles=read(B/'roles_v2_full_TRE_ONLY.parquet');assert X.final_role.eq('discovery').all()
assert set(X.participant_id)==set(roles.loc[roles.final_role.eq('discovery'),'participant_id'])
reserved=set(roles.loc[roles.final_role.eq('confirmation_reserved'),'participant_id']);assert not set(X.participant_id)&reserved
assert roles.groupby('split_unit_id').final_role.nunique().max()==1
assert X.groupby('split_unit_id').cv_fold.nunique().max()==1
protected=json.loads((B/'protected_v1_hashes.json').read_text())
for path,expected in protected.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
cfg=[tuple(c) for c in json.loads((O/'configuration_order.json').read_text())]
z={};context=O/'fold_preprocessing/full'
for site in ['oral','gut']:
 with np.load(io.BytesIO((B/(site+'_raw_TRE_ONLY.npz')).read_bytes())) as f:raw=f['abundance']
 m=MicrobiomeCLR();m.state=json.loads((context/(site+'_transform.json')).read_text())['CLR'];taxa=json.loads((B/(site+'_taxa.json')).read_text());z[site]=m.transform(raw,taxa,site=site)['values']
zz=np.column_stack([z['oral'],z['gut']]);table=pd.read_csv(O/'discovery_comparison.csv');stability=pd.read_csv(O/'score_stability.csv');control=pd.read_csv(O/'synthetic_controls.csv')
assert len(table)==18 and len(stability)==36 and len(control)==3
K=['participant_id','cohort','research_stage'];verified_scores=0;verified_predictions=0;compact=[]
with np.load(io.BytesIO((context/'transformed_discovery_TRE_ONLY.npz').read_bytes())) as f:diets={'UPF':f['UPF_u'],'fiber_density':f['fiber_u']}
for row in table.to_dict('records'):
 folder=O/'models'/row['analysis_id']/row['exposure_id'];model=json.loads((folder/'scores_and_predictor.json').read_text())
 assert not model['confirmation_used'] and model['no_unpenalized_compressed_refit']
 predictions=read(folder/'development_CV_predictions_TRE_ONLY.parquet');scores=read(folder/'discovery_frozen_scores_TRE_ONLY.parquet')
 assert not predictions.duplicated(K).any() and not scores.duplicated(K).any()
 assert not set(predictions.participant_id)&reserved and not set(scores.participant_id)&reserved
 assert predictions.groupby('split_unit_id').cv_fold.nunique().max()==1
 for prefix in ['null','interaction']:
  met=metrics(predictions.Y_fold_processed,predictions[prefix+'_prediction'])
  for field in ['MSE','R2']:np.testing.assert_allclose(met[field],row[prefix+'_'+field],rtol=1e-10,atol=1e-10)
  verified_predictions+=1
 assert len(predictions)==row['n']
 with np.load(io.BytesIO((folder/'selected_CV_losses_TRE_ONLY.npz').read_bytes())) as f:
  jn,ji,means=select_minimum(f['losses'],cfg)
  assert jn==model['selection']['main_only_selected_index'] and ji==model['selection']['interaction_selected_index']
  for j,prefix in [(jn,'null'),(ji,'interaction')]:
   r=predictions[K].merge(X[K].assign(input_row=np.arange(len(X))),on=K,validate='one_to_one').input_row.to_numpy(int)
   np.testing.assert_allclose(f['all_predictions'][r,j],predictions[prefix+'_prediction'],rtol=1e-12,atol=1e-12)
  np.testing.assert_allclose(means,model['selection']['mean_validation_MSE'])
 with np.load(io.BytesIO((folder/'full_fit_readback_TRE_ONLY.npz').read_bytes())) as f:
  tr=f['training'];u=diets[row['exposure_id']];valid=np.flatnonzero(tr)
  calc=f['background'][tr]@np.asarray(model['background_coef'])+zz[tr]@np.asarray(model['CLR_main_coef'])+u[tr]*(zz[tr]@np.asarray(model['CLR_interaction_coef']))
  np.testing.assert_allclose(calc,f['prediction'][tr],rtol=1e-10,atol=1e-10)
  calcnull=f['background'][tr]@np.asarray(model['matched_null_background_coef'])+zz[tr]@np.asarray(model['matched_null_CLR_main_coef'])
  np.testing.assert_allclose(calcnull,f['null_prediction'][tr],rtol=1e-10,atol=1e-10)
  rr=scores[K].merge(X[K].assign(input_row=np.arange(len(X))),on=K,validate='one_to_one').input_row.to_numpy(int);np.testing.assert_array_equal(rr,valid)
  for site in ['oral','gut']:
   direction=model['directions'][site]
   if direction is None:continue
   vals=score(z[site][rr],direction);np.testing.assert_allclose(vals,scores[site+'_score'],rtol=1e-10,atol=1e-10)
   np.testing.assert_allclose(vals.mean(),0,atol=1e-10);np.testing.assert_allclose(vals.std(),1,atol=1e-10);verified_scores+=1
 rec={k:row[k] for k in ['analysis_id','exposure_id','n','training_n_min','training_n_max','null_R2','interaction_R2','delta_R2','prediction_winner','oral_stability','gut_stability']};compact.append(rec)
 assert not model['selection']['null_predictive_veto'] and not model['selection']['one_SE']
assert not stability.hard_stability_gate.any()
summary=json.loads((O/'summary.json').read_text());assert verified_scores==summary['frozen_score_dimensions']
manifest=[{'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in B.rglob('*') if p.is_file()]
D=O/'delivery';D.mkdir(exist_ok=False);(D/'artifact_manifest_TRE_ONLY.json').write_text(json.dumps(manifest,indent=2)+'\n')
status={'status':'simplified_v2_readback_verified','time_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'summary':summary,'roles':json.loads((B/'role_summary.json').read_text()),'protected_v1_files_unchanged':len(protected),'CV_metric_readbacks':verified_predictions,'score_formula_readbacks':verified_scores,'prediction_formula_readbacks':36,'actual_exit_43':0,'confirmation_Y_reads_this_rerun':0,'formal_P_values_computed':False,'primary_family_55_Holm_unchanged':True,'post_selection_CV_optimism_disclosed':True,'next':'W22 freeze needed before any confirmation analysis'}
(D/'verified_status.json').write_text(json.dumps(status,indent=2)+'\n');pd.DataFrame(compact).to_csv(D/'compact_results.csv',index=False)
print('V2_VERIFIED_STATUS',json.dumps(status),flush=True)
for r in compact:print('V2_VERIFIED_MODULE',json.dumps(r),flush=True)
for r in control.to_dict('records'):print('V2_VERIFIED_CONTROL',json.dumps(r),flush=True)
