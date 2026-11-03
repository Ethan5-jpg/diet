"""Read back score formulas and metrics; seal W19-W20 handoff without refitting."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,hashlib,datetime,io
import numpy as np,pandas as pd,pyarrow.parquet as pq
from w19_baseline_core import metrics
from preprocessing_v1_20261030 import MicrobiomeCLR
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');L=Path('/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1')
B=P/'data_qc/w20_discovery_20261101_v1';W=P/'data_qc/w19_discovery_20261101_v1';O=B/'score_results_v1';D=O/'delivery';D.mkdir(exist_ok=False)
assert (L/'38_actual_exit_code').read_text().strip()=='0'
spec=json.loads((B/'execution_spec.json').read_text());summary=json.loads((O/'summary.json').read_text());table=pd.read_csv(O/'discovery_score_comparison.csv');st=pd.read_csv(O/'discovery_score_stability.csv');assert len(table)==18 and len(st)==36
X=pq.ParquetFile(W/'discovery_covariates_food_TRE_ONLY.parquet').read().to_pandas(ignore_metadata=True);assert X.final_role.eq('discovery').all()
K=['participant_id','cohort','research_stage'];z={}
for site in ['oral','gut']:
 with np.load(io.BytesIO((W/(site+'_raw_TRE_ONLY.npz')).read_bytes())) as f:raw=f['abundance']
 state=json.loads((W/'baseline_results_v1/fold_preprocessing/full'/(site+'_transform.json')).read_text())['CLR'];m=MicrobiomeCLR();m.state=state;taxa=json.loads((W/(site+'_taxa.json')).read_text());z[site]=m.transform(raw,taxa,site=site)['values']
manifest=[];verified_scores=0;modules=[];full_null=0
selections=json.loads((O/'nested_selection_log.json').read_text());inner_best_nonnull=0;inner_selected_nonnull=0;null_within_1SE=0
for r in selections:
 m=r['inner_oneSE'];means=np.asarray(m['means']);assert int(np.argmin(means))==m['best'];selected=m['selected'];assert means[selected]<=means[m['best']]+m['best_SE']+1e-12
 inner_best_nonnull+=m['best']!=0;inner_selected_nonnull+=r['selected_config'][1] is not None;null_within_1SE+=means[0]<=means[m['best']]+m['best_SE']+1e-12
assert len(selections)==90

for endpoint in spec['endpoints']:
 for exposure in spec['exposures']:
  dest=P/'models'/endpoint/exposure/'ridge_scores/w20_20261101_v1';model=json.loads((dest/'scores_and_model.json').read_text());assert not model['confirmation_used'] and model['q']<=2
  scores=pq.ParquetFile(dest/'discovery_stable_scores_TRE_ONLY.parquet').read().to_pandas(ignore_metadata=True)
  pred=pq.ParquetFile(dest/'nested_OOF_predictions_TRE_ONLY.parquet').read().to_pandas(ignore_metadata=True);assert pred.final_role.eq('discovery').all() and scores.final_role.eq('discovery').all()
  assert not pred.duplicated(K).any() and not scores.duplicated(K).any() and pred.groupby('split_unit_id').outer_fold.nunique().max()==1
  score_rows=scores[K].merge(X[K].assign(input_row=np.arange(len(X))),on=K,validate='one_to_one').input_row.to_numpy(int)
  q=0
  meta=model['selection'];means=np.asarray(meta['means']);assert means[meta['selected']]<=means[meta['best']]+meta['best_SE']+1e-12
  full_null+=model['selected_config'][1] is None
  for site in ['oral','gut']:
   direction=model['directions'][site];status=st.loc[(st.analysis_id==endpoint)&(st.exposure_id==exposure)&(st.site==site)].iloc[0]
   if direction is None:
    assert model['q_by_site'][site]==0 and site+'_score' not in scores;continue
   q+=1;assert bool(status.stable_direction) and int(status.active_outer_folds)>=4 and status.median_absolute_heldfold_correlation>=.5
   w=np.asarray(direction['weights']);assert len(w)==z[site].shape[1] and np.isfinite(w).all()
   # Apply serialized formula to raw microbiome through frozen transforms; no D or Y argument.
   calc=z[site][score_rows]@w-direction['center'];np.testing.assert_allclose(calc,scores[site+'_score'],rtol=1e-12,atol=1e-12)
   np.testing.assert_allclose(calc.mean(),0,atol=1e-10);np.testing.assert_allclose(calc.std(),1,atol=1e-10);verified_scores+=1
  assert q==model['q']
  met=metrics(pred.Y_fold_processed,pred.fold_selected_prediction);row=table.loc[(table.analysis_id==endpoint)&(table.exposure_id==exposure)].iloc[0]
  for name in ['MSE','RMSE','MAE','R2']:np.testing.assert_allclose(met[name],row[name],rtol=1e-10,atol=1e-10)
  assert len(pred)==row['n'];assert len(model['low_model_columns'])==len(model['low_model_coefficients'])==model['low_model_rank']
  for p in dest.iterdir():
   if p.is_file():manifest.append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
  modules.append({'analysis_id':endpoint,'exposure_id':exposure,'oral_q':model['q_by_site']['oral'],'gut_q':model['q_by_site']['gut'],'OOF_R2':float(row.R2),'null_R2':float(row.null_R2),'OOF_n':len(pred)})
assert verified_scores==summary['active_score_dimensions']
for p in list(O.glob('*.csv'))+list(O.glob('*.json'))+[B/'execution_spec.json']:
 manifest.append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(D/'artifact_manifest_TRE_ONLY.json').write_text(json.dumps(manifest,indent=2)+'\n')
for tag in ['37','38','39']:
 for suffix in ['.log','_actual_exit_code']:
  p=L/(tag+suffix)
  if p.exists():(D/p.name).write_bytes(p.read_bytes())
for name in ['37_register_test_W20.py','38_run_W20_scores.py','38_wait_record.py','w20_score_core.py','40_verify_seal_W20.py']:
 p=P/'scripts/preprocessing_20261030_v1'/name;(D/name).write_bytes(p.read_bytes())
status={'status':'W19_W20_discovery_delivery_verified','timestamp_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'W19':{'modules':18,'baselines':72,'OOF_readbacks':72,'actual_exit':0},'W20':{**summary,'actual_exit':0,'serialized_stable_score_formulas_verified':verified_scores},
 'confirmation_Y_reads_during_W19_W20':0,'formal_confirmation_started':False,'next_workpackage':'W22','W21':'optional challenger has not been enabled',
 'human_cross_review_recorded':False,'new_Claude_review_claimed':False,'support_and_rank_freeze_pending':True,'selection_readback':{'outer_records':len(selections),'outer_best_nonnull':int(inner_best_nonnull),'outer_selected_nonnull':int(inner_selected_nonnull),'outer_null_within_1SE':int(null_within_1SE),'full_selected_null_modules':int(full_null)}}
(D/'delivery_status.json').write_text(json.dumps(status,indent=2)+'\n');pd.DataFrame(modules).to_csv(D/'compact_results.csv',index=False)
(D/'TEAM_HANDOFF.md').write_text('''# W19–W20交接

主效应基准与发现期交互评分均已完成，全部只用discovery角色。W19的18个组合有72套基准；W20保存18套评分模型和36条部位稳定性记录。q=0按事前规则保留，不由确认数据寻找替代方向。没有正式P值或确认结果。

用户线：血糖、六肝、后续Daily；同事线：总脂、VAT。两线共用角色、训练分折与55位置预算。模型位于models/<analysis_id>/<exposure_id>/baseline_bundle/w19_20261101_v1及ridge_scores/w20_20261101_v1；折外预测、参数选择和稳定性表在data_qc/w19_discovery_20261101_v1与w20_discovery_20261101_v1中。

评分应用仅需固定菌群变换与物种权重，不读该人的饮食／结局。已从保存的公式对原始菌群重算稳定评分并核对。W20的OOF指标评价每折选择的压缩程序；全发现稳定性门槛是发现后处理，不能把该指标称为无偏的最终门槛模型验证。稳定性相关使用全发现模型作参考，是可重复性诊断，非独立验证。

下一步W22复核全部模块的评分／秩、共同支持、对比、最终N与inactive位置并全局封存；W21可选挑战者未启用。首次正式确认须等全局包封存后；此交接不冒称人工交叉复核或新一轮Claude验收。
''')
print('W20_SELECTION_READBACK',json.dumps(status['selection_readback']),flush=True)
print('W19_W20_DELIVERY_VERIFIED',json.dumps(status),flush=True)
for r in modules:print('W20_VERIFIED_MODULE',json.dumps(r),flush=True)
