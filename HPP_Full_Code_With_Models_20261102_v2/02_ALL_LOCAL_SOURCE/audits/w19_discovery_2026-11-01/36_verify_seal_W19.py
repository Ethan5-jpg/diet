"""Read back W19 delivery, recompute metrics and seal artifacts; no new fitting."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,hashlib,datetime
import numpy as np,pandas as pd,pyarrow.parquet as pq
from w19_baseline_core import metrics
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');B=P/'data_qc/w19_discovery_20261101_v1';O=B/'baseline_results_v1'
L=Path('/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1')
assert (L/'34_actual_exit_code').read_text().strip()=='0'
D=O/'delivery';D.mkdir(exist_ok=False)
spec=json.loads((B/'execution_spec.json').read_text());result=pd.read_csv(O/'discovery_baseline_comparison.csv');assert len(result)==72
folds=pq.ParquetFile(B/'discovery_roles_folds_TRE_ONLY.parquet').read().to_pandas(ignore_metadata=True);assert folds.final_role.eq('discovery').all()
assert folds.groupby('split_unit_id').outer_fold.nunique().max()==1
models=['host_only','oral_main','gut_main','oral_gut_main'];readback=0;model_files=[];compact=[]
for endpoint in spec['endpoints']:
 for exposure in spec['exposures']:
  dest=P/'models'/endpoint/exposure/'baseline_bundle/w19_20261101_v1'
  df=pq.ParquetFile(dest/'nested_OOF_predictions_TRE_ONLY.parquet').read().to_pandas(ignore_metadata=True)
  assert df.final_role.eq('discovery').all() and not df.duplicated(['participant_id','cohort','research_stage']).any()
  assert df.groupby('split_unit_id').outer_fold.nunique().max()==1
  assert set(df.participant_id)<=set(folds.participant_id)
  for model in models:
   m=json.loads((dest/(model+'.json')).read_text());assert not m['confirmation_data_used'] and not m['contains_microbiome_interactions']
   assert len(m['host_columns'])==len(m['host_coefficients']) and len(m['microbiome_main_columns'])==len(m['microbiome_main_coefficients'])
   check=metrics(df.Y_fold_processed,df[model+'_prediction']);row=result.loc[(result.analysis_id==endpoint)&(result.exposure_id==exposure)&(result.model==model)].iloc[0]
   for name in ['MSE','RMSE','MAE','R2']:np.testing.assert_allclose(check[name],row[name],rtol=1e-10,atol=1e-10)
   assert check['n']==row['n'];readback+=1
   model_files.append(dest/(model+'.json'))
  model_files.extend([p for p in dest.iterdir() if p.is_file() and p.name not in [m+'.json' for m in models]])
  subset=result.loc[(result.analysis_id==endpoint)&(result.exposure_id==exposure)]
  host=float(subset.loc[subset.model.eq('host_only'),'R2'].iloc[0]);both=float(subset.loc[subset.model.eq('oral_gut_main'),'R2'].iloc[0])
  compact.append({'analysis_id':endpoint,'exposure_id':exposure,'OOF_people':len(df),'host_R2':host,'oral_gut_R2':both,'delta_R2':both-host})
manifest=[]
for p in sorted(set(model_files+list(B.glob('*.json'))+list(O.glob('*.json'))+list(O.glob('*.csv')))):
 manifest.append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(D/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for n in ['32','33','34','35']:
 for suffix in ['.log','_actual_exit_code']:
  p=L/(n+suffix)
  if p.exists():(D/p.name).write_bytes(p.read_bytes())
for p in (P/'scripts/preprocessing_20261030_v1').glob('*W19*.py'):
 (D/p.name).write_bytes(p.read_bytes())
(D/'w19_baseline_core.py').write_bytes((P/'scripts/preprocessing_20261030_v1/w19_baseline_core.py').read_bytes())
report={'status':'W19_complete_verified','timestamp_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_recipe_sha256':hashlib.sha256((B/'execution_spec.json').read_bytes()).hexdigest(),'baseline_bundles':18,'baseline_models':72,'OOF_metric_readbacks':readback,'outer_model_evaluations':360,'folds':5,'inner_folds':3,'confirmation_Y_reads':0,'real_discovery_models_fitted':True,'interaction_scores_fitted':False,'next':'W20','researcher_cross_review_recorded':False,'Claude_review_claimed':False}
(D/'delivery_status.json').write_text(json.dumps(report,indent=2)+'\n');pd.DataFrame(compact).to_csv(D/'compact_results.csv',index=False)
(D/'TEAM_HANDOFF.md').write_text('''# W19 交付

18个结局×饮食组合，四套无菌群交互主效应基准，72个最终模型包。5外折／3内折均按家庭，预处理逐训练折拟合；OOF预测和逐折选择、主效应列及系数均可审查。全发现组重新拟合的参考模型另存，不能把其训练性能当作OOF。

本轮只使用discovery角色。新的真实训练结局模型已经拟合，确认结局未读取。W20仍需学习block ridge低维交互评分，依事前null／oneSE和稳定性规则决定q，不能按确认结果倒选。W19的预测改善不等于饮食×菌群交互证据；本轮没有正式p值或确认关联结论。

用户线为血糖和六肝，同事线为总脂、VAT。模型包：models/<analysis_id>/<exposure_id>/baseline_bundle/w19_20261101_v1。结果表：data_qc/w19_discovery_20261101_v1/baseline_results_v1/discovery_baseline_comparison.csv。两线共用同版家庭角色、预处理及55位置家族。
''')
print('W19_DELIVERY_VERIFIED',json.dumps(report),flush=True)
for r in compact:print('W19_RESULT',json.dumps(r),flush=True)
