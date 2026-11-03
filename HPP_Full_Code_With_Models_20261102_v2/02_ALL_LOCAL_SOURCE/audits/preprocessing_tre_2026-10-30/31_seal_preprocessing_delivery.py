"""Seal processed input delivery and aggregate handoff; no new model fits."""
from pathlib import Path
import json,hashlib,datetime
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
Q=P/'data_qc/preprocessing_20261030_v1';S=P/'scripts/preprocessing_20261030_v1'
L=Path('/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1')
for tag in ['25','26','27b','28b','29b']:assert (L/(tag+'_actual_exit_code')).read_text().strip()=='0',tag
for tag in ['27','28']:assert (L/(tag+'_actual_exit_code')).read_text().strip()=='1',tag
B=Q/'preprocessing_delivery_v1';B.mkdir(exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
n=0
for run in ['w13_microbiome_qc_20261029','w14_eligibility_20261029','w15_split_access_20261029']:
 for e in json.loads((P/'data_qc'/run/'final_artifact_manifest.json').read_text()):
  assert sha(e['path'])==e['sha256'];n+=1
summary=json.loads((Q/'analysis_ready_v1/summary.json').read_text())
assert len(summary['person_tables'])==20 and not summary['associations_fitted'] and not summary['supervised_scores_fitted']
assert sum(x['primary'] for x in summary['person_tables'])==18
assert summary['daily']['people']>0
pointer=json.loads((P/'config/preprocessing_20261030_v1/active_execution.json').read_text())
assert sha(pointer['roles'])==pointer['roles_sha256']
assert sha(pointer['execution_design'])==pointer['execution_design_sha256']
roots=[Q/'food_qc_v1',Q/'foodid_interval_geometry_v1b',Q/'microbiome_transformed_v1',Q/'phenotypes_transformed_v1b',Q/'UPF_probability_inputs_v1b',Q/'analysis_ready_v1',S,P/'config/preprocessing_20261030_v1',P/'freeze/w18_preprocessing_20261030_v1']
manifest=[]
for root in roots:
 for p in sorted(root.rglob('*')):
  if p.is_file():manifest.append({'path':str(p.relative_to(P)),'bytes':p.stat().st_size,'sha256':sha(p)})
(B/'artifact_and_source_manifest_TRE_ONLY.json').write_text(json.dumps(manifest,indent=2)+'\n')
closed=B/'closed_runtime_snapshots';closed.mkdir()
for p in sorted(L.iterdir()):
 if p.is_file() and p.name!='31.log' and (p.suffix=='.log' or p.name.endswith('_actual_exit_code')):
  b=p.read_bytes()
  with (closed/p.name).open('xb') as f:f.write(b)
  assert (closed/p.name).read_bytes()==b
status={'status':'preprocessing_inputs_ready','timestamp_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'active_execution':str(P/'config/preprocessing_20261030_v1/active_execution.json'),
 'roles_sha256':pointer['roles_sha256'],'design_sha256':pointer['execution_design_sha256'],
 'successful_stage_exits':{t:0 for t in ['25','26','27b','28b','29b']},'preserved_implementation_failures':{'27':1,'28':1},
 'old_W13_W14_W15_artifacts_hash_verified':n,'files_hashed':len(manifest),
 'person_tables':20,'primary_person_tables':18,'secondary_BMI_tables':2,'daily_tables':1,
 'microbiome_features':{'oral':131,'gut':509},'phenotypes_processed':10,
 'real_Y_technically_read_for_preprocessing':True,'real_associations_fitted':False,'real_supervised_scores_fitted':False,
 'researcher_joint_signoff_claimed':False,'new_Claude_review_claimed':False,
 'next_workpackages':['W19','W20'],'confirmation_association_not_opened':True}
(B/'delivery_status.json').write_text(json.dumps(status,indent=2)+'\n')
(B/'analysis_ready_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(B/'TEAM_HANDOFF.md').write_text('''# 预处理交接

数据准备及训练／确认角色已经落实。入口为 config/preprocessing_20261030_v1/active_execution.json。旧 W13–W15 产物保持原哈希，不用旧 W15 未分配角色替代新入口。

用户线：D_Glu、六项 D_Liver_*、Daily_CGM/fiber_density。同事线：D_TotalFat、D_VAT，D_BMI 为次要。两线可以从 W19/W20 并行，不需要先跑完所有分析；共同维护 55 位置家族，禁止各线重新分人或家庭。

逐结局／饮食表在 data_qc/preprocessing_20261030_v1/analysis_ready_v1，含原始及处理后表型、角色、家庭、协变量、饮食和固定菌群矩阵行引用。summary.json 和 participant_flow.csv 给出当前准入人数。表内包括确认数据：后续训练脚本必须先按 final_role==discovery 读取训练成员，不能把已生成表视为确认关联已获授权。W22 仍需封存评分、秩、对比及最终支持范围。

microbiome_transformed_v1 是全发现组拟合的参考矩阵。交叉验证必须用原始固定样本输入在每个训练折重拟合物种过滤、替零、标准化、表型阈值及协变量编码。全发现组固定参数供最后拟合和确认应用。

UPF_probability_inputs_v1b 保存上下界、模型期望、共享 FoodID 概率和标签抽样；非线性模型对整套基函数积分，不能仅把期望代入，也不能把模型期望当真实摄入。食品概率模型在训练折内重拟合；保留既定概率偏移敏感性。

人级至少三个新 QC 合格饮食日，人级 CGM 至少三个合格 CGM 日，不要求饮食逐日相交。Daily 为实际共同日期至少三日，等人权重，不继承人级表型离群排除。当前人数为分析输入准入，后续评分秩和支持检查可能再减少。

没有运行真实监督评分或关联；本交接不冒称研究者共同签认或新一轮 Claude 验收。27、28 为实现失败，27b、28b 保留修复版本，不覆盖失败目录。
''')
assert json.loads((B/'delivery_status.json').read_text())==status
print('PREPROCESSING_DELIVERY_SEALED',json.dumps(status),flush=True)
for r in summary['person_tables']:
 print('COHORT',r['analysis_id'],r['exposure_id'],r['discovery'],r['confirmation'],r['units'],r['people_ge7days'],flush=True)
print('DAILY',json.dumps(summary['daily']),flush=True)
