"""Seal completed method assessment and our closed runtime logs, no new Y reads."""
from pathlib import Path
import hashlib,json,datetime
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
S=P/'scripts/preprocessing_20261030_v1'
Q=P/'data_qc/preprocessing_20261030_v1'
L=Path('/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1')
for tag in ['01','02','03','04','05','06b','07','08','09','10','12','13','14b','15','16','17','18','19']:
    assert (L/(tag+'_actual_exit_code')).read_text().strip()=='0',tag
for tag in ['06','14']:assert (L/(tag+'_actual_exit_code')).read_text().strip()=='1',tag
assert not (L/'20_actual_exit_code').exists()
assert not (S/'20_assumption_stress.py').exists()
v=json.loads((Q/'learned_score_full55_calibration_v1.json').read_text())
assert v['all_null_gates_passed'] and v['strong_signal_power_gate_passed']
assert v['new_real_Y_reads']==0 and not v['real_production_activated']
out=Q/'method_assessment_v1';out.mkdir(exist_ok=False)
exits={p.name:p.read_text().strip() for p in L.glob('*_actual_exit_code')}
decision={'version':'UPF_method_assessment_20261030_v1',
 'timestamp_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'selected_route':'model_based_regression_calibration_shared_FoodID',
 'status':'assessment_complete_method_selected_for_W18_execution_draft',
 'user_direction':'stop expanding method testing; proceed with a practical execution plan',
 'assumptions':['known-to-unknown food category probability transport',
                'one coherent classification per FoodID','adequate conditional mean dictionary'],
 'uncertainty':['shared source-group probability estimation','one shared latent label per FoodID','family-cluster outcome covariance'],
 'retain':['original labels and unresolved classification','UPF bounds','predeclared probability-odds sensitivity'],
 'no_midpoint_as_truth':True,'unknown_not_zero':True,
 'complete_candidate_family_null_gates_passed':True,
 'strong_synthetic_signal_detection':v['power'][-1],
 'weak_synthetic_signal_detection':v['power'][0],
 'statistical_general_validity_claimed':False,
 'actual_W18_roles_frozen':False,'researcher_joint_review_recorded':False,
 'new_real_Y_reads':0,'real_microbiome_or_phenotype_parameters_fitted':False,
 'real_supervised_scores_fitted':False,
 'cancelled_before_execution':['20_assumption_stress.py'],
 'recorded_exits':exits,
 'next':'register W18 executable diet/split/dictionary/conditions/family; then training-only transformations'}
(out/'method_decision.json').write_text(json.dumps(decision,indent=2,ensure_ascii=False)+'\n')
(out/'MEMO.md').write_text('''# UPF 方法评估收口

已选择共享 FoodID 的模型式回归校准，用于 W18 执行版。用户要求不过度扩展防御性测试，因此取消未启动的 20 额外压力测试。

18 的五个预设场景每个 400 次，完整候选 55 位置家族的开发校准门槛通过；强合成信号检出 85/100，弱信号 0/100。它是本项目合成原型评估，不是实际关联结果。

保留食品分类的原始证据和 UPF 上下界。未知食品不当作非UPF，分类期望不当作真实个人摄入。主要模型假设与少量预设概率偏移敏感性写入 W18，不将所有潜在问题都追加为前置门槛。

W18 的最终角色、窗口／条件／预算登记及研究者共同签认尚未完成。真实菌群／人级表型参数未拟合，真实结局关联和监督评分未启动。下一步直接整理 W18 执行版。

06 与 14 的真实失败保留；06b、14b 的恢复不覆盖失败。11 是本地旧草稿，未部署。执行状态与统计校准状态分别记录。
''')
runtime=out/'closed_runtime_snapshots';runtime.mkdir()
for p in sorted(L.iterdir()):
    if p.is_file() and p.name!='21.log' and (p.suffix=='.log' or p.name.endswith('_actual_exit_code')):
        b=p.read_bytes();t=runtime/p.name
        with t.open('xb') as f:f.write(b)
        assert t.read_bytes()==b
names=['catalogue_calibration_v1.json','estimated_probability_calibration_v1.json',
       'learned_score_full55_spec_v1.json','learned_score_full55_calibration_v1.json']
for name in names:
    b=(Q/name).read_bytes();t=out/name
    with t.open('xb') as f:f.write(b)
    assert t.read_bytes()==b
manifest=[]
for p in sorted(S.glob('*.py')):
    manifest.append({'path':str(p.relative_to(P)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(out/'deployed_source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for p in [out/'method_decision.json',out/'deployed_source_manifest.json']:json.loads(p.read_text())
print('METHOD_ASSESSMENT_SEALED',json.dumps({'directory':str(out),'route':decision['selected_route'],
    'null_gates_passed':True,'strong_signal_detection':.85,'new_real_Y_reads':0,
    'W18_roles_frozen':False,'additional_20_not_started':True,'source_files_hashed':len(manifest),
    'closed_runtime_files_copied':len(list(runtime.iterdir())),
    'decision_sha256':hashlib.sha256((out/'method_decision.json').read_bytes()).hexdigest()}),flush=True)
