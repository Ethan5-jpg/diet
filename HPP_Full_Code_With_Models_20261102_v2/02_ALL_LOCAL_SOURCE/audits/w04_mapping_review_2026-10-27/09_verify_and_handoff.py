# -*- coding: utf-8 -*-
from pathlib import Path
import json,hashlib,ast
import pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');Q=P/'data_qc/w04_mapping_review_20261027'
L=Path('/home/ec2-user/multisystem_runtime_20261021/w04_mapping_review_20261027')
S=P/'scripts/w04_mapping_review_20261027'
EXPECTED={'07_find_existing_food_dictionaries.py': 'b23a2a44ebfba57071e40deecd4c7c8867816191653c727d20f157db02e22125', '04_diagnose_unknowns_and_aggregation.py': 'f0e3c31dc08b86b897a736e5f0eea084ff3d06126133f62874c9a4354e540b2a', '08_inspect_mapping_candidates_and_raw.py': '4dd2fd72b48cf91fe5ba41bdc2bce52bc72a418716ff5e478a6125adfeb4adca', '06_recover_foodid_verified_io.py': '19aed13a6023184ac815b4bb0eaf572cb0e831c9567095371d4ac49f369f817f', '02_search_mapping_candidates.py': 'd3eb65771ac2f911d58d596bff1e273d924a505c58dac5ef81ffc26471c7de9b', '03_audit_event_mapping_days.py': '20e1da0190924834db277acefd34922f2d14f1c00d73afebd42c27776e03235d', '01_inventory_mapping_sources.py': '3b20b3d5eada9b3b66e20e2beea6f5e85142dac5c2ac9770c2aa68ec12803410', '05_recover_foodid_and_finalize.py': '46340af6fbc3abf6a8e770a4ccaabdac71708d6dfb2c4bc5505fc25730a5a321'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,h in EXPECTED.items():assert sha(S/name)==h,(name,'code hash mismatch')
exits={n:int((L/(n+'_actual_exit_code')).read_text()) for n in ['01','02','03','04','05','06','07','08']}
assert exits=={'01':0,'02':0,'03':0,'04':1,'05':1,'06':0,'07':0,'08':0}
a=json.loads((Q/'event_mapping_day_summary.json').read_text());b=json.loads((Q/'recovery_and_aggregation_summary.json').read_text());raw=json.loads((Q/'raw_food_label_audit_summary.json').read_text())
for x in a['fingerprints']:
 st=Path(x['path']).stat();assert st.st_size==x['bytes'] and st.st_mtime_ns==x['mtime_ns']
D=pq.ParquetFile(Q/'day_mapping_diagnostics_TRE_ONLY.parquet').read().to_pandas(ignore_metadata=True)
assert len(D)==164625 and int(D.index_matched.sum())==149912
base=D[(D.cohort=='10k')&(D.research_stage=='00_00_visit')]
indexed=base[base.index_matched]
flow={'baseline_event_days_before_index':len(base),'baseline_people_before_index':base.participant_id.nunique(),'baseline_event_days_after_index':len(indexed),'baseline_people_after_index':indexed.participant_id.nunique(),'baseline_index_excluded_days':len(base)-len(indexed),'baseline_people_no_remaining_indexed_day':base.participant_id.nunique()-indexed.participant_id.nunique()}
assert len(indexed)==116906
mirrors=json.loads((Q/'binary_mirror_verification.json').read_text())
for x in mirrors:
 assert x['cloud_readback_verified'] and sha(Path(x['cloud']))==sha(Path(x['runtime']))==x['sha256']
 assert pq.ParquetFile(x['cloud']).metadata.num_rows==pq.ParquetFile(x['runtime']).metadata.num_rows
source=json.loads((Q/'existing_food_dictionary_sources_TRE_ONLY.json').read_text())
assert source['complete_within_depth']
assert json.loads((Q/'inventory_TRE_ONLY.json').read_text())['complete']
builders=[]
for z in source['builders']:
 t=ast.parse(Path(z['path']).read_text());constants={}
 for n in t.body:
  if isinstance(n,ast.Assign):
   for target in n.targets:
    if isinstance(target,ast.Name) and (target.id=='PROJECT_DIR' or target.id.startswith('DEFAULT_')):
     constants[target.id]=ast.unparse(n.value)
 builders.append({'path':z['path'],'constants':constants})
 print('BUILDER_DEFAULTS',json.dumps(builders[-1]),flush=True)
policy={'date':'2026-10-27','role':'diagnostic implementation supplement, not frozen scientific protocol','primary_food_day_rule':'actual food events AND valid matching daily index; exclude day, not automatically participant','source_energy_multiplier':1,'nutrient_semantics_source':'W05 verified actual event-intake; never second weight/100','NOVA_catalog':'643-name previous NC-Diet catalog; trim/casefold exact match','FoodID_fallback':'tested unique known same-ID labels for null short names only; recovered zero events','UPF_unknown':'point NA when unknown energy>0; preserve total-energy denominator and bounds; no coverage threshold adopted','UPF_zero_energy_unknown':'algebraically identifiable share when all energy valid and unknown energy exactly zero; differs from legacy all-row-complete flag','aggregation':'equal-day and cumulative computed on identical valid days for diagnostic comparison; primary choice not frozen','minimum_days':'ge7 is diagnostic display, not newly adopted admission for all outcomes','I_MEDAS':'old local and reference code located; version/proxy/current production not resolved; no new individual score computed','formal_selected_exposures':None,'formal_hypothesis_budget_changed':False,'outcome_or_microbiome_values_read':False,'raw_intake_pooled_or_visit_imputed':False}
cfg=P/'config/w04_followup_20261027';cfg.mkdir(exist_ok=True)
cp=cfg/'diagnostic_policy.json';cp.write_text(json.dumps(policy,indent=2));assert json.loads(cp.read_text())==policy
verification={'actual_exits':exits,'source_fingerprints_unchanged':True,'code_hashes':EXPECTED,'full_processed_rows':a['rows_scanned'],'full_raw_label_rows':raw['raw_rows_scanned'],'binary_readback_and_cloud_mirrors_verified':True,'baseline_flow':flow,'baseline_diet':a['scopes']['baseline_10k'],'recovery':b['recovery']['baseline_10k'],'aggregation':b['aggregation'],'canonical_food_IDs':b['canonical_food_IDs'],'unambiguous_NOVA_IDs':b['unambiguous_NOVA_IDs'],'raw_label_summary':raw,'builder_defaults':builders,'no_new_outcome_analysis':True,'final_cohort_N_computed':False}
vp=Q/'verification.json';vp.write_text(json.dumps(verification,indent=2));assert json.loads(vp.read_text())==verification
memo='''# W04映射与六问题核查：代码备忘录

2026-10-27，pheno环境，经Chrome/WorkSpaces界面执行。原始数据只读；无代谢结局或菌群数值读取。

## 跑了什么代码

01_inventory_mapping_sources.py：扫描当前饮食模块目录、schema和footer。
02_search_mapping_candidates.py：原始字典与两个项目目录中映射候选搜索。
03_audit_event_mapping_days.py：完整处理后事件投影，旧643名称分类连接、FoodID一致性、日索引和营养素完整性诊断。
04_diagnose_unknowns_and_aggregation.py：未分类能量及两种聚合；旧脚本路径手工转写错误导致exit1，保留失败。
05_recover_foodid_and_finalize.py：独立读回发现上一份云盘Parquet为0字节，exit1；没有接受坏产物。
06_recover_foodid_verified_io.py：从有效逐日诊断重建比较；FoodID补链诊断；先在运行盘保存Parquet，再复制云盘并验证hash与行数。exit0。
07_find_existing_food_dictionaries.py：定位已有映射构建器的输入输出定义，扩大字典文件名搜索。
08_inspect_mapping_candidates_and_raw.py：候选字段核对、完整raw食品标签投影；不混入raw摄入、不补undefined访视。
09_verify_and_handoff.py：独立源指纹／源码hash／退出码／行数／二进制副本核验、当前诊断配置及MEMO。

代码目录：P/scripts/w04_mapping_review_20261027。
Result目录：P/data_qc/w04_mapping_review_20261027。
运行日志与实际退出码：L/01.log等、L/01_actual_exit_code等。
二进制运行盘原件：L/06_artifacts；云盘副本经哈希及Parquet行数读回验证。
当前诊断配置：P/config/w04_followup_20261027/diagnostic_policy.json。

P=/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021
L=/home/ec2-user/multisystem_runtime_20261021/w04_mapping_review_20261027

## 六问题当前状态

1. 找到当前FoodID与短名称／类别的关系，未发现同ID非缺失标签多值冲突；旧643名称可做有界精确连接。但当前挂载无diet_logging_auxiliary／USDA模块，旧项目只有构建代码、未发现具相应列的映射产物；版本与未分类仍未解决。
2. 实测严格缺失处理的天数和人数损失，已保存UPF上下界与覆盖诊断；没有因保留更多人自动选覆盖阈值，也不把未知当非UPF。纤维缺失／负值未补零。
3. 两种个人聚合在相同有效日上完成数值比较；正式主口径未冻结。
4. 已定位本地适配及参考I-MEDAS代码，混合日／人聚合和代理差异须明确选择；尚未生成当前I-MEDAS生产分数。
5. UPF／纤维与V3原替代设计的角色和检验预算仍待科学选择；未按任何新结局结果选择。
6. 本地历史报告第17节已有总能量分母，不能将本次分母称全新；精确历史人员／家系／访视交集属于后续W15，未宣称未接触确认。

至少7日是本次诊断展示，不等于新冻结各结局共同准入。报告人数没有要求CGM、body、liver或双部位菌群完整；不是最终模型N。raw为食品字典核对来源，不提供新基线摄入。

详细聚合事实、失败记录、来源指纹与hash见verification.json。个体和食品明细留TRE，不下载。
'''
mp=P/'memos/w04_mapping_review_20261027';mp.mkdir(exist_ok=True)
for p in [mp/'MEMO.md',Q/'MEMO.md']:
 p.write_text(memo);assert p.read_text()==memo
manifest=[]
for p in sorted(Q.iterdir()):
 if p.is_file() and p.name!='artifact_manifest.json':manifest.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
(Q/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
print('ACTUAL_EXITS_01_TO_08',json.dumps(exits),flush=True)
print('BASELINE_FLOW',json.dumps(flow),flush=True)
print('BASELINE_COMPLETE_DAYS',json.dumps({k:a['scopes']['baseline_10k'][k] for k in ['index_matched_days','energy_valid_days','NOVA_complete_days','fiber_complete_days','known_energy_NOVA_coverage_fraction','person_visit_day_counts']}),flush=True)
print('RECOVERY_BASELINE',json.dumps(b['recovery']['baseline_10k']),flush=True)
print('AGGREGATION',json.dumps(b['aggregation']),flush=True)
print('RAW_LABELS',json.dumps(raw),flush=True)
print('VERIFIED_ARTIFACTS',len(manifest),'ALL_CODE_SOURCE_BINARY_CHECKS_PASS',flush=True)
print('RESULT',str(Q),'MEMO',str(mp/'MEMO.md'),flush=True)
print('W04_REVIEW_HANDOFF_DONE_NO_FINAL_ANALYSIS_N',flush=True)
