# -*- coding: utf-8 -*-
from pathlib import Path
if Path(__file__).with_name("local_records_applied.flag").exists():
    raise SystemExit("One-shot historical record builder already applied; edit current artifacts or write a new dated update.")
from pathlib import Path
import json,hashlib,shutil
root=Path.cwd();a=root/'audits/w04_mapping_review_2026-10-27';cfg=root/'config/w04_followup_20261027';cfg.mkdir(exist_ok=True)
D=root/'outputs/researchwrite/hpp_multisystem_20261021/team_checklist_20261026'
P='/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021';L='/home/ec2-user/multisystem_runtime_20261021/w04_mapping_review_20261027'
facts={'date':'2026-10-27','provenance':'manual transcription of aggregate console readback; see 09_final_verification.png; not a download of TRE source data','processed_event_rows_scanned':2640959,'raw_food_label_rows_scanned':6485022,'processed_unique_food_IDs':7770,'classified_FoodIDs_from_existing_catalog':1933,'raw_food_IDs':8032,'extra_NOVA_IDs_from_raw':0,'same_FoodID_multiple_nonmissing_names':0,'same_FoodID_multiple_categories':0,'same_FoodID_NOVA_conflicts':0,'exact_catalog_size':643,'baseline':{'people_before_index':9737,'event_days_before_index':125374,'people_after_index':9087,'event_days_after_index':116906,'index_excluded_days':8468,'people_without_any_indexed_day':650,'people_ge7_indexed_days':8849,'energy_valid_days':113517,'strict_NOVA_complete_days':35057,'strict_NOVA_people_ge1':7957,'strict_NOVA_people_ge7':1707,'zero_unknown_energy_NOVA_days':35196,'zero_unknown_energy_NOVA_people_ge1':7970,'zero_unknown_energy_NOVA_people_ge7':1714,'fiber_complete_days':60574,'fiber_complete_people_ge1':8677,'fiber_complete_people_ge7':4662,'known_energy_NOVA_coverage':0.881640950084934,'daily_UPF_bound_width_median_pp':6.651987649625078,'daily_UPF_bound_width_p90_pp':31.657813322169368},'coverage_diagnostics_bounds_only':{'90':{'days':66603,'people_ge7':5526},'95':{'days':51526,'people_ge7':3609},'99':{'days':38741,'people_ge7':2070}},'FoodID_recovered_events':0,'aggregation_same_valid_days':{'UPF':{'n_ge7':1707,'median_absolute_difference_pp':0.7463514859712141,'p95_absolute_difference_pp':3.748074355377042,'spearman':0.9844405194777822},'fiber':{'n_ge7':4662,'median_absolute_difference_g_per_1000kcal':0.31090793581927834,'p95_absolute_difference_g_per_1000kcal':1.3616324889441715,'spearman':0.9895057489549277}},'actual_exit_codes':{'01':0,'02':0,'03':0,'04':1,'05':1,'06':0,'07':0,'08':0,'09':0},'server_verified_artifact_count_at_09':21,'binary_cloud_mirrors_verified':3,'external_sources_allowed':False,'not_final_cohort_N':True,'minimum_7_days_is_diagnostic_not_new_global_admission':True,'no_new_outcome_or_microbiome_analysis':True,'formal_exposure_selection_frozen':False}
(a/'observed_aggregate_summary.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2)+'\n')
recon={'date':'2026-10-27','base_spec':'config/w04_diet_features_20261026/diet_feature_spec_draft.yaml','status':'implementation_evidence_updated_formal_protocol_not_frozen','allowed_sources':'current attached HPP data and existing user-provided classification material','additional_external_sources_allowed':False,'usda_candidate':'not found in current diet_logging module; no further external search','event_energy_multiplier':1,'energy_semantics_evidence':'W05 actual intake; never multiply weight/100 again','food_day_admission':'actual events AND valid matching daily index, per user; not automatic person deletion','legacy_NOVA_match':'exact short_food_name after trim/casefold; source/version still needs documentation','unique_FoodID_fallback':'tested for missing names; recovered zero events','UPF_unknown_policy':'preserve bounds; positive unknown energy implies no exact point; energy-zero exception reported separately from legacy row-complete flag','coverage_threshold_frozen':False,'minimum_days_frozen_here':False,'person_aggregation_frozen':False,'recommended_aggregation':'equal qualified-day mean, cumulative ratio diagnostic/sensitivity candidate','I_MEDAS_prior_code_located':True,'I_MEDAS_current_production_implemented':False,'I_MEDAS_pending':['reference mixed daily/person versus local daily-score averaging','proxy Med Oil and fats does not identify hummus/tahini specifically','category aliases and current production provenance'],'formal_primary_exposures':None,'test_budget_changed':False,'known_historical_total_energy_NOVA_contact':True,'known_historical_equal_person_weight_contact':True,'exact_participant_visit_overlap_verified':False,'report':'reports/HPP_W04_Mapping_Six_Issues_2026-10-27.md','server_code':P+'/scripts/w04_mapping_review_20261027','server_result':P+'/data_qc/w04_mapping_review_20261027','server_MEMO':P+'/memos/w04_mapping_review_20261027/MEMO.md'}
(cfg/'implementation_reconciliation.json').write_text(json.dumps(recon,ensure_ascii=False,indent=2)+'\n')
report='''# W04：映射与六项问题逐一补核

2026-10-27。当前HPP服务器，pheno环境，经Chrome/WorkSpaces界面执行。仅做食品映射、饮食完整性和汇总方式诊断，没有读取新代谢结局或菌群数值。本次没有Astra复审。

**用户最新限制：只用当前HPP数据与已有分类材料，不再寻找或引入额外外部映射源。** 当前饮食目录未找到USDA辅助映射，关闭这条外部查找路线；未解决分类仍保留未知。下列人数是预清洗饮食可用性诊断，尚未要求任何代谢结局或双部位菌群完整，不能当最终模型N。至少7日仅作诊断展示，并非本轮新冻结所有结局的统一准入。

## 1. 六项问题现在分别到了哪里

| 问题 | 本次实际推进 | 仍未解决／下一步 |
|---|---|---|
| 食品分类连接 | 明确FoodID→短名称／类别关系；全处理后源7770个FoodID，未发现同ID非缺失标签多值或NOVA冲突。旧643名称可做精确连接 | 仅1933个FoodID能通过现有名称分类取得有效NOVA；全覆盖和数据库版本并未解决。只在内部材料范围继续核对 |
| UPF／纤维缺失 | 已实测天数和人数损失，保存覆盖及UPF上下界；检验FoodID补链与零能量未知例外 | 补链恢复0条。仍需W06处理事件编辑／重复、营养素缺失／负值及日完整性；没有自动选择覆盖阈值 |
| 日到个人汇总 | 两种算法在相同有效日上计算并独立读回 | 主口径仍需冻结；建议典型日问题用日均，不根据代谢显著性选择 |
| I-MEDAS | 找到了旧安全模块、notebook、配置与参考实现，明确并非同一聚合版本 | 当前生产版本、类别别名、hummus／tahini代理仍未验收；没有算当前I-MEDAS总分，不阻塞其他候选 |
| 主暴露及检验预算 | UPF、纤维和原V3替代设计的不同问题已区分；有实际测量可用性证据 | 尚未选主／次角色或更改预算，不能把本次短名单当正式替换V3 |
| 历史分析接触 | 重新核对10月13日报告：第17节已有总能量NOVA4与个人等总权重 | 这两点不能称为新设计。逐人／家系／访视与开发接触账本待W15；没有宣称全新未接触确认组 |

## 2. 原始目录和映射文件到底查了什么

当前原始目录：`/home/ec2-user/studies/hpp_datasets/diet_logging/`。完整遍历该模块及metadata，仅三张Parquet、三份字典CSV；没有单独映射文件。

| 文件 | 元信息行数 | 用途 |
|---|---:|---|
| diet_logging.parquet | 149,912 | 饮食日索引 |
| diet_logging_events.parquet | 2,640,959 | 处理后事件；完整投影扫描作为本次饮食诊断来源 |
| raw_diet_logging_events.parquet | 6,485,022 | 本次仅完整投影FoodID及食品标签；没有混入raw摄入量或补访视 |

当前挂载的hpp_datasets根目录没有diet_logging_auxiliary／USDA模块。不能据此说整个HPP从未有过该表，也不能判断截图中的表实际字段。你提供的截图是线索，不是已读到文件的证据。

搜索旧项目与V2项目映射名称，再扩大旧项目字典／分类文件搜索，限定深度内完成。找到了`script/amd/03_prepare_adm_mapping.py`和`new_diet_indicators/03_prepare_diet_indicator_mapping.py`。后者声明构建FoodID的NOVA4／EAT映射，默认引用`outputs/02_food_dictionary/food_dictionary.csv`、`outputs/03_score_mapping/hpdi/hpdi_food_id_mapping.csv`、`config/diet_adherence_foods.csv`，输出`new_diet_indicator_food_id_mapping.csv`。本次没有找到可直接复用、具备相关FoodID分类列的产物，不能把构建代码当成已可用字典。

**当前可用路径：事件FoodID→事件标准短名称→你已有643名称NOVA表。** 采用去首尾空格及大小写统一后的精确匹配，没有模糊匹配或按照代谢结果人工补分类。目录原本2个NOVA空项仍为未知。FoodID补链测试只针对缺短名称且同ID已有唯一分类的情况，不修改原始字段；实测恢复0条。

raw标签有8032个FoodID，但相同旧目录下有效NOVA仍只有1933个ID，没有额外可用ID。raw的访视占位问题不能由此解决，本次也没有这么做。

## 3. 基线人数和天数：逐步看，不取所有结局交集

下表仅cohort=10k、research_stage=00_00_visit。基线单访视下人—访视数等于人数。

| 条件 | 至少1个相应记录日的人数 | 相应记录日数 | 至少7个相应记录日的人数 |
|---|---:|---:|---:|
| 有带有效身份和日期的处理后食物事件 | 9,737 | 125,374 | 本表未另算 |
| 再要求日期匹配饮食日索引 | 9,087 | 116,906 | 8,849 |
| 能量完整、总能量>0，且每条食品NOVA均完整（旧严格规则） | 7,957 | 35,057 | 1,707 |
| 能量完整、总能量>0，未知分类能量恰为0（能量占比可确定） | 7,970 | 35,196 | 1,714 |
| 能量完整、总能量>0，且纤维输入完整非负 | 8,677 | 60,574 | 4,662 |

匹配日索引排除8468个基线事件日，占基线125374日的6.75%；650人没有剩余匹配日，因此自然退出这个饮食日视图。不是额外按人随意删除。全源含其他访视的无索引日期仍是14713/164625=8.94%，不能与基线口径混写。

日索引匹配后，能量输入完整且正的日共113517日。本轮尚未判定记录是否覆盖全天、编辑／重复事件是否全部处理或极端能量是否合理。因此“能量有效”只是上述有限条件，“有日索引”也不是完整记录全天的证明。

NOVA完整日数中位数为3，纤维完整日数中位数为7；这两个中位数以全部9087个索引匹配人—访视为分母，含某特征0个有效日者。不能解释成每个人都有相应天数。

## 4. 未知分类：影响已量化，不能假定缺口消失

基线索引匹配事件中，**已知有效能量加权的NOVA分类覆盖为88.16%**，不是每个人都覆盖88.16%，也不是食品ID覆盖率。7770个ID中1933个可分类，大量长尾食品贡献的能量占比相对较小，但仍使许多天不满足完全分类。

在能量完整的索引日中，UPF上下界宽度中位数约6.65个百分点，90分位约31.66个百分点。上下界是食品分类不确定范围，不是模型效应置信区间。

下表只看能量覆盖时的诊断；分类不完整时仍是区间，**不是已经得到精确UPF值，也不是已决定这些阈值作为主分析准入**。

| 已分类能量覆盖 | 满足条件的基线记录日 | 至少7日的候选人数 | 当日UPF区间最大宽度 |
|---|---:|---:|---|
| ≥90% | 66,603 | 5,526 | ≤10个百分点 |
| ≥95% | 51,526 | 3,609 | ≤5个百分点 |
| ≥99% | 38,741 | 2,070 | ≤1个百分点 |

未知分类但能量为0时，可以不影响能量占比；本次从旧严格规则1707人只增加到1714人，不能靠这个小改动恢复大队列。纤维缺失也不因NOVA已知而自动解决，不能从零能量或宽食物类别直接推为0g纤维。

未分类FoodID及产品名的内部清单已留TRE，包括`remaining_unclassified_foodID_products_TRE_ONLY.json`及按基线索引日汇总的`unclassified_foods_baseline_indexed_TRE_ONLY.json`。有些食品短名称缺失但产品名存在，有些连产品名也缺失。类别与名称也可不一致，例如产品“Lamb chops”出现在Processed meat products类，说明不能把宽类别直接当NOVA金标准。没有自动据这类例子改标签。

## 5. 两种聚合口径实际差多少

只在相同有效日比较，不混入不同缺失样本。下表采用旧严格NOVA日规则和完整纤维日，各自至少7日，属于诊断子集。

| 指标 | 诊断人数 | 两种个人值绝对差的中位数 | 绝对差95分位 | Spearman相关 |
|---|---:|---:|---:|---:|
| UPF日均 vs 累计能量比 | 1,707 | 0.746个百分点 | 3.748个百分点 | 0.9844 |
| 纤维密度日均 vs 累计比 | 4,662 | 0.311 g/1000 kcal | 1.362 g/1000 kcal | 0.9895 |

相关很高不代表定义等价。日均表达典型记录日，累计比表达窗口能量加权构成。当前建议日均为优先主口径，尚未代用户冻结；没有看新代谢关联来选择。

## 6. I-MEDAS和历史重叠的具体更正

旧安全模块确实存在`2026-09-14-score-safety-v2`；旧notebook的NOVA_pct是0–1，当前候选UPF定义为0–100，不能混用单位。旧实现日UPF与日纤维密度取有效日均值，食物类别能量比另用窗口累计比；旧代码不是所有特征采用同一种聚合。

本地适配I-MEDAS为14个二元组件＋一个0–3分组件，总分0–17，输入类别能量为kcal。参考实现部分组件先取个人日均摄入再判阈值，本地适配逐日判阈值后平均。假设阈值≥60 kcal，两日0和120：日得分均值0.5；先均值再判阈值得1。不能混合为一个未命名版本。`Med Oil and fats`也不能证明准确识别hummus／tahini。

所以当前不是“没配方”，而是“配方及多个实现已找到，但生产版本、类别对应和代理仍需确认”。它是背景候选，不应因其缺失将本来可用的UPF／纤维或某结局参与者排掉。

10月13日报告第16节用已分类能量分母，第17节已使用总能量分母、比较过两种分母，也采用个人等总权重。**本项目不能以总能量分母或个人等总权重作为独立于历史的新颖性依据。** 增量应来自双部位、分结局、可重复的异质性证据及可靠验证。当前只能登记已知接触；实际人员／家庭／访视交集未生成。

## 7. 按顺序如何继续解决

1. 在限定的内部材料中整理现有FoodID—名称—NOVA证据及未分类清单；新增解释须有来源与理由，不能按结果补标签。没有足够信息的条目继续未知，不再外搜USDA或其他新映射。
2. W06完成事件编辑／重复、日质量和营养素缺失检查，分别输出UPF与纤维的日／人资格。不把本次≥7日诊断变成所有结局共同门槛。
3. 根据测量质量、覆盖与精度，在新结局分析前确定主暴露角色、汇总口径和检验预算；不能仅为了增加N把未知当非UPF。
4. I-MEDAS单独核版本与代理，允许保持背景候选；不阻塞其他暴露。
5. W15补历史接触及交集账本，W18／W22按真实情况冻结发现／确认安排。

## 8. 代码、Result、备忘录及失败记录

P=`/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021`。
L=`/home/ec2-user/multisystem_runtime_20261021/w04_mapping_review_20261027`。

| 代码 | 功能 | 实际状态 |
|---|---|---|
| 01_inventory_mapping_sources.py | 饮食目录、schema／footer | exit0 |
| 02_search_mapping_candidates.py | 字典字段和映射名称搜索 | exit0 |
| 03_audit_event_mapping_days.py | 完整处理后事件、名称连接、索引和完整性 | exit0 |
| 04_diagnose_unknowns_and_aggregation.py | 未分类及汇总诊断；旧脚本文件名转写错误 | exit1，部分结果保留，不作完整成功交付 |
| 05_recover_foodid_and_finalize.py | 独立读回发现旧诊断Parquet为0字节 | exit1，拒绝坏产物 |
| 06_recover_foodid_verified_io.py | 从有效日表重建比较，FoodID补链及运行盘→云盘哈希／行数验证 | exit0，3份二进制副本通过 |
| 07_find_existing_food_dictionaries.py | 旧构建器参数及分类字典文件定位 | exit0 |
| 08_inspect_mapping_candidates_and_raw.py | 候选字段、完整raw食品标签扫描 | exit0 |
| 09_verify_and_handoff.py | 源指纹、8源码hash、退出码、行数与结果独立验收 | exit0，21项产物清单 |
| 10_record_internal_source_scope.py | 记录用户仅限内部来源的最新决定 | 以随后实际退出记录为准；不做数据重算 |

服务器代码：P/scripts/w04_mapping_review_20261027。
服务器Result：P/data_qc/w04_mapping_review_20261027，包括verification.json、artifact_manifest.json、event_mapping_day_summary.json、recovery_and_aggregation_summary.json和各TRE_ONLY诊断。
服务器MEMO：P/memos/w04_mapping_review_20261027/MEMO.md。
服务器当前配置：P/config/w04_followup_20261027/diagnostic_policy.json。
运行盘二进制原件：L/06_artifacts；日志L/01.log等、实际退出码L/01_actual_exit_code等。

原始数据没有改动；云盘写入异常发生在本轮自己的诊断产物，修正版保留运行盘原件，复制后独立核对。没有因此放宽科学准入。

本地：[代码备忘录](../audits/w04_mapping_review_2026-10-27/MEMO.md)、[聚合转录](../audits/w04_mapping_review_2026-10-27/observed_aggregate_summary.json)、[当前实施补充](../config/w04_followup_20261027/implementation_reconciliation.json)。原始记录、个体表和完整映射清单留TRE；本地数字是屏幕聚合读回转录。

![最终来源及结果独立验收](../audits/w04_mapping_review_2026-10-27/09_final_verification.png)
'''
(root/'reports/HPP_W04_Mapping_Six_Issues_2026-10-27.md').write_text(report)
memo=f'''# W04服务器映射核查：本地备忘录\n\n2026-10-27。完整功能、来源边界、人数口径、退出状态及失败修复见[报告](../../reports/HPP_W04_Mapping_Six_Issues_2026-10-27.md)。\n\n跑了什么：01/02来源清单和字典搜索；03完整处理后事件投影；04/05保留失败；06重建并独立验证二进制；07/08旧构建器及raw标签；09封口验收；10仅登记用户内部来源限制。09已实际exit0；10另以其实际退出记录确认。\n\nCode：`{P}/scripts/w04_mapping_review_20261027/`。\nResult：`{P}/data_qc/w04_mapping_review_20261027/`。\n远程MEMO：`{P}/memos/w04_mapping_review_20261027/MEMO.md`。\n运行日志／退出码：`{L}`；二进制原件在`{L}/06_artifacts`。3份云盘镜像已验证hash与Parquet行数。\n\n只用当前HPP与用户已有分类材料，不再引入外部源。FoodID结构已核实，NOVA覆盖缺口未消除；补链恢复0条，raw没有增加可分类ID。日索引后基线9087人、116906日；≥7日只是诊断：旧严格UPF1707，零未知能量1714，完整纤维4662。均非最终模型N。\n\n没有新结局或菌群数值读取，没有正式暴露／聚合／阈值／检验预算冻结。I-MEDAS已找到配方与代码，生产版本待确认。旧报告第17节已有总能量分母及个人等总权重，已更正重叠登记。\n\n本地汇总为截图聚合转录，非TRE原始表下载。旧V3科学方案和用户原补表不改；后续先读本次实施补充，避免重复上服务器问已知数字。\n'''
(a/'MEMO.md').write_text(memo)
# Correct current reading guidance while preserving pre-update snapshots.
p=root/'reports/HPP_W04_Diet_Feature_Definitions_2026-10-26.md';s=p.read_text();first=s.find('\n',s.find('\n')+1)
s=s[:first]+ '\n\n> **2026-10-27补核已完成：** [映射、实际日数／人数及六问题当前状态](HPP_W04_Mapping_Six_Issues_2026-10-27.html)。以下原始定义工作仍保留10月26日时点；当前单位、代码可用性及历史分母更正以补核为准。\n'+s[first:]
s=s.replace('I-MEDAS类别量纲、平均份额×平均能量、油脂／hummus代理及原实现仍有待核；','10月27日已找到旧I-MEDAS配方和代码，输入类别kcal及旧聚合可追溯；参考／本地版本选择、类别对应、油脂／hummus代理和生产版本仍待验收；')
s=s.replace('旧报告称NOVA4按已分类能量作分母，且原日级模型按人日等权；新候选为全部记录能量分母，当前V3日级要求每人等总权重1/n_i。旧代码未在本轮重审，这些先作为历史报告层面的已知差异登记，不把旧结果直接拼入新确认。','10月27日更正：旧报告第16节用已分类能量和人日等权，第17节已采用总能量分母及个人等总权重，并比较两种分母。当前这两点不能作为未曾探索的新颖性依据；不同定义不能直接拼接进新确认。')
p.write_text(s)
p=root/'config/w04_diet_features_20261026/prior_work_overlap.md';s=p.read_text();s=s.replace('未读完整稿／代码／分析名单','截至10月26日未读完整稿／代码／分析名单；10月27日补核部分旧代码，生产版本及名单仍待核')
s=s.replace('报告描述旧NOVA4按“已分类能量”作分母，旧日级按人日等权；本轮未重新审计旧源码，不把报告口径提升成代码复核事实','10月27日重核：第16节按已分类能量，第17节已按总能量且个人等总权重，并比较两种分母；不能仅以新分母／权重声称独立于历史探索')
s=s.replace('新V3采用全记录能量候选、1/n_i个人等总权重、双部位、多域与共享pair确认；这些需要新验证，不是旧结果自然延续。','总能量分母及个人等总权重已在旧第17节出现；本项目增量须由双部位、多域与共享pair等实际新证据承担，且需要新验证。')
s=s.replace('旧报告称已分类能量分母。两者不能不改名就拼接或比较系数。','旧报告第16节为已分类分母，第17节已用总能量并做两分母比较。这些定义不能混用，也不能把当前总能量分母称为从未探索。')
s+='\n\n## 2026-10-27补核\n\n已定位旧安全模块、notebook与I-MEDAS配置／参考代码。当前FoodID与目录及实际缺失影响见[W04补核](../../reports/HPP_W04_Mapping_Six_Issues_2026-10-27.md)。本次按用户限定，只用内部HPP与已有材料，不再外搜新映射。食品定义复用不等于精确历史人员交集已完成；W15仍需核对。\n';p.write_text(s)
p=root/'config/w04_diet_features_20261026/decision_log.md';s=p.read_text();s+='\n\n## 2026-10-27实施证据补充（不冻结正式方案）\n\n旧配方和保守缺失实现已经找到；W05事件摄入量乘数1已验收。当前源全量映射／有效日诊断已完成，见[补核报告](../../reports/HPP_W04_Mapping_Six_Issues_2026-10-27.md)及[当前实施配置](../w04_followup_20261027/implementation_reconciliation.json)。仅用HPP及已有分类，不再外搜；未知保留未知。旧报告第17节总能量分母和个人等总权重接触已更正。原草案中的“旧代码不可得／仅旧分母”等过时实施说明由本补充替代；正式A/B、主聚合、I-MEDAS版本、覆盖门槛和检验预算仍未冻结。\n';p.write_text(s)
# Minimal study-level contact ledger; no fabricated ID overlap.
(cfg/'historical_contact_ledger_draft.json').write_text(json.dumps({'status':'study_level_known_contact_only_exact_ID_overlap_pending_W15','rows':[{'source':'reports/HPP_饮食-菌群-血糖_完整结果报告_2026-10-13.html','section':'16','exposure':'NOVA4 classified-energy denominator; fiber density','outcome':'daily CGM','known_result_contact':True,'exact_ID_overlap':'unknown'},{'source':'same historical report','section':'17','exposure':'NOVA4 total-energy and classified-energy comparison','outcome':'between/within-person CGM; equal-person weights','known_result_contact':True,'exact_ID_overlap':'unknown'},{'source':'previous NC-Diet project','section':'code and user-provided tables','exposure':'NOVA; fiber; adapted diet scores','outcome':'oral microbiome per user background','known_result_contact':'needs per-analysis ledger','exact_ID_overlap':'unknown'}]},ensure_ascii=False,indent=2)+'\n')
# Current task table retains W04 original definition completion, with explicit follow-up scope.
p=D/'workpackage_progress.json';j=json.loads(p.read_text());t=j['tasks']['W04'];t['short_status']='定义完成；映射／有效日实测补核完成，覆盖缺口和正式选择仍待解决'
t['status_text']+=' 2026-10-27服务器补核：处理后2640959行及raw食品标签6485022行全扫描；当前无USDA辅助模块，用户限定不再外搜。基线日索引后9087人116906日；≥7日诊断旧严格UPF1707、零未知能量1714、完整纤维4662，非最终模型N。09实际exit0，源／hash／副本读回通过；04／05失败保留并由06修复。六问题状态见新报告，未改正式V3预算或暴露。'
t['what_ran']+=' 后续01—03目录／连接／全源日审计，04／05失败留档，06修复并核结果，07／08旧字典及raw标签，09独立验收，10记录用户内部来源限制。'
t['code_location']+=' 补核服务器 `'+P+'/scripts/w04_mapping_review_20261027`；本地 `audits/w04_mapping_review_2026-10-27/`。'
t['result_location']+=' 补核服务器 `'+P+'/data_qc/w04_mapping_review_20261027`；备忘录 `'+P+'/memos/w04_mapping_review_20261027/MEMO.md`；[六问题与人数补核](HPP_W04_Mapping_Six_Issues_2026-10-27.html)。'
j['latest_followup']='W04_mapping_six_issues_20261027';p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n')
p=D/'execution_update.md';s=p.read_text();s+='\n\n**2026-10-27 W04六问题补核：**[映射／人数／六项状态](HPP_W04_Mapping_Six_Issues_2026-10-27.html)。已上服务器实测，不再仅是本地定义草案。基线索引后9087人116906日；旧严格UPF≥7日1707人、零未知能量1714人、完整纤维4662人，均非正式模型N或各结局统一门槛。FoodID补链和raw标签未恢复新分类；用户限定内部材料，不再外搜。两种聚合已作同日比较；I-MEDAS旧代码找到、生产版本待验，主暴露／聚合／预算尚未冻结。04路径转写、05云盘0字节失败均保留，06修复，09独立验收exit0。第17节总能量分母及个人等总权重的历史接触已更正。\n';p.write_text(s)
current='''\n\n### 2026-10-27 W04映射与有效日核查\n\n已在当前服务器完成处理后2640959事件全投影和raw6485022行食品标签投影。基线事件9737人125374日，按用户要求匹配日索引后9087人116906日，排除8468日；650人无剩余匹配日。至少7日仅作诊断：索引日8849人，旧严格NOVA日1707人，零未知能量NOVA日1714人，完整纤维日4662人。没有取CGM／body／liver／双菌群交集，不是最终模型N。\n\n当前diet_logging模块无USDA辅助映射；用户限定HPP与已有分类材料，不再外搜。旧643名称精确连接的基线已知能量NOVA覆盖88.16%，FoodID补链恢复0条，raw没有增加可分类ID。原始数据只读；04／05失败由06修复，3份二进制副本读回通过，09独立验收exit0。旧I-MEDAS代码已找到，生产版本待核；旧报告第17节已有总能量分母及个人等总权重，不能称这些是新颖性。\n\n报告：REPORT_LINK。代码：`/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/scripts/w04_mapping_review_20261027`；Result在同项目`data_qc/w04_mapping_review_20261027`，MEMO在`memos/w04_mapping_review_20261027/MEMO.md`。当前本地实施补充：CONFIG_LINK。没有冻结主暴露、聚合、阈值或新检验预算。\n'''
for n in ['HPP_LOCAL_VERIFIED_FACTS.md','HPP_PROJECT_BACKGROUND.md','reports/HPP_CURRENT_PLAN.md']:
 p=root/n;prefix='../' if n.startswith('reports/') else ''
 reportlink='[W04六项补核]('+('HPP_W04_Mapping_Six_Issues_2026-10-27.md' if n.startswith('reports/') else 'reports/HPP_W04_Mapping_Six_Issues_2026-10-27.md')+')'
 configlink='[实施配置]('+prefix+'config/w04_followup_20261027/implementation_reconciliation.json)'
 p.write_text(p.read_text()+current.replace('REPORT_LINK',reportlink).replace('CONFIG_LINK',configlink))
# Extend protected background append chain, with exact pre-append snapshots.
inputs=json.loads((a/'local_update_input_manifest.json').read_text());records=[]
for n in ['HPP_DATA_BACKGROUND.md','HPP_SERVER_BACKGROUND.md']:
 p=root/n;prev=next(x for x in inputs['records'] if x['path']==n);assert len(p.read_bytes())==prev['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==prev['sha256']
 p.write_text(p.read_text()+current.replace('REPORT_LINK','[W04六项补核](reports/HPP_W04_Mapping_Six_Issues_2026-10-27.md)').replace('CONFIG_LINK','[实施配置](config/w04_followup_20261027/implementation_reconciliation.json)'))
 records.append(dict(prev,append_only=True,after_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(a/'background_append_manifest.json').write_text(json.dumps({'date':'2026-10-27','records':records},ensure_ascii=False,indent=2)+'\n')
p=D/'validate_checklist.py';s=p.read_text();needle="('audits/w03_followup_2026-10-27/background_append_manifest.json', '2026-10-27 W03肝表补核与用户规则澄清')]";assert needle in s;s=s.replace(needle,needle[:-1]+",\n        ('audits/w04_mapping_review_2026-10-27/background_append_manifest.json', '2026-10-27 W04映射与有效日核查')]");p.write_text(s)
print('Local W04 report, facts, six-issue reconciliation and progress updated.')
