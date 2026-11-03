"""Persist manually observed TRE aggregates and append-only local handoff. Run once."""
from pathlib import Path
import json, hashlib

R=Path(__file__).resolve().parents[2]; A=Path(__file__).parent
D=R/'outputs/researchwrite/hpp_multisystem_20261021/team_checklist_20261026'
P='/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021'
Q=P+'/data_qc/w11_w12_covariates_samples_20261028'
L='/home/ec2-user/multisystem_runtime_20261021/w11_w12_covariates_samples_20261028'
stem='HPP_W11_W12_Covariates_Sample_Candidates_2026-10-29'
assert not (A/'observed_aggregate.json').exists(), 'One-time recorder; do not repeat background append.'
coverage=[
 ('日历年年龄',10417,707),('源表性别',11103,21),('吸烟状态',8487,2637),
 ('年龄＋性别均可用',10417,707),('年龄＋性别＋吸烟均可用',8219,2905),
 ('口腔抗生素题频率',10004,1120),('步行天数',8474,2650),('中等活动天数',8356,2768),
 ('剧烈活动天数',2229,8895),('睡眠时长',8435,2689),('饮酒频率',8526,2598)]
medcoverage=[('ATC5_A10BA02_listed',2182,8942),('ATC5_A10A_prefix_listed',2102,9022),
 ('ATC5_A10_prefix_listed',2244,8880),('ATC5_C10AA_prefix_listed',2779,8345),
 ('ATC5_C10_prefix_listed',3259,7865),('ATC5_C02_C03_C07_C08_C09_prefix_listed',2650,8474),
 ('ATC5_J01_prefix_listed',2248,8876)]
x={'date':'2026-10-29','provenance':'manual transcription of aggregate-only TRE UI outputs; no individual-data file download',
 'actual_exits':{f'{i:02}':0 for i in range(1,13)},
 'W10_gate':{'final_exit07':0,'artifact_hashes_verified':8},
 'W11':{'registry_person_visits':20407,'registry_unique_people':11126,'baseline_union_people':11124,
 'baseline_coverage':[{'variable':n,'available':a,'missing_unavailable':m} for n,a,m in coverage],
 'baseline_medication_code_availability':[{'variable':n,'available_coded_state':a,'unknown':m} for n,a,m in medcoverage],
 'baseline_medication_source_people':10269,
 'baseline_smoking':{'No':7322,'unknown':2637,'occasional':645,'most_or_all_days':520},
 'baseline_oral_health_frequency':{'Never':8510,'Rarely':1474,'Often':20,'unknown':1120},
 'directed_family_household_edges':1780,'household_edges':1353,'family_edges':427,
 'observed_components_size_gt_one':1086,'baseline_people_in_observed_components_size_gt_one':2244,
 'repeated_person_visit_key_groups':{'lifestyle':140,'family_questionnaire':140,'population':0},
 'medication_source_rows':43417,'valid_key_medication_rows':42618,
 'medication_start_precision':{'start_unavailable':37209,'reported_year_interval':2722,'reported_year_month_interval':2576,'source_start_month_date_conflict':111},
 'medication_start_vs_food':{'unknown':40685,'start_before_food_window':1486,'start_after_food_window':295,'start_interval_spans_food_window_boundaries':152}},
 'W12':{'oral_index_rows':8816,'oral_valid_index_person_visits':8402,'oral_invalid_index_rows':414,
 'oral_baseline_index_people':8294,'oral_abundance_rows':8834,'oral_resolved_abundance_all_stages':8398,
 'oral_unresolved_abundance_unknown_stage':436,'oral_index_only_rows':418,'oral_valid_index_only_rows':4,
 'oral_baseline_valid_index_only_rows':3,'oral_candidate_rows':9252,'oral_baseline_identity_ready_people':8291,
 'oral_ready_baseline_food_lag_n':7186,'oral_ready_baseline_signed_food_lag_median_days':68,
 'gut_index_rows':10694,'gut_unique_index_person_visits':10670,'gut_repeated_index_key_groups':22,
 'gut_baseline_index_rows':8712,'gut_baseline_unique_index_people':8694,
 'gut_abundance_rows':11295,'gut_unique_abundance_person_visits':11269,'gut_repeated_abundance_key_groups':24,
 'gut_baseline_abundance_rows':9105,'gut_baseline_unique_abundance_people':9087,
 'gut_identity_ready_all_stages':10646,'gut_baseline_identity_ready_people':8676,
 'gut_held_rows':649,'gut_index_other_stage_only':185,'gut_person_absent_primary_index':414,
 'gut_repeated_profile_rows_needing_specimen_link':50,'gut_repeated_profile_groups':24,
 'gut_baseline_held_rows':429,'gut_index_only_rows':0,
 'gut_ready_baseline_food_lag_n':7632,'gut_ready_baseline_signed_food_lag_median_days':9,
 'candidate_rows':20547,'abundance_candidate_rows':20129,'index_only_rows':418,
 'baseline_joint_identity_ready_people_not_final_QC':7214,'final_samples_selected':0,'W13_QC_done':False},
 'verification':{'covariate_rows':20407,'questionnaire_cells':122442,'medication_class_cells':142849,
 'medication_start_rows':42618,'family_edges':1780,'oral_health_cells':20407,'candidate_rows':20547,
 'artifact_hashes':13,'archived_prior_revisions':4,'binary_mirrors':4,
 'script10_abundance_lag_rows':20129,'script12_index_only_lag_rows':418,'script12_lag_fields':836},
 'raw_individual_exported':False,'model_fitted':False,'imputation_fitted':False,'protocol_frozen':False,
 'team_human_review_claimed':False,'fresh_Astra_review_claimed':False}
(A/'observed_aggregate.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
covtable='| 项目 | 可用人数 | 缺失／不可用 |\n|---|---:|---:|\n'+'\n'.join(f'| {n} | {a:,} | {m:,} |' for n,a,m in coverage)
medtable='| 代码状态字段 | 可判定编码状态 | unknown |\n|---|---:|---:|\n'+'\n'.join(f'| `{n}` | {a:,} | {m:,} |' for n,a,m in medcoverage)
steps=[
 ('01_preflight.py','读W10最终exit及8产物哈希，核W11/W12来源schema和字典'),
 ('02_read_contracts.py','字段、时间、样本契约与编码文件定位'),
 ('03_compact_contract.py','精简schema与codebook预核'),
 ('04_covariate_value_contract.py','协变量缺失、编码、键重复预核'),
 ('05_sample_identity_probe.py','身份投影与肠道连接基数、日期字段诊断'),
 ('06_focused_values_and_codes.py','精确源编码、药物开始时间、家系与样本证据核对'),
 ('07_build_covariates.py','构建union协变量登记、药物开始区间及ATC状态、家系边'),
 ('08_build_sample_candidates.py','重建全部候选、口腔访视桥、肠道暂挂原因、日期距离与随机优先级'),
 ('09_complete_oral_health.py','按真实字段拼写补齐口腔题；存档4个旧派生版本并发布当前哈希'),
 ('10_independent_verify.py','从原始投影独立重建值、键、家系及样本连接；复核20129个丰度候选的日期距离、哈希与副本'),
 ('11_finalize_and_memos.py','核实际exit与当前哈希；写3份远程备忘录、交付清单及执行状态'),
 ('12_verify_index_only_lags.py','另核418条index-only记录的836个距离字段，补齐全部20547候选的距离验证')]
steptable='| 代码 | 功能 | 实际退出 |\n|---|---|---|\n'+'\n'.join(f'| `{n}` | {f} | 0 |' for n,f in steps)
report=f'''# HPP W11/W12：协变量登记与双部位候选样本

2026-10-29。使用HPP Server Skill，经Chrome → WorkSpaces → pheno在当前服务器实际运行。**W10结构门槛核对通过；W11来源／编码／时间登记和W12候选映射、选样规则已交付并独立复核。W13菌群技术QC及最终选样尚未执行。**

这是预处理与数据可用性结果，没有拟合生物学关联模型、填补缺失、生成发现／确认名单或冻结最终调整集。W11/W12的勾选表示本包技术交付完成，不表示所有候选均已成为正式分析样本。

## 1. 先看人数：每一个分母分别是什么

| 层级 | 实测结果 | 正确解释 |
|---|---:|---|
| W11全访视协变量登记 | 20,407个人—访视，11,126人 | 必要来源合法键的并集 |
| W11基线登记分母 | 11,124人 | `10k＋00_00_visit`，不是完整病例交集 |
| 基线年龄＋性别可用 | 10,417人 | 还未要求饮食、菌群或某结局 |
| 基线再要求吸烟可用 | 8,219人 | 同上，不能直接当最终模型N |
| W12口腔基线身份可对应 | 8,291人 | 等待W13矩阵及技术QC |
| W12肠道基线身份可对应 | 8,676人 | 等待W13矩阵及技术QC |
| W12双部位基线身份交集 | 7,214人 | 不是最终饮食—双菌群—结局人数 |
| W12全部候选记录 | 20,547行 | 含跨访视及index-only记录，不能当人数 |

例如，总脂模块最终要取“本模块饮食资格＋O/G资格＋总脂资格＋本模块条件变量”的交集；不因为肝脏或CGM缺失删人。血糖另按自己的共同合格日建表。**7,214只是目前双部位身份层的候选交集，不能拿来统一限制所有单部位分析，也不是所有结局的最终样本量。**

## 2. W11具体做了什么、怎样处理缺失

主键始终为`participant_id＋cohort＋research_stage`。空值、占位或无效ID不编造成访视。各必要来源有效键取并集后逐字段接入，唯一键与连接基数全量检查；没有先做全域complete-case删除。

人口学在个人／队列层连接；同访视重复问卷按每个字段的非缺失值一致性汇总，有矛盾就标记并留缺，不随意取最后一条。生活方式和家系问卷各有140组重复人—访视键，人口学键无重复；“140组”不是额外行数。

以下全部以基线登记11,124人为分母：

{covtable}

**年龄是日历年近似年龄。** 当前来源只有出生年／月，没有精确生日；用参考年份减出生年，不伪造DOB。参考日优先取同访视索引饮食窗口中点，再用人体测量明确日期、首个合格CGM日兜底；仍无参考日则缺失。此处近似年龄会有未过生日造成的年内误差。

**性别和吸烟按真实源编码。** 性别code9为Female=0、Male=1，没有悄悄换成gender。吸烟code100347的0=No、1=大多数／每天、2=偶尔、−3=拒答；拒答和冲突不是No。基线No 7,322人、偶尔645人、大多数／每天520人、unknown 2,637人。

**活动、睡眠、饮酒是候选敏感性变量。** 活动天数只用0–7数学范围，睡眠小时0–24；没有按样本分位数挑人。剧烈活动仅2,229人有数值，其缺失可能涉及问卷跳转，未把缺失当0，也没有要求全体这些可选变量齐全。

**口腔健康题已补齐。** 实际源字段拼写为`oralhealth_antibiotics_for_ginvitis`，code021_02是Never／Rarely／Often。基线分别8,510／1,474／20人，unknown 1,120人。它是这道口腔问题的频率，不能改名成“近期全身抗生素使用”。07最初按另一拼写条件查找而未纳入该题；09按实际源拼写补齐，4个旧派生版本存档，当前registry行数和主键不变。并非源数据被修改。

## 3. 用药：有开始时间，但不能据此确定记录饮食时仍在服药

原表43,417条药物记录，其中42,618条有合法显式人—访视键进入时间登记；其余799条未接入有效键。基线有药物来源记录10,269人，但这个人数不是“正在服药人数”。

| 42,618条合法键药物记录的开始信息 | 条数 |
|---|---:|
| 开始信息不可用 | 37,209 |
| 只能定位到报告年份区间 | 2,722 |
| 能定位到报告年月区间 | 2,576 |
| 来源年月／日期冲突 | 111 |

年级信息保留整年的不确定区间；月级信息保留当月首日至末日。没有猜成某一天。**源表没有可据以确认停药／持续服用的字段，记录日也不等于用药开始日。**

与饮食窗口相比，42,618条记录中开始区间明确在前1,486条、在后295条、跨窗口边界152条，其余40,685条unknown。这是药物记录数，涉及多访视，不能用作基线人群百分比。开始在前也不证明窗口内仍然在用。

ATC状态编码表如下，“可判定”包含已列出和在提供的完整编码记录中未列出两种状态，**不是该类药物使用者人数**：

{medtable}

1表示记录列出该代码，0表示在所提供且编码完整的记录中未列出；没有来源／映射不完整且未观察到阳性，则unknown。0不是证实没有服用。关键药物的D×药物交互仍以有效用药时序语义为前提，不能把当前登记直接宣称“饮食记录时在用药”。

## 4. 家系与技术变量的边界

从自报家庭／同住参与者ID重建1,780条唯一有向报告边，其中同住1,353、家系427。观测图中大小>1的连通组1,086个，基线登记中2,244人处于这类组。连通图可以含当前registry之外被报告的参与者；这些组数不是最终分析家庭数。

这是后续统一分组／避免已知亲属跨组的证据，不是全体基因亲缘关系的认证；无自报连接不代表彼此无亲缘。尚未分发现／确认组。

能量、饮食窗口日历、CGM设备、菌群测序深度／批次登记为模块专属接口；当前W11不把错误窗口的能量或未选样本批次填进一张总表。D×年龄、D×性别和条件性的D×关键药物已登记为待协议冻结候选；样条节点、编码、尺度和插补仅将来在相应训练范围fit。

BMI、体脂、VAT、肝脏等并列结局没有自动互相调整。**一般糖尿病／疾病“在饮食时已存在”的变量仍未验收**：来源存在不等于时间定义有效，没有拿眼病问卷中的糖尿病问题冒充一般糖尿病诊断。该项及用药时序若要进正式条件集，须在冻结前明确；不是现在把unknown补成No。

## 5. W12口腔样本怎样找到对应访视

主口腔索引8,816行，其中8,402个合法人—访视，基线8,294人；另414行键无效。口腔丰度汇总8,834行的cohort／stage原字段为占位，不能直接按原键连接。

本轮用“同人＋来源采样日＋唯一有效主索引”恢复访视证据，全量8,834行重建并与既有W03桥核对：全访视8,398条可解，基线8,291人；436条丰度记录无法唯一恢复访视，留作暂挂。**436条访视未定，不能都标为基线。**

此外418条索引记录没有对应丰度行：414条索引键本来无效，4条是合法index-only，其中基线3条。口腔candidate共9,252行=8,834丰度行＋418索引独有行。主索引固定，另一个同规模UUID版本只登记，不混合替换。

这里用采样日找回访视，是身份溯源；不等于要求菌群与饮食同日，也没有引入30／60／90天排除。

## 6. W12肠道：原来599和22组到底怎样了

肠道主索引10,694行、10,670个唯一人—访视，其中22个键重复；基线8,712行、8,694人。肠道丰度11,295行、11,269个唯一键，其中24个键重复；基线9,105行、9,087人。

| 连接状态 | 全访视行数 | 本次处理 |
|---|---:|---|
| 索引和丰度一对一、访视身份可信 | 10,646 | 进入身份可用候选层，基线8,676人 |
| 原599未匹配中：同人同队列仅其他stage有索引 | 185 | 暂挂，不跨stage强行补接 |
| 原599未匹配中：当前主索引完全没有这个人 | 414 | 暂挂，登记来源范围不一致 |
| 重复访视丰度记录，需要标本层映射 | 50 | 24组，含22组双侧多行；不做多对多扩张 |
| 合计 | 11,295 | 每条丰度行只保留一次 |

649=185＋414＋50条暂挂。基线暂挂429**行**，不是429个独立人；上表599问题已定位来源结构，但尚未找回可信标本连接。

聚合肠道丰度本身没有采样日期或sample_name，索引有。`array_index`相等目前只是候选连接提案，不是已证明两表共享同一个生物样本ID命名空间；不能据此把50行任意随机配对。**你的“random选一个”用于已经核实身份、通过技术QC的生物重复样本，不用于修补不确定的两表连接。**

## 7. 最终选样怎样执行：沿用你的随机规则

固定来源版本 → 核实身份及访视 → 区分技术重测和生物重复、执行所需技术QC → 对同人同cohort同stage同部位的合格生物样本随机选一个。

种子固定20261027，用`SHA256(seed|site|candidate_id)`建立不可随结果改变的优先级，在合格且身份明确的不同生物样本中选最小值。不会反复试种子，不按表型、显著性或离饮食最近选样。技术重测不自动当成新的生物样本。**W12只登记候选和优先级，W13才执行最终QC、选样及锁定；本次最终选定样本数=0。**锁定后同一个样本应跨结局模块复用。

采样日期和lag只描述。基线身份可用且能计算距最近实际索引饮食日的候选中：口腔7,186人，带符号距离中位+68天；肠道7,632人，中位+9天。正值表示菌群采在最近饮食日之后。不是距CGM的旧69／10天，也不是两个部位相同分母；不据此排除same-visit，亦不据此证明长期稳定或因果方向。

## 8. 实际复核与运行记录

W10最终07实际exit0及8份产物SHA复核通过，剩余仪器QC／肝旧汇总差异限制照常保留，不冒称W10没有任何限制。

W11/W12的01–12全部实际exit0。独立重读原始必要列，核对20,407个协变量键、122,442个问卷单元、142,849个药物类别单元、42,618条开始时间记录、1,780条家系边、20,407个口腔题单元及全部20,547条候选身份。10核全部20,129条丰度候选的日期距离；12补核418条index-only、836个距离字段，因此全候选距离覆盖完整。

当前13产物哈希、4个旧派生版本、4份二进制副本核验通过。Parquet先写runtime本地盘再复制V2云盘，SHA及行数验证；没有直接在挂载盘流式写二进制。

05预核用未锚定的`kit`筛列时附带命中了少量名称含kit的数值菌列；未据其进行选择或建模、未导出，生产08改用精确白名单。此预核不等于完成菌群矩阵QC。未宣称新的Astra复审或同事人工复核完成。

本地仅保存自编代码、人工转录聚合及聚合输出截图，个体ID、精确日期、完整字典和矩阵均留TRE。远程11清单保留生成当时的范围；之后12的补充验证单独存档，没有改写前一版交付证据。

## 9. 代码、Result和备忘录

以下P、Q、L是服务器路径，不能作为Mac文件直接打开：

- P：`{P}`
- Q：`{Q}`
- L：`{L}`

| 内容 | 服务器位置 |
|---|---|
| 代码01–12 | `P/scripts/w11_w12_covariates_samples_20261028/` |
| W11共享协变量表 | `P/analysis/shared/covariate_registry_TRE_ONLY.parquet` |
| 条件变量角色／缺失表 | `P/config/covariate_roles.yaml`、`P/data_qc/covariate_missingness.csv` |
| 用药时间／家系边 | `Q/medication_timing_TRE_ONLY.parquet`、`Q/relatedness_edges_TRE_ONLY.parquet` |
| W12共享候选表 | `P/analysis/shared/sample_candidates_TRE_ONLY.parquet` |
| 选样规则／原因表 | `P/config/sample_selection_rules.yaml`、`P/data_qc/sample_identity_reasons.csv` |
| 结果／源清单／独立复核／交付 | `Q/`，核心`delivery_TRE_ONLY.json`、`independent_verification_TRE_ONLY.json` |
| 当前W11哈希及旧版本 | `Q/w11_current_artifact_manifest_TRE_ONLY.json`、`Q/w11_before_oral_health_completion/` |
| 远程合并备忘录 | `P/memos/w11_w12_covariates_samples_20261028/MEMO.md` |
| 分包备忘录 | `P/memos/w11_covariates_20261028/MEMO.md`、`P/memos/w12_sample_candidates_20261028/MEMO.md` |
| 12补充验证 | `Q/index_only_lag_verification.json`、`P/memos/w11_w12_covariates_samples_20261028/VALIDATION_SUPPLEMENT.md` |
| 实际日志和退出文件 | `L/01.log`…`12.log`及对应`NN_actual_exit_code` |

本地[代码功能MEMO](../audits/w11_w12_covariates_samples_2026-10-28/MEMO.md)、[聚合缓存](../audits/w11_w12_covariates_samples_2026-10-28/observed_aggregate.json)、[11退出成功证据](../audits/w11_w12_covariates_samples_2026-10-28/11_final_delivery_exit0.png)、[12补充核查证据](../audits/w11_w12_covariates_samples_2026-10-28/12_full_lag_coverage_exit0.png)可直接打开。

## 10. 下一步与尚未解决项

下一步W13：菌群矩阵非负／有限／总量／稀疏度及实有技术标记验收；优先追溯649条肠道暂挂与口腔436条未定访视的来源证据，无法解决者保留原因、按可信身份规则处置。区分技术与生物重复后再实施固定随机选样。

W14再按每个结局自己的饮食、O/G、结局及条件变量资格报告人数；不全域取交集。后续冻结正式调整集时，处理一般疾病／在用药时序是否可识别、缺失策略、最少日数、主次终点和交互检验预算；W15安排家系／历史结果接触与统一角色。未做的内容仍留在后续包，不以W11/W12已勾选代替验收。
'''
(R/'reports'/f'{stem}.md').write_text(report)
memo=f'''# W11/W12代码与Result备忘录

开始2026-10-28，完成2026-10-29。HPP Server Skill，Chrome → WorkSpaces → pheno。原始源只读；本地聚合人工转录，未下载个体表。下列01–12都已从实际终端看到exit0。

{steptable}

Code：`{P}/scripts/w11_w12_covariates_samples_20261028/`

Result：`{Q}/`

Runtime／实际日志与exit：`{L}/`

远程MEMO：`{P}/memos/w11_w12_covariates_samples_20261028/MEMO.md`；两个分包同目录名分别`w11_covariates_20261028`、`w12_sample_candidates_20261028`。12追加`VALIDATION_SUPPLEMENT.md`和`Q/index_only_lag_verification.json`，不改11证据。

共享表：`P/analysis/shared/covariate_registry_TRE_ONLY.parquet`、`P/analysis/shared/sample_candidates_TRE_ONLY.parquet`。配置：`P/config/covariate_roles.yaml`、`P/config/sample_selection_rules.yaml`。缺失／原因CSV：`P/data_qc/covariate_missingness.csv`、`P/data_qc/sample_identity_reasons.csv`。P、Q同上。

registry20407人—访视／11126人，基线union11124人；年龄性别10417、加吸烟8219；口腔题10004。W12基线身份可用O8291／G8676／双部位7214，非最终分析N。总候选20547行。599肠道未匹配拆为185其他stage＋414主索引无人；另50重复丰度行（24组）待标本证据。不做多对多扩张、无日期差门槛。随机seed20261027，仅对身份和QC均过的生物样本在W13执行；当前final_selected0。

年龄为日历年近似值；药物有开始年／月但无停用持续证据，编码0不等于未服药；口腔题Never/Rarely/Often不是近期全身抗生素；家系只是自报连接。一般疾病时间定义、最终调整集与技术变量实例留后续验收，未做插补／fit／关联模型／发现确认分组。01–12实际0、13产物hash／4旧版本／4二进制镜像核查通过。未冒称同事人工或Astra新审查。

07漏查口腔字段由09补齐；旧4派生版本及初次manifest完整保留，当前manifest单独发布。05的kit子串预核附带少量菌列，生产08精确白名单排除，不冒称W13矩阵QC完成。

[完整报告](../../reports/{stem}.html) · [人工转录聚合](observed_aggregate.json) · [11实际退出](11_final_delivery_exit0.png) · [12实际退出](12_full_lag_coverage_exit0.png)
'''
(A/'MEMO.md').write_text(memo)
v={'passed':True,'provenance':x['provenance'],'actual_exit_codes':x['actual_exits'],
 'covariate_rows':20407,'candidate_rows':20547,'artifact_hashes_verified':13,'prior_revisions_verified':4,
 'binary_mirrors':4,'full_lag_coverage_verified':True,'union_not_complete_case':True,
 'W13_technical_QC_done':False,'final_samples_selected':0,'primary_protocol_frozen':False,
 'raw_individual_exported':False,'team_human_review_claimed':False,
 'evidence':['audits/w11_w12_covariates_samples_2026-10-28/11_final_delivery_exit0.png',
 'audits/w11_w12_covariates_samples_2026-10-28/12_full_lag_coverage_exit0.png']}
(A/'delivery_validation.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
marker='2026-10-29 W11协变量与W12候选样本'
addition=f'''

## {marker}

W10实际exit0／8产物哈希gate通过后，W11和W12于当前TRE/pheno运行并独立复核，01–12实际exit0。协变量union20407人—访视／11126人，基线11124人；年龄＋性别10417，加吸烟8219，口腔题10004可用。均非最终饮食／菌群／结局交集。年龄为参考年减出生年，非精确生日年龄；口腔字段实际拼写oralhealth_antibiotics_for_ginvitis，code021_02为Never/Rarely/Often，不是全身近期抗生素。09补齐该题，4旧派生版本存档，键数不变。一般疾病时间定义仍未验收。

药物43417原始行中42618合法键行；开始信息37209缺失、2722年区间、2576年月区间、111源冲突；与饮食开始时间40685条unknown，无停用／持续信息，ATC未列出不等于未服用。自报家系同住1780有向边，观测组>1的基线2244人，无连接不等于无亲缘；未分发现确认组。可选活动／睡眠等不要求全齐，无缺失填0、无模型／插补fit。技术变量保留窗口／样本接口，最终条件集后续冻结。

W12全候选20547行=20129丰度＋418口腔index-only。基线身份可用口腔8291／肠道8676／双部位7214人，非最终技术QC或模型N。口腔全访视8398丰度行恢复身份，436丰度行访视未定不可全称基线。肠道599未匹配拆185同人仅其他stage有索引＋414主索引无人；另50重复丰度行、24组待标本证据，含22组双侧多行，全部暂挂不随机强配。每条源丰度只出现一次，未多对多扩张。原肠道聚合无采样日／sample_name，不能把array_index相等当标本ID已认证。

same-visit准入不加日期差门槛；距最近实际索引饮食日signed lag中位O+68（n7186）、G+9（n7632），仅描述，不是旧CGM距离。最终选样沿用用户随机规则：固定版本、身份／技术QC后，对合格生物样本固定seed20261027与SHA优先级，W13执行；本次选定0、W13未做。13产物hash、4旧版本、4二进制副本及所有20547候选lag均独立复核。

Code：`{P}/scripts/w11_w12_covariates_samples_20261028/`；Result：`{Q}/`；MEMO：`{P}/memos/w11_w12_covariates_samples_20261028/MEMO.md`。共享表位于P/analysis/shared的covariate_registry_TRE_ONLY.parquet与sample_candidates_TRE_ONLY.parquet。12的VALIDATION_SUPPLEMENT.md单存，11交付清单不覆盖。个体表／ID／日期／完整字典留TRE；本地仅自编源、人工聚合与聚合截图。

本地[完整报告](reports/{stem}.html)、[代码功能备忘录](audits/w11_w12_covariates_samples_2026-10-28/MEMO.md)、[聚合缓存](audits/w11_w12_covariates_samples_2026-10-28/observed_aggregate.json)。下一步W13矩阵／实有技术QC及最终选样，然后W14分结局建表；未冒称新的Astra／团队人工复核。
'''
records=[]
for name in ['HPP_DATA_BACKGROUND.md','HPP_SERVER_BACKGROUND.md']:
 p=R/name; b=p.read_bytes(); assert marker.encode() not in b
 snap=A/'before_w11_w12_update'/name;snap.parent.mkdir(parents=True,exist_ok=True);assert not snap.exists();snap.write_bytes(b)
 p.write_bytes(b+addition.encode());assert p.read_bytes().startswith(b)
 records.append({'path':name,'snapshot':str(snap.relative_to(R)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'append_only':True,'after_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(A/'background_append_manifest.json').write_text(json.dumps({'date':'2026-10-29','records':records},indent=2)+'\n')
for name in ['HPP_LOCAL_VERIFIED_FACTS.md','HPP_PROJECT_BACKGROUND.md','reports/HPP_CURRENT_PLAN.md']:
 p=R/name;t=addition
 if name.startswith('reports/'):t=t.replace('(reports/','(').replace('(audits/','(../audits/')
 assert marker not in p.read_text();p.write_text(p.read_text()+t)
progress=D/'workpackage_progress.json';p=json.loads(progress.read_text())
for w,title,status,ran in [
 ('W11','协变量来源／编码／时间登记完成；正式条件集待冻结','registry20407人—访视、基线11124；年龄性别10417、加吸烟8219，口腔题10004。union而非完整病例交集。药物开始区间可用性／无持续证据、一般疾病时间定义缺口、自报家系范围均登记；技术变量留模块接口。未fit或分组。','01–07来源与编码／时间预核及union构建；09补齐实际口腔字段并保存4旧版本；10独立复核；11备忘录／交付。'),
 ('W12','全部候选／身份映射／随机规则完成；W13技术QC和最终选样未执行','候选20547行，基线身份可用O8291/G8676/双部位7214，非最终N。肠道599未匹配拆185其他stage＋414索引无人；另50重复行需标本证据；口腔436行访视未定。无日期差门槛，随机seed20261027仅待身份和技术QC后的生物样本；选定0。','01–06索引／来源预核；08全部候选及随机契约；10全部身份与丰度候选lag独立复核；11交付；12补齐418个index-only的836个距离字段。')]:
 p['tasks'][w]={'checked':True,'short_status':title,'status_text':'2026-10-29当前TRE实跑，'+status+' 01–12全部实际exit0，13产物hash／4旧版本／4二进制副本验证；未冒称同事人工或新Astra复审。','what_ran':ran,
 'code_location':f'服务器 `{P}/scripts/w11_w12_covariates_samples_20261028/`；本地[代码功能MEMO](../audits/w11_w12_covariates_samples_2026-10-28/MEMO.md)。',
 'result_location':f'服务器 `{Q}/`；共享表 `P/analysis/shared/`，配置 `P/config/`；P同代码项目根。远程MEMO `P/memos/w11_w12_covariates_samples_20261028/MEMO.md`；本地[结果与人数]({stem}.html)、[聚合缓存](../audits/w11_w12_covariates_samples_2026-10-28/observed_aggregate.json)。'}
p['tasks']['W12']['action_override']='按同人同队列同访视与可信身份建立全部候选map。沿用用户已确认规则：固定来源版本→身份／技术QC与技术重测处理→对合格生物重复样本按固定seed20261027、SHA256(seed|site|candidate_id)优先级随机选一个；不按结局、效应或最近日期选样。身份不确定者暂挂，不能以random修补两表连接。记录日期缺失及signed lag，不加固定日期差门槛。W12只登记契约，最终QC和选样由W13执行并锁定。'
p['evidence']='audits/w11_w12_covariates_samples_2026-10-28/MEMO.md';p['latest_run_id']='w11_w12_covariates_samples_20261028';p['latest_local_workpackage']='W11_W12';p['latest_local_run_id']=p['latest_run_id'];p['next_recommended_workpackage']='W13';p['next_parallel_workpackages']=[]
progress.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
u=D/'execution_update.md';u.write_text(u.read_text()+f'\n\n### {marker}\n\nW11来源／时间／编码登记、W12全部候选身份映射与用户随机规则已完成并独立复核。W11基线union11124，年龄性别10417／加吸烟8219；W12基线身份层O8291／G8676／双部位7214，非最终N。01–12实际exit0。用药活跃时段、疾病时间语义、家系完整性与样本身份暂挂均有记录。下一步W13矩阵／技术QC及最终随机选样，当前选定0。\n\n[完整结果及筛选解释]({stem}.html) · [跑了什么、Code／Result／MEMO](../audits/w11_w12_covariates_samples_2026-10-28/MEMO.md)。\n')
run=A/'RUNNING.md';run.write_text(run.read_text()+'\n\n## 2026-10-29最终实测更新\n\n此前进行中状态均为历史。09、10、11和12已实际运行exit0，聚合截图11_final_delivery_exit0.png与12_full_lag_coverage_exit0.png已存。全部01–12实际0；13当前产物hash／4旧派生版本／4镜像通过。12补齐418索引独有行的836个lag，未改数据或选样。W11/W12技术交付完成，W13未做、最终选定0。当前代码／Result路径及人数以MEMO.md和observed_aggregate.json为准。\n')
print('Saved W11/W12 local report, aggregates, memo, progress and append-only backgrounds.')
