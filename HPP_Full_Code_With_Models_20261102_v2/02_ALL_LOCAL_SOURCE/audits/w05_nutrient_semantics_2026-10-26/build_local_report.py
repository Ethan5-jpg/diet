"""Build W05 local aggregate-only report and reference config; never read TRE data."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent
v=json.loads((D/'observed_aggregate.json').read_text());run=v['run'];P='/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021';Q=f'{P}/data_qc/{run}'
fields=['weight_g','calories_kcal','carbohydrate_g','lipid_g','protein_g','alcohol_g','dietary_fiber_g']
cfg={'status':'event_unit_semantics_verified_quality_cleaning_pending','scope':'Local authored reference; canonical executed config is the TRE run file. No individual data exported.','tre_canonical_config':f'{P}/config/{run}/nutrient_semantics.yaml','run':run,'official_reference':'https://knowledgebase.pheno.ai/datasets/005-diet_logging.html','primary_source':'diet_logging/diet_logging_events.parquet','raw_source':'diagnostic only; never pool with processed or infer baseline from undefined','fields':{f:{'unit':'kcal' if f=='calories_kcal' else 'g','basis':'logged_food_event_weight' if f=='weight_g' else 'logged_food_event_intake','multiplier':1.0,'weight_rescaling':False,'missing':'preserve_NA','invalid':'flag; W06 must adjudicate before feature aggregation'} for f in fields},'energy':'preserve source calories_kcal; macro energy diagnostics are not replacements','alcohol':'observed alcohol_g only; no energy-residual recovery, assumed ABV or missing-to-zero','unresolved':['W06 edits/duplicates/missing/extreme events and raw visit lineage','W07 NOVA mapping/version','carbohydrate includes/excludes fiber, fiber/polyol energy factors','mixed recipes and raw/cooked ingredient composition'],'individual_features_computed':False,'formal_protocol_frozen':False}
cp=ROOT/'config/nutrient_semantics.yaml';cp.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
lines=['# W05 营养素单位、转换与质量核查','',f'执行日期：2026-10-26；服务器运行 `{run}`。**W05单位含义与转换规则已验收；完整饮食清洗、食物映射及正式特征仍未完成。**','',
'## 1. 最重要的结论','',
'当前HPP事件表中的能量、纤维、碳水、脂肪、蛋白和酒精已经对应这条食物记录的摄入量。重量是这条记录的食物重量。**原值保留，转换乘数为1；不能再统一乘 `weight_g/100`。** 当前服务器字典逐字段校验通过，且与[Pheno官方饮食数据说明](https://knowledgebase.pheno.ai/datasets/005-diet_logging.html)相符。','',
'这确认的是记录字段的含义，不证明全部记录有效、没有漏报，或App记录时间就是实际进食时间。三种食物层级仍分开：日索引提供记录日信息；处理后事件作为候选营养输入；raw事件用于后续编辑与来源核查，不与处理后事件累加。','',
'## 2. 实际做了什么、扫描了多少','',
'| 来源 | 本轮处理 | 全源事件行数 | 明确满足10k基线字面条件的事件行数 |','|---|---|---:|---:|',
'| 处理后食物事件 | 7个数值字段、food_id及cohort/stage完整分批投影 | 2,640,959 | 2,059,935 |',
'| 原始食物事件 | 同一投影，独立扫描，不并入摄入总量 | 6,485,022 | 0，不能解读为没有基线参与者，见下文 |',
'| 饮食日索引 | schema与元信息核查；没有上述营养素字段 | 149,912（metadata） | 本轮不重算 |','',
'两层事件合计扫描9,125,981行，**这是扫描工作量，不能把两表相加作为实际食物摄入记录数，更不是人数。** 基线字面条件为 `cohort=10k` 且 `research_stage=00_00_visit`；本轮没有重做参与者准入或产生新的分析N。既有W03的9737个食物基线人—访视也不是本轮最终分析N。','',
'07补核确认：raw的6,485,022行全部为cohort=10k、research_stage=undefined。因此原始表的字面基线子集为空，本报告只解释其全源数值质量，不能用它宣称零名基线参与者，更不能把占位访视自动改成基线。另以07脚本完整投影cohort/stage解释这一范围，结果见服务器 `raw_stage_scope.json` 与本地聚合缓存。W06需要建立raw与处理后事件的来源／编辑关联。','',
'主程序、处理后扫描、raw扫描实际退出码均0；独立交付核验退出0。生成28行“来源×范围×字段”聚合审计Parquet，并重新读取源文件验证12个事件例子。例子保留在TRE。两次早期失败及修正均保留：先修正字典说明列名，再关闭Pandas元信息自动恢复索引；没有把失败运行标成成功。','',
'## 3. 缺失与异常：不要误当作0','',
'| 范围 | 字段 | 缺失事件数（比例） | 负值事件数 | 零值事件数 |','|---|---|---:|---:|---:|']
labels={'processed_all':'处理后·全源','processed_baseline_10k_literal':'处理后·明确基线','raw_all':'raw·全源'}
for scope,s in v['counts'].items():
 for f in ['weight_g','calories_kcal','dietary_fiber_g','alcohol_g']:
  z=s[f];lines.append(f"| {labels[scope]} | `{f}` | {z['missing']:,}（{z['missing_percent']:.3f}%） | {z['negative']:,} | {z['zero']:,} |")
lines+=['',
'处理后全源另有324条“正能量但重量缺失或不大于0”、3条“纤维超过正重量”；明确基线分别277条、3条。raw全源分别579条、7条。不同异常可能重叠，不能直接把数量相加当作独立异常总数。','',
'处理后全源能量与重量没有负值；纤维最小−13.3g、最大1545.12g，酒精最大3476.25g。这些范围提示需要逐项追查食物ID／来源，而非证明极值合理。字典重量2000g与能量5000kcal上限之外为0条，也不等于范围内的数值都合理。零纤维可能合理，零能量也可能对应某些食物；必须和缺失、负值分开，W06再确定规则与人数流转。','',
'缺失纤维不能在求和时悄悄跳过并解释成全天真实纤维；已观测部分的和需要标注完整度，日／人层面的可用规则待W06。能量缺失会影响UPF分子和总能量分母；纤维缺失则还影响纤维密度。这里没有直接删除或填补任何源记录。','',
'## 4. 为什么确信不应再按重量换算','',
'依据首先是当前字段字典，其次是数值行为的印证。对同一食品ID，取全源正重量最小／最大记录，要求重量至少相差两倍，并要求对应营养素正值且可观测：','',
'| 来源／营养素 | 可比较食品ID数 | 每克含量在1%以内一致 | 事件总量近似恒定 |','|---|---:|---:|---:|',
'| 处理后／能量 | 6,864 | 6,864 | 0 |','| 处理后／纤维 | 4,763 | 4,763 | 0 |','| raw／能量 | 7,436 | 7,436 | 0 |','| raw／纤维 | 5,165 | 5,165 | 0 |','',
'这是选定条件下同食品ID的份量一致性检查，不是随机抽样、菌群稳定性或进食反应可重复性检验。1%是数值比较容差，不是临床排除阈值。字典与数据行为共同支持事件摄入量，不能单靠相关系数决定单位。','',
'**合成例子（教学用，不是导出的受试者记录）：**某食品100g记录为250kcal、5g纤维；200g记录已经是500kcal、10g纤维。正确汇总是750kcal、15g。若再乘重量/100，第二条就被错误变成1000kcal、20g，最终得到1250kcal、25g。12个真实例子及原始行号、正确保留值、错误二次换算对照都在服务器，并已独立从源行读回核对。','',
'## 5. 能量重建与酒精、纤维的边界','',
'记C为碳水g、L为脂肪g、P为蛋白g、A为酒精g、F为纤维g。以下是处理后全源的诊断，均只用公式所需字段已观测的记录；不同公式的分母不同，且包含尚未清洗的异常值。','',
'| 诊断公式 | 可计算事件数 | 与源能量的平均绝对差（kcal/事件） | 差值>1kcal事件数 |','|---|---:|---:|---:|']
for name,z in v['energy_diagnostics_observed']['processed_all'].items():lines.append(f"| {name} | {z['n']:,} | {z['mae_kcal']:.4f} | {z['gt1']:,} |")
lines+=['',
'加上纤维项后的平均误差较小，**不能据此认定源数据库一定采用该纤维供能算法**：碳水是否包含纤维、糖醇／有机酸、食品数据库来源和舍入等尚不清楚。这里也没有在相同完整案例子集上比较所有公式。保留源能量和各诊断重建量；不按最小误差选公式替换源能量。','',
'旧项目代码存在用能量残差反推酒精，或用默认酒精浓度与重量估算、最后缺失补0的逻辑；这些是历史研究代码假设，不是本轮已验证规则。本轮只保留观测到的 `alcohol_g`，缺失继续未知，不默认食品g等于饮料mL。','',
'混合菜谱／生熟重方面，当前字段说明是事件摄入量，故不额外统一做干重、生重、熟重或菜谱倍数转换；底层配方和营养数据库的精确来源仍未确认。若以后引入每100g食物主表，必须为那张新表单独核查单位和版本，再在明确关联后换算，不能把它与事件字段混用。','',
'## 6. 目前可以继续什么，哪些仍未放行','',
'- W05单位与事件转换规则已完成，可进入W06；UPF和纤维两个候选没有因单位不明而被否决。','- W06处理编辑／重复、缺失、负值、可疑极值、日记录完整度和raw来源关联；此处尚未产出清洗后日表。','- W07核对食品ID、版本与NOVA映射；没有生成正式UPF值，也没有把未分类食物当非UPF。','- 用确认后的源能量与通过QC的纤维构造候选 `1000×F/E`；是否等权日均、完整度准入和最终协议另按W06／W18处理。','- “严格等能量碳水—脂肪替代”仍需解决供能闭合与酒精／纤维约定，不能仅凭本次单位验收就宣称成立。','- 未读临床结局或菌群数值、未训练模型、未划分发现／确认名单、未改变V3科学设计或冻结正式主方案。','',
'## 7. 跑了什么代码、代码位置、Result在哪里','',
'| 代码 | 功能 |','|---|---|','| `01_probe_sources.py` | 当前三层表schema、目录与相关字典字段探查，不打印参与者值 |','| `05_validate_nutrients.py` | 当前字典断言、两表完整数值投影、质量／份量／能量诊断、源例子读回、配置和MEMO |','| `06_verify_w05_delivery.py` | 主程序及子进程实际exit、输入指纹、产物hash、Parquet行数、配置与MEMO独立读回 |','| `07_confirm_raw_stage_scope.py` | raw表完整cohort/stage投影，解释字面基线子集，不生成访视 |','| `build_local_report.py` | 从人工转录的聚合观察构建本地报告和参考配置，不访问TRE |','',
f'服务器代码：`{P}/scripts/{run}/`。', '',f'服务器Result：`{Q}/`；核心文件 `nutrient_conversion_audit.parquet`、两层 `*_summary.json`、`summary.json`、`verification.json`、`delivery_readback.json`、`raw_stage_scope.json`；事件例子与份量对照 `_TRE_ONLY.parquet` 不导出。','',
f'服务器配置：`{P}/config/{run}/nutrient_semantics.yaml`；当前指针配置 `{P}/config/nutrient_semantics.yaml`。','',
f'服务器备忘录：`{P}/memos/{run}/MEMO.md` 与 `RAW_SCOPE_MEMO.md`；交接入口 `{P}/W05_HANDOFF.md`、`W05_LATEST_RUN.json`。','',
'本地自编代码、失败记录、聚合缓存及截图：[W05目录](../audits/w05_nutrient_semantics_2026-10-26/MEMO.md)；本地参考配置：[nutrient_semantics.yaml](../config/nutrient_semantics.yaml)。本地JSON是聚合画面的人工转录，完整聚合Parquet和字典证据仍留TRE；不冒称下载了源文件或字节完全一致的服务器配置。','',
'## 8. 服务器验收画面','',
'![W05聚合读回与实际退出码](../audits/w05_nutrient_semantics_2026-10-26/remote_delivery_page1.png)','',
'![W05份量与能量诊断](../audits/w05_nutrient_semantics_2026-10-26/remote_delivery_page2.png)','']
report='\n'.join(lines);(ROOT/'reports/HPP_W05_Nutrient_Semantics_2026-10-26.md').write_text(report)
(D/'REPORT.md').write_text(report.replace('../audits/w05_nutrient_semantics_2026-10-26/','').replace('../config/','../../config/'))
memo=f'''# W05代码与结果备忘录\n\n执行run：{run}；当前主机ip-10-0-91-109；pheno。\n\n跑了什么：05_validate_nutrients.py校验当前两层事件字典、完整投影扫描9,125,981行，检查缺失/负值/份量比例及4类能量公式，源文件独立读回12例。06_verify_w05_delivery.py核对实际退出及产物；07_confirm_raw_stage_scope.py单独解释raw访视范围。\n\n本地代码：本目录01、05、06、07及build_local_report.py；前两次失败源码和failure_log.json保留，成功05源码SHA256为3d6bd89f84cf17eefbc5f1a13599fbb4f4b3b1b77bdcc48ddffa87effa2835ce。修改后重跑会生成新run，不覆盖旧扫描。\n\n服务器代码：{P}/scripts/{run}\nResult：{Q}\n配置：{P}/config/{run}/nutrient_semantics.yaml\nMEMO：{P}/memos/{run}/MEMO.md；raw范围补充备忘录RAW_SCOPE_MEMO.md。\n实际退出：主程序0，processed/raw各0，06验证0；07实际退出0，全量确认raw访视占位。\n\n关键规则：事件摄入量原值，乘数1，不再按重量/100换算；缺失保留未知；源能量保留；不用能量残差/默认ABV补酒精。W06完整清洗、W07NOVA映射及纤维能量/菜谱约定未完成。W05不等于最终日表、合格N或模型结果。\n\n本地observed_aggregate.json仅人工转录聚合截图，不含个体ID/日期/事件明细。原始数据和真实例子留TRE。\n'''
(D/'MEMO.md').write_text(memo)
print('Built W05 MD, aggregate reference config and MEMO.')
