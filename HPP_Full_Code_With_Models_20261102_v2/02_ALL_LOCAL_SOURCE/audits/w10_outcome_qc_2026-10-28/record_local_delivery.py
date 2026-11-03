# -*- coding: utf-8 -*-
"""Create local aggregate handoff/report and dated, append-only background updates."""
from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[2];A=Path(__file__).parent;D=ROOT/'outputs/researchwrite/hpp_multisystem_20261021/team_checklist_20261026';P='/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021';L='/home/ec2-user/multisystem_runtime_20261021/w10_outcome_qc_20261028'
x=json.loads((A/'observed_aggregate.json').read_text());x['status']='W10_technical_delivery_and_final_exit07_verified';x['actual_exits']['07']=0;x['CGM']['all_stage_availability_visits']=9489;x['liver_rounding_envelope']=[{'outcome_id':oid,'comparable_all':n,'rounding_compatible_all':compatible,'larger_all':large,'baseline_comparable':nb,'baseline_larger':lb,'baseline_gt_0_01':g} for oid,n,compatible,large,nb,lb,g in [('liver_elasticity',8936,8859,77,6757,53,49),('liver_velocity',8936,8894,42,6757,28,19),('liver_viscosity',8935,8877,58,6756,40,38),('liver_dispersion',8935,8840,95,6756,71,69)]];(A/'observed_aggregate.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
rows=x['counts'];stem='HPP_W10_Outcome_Fields_QC_2026-10-28'
table='| 结局 | 基线来源人数 | 本结局基本QC后 | 同访视有索引饮食日≥1 | 同访视有能量合格饮食日≥1 |\n|---|---:|---:|---:|---:|\n'+'\n'.join(f"| {r['label']} | {r['baseline_source_people']:,} | {r['baseline_eligible_people']:,} | {r['baseline_with_indexed_food_same_visit']:,} | {r['baseline_with_energy_food_same_visit']:,} |" for r in rows)
fields='| 结局 | 原始字段 | 输出单位 |\n|---|---|---|\n'+'\n'.join(f"| {r['label']} | `{r['source_field']}` | {r['unit']} |" for r in rows)
roundtable='| 肝指标 | 可比较的基线人数 | 超出两位小数舍入幅度的人数 | 比例 |\n|---|---:|---:|---:|\n'+'\n'.join(f"| {next(v['label'] for v in rows if v['outcome_id']==r['outcome_id'])} | {r['baseline_comparable']:,} | {r['baseline_larger']} | {100*r['baseline_larger']/r['baseline_comparable']:.2f}% |" for r in x['liver_rounding_envelope'])
report=f'''# HPP W10：结局字段、单位与基本质量核查

2026-10-28，已在当前HPP服务器通过Chrome → WorkSpaces → pheno实际运行并独立复核。**W10的字段和基本数值QC交付已完成；最终分析队列、制造商级技术QC、主次终点与模型协议冻结是不同状态。**

本轮没有训练关联／预测模型，没有按显著性选择字段，没有生成发现／确认分组。各结局独立准入，不取全部表型的全交集；UPF分类问题不阻塞本包。

## 1. 人数要怎样读

下表全部是`cohort=10k`、`research_stage=00_00_visit`。每列按**同一参与者、同一cohort、同一明确访视**计人，没有按具体日期差筛人。BMI、总脂、VAT及每个肝指标分别独立统计。

{table}

“本结局基本QC后”仅要求这个结局本身有有效值；不要求同时有饮食、CGM、其他肝指标或双菌群。“有饮食日≥1”只是同访视饮食来源存在性诊断，不是冻结的最少记录天数，也不是最终模型人数。未要求UPF精确点值、纤维完整、协变量完整、O/G有效或家庭角色。W11–W14将按实际问题继续形成独立资格／窗口／协变量层。

血糖方面，W09已提供**9,402名基线参与者**的合格CGM日；W10发布9,489个全访视CGM可用日计数接口，登记计算公式，**没有生成整段CGM均值来替代模块专属结局**。人级Y在W14针对相应暴露／pair的共同合格日重算。

## 2. 本轮实际使用的字段

{fields}

总脂当前登记**脂肪重量g**，VAT登记**DXA估计体积cm³**；没有把重量、比例、面积、体积混作同一个结局。总脂／VAT是供后续协议冻结的明确字段候选；BMI仍是次要候选，未自动增加主要终点预算。没有做log VAT或按结果选尺度。六项肝表型沿用你已确定的范围，没有添加min／max／SD结局或合成总分。

弹性与剪切波速度不能当成肝脂肪量；衰减是与肝脂相关的超声代理，不等于组织学脂肪百分比。黏度、频散和声速不预设“越高一定越健康／不健康”。

## 3. 具体怎样筛选、怎样合并

1. 从只读原始表读取必要列，核对本次来源行数、文件大小、修改时间和字典单位。先检查Arrow schema，再按批投影读取；不靠本地旧数据替代远程来源。
2. 必须有合法`participant_id＋cohort＋research_stage`，拒绝空值／占位和无效ID，不把undefined补成基线。每域的来源与排除数保存在TRE。
3. 有限数值加数学定义检查：脂肪重量、BMI及六项肝测量须大于0；VAT允许0并单独标记。不用样本分位数删极端值，不臆造临床阈值或厂商QC。
4. 身体／BMI当前来源的人—访视键唯一，直接保留原字段。BMI不因身高／体重组件缺失自动删掉已提供的有效BMI，也不在本轮补算缺失BMI。
5. 肝脏按你指定的规则，对同访视**每个字段自己的有效明细**等权取算术均值。保留每项有效测量数；全缺失就是缺失，不补0；不要求六项同时有效。
6. 日期仅保留为描述性元信息；不猜数值timestamp的单位，不因timestamp缺失、日期不唯一或距饮食较远而排除同访视状态结局。日级CGM仍按W09真实日历日和点覆盖QC处理。

因此，总脂6,671→6,595人：75人该字段缺失、1人值为0而未通过脂肪重量检查。VAT6,671→6,513：158人缺失，**31人的零值仍保留并标记**，尚不能解释为完全没有内脏脂肪。BMI10,402→10,382：20人该字段缺失。肝脏6,782人中，各字段全缺失的分别为24、24、25、25、24、18人；其余按字段独立取均值。

这些是**基本数据可用性规则**。当前检查的表中没有逐条制造商／设备QC标记，不能声称完成了完整仪器层面的质量认证；将这一限制随字典传给后续分析，额外技术排除须有来源、先登记后执行。

## 4. 三个实际发现与处理

**体脂比例的尺度与字典不一致。** 两个非选定比例列实际存储为0–1，字典却写%。8,989个可比记录与脂肪重量／相应总质量之比核对，最大绝对误差小于5×10⁻¹⁰。例如存储0.30应解释为30%，不能当0.30%。本轮选定总脂重量，比例列仅作QC诊断；将来启用百分比需显式×100、保留原值和版本，不能静默改源表。

**BMI定义一致。** 全访视12,925个可比较记录，源BMI与`体重kg/(身高cm/100)²`完全一致，最大绝对误差0；没有因此补算缺失或额外改变准入。

**肝脏旧汇总不能直接当成本次明细均值。** 四个旧汇总字段与明细均值多数差异在±0.005以内，与保留两位小数兼容；但仍有少量超过这个幅度：

{roundtable}

这些较大差异的上游生成原因，当前字典没有说明，不能编造成某种质量剔除或“旧表做错了”。全访视最大绝对差分别1.385 kPa、0.392 m/s、0.925 Pa·s、1.64 (m/s)/kHz；这是差异诊断，不据此换终点或挑人。原始明细额外有32个人—访视键没有旧汇总，本次六项仍按明细各自有效性处理，缺旧汇总不会直接删人。

本项目按已确定的规则，统一从明细构建六项算术均值。旧汇总保留作来源对照，不能混用、不能声称完全复现官方汇总算法。`liver_aggregate_diagnostic.json`及`liver_rounding_envelope_diagnostic.json`记录全部比较。

## 5. 血糖结局的接口

人级候选公式是**本模块共同合格日内，日均血糖的等权平均**，单位mg/dL。某模块若只用8个共同合格日，就用这8天计算饮食和血糖；不能把饮食8天与整段13天的均值直接配起来。pair同样从对应基础资格和共同窗口重算。

W10在`outcome_dictionary.yaml`登记此公式及W09规则路径，另存`cgm_outcome_availability_TRE_ONLY.parquet`。九个静态结局的长表没有伪造一列已完成的CGM人级Y。≥7日等最少天数仍未全局冻结。

## 6. 验证了什么

- 九个静态结局，共84,962条全访视人—访视—结局记录；测量级QC共192,434条。它们是长表行数，**不是独立参与者人数**。
- `05`从所有选定原始列独立重读，以逐行累加／`math.fsum`重新构建每个组的均值、有效数、来源数及零值数；与构建表全量核对通过。
- 各结局基线人数与同访视饮食存在性计数独立复核；没有无关域缺失导致的全局删人。
- 三份Parquet先写runtime本地盘，再字节复制到V2云盘；SHA与Parquet行数验证通过；八份构建产物hash再次核对通过。
- 脚本01、修正版02b、03、04、05、06、07实际退出均0。初版02因Pandas自动恢复索引、键不在DataFrame列而失败，退出1及日志保留；02b用`ignore_metadata=True`修复，没有改原始数据或把失败覆盖成成功。
- 最新same-visit与肝六项均值用户规则已同步到TRE独立版本目录。没有冒称同事人工复核或新的Astra评审完成。

## 7. 代码、Result、备忘录在哪里

以下是**服务器路径**，不伪装成本地链接。项目根`P={P}`。

| 内容 | 服务器位置 |
|---|---|
| 01–07脚本（含初次失败02与修复02b） | `P/scripts/w10_outcome_qc_20261028/` |
| 聚合结果、明细QC、源清单、独立验证、差异诊断、交付记录 | `P/data_qc/w10_outcome_qc_20261028/` |
| 九静态结局共享长表 | `P/analysis/shared/outcome_registry_TRE_ONLY.parquet` |
| 字段字典与CGM计算契约 | `P/config/outcome_dictionary.yaml`（JSON兼容YAML） |
| 用户最新规则快照 | `P/config/w10_user_rules_20261028/` |
| 聚合人数／字段QC表 | `P/data_qc/outcome_field_QC.csv` |
| 远程代码功能备忘录 | `P/memos/w10_outcome_qc_20261028/MEMO.md` |
| 本地运行盘与实际日志／退出文件 | `{L}/` |

本地[代码功能备忘录](../audits/w10_outcome_qc_2026-10-28/MEMO.md)、[聚合缓存](../audits/w10_outcome_qc_2026-10-28/observed_aggregate.json)、[退出0证据](../audits/w10_outcome_qc_2026-10-28/07_final_delivery_exit0.png)、[本轮自编构建代码](../audits/w10_outcome_qc_2026-10-28/04_build_outcomes.py)、[独立复核代码](../audits/w10_outcome_qc_2026-10-28/05_independent_verify_and_memo.py)均已保存。本地只有自编代码、聚合人工转录与聚合日志截图，原始个体数据、完整字典、ID、日期和特征矩阵留TRE。

## 8. 下一步

W10基本字段和接口可供后续使用。接下来是W11协变量；W12→W13完成双部位样本映射、技术QC和一次固定选样，可由团队另一条线推进。之后W14才汇合成各结局自己的分析表，分别报告每一步的实际人数。正式统计模型、终点检验预算及最少有效天数须按计划在后续统一冻结。
'''
(ROOT/'reports'/f'{stem}.md').write_text(report)
memo=f'''# W10本地代码与交付备忘录

日期：2026-10-28。通过HPP Server Skill的Chrome/WorkSpaces运行，pheno Python。原始数据只读，个体表均留TRE。本地数值来自屏幕聚合人工转录。

| 代码 | 功能 | 实际结果 |
|---|---|---|
| 01_probe_contract.py | 五张表schema／当前字典／字段单位预核，不读个体行 | exit0 |
| 02_probe_values.py | 身体与BMI必要列的单变量预核；初版自动恢复Pandas索引导致cohort不在列中 | exit1，保留 |
| 02b_probe_values_index_fix.py | 显式ignore_metadata=True，修复列／索引语义；不修改源 | exit0 |
| 03_probe_final_fields.py | 精确总脂／VAT／肝六字段、单位、缺失和零值诊断 | exit0 |
| 04_build_outcomes.py | 各结局独立基本QC，六肝明细均值，9静态结局长表，CGM可用日接口，字典与人数 | exit0 |
| 05_independent_verify_and_memo.py | 原始列全量独立重算84962个结局组，守恒／食物存在性／SHA／3副本验证 | exit0 |
| 06_liver_aggregate_diagnostic.py | 比较官方旧汇总和明细均值，量化差异，不换结局 | exit0 |
| 07_finalize_delivery.py | 量化两位小数舍入兼容幅度；最终hash、退出读回、交付清单、远程状态与MEMO | exit0 |

远程Code：{P}/scripts/w10_outcome_qc_20261028/

Result：{P}/data_qc/w10_outcome_qc_20261028/

MEMO：{P}/memos/w10_outcome_qc_20261028/MEMO.md

共享长表：{P}/analysis/shared/outcome_registry_TRE_ONLY.parquet

字典：{P}/config/outcome_dictionary.yaml；聚合CSV：{P}/data_qc/outcome_field_QC.csv

Runtime：{L}/，保存01／02／02b／03–07.log及实际exit记录。07_actual_exit_code由外层包装器在脚本退出后写入，已屏幕确认0。

基线有效：总脂6595，VAT6513，BMI10382，肝弹性6758／速度6758／黏度6757／频散6757／衰减6758／声速6764。不是最终双菌群／协变量交集N。CGM9402人可用日，Y按模块窗口在W14生成。本轮未冻结模型最少天数、主次预算，未训练关联模型，未冒称制造商QC或团队人工复核完成。

已知事项：当前源无设备逐条QC标记；VAT31基线零值保留标记；非选定体脂百分比实际fraction尺度；官方旧肝均值多数差异兼容两位小数舍入，基线各项53／28／40／71人差异超过此幅度，上游原因未确定。按用户已选明细等权均值构建，全部保留对照。

本地报告：[W10完整报告](../../reports/{stem}.md)。聚合缓存：[observed_aggregate.json](observed_aggregate.json)。
''';(A/'MEMO.md').write_text(memo)
validation={'passed':True,'provenance':'manually_observed_TRE_aggregate_outputs_and_actual_exit_UI; local validator does not independently access TRE individual data','actual_exit_codes':{k:v for k,v in x['actual_exits'].items() if k!='02_initial'},'initial_failed_exit_retained':1,'independent_groups_recomputed':84962,'registry_rows':84962,'measurement_QC_rows':192434,'static_outcomes':9,'liver_outcomes':6,'binary_mirrors':3,'build_hashes_verified':8,'per_outcome_eligibility':True,'CGM_recipe_registered':True,'CGM_person_values_materialized':False,'same_visit_without_timestamp_admission':True,'manufacturer_QC_claimed':False,'primary_protocol_frozen':False,'raw_individual_exported':False,'team_human_review_claimed':False,'evidence':'audits/w10_outcome_qc_2026-10-28/07_final_delivery_exit0.png'};(A/'delivery_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
marker='2026-10-28 W10结局字段与基本QC';addition=f'''\n\n## {marker}

已通过当前HPP服务器pheno构建并独立复核9静态结局长表：84962条人—访视—结局记录，192434条测量QC；每个字段独立准入。基线本结局基本QC后：总脂重量6595、VAT体积6513、BMI10382；六肝明细均值分别6758／6758／6757／6757／6758／6764（弹性／速度／黏度／频散／衰减／声速）。均非最终双菌群／协变量分析N。Body／肝同人同cohort同stage，不以timestamp或日期差准入；肝六项各自有效测量等权算术均值，最新用户规则已同步TRE。

总脂原字段body_comp_total_fat_mass单位g；VAT total_scan_vat_volume单位cm³；BMI kg/m²。VAT基线31个零值保留标记，不解释为完全无内脏脂肪。体脂比例非选定列实际0–1，与字典%不同；8989条质量比复核误差<5e-10。BMI12925条公式对照误差0。旧肝汇总多数差异兼容两位小数舍入，但基线各53／28／40／71人超出±0.005，上游原因未给；不混用旧汇总，统一按用户明细均值。当前源无逐条设备／制造商QC标记，完成的是字段、单位和基本数值QC，不冒称完整仪器QC。

CGM基线9402人有合格日；W10登记模块共同日等权日均公式和可用日接口，W14按模块窗口生成Y，不用整段均值顶替。正式最少天数、主次预算和最终N未冻结。01／02b／03–07实际exit0；初版02索引恢复问题exit1保留。84962组原始列独立重算、8产物hash及3二进制副本验收通过；无关联模型、无个体导出、无新Astra／团队人工复核声明。

代码：`{P}/scripts/w10_outcome_qc_20261028/`；Result：`{P}/data_qc/w10_outcome_qc_20261028/`；MEMO：`{P}/memos/w10_outcome_qc_20261028/MEMO.md`。共享表`P/analysis/shared/outcome_registry_TRE_ONLY.parquet`，字典`P/config/outcome_dictionary.yaml`，P同上述项目根。本地[完整W10报告](reports/{stem}.html)、[MEMO](audits/w10_outcome_qc_2026-10-28/MEMO.md)、[聚合缓存](audits/w10_outcome_qc_2026-10-28/observed_aggregate.json)。下一步W11，菌群W12→W13独立后汇合W14。
'''
records=[]
for name in ['HPP_DATA_BACKGROUND.md','HPP_SERVER_BACKGROUND.md']:
 p=ROOT/name;b=p.read_bytes();assert marker.encode() not in b;snap=A/'before_w10_update'/name;snap.parent.mkdir(parents=True,exist_ok=True);assert not snap.exists();snap.write_bytes(b);p.write_bytes(b+addition.encode());assert p.read_bytes().startswith(b);records.append({'path':name,'snapshot':str(snap.relative_to(ROOT)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'append_only':True,'after_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(A/'background_append_manifest.json').write_text(json.dumps({'date':'2026-10-28','records':records},indent=2))
for name in ['HPP_LOCAL_VERIFIED_FACTS.md','HPP_PROJECT_BACKGROUND.md','reports/HPP_CURRENT_PLAN.md']:
 p=ROOT/name;t=addition
 if name.startswith('reports/'):t=t.replace('(reports/','(').replace('(audits/','(../audits/')
 assert marker not in p.read_text();p.write_text(p.read_text()+t)
progress=D/'workpackage_progress.json';p=json.loads(progress.read_text());p['tasks']['W10']={'checked':True,'short_status':'字段／单位／独立基本QC和接口完成；仪器QC限制与最终协议待后续','status_text':'2026-10-28在TRE构建9静态结局并从原始列独立全量重算84962组，3副本／8产物hash通过；01、02b、03–07实际exit0，初版02失败保留。基线总脂6595、VAT6513、BMI10382，肝六项6758/6758/6757/6757/6758/6764；不是最终双菌群N。各域same-visit、独立有效标记；CGM仅登记共同日等权均值接口，W14重算模块Y。当前源缺设备逐条QC标记；VAT零值／体脂fraction尺度／旧肝汇总差异已记，不冒称完整仪器QC或人工复核完成。','what_ran':'01 schema／字典；02初版键索引错误exit1保留，02b修复；03精确字段诊断；04各结局独立构建；05全原始列独立均值与计数／hash复核；06旧肝汇总比较；07舍入幅度／最终交付与备忘录。','code_location':f'服务器 `{P}/scripts/w10_outcome_qc_20261028/`；本地自编源及[代码功能MEMO](../audits/w10_outcome_qc_2026-10-28/MEMO.md)。','result_location':f'服务器 `{P}/data_qc/w10_outcome_qc_20261028/`；共享表 `P/analysis/shared/outcome_registry_TRE_ONLY.parquet`，字典 `P/config/outcome_dictionary.yaml`，远程备忘录 `P/memos/w10_outcome_qc_20261028/MEMO.md`，P同代码项目根。本地[字段与筛选人数]({stem}.html)、[聚合缓存](../audits/w10_outcome_qc_2026-10-28/observed_aggregate.json)。'}
p['tasks']['W03']['status_text']+=' 2026-10-28最新same-visit及肝六项均值规则已由W10同步独立版本并实际执行，见W10当前状态；前述待同步为历史状态。';p['evidence']='audits/w10_outcome_qc_2026-10-28/MEMO.md';p['latest_run_id']='w10_outcome_qc_20261028';p['latest_local_workpackage']='W10';p['latest_local_run_id']=p['latest_run_id'];p['next_recommended_workpackage']='W11';p['next_parallel_workpackages']=['W12','W13'];progress.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
q=D/'execution_update.md';q.write_text(q.read_text()+f'\n\n### {marker}\n\nW10字段与基本QC、独立复核交付已完成，9静态结局／6肝均值各自准入；基线有效数见报告，不是最终菌群模型N。CGM人级Y留W14按模块共同日生成，原单位／无timestamp准入、设备QC缺口和旧汇总差异均有记录。\n\n[本轮结果及筛选人数]({stem}.html) · [代码／Result／MEMO](../audits/w10_outcome_qc_2026-10-28/MEMO.md)。\n')
print('Local W10 report, aggregate cache, memo, progress and append-only backgrounds saved.')
