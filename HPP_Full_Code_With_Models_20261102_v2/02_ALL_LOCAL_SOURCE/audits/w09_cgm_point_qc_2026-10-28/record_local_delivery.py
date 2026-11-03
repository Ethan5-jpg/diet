# -*- coding: utf-8 -*-
"""Run only after aggregate TRE results and actual exits have been independently observed."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[2];A=Path(__file__).resolve().parent;D=ROOT/'outputs/researchwrite/hpp_multisystem_20261021/team_checklist_20261026';P='/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021';L='/home/ec2-user/multisystem_runtime_20261021/w09_cgm_point_qc_20261028';stem='HPP_W09_CGM_QC_W08_Common_Days_2026-10-28';marker='2026-10-28 W09点级QC与W08共同日补齐'
r=json.loads((A/'observed_aggregate.json').read_text());assert r['W09_verified'] and r['W08_verified'] and all(v==0 for v in r['actual_exit_codes'].values());assert r['all_fragments_verified']==9497
base=r['W09_counts']['baseline_10k_selected'];common=r['W08_counts']['baseline_10k'];rows=[]
for label,x in [('CGM首末观测间日历跨度（含空白日）',base['observed_calendar']),('CGM日级QC合格',base['qualified_CGM']),('索引匹配食物事件日 ∩ 合格CGM日',common['indexed_common']),('其中饮食能量完整',common['energy_common']),('其中纤维密度完整',common['fiber_common']),('其中UPF未知分类能量为0',common['UPF_zeroU_common']),('其中UPF所有事件均已分类',common['UPF_strict_common'])]:
 rows.append(f"|{label}|{x['people']:,}|{x['days']:,}|{x['visits_ge7_DIAGNOSTIC']:,}|{x['median_days']:g}|")
table='|筛选层级（10k基线）|至少1日的人数|日数|至少7日人数：仅诊断|每人对应日数中位数|\n|---|---:|---:|---:|---:|\n'+'\n'.join(rows)
flow_names={'at_least_one_valid_point':'当天至少1个有效血糖点','coverage_at_least_80pct':'日覆盖达到80%','max_uncovered_gap_at_most_2h':'最长未覆盖段不超过2小时','no_conflicting_timestamp':'当天无同时间不同值冲突','regular_15min_cadence':'15分钟采样节奏可确认','connection_no_missing_timestamp':'所在连接无缺失时间戳'}
flow='|步骤|该步排除的日数|剩余日数|仍有至少1日人数|\n|---|---:|---:|---:|\n'+'\n'.join(f"|{flow_names.get(x['step'],x['step'])}|{x['removed_days_at_this_step']:,}|{x['remaining_days']:,}|{x['remaining_people']:,}|" for x in r['CGM_day_flow'])
report=f'''# W09 CGM点级核查与W08真实共同日结果

2026-10-28。**W09全分片点级核查、日表重建及技术验收完成；W08真实共同日及窗口已补齐并独立复核。** 原饮食自身窗口保持不变。代码实际退出码见文末，本轮未拟合预测、交互或因果模型。

## 1. 筛选后到底多少人

{table}

这些是**饮食—CGM模块的当前可用性**，尚未交入口腔／肠道QC、协变量或最终模型准入；不能说是最终菌群模型N。没有与body／liver取交集，也不会以此删掉非CGM域的人。

“至少7日”仍是质量诊断，没有冻结成全项目门槛。表中每种暴露使用自己的有效日集；“能量完整”之下的纤维与UPF是并列分支。日数是人—访视—真实日期计数；第一行还保留首末观测之间完全没有有效点的空白日期，用于揭示缺段，不代表这些日有合格观测。“共同日数中位数”是先数每人有多少共同日期，再对人数取中位数，不是把时间差取平均。

例如：10月1–14日有索引匹配食物事件、10月3–15日有CGM，若10月8日CGM覆盖不足，则名义共同日期12日，合格共同日期11日；纤维若其中2日缺失，只用另外9日算纤维。不会删除整个人，也不会用14日饮食均值冒充这9日均值。

完整能量共同窗口中，所有日都有精确UPF点值的基线参与者为**133人**，没有另加≥7日条件；原来完整饮食自身窗口的45人是不同的窗口，不能据此声称食品分类改善。当前UPF精确点值子集虽然覆盖6779人，但每人精确日数中位仅3日；纤维为6日。这是后续暴露验收和测量精度要处理的实际问题，不能只报至少1日人数。

本次CGM基线索引有9410个连接，归属9405人；5人同访视有多个连接。提供的基线选定序列可见日历跨度为1–13日，中位13日。计划两周佩戴与交付点序列可见13日是不同概念，不能据此补出第14日。

## 2. CGM日级QC具体怎样筛

单位和有效范围来自**服务器本次挂载的glucose数据字典**：mg/dL，40–500，15分钟采样。时间戳为Asia/Jerusalem。提供的HPP点序列可能已经经过上游QC，本轮不能恢复上游删除的点。HPP公开说明的两周佩戴、15分钟记录是计划方案，实际可用日按本轮观测重算。[HPP CGM官方说明](https://knowledgebase.pheno.ai/datasets/017-cgm.html)

本次执行口径：有限且字典范围内的点；相同时间和值的重复保留谱系后折叠；同时间不同值标记冲突并隔离该日；不跨缺口插值。一天按以色列真实日界，夏令时日可能23／25小时。

覆盖率定义为**实际观测点各自±7.5分钟采样时段的并集长度÷真实日长**，重叠不重复累计；这是观测覆盖计算，不给缺口捏造血糖值。工作日级候选要求覆盖≥80%、最长未覆盖段≤2小时、采样节奏可确认且无时间冲突／缺失。70/80/90%与1/2/4小时组合敏感性已保留；最终模型阈值与最少天数仍须协议冻结。

{flow}

上表按顺序归因，排除数可以相加。某日可能同时覆盖不足和缺口过长，不能把两个边际计数重复相加。

## 3. 点级、连接、夏令时和旧日表

```json
{json.dumps(r['point_accounting'],ensure_ascii=False,indent=2)}
```

共9497个片全部读取，处理状态、源hash、逐片点数守恒和单位可追溯。多连接的处理结果、日长与旧表比较如下，均为聚合数，不含参与者ID／日期：

```json
{json.dumps({k:r[k] for k in ['multi_connection_aggregate','day_lengths','legacy_reconciliation','precision_check']},ensure_ascii=False,indent=2)}
```

连接选择预先定义为：合格CGM日与W06能量完整的索引匹配食物日共同数最多→真实开始最早→稳定连接ID。选择不使用血糖高低、病例或饮食关联效应；不同连接没有自动拼成一个窗口。多连接存在不证明换传感器，原因仍需临床记录支持。

固定源序号的曲线文件留TRE，曲线的生成与自动数值核验、人工目视复核分别记录，不能互相冒充。随机遮去20%点的均值精度检查只模拟随机缺失，不证明实际缺失随机，也不支持因果结论。

旧日表的104845个日期全部能从提供点重建，104845个两侧均值均可比较，误差仅浮点舍入。重建另外出现557个旧表未列的日期：2日无点而被QC排除；555日有有效点且通过QC，其中553日有96点、另两日分别93和94点。为什么上游旧汇总表未列这555日，当前元信息没有给原因；不能编造成设备更换或没有观测。

**这555日没有进入本轮基线饮食共同日。** 只限旧日表存在的日期时，基线饮食共同日仍为8288人87856日，能量共同仍为8235人85413日，纤维共同仍为7613人45695日，相关≥7日人数也不变。因此，它不影响当前W08基线人数。新增`legacy_daily_presence_selected_TRE_ONLY.parquet`供后续CGM单独分析／其他窗口作预设敏感性；差异及验证在`legacy_difference_audit.json`，代码11。

## 4. W08怎样补齐

新增`food_cgm_common_daily_TRE_ONLY.parquet`和`food_cgm_common_windows_TRE_ONLY.parquet`。按唯一同人／队列／访视／真实日历日与合格CGM表一对一连接，然后重算各暴露的有效日数、分子、同日能量分母、每天等权均值、累计比和UPF上下界。UPF未知不归成非UPF、纤维缺失不填0。

```json
{json.dumps(r['window_recalculation'],ensure_ascii=False,indent=2)}
```

原`food_windows_TRE_ONLY.parquet`的hash复验未变，因此非CGM结局保留自己的饮食窗口。共同日独立核验包括完整键集合、日值、每个窗口公式及副本hash，不只验证文件存在。

## 5. 还需要掌握的问题

- **食物App的记录日不等于独立验证过的进食时刻。** 本轮支持日历日分析；逐餐／分钟级反应模型仍需食物时钟证据。
- **NOVA未知分类仍存在。** 精确UPF子集与完整共同窗口UPF界限是两个不同对象；本轮CGM QC不能补回食物加工等级。
- **≥7日、主要饮食聚合方式与主结局未冻结。** 不按模型显著性挑阈值；QC敏感性网格已备好。
- **旧日表遗漏555个合格点序列日期的上游原因仍不清楚。** 已留来源标记与敏感性入口；它们不影响当前基线饮食共同日。
- **尚无最终菌群模型N。** 需W10–W14分别完成各结局／协变量／双部位菌群的same-visit与技术QC。本轮不新增固定菌群采样日差门槛。

## 6. 本轮运行中修正的代码问题

初版03已完成点扫描，但对旧日表误用了`cgm_mean`字段名；实际字段是`cgm_daily_mean`，保护断言因此停止。修正版`03b_build_point_qc_legacy_fix.py`把旧表字段预检提前，重新完成构建，并保存扫描检查点。初版`03.log`与`03_initial_actual_exit_code=1`保留，修正版真实退出看`03b_actual_exit_code`，没有把初版失败当成功。

随后键类型核查确认：饮食date为ISO字符串，初版CGM为Arrow date32；类型不同造成共同日伪0。09仅统一日期的表示方式，不改实际日期，从完整点扫描检查点重建连接选择与日表；旧派生产物及hash归档保留。最终独立验收为04b，原04是修复前的历史验收。

## 7. 代码、结果、备忘录

服务器项目根P：`{P}`。

|位置|用途|
|---|---|
|`P/scripts/w09_cgm_point_qc_20261028/`|01–07源码及后续修复文件；具体版本以源码manifest和执行日志为准|
|`P/data_qc/w09_cgm_point_qc_20261028/`|summary、逐片处理状态、连接决策、旧日表对照、两份独立verification|
|`P/data_qc/cgm_point_connection_QC.parquet`|连接粒度QC及选择状态|
|`P/analysis/shared/cgm_daily_TRE_ONLY.parquet`|唯一连接选择后的CGM日表，保留合格标记|
|`P/analysis/shared/food_cgm_common_daily_TRE_ONLY.parquet`|W08合格真实共同日|
|`P/analysis/shared/food_cgm_common_windows_TRE_ONLY.parquet`|在实际共同日集上重新计算的饮食窗口|
|`P/config/cgm_point_daily_rules_W09_20261028.json`|规则及未冻结项|
|`P/data_qc/w09_cgm_point_qc_20261028/legacy_daily_presence_selected_TRE_ONLY.parquet`|旧日表是否覆盖每个选定连接日期，供敏感性分析|
|`P/data_qc/w09_cgm_point_qc_20261028/legacy_difference_audit.json`|旧日表缺557日的具体结构与基线共同日不受影响的验收|
|`P/memos/w09_cgm_point_qc_20261028/MEMO.md`|逐代码功能、环境、筛选、验证、限制、结果位置|
|`{L}`|日志、PID、实际退出码、运行盘副本|

本地[自编代码与备忘录](../audits/w09_cgm_point_qc_2026-10-28/MEMO.md)、[已观察聚合缓存](../audits/w09_cgm_point_qc_2026-10-28/observed_aggregate.json)。个体数据与源曲线不下载。

```json
{json.dumps(r['actual_exit_codes'],indent=2)}
```

技术验收完成不代表团队另一成员已人工复核，也不代表研究假设被证实。
'''
(ROOT/'reports'/f'{stem}.md').write_text(report)
memo=f'''# W09与W08共同日代码备忘录

报告：[完整结果](../../reports/{stem}.md)。

Code：`{P}/scripts/w09_cgm_point_qc_20261028/`。
Result：`{P}/data_qc/w09_cgm_point_qc_20261028/`。
MEMO：`{P}/memos/w09_cgm_point_qc_20261028/MEMO.md`。
Runtime：`{L}`。

|代码|功能|
|---|---|
|01_probe_cgm_contract.py|字典、表schema、片数、三代表片采样节奏|
|02_probe_index_contract.py|索引键、设备、单位与多连接数量|
|03_build_point_qc.py|初版，全片扫描后旧表字段断言失败，保留日志|
|03b_build_point_qc_legacy_fix.py|生产修正版：旧表实际cgm_daily_mean字段提前预检，全片QC/日表及检查点|
|04_independent_verify_cgm.py|修复日期键之前的独立点级复验，历史保留|
|09_fix_calendar_contract.py|将CGM日键表示统一为W06的ISO字符串，从扫描检查点重建连接选择/日表/对账，归档旧派生产物|
|04b_verify_corrected_calendar.py|最终独立全片hash/守恒/日均/覆盖及ISO日期键验收|
|05_build_w08_common.py|真实共同日与窗口重算、暴露分布与观测期重复性|
|06_independent_verify_common.py|完整共同键、日值、全部窗口公式和副本hash复验|
|07_finalize_and_memo.py|汇总真实exit与逐步筛选、写TRE备忘录|
|08_display_aggregate.py与10_status_and_read.py|只显示聚合人数、真实退出和验证；不导出个体|
|11_legacy_difference_and_delivery.py|核对旧表遗漏日期、保存来源标记与限制旧日期的敏感性人数，附记远程MEMO|

若有修复脚本，保留失败日志，最终执行版本见observed_aggregate及源码manifest，不能把未运行草稿当实际代码。当前本地代码为自编输入，未下载远程个体数据。

实际生产执行顺序：03b→09日期契约修复→04b最终点级复验→05共同日→06独立窗口复验→07备忘→11旧日表差异补核。原03／04是历史版本，不是最终产物验收依据。

本轮直接typeText的只读tail命令曾只收到t，出现command not found；科研源代码均走官方Clipboard、hash校验及实际退出验证。后续不要用原生逐字输入长命令，优先复用已验证的剪贴板路线。

实际状态：W09 verified={r['W09_verified']}，W08 verified={r['W08_verified']}；exit={r['actual_exit_codes']}。
''';(A/'MEMO.md').write_text(memo)
addition=f'''\n\n## {marker}\n\nW09全9497片点级QC与W08真实共同日／窗口已验收；实际exit与独立证据见[聚合缓存](audits/w09_cgm_point_qc_2026-10-28/observed_aggregate.json)。基线CGM合格{base['qualified_CGM']['people']}人／{base['qualified_CGM']['days']}日；索引匹配食物共同日{common['indexed_common']['people']}人／{common['indexed_common']['days']}日；能量共同{common['energy_common']['people']}人／{common['energy_common']['days']}日；纤维共同{common['fiber_common']['people']}人／{common['fiber_common']['days']}日；UPF零未知能量共同{common['UPF_zeroU_common']['people']}人／{common['UPF_zeroU_common']['days']}日。均非最终菌群模型N，≥7日只诊断，未全域取交集。\n\n共享date键统一YYYY-MM-DD字符串，与W06饮食日保持一致；源CGM date32不直接与字符串比较，真实日历日期未改。mg/dL、字典40–500、15min；Israel日历真实日长；覆盖为观测±7.5min时段并集，无血糖插值。80%／2h为执行候选，网格保留；连接选最多能量合格食物共同日→最早起始→稳定ID。原饮食独立窗口hash未变，UPF未知与进食时钟限制未掩盖。\n\nCode：`{P}/scripts/w09_cgm_point_qc_20261028/`。Result：`{P}/data_qc/w09_cgm_point_qc_20261028/`。MEMO：`{P}/memos/w09_cgm_point_qc_20261028/MEMO.md`。共享`cgm_daily_TRE_ONLY.parquet`、`food_cgm_common_daily_TRE_ONLY.parquet`、`food_cgm_common_windows_TRE_ONLY.parquet`在P/analysis/shared/。\n\n[完整报告](reports/{stem}.md) · [本地代码备忘](audits/w09_cgm_point_qc_2026-10-28/MEMO.md)。\n'''
before=A/'before_w09_update';before.mkdir(exist_ok=True);records=[]
for name in ['HPP_DATA_BACKGROUND.md','HPP_SERVER_BACKGROUND.md']:
 p=ROOT/name;b=p.read_bytes();assert marker.encode() not in b;snap=before/name;snap.write_bytes(b);p.write_bytes(b+addition.encode());records.append({'path':name,'snapshot':str(snap.relative_to(ROOT)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'append_only':True,'after_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(A/'background_append_manifest.json').write_text(json.dumps({'date':'2026-10-28','records':records},indent=2))
for name in ['HPP_LOCAL_VERIFIED_FACTS.md','HPP_PROJECT_BACKGROUND.md','reports/HPP_CURRENT_PLAN.md']:
 p=ROOT/name;t=addition
 if name.startswith('reports/'):t=t.replace('(reports/','(').replace('(audits/','(../audits/')
 assert marker not in p.read_text();p.write_text(p.read_text()+t)
x=json.loads((D/'workpackage_progress.json').read_text());x['evidence']='audits/w09_cgm_point_qc_2026-10-28/MEMO.md'
loc={'code_location':f'服务器 `{P}/scripts/w09_cgm_point_qc_20261028/`；本地自编源与[MEMO](../audits/w09_cgm_point_qc_2026-10-28/MEMO.md)。','result_location':f'服务器 `{P}/data_qc/w09_cgm_point_qc_20261028/`；共享表在`P/analysis/shared/`；远程`P/memos/w09_cgm_point_qc_20261028/MEMO.md`。本地[结果与筛选人数]({stem}.html)。P同代码项目根。'}
x['tasks']['W09']=dict(loc,checked=True,short_status='全分片点级QC／日表与独立技术验收完成，候选阈值网格保留',status_text=f"2026-10-28全9497片；单位／采样间隔／点数守恒／DST覆盖／连接选择及旧日表比较已验收，实际exit见报告。基线合格CGM{base['qualified_CGM']['people']}人{base['qualified_CGM']['days']}日。80%/2h执行候选并保留敏感性网格；未冻结模型最少天数或研究假设，团队人工复核未冒称完成。",what_ran='01／02源契约；03全分片QC；04独立hash／守恒／固定片日均与覆盖复算；07汇总备忘；若有修复记录见MEMO，实际退出均已核。')
x['tasks']['W08']=dict(loc,checked=True,short_status='饮食自身及真实CGM共同日／窗口均完成技术验收',status_text=f"2026-10-28在原饮食视图基础补齐真实共同日并重算窗口，独立键／日值／全部窗口公式与副本hash通过。基线共同{common['indexed_common']['people']}人{common['indexed_common']['days']}日；纤维{common['fiber_common']['people']}人{common['fiber_common']['days']}日。原饮食窗口未变，未交菌群／其他结局或冻结主暴露；进食时钟未验证。",what_ran='先前W07/W08的01／02已验收；新增05构建真实共同日和窗口、06独立全键与公式验收、07汇总。')
x['latest_run_id']='w09_cgm_point_qc_20261028';x['latest_local_workpackage']='W09_and_W08_common';x['latest_local_run_id']=x['latest_run_id'];(D/'workpackage_progress.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
p=D/'execution_update.md';p.write_text(p.read_text()+f'\n\n### {marker}\n\nW09全量点级QC及W08真实共同日／窗口技术验收完成。基线合格CGM{base["qualified_CGM"]["people"]}人，共同饮食{common["indexed_common"]["people"]}人，纤维共同{common["fiber_common"]["people"]}人；不是最终菌群模型N，最少天数未全局冻结。\n\n[完整结果与流转]({stem}.html) · [代码／Result／MEMO](../audits/w09_cgm_point_qc_2026-10-28/MEMO.md)。\n')
(A/'delivery_validation.json').write_text(json.dumps({'passed':True,'actual_exit_codes':r['actual_exit_codes'],'W09_verified':True,'W08_real_common_verified':True,'fragments':9497,'independent_fragment_hashes':9497,'raw_individual_exported':False,'primary_protocol_frozen':False},indent=2))
print('W09 / W08 local report and progress recorded; rebuild HTML and checklist next.')
