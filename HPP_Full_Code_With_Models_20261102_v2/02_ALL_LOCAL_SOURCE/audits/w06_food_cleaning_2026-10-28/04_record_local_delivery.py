# -*- coding: utf-8 -*-
"""Record manually transcribed TRE aggregate evidence, never download individual records."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[2];A=ROOT/'audits/w06_food_cleaning_2026-10-28'
P='/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021';L='/home/ec2-user/multisystem_runtime_20261021/w06_food_cleaning_20261028'
h=lambda b:hashlib.sha256(b).hexdigest()
r={'date':'2026-10-28','provenance':'Manually transcribed aggregate terminal output via authorized HPP browser UI; no participant rows exported','actual_exit_codes':{'01':0,'02':0,'03':0},'processed_rows':2640959,'raw_rows_scanned':6485022,'index_rows':149912,'clean_event_rows':2420902,'event_rows_not_in_primary_view':220057,'day_qc_rows':164625,'all_source':{'event_days':164625,'indexed_event_days':149912,'event_days_without_index':14713,'indexed_event_participants':9531,'indexed_event_participant_visits':12085,'energy_days':145325,'fiber_days':76977,'index_only_days':0,'index_food_count_mismatch_days':0},'baseline_10k':{'event_days':125374,'event_participants':9737,'indexed_event_days':116906,'indexed_event_participants':9087,'indexed_ge7_DIAGNOSTIC':8849,'energy_days':113517,'energy_participants':9065,'energy_ge7_DIAGNOSTIC':8649,'fiber_days':60574,'fiber_participants':8677,'fiber_ge7_DIAGNOSTIC':4662,'event_days_without_index':8468,'index_food_count_mismatch_days':0},'signatures':{'exact_repeated_groups':0,'same_food_time_repeated_groups':826,'same_food_time_flagged_rows':1659,'rows_deduplicated':0,'processed_unique_raw_shared_signature_matches':2640959,'processed_rows_without_raw_match':0,'processed_rows_with_multiple_raw_matches':0},'clock':{'all_processed_rows':2640959,'baseline_processed_rows':2059935,'all_date_vs_local_conflicts':1066,'baseline_date_vs_local_conflicts':653,'date_vs_aware_conflicts':0,'missing_local':0,'missing_aware':0,'all_local_equals_aware_wall':2626229,'all_local_equals_UTC_naive':833,'all_local_other_than_both':13897,'conflict_local_equals_UTC_naive':34,'conflict_local_equals_wall':0,'conflict_day_offset_minus1':1039,'conflict_day_offset_plus1':27,'conflict_seconds_top':[[-25200,585],[-36000,145],[-3600,95],[-7200,47],[-28800,41],[-21600,39],[21600,10],[18000,8],[12600,4],[-10800,3],[14400,3],[3600,2]]},'verification':{'passed':True,'source_examples':12,'all_seven_nutrients':True,'event_day_conservation':True,'binary_mirrors':5,'source_metadata_unchanged':True,'outcomes_read':False,'human_team_review_done':False},'remote':{'code':P+'/scripts/w06_food_cleaning_20261028','result':P+'/data_qc/w06_food_cleaning_20261028','memo':P+'/memos/w06_food_cleaning_20261028/MEMO.md','runtime':L},'unresolved':['Source tables do not expose an auditable event lifecycle ID/edit/delete/backfill history.','Food logging timestamp is not independently validated ingestion time; 1066 local calendar conflicts retain unknown causes.','No global minimum record days, complete-intake threshold, or formal dietary exposure design frozen.','NOVA mapping and dietary features remain W07/W08.']}
assert r['processed_rows']==r['clean_event_rows']+r['event_rows_not_in_primary_view']
assert r['all_source']['event_days']==r['all_source']['indexed_event_days']+r['all_source']['event_days_without_index']
assert r['baseline_10k']['event_days']==r['baseline_10k']['indexed_event_days']+r['baseline_10k']['event_days_without_index']
assert r['clock']['conflict_day_offset_minus1']+r['clock']['conflict_day_offset_plus1']==1066
assert r['clock']['all_local_equals_aware_wall']+r['clock']['all_local_equals_UTC_naive']+r['clock']['all_local_other_than_both']==r['processed_rows']
(A/'observed_aggregate.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
report=f'''# W06：食物事件与记录日清洗——执行结果

日期：2026-10-28。**清洗及独立技术验收完成；餐级进食时间语义仍未确认。** 原始数据只读，个体表留在TRE；本页仅有经屏幕核验的聚合结果。未读取血糖、体成分、肝脏等结局，未训练模型。

## 1. 这一步到底做了什么

处理事件表是研究用的摄入来源。raw表只用于核对记录关系，既没有并入摄入量，也没有把其中的undefined访视补成基线。按已经确认的规则：只有**有效人／队列／访视／日期、有处理事件、并匹配日索引**的日期进入主食物视图。

营养素保留原值，并新增`*_clean`列：负值、非有限数和缺失保持NA；已确认的事件摄入量乘数为1。某项营养素缺失只影响相应指标，不能把整条记录所有成分一起当0。未按BMI、血糖或关联显著性筛人。

主程序完整扫描处理事件**2,640,959行**、raw事件**6,485,022行**和日索引**149,912行**。保留原始行号，生成保留／排除谱系，没有合并或自动去重。

## 2. 人数与日期：基线10k队列

下表全部是**饮食自身可用性**，还没有与口腔、肠道、CGM、body或liver相交，不是最终模型人数。基线只有一个本次指定访视，因此这里的人数对应独立参与者。

|视图／条件|至少1个相应日期的人数|相应日期数|至少7个相应日期的人数，仅诊断|
|---|---:|---:|---:|
|处理表中有食物事件|9,737|125,374|不用于本次准入|
|有事件且匹配日索引|9,087|116,906|8,849|
|在上一视图内，所有事件能量有效且当日总能量>0|9,065|113,517|8,649|
|在能量可计算日内，所有事件纤维也有效|8,677|60,574|4,662|

从125,374个事件日期中排除**8,468日（{8468/125374*100:.2f}%）**，原因是缺少匹配日索引；650人没有剩余索引匹配日。不是把有一个坏日的人整人剔除。

例子：一个人原有14个事件日期，其中2日缺索引，留下12日；若其中3日有缺失纤维，则能量最多使用那12日内能量合格的日期，纤维只能使用纤维也合格的日期。不同候选的有效日数各自计算。

**≥7日尚未冻结为全项目准入标准。** W06没有因为候选不同而强制所有分析使用同一交集，也没有证明这些日期记录了全天全部摄入。

## 3. 全来源规模与守恒

全来源包含多个访视，人数和人—访视数必须分开。

|项目|结果|
|处理事件总行数|2,640,959|
|进入主事件视图|2,420,902|
|未进入主事件视图|220,057，具体处置保留于服务器逐行谱系和排除表|
|事件日期键总数／日QC表行数|164,625|
|有事件且匹配日索引|149,912日；9,531人；12,085个人—访视|
|有事件但无日索引|14,713日，占事件日期8.94%|
|只有索引而没有处理事件|0日|
|索引food_items与事件条数不一致|0个索引匹配事件日|
|能量可计算日|145,325|
|纤维也可计算日|76,977|

独立复核逐批验证：2,640,959个源行号连续、主事件行号严格递增且不扩行、每个日期事件条数与日表一致，七种营养素的有效值／缺失数及事件→日汇总守恒。

## 4. 是不是同样的食物记录重复了

**没有发现所有源字段完全相同的处理事件。**

|核查|结果|处理|
|整行相同的签名组|0组|无需按整行重复删除|
|同人、队列、访视、源日期、带时区时间和FoodID相同|826组，涉及1,659行|作为候选重复标记，全部保留|
|实际去重／合并|0行|不根据“看起来像同一餐”删除|

上述826组在其他源字段上有差异，否则就会构成整行重复。仅凭同FoodID同时间，不能判为重复摄入，也不能判为用户编辑。原字段和源行号均保留，后续可以定位。

另一个比较清楚的发现是：**每条处理事件都能在raw表中找到唯一相同的共享字段签名：2,640,959条匹配，0条缺匹配，0条多匹配。** 这个比较忽略两表不一致的访视／logging_day字段，不是用raw的undefined阶段重新建立研究访视，也不将两表合并计摄入。

raw比处理表多出的记录**不能直接解释为删除或编辑历史**。当前暴露的标量字段没有可靠事件生命周期ID及编辑／删除／补记链，不能从表大小差还原过程。

## 5. 历史1066条日期冲突：现在知道什么

字典把`collection_date`解释为食物记录时间；`local_timestamp`为无时区的食物记录本地时间；`collection_timestamp`明确带有`Asia/Jerusalem`时区。**它们的说明尚不足以把记录时刻直接认定为实际进食时刻。**

|对照|全处理表|基线10k|
|---|---:|---:|
|处理事件行数|2,640,959|2,059,935|
|源日期与带时区时间的以色列日历日期冲突|0|0|
|源日期与无时区local日期冲突|1,066（{1066/2640959*100:.4f}%）|653（{653/2059935*100:.4f}%）|
|缺少local_timestamp|0|0|
|缺少collection_timestamp|0|0|

全源1,066条冲突中：local日历日期早1日的**1,039条**，晚1日的**27条**。但“跨日”只是观察到的结果，不能自动解释为补记或跨午夜进餐。

- 仅**34条**冲突记录的local时刻恰好等于带时区时间转换后的UTC无时区时刻。
- 冲突中的时差包括−7小时585条、−10小时145条、−1小时95条、−2小时47条等。因此，不能把所有冲突都归因于统一的UTC／以色列时区差或夏令时。
- 全源local与以色列墙上时间完全一致2,626,229条；等于UTC无时区833条；两者都不相同13,897条。多数时间差不会造成日历日期冲突。

**当前规则：**保留与带时区时间日历日期一致的源日期；local冲突只标记，不作为整日排除标准，不以local时间重分日期。日级食物视图可以交给W08/W09继续按真实日期对齐；正式逐餐／分钟反应模型仍需额外确认实际进食时刻与记录延迟。

## 6. 到服务器哪里找代码和Result

项目根P：`{P}`。

|自编代码|功能|实际状态|
|---|---|---|
|`P/scripts/w06_food_cleaning_20261028/01_inspect_source_contract.py`|schema、字典、资源、来源元信息|exit 0|
|`P/scripts/w06_food_cleaning_20261028/02_build_food_views.py`|完整分批扫描、两层事件签名、时钟诊断、事件／日视图与守恒|exit 0|
|`P/scripts/w06_food_cleaning_20261028/03_independent_verify.py`|独立读回行谱系、七种营养素、12条源记录、5副本hash与行数|exit 0|

|产物|位置／用途|
|---|---|
|主事件视图|`P/analysis/shared/food_events_clean_TRE_ONLY.parquet`：2,420,902行；只含准入事件，保留源值、清洗值及QC标记|
|全源逐行谱系|`P/analysis/shared/food_event_lineage_TRE_ONLY.parquet`：2,640,959行；每行保留／排除和重复／时间标记|
|日QC表|`P/analysis/shared/food_day_qc_TRE_ONLY.parquet`：164,625行；包含未准入日期，使用时必须筛primary_day及适用暴露条件|
|规则配置|`P/config/food_day_rules.yaml`；JSON兼容YAML，明确无全局最少天数|
|排除汇总|`P/data_qc/food_exclusions.csv`|
|Result|`P/data_qc/w06_food_cleaning_20261028/`：源契约、w06_summary_TRE_ONLY.json、binary_mirror_verification.json、verification.json|
|远程备忘录|`P/memos/w06_food_cleaning_20261028/MEMO.md`，包含代码用途、计数和独立复核|
|运行盘／日志|`{L}`：01.log、02.log、03.log及各实际退出码|

二进制先写服务器运行盘，再逐字节复制到V2云盘；5个副本的字节数、SHA和Parquet行数已核对。源表大小、mtime和行数未变。峰值RSS约1.08 GiB。

本地保存[自编代码与备忘录](../audits/w06_food_cleaning_2026-10-28/MEMO.md)、[聚合缓存](../audits/w06_food_cleaning_2026-10-28/observed_aggregate.json)、[独立复核与退出码截图](../audits/w06_food_cleaning_2026-10-28/03_independent_and_exit_codes.png)。未导出个体数据。

## 7. 剩下什么、下一步做什么

1. **W06已完成数据清洗与独立技术验收**；同事乙的人工复核仍待进行。
2. **1066条local冲突的具体行为原因尚未还原**；原始表没有提供足够编辑／补记历史，保留诊断与餐级分析的前置条件。
3. **W07**继续内部FoodID／NOVA crosswalk和未知分类的覆盖评估，不新增外部映射源。
4. **W08**据候选各自有效日生成日级与窗口视图；**W09**单独处理CGM point QC和共同日期。
5. 能量范围、记录跨度、最少天数及主暴露／检验预算仍需按研究问题冻结，不能按结局效果选择。

W06没有生成新的饮食—菌群—代谢关联结论，也没有更改正式V3的科学问题和模型预算。
'''
(ROOT/'reports/HPP_W06_Food_Events_Day_QC_2026-10-28.md').write_text(report)
memo=f'''# W06代码、Result和验收备忘录

2026-10-28：清洗与独立技术验收已完成。01／02／03实际退出码均0，证据见本目录`03_independent_and_exit_codes.png`。

|代码|功能|远程文件|
|---|---|---|
|01_inspect_source_contract.py|三张源表schema／字典及环境|{P}/scripts/w06_food_cleaning_20261028/01_inspect_source_contract.py|
|02_build_food_views.py|全源分批处理、raw签名对照、时钟诊断、事件谱系／日QC、规则及副本核验|同目录02_build_food_views.py|
|03_independent_verify.py|重新读取全部谱系和事件、七种营养素、12例源记录及副本哈希|同目录03_independent_verify.py|
|w06a.py／w06b.py／w06c.py|仅封装本次自编代码，解码／源码SHA／落盘／运行|传输封装；不依赖TemporaryFiles中的上传副本|
|04_record_local_delivery.py|人工转录已见聚合证据，写本地缓存与报告|只在本地运行，不读TRE个体文件|

Result：`{P}/data_qc/w06_food_cleaning_20261028/`。

日志及实际退出码：`{L}`。02启动器PID5045；03启动器PID13778（不是长期运行服务器标识）。

远程备忘录：`{P}/memos/w06_food_cleaning_20261028/MEMO.md`。

主事件2,420,902行；谱系2,640,959行；日QC164,625行；副本5项。完整结果见[报告](../../reports/HPP_W06_Food_Events_Day_QC_2026-10-28.md)。

## 边界

- 日期准入遵循用户决定：有事件且匹配日索引。
- raw仅诊断，不并入摄入或补访视；所有processed共享字段签名在raw唯一匹配。
- 整行重复0组，同食物同时间826组标记保留，无自动去重。
- 1066条local日期冲突仅标记；与带时区以色列日期冲突0。记录时刻不冒称实际进食时刻，具体冲突成因未还原。
- ≥7日仅诊断，无全局准入冻结；完整摄入未被证明。
- 无结局读取、无新增关联模型、无人类同事复核完成的主张。

## 操作记录与恢复

本轮剪贴板显示Pasted但终端无法粘贴，上传的自编w06a.py在TemporaryFiles可见但未找到映射到EC2的文件路径。最终采用自编代码十六进制文本＋显式括号按键，远程解码后核SHA，pheno启动并保存真实退出状态。仅上传／输入自编源码，无用户原始数据外传。

远程视频可能延迟；一次尚未执行的重复tail命令已清除，不影响数据。原始数据保持只读。三个科研脚本无失败重跑；界面传输试验不当作数据分析结果。
'''
(A/'MEMO.md').write_text(memo)
# Preserve dated historical backgrounds exactly, append once with chain evidence.
marker='2026-10-28 W06食物事件与记录日清洗'
records=[];before=A/'before_w06_update';before.mkdir(exist_ok=True)
append=f'''\n\n## {marker}\n\n真实TRE执行01／02／03均exit0，独立技术验收通过；人工团队复核仍待进行。处理事件2640959行、raw诊断6485022行，主事件2420902行、日QC164625行。全源149912个索引匹配事件日，无索引14713日；基线9087人116906日，能量可用9065人113517日，纤维可用8677人60574日；≥7日分别8849／8649／4662，仅诊断，非最终模型N。\n\n整行重复0组；同食物同时间826组1659行，只标记无自动去重。所有processed在raw中有唯一共享字段签名匹配，不赋raw访视。local日期冲突全源1066／基线653；源日期与带Asia/Jerusalem时间日期冲突0，冲突只有34条符合UTC无时区模式，不能统一归因于时区或补记。日志时间尚未验证为实际进食时间，日级源日期可继续使用，逐餐精细模型前置条件未解除。\n\nCode：`{P}/scripts/w06_food_cleaning_20261028/`。Result：`{P}/data_qc/w06_food_cleaning_20261028/`。远程MEMO：`{P}/memos/w06_food_cleaning_20261028/MEMO.md`。共享产物：`P/analysis/shared/food_events_clean_TRE_ONLY.parquet`、`food_event_lineage_TRE_ONLY.parquet`、`food_day_qc_TRE_ONLY.parquet`；规则`P/config/food_day_rules.yaml`，排除表`P/data_qc/food_exclusions.csv`。\n\n本地[W06报告](reports/HPP_W06_Food_Events_Day_QC_2026-10-28.md)、[代码MEMO](audits/w06_food_cleaning_2026-10-28/MEMO.md)、[聚合缓存](audits/w06_food_cleaning_2026-10-28/observed_aggregate.json)。\n'''
for name in ['HPP_DATA_BACKGROUND.md','HPP_SERVER_BACKGROUND.md']:
 p=ROOT/name;old=p.read_bytes();assert marker.encode() not in old
 snap=before/name;snap.write_bytes(old);p.write_bytes(old+append.encode());new=p.read_bytes();assert new.startswith(old)
 records.append({'path':name,'snapshot':str(snap.relative_to(ROOT)),'bytes':len(old),'sha256':h(old),'append_only':True,'after_sha256':h(new)})
(A/'background_append_manifest.json').write_text(json.dumps({'date':'2026-10-28','records':records},indent=2))
for name in ['HPP_LOCAL_VERIFIED_FACTS.md','HPP_PROJECT_BACKGROUND.md','reports/HPP_CURRENT_PLAN.md']:
 p=ROOT/name;s=p.read_text();assert marker not in s;p.write_text(s+append)
progress=ROOT/'outputs/researchwrite/hpp_multisystem_20261021/team_checklist_20261026/workpackage_progress.json';x=json.loads(progress.read_text());x['evidence']='audits/w06_food_cleaning_2026-10-28/MEMO.md';w=x['tasks']['W06'];w.update(checked=True,short_status='清洗及独立技术验收完成；餐级时间语义／团队人工复核待确认',status_text='2026-10-28三脚本均实际exit0。主事件2420902行、全源谱系2640959行、日QC164625行；七营养素／12例源行／5副本及守恒复验通过。基线索引后9087人116906日；能量9065人113517日、纤维8677人60574日。无整行重复，826同食物时间组只标记；local日期冲突1066原因未还原，与带时区源日期冲突0。未读结局，未冻结≥7日；团队人工复核待进行。',what_ran='01_inspect_source_contract.py；02_build_food_views.py；03_independent_verify.py。来源契约、完整事件／raw签名与时钟诊断、主事件／谱系／日QC、独立读回及12源行／5副本校验；均exit0。',result_location=f'服务器 `{P}/data_qc/w06_food_cleaning_20261028/`（summary／verification）；共享表在 `P/analysis/shared/`，规则 `P/config/food_day_rules.yaml`、排除表 `P/data_qc/food_exclusions.csv`；远程 `P/memos/w06_food_cleaning_20261028/MEMO.md`。本地[W06报告](HPP_W06_Food_Events_Day_QC_2026-10-28.html)、[聚合与MEMO](../audits/w06_food_cleaning_2026-10-28/MEMO.md)。P同代码项目根。');progress.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
(A/'delivery_validation.json').write_text(json.dumps({'passed':True,'actual_exit_codes':r['actual_exit_codes'],'processed_rows':2640959,'clean_rows':2420902,'day_rows':164625,'source_examples':12,'mirrors':5,'outcomes_read':False,'human_team_review_done':False,'aggregate_arithmetic_verified':True,'local_is_aggregate_transcription':True},indent=2))
print('W06 local report, aggregate cache, memo, append-only backgrounds and checked progress updated.')
