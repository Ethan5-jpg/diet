# -*- coding: utf-8 -*-
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[2];A=ROOT/'audits/w07_w08_diet_views_2026-10-28';P='/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021';L='/home/ec2-user/multisystem_runtime_20261021/w07_w08_diet_views_20261028'
h=lambda b:hashlib.sha256(b).hexdigest()
r={'date':'2026-10-28','provenance':'Manual transcription of aggregate TRE terminal output; individual features remain in TRE','build_actual_exit':0,'independent_verify_actual_exit':0,'W07_technical_crosswalk_complete':True,'W08_food_only_complete':True,'W08_CGM_common_complete':False,'W08_pending_dependency':'W09 verified CGM point QC and eligible-day table','FoodIDs':7770,'known_NOVA_IDs':1933,'unknown_NOVA_IDs':5837,'multiple_names_per_ID':0,'multiple_categories_per_ID':0,'mapping_version':'legacy_NCDiet_NOVA_643_419192dbf31a_FoodID_W07_20261028','daily_rows':149912,'window_rows':12085,'all_source_participants':9531,'baseline':{'indexed':{'people_ge1':9087,'days':116906,'people_ge7_diagnostic':8849},'energy':{'people_ge1':9065,'days':113517,'people_ge7_diagnostic':8649},'UPF_zero_unknown_energy':{'people_ge1':7970,'days':35196,'people_ge7_diagnostic':1714},'UPF_all_rows_classified':{'people_ge1':7957,'days':35057,'people_ge7_diagnostic':1707},'fiber':{'people_ge1':8677,'days':60574,'people_ge7_diagnostic':4662},'NOVA_energy_coverage_on_energy_eligible_days':0.8819395279080699,'UPF_interval_width_median_pp':6.651987649625077,'UPF_interval_width_p99_pp':60.53859186562093,'exact_point_over_entire_energy_window_people_no_minimum_day_requirement':45},'all_source':{'UPF_zeroU_days':45095,'UPF_zeroU_person_visits':10469,'UPF_zeroU_people':8514,'UPF_zeroU_visits_ge7_diagnostic':2123,'UPF_strict_days':44907,'UPF_strict_person_visits':10450,'UPF_strict_people':8502,'UPF_strict_visits_ge7_diagnostic':2112,'NOVA_energy_coverage':0.8798852317410574,'exact_full_energy_windows_no_minimum_days':84},'baseline_repeatability':{'fiber':{'n_ge7':4662,'odd_even_spearman':0.7196793406521989},'UPF_zeroU':{'n_ge7':1714,'odd_even_spearman':0.5979745527593505},'UPF_strict':{'n_ge7':1707,'odd_even_spearman':0.597785874760113}},'baseline_feature_distribution_rounded':{'energy_daily_kcal':{'min':500.0079,'median':1653.498,'max':4999.0916},'fiber_daily_density':{'zero_days':33,'median':13.247225446813946,'p99':37.39703232462661,'max':99.40450768531622},'fiber_equal_day_window_mean':{'median':13.56078840124702,'p99':30.31298065562956,'max':62.53190677151679},'fiber_window_cumulative_ratio':{'median':13.33504193477001,'p99':29.851803099480467},'UPF_zeroU_daily_pp':{'zero_days':6063,'median':15.191289388761657,'p99':71.53864833848421,'max':100.0},'UPF_zeroU_equal_day_window_mean_pp':{'median':17.795083730291722,'p99':55.68754955759877},'UPF_zeroU_window_cumulative_ratio_pp':{'median':17.907558671767468,'p99':55.478392717124194}},'verification':{'pass':True,'mirrors':7,'admitted_event_rows_independently_reread':2420902,'source_days_independently_verified':3,'CGM_common_recomputation_synthetic_pass':True,'outcomes_read':False,'human_team_review_done':False},'peak_RSS_KiB':342904,'global_minimum_days_frozen':False,'primary_aggregation_frozen':False,'formal_exposures_frozen':False,'additional_external_mapping_used':False,'optional_source_categories_or_I_MEDAS_built':False,'raw_source_category_count':33,'code':P+'/scripts/w07_w08_diet_views_20261028','result':P+'/data_qc/w07_w08_diet_views_20261028','memo':P+'/memos/w07_w08_diet_views_20261028/MEMO.md','runtime':L,'correction':'Initial commentary read exact full-window baseline count as 15; rechecked output is 45. This cache/report use 45.'}
assert r['FoodIDs']==r['known_NOVA_IDs']+r['unknown_NOVA_IDs'];assert r['baseline']['UPF_zero_unknown_energy']['days']>=r['baseline']['UPF_all_rows_classified']['days']
(A/'observed_aggregate.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
report=f'''# W07、W08执行结果：FoodID映射与饮食特征视图

2026-10-28。**W07版本化映射及技术验收完成；W08饮食自身的日级／窗口视图及技术验收完成。W08的真实CGM共同日分支仍等待W09点级QC，未标作全部完成。** 构建与独立复核实际退出码均0。

本报告是TRE终端聚合结果的本地记录。个体饮食特征、FoodID产品明细和日日期保留在服务器。本轮没有读取菌群／代谢结局、没有拟合关联或预测模型，没有改变正式V3主暴露和检验预算。

## 1. 生成了什么

|产物|行数|作用|
|---|---:|---|
|FoodID crosswalk|7,770|每个当前处理事件源中的FoodID一行，保存名称、产品、源类别、继承NOVA、未知原因及版本|
|NOVA覆盖与区间日表|149,912|每日总能量E、已知UPF能量K、未知分类能量U、覆盖及上下界|
|饮食日级候选表|149,912|W06索引匹配事件日；保留无有效暴露的日并标记缺失|
|饮食窗口表|12,085个人—访视|9,531位参与者，可能有多个访视；各候选使用自己的有效日集|
|真实CGM共同日表|未生成|等待W09合格日与点级QC；没有以旧日索引存在冒充点级质控通过|

没有取CGM、body、liver、口腔或肠道的全域交集。基线10k的9,087个饮食窗口对应9,087位参与者。

## 2. W07映射采用什么依据

复用用户提供、W04已核对来源的**643个食物短名称及旧NOVA分类**，只做去首尾空格和casefold后的精确匹配。当前FoodID的全部非缺失短名称必须一致地对应一个有效旧分类，才能继承；没有模糊匹配，没有从产品名猜加工等级，也没有引入外部映射源。

版本：`legacy_NCDiet_NOVA_643_419192dbf31a_FoodID_W07_20261028`。

|核查|结果|
|---|---:|
|完整处理源事件扫描|2,640,959行|
|当前FoodID|7,770|
|能继承有效NOVA的FoodID|1,933|
|NOVA未知的FoodID|5,837|
|同FoodID多个短名称|0|
|同FoodID多个源类别|0|
|W06准入事件用于计算|2,420,902行|

`Stevia Sweetner`和`Spirulina`继续保持原目录空白。未匹配／原目录空白／无名称／冲突分别登记。未知产品按影响的已知能量排序，留TRE供后续人工核对。

**这是可追溯的旧分类迁移，不是对当前每个产品配方重新完成NOVA人工鉴定。** crosswalk保留`manual_current_product_NOVA_review_completed=False`。技术映射通过不意味着未知分类消失，也不意味着已经证明旧目录是当前HPP官方数据库版本。

## 3. UPF未知分类如何处理

每日能量合格，要求所有保留事件能量有效、非负且当日总能量>0。

- 区间下界：`100×K/E`。
- 区间上界：`100×(K+U)/E`。
- `UPF_point_zero_unknown_energy_pp`：仅在`U=0`时给精确点；未知分类条目若记录能量恰为0，在记录能量口径下不扩大区间。
- `UPF_point_all_rows_classified_pp`：更严格，要求所有事件NOVA均已分类；它与上一版本分列保存。

例子：总能量2,000 kcal，已知UPF 600 kcal，未知分类200 kcal，得到**30%–40%**，精确点值保持缺失。分母始终是全部保留记录的总能量。

基线能量合格日的能量加权NOVA覆盖为**88.19%**，日UPF区间宽度中位数**6.65个百分点**、p99约**60.54个百分点**。此口径只针对能量完整合格日；W04旧“所有索引匹配日”的覆盖诊断不能不加说明地直接混用。

## 4. 基线到底有多少人、多少天

下表是**饮食自身可用性**，不是最终模型样本数。≥7日只是诊断，尚未冻结为统一准入标准。

|视图|至少1个对应有效日的人数|有效日数|至少7个对应有效日的人数|
|---|---:|---:|---:|
|有事件且匹配索引|9,087|116,906|8,849|
|能量可计算|9,065|113,517|8,649|
|UPF：未知分类能量为0|7,970|35,196|1,714|
|UPF：所有事件已分类|7,957|35,057|1,707|
|纤维密度可计算|8,677|60,574|4,662|

**整个能量合格窗口都有精确UPF的基线参与者是45人**，这里没有额外要求至少7日。因此，45人与1,714人不是同一个条件下的数字。

- 1,714人：至少有7个可以算精确UPF的日期，其他日期仍可能有分类缺口。
- 45人：这个人的所有能量合格日期都没有未知NOVA能量，但其窗口可能很短。

例如一个人14个能量合格日中，8日UPF分类完整、6日有缺口：他可以进入“≥7个精确UPF日”诊断，但8日均值不等于整个14日窗口的精确均值。窗口表同时保存有效日子集点值和覆盖完整能量窗口的上下界。

早先聊天曾把45误读为15，本报告与本地聚合缓存均已按重新核对的45更正。

## 5. W08每天、每个窗口怎么算

纤维日密度：`1000×当天完整纤维g／当天完整能量kcal`。纤维缺失的日期不填0，也不牵连其他候选。

每个候选分别保存：有效日数、首末有效日期、分子累计量、同日集能量分母、日等权均值和累计比。没有用纤维有效日替代UPF有效日，也没有把完整窗口的能量分母配上某个子集的UPF分子。

|变量示例|意义|
|---|---|
|`UPF_zeroU_equal_day_mean`|UPF精确日子集的每天等权平均，单位百分点|
|`UPF_zeroU_cumulative_ratio`|同一UPF精确日子集的累计UPF能量／累计总能量|
|`UPF_lower_equal_day_mean_all_energy_days`|全部能量合格日的UPF下界等权均值|
|`UPF_upper_cumulative_all_energy_days`|全部能量合格日累计比的上界|
|`UPF_exact_full_energy_window_*`|只有所有能量合格日均无未知能量时才提供的完整窗口精确点|
|`fiber_equal_day_mean`|纤维有效日密度等权平均，g/1000 kcal|
|`fiber_cumulative_ratio`|同一纤维有效日集的1000×累计纤维／累计能量|

两日UPF分别80%和10%，总能量分别1,000和3,000 kcal：等权均值是**45%**，累计比是**27.5%**。两个结果都保存，尚未按关联结果选主口径。

原始日表也保留计数、日期、映射版本和缺失状态，因此可以回溯一个窗口值到底用了哪些日。W08未把有记录日改称“完整全天摄入”。

## 6. 分布与观察期重复性

以下为基线、各自有效样本的描述，日分布与人级分布的观察单位不同。

|候选|有效日值中位数|个人有效日等权均值中位数|个人同日集累计比中位数|
|---|---:|---:|---:|
|UPF，未知能量为0|15.19%|17.80%|17.91%|
|纤维密度|13.25 g/1000 kcal|13.56 g/1000 kcal|13.34 g/1000 kcal|

基线能量合格日观测能量范围约500.01–4,999.09 kcal，中位数1,653.50 kcal；这是本次源表的观测范围，**不是W08新设的热量排除阈值**。纤维日密度p99约37.40、最大99.40 g/1000 kcal；极端值没有按结局关联进行删改，后续冻结模型前仍需按既定规则复核。

把每个人按时间排序后的有效日分成奇数日、偶数日，分别求均值；仅对至少7个相应有效日、两半均有值的人做描述。

|候选|诊断人数|奇偶日均值Spearman相关|
|---|---:|---:|
|UPF，未知能量为0|1,714|0.598|
|UPF，所有事件已分类|1,707|0.598|
|纤维密度|4,662|0.720|

这反映的是**本次观察期内的重复性**，不能当作长期饮食稳定性、发现／确认分组或模型预测成绩；各候选样本不同，也没有凭相关大小冻结主暴露。

## 7. 哪些已经验收，哪些还在等

**已验收：**FoodID唯一性、643目录分类一致性、逐日公式与上下界、W06日QC一对一对应、窗口各自日集、7个副本hash／行数；独立复核重新扫描2,420,902条准入事件并核算3个实际日期。真实构建和复核均exit0，峰值RSS约0.33 GiB。

**W08真实CGM分支尚未完成：**按任务表，`food_cgm_common`必须等待W09原始点级QC及有效日表。已生成`common_day_adapter.py`，要求CGM日期键唯一，并显式标记`point_qc_verified=True`和`cgm_day_eligible`。它按共同日重新求窗口，而不是复制饮食全窗口均值。

合成验证中，饮食全窗口45%的日均值，在CGM只覆盖第一日时，重新计算为80%；重复CGM键或未经点级验证的输入会被拒绝。**这些是代码测试，不是实际CGM结果。**

四项食物来源解释和适配I-MEDAS仍未凭NOVA字段／未确认类别代理强行生产。本轮33个原始food_category的清单留TRE，后续可单独核对应关系；它们不作为两个核心候选的完整病例准入条件。

人工当前产品NOVA审定、正式主暴露、日均／累计比的主口径、最少天数及检验预算均未在本轮冻结。

## 8. 代码、Result与MEMO在哪里

服务器项目根P：`{P}`。

|代码|功能|
|---|---|
|`P/scripts/w07_w08_diet_views_20261028/01_build_w07_w08.py`|完整食品字典、版本化映射、日覆盖／候选、各自窗口、分布与观察期重复性、QC网页及副本|
|同目录`02_verify_w07_w08.py`|独立源分类核对、所有公式／窗口／W06读回、准入源重扫、3日期核算及共同日合成测试|
|同目录`common_day_adapter.py`|供W09后接入的共同日过滤与窗口重算函数；真实CGM数据尚未接入|

|Result|位置|
|---|---|
|FoodID版本表|`P/config/food_id_crosswalk_versioned.parquet`|
|NOVA日覆盖|`P/analysis/shared/nova_coverage_TRE_ONLY.parquet`|
|饮食日表|`P/analysis/shared/food_daily_TRE_ONLY.parquet`|
|饮食窗口表|`P/analysis/shared/food_windows_TRE_ONLY.parquet`|
|映射状态汇总|`P/data_qc/nova_mapping_audit.csv`|
|规则|`P/config/diet_views_rules_W07_W08_20261028.json`|
|TRE内QC阅读页|`P/data_qc/diet_candidate_QC.html`|
|汇总、验证、未知产品人工复核清单|`P/data_qc/w07_w08_diet_views_20261028/`|
|远程MEMO|`P/memos/w07_w08_diet_views_20261028/MEMO.md`|
|日志／实际退出码|`{L}`|

本地：[代码与备忘录](../audits/w07_w08_diet_views_2026-10-28/MEMO.md) · [聚合缓存](../audits/w07_w08_diet_views_2026-10-28/observed_aggregate.json) · [构建及独立验收证据](../audits/w07_w08_diet_views_2026-10-28/01_build_and_verification.png) · [分布及产物证据](../audits/w07_w08_diet_views_2026-10-28/02_feature_distributions.png)。
'''
(ROOT/'reports/HPP_W07_W08_Diet_Feature_Views_2026-10-28.md').write_text(report)
memo=f'''# W07／W08代码与Result备忘录

日期2026-10-28。01构建、02独立复核实际退出码均0。

- W07：版本化旧分类FoodID迁移完成，未知保持未知；当前产品新配方人工NOVA审定未冒称完成。
- W08：饮食自身日级／窗口视图完成；真实CGM共同日分支等待W09，未标作W08全部完成。

## 跑了什么代码

|代码|功能|实际状态|
|---|---|---|
|01_build_w07_w08.py|全处理源2640959行建FoodID字典；准入2420902事件算NOVA覆盖、UPF两种精确条件／区间、纤维、各自窗口／观察期QC|exit0|
|02_verify_w07_w08.py|独立核对继承目录、W06日QC、日／窗口所有公式，重扫2420902行核3个日期，副本和合成CGM窗口重算|exit0|
|common_day_adapter.py（服务器生成）|CGM合格共同日过滤和窗口重算；要求唯一日键和点QC验证|只做合成验证，未接真实CGM|
|prepare_transfer.py／launch.py|自编源码打包、SHA、pheno顺序运行并留真实退出码|已核包SHA并启动PID28477；两个子程序exit0|
|03_record_delivery.py|本地转录聚合、报告、备忘录和进度更新|只读本地资料，不拉取个体数据|

Code：`{P}/scripts/w07_w08_diet_views_20261028`。

Result：`{P}/data_qc/w07_w08_diet_views_20261028`。

远程MEMO：`{P}/memos/w07_w08_diet_views_20261028/MEMO.md`。

日志／退出码：`{L}`。

7个关闭后逐字节副本已验SHA和行数。映射7770行、NOVA覆盖及日表各149912行，窗口12085行。三项Parquet不是相同粒度，不可直接按行拼接。

## 当前人数与限定

基线索引匹配9087人116906日；UPF零未知能量点值7970人35196日，≥7日1714；严格全事件分类7957人35057日，≥7日1707；纤维8677人60574日，≥7日4662。整个能量窗口无未知NOVA能量45人，不限至少7日；初次聊天误读15已更正。

完整结果与说明：[W07/W08报告](../../reports/HPP_W07_W08_Diet_Feature_Views_2026-10-28.md)。

## 给后续W09接续

1. 先完成原始CGM点级QC、连接选择及有效日表，不能把旧日索引有记录当作合格。
2. 适配输入唯一键`participant_id/cohort/research_stage/date`；提供布尔`cgm_day_eligible`及`point_qc_verified=True`，日历／日期类型按已验合同统一。
3. 通过服务器`common_day_adapter.py`的`common_view(food_daily,cgm_days)`取合格共同日，再调用`window(common_daily)`重算窗口，并把`window_id`命名为`food_cgm_common`。若无合格共同日，应明确停止并输出空集原因，不能默认拿食物全窗口均值替代。
4. 实际CGM共同日、窗口及与全饮食窗口的差异须另验，不以本次合成测试冒称完成。

本地`common_day_adapter_local.py`是从同一自编源函数重新AST生成的阅读副本，Python版本格式化不同，hash不等于远程库；canonical代码是01源及服务器生成库。未下载原始个体数据。
'''
(A/'MEMO.md').write_text(memo)
# Date-append only for protected background notes.
marker='2026-10-28 W07与W08饮食特征视图';before=A/'before_w07_w08_update';before.mkdir(exist_ok=True);records=[]
addition=f'''\n\n## {marker}\n\n01构建／02独立复核实际均exit0；W07技术映射完成，W08饮食自身日级与窗口视图验收通过，真实CGM共同日分支仍等W09，不能标W08全部完成。映射7770个FoodID，1933可继承643旧目录NOVA，5837未知；新外部映射0，当前产品人工NOVA审定未完成。NOVA日覆盖与食物日表各149912行，食物窗口12085个人—访视（9531人）。\n\n基线：索引9087人116906日；能量9065人113517日；零未知能量UPF7970人35196日（≥7日1714）；严格全事件分类UPF7957人35057日（≥7日1707）；纤维8677人60574日（≥7日4662）。整个能量合格窗口UPF精确点45人，不限≥7日；早先口头误读15已纠正。≥7仅诊断、主暴露及聚合未冻结。能量合格日NOVA能量覆盖88.19%，日区间宽中位6.65pp。\n\nCode：`{P}/scripts/w07_w08_diet_views_20261028/`。Result：`{P}/data_qc/w07_w08_diet_views_20261028/`。MEMO：`{P}/memos/w07_w08_diet_views_20261028/MEMO.md`。共享表`P/analysis/shared/food_daily_TRE_ONLY.parquet`、`food_windows_TRE_ONLY.parquet`、`nova_coverage_TRE_ONLY.parquet`；FoodID表`P/config/food_id_crosswalk_versioned.parquet`。独立重扫2420902准入源事件、核3日期、7副本hash及公式通过。\n\n[W07/W08报告](reports/HPP_W07_W08_Diet_Feature_Views_2026-10-28.md) · [代码MEMO](audits/w07_w08_diet_views_2026-10-28/MEMO.md) · [聚合缓存](audits/w07_w08_diet_views_2026-10-28/observed_aggregate.json)。\n'''
for name in ['HPP_DATA_BACKGROUND.md','HPP_SERVER_BACKGROUND.md']:
 p=ROOT/name;old=p.read_bytes();assert marker.encode() not in old;snap=before/name;snap.write_bytes(old);p.write_bytes(old+addition.encode());new=p.read_bytes();assert new.startswith(old);records.append({'path':name,'snapshot':str(snap.relative_to(ROOT)),'bytes':len(old),'sha256':h(old),'append_only':True,'after_sha256':h(new)})
(A/'background_append_manifest.json').write_text(json.dumps({'date':'2026-10-28','records':records},indent=2))
for name in ['HPP_LOCAL_VERIFIED_FACTS.md','HPP_PROJECT_BACKGROUND.md','reports/HPP_CURRENT_PLAN.md']:
 p=ROOT/name;t=addition
 if name.startswith('reports/'):t=t.replace('(reports/HPP_','(HPP_').replace('(audits/w07_','(../audits/w07_')
 assert marker not in p.read_text();p.write_text(p.read_text()+t)
D=ROOT/'outputs/researchwrite/hpp_multisystem_20261021/team_checklist_20261026';p=D/'workpackage_progress.json';x=json.loads(p.read_text());x['evidence']='audits/w07_w08_diet_views_2026-10-28/MEMO.md'
common={'what_ran':'01_build_w07_w08.py构建；02_verify_w07_w08.py独立源／公式／副本与共同日合成验收，均exit0。','code_location':f'服务器 `{P}/scripts/w07_w08_diet_views_20261028/`；本地自编源与[MEMO](../audits/w07_w08_diet_views_2026-10-28/MEMO.md)。','result_location':f'服务器 `{P}/data_qc/w07_w08_diet_views_20261028/`；共享视图 `P/analysis/shared/`、映射及规则 `P/config/`；远程 `P/memos/w07_w08_diet_views_20261028/MEMO.md`。P同代码项目根。本地[W07/W08结果](HPP_W07_W08_Diet_Feature_Views_2026-10-28.html)。'}
x['tasks']['W07']=dict(common,checked=True,short_status='FoodID迁移与覆盖验收完成；未知分类保留、当前产品人工审定待进行',status_text='2026-10-28构建／独立验收exit0。7770个FoodID，1933继承有效旧NOVA、5837未知，无一ID多名称／多类别。全处理源与2420902准入行完整核查，分类／事件不扩行。基线能量合格日覆盖88.19%；保留上下界及两种严格度点值，不把未知当非UPF；未新用外部映射或冻结主暴露，人工产品NOVA审定及团队复核待进行。')
x['tasks']['W08']=dict(common,checked=False,short_status='饮食日／窗口视图及技术验收完成；CGM共同日真实分支等待W09',status_text='2026-10-28饮食自身日表149912行、窗口12085个人—访视，日均／同日集累计比／全窗口区间及各自日集分开，7副本、3实际日期独立核算及全部公式通过，01／02均exit0。基线≥7日诊断UPF零未知1714、严格1707、纤维4662；完整能量窗口精确UPF45人（不限7日）。共同日适配函数合成通过，但未生成真实CGM共同日／窗口，需W09点级QC，因此不勾全部完成。未冻结最少天数、主聚合或主暴露，未读结局。')
x['latest_run_id']='w07_w08_diet_views_20261028';x['latest_local_workpackage']='W08_food_only';x['latest_local_run_id']='w07_w08_diet_views_20261028';p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
p=D/'execution_update.md';p.write_text(p.read_text()+'''\n\n### 2026-10-28 W07／W08实际执行\n\nW07版本化映射与覆盖已技术验收；W08饮食自身日／窗口视图已验收，真实CGM共同日分支待W09，因此W08不勾全部完成。构建／独立复核均exit0。FoodID7770（NOVA已知1933），日表149912，窗口12085人—访视；7副本、全准入源重扫及3日期核算通过。基线≥7日诊断：零未知能量UPF1714、严格UPF1707、纤维4662。完整能量窗口精确UPF45人，不限≥7日。\n\n[完整报告](HPP_W07_W08_Diet_Feature_Views_2026-10-28.html) · [代码、Result及备忘录](../audits/w07_w08_diet_views_2026-10-28/MEMO.md)。\n''')
(A/'delivery_validation.json').write_text(json.dumps({'passed':True,'build_actual_exit':0,'independent_verify_actual_exit':0,'crosswalk_rows':7770,'daily_rows':149912,'window_rows':12085,'source_days_verified':3,'admitted_event_rows_reread':2420902,'mirrors_verified':7,'W07_technical_complete':True,'W08_food_only_complete':True,'W08_real_CGM_common_complete':False,'outcomes_read':False,'human_team_review_done':False},indent=2))
print('W07/W08 local report, aggregate cache, MEMO and honest branch status recorded')
