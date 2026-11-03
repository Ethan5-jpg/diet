"""Write observed aggregate W03 results and local lookup cache; no remote/file-data access."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];A=Path(__file__).resolve().parent
D=ROOT/'outputs/researchwrite/hpp_multisystem_20261021/team_checklist_20261026'
P='/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021';RUN='w03_keys_20261026T134938Z'
Q=P+'/data_qc/'+RUN;C=P+'/config/'+RUN;S=P+'/scripts/'+RUN;M=P+'/manifests/'+RUN;MEMO=P+'/memos/'+RUN+'/MEMO.md'
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def main():
 snap=A/'before_w03_update';snap.mkdir(exist_ok=False);records=[]
 names=['HPP_SERVER_BACKGROUND.md','HPP_DATA_BACKGROUND.md','HPP_PROJECT_BACKGROUND.md','reports/HPP_CURRENT_PLAN.md','reports/HPP_Team_Execution_Checklist_2026-10-26.md']
 for name in names:
  b=(ROOT/name).read_bytes();p=snap/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
  records.append({'path':name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'snapshot':str(p.relative_to(ROOT))})
 save(snap/'manifest.json',records)
 counts=json.loads((A/'observed_source_counts.json').read_text())
 assert sum(r['source_rows'] for r in counts['sources'])==3021035
 observed={'provenance':'Manually transcribed from aggregate TRE terminal output and saved screenshots, not exported individual rows','run_id':RUN,'date':'2026-10-26','sources_full_scanned':14,'rows_key_date_projected':3021035,'audit_and_followups_actual_exits':{'02':0,'03':0,'04':0,'05':0},'source_row_conservations_passed':14,'dictionary_exact_field_evidence_rows':57,'output_hashes_verified':38,'invalid_original_key_rows':9662,'identity_exception_records':88183,'join_profiles':10,'raw_many_to_many_gut_person_visit_groups':22,'oral_index_version_key_date_difference_groups':[0,0],'oral_original_stage_placeholder_rows':8834,'oral_index_original_stage_placeholder_rows_each_version':414,'oral_bridge_unique_exact_matches':8398,'oral_bridge_baseline_rows_and_unique_person_visits':8291,'oral_bridge_unmatched_held':436,'oral_bridge_multiple_abundance_rows_per_resolved_visit':0,'food_local_timestamp_calendar_conflicts':1066,'food_collection_date_vs_aware_timestamp_conflicts':0,'outcome_values_read':False,'raw_CGM_points_QC_performed':False,'final_biological_sample_registry_built':False,'final_eligible_N_recomputed':False,'code_directory':S,'contract':C+'/key_contract.yaml','results_directory':Q,'memo':MEMO}
 save(A/'observed_aggregate.json',observed)
 # Source paths from our own corrected authored collector, plus metadata rows manually observed.
 tree=ast.parse((ROOT/'audits/w02_source_inventory_2026-10-26/02_source_inventory_corrected.py').read_text());core=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CORE' for t in n.targets))
 for row in counts['sources']:row['source_path']='/home/ec2-user/studies/hpp_datasets/'+core[row['role']]
 save(A/'observed_source_counts.json',counts)
 table='| 来源 | 全表行数 | 10k基线人—访视（原始有效键） | 原始键异常行 |\n|---|---:|---:|---:|\n'
 for r in counts['sources']:table+=f"| {r['role']} | {r['source_rows']} | {r['baseline_person_visits_original_keys']} | {r['invalid_original_key_rows']} |\n"
 report=f'''# W03 身份、日期与连接规则：已核查结果

2026-10-26；运行 `{RUN}`。主扫描及03／04／05跟进实际均退出0。14个来源完整扫描了身份与日期投影，共3,021,035行；14个来源行数守恒通过，38个产物hash读回通过，字典中57条精确字段证据留TRE。没有读取血糖／体脂／肝脏／菌群丰度数值，没有模型或最终合格N。

## 最重要的发现

**口腔物种表的8834行 `research_stage` 均为占位值。**不能按原始三字段直接合并，也不能把 `undefined` 替换为 `00_00_visit`。两版口腔索引各有414行访视占位；其余有效键／日期／重复数在两版之间完全一致，但没有检验其他特征值，不能因此把版本混合。

定向复核只使用同一参与者、来源表明确的同一采样日、唯一且有效的口腔索引行，并尊重丰度表中已有有效cohort/stage字段。结果：8398行有唯一访视证据，其中8291行／8291唯一人—访视属于10k基线，436行无精确匹配而隔离。已匹配者没有重复丰度行对应同一人—访视。**这不是最终菌群QC名单或分析N**；一人一访视内生物样本身份和技术QC仍由W11验收。严格日期仅用于修复缺失的访视来源证据，不是对所有菌群施加采样日期准入门槛；不按最近日期补配、不因距离饮食／CGM久自动排除。

肠道物种与索引按三字段直接连接，有22个分组两侧都多行，可能产生多对多行膨胀；另有599个丰度侧person-visit键未在当前有效索引匹配。这些是全源范围的原始键分组数，非基线合格人数。先保留待W11版本／样本选择，不能随意去重或把矩阵扩张后的行数当人数。

饮食主 `collection_date` 与带Asia/Jerusalem时区的 `collection_timestamp` 日历日期冲突0；与naive `local_timestamp` 比较有1066行日期冲突（全表2,640,959行）。这不证明真实进食时刻已验收；W06负责语义与清洗。Body数值型时间戳没有猜epoch／单位，描述日期保留明确的collection_date。DST点级覆盖尚未检查。

## 人数口径

以下仅为来源存在性／键核查。原始有效三字段基线键以 `cohort=10k`、`research_stage=00_00_visit` 计数，未要求饮食／双菌群／结局交集，未要求临床字段有效。

{table}
口腔物种原始键基线计数0，表示访视字段不能直接用，**不表示没有口腔数据**。新增精确来源桥接的8291基线人—访视另列，不回写原始表、也不把它静默代入上表。不要把9096个DXA全表行数、6671基线存在数、旧6281饮食日期配对数或后续结局完整病例N混为同一个数。

## 连接契约

| 数据连接 | 本轮证据及规则 |
|---|---|
| 人与访视 | participant_id＋cohort＋research_stage；空值／占位隔离；没有独立visit_id时明确只具显式stage证据 |
| 食物事件→饮食日索引 | 先聚合到人—访视—日期；全源149912共同日期键，14713事件日期键无索引；不能将事件行当日数 |
| CGM日指标→连接 | 人—访视＋connection_id；9497连接键全匹配；5个人—访视有多连接，留W09按预定质量规则选择 |
| 饮食→CGM日指标 | 人—访视—真实日期；本轮全源89075共同日期键，非89075人，也不是旧基线共同日人数重算 |
| 肝汇总→明细 | 汇总侧1、明细侧多（多次测量）；8937共同人—访视键，32个明细侧键无汇总；不能拿第一行日期强行填汇总 |
| 口腔丰度→索引 | 原始stage不可用；使用另存的唯一精确日期证据桥，436行仍隔离，W11最终样本／QC后再分析 |
| 肠道丰度→索引 | 22组原始多对多、599丰度侧未匹配键；W11解决后才接入模型 |
| Body或liver→饮食 | 各自必要表连接；不要求CGM或其他无关结局齐全 |

本轮核的是键集合／基数，未物化原始多对多合并；并非所有下游join都已成功运行。未来执行仍须检查右表唯一性、预期丢失和行数守恒，发生未知膨胀停止相应模块。多日期／长lag不直接当身份失败。

## 保存位置

服务器P：`{P}`。

- 代码：`{S}`，包括02主扫描、03验证、04口腔证据桥、05日志快照。
- 规则：`{C}/key_contract.yaml`。
- Result：`{Q}`；含日期来源CSV、连接基数JSON、异常Parquet、口腔桥接Parquet及聚合summary。
- 验证：`{Q}/verification.json`；manifest与hash：`{M}`。
- MEMO：`{MEMO}`；完整02／03／04日志及实际退出记录：`{P}/logs/{RUN}`。
- 个体键暂存SQLite：`/home/ec2-user/multisystem_runtime_20261021/{RUN}/identity.sqlite`，仅TRE。

本机[MEMO](MEMO.md)、[聚合JSON](observed_aggregate.json)、[来源计数与路径JSON](observed_source_counts.json)、[验收截图](verification_screen.png)及根目录 `HPP_LOCAL_VERIFIED_FACTS.md` 已保留可复用信息。个体日期、ID、完整字典与异常表留TRE。

W03完成的是核查和带明确停止条件的键契约；W06、W09、W10、W11各自仍需执行。未冻结发现／确认名单，未决定最终菌群主样本，也未变更V3科学问题、分结局建队列或same-visit策略。
'''
 (A/'REPORT.md').write_text(report)
 memo=f'''# W03代码备忘录

日期2026-10-26。用户授权执行W03，并要求更新W01/W02结果和本地缓存。HPP Server Skill：Chrome→既有WorkSpaces→MATE Terminal，所有真实数据访问／写入／执行均经浏览器；不SSH、不下载个体数据。研究Python固定 `/home/ec2-user/miniconda/envs/pheno/bin/python`，本轮无安装依赖。

## 跑了什么代码

| 脚本 | 功能 | 本轮运行 |
|---|---|---|
| 01_probe_contract_inputs.py | 读W02既有manifest，查看键／日期schema类型和预算；无数据行 | inline实际输出PROBE_DONE |
| 02_audit_identity_dates.py | 14来源完整身份／日期投影，SQLite按来源／粒度计数；占位异常、日期冲突、10对join基数、契约及异常Parquet | 实际exit 0，3,021,035行；14源完整 |
| 03_verify_and_handoff.py | 独立核14源原始行＝有效＋异常；57条字典字段证据；区分异常字段；两版口腔键／日期比较；保存日志 | 实际exit 0 |
| 04_resolve_oral_visit_evidence.py | 全扫3个口腔表的键／日期；仅按唯一精确来源证据桥接访视，不改原始列、不用最近日期 | 实际exit 0；8398明确、8291基线、436待核 |
| 05_snapshot_final_status.py | 读02/03/04实际退出文件，完整日志保存到V2，更新38产物hash及项目状态 | 实际exit 0已在终端看到 |
| validate_contract_helpers.py | 5个合成检查：多对多识别、占位／空格键拒绝、数值时间不猜测、保留本地日历、访视分离 | 本地exit 0；非真实数据结果 |
| record_local_handoff.py | 手工观察聚合结果的本机登记，写报告／缓存／进度及背景；不访问TRE | 本地实际运行另见文档构建记录 |

代码位置：本地本目录自编源码；服务器 `{S}`。Result、config、MEMO路径见[REPORT](REPORT.md)。远程临时部署及stdout前缀为 `/tmp/hpp_w03_f0cadbf2`（02）、`/tmp/hpp_w03_verify_6653b947`（03）、`/tmp/hpp_w03_bridge_ea236556`（04）、`/tmp/hpp_w03_final_cac0cef9`（05）；各有.py/.log/.exit。02/03/04最终完整日志和实际退出快照已在V2 logs。最终核验截图留本机。

02每源独立串行子进程，2048行batch；必要身份／日期列投影≤1GiB、3GiB地址空间、父进程监视768MiB RSS／可用内存下限和超时；文件前后bytes/mtime及metadata行数一致，完整行数断言。旧原始食物事件编辑版本留W06，原始CGM点留W09；不读丰度和表型数值。resource内存守护是轮询，不能承诺瞬时RSS永不越限。

## 修改入口

来源与身份列：02的ROLES/KEY/DATE/EXTRA；缺字段不猜命名，不把undefined改基线。日期规则只保留真实类型，Body numeric不解析。各join的specs与join_profile负责显式粒度／键集合、原始潜在膨胀；真正下游join还须独立断言。

口腔桥：04的精确参与者＋来源采样日＋唯一索引规则；尊重原本有效cohort/stage，不按最近日期，不允许多个候选。这个日期匹配只是恢复访视来源证据，非菌群lag准入。436未匹配保持held，W11补版本／生物样本QC与跨结局锁定。同一日同人不自动等于同一生物样本。

01/02/03最初检查报告保留原始oral键全不通过的事实；04是新增证据桥，不倒改旧统计。后续使用contract及bridge，不能读原始oral stage直合。02_SUMMARY的原始基线0／9662原始异常行不被悄悄改成桥后计数。meta匹配不证明所有原始值相同，不对未读特征声明版本等价。

## 关键结果与后续

核查／键规则完成，有下游条件：436oral访视待核、22gut多对多、1066饮食naive日期冲突，分别W11／W06跟进。W09原始点级／DST、W10临床字段、W11最终样本QC尚未做；无最终合格N／名单／模型。W01沿用免重跑，W02 metadata完成。用户可以先从本地 `HPP_LOCAL_VERIFIED_FACTS.md` 查现有证据，不应因问历史结果而重连重扫。

## 源码hash

'''
 memo+='\n'.join('- `'+f.name+'`：`'+hashlib.sha256(f.read_bytes()).hexdigest()+'`' for f in sorted(A.glob('*.py')))+'\n'
 (A/'MEMO.md').write_text(memo)
 cache=f'''# HPP本地已核实事实：优先查询入口

最后更新2026-10-26。后续协作者先读本文件、项目背景及本地audit，不因询问历史数字重新连接服务器。**这是带日期的本地证据缓存，不是实时服务器状态。**只有要读新数据、重算、检查来源变化或现有记录确实缺失时才访问TRE。原始数据、个体ID／日期、字典原文与异常表仍留TRE。

| 工作包 | 已核实状态 | 本地入口 |
|---|---|---|
| W01 | 用户指定沿用同一服务器，免完整重跑；W02中确认主机／pheno／P/L与写读。不冒称新做容量基准或完整继承矩阵 | [W02备忘录](audits/w02_source_inventory_2026-10-26/MEMO.md) |
| W02 | 26/26来源metadata可读，8/8历史元信息匹配，29字典文件／991文本候选，14份输出hash；gut物种文件名前缀已修正 | [W02报告](audits/w02_source_inventory_2026-10-26/REPORT.md) |
| W03 | 14源全量键／日期投影3,021,035行；核查与带闸门契约完成；38产物hash，02/03/04/05实际exit均0 | [W03报告](audits/w03_identity_join_2026-10-26/REPORT.md)、[聚合JSON](audits/w03_identity_join_2026-10-26/observed_aggregate.json) |

## 服务器位置

- 当前核查主机：`ip-10-0-91-109.ec2.internal`。
- 原始数据R：`/home/ec2-user/studies/hpp_datasets`（只读）。
- 云盘V2：`/home/ec2-user/studies/diet_gut_cgm_v2`。
- 项目P：`{P}`；本地运行暂存L在服务器 `/home/ec2-user/multisystem_runtime_20261021`。
- Python：`/home/ec2-user/miniconda/envs/pheno/bin/python`。
- 当前W02 run：`w02_sources_20261026T130908Z`；W03 run：`{RUN}`。
- 服务器代码／Result／MEMO精确位置在两个本地报告，不必为查路径登录。

## 已确认的数据结构和连接问题

1. Food有日索引、处理后事件、原始编辑事件；一条事件非一餐。处理后事件全源2,640,959行。历史10月18日基线事件记录9737人、125374人日、每人中位14记录日、每个人日中位16事件。本轮按有效原始键复验事件基线存在9737人—访视，不等于完整摄入或模型合格N。
2. CGM连接和历史指标各9497行，原始目录9497片段（不是9497人）；W03原始键基线存在9405人—访视。历史饮食—CGM基线8367人、≥7真实共同日7065人、共同日中位12天来自10月18日完整日期核查；本轮未重做该资格和点级覆盖。5个人—访视多连接留W09处理。
3. Body／liver与CGM分别建自身最大必要队列，不取全域交集。W03来源基线存在：anthropometrics10402、body_composition6671、liver_detail6782、liver_aggregated6758人—访视；均未检验特定Y有效或饮食O/G交集，不能作为最终N。
4. Oral物种8834行原始research_stage均占位；唯一精确索引／采样日证据桥8398行，其中8291基线行／唯一人—访视，436行held。8291不是样本QC后或饮食／结局交集的最终N。两版口腔索引键／日期／重复数无差异，不证明全部特征值相同，不混版本。
5. Gut原始索引基线存在8694、物种丰度基线存在9087人—访视；两者不等数。全源22组原始多对多、599个丰度侧未匹配人—访视键，W11处理。真实物种路径为 `R/gut_microbiome/abundance/metaphlan4/aggregated/metaphlan_abundance_species.parquet`；旧R16漏前缀。
6. Food主日期与带时区时间戳的日期冲突0；naive local_timestamp冲突1066行；App记录时间是否实际进食仍未知。Body一个时间戳数值型，未猜epoch；Arrow多数aware timestamp带Asia/Jerusalem，不能写死UTC偏移。原始CGM点覆盖／DST仍待W09。
7. 菌群same visit策略未加固定lag／日期唯一／必须前采门槛。口腔“同采样日”是修复原始访视占位的来源证据，不是要求菌群与饮食同日。

## 数字要怎样查

来源总行数、原始键基线存在数、异常行、多日期分组及输入路径：[完整来源JSON](audits/w03_identity_join_2026-10-26/observed_source_counts.json)。不同表分母不要混用。正式各域E/D、pair合格N、样本QC通过数、发现／确认名单目前未生成。

代码功能／修改入口／实际退出及截图：[W03 MEMO](audits/w03_identity_join_2026-10-26/MEMO.md)。当前任务状态权威源：`outputs/researchwrite/hpp_multisystem_20261021/team_checklist_20261026/workpackage_progress.json`，重建MD/HTML时会保留。V3科学设计不因本次核查自动改变。
'''
 (ROOT/'HPP_LOCAL_VERIFIED_FACTS.md').write_text(cache)
 progress=json.loads((D/'workpackage_progress.json').read_text());progress['latest_run_id']=RUN;progress['evidence']='audits/w03_identity_join_2026-10-26/MEMO.md'
 progress['tasks']['W01']['result_location']+=' 本地汇总：[W01／W02已核实记录](../HPP_LOCAL_VERIFIED_FACTS.md)。'
 progress['tasks']['W02']['result_location']+=' 本地：[W02执行报告](../audits/w02_source_inventory_2026-10-26/REPORT.md)及[MEMO](../audits/w02_source_inventory_2026-10-26/MEMO.md)。'
 progress['tasks']['W03']={'checked':True,'short_status':'核查／契约完成；口腔436行隔离、样本QC待W11；Codex执行','status_text':'2026-10-26已完成14来源完整键／日期投影及连接规则核验；3,021,035行，02／03／04／05实际exit均0。口腔原始stage占位，经精确来源桥8398行明确（8291基线），436隔离；gut22组多对多留W11。契约完成不等于最终样本QC／合格N，未选最终菌群样本，乙／甲人工复核待进行。','what_ran':'02_audit_identity_dates.py（14源完整键／日期扫描）；03_verify_and_handoff.py（守恒／字典／版本比较）；04_resolve_oral_visit_evidence.py（口腔来源桥）；05_snapshot_final_status.py（退出／日志／38产物hash）。','code_location':'`'+S+'`；本地自编代码在 `audits/w03_identity_join_2026-10-26/`，功能见[MEMO](../audits/w03_identity_join_2026-10-26/MEMO.md)。','result_location':'`'+Q+'`；契约 `'+C+'/key_contract.yaml`；远程备忘录 `'+MEMO+'`。本地：[W03报告](../audits/w03_identity_join_2026-10-26/REPORT.md)、[聚合缓存](../HPP_LOCAL_VERIFIED_FACTS.md)。'}
 save(D/'workpackage_progress.json',progress)
 (D/'execution_update.md').write_text(f'''## 当前实际执行进度：2026-10-26（W03已运行）

| 工作包 | 当前状态 | 已有结果 |
|---|---|---|
| W01 | 用户指定沿用、免完整重跑；保留空框并不阻塞 | 原主机／pheno与P/L、W02写入读回通过；没有冒称重新完成全部W01 |
| W02 | 元信息来源盘点已完成 | 26/26来源、8/8历史元信息匹配，路径修正、清单／预算／MEMO已保存 |
| W03 | **核查及带停止条件的键契约完成** | 14源全量键／日期投影3,021,035行；行数守恒14/14，38产物hash；主扫描和跟进实际exit均0 |

W02／W03由Codex在TRE代执行，团队成员人工复核未冒称完成。下面总表保留“跑了什么代码／代码位置／Result位置”三项，并补上本地报告链接。

**W03发现了口腔原始访视字段问题，并保存了来源证据桥：**8834丰度行research_stage均占位。仅用唯一精确参与者／来源采样日／有效索引证据，8398行可明确，其中8291基线；436行继续隔离。8291不是最终QC或分析N。gut另有22组原始多对多；饮食naive日期冲突1066行。分别留W11／W06处理，不能为提高N直接补访视或去重。

后续问历史字段、位置或数字，先看[本地已核实事实](../HPP_LOCAL_VERIFIED_FACTS.md)、[W02报告](../audits/w02_source_inventory_2026-10-26/REPORT.md)、[W03报告](../audits/w03_identity_join_2026-10-26/REPORT.md)和MEMO；无需为重复查询重登服务器。个体ID／日期／异常表留TRE。

当前可继续W04特征定义／W05营养素语义；W11须使用本轮口腔桥并处理436待核及gut重复，正式模型前必须完成。原始CGM点级／DST仍待W09，临床有效性待W10。V3分域队列与菌群same-visit规则保留，没有新增固定lag准入；精确采样日仅作口腔访视证据恢复。后文首轮步骤是原始排程，以本节和任务卡实际状态为准。
''')
 section=f'''\n\n## 2026-10-26 W03实际核查与本地查询入口

后续优先读本地 `HPP_LOCAL_VERIFIED_FACTS.md`、`audits/w03_identity_join_2026-10-26/REPORT.md` 和MEMO，不因查询已有结果重连TRE。W01沿用免完整重跑，W02元信息完成；本轮W03运行 `{RUN}` 已完成14源完整键／日期投影3,021,035行、14源守恒及38产物hash，02/03/04/05实际exit均0。没有读取表型／菌群丰度数值、训练、生成最终合格N或冻结参与者角色。

口腔物种8834行原始research_stage均占位；通过唯一的同参与者／来源采样日／有效索引记录恢复访视证据8398行，其中8291基线人—访视，436行隔离。不能将原始undefined统一改基线；这不是最终样本QC名单。gut22组原始多对多、599丰度侧未匹配键留W11；饮食naive日期冲突1066行留W06。Body numeric时间不猜epoch，原始CGM点级／DST待W09。菌群same-visit未增加固定lag或日期唯一限制；口腔精确日期桥只用于修复访视来源证据。

来源基线存在数：food事件9737、CGM连接9405、anthropometrics10402、DXA6671、肝明细6782／汇总6758、口腔索引8294／证据桥8291、gut索引8694／丰度9087人—访视；均非饮食＋双菌群＋Y质量后最终N。全表行数与基线数详见本地来源JSON；不要混旧日期配对分母。

服务器代码 `{S}`；结果 `{Q}`；键契约 `{C}/key_contract.yaml`；MEMO `{MEMO}`；完整日志在P/logs/{RUN}。个体异常及桥接表只留TRE，本机保存自编代码、聚合数、规则摘要、路径和截图。
'''
 for name in names[:4]:
  with (ROOT/name).open('a') as f:f.write(section)
  rec=next(r for r in records if r['path']==name);rec['after_sha256']=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
 save(A/'background_append_manifest.json',{'scope':'Append-only W03 update; before snapshots retain earlier W02 facts and original science','records':records[:4]})
 state=json.loads((D/'state.json').read_text());state.update(stage='W03_key_contract_complete_with_downstream_gates',W03='key_date_audit_and_contract_complete_with_held_exceptions',latest_run_id=RUN,participant_key_date_rows_projected_this_run=3021035,outcome_values_read_this_run=False,scientific_tasks_executed_this_turn=False);save(D/'state.json',state)
 with (D/'MEMO.md').open('a') as f:f.write('\n\n## W03实际执行后更新\n\nW03完成14源3,021,035行键／日期核查及口腔明确访视证据桥；83​98行已明确、8291基线、436held；38产物hash与02/03/04/05实际退出0。来源报告、缓存及MEMO保存本机；个体留TRE。任务进度源加入W03，并给W01/W02补本地报告。详见audit W03 MEMO。文档构建／validator需同步支持已核验W02＋W03状态及有快照的背景追加链，科学源仍保持原hash。\n')
 print('W03 report, memo, local facts cache, persistent task status and dated backgrounds recorded.')
if __name__=='__main__':main()
