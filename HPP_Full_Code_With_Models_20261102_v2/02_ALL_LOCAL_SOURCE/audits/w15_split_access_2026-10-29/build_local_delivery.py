"""Local aggregate-only delivery and living-status update after remote exit 05=0."""
from pathlib import Path
import json,hashlib,html,re,shutil
D=Path(__file__).resolve().parent;W=D.parent.parent;C=W/'config/w15_split_access_20261029';C.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ag={'access_enforcement':'workflow_API_guard_not_filesystem_ACL','actor_declarations':4,'actual_final_roles_assigned_people':0,'artifacts':10,'candidate_multi_person_units':529,'candidate_people':6305,'candidate_singleton_units':5170,'candidate_units':5699,'design_core_selected_people':0,'family_household_source_edges':1780,'final_split_fraction':None,'full_graph_nodes':11269,'guard_and_family_tests':19,'historically_unexposed_validation_established_people':0,'history_event_rows':8,'history_exact_ID_overlap':'unknown','independent_verification_checks':61,'largest_candidate_unit':16,'module_membership_rows':54151,'modules':11,'new_scientific_Y_access_opened_people':0,'people_in_multi_person_candidate_units':1135,'people_with_known_full_component_size_gt1':1440,'relevant_full_graph_nodes':6726,'seed':'W15_global_person_family_order_20261029_v1','source_Y_value_columns_read':0}
digest=hashlib.sha256(json.dumps(ag,sort_keys=True,separators=(',',':')).encode()).hexdigest();assert digest=='ac54b8aad00e6228c107e6e781306b9243b864ebba5d071faaa1b64c8e0d5284',digest
(D/'observed_aggregate.json').write_text(json.dumps(ag,ensure_ascii=False,indent=2)+'\n')
P='/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021';L='/home/ec2-user/multisystem_runtime_20261021/w15_split_access_20261029';Q=P+'/data_qc/w15_split_access_20261029'
counts={'D_BMI':6290,'D_Glu':5876,'Daily_CGM':5811,'D_Liver_speed_of_sound':4597,'D_Liver_velocity':4591,'D_Liver_viscosity':4591,'D_Liver_elasticity':4591,'D_Liver_attenuation':4591,'D_Liver_dispersion':4591,'D_TotalFat':4338,'D_VAT':4284};assert sum(counts.values())==54151
evidence={'run_id':'w15_split_access_20261029','status':'W15_complete_global_order_history_ledger_workflow_guard','actual_exit_codes':{t:0 for t in ['01','02','03','04','05']},'aggregate_sha256':digest,'remote_aggregate_SHA256_visually_matched':True,'independent_BFS_checks':61,'published_readback_checks':82,'production_artifacts_verified':10,'new_scientific_Y_values_read':0,'individual_data_exported':False,'module_membership_counts':counts,'historical_contact_evidence':'user declaration plus existing W04 records; colleague self confirmation and exact-ID history audit incomplete','human_review_complete':False,'access_control_scope':'workflow/API guard; no filesystem ACL or cryptographic sealing','remote_completion':Q+'/completion_status.json','remote_final_verification':Q+'/final_delivery_verification.json','actual_exit_records':L+'/{01,02,03,04,05}_actual_exit_code','next_workpackage':'W16'}
(C/'execution_status.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
status='W15已在TRE完成全项目个人／已知家系顺序、历史接触账本和默认拒绝的流程访问接口；01–05实际exit均0。6305人、5699单元、11模块54151成员记录；529多候选人单元含1135人。10正式产物，61独立BFS核验＋82正式读回检查通过。用户已有历史接触、同事待本人确认、AI已读历史报告并用于方案讨论、其他会话／人工运行未核实均已登记；历史精确样本重叠unknown。可选设计核心未启用，新开放科学Y为0，最终比例／角色留W18。保护为流程／API层，非操作系统权限隔离；家系仅实有自报连接，不能据新划分宣称历史未接触验证。下一步W16。'
report=f'''# W15 全项目顺序、历史接触账本与访问策略

2026-10-29，运行 `w15_split_access_20261029`。**W15已完成构建、独立核验、发布和最终读回；五步实际退出码均为0。** Codex代执行及独立代码复验，未冒称甲／乙本人或额外分析助手人工复核已完成。

{status}

## 本轮交付

| 项目 | 核验结果 |
|---|---:|
| W14基线候选并集人数 | 6,305 |
| 全项目个人／已知家系随机单元 | 5,699 |
| 多候选人单元 | 529 |
| 上述单元内候选人数 | 1,135 |
| 单候选人单元 | 5,170 |
| 最大候选人单元人数 | 16 |
| 已知完整连通分量大于1的候选人数 | 1,440 |
| W11家庭／同住连接记录 | 1,780 |
| 完整图节点／与候选有关的图节点 | 11,269／6,726 |
| 模块／模块成员记录 | 11／54,151 |
| 历史接触事件／声明主体 | 8／4 |
| 正式产物 | 10 |
| 设计核心人数／新开放科学Y／最终已分配角色 | 0／0／0 |

54,151是人×模块成员记录数，不是独立人数或模型N。设计核心未启用，最终发现／确认比例与角色均未分配；不提前固定7:3。当前统一状态为 `unopened_pending_design_freeze`。

## 随机顺序与家庭一致性

种子 **`W15_global_person_family_order_20261029_v1`** 在候选成员投影、分组和排序前单独登记；没有搜索种子，也没有沿用W12样本选择种子。全量W11自报家庭／同住边先构图，再取W14候选；非候选节点形成的连接保留，避免把经第三人相连的候选拆开。节点以`cohort + participant_id`命名，当前候选均为10k、00_00_visit；不能把不同cohort的相同数字ID擅自视为同人。

完整分量成员排序序列生成SHA256标识，再以 `SHA256(seed|component_id)` 排序；同单元内个人顺序也以种子哈希确定。同一人的全部模块、日期、暴露和未来资格子集均继承这一顺序。家系信息新增导致冲突时先暂挂核查，不静默重随机。自报连接不代表完整遗传家系；没有已知边的单元不能被称为“已证明无亲缘”。

可选设计核心当前未启用。若W17需要实际设计摘要，须先固定整个单元的顺序前缀，登记人数、用途和名单并封存回执，再读取指定信息；已开放／训练者只留发现侧，不回流确认。最终发现／确认分配由W18冻结。

## 历史接触登记

| 主体／事件 | 如实登记 |
|---|---|
| 你（研究者甲） | 已有研究层面历史结果接触；具体逐项阅读范围及精确样本重叠不确定 |
| 同事（研究者乙） | 不确定，待本人确认；总脂／VAT职责不能证明其是否看过血糖结果 |
| AI分析助手 | 已读取历史探索报告，并用于方案讨论；包括已有W04账本中的早期报告第16／17节；精确ID重叠未知 |
| 其他会话／人工运行 | 未核实；本会话不能替这些活动作保证 |
| 已登记W14 | 数据整理、QC、技术性Y构造与候选队列；未训练模型、未运行本轮关联；不据此否认其他运行 |
| 既往NC-Diet／口腔饮食及HPP文献背景 | 已有方法与方案背景接触，具体结果／样本范围仍不确定 |

CSV中研究层面的历史事件不填写虚构的个体ID。每个候选人的历史精确重叠状态均保留unknown；W14技术处理的当前人员范围通过成员表关联。历史接触不会自动使后续研究无效，但重新划分同一批人不会消除已有接触。能否称独立验证，仍须核对具体样本、结果及用途；本轮没有确立“历史未接触验证集”。W15没有为补账而重新打开历史效应表。

## 访问策略及执行范围

`w15_access_guard.py` 默认只允许成员键、模块、顺序及当前角色等技术元信息。实际读回通过该接口逐一读取11模块的盲成员表；请求Y、未登记模块会失败。结局／暴露值、关联、损失、选择性图和真实监督训练请求被拒绝。发现／已开放设计核心回到确认侧的角色转换也被拒绝；W15接口即使收到伪造的后续回执也不会授权真实Y，后续须另行验收W18／W22读取器。

**这是流程／API层控制；没有改变原账户及源表的文件系统权限，也没有做加密隔离。** 后续正式脚本必须接入受控读取器和冻结回执，不能把这次交付解释为其他人已无法直接读取源文件。W15按白名单分批投影来源元信息，不读科学Y／暴露值列；完整性哈希读取文件字节属于技术处理，不是查看统计结果。原始个人数据、成员名单及家系映射均留TRE，本地仅保存通用代码、聚合数及验收截图。

## 验收证据

| 阶段 | 实际结果 |
|---|---|
| 01 预登记种子与规则、核W14前置及读取预算 | exit 0 |
| 02 完整家系图、统一顺序、模块继承、8事件账本与访问接口 | exit 0；19项构建内检查 |
| 03 独立重建与验证 | exit 0；61项检查 |
| 04 正式发布与摘要哈希 | exit 0；10产物写入并读回 |
| 05 正式文件独立读回与真实盲接口验收 | exit 0；82项检查 |

独立核验以BFS代替构建时的union-find，批量127行，完整比较6,305人、54,151模块成员和全量分量；检查W11分区一致、乱序／反向／重复边不改变结果、非候选桥接及跨cohort命名合成例、多个假设切点／折的家庭一致性、unknown历史状态以及拒绝非法读取／角色回流。假设切点和折只是测试，没有产生正式角色／折。

最终读回验证10个正式产物的hash、W14全部36产物与四个W15输入hash、现行协议及分析登记未改；逐一调用盲成员读取器，并确认未登记模块和Y列被拒。源表未重写，无新模型／关联／损失或选择性图。最高已观察RSS：构建166,408 KiB，独立核验183,056 KiB。

模块成员数：{', '.join(k+'='+str(v) for k,v in counts.items())}。

聚合摘要采用排序键、紧凑JSON的SHA256；服务器显示值与本地独立重建完全相同：

`{digest}`

## 文件与接续

服务器项目根 `P={P}`；实际退出与本轮完整日志在 `{L}`。

| 远程产物（均相对P） | 用途 |
|---|---|
| `freeze/split_order_TRE_ONLY.parquet` | 6,305人统一顺序、家系单元与尚未开放状态 |
| `freeze/access_ledger_TRE_ONLY.csv` | 8条历史接触事件及证据范围 |
| `config/access_policy.yaml` | 已预登记规则；JSON写法为有效YAML |
| `freeze/module_membership_TRE_ONLY.parquet` | 11模块继承的顺序／角色 |
| `freeze/known_related_membership_TRE_ONLY.parquet` | 6,726个相关完整图节点含非候选桥接者 |
| `freeze/person_access_status_TRE_ONLY.parquet` | 每人历史unknown与当前科学访问状态 |
| `freeze/design_core_receipt.json` | 可选设计核心未启动回执 |
| `scripts/w15_split_access_20261029/w15_access_guard.py` | 实际验收的受控盲成员读取器 |
| `config/user_history_declaration_W15_20261029.json` | 用户本轮声明与解释边界 |
| `data_qc/w15_split_access_20261029/read_projection_manifest.json` | 只读元信息列的记录 |
| `config/w15_split_access_lock.json` | 版本、种子、输入与产物hash指针 |
| `data_qc/w15_split_access_20261029/` | 预登记、独立验证、正式清单、聚合摘要及完成记录 |
| `memos/w15_split_access_20261029/MEMO.md` | 服务器交接说明 |
| `W15_HANDOFF.md`、`W15_COMPLETED.md` | 接续入口和完成标记；须同时核05实际exit0 |

本地[功能备忘录](../audits/w15_split_access_2026-10-29/MEMO.md)、[聚合缓存](../audits/w15_split_access_2026-10-29/observed_aggregate.json)、[状态回执](../config/w15_split_access_20261029/execution_status.json)。[最终验收截图](../audits/w15_split_access_2026-10-29/05_final_readback_exit0.jpg)。

W15完成表示基础候选数据与全项目顺序／接触管理接口已交付；正式分析还需W16软件接口、W17设计验收和W18冻结。下一包W16未执行。本轮不改变W14已确认的三套用药研究窗口，正式选择仍留W18。
'''
rp=W/'reports/HPP_W15_Split_Access_Ledger_2026-10-29.md';rp.write_text(report)
memo=f'''# W15 代码与接续备忘

{status}

运行 `w15_split_access_20261029`，pheno，经Chrome WorkSpaces UI传输并执行；全部脚本传输前本地compile及SHA256，远程解压、hash、compile后保存到独立目录再执行，launcher保存实际退出码及关闭日志。未用SSH或服务器API。

| 本地／服务器同名脚本 | 功能 |
|---|---|
| `01_preflight_commit.py` | 前置核验、种子与规则先行登记、用户声明、schema预算和输入hash |
| `02_build_order_and_ledger.py` | full-graph union-find、统一单元／人顺序、模块继承、历史账本、访问guard、19项检查 |
| `03_independent_verify.py` | full-graph BFS、完整键集和61项独立核验 |
| `04_publish.py` | 10产物正式发布、锁和聚合摘要、memos；拒绝覆盖旧文件 |
| `05_final_readback.py` | 正式hash／源未变、实际盲读取器11模块、82检查、完成回执 |
| `prepare_transfer.py` | 通用本地compile/hash/zlib-base64及浏览器终端launcher命令 |
| `build_local_delivery.py` | 按已见屏幕重建聚合摘要，核hash；本地报告／进度完整写回 |
| `validate_local_delivery.py` | 本地交付与状态一致性核验 |

五步实际退出0，各有截图。服务器脚本 `P/scripts/w15_split_access_20261029`；正式结果 `{Q}`，运行日志 `{L}`；P=`{P}`。代码与个体表不混放；原始数据只读。CSV历史记录是研究层面声明，不伪造具体ID。

重要边界：5,699是已知连接单元；5,170单候选人单元并非已证无亲缘。历史精确ID重叠unknown，同事本人未确认，其他会话未核实。设计核心0、新科学Y0、正式角色0、比例null。流程guard并非OS隔离。W14全部36产物、四个输入及现行protocol／analysis_register的hash未变。三套用药窗口不改，W18决定。

聚合SHA256 `{digest}`；本地observed_aggregate与屏幕相符，未取回个体表。下一W16，W17可选设计核心须先名单／用途／数量回执再开放，W18最终角色和比例，W22全局确认访问。现有日志中的完成标记只有合并05实际exit0才成立；05退出码在runtime由外层launcher落盘。
'''
(D/'MEMO.md').write_text(memo)
# Standalone HTML with readable tables and preserved full prose, using a small deterministic renderer.
def inline(s):
 s=html.escape(s);s=re.sub(r'`([^`]+)`',r'<code>\1</code>',s);s=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',s);s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',s);return s
lines=report.splitlines();parts=[];table=False;para=[]
def flush():
 if para:parts.append('<p>'+inline(' '.join(para))+'</p>');para.clear()
for line in lines:
 if line.startswith('|'):
  flush()
  if not table:parts.append('<table>');table=True
  cells=[x.strip() for x in line.strip('|').split('|')]
  if all(re.fullmatch(r':?-+:?',x) for x in cells):continue
  parts.append('<tr>'+''.join('<td>'+inline(x)+'</td>' for x in cells)+'</tr>');continue
 if table:parts.append('</table>');table=False
 if not line.strip():flush();continue
 if line.startswith('#'):
  flush();n=len(line)-len(line.lstrip('#'));parts.append(f'<h{n}>'+inline(line[n:].strip())+f'</h{n}>')
 else:para.append(line)
flush()
if table:parts.append('</table>')
page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>W15 顺序与接触账本验收</title><style>body{font-family:system-ui,-apple-system,sans-serif;max-width:1080px;margin:40px auto;padding:0 24px;color:#183042;line-height:1.75;background:#f7f9fc}h1,h2{color:#0e5368}table{border-collapse:collapse;width:100%;margin:20px 0;background:white}td{border:1px solid #d6e1e8;padding:9px 12px;vertical-align:top}tr:first-child{font-weight:600;background:#e8f2f5}code{overflow-wrap:anywhere;font-size:.88em;background:#eaf0f3;padding:2px 4px}a{color:#096b90}p{overflow-wrap:anywhere}</style><main>'+''.join(parts)+'</main></html>'
hp=rp.with_suffix('.html');hp.write_text(page)
# Preserve current living documents and progress before any mutation.
paths=['reports/HPP_CURRENT_PLAN.md','reports/HPP_Team_Execution_Checklist_2026-10-26.md','reports/HPP_Project_TODO_v3_2026-10-23.md','reports/HPP_Project_Roadmap_v3_2026-10-23.md','HPP_LOCAL_VERIFIED_FACTS.md','HPP_PROJECT_BACKGROUND.md','HPP_DATA_BACKGROUND.md','HPP_SERVER_BACKGROUND.md','outputs/researchwrite/hpp_multisystem_20261021/team_checklist_20261026/workpackage_progress.json','reports/HPP_Team_Execution_Checklist_2026-10-26.html']
snap=D/'before_w15_update';snap.mkdir(exist_ok=False);snapmanifest=[]
for rel in paths:
 p=W/rel;dst=snap/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst);snapmanifest.append({'path':rel,'sha256':sha(dst)})
(D/'before_w15_manifest.json').write_text(json.dumps(snapmanifest,ensure_ascii=False,indent=2)+'\n')
for rel in paths[:8]:
 p=W/rel;s=p.read_text();s=s.replace('**当前实际进度（W14，2026-10-29）：**','**历史交付进度（W14，2026-10-29；当前见W15）：**').replace('**当前实际进度（W13，2026-10-29）：**','**历史交付进度（W13，2026-10-29；当前见W15）：**')
 link=('HPP_W15_Split_Access_Ledger_2026-10-29.md' if rel.startswith('reports/') else 'reports/HPP_W15_Split_Access_Ledger_2026-10-29.md')
 banner='> **当前实际进度（W15，2026-10-29）：** '+status+' [W15报告]('+link+')。\n\n'
 first=s.find('\n');s=s[:first+1]+'\n'+banner+s[first+1:].lstrip('\n')
 if rel=='reports/HPP_CURRENT_PLAN.md':s=s.replace('W04–W14已按各包范围交付，下一步W15；','W04–W15已按各包范围交付，下一步W16；')
 if rel=='reports/HPP_Team_Execution_Checklist_2026-10-26.md':
  s=s.replace('- [ ] **W15 建全项目统一角色与历史接触账本**（主责：乙；姓名：____）','- [x] **W15 建全项目统一角色与历史接触账本**（主责：乙；姓名：____）（Codex代执行；独立代码验收通过，人工复核待进行；最终角色留W18）')
  start=s.index('### W15 建全项目统一角色与历史接触账本');end=s.index('### W16 ',start);block=s[start:end];assert '**状态：** 待执行。' in block;block=block.replace('**状态：** 待执行。','**状态：** '+status+' [完整报告](HPP_W15_Split_Access_Ledger_2026-10-29.md)。\n\n**本次实施范围：** 可选设计核心未启用；后续需要时先固定全单元前缀及回执再开放。当前只建立盲成员接口和拒绝策略，没有文件系统权限隔离或新科学结局访问。');s=s[:start]+block+s[end:]
 s+='\n\n## 2026-10-29 W15正式交付\n\n'+status+'\n\n本次以[W15报告]('+link+')及TRE最终读回／05实际exit0为准；前文“下一步W15”等均为历史进度。研究者本人复核和历史精确样本核对未冒称完成。\n';p.write_text(s)
pp=W/paths[8];progress=json.loads(pp.read_text());progress['tasks']['W15']={'checked':True,'short_status':'统一顺序／历史账本／流程访问接口交付；最终角色W18','status_text':status,'what_ran':'01 precommit；02 full-graph union-find及19检查；03独立BFS61检查；04正式发布10产物；05读回82检查，全部实际exit0','code_location':P+'/scripts/w15_split_access_20261029','result_location':Q,'local_report':'reports/HPP_W15_Split_Access_Ledger_2026-10-29.md','human_review_complete':False,'final_role_freeze_complete':False,'design_core_activated':False,'historical_exact_ID_overlap':'unknown','access_enforcement':'workflow_API_guard_not_filesystem_ACL','aggregate_sha256':digest,'next_execution_note':'W16合成变换／估计接口；W17设计验收；最终比例与角色W18'}
progress['latest_run_id']=progress['latest_local_run_id']='w15_split_access_20261029';progress['latest_local_workpackage']='W15';progress['evidence']='audits/w15_split_access_2026-10-29/MEMO.md';progress['next_recommended_workpackage']='W16';progress['parallel_preparation']=['W17合成校准／设计验收准备；设计核心未开放'];progress['tasks']['W14']['next_execution_note']='W14已完成，W15顺序／账本也已交付；下一步W16';pp.write_text(json.dumps(progress,ensure_ascii=False,indent=2)+'\n')
# Add an authoritative current notice to the existing interactive checklist without changing its older content.
team=W/paths[9];ts=team.read_text();notice='<aside data-w15-current="true" style="margin:16px;padding:20px;border:2px solid #16788c;background:#edf8fa;color:#173e4c"><h2>2026-10-29：W15已验收，下一步W16</h2><p>'+html.escape(status)+'</p><a href="HPP_W15_Split_Access_Ledger_2026-10-29.html">W15完整验收报告</a><p>下方旧状态文字为历史记录；现行完成状态以共享workpackage_progress.json及本提示为准。</p></aside>'
ts=re.sub(r'(<body\b[^>]*>)',lambda m:m.group(1)+notice,ts,count=1,flags=re.I);team.write_text(ts)
(D/'local_delivery_manifest.json').write_text(json.dumps({'report_markdown_sha256':sha(rp),'report_html_sha256':sha(hp),'status_json_sha256':sha(C/'execution_status.json'),'aggregate_sha256':digest,'updated_files':paths,'before_snapshot':'before_w15_update'},ensure_ascii=False,indent=2)+'\n')
print('LOCAL_W15_DELIVERY_WRITTEN',digest,len(paths),flush=True)
