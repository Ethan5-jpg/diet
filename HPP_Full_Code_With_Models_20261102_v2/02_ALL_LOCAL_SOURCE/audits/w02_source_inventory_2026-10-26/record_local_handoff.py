"""Record manually observed aggregate W02 evidence; never access remote data."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
A = Path(__file__).resolve().parent
D = ROOT/'outputs/researchwrite/hpp_multisystem_20261021/team_checklist_20261026'
P = '/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021'
RUN = 'w02_sources_20261026T130908Z'
CODE = P+'/scripts/'+RUN
M = P+'/manifests/'+RUN
Q = P+'/data_qc/'+RUN
MEMO = P+'/memos/'+RUN+'/MEMO.md'
GUT = '/home/ec2-user/studies/hpp_datasets/gut_microbiome/abundance/metaphlan4/aggregated/metaphlan_abundance_species.parquet'

def jwrite(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')

def main():
    # Preserve the prior document versions; never overwrite these snapshots.
    snap = A/'before_w02_update'
    snap.mkdir(exist_ok=False)
    names = ['HPP_SERVER_BACKGROUND.md', 'HPP_DATA_BACKGROUND.md', 'HPP_PROJECT_BACKGROUND.md', 'reports/HPP_CURRENT_PLAN.md', 'reports/HPP_Team_Execution_Checklist_2026-10-26.md']
    before = []
    for name in names:
        data = (ROOT/name).read_bytes()
        target = snap/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        before.append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'snapshot': str(target.relative_to(ROOT))})
    jwrite(snap/'manifest.json', before)
    aggregate = {
        'date': '2026-10-26', 'run_id': RUN, 'provenance': 'Manually transcribed from observed TRE terminal output; not downloaded participant data or remote manifest',
        'host': 'ip-10-0-91-109.ec2.internal', 'source_files': 26, 'metadata_sources_read': 26,
        'historical_metadata_matches': 8, 'historical_compared': 8, 'dictionary_files': 29,
        'dictionary_term_match_candidates': 991, 'cgm_fragment_files': 9497, 'cgm_representative_metadata_read': 3,
        'participant_rows_read': 0, 'artifact_hashes_verified': True, 'artifacts_verified': 14,
        'initial_collector_actual_exit': 2, 'targeted_correction_actual_exit': 0, 'finalizer_actual_exit': 0,
        'gut_species_correct_path': GUT, 'code_directory': CODE, 'manifest_directory': M,
        'result_directory': Q, 'remote_memo': MEMO,
        'w01': 'full_rerun_waived_by_user_existing_environment_reused',
        'data_QC_or_cohort_N_or_models_or_split_performed': False,
        'proof_screenshot': 'audits/w02_source_inventory_2026-10-26/verification_screen.png',
        'interpretation': '26 files, not 26 cohorts; 991 text matches, not 991 adopted features; footer/metadata consistency does not establish unchanged data-page values.'}
    assert (A/'verification_screen.png').is_file()
    jwrite(A/'observed_aggregate.json', aggregate)
    progress = {'date': '2026-10-26', 'source': 'TRE execution; Codex via authorized HPP browser skill',
        'verified_storage': True, 'run_id': RUN, 'evidence': str((A/'MEMO.md').relative_to(ROOT)), 'tasks': {
        'W01': {'checked': False, 'short_status': '用户指定沿用，免完整重跑，不阻塞W02',
            'status_text': '用户指定沿用原服务器，完整W01免重跑；仅核当前主机／路径及W02输出写入读回。未新做完整环境基准、容量验收或历史结论继承矩阵。',
            'what_ran': '沿用既有环境；W02中核当前主机、Python及路径、输出关闭后读回。未完整执行W01。',
            'code_location': '`'+CODE+'/02_source_inventory.py`（W02中的最小环境记录）；没有新增完整W01脚本。',
            'result_location': '`'+M+'/environment.json`；W02验收记录 `'+Q+'/verification.json`。'},
        'W02': {'checked': True, 'short_status': '元信息清单已完成；Codex代甲执行，同事尚未人工复核',
            'status_text': '2026-10-26元信息清单与读取计划已运行验收：26/26来源；8/8历史元信息匹配；纠正gut物种表路径。首次退出2保留，单源修正与最终核验实际均退出0；未读个体行。临床语义、QC、合格N仍待后续任务；乙尚未人工复核。',
            'what_ran': '`02_source_inventory.py`：schema／字典／历史元信息比较及读取预算；`03_fix_gut_source_and_verify.py`：修正肠道文件路径并单源复验；`04_finalize_w02.py`：保存、读回、hash和交接核验。',
            'code_location': '`'+CODE+'`；可复用收集器为其中 `02_source_inventory_corrected.py`。',
            'result_location': '`'+Q+'`（REPORT、source_delta、verification）；来源清单 `'+M+'/source_manifest.json`；读取计划 `'+P+'/config/'+RUN+'/read_plan.yaml`；备忘录 `'+MEMO+'`。'}}}
    jwrite(D/'workpackage_progress.json', progress)
    (D/'execution_update.md').write_text('''## 当前实际执行进度：2026-10-26

**W01按用户要求沿用现有服务器，免完整重跑；W02来源元信息清单已完成。**W01在总表保留空框并明确“免重跑”，不作为阻塞或虚构完整验收。W02由Codex通过HPP Server Skill在TRE执行，乙尚未人工复核。后文启动步骤为原始排程，实际状态以本节和工作卡状态为准。

- 26个来源文件均成功读取schema／footer；8个可比历史来源的大小、修改时间、元信息行数及既有投影列存在性匹配。不是参与者数据值全量比对。
- 盘点29个字典文件，产生991条关键词候选匹配；这些不是已验收变量。CGM目录9497个片段，仅3个代表文件读了元信息，仍未完成点级QC。
- 已保存来源清单、字段候选、逐来源读取预算及MEMO；P/L已实际创建并写入读回。
- **路径修正：gut物种表实际文件名为 `metaphlan_abundance_species.parquet`。**后续使用本轮清单，不复制旧R16的 `abundance_species.parquet`。
- 本轮读取参与者数据行0条；没有新的人数、饮食特征、队列、发现／确认分组或模型结果。接下来可做W03身份／join契约及W04特征定义；W05再核营养素语义。

本地代码与[执行备忘录](../audits/w02_source_inventory_2026-10-26/MEMO.md)、[聚合报告](../audits/w02_source_inventory_2026-10-26/REPORT.md)；具体服务器代码／Result位置已填入下方总表。科学方案与分工保持V3。
''')
    report = f'''# W02 来源盘点：执行结果

2026-10-26。运行 `{RUN}`；当前同一主机 `ip-10-0-91-109.ec2.internal`。W01由用户明确免完整重跑，沿用既有服务器；W02已通过HPP浏览器技能实际执行。

| 项目 | 实测结果 | 含义 |
|---|---:|---|
| 计划／候选来源Parquet | 26/26成功 | 15个核心来源加11个协变量候选文件；不是26队列 |
| 可对照历史元信息 | 8/8匹配 | bytes、mtime_ns、rows_metadata及原投影字段存在性，不保证数据页内容未变 |
| 字典文件盘点 | 29 | 界定大小／深度后盘点，并非所有词典全覆盖 |
| 关键词匹配候选 | 991 | 文本匹配候选，不是991个已选／有效变量 |
| CGM序列目录片段 | 9497 | 仅3个代表文件footer/schema，不是9497份质量通过 |
| 参与者数据行读取 | 0 | 没有重算人数、做数据QC或看关联 |
| 输出文件hash核验 | 14个通过 | 不等同于输入数据页的完整hash |
| 首次收集退出码 | 2 | 老gut文件名错误，保留失败证据 |
| 单源修正／最终核验退出码 | 0／0 | 实际退出文件已通过终端读回 |

真实肠道物种路径：

`{GUT}`

旧V3路径R16遗漏 `metaphlan_` 前缀。本轮只修正该来源，没有重扫其余25个成功来源。旧失败清单／报告留作 `.initial_partial.*`，不覆盖失败事实。

## 结果在哪

以下为服务器TRE路径，不是Mac本地路径。

| 用途 | 位置 |
|---|---|
| 全部源码 | `{CODE}` |
| 现行来源清单 | `{M}/source_manifest.json` |
| 汇总及hash清单 | `{M}/summary.json`、`artifact_manifest.json` |
| 环境最小记录 | `{M}/environment.json` |
| 增量读取计划 | `{P}/config/{RUN}/read_plan.yaml` |
| 报告／来源差异／验证 | `{Q}` |
| 字典候选（仅TRE） | `{Q}/dictionary_candidates_TRE_ONLY.json` |
| 远程备忘录 | `{MEMO}` |
| 02及03实际日志／退出记录 | `{P}/logs/{RUN}` |
| 项目交接入口 | `{P}/W02_HANDOFF.md`、`PROJECT_EXECUTION_STATE.json` |
| 服务器运行暂存 | `/home/ec2-user/multisystem_runtime_20261021/{RUN}` |

当前完成范围是“来源已定位、schema可读取、历史结构比较与读取计划已保存”。食品ID/NOVA覆盖、事件营养素单位、进食时间、身份连接、协变量编码与可用率、家系ID、O/G样本QC、CGM点级覆盖、各域真实N仍需后续任务验收。family_history不能直接当家庭聚类ID。已有临床日期／人数仍引用2026-10-18核查，不升级为本轮新QC结论。

W01完整包、磁盘容量基准及继承矩阵没有新做；本轮仅核当前主机／路径、使用既有pheno及W02产物保存读回。未训练模型、没有冻结实际发现／确认名单、没有查看确认关联。

证据来自终端可见的聚合输出，未下载参与者数据或TRE字典／schema文件。详见[MEMO](MEMO.md)及[验收截图](verification_screen.png)。
'''
    (A/'REPORT.md').write_text(report)
    hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(A.glob('*.py'))}
    memo = f'''# W02 代码与执行备忘录

日期：2026-10-26。用户要求调用HPP Server Skill实际运行W02；明确同一服务器W01可免完整重跑。Codex执行，没有新调用Astra，没有冒称同事已复核。

## 实际做了什么

通过Chrome WorkSpaces既有会话、远程MATE Terminal、现有pheno Python执行。仅来源目录、schema/footer、字典及历史manifest；参与者行读取0，原始数据只读。源码在本机编写，经浏览器剪贴板送入TRE，编译／SHA256核对后用指定pheno运行；没有SSH、原始数据下载或新装依赖。

当前P：`{P}`，L：`/home/ec2-user/multisystem_runtime_20261021`，均实际创建并完成W02文件全写／关闭读回。原始根R：`/home/ec2-user/studies/hpp_datasets`。

## 每份代码的功能、结果与修改入口

1. `02_source_inventory.py`：初始收集器。15核心＋11候选来源schema、footer指纹、候选字段、逐列读取预算；对照2026-10-18历史8来源；有限字典盘点；CGM目录枚举及3个footer。每个worker串行，60秒超时、512MiB RSS、2GiB地址空间、8MiB footer预算。**实际退出2**：25来源成功，老gut路径不存在。
2. `02_source_inventory_corrected.py`：只修改gut文件常量后的现行收集器；用于将来新run。**本轮未完整重跑该脚本main**，03只复用其单来源函数。后续增改来源改CORE；候选字段改IDENTITY/FOOD/TERMS；预算改guarded_meta/read_plan构造，先核内存。
3. `03_fix_gut_source_and_verify.py`：从终端观测真实同目录文件后，只复验正确gut来源；保存原partial证据，更新来源／预算／差异／summary，校验11份文件。**实际退出0**。此脚本只针对本次run，不当通用pipeline重跑。
4. `04_finalize_w02.py`：检查03实际退出0，重新验证文件hash，保存02/03日志和退出记录、追加MEMO，写项目状态与handoff。**实际退出0**，14份输出hash通过。完成实际退出在 `/tmp/hpp_w02_final_cb3a98f2.exit`，终端已读回；04临时console为 `/tmp/hpp_w02_final_cb3a98f2.log`。本地截图保存完整聚合结果。
5. `record_local_handoff.py`：本机文档登记脚本；只把终端已观察的聚合数和代码位置登记到本地REPORT／MEMO／progress和四份背景，不访问TRE或读取原始数据。运行一次，旧文档快照保留在 `before_w02_update/`。

本轮远程调用方式为 `{('/home/ec2-user/miniconda/envs/pheno/bin/python')} -u /tmp/hpp_w02_b1da099d.py`（02），同解释器 `/tmp/hpp_w02_fix_368e34b9.py`（03）、`/tmp/hpp_w02_final_cb3a98f2.py`（04），由小型subprocess启动器将stdout/stderr与实际退出码分别写 `/tmp`。其持久源码副本位于 `{CODE}`；原collector的源码哈希在部署前后核对。路径细节及远程产物见[REPORT](REPORT.md)。

## 结果与边界

26/26来源metadata成功，8/8历史元信息匹配；29字典文件、991文本候选；9497CGM片段、3代表footer；参与者行0；最后14份输出hash通过。来源schema可读不等于单位已验收、QC通过或最终N可用。不同队列不取全域交集、O/G same-visit准入等V3规则未改。

旧gut R16漏前缀，现为 `{GUT}`。旧文档保留历史依据，本轮manifest优先。初始失败并非数据丢失。旧来源大小/mtime/metadata行数一致也不能排除原始数据内容变化，不据此把旧人数当新版最终样本数。

W01完整重跑已由用户免除；只记录最小当前环境与W02写读通过，未虚构全部历史继承表。当前执行者为Codex，乙人工复核仍待。W03身份／join及W04定义可接续；W05核营养素语义后再生成正式饮食暴露。本轮没有开展这些后续任务。

## 源码hash

'''
    memo += '\n'.join('- `'+name+'`：`'+value+'`' for name,value in hashes.items())+'\n'
    memo += '\n原始与修正源码保留；不要将本次03/04脚本直接用于别的run。新运行应给新run_id，保留旧记录。数据和字典留TRE，本机只有自编代码、手工转记聚合数、备忘录与不含个体的验收截图。\n'
    (A/'MEMO.md').write_text(memo)
    background_section = f'''\n\n## 2026-10-26 W02实际执行更新（覆盖旧“尚未运行”状态）

W01按用户要求沿用现有服务器，免完整重跑；本轮只核当前主机／路径及W02输出写入读回。通过HPP浏览器技能实际完成W02元信息盘点，运行 `{RUN}`。P=`{P}`，L=`/home/ec2-user/multisystem_runtime_20261021` **现已实际创建并写入读回成功**；旧“候选／尚未创建”为历史记录。

26/26来源schema/footer读取成功；8/8可比历史来源元信息匹配。29字典文件、991关键词候选；CGM目录9497片段，仅3个代表footer。14份输出hash通过，单源修正与最终核验实际退出0；首次退出2保留。**本轮参与者数据行读取0；没有新合格N、个体QC、角色分组或模型。**既有人数与真实日期结论仍截止2026-10-18，元信息匹配不证明数据页内容全同。

**实源修正：**gut物种表应为 `{GUT}`。旧V3 R16的 `abundance_species.parquet` 不存在，遗漏了 `metaphlan_` 前缀；以后以本轮来源manifest为准，不静默改写旧失败记录。

服务器代码 `{CODE}`；来源清单 `{M}/source_manifest.json`；读取计划 `{P}/config/{RUN}/read_plan.yaml`；Result `{Q}`；MEMO `{MEMO}`；根交接 `{P}/W02_HANDOFF.md`。本机自编代码和聚合说明：`audits/w02_source_inventory_2026-10-26/MEMO.md`，不含个体数据。W02已定位字段候选，不代表食品单位、FoodID/NOVA、family ID、进食时间、CGM点级或菌群技术QC已验收。下一步W03身份／join、W04定义、W05营养素语义；未擅自执行。
'''
    # Append-only keeps the previously recorded source bytes verifiable.
    for name in names[:4]:
        with (ROOT/name).open('a') as f:
            f.write(background_section)
    jwrite(A/'background_append_manifest.json', {'date':'2026-10-26', 'scope':'Append-only dated W02 metadata execution; preserve prior bytes and all V3 scientific source files', 'records':[dict(x, after_sha256=hashlib.sha256((ROOT/x['path']).read_bytes()).hexdigest()) for x in before[:4]]})
    state = json.loads((D/'state.json').read_text())
    state.update(stage='W02_source_inventory_complete_scientific_analysis_pending', tre_accessed=True,
                 metadata_inventory_performed=True, workpackage_progress='workpackage_progress.json',
                 W01='user_waived_full_rerun_existing_server_reused', W02='source_inventory_complete',
                 source_inventory_run_id=RUN, participant_rows_read_this_run=0,
                 scientific_tasks_executed_this_turn=False, participant_roles_frozen=False)
    jwrite(D/'state.json', state)
    for name in ['MEMO.md','review.md','00_scope.md','01_research_canon.md','02_evidence_table.md','03_argument_map.md','04_section_contracts.md']:
        with (D/name).open('a') as f:
            f.write('\n\n## 同日后续：W02已实际运行\n\n用户授权HPP Server Skill执行，W01免完整重跑，沿用同一服务器。W02的26来源元信息盘点和读取计划已通过，未读参与者行，未做科学模型或分组。实际进度源为 `workpackage_progress.json`；前文“没有TRE访问／所有任务未执行”描述的是初版文档构建时点，不能当作最新状态。详见 `audits/w02_source_inventory_2026-10-26/MEMO.md`。四份背景只追加有日期的新事实；核心V3设计保持。\n')
    print('Local W02 evidence, progress, snapshots, and append-only background updates recorded.')

if __name__ == '__main__':
    main()
