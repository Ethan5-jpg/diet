# -*- coding: utf-8 -*-
from pathlib import Path
import json,hashlib,shutil,re
root=Path.cwd();run=root/'audits/w13_microbiome_qc_2026-10-29'
agg={'candidate_rows':20547,'selected_rows':19059,'oral_selected_rows':8398,'oral_baseline_people':8291,'oral_taxa':737,'gut_selected_rows':10661,'gut_baseline_people':8690,'gut_taxa':2088,'baseline_joint_people':7228,'recovered_biological_candidate_rows':34,'selected_recovered_candidate_rows':17,'held_oral_candidate_rows':854,'held_gut_candidate_rows':615,'source_duplicate_flag_fail_rows':2,'independent_full_source_pass':True,'oral_null_contract_inferred':True,'no_model_fit':True}
digest=hashlib.sha256(json.dumps(agg,sort_keys=True,separators=(',',':')).encode()).hexdigest();assert digest=='bf928a386b378f565fa1b2f76c597d4e7fc2c3e7986d059ec498a54c585bb538'
obs={'date':'2026-10-29','provenance':'Manually transcribed approved aggregates from remote browser screenshots; no participant data exported','aggregate':agg,'canonical_sha256':digest,'matrix_scan':{'source_rows':20129,'raw_cells':30094618,'selected_cells_exactly_reverified':28449494,'identity_links_rebuilt':19078,'technical_metadata_cells_rechecked':51118,'native_recovered_cells_rechecked':70992},'baseline_joint_W12_identity_candidate_people':7214,'baseline_joint_W13_locked_people':7228,'actual_exits':{'01':0,'02':0,'03':1,'03b':0,'04':0,'05':0,'06':0,'07':0,'08':0},'failure':'03 bounded directory inventory timed out; exact-pointer 03b passed; originals preserved','no_independent_human_or_new_Opus_review_claimed':True}
(run/'observed_aggregate.json').write_text(json.dumps(obs,ensure_ascii=False,indent=2)+'\n')
P='/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021';rid='w13_microbiome_qc_20261029';report='reports/HPP_W13_Microbiome_QC_Sample_Lock_2026-10-29.md'
text=f'''# HPP W13：菌群矩阵质控与主样本锁定

2026-10-29。状态：已在当前TRE实跑、独立代码复核并发布后读回通过。Codex代执行，不冒称团队人工复核或新一轮Opus审查。未训练模型、未运行科学关联。

## 已同步的方案

执行前将`multisystem_v3_user_amendment_20261029`方案、登记表与访视政策逐字节核SHA后同步。六项肝表型各自正式分析；人级CGM与序列分析分开；用药按研究窗口三状态；跨域比较暂缓；UPF保留上下界与覆盖率替代，补齐NOVA不门控W13。旧W10/W11/W12带hash的配置保留，改用版本化覆盖配置及`P/config/ACTIVE_PROTOCOL.json`入口。01核验W12的13项产物hash及实际退出码，08再次核原始同步文件和保留配置hash。

这次同步没有重建新的CGM人级表型或研究窗口用药状态；这两项须在W14按新规则落地。

## 实际样本与人数

| 指标 | 口腔 | 肠道 |
|---|---:|---:|
| 原始丰度行 | 8,834 | 11,295 |
| 保留全部源物种列 | 737 | 2,088 |
| 候选行（含仅索引） | 9,252 | 11,295 |
| 身份可用候选行 | 8,398 | 10,680 |
| 身份暂挂候选行 | 854 | 615 |
| 身份可用但源技术标记失败 | 0 | 2 |
| 同组其他合格生物重复未入选 | 0 | 17 |
| 锁定人—cohort—stage—部位行 | 8,398 | 10,661 |
| 锁定样本对应不同人（所有源阶段） | 8,398 | 8,836 |
| **10k基线锁定人数** | **8,291** | **8,690** |

全部候选20,547行，主样本19,059行。基线双部位交集**7,228人**，W12仅身份候选为7,214人。二者范围／处理阶段不同，后者不再作当前锁定数。上述人数均未与饮食、特定结局和条件变量取交集，不是最终模型N。

口腔854暂挂由436丰度行访视证据不足、414无有效访视的仅索引候选、4有效访视但无丰度候选组成。436丰度行并非没有采样日期，仍缺唯一有效访视匹配。肠道615暂挂由414索引无人、185仅其他阶段、16重复标本身份未充分解决组成；不靠随机数修复连接。

## 技术核查与零值解释

完整扫描20,129丰度行、30,094,618物种单元：两部位均无负值、非空NaN/Inf、显式零和全零行；观察到的数值均为正，组成和约100。口腔组成和99.99991–100.00009；肠道99.9985–100.00017。源为百分比，确定性除100生成比例矩阵。数值闭合容差仅为源小数舍入误差`max(0.001,n_taxa×0.5e-5+1e-8)`个百分点。

口腔6,228,174、肠道21,563,957个Arrow NULL位于**已存在的稀疏组成谱内**。预先固定的3个肠道一对一样本与原始MetaPhlAn文本逐格核对，每个2,088物种均一致，共6,264格，其中5,822格是原文未报告的物种。原始表不改；派生视图把这种隐式未检出记0，同时保存每个原始单元空值掩码。

**口腔没有同等原始文本，不能宣称直接验证。**其零值编码是依据共同HPP聚合管线、全源稀疏模式、无非空非有限值及全谱闭合所作推定，单独登记为限制。0表示源谱未检出／未报告，不等于生物学不存在。缺失样本始终缺失，不制造全零行。后续如有反证须版本化重建并重验。

相对丰度百分比含义见[MetaPhlAn官方说明](https://github.com/biobakery/biobakery/wiki/MetaPhlAn-4.1)；官方说明本身不能证明HPP的NULL存储语义。

已关联口腔索引的raw read count均正；肠道raw／trimmed均正，trimmed不超过raw。肠道HPP已有`excessive_duplicate_reads`布尔标记2条为真，字典定义为绝对重复数超过1e5，按源技术失败排除。没有以分位数另定最低深度门槛，没有声称缺失的污染／口腔修剪信息通过QC。相同口腔丰度谱组未按跨人相似度自动去重。肠道16个全源未检出物种仍保留，检出率筛选留训练内。

## 重复标本恢复与随机锁定

50条肠道重复丰度记录全部调查。48条有唯一原始谱匹配，46条满足组内双射；仅34条还同时具备不同标本名称和实际采样日期的生物重复证据，恢复为候选。其余16条继续暂挂。另读3个固定零值验证谱，合计51个原始文本。

在身份及实有技术QC之后，按W12已固定的seed20261027与`SHA256(seed|site|candidate_id)`最小值选一。34条恢复候选中17条入选，17条同组其他生物重复不选。不按最近日期、Y、效应、窗口或模型改变样本；全模块复用同一锁。日期用于标本证据，没有新加lag准入门槛。`sample_id=candidate_id`为固定版本的丰度源行定位符，不冒充跨来源通用生物标本UUID。

## 独立验收及失败保留

06用不同batch127重读全部来源：独立重建19,078身份连接、51,118技术元数据单元，重新核34条恢复记录的70,992原始谱单元，逐格精确比较全部28,449,494入选丰度单元及相同数量原始空值掩码，重新哈希各组赢家。05另核合成失败出口及行序不变性。未读取饮食、CGM或其他Y数值来选样。

01、02、03b、04、05、06、07、08实际exit均0。03目录枚举碰到45秒保护而exit1，未覆盖；03b改查确切源指针通过。07正式发布，08独立核正式产物hash、行数、一对一样本／矩阵／掩码、锁定状态、配置与旧配置完整性。独立代码复核与人工审查分开记录。

聚合转写SHA256：`{digest}`。本地同schema重新计算与屏幕打印hash一致，详见[聚合缓存](../audits/w13_microbiome_qc_2026-10-29/observed_aggregate.json)。

## 代码与产物位置

P=`{P}`；run=`{rid}`。以下为TRE路径，不是本地可点击文件。

- Code：`P/scripts/{rid}/`；功能见[本地MEMO](../audits/w13_microbiome_qc_2026-10-29/MEMO.md)。
- Result：`P/data_qc/{rid}/`，包括完整source证据、原始QC、独立验收、final_artifact_manifest、aggregate_delivery及final_delivery_verification。
- 共享：`P/analysis/shared/sample_registry_TRE_ONLY.parquet`、`microbiome_sample_flags_TRE_ONLY.parquet`、`oral_selected_abundance_TRE_ONLY.parquet`、`gut_selected_abundance_TRE_ONLY.parquet`及两部位`source_null_masks`。
- 规则：`P/config/w13_sample_selection_lock.json`；入口`P/config/ACTIVE_PROTOCOL.json`；同步版本`P/config/protocol_amendment_20261029/`。
- QC／原因：`P/data_qc/microbiome_abundance_QC.csv`、`sample_selection_reasons.csv`；个体证据留TRE。
- 远程MEMO：`P/memos/{rid}/MEMO.md`；已关闭01–07日志及退出码保存于`P/logs/{rid}/`。08运行日志与实际退出码在L，最终读回结果已存P。
- Runtime L：`/home/ec2-user/multisystem_runtime_20261021/{rid}`。

## 接续W14

按结局分别建立候选E与人数流转，复用锁定O/G；六项肝表型分别准入，不能要求六项齐全。新建CGM人级访视汇总时不要求饮食共同日；日级序列保留真实日期对齐。明确研究窗口并保留用药／确认未用药／unknown，不把记录缺失当未用药。继续登记UPF上下界／覆盖率替代的模型接口，正式暴露与假设家族在W18/W22冻结。跨域比较暂缓。**W14尚未执行，未生成新的最终分析N。**
'''
(root/report).write_text(text)
memo='''# W13 本地代码与接续MEMO

本地仅有自编代码、聚合转写、截图与规则，不保存个体数据。完整结果见 ../../reports/HPP_W13_Microbiome_QC_Sample_Lock_2026-10-29.md。

| 脚本 | 实际功能／状态 |
|---|---|
| 01_sync_and_preflight.py | 同步用户修订的原文bytes及hash，保留旧配置，核W12，源schema／预算；exit0 |
| 02_scan_and_identity_probe.py | 全量数值QC、索引元数据与初步指针检查；exit0 |
| 03_resolve_storage_and_specimen_evidence.py | 有界目录枚举超时保护；exit1留存 |
| 03b_resolve_exact_source_evidence.py | 确切来源指针／字典与存储证据；exit0 |
| 04_match_native_profiles.py | 固定3谱空值核对、重复原始谱身份恢复；exit0 |
| 05_build_provisional_qc_and_selection.py | 结局盲QC、seed哈希选样、比例矩阵与原空值掩码；exit0 |
| 06_independent_source_and_selection_verify.py | 全源独立重扫、身份／元数据／选中每格／掩码／赢家重建；exit0 |
| 07_publish_locked_registry.py | 正式发布锁定registry、flags、矩阵、配置、QC／原因CSV、manifest及远程MEMO；exit0 |
| 08_final_readback.py | 正式产物hash／一对一／新旧配置／聚合hash读回；exit0 |
| prepare_transfer.py | 本地源码compile、压缩传输命令；远程SHA与syntax核验后才写独立脚本并运行，保存真实退出码 |
| record_local_delivery.py | 根据屏幕聚合及同schema哈希比对记录本地报告与待办；不接触服务器 |

按HPP browser skill操作，仅Chrome→WorkSpaces UI传入运行；pheno环境、4GiB地址空间保护、矩阵分批读取。失败源码／日志保留，未更改原始源。source_manifest_NN.json记录传入源码哈希；NN_command.txt是自编代码传输包装，无受试者信息。正式服务器路径与下一步见报告。

本地下一步：W14按用户修订构建各结局候选表，未知用药保持unknown；旧共同日CGM接口不能改名冒充新访视表。共享菌群锁在所有结局复用。原始数据与个体证据仅TRE。
'''
(run/'MEMO.md').write_text(memo)
# Snapshot existing living documents, then make dated additions and targeted current-status replacements.
paths=['reports/HPP_CURRENT_PLAN.md','reports/HPP_Team_Execution_Checklist_2026-10-26.md','reports/HPP_Project_TODO_v3_2026-10-23.md','reports/HPP_Project_Roadmap_v3_2026-10-23.md','HPP_LOCAL_VERIFIED_FACTS.md','HPP_PROJECT_BACKGROUND.md','HPP_DATA_BACKGROUND.md','HPP_SERVER_BACKGROUND.md','outputs/researchwrite/hpp_multisystem_20261021/team_checklist_20261026/workpackage_progress.json']
for rel in paths:
 p=root/rel;b=run/'before_w13_update'/rel;b.parent.mkdir(parents=True,exist_ok=True);assert not b.exists();shutil.copyfile(p,b)
summary='2026-10-29 W13已在TRE完成并独立读回验收：先同步用户修订的版本化配置，再完整核20,129丰度行、30,094,618物种单元，锁定口腔8,398／肠道10,661人—访视—部位行。基线O8,291、G8,690、双部位7,228人，非饮食／结局模型N。恢复34条有原始谱和不同采样日证据的肠道生物重复候选，17条入选；2条源重复reads异常排除，身份暂挂O854／G615。口腔隐式零编码是推定，非直接原文验证；原始空值掩码保留，无全源物种筛选、模型或关联。01/02/03b/04/05/06/07/08实际exit0，03目录保护exit1留存。下一步W14：六肝表型分别建表、人级CGM独立访视汇总、研究窗口用药三状态；跨域比较暂缓。'
for rel in paths:
 if rel.endswith('.json'):continue
 p=root/rel;s=p.read_text();link=('HPP_W13_Microbiome_QC_Sample_Lock_2026-10-29.html' if rel.startswith('reports/') else report)
 # Add authoritative newest status before old dated history.
 pos=s.find('\n')+1;s=s[:pos]+f'\n> **当前实际进度（W13，2026-10-29）：** {summary} [完整报告]({link})。\n'+s[pos:]
 if rel=='reports/HPP_CURRENT_PLAN.md':
  s=s.replace('W04–W12已按各包范围交付，下一步W13','W04–W13已按各包范围交付，下一步W14').replace('W13未执行，模型与统计校准未运行','W13已按实有QC范围完成，模型与统计校准未运行')
  s+='\n\n## 2026-10-29 W13正式交付\n\n'+summary+f'\n\n代码`{P}/scripts/{rid}`；Result`{P}/data_qc/{rid}`；MEMO`{P}/memos/{rid}/MEMO.md`；当前入口`P/config/ACTIVE_PROTOCOL.json`，锁`P/config/w13_sample_selection_lock.json`。详见[W13报告]({link})及[本地代码MEMO](../audits/w13_microbiome_qc_2026-10-29/MEMO.md)。此前“尚未执行／仅本地修订／未再连接TRE”为对应日期历史状态，本节已取代。\n'
 if rel=='reports/HPP_Team_Execution_Checklist_2026-10-26.md':
  s=s.replace('- [ ] **W13 验收 O/G 矩阵和实有技术质量**（主责：乙；姓名：____）','- [x] **W13 验收 O/G 矩阵和实有技术质量**（主责：乙；Codex代执行；独立代码复核通过，人工复核未冒称完成）')
  entry=f'\n  - 跑了什么：01同步配置、02全源QC、03目录保护失败留存、03b确切指针、04原始谱证据、05选样与矩阵、06独立复核、07发布、08最终读回；成功脚本均实际exit0。\n  - 代码位置：`{P}/scripts/{rid}/`；[本地功能MEMO](../audits/w13_microbiome_qc_2026-10-29/MEMO.md)。\n  - Result位置：`{P}/data_qc/{rid}/`；共享registry／flags／矩阵／掩码在`P/analysis/shared`；远程MEMO`P/memos/{rid}/MEMO.md`；[结果与人数]({link})。\n'
  anchor='- [ ] **W14';i=s.find(anchor,s.find('- [x] **W13'));assert i>0;s=s[:i]+entry+s[i:]
  i=s.index('### W13 验收 O/G 矩阵和实有技术质量');j=s.index('### W14',i);part=s[i:j].replace('**状态：** 待执行。','**状态：** '+summary+f' 代码／产物／完整说明见[W13报告]({link})。');s=s[:i]+part+s[j:]
 p.write_text(s)
p=root/paths[-1];d=json.loads(p.read_text());d.update({'date':'2026-10-29','latest_run_id':rid,'latest_local_workpackage':'W13','latest_local_run_id':rid,'evidence':'audits/w13_microbiome_qc_2026-10-29/MEMO.md','next_recommended_workpackage':'W14','TRE_sync_pending_for_user_amendment':False,'TRE_sync_receipt':P+'/data_qc/'+rid+'/protocol_sync_receipt.json','TRE_current_protocol_pointer':P+'/config/ACTIVE_PROTOCOL.json','parallel_preparation':['W15角色与历史接触账本']});d['tasks']['W13']={'checked':True,'short_status':'实有矩阵QC与主样本锁定完成；口腔零编码推定及不可测QC限制已记','status_text':summary,'what_ran':'01—08（03目录超时保护，03b确切指针替代）；06独立全源与每个入选单元复核，08正式产物读回。','code_location':P+'/scripts/'+rid,'result_location':P+'/data_qc/'+rid,'local_report':report,'human_review_complete':False};d['tasks']['W12']['short_status']='W12候选与规则完成；W13已随后完成QC／锁定，旧选定0仅指W12时点';d['tasks']['W14']['checked']=False;d['tasks']['W14']['next_execution_note']='配置已同步、菌群已锁；按新CGM访视表型与用药三状态分别建表，不把W10旧共同日接口改名。';p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
(run/'local_update_paths.json').write_text(json.dumps(paths+[report],ensure_ascii=False,indent=2)+'\n')
print('RECORDED',digest)
