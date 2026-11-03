"""Archive real Claude results and expose conditional acceptance without changing TRE artifacts."""
from pathlib import Path
import json,hashlib,datetime,shutil,re,html
D=Path(__file__).resolve().parent;W=D.parent.parent;T=Path('/private/tmp/hpp_claude_w08_w15_20261029');B=T/'packet';tags=['A_W08_W10','B_W11_W13','C_W14_W15','D_CONSOLIDATED'];reviews=[];checks=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ok(n,x):
 assert x,n
 checks.append(n)
for tag in tags:
 meta=json.loads((T/(tag+'_invocation.json')).read_text());result=json.loads((T/(tag+'_result.json')).read_text());ok('actual_exit_'+tag,meta['actual_exit_code']==0 and (T/(tag+'_actual_exit_code.txt')).read_text().strip()=='0');ok('successful_result_'+tag,result['type']=='result' and result.get('subtype')=='success' and not result.get('is_error'));ok('permission_denials_'+tag,not meta.get('permission_denials'))
 events=[]
 for line in (T/(tag+'_transcript.jsonl')).read_text().splitlines():
  try:events.append(json.loads(line))
  except ValueError:pass
 init=next(e for e in events if e.get('type')=='system' and e.get('subtype')=='init');calls=[c for e in events if e.get('type')=='assistant' for c in e.get('message',{}).get('content',[]) if c.get('type')=='tool_use'];reads=[c.get('input',{}).get('file_path') for c in calls if c.get('name')=='Read'];ok('read_only_tools_'+tag,set(init['tools'])<=set(['Read','Glob','Grep']) and all(c.get('name') in ['Read','Glob','Grep'] for c in calls));ok('read_paths_in_packet_'+tag,all(p and str(Path(p).resolve()).startswith(str(B.resolve())+'/') for p in reads));ok('raw_answer_preserved_'+tag,(T/(tag+'_review.md')).read_text()==result['result']+'\n')
 files=[]
 for suffix in ['_prompt.txt','_invocation.json','_transcript.jsonl','_result.json','_review.md','_actual_exit_code.txt','_stderr.log']:
  p=D/(tag+suffix);ok('artifact_'+tag+suffix,p.exists() and sha(p)==sha(T/(tag+suffix)));files.append({'name':p.name,'sha256':sha(p),'bytes':p.stat().st_size})
 reviews.append({'tag':tag,'actual_exit_code':0,'result_subtype':result['subtype'],'model':init['model'],'tools':init['tools'],'tool_calls':len(calls),'read_calls':len(reads),'unique_files_read':len(set(reads)),'review_sha256':sha(T/(tag+'_review.md')),'artifacts':files})
source_manifest=json.loads((T/'input_manifest.json').read_text());ok('base_input_count',len(source_manifest)==153)
for m in source_manifest:ok('base_unchanged_'+m['path'],sha(B/m['path'])==m['sha256'] and (B/m['path']).stat().st_size==m['bytes'])
ok('snapshot_helper_passed',json.loads((T/'local_snapshot_integrity.json').read_text())['flat_script_manifest_checks']==41)
raw=(T/'D_CONSOLIDATED_review.md').read_text();report=W/'reports/HPP_Claude_W08_W15_Review_2026-10-29.md';summary='Claude已完成W08–W15分组三轮审查及一次跨包汇总，四次CLI实际exit均0，模型标识claude-opus-5-5。结论为有条件通过：未发现已发生且使已发布产物失效的实现或数值错误；可进入W16菌群处理／合成接口，任何真实Y读取前必须补受控读取和逐次回执。肝重复、相同丰度谱、家系跨cohort、用药evidence及核验盲区等需一次TRE只读补查；另有文档与未来更新规则待补。本次未独立重跑TRE，不替代研究者人工签字。'
cover='''# W08–W15 Claude 独立复核交付

2026-10-29（America/Los_Angeles）。**结论：有条件通过；可以继续 W16 的菌群处理与合成软件接口。**

'''+summary+'''

本次已实际在项目根调用本机 Claude Code 2.1.314，按 W08–W10、W11–W13、W14–W15 分三组审查，再用第四次调用统一复核跨包衔接。四次返回模型均为 `claude-opus-5-5`，实际退出码均0，无权限拒绝，工具只有 Read／Glob／Grep；全部观察到的文件读取都在验收材料内。不是研究者人工签字。

原始153份输入快照逐项核hash；另补7张已查看的聚合技术截图、6份启动包装代码，以及从W13实际传输代码解码出的3份协议原文（SHA与内嵌值一致）。Codex另做本地确定性完整性检查：153份原始快照审查前后字节未变，41份单脚本清单与对应代码hash一致。此检查单独登记，不冒称Claude自行计算了hash，也不证明TRE全部数值语义。

## 最优先的补做

1. **任何真实Y读取前：** 补 W15 的实际Y受控读取器、读取前冻结回执校验、逐次不可覆盖访问回执及后续脚本preflight。现有guard只保护盲成员表，不能管理含Y的真实表。没有发现已经发生的泄漏；此项是未来读取前必须关闭的机制缺口。FUSE已观察到不支持append，回执应逐次写新文件。
2. **一次TRE只读补查：** 合并肝明细重复诊断、相同丰度谱分组、深度缺失、家系跨cohort／未命中节点、用药evidence／listed_rows独立重算、W14核验覆盖盲区和W09退出来源。只读聚合，先判断实际影响，不按评审猜测重建或排除样本。
3. **版本化文档附录：** 补家系更新／追加规则、历史接触领域与W08–W13技术接触，统一旧CGM接口、人数勘误、QC标签及旧完成标记说明。冻结的人员顺序与服务器产物不在本轮改写。

补充证据已关闭W08 9056／9065及覆盖率争议、SD未计算疑点、双部位E是否符合协议和可选设计核心等疑点；家系单元更新风险、深度QC标签也已降级。不能把首轮报告中尚未复核的条目一概当成已确认错误；以下跨包最终报告为本轮汇总结论，首轮报告原样保留。

## 原始审查与证据

'''
for tag,label in [('A_W08_W10','第一组：W08–W10'),('B_W11_W13','第二组：W11–W13'),('C_W14_W15','第三组：W14–W15'),('D_CONSOLIDATED','跨包最终复核原始答复')]:cover+=f'- [{label}](../audits/claude_w08_w15_review_2026-10-29/{tag}_review.md)\n'
cover+='\n[最终运行与hash清单](../audits/claude_w08_w15_review_2026-10-29/FINAL_REVIEW_MANIFEST.json)；[本地交付验证](../audits/claude_w08_w15_review_2026-10-29/final_delivery_validation.json)。完整prompt、JSON结果、逐条转录和实际退出码均保存在同一验收目录。未取回TRE个体表，未修改服务器产物，未运行新科学关联或模型。\n\n## Claude 最终报告原文\n\n'
report.write_text(cover+raw)
aliases={'w07_w08':'w07_w08_diet_views_2026-10-28','w09':'w09_cgm_point_qc_2026-10-28','w10':'w10_outcome_qc_2026-10-28','w11':'w11_w12_covariates_samples_2026-10-28','w13':'w13_microbiome_qc_2026-10-29','w14':'w14_eligibility_2026-10-29','w15':'w15_split_access_2026-10-29'}
index='# Claude W08–W15 复核材料索引\n\n'+summary+'\n\n主报告：[完整汇总](../../reports/HPP_Claude_W08_W15_Review_2026-10-29.md)。\n\n原始首轮／汇总答复、prompts、转录、实际退出、模型返回值均在本目录。原始153份文件在`packet/`；新增协议、截图、启动包装分别在`supplemental_protocol/`、`supplemental_evidence/`、`supplemental_launchers/`。临时缓存位于`/private/tmp/hpp_claude_w08_w15_20261029/packet`，用于避免iCloud重新回收内容。\n\n最终报告路径简写对应如下（均在原始packet/audits中）：\n\n'
for a,b in aliases.items():index+=f'- `{a}` → `packet/audits/{b}/`\n'
index+='\n最终15个X编号分别包含当前机制缺口、核验覆盖、实际影响待证、未来更新风险和文档项；并非15个已发生的数值错误。以最终报告的状态列与截止阶段为准。后续最小补做清单已整理，本次没有执行修复或TRE补查。\n';(D/'INDEX.md').write_text(index)
# Copy frozen pre-review status files for reversibility, then append external review status only.
snapshot=json.loads((T/'status_before_review_manifest.json').read_text());snap=D/'before_review_status_update';snap.mkdir(exist_ok=False)
for m in snapshot:
 p=W/m['path'];ok('no_concurrent_status_change_'+m['path'],sha(p)==m['sha256']);dst=snap/m['path'];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(T/'status_before_review'/m['path'],dst)
(D/'before_review_status_manifest.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')
progress_path=W/'outputs/researchwrite/hpp_multisystem_20261021/team_checklist_20261026/workpackage_progress.json';progress=json.loads(progress_path.read_text());review_status={'reviewer':'Claude via local Claude Code','model':'claude-opus-5-5','date_local':'2026-10-29','kind':'static_code_and_existing_execution_evidence','status':'conditional_acceptance','report':'reports/HPP_Claude_W08_W15_Review_2026-10-29.md','TRE_independent_replay':False,'server_artifacts_changed':False,'actual_review_exit_codes':{tag:0 for tag in tags},'allowed_next_scope':'W16 microbiome processing and synthetic interfaces; no Y access','before_real_Y':'X-01 access reader/immutable receipts and X-04 family identity checks; preregister any design core','before_W18':'TRE aggregate supplemental verification plus documented disposition of remaining findings','human_signoff_complete':False}
progress['claude_W08_W15_review_20261029']=review_status
for i in range(8,16):
 key='W'+str(i).zfill(2)
 if key in progress['tasks']:progress['tasks'][key]['claude_review']={'status':'conditional_acceptance','report':review_status['report'],'TRE_independent_replay':False,'human_review_complete':False}
progress_path.write_text(json.dumps(progress,ensure_ascii=False,indent=2)+'\n')
for rel in ['reports/HPP_CURRENT_PLAN.md','reports/HPP_Team_Execution_Checklist_2026-10-26.md']:
 p=W/rel;s=p.read_text();first=s.index('\n');banner='> **Claude复核（2026-10-29，有条件通过）：** '+summary+' [完整验收报告](HPP_Claude_W08_W15_Review_2026-10-29.md)。\n\n';s=s[:first+1]+'\n'+banner+s[first+1:].lstrip('\n');p.write_text(s)
p=W/'reports/HPP_Team_Execution_Checklist_2026-10-26.html';s=p.read_text();notice='<div data-claude-w08-w15-review="true" style="position:static;margin:16px;padding:20px;border:2px solid #9b7425;background:#fff9e8;color:#40330c"><strong>2026-10-29：Claude复核有条件通过</strong><p>'+html.escape(summary)+'</p><a href="HPP_Claude_W08_W15_Review_2026-10-29.md">完整Claude验收报告与最小补做清单</a></div>';s=re.sub(r'(<body\b[^>]*>)',lambda m:m.group(1)+notice,s,count=1,flags=re.I);p.write_text(s)
manifest={'status':'Claude_review_complete_conditional_acceptance','date_local':'2026-10-29','timezone':'America/Los_Angeles','review_groups':reviews,'original_inputs':153,'original_input_manifest_sha256':sha(D/'input_manifest.json'),'original_packet_bytes_unchanged':True,'local_script_hash_checks':41,'supplemental_protocol':json.loads((D/'supplemental_protocol/DECODE_PROVENANCE.json').read_text()),'supplemental_evidence':json.loads((D/'supplemental_evidence/MANIFEST.json').read_text()),'supplemental_launchers':json.loads((D/'supplemental_launchers/MANIFEST.json').read_text()),'main_report':str(report),'main_report_sha256':sha(report),'index_sha256':sha(D/'INDEX.md'),'status_updates':[{'path':m['path'],'before_sha256':m['sha256'],'after_sha256':sha(W/m['path'])} for m in snapshot],'no_TRE_individual_table_export':True,'no_server_changes':True,'no_new_scientific_models_or_associations':True,'TRE_independent_replay':False,'human_signoff':False}
(D/'FINAL_REVIEW_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');ok('report_has_actual_consolidation',report.read_text().endswith(raw));ok('all_eight_tasks_keep_delivery_status',all(progress['tasks'].get('W'+str(i).zfill(2),{}).get('claude_review',{}).get('status')=='conditional_acceptance' for i in range(8,16) if 'W'+str(i).zfill(2) in progress['tasks']));ok('future_W16_not_claimed',not progress['tasks'].get('W16',{}).get('checked',False));ok('review_notice_in_MD',all('Claude复核（2026-10-29，有条件通过）' in (W/p).read_text() for p in ['reports/HPP_CURRENT_PLAN.md','reports/HPP_Team_Execution_Checklist_2026-10-26.md']))
(D/'REVIEW_STATUS.json').write_text(json.dumps(review_status,ensure_ascii=False,indent=2)+'\n');validation={'status':'passed','checks':len(checks),'check_names':checks,'main_report_sha256':sha(report),'real_CLI_invocations':4,'actual_exit_codes':[0,0,0,0],'base_inputs_unchanged':153,'local_script_manifest_checks':41,'server_changes':False};(D/'final_delivery_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
print('CLAUDE_REVIEW_DELIVERY_VERIFIED',len(checks),'report',report,'review_invocations',len(reviews),flush=True)
