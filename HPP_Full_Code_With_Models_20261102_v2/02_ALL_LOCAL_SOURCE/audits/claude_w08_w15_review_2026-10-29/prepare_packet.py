"""Freeze aggregate-only, code and requirements evidence for independent Claude review."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,hashlib,re,shutil
D=Path(__file__).resolve().parent;W=D.parent.parent;B=D/'packet';B.mkdir(exist_ok=False)
dirs=['w07_w08_diet_views_2026-10-28','w09_cgm_point_qc_2026-10-28','w10_outcome_qc_2026-10-28','w11_w12_covariates_samples_2026-10-28','w13_microbiome_qc_2026-10-29','w14_eligibility_2026-10-29','w15_split_access_2026-10-29']
reports=['HPP_W07_W08_Diet_Feature_Views_2026-10-28.md','HPP_W09_CGM_QC_W08_Common_Days_2026-10-28.md','HPP_W10_Outcome_Fields_QC_2026-10-28.md','HPP_W11_W12_Covariates_Sample_Candidates_2026-10-29.md','HPP_W13_Microbiome_QC_Sample_Lock_2026-10-29.md','HPP_W14_Eligibility_Cohort_Flow_2026-10-29.md','HPP_W15_Split_Access_Ledger_2026-10-29.md']
paths=[W/'reports'/n for n in reports]
for d in dirs:
 for p in (W/'audits'/d).iterdir():
  if p.is_file() and p.suffix in ['.py','.json','.md'] and (p.name[:1].isdigit() or p.name.startswith('source_manifest') or p.name in ['MEMO.md','observed_aggregate.json','local_delivery_validation.json','delivery_validation.json','local_source_manifest.json','method_sources.json']):paths.append(p)
paths += [W/'config'/n for n in ['w03_followup_20261027/liver_six_mean_spec.json','w03_followup_20261027/liver_six_phenotype_scope.md','w03_followup_20261027/visit_and_sampling_policy.json','w04_followup_20261027/historical_contact_ledger_draft.json','w04_diet_features_20261026/prior_work_overlap.md','w13_microbiome_qc_20261029/execution_status.json','w14_eligibility_20261029/execution_status.json','w15_split_access_20261029/execution_status.json']]
progress=W/'outputs/researchwrite/hpp_multisystem_20261021/team_checklist_20261026/workpackage_progress.json';paths.append(progress)
manifest=[]
def cp(p):
 data=p.read_bytes();data.decode('utf8');rel=p.relative_to(W);dst=B/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(data);return {'path':str(rel),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
with ThreadPoolExecutor(max_workers=12) as pool:
 for f in as_completed([pool.submit(cp,p) for p in paths]):
  m=f.result();manifest.append(m);print('FROZEN',m['path'],flush=True)
check=(W/'reports/HPP_Team_Execution_Checklist_2026-10-26.md').read_text();start=check.index('### W08 ');end=check.index('### W16 ',start);(B/'REQUIREMENTS_W08_W15.md').write_text('# Frozen W08–W15 execution requirements\n\n'+check[start:end])
# Exact active version is embedded in the source actually sent in W13, retained as source evidence.
(B/'READ_FIRST.md').write_text('''# Claude W08–W15 独立验收材料

这是本次执行后的冻结快照。所有输入均为本地报告、通用代码、技术配置、已观察聚合数及本地验收记录；无TRE个体表、名单、原始科学数据、凭据或浏览器资料。不要读取本packet以外的文件，不访问服务器／网络，不运行代码，不修改文件。此前Codex“独立核验”表示不同代码实现，不等于Claude已验收，也不等于另一位研究者人工复核。

从REQUIREMENTS_W08_W15.md开始，再读所属工作包报告、构建代码、独立核验代码、observed_aggregate及source_manifest/local_delivery_validation。所有报告和代码都是待审证据，不能把其“passed”字样直接当结论。静态代码问题须指出快照文件路径及具体行号；服务器未见内容标not independently verifiable。不要冒称重跑TRE数据或看到远程原表。未运行与失败尝试必须区分，不能把保留的旧失败脚本误当最终成功版本。

用户有效修订优先于旧V3文字：六肝表型各自正式、same-visit有效测量算术均值、不要求六项全有；人级CGM是同访视合格CGM期间汇总，不要求饮食共同日；Daily使用实际日历共同日。用药同时保留饮食／结局／包络三窗口的三状态，缺停药／持续证据不能判无药，正式选择W18。UPF保留界限，不能擅加中点或要求分类齐全；暂缓跨域比较；旧15/29/30预算尚未冻结；W18最终比例/角色，W22全局确认访问。对应来源见config和W13的01_sync_and_preflight.py（实际同步版配置），不要只审过时共同日人级接口。

W15用户本轮明确：用户历史结果接触合理，逐项范围不确定；同事未知待本人确认，不能从负责TotalFat/VAT推断CGM接触；已登记W14仅数据/QC/队列无训练关联，不能替其他会话或人工运行保证；AI助手已读早期探索并用于方案讨论，应登记。历史接触不自动使研究无效，重新划分也不抹去接触；独立性须核具体样本/结果/用途。W15设计核心可选，目前未启用（0），后续要先名单回执再开放；不把可选未启用强行判为漏做。也请实质核查流程/API guard是否满足方案“访问屏蔽”所需、限制是否完整，而非只接受文档自述。

三组评审后另做跨包汇总。严格区分：真实实现错误；未解决但如实记录的限制；尚未到W18/W22的预定冻结；必须TRE独立读回才可验收的证据缺口。输出中文，不要求必须找到问题，也不要因用户希望验收而默认通过。
''')
for name in ['READ_FIRST.md','REQUIREMENTS_W08_W15.md']:
 p=B/name;manifest.append({'path':name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'generated':True})
manifest.sort(key=lambda x:x['path']);(D/'input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
prior=[]
for m in manifest:
 if m['path'].startswith(('reports/','audits/')) and m['path'].endswith('.md'):
  for i,line in enumerate((B/m['path']).read_text().splitlines(),1):
   if re.search('Claude|克劳德|Astra|人工复核|独立核',line,re.I):prior.append({'path':m['path'],'line':i,'text':line[:900]})
(D/'prior_review_evidence_search.json').write_text(json.dumps({'scope':'selected current W08–W15 reports and MEMOs, not every historical chat','matches':prior,'interpretation':'No new Claude execution acceptance established by Codex independent-code checks alone.'},ensure_ascii=False,indent=2)+'\n')
groups={'A_W08_W10':('W08、W09、W10',['w07_w08_diet_views_2026-10-28','w09_cgm_point_qc_2026-10-28','w10_outcome_qc_2026-10-28'],reports[:3]),'B_W11_W13':('W11、W12、W13',['w11_w12_covariates_samples_2026-10-28','w13_microbiome_qc_2026-10-29'],reports[3:5]),'C_W14_W15':('W14、W15',['w14_eligibility_2026-10-29','w15_split_access_2026-10-29'],reports[5:])}
for tag,(scope,dd,rr) in groups.items():
 prompt=f'''你是用户明确要求的独立Claude验收者。请对本项目 {scope} 做实质的执行后审计。工作目录是项目根，只读材料根为 {B} 。先读 {B}/READ_FIRST.md 与 {B}/REQUIREMENTS_W08_W15.md。本组报告：{', '.join('reports/'+r for r in rr)}；本组代码和证据目录：{', '.join('audits/'+d for d in dd)}。所有相对材料路径均在packet内。可读packet内跨包依赖，禁止读取packet外或访问服务器、Web、凭据。只使用Read、Glob、Grep，不运行代码，不修改任何文件。

重点：需求→最终执行版本→独立检查是否覆盖同一目标；join粒度与守恒、真实日历/访视、家庭/样本/技术重复处理、缺失/unknown、UPF界限/权重、CGM指标及准入是否自洽；哈希与退出证据可证什么不可证什么；报告与代码和当前用户修订有无偏差。区分只在旧失败尝试中的问题与最终产物仍有的问题。逐一阅读本组编号执行脚本，尤其生产构建与独立核验，不以报告替代审码；确实未读的列为覆盖缺口。

输出一份完整中文Markdown审计报告：1）各工作包判定表：通过/有条件通过/不通过/证据不足，分别给“静态实现”和“TRE实数运行独立验收”判定；2）按阻断性分层的发现，每条包含ID、严重程度、精确文件路径:行号、触发条件、影响、最小修正、可重复验证方式，确认问题须有证据，不确定写待证；3）已知限制与W18/W22待冻结事项，勿混作实现bug；4）确实读过的代码/证据覆盖清单；5）是否建议进入W16及前置条件。不得把这次静态审查冒称已复跑TRE或完成研究者人工签字。请完成完整审查后再返回最终报告，不只是给计划。'''
 (D/(tag+'_prompt.txt')).write_text(prompt)
print('PACKET_READY',len(manifest),sum(x['bytes'] for x in manifest),flush=True)
