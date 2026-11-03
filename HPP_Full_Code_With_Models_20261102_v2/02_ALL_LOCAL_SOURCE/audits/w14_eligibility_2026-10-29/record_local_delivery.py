# -*- coding: utf-8 -*-
"""Record screen-transcribed aggregates only after remote final exit-zero evidence."""
from pathlib import Path
import json,hashlib,re,shutil,html,ast
root=Path.cwd();run=root/'audits/w14_eligibility_2026-10-29';obs=json.loads((run/'observed_aggregate.json').read_text());agg=obs['aggregate'];assert obs['actual_exits']['08b']==0 and obs['remote_final_readback_passed'];h=hashlib.sha256(json.dumps(agg,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest();assert h==obs['canonical_sha256'];P='/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021';rid='w14_eligibility_20261029';report='reports/HPP_W14_Eligibility_Cohort_Flow_2026-10-29.md'
by={r['analysis_id']:{x['exposure_id']:x for x in agg['counts'] if x['analysis_id']==r['analysis_id']} for r in agg['counts']};names={'D_Glu':'CGM血糖访视均值','D_TotalFat':'总体脂肪重量','D_VAT':'VAT体积','D_Liver_elasticity':'肝弹性','D_Liver_velocity':'肝剪切波速度','D_Liver_viscosity':'肝黏度','D_Liver_dispersion':'肝频散','D_Liver_attenuation':'肝衰减系数','D_Liver_speed_of_sound':'肝超声声速','D_BMI':'BMI（次要）','Daily_CGM':'日级CGM'}
table='| 模块 | 基础E | UPF界限可用≥1日 | 纤维可用≥1日 |\n|---|---:|---:|---:|\n'
for a in names:
 u=by[a]['UPF_bounds'];f=by[a]['fiber_density'];table+=f"| {names[a]} | {u['base_E']:,} | {u['window_ge1']:,} | {f['window_ge1']:,} |\n"
qc='| 模块 | UPF≥7日诊断 | 纤维≥7日诊断 | UPF≥1日＋年龄性别吸烟 | 纤维≥1日＋年龄性别吸烟 |\n|---|---:|---:|---:|---:|\n'
for a in names:
 u=by[a]['UPF_bounds'];f=by[a]['fiber_density'];qc+=f"| {names[a]} | {u['ge7_diagnostic']:,} | {f['ge7_diagnostic']:,} | {u['age_sex_smoking_candidate']:,} | {f['age_sex_smoking_candidate']:,} |\n"
text=f'''# HPP W14：基础资格、候选分析表与人数流转

2026-10-29。已通过当前Chrome→WorkSpaces服务器在pheno环境完成构建、独立重建、正式发布及发布后读回；01、02、03、03b、04、05、06、07、08b实际exit均0；08追加归档遇挂载盘限制exit1，保留并由08b修复。Codex代执行，不冒称同事人工复核或新Astra／Opus评审。未训练模型、未运行本轮关联、未分发现／确认角色。

## 范围与当前人数

本次研究表明确采用10k队列的`00_00_visit`基线，每个人级模块一人一行。来源并集保留所有实际访视20,407行，以scope标记区分；不把不同访视混成独立人。正式资格registry含10个人级模块及Daily_CGM共224,477行。六个肝表型各自建表，不要求其他五项齐全；总脂、VAT、血糖各用自身必要来源；BMI保留独立次要模块。跨域pair及Common4未建，不作为当前交付门槛。

基础E要求：同人同cohort同stage，有日索引且有实际食物事件的记录日、W13锁定口腔样本、W13锁定肠道样本及本结局基本QC合格。窗口、条件变量和评分支持分开。下表≥1日仅表示该饮食候选具备来源可用性，**不是已冻结最少天数或最终模型N**。日级行的基础人数另加实际合格共同日，因而与人级E_Glu不同。

{table}

后续W15角色候选并集为**6,305人**，包含上述可用基础候选（含次要BMI）；重叠模块不能相加作独立总N。只准备名单和已有家系来源，不生成角色或比例。

## 候选窗口与条件完整性诊断

≥7日沿用待验收诊断，不是本轮决定的主门槛。末两列从各自≥1日窗口分支出发检查年龄、性别和吸烟可用性，不以≥7日为前置；此条件集也尚未正式冻结。所有正式window／covariate／support／final资格、q/rank、发现／确认及final N仍为null。

{qc}

年龄性别可用人数与对应≥1日窗口人数相同。真实family N仍null；流转表分别给出已观察关联组、未连接／未观察人数及工作簇代理数，无连接不等于无亲缘。

## CGM两层接口

新建人级`glucose_visit_equal_day_mean`，单位mg/dL：同人同cohort同stage，按W09合格CGM日的日均血糖等权平均。连接按合格CGM日数最多、最早起始、稳定connection_id选择，完全不依赖食物共同日或血糖值，不拼接连接。共9,489个访视，基线9,402人、103,925合格日，≥7日诊断8,202人；与旧日级接口相比3个人—访视更换连接。新汇总与原共同日接口分开命名和保存。

日级保留W09真实日期连接和原选定连接，不要求DXA或肝脏。基于E_Glu获得5,811人有实际合格共同日；UPF界限5,779人／60,134个人日，纤维5,327人／32,129个人日。每个暴露分支的可用日候选权重为1/n_i，逐人核总和为1。纤维有4,846人的原始个人内暴露变异为正，仅为暴露信息诊断，不等于最终残差设计秩或评分支持。UPF上下界变化不能证明真实标量的个人内变化，该信息保持待定；零未知能量精确日仅保留明确命名的敏感性子集。

## 饮食与用药处理

UPF主候选保留所有能量合格日的等权日均及累计比上下界、未知能量和覆盖率；不取中点，不把下界改称真实UPF标量。历史零未知能量／严格全分类子集独立命名，不替代整个窗口。纤维保留等权日均与累计比两种口径。没有新增外部分型或补齐NOVA。

用户本轮明确确认保留三套用药窗口：饮食记录期、结局测量期、两者最早至最晚的包络期，正式模型使用哪套留W18。每个模块、暴露、窗口和药物类分别保存状态及证据；包络期须饮食和结局日期完整，且不表示连续测量。Daily的三个窗口均退化为同一个实际共同日，统一存为`med_actual_day_*`。

状态为用药／确认未用药／无法确定。当前源只能在已知ATC编码的开始时间区间完全落在窗口时证明“期内至少发生一次开始用药”，不能证明连续服药。先于窗口的开始记录缺持续证据、缺开始日期、区间跨边界、未列该ATC或没有记录均为unknown；当前没有覆盖研究期的明确无用药证据；三套窗口本次合计2,030,280条模块×候选×药物类状态全部为unknown，各窗口676,760条（是重复模块状态数，不是独立人数），**未确认任何窗口内用药或未用药，不把unknown改0**。保留metformin、insulin、降糖、statin、降脂、降压和全身抗生素七类，仍为模型候选而非自动调整／排除项。

## 验收与实际边界

独立脚本以batch127重读来源，核28个构建产物和3个CGM副本；重建9,489个访视均值、96,680个人级候选行、2,030,280个用药状态、123,266个日级行（两个候选表各61,633行），并核11,106组候选个人权重。独立验收实际exit0，峰值RSS约1.33GiB；构建约2.59GiB，地址空间上限4GiB。

正式发布36项产物。最终读回核全部hash／行数、原11输入未改、ACTIVE_PROTOCOL及W13样本锁未改、11模块registry一对一、分层人数守恒、最终状态null及6,305人并集一致。08在追加MEMO时因FUSE不支持append而exit1；08b改写全新完成凭证并重新核36产物hash，实际exit0。失败源码／日志保留，原数据产物及07验收结果未改。W13口腔隐式零编码推定和仪器QC来源限制继续存在，W14不将其升级为直接验证。

本地只存自编源码、人工聚合转写、规则摘要与聚合截图，个体ID、日期、明细及完整服务器产物只留TRE。聚合转写SHA256与屏幕一致：`{h}`。

## 路径与接续

以下P为TRE项目根`{P}`，不是本地文件。

- Code：`P/scripts/{rid}/`；本地自编源码与功能说明见[MEMO](../audits/w14_eligibility_2026-10-29/MEMO.md)。
- Result：`P/data_qc/{rid}/`，含原输入manifest、两版执行契约、独立验收、最终manifest、聚合、读回及completion_status_08b.json（08旧临时状态由此取代）。
- 主接口：`P/analysis/eligibility_registry_TRE_ONLY.parquet`；`P/analysis/<analysis_id>/<exposure_id>/base_table_TRE_ONLY.parquet`；日级另有`within_person_information_TRE_ONLY.parquet`；`P/results/cohort_flow_blind.csv`。
- 共享：`P/analysis/shared/cgm_visit_phenotypes_TRE_ONLY.parquet`、`cgm_visit_qualified_daily_TRE_ONLY.parquet`、`cgm_person_connection_selection_TRE_ONLY.parquet`、`medication_window_status_TRE_ONLY.parquet`、`source_availability_union_TRE_ONLY.parquet`、`W15_candidate_union_TRE_ONLY.parquet`。
- 配置：`P/config/w14_candidate_publication.json`、`eligibility_rules_W14_20261029.json`、`outcome_dictionary_W14_20261029.json`、`medication_rules_W14_20261029.json`；原ACTIVE_PROTOCOL和W13锁不覆盖。
- MEMO：`P/memos/{rid}/MEMO.md`及`FINAL_READBACK.md`，根交接`P/W14_HANDOFF.md`，最终完成凭证`P/W14_COMPLETED_08b.md`及MEMO目录`COMPLETION_STATUS_08b.md`；01–08日志（含失败）和实际退出码在`P/logs/{rid}/`。
- Runtime：`/home/ec2-user/multisystem_runtime_20261021/{rid}`，08b日志与实际退出码在此；新完成凭证存P。

下一包W15：统一人／家庭角色与历史接触账本；W16可准备合成接口。正式饮食／UPF测量接口、最少日数、聚合、条件集、用药窗口选择、角色比例及家族在W18/W22冻结。未擅自启动W15、真实训练或关联。
'''
(root/report).write_text(text)
memo='''# W14 本地代码与接续MEMO

仅存自编代码、人工聚合及截图，不存个体表。服务器产物和完整科学边界见 ../../reports/HPP_W14_Eligibility_Cohort_Flow_2026-10-29.md。

| 脚本 | 功能 |
|---|---|
| 01_preflight.py | 核W13锁、现行协议、输入schema／预算及来源指纹；原已运行exit0 |
| 02_build_CGM_visit.py | 按独立合格CGM期间构建人级等权日均表，连接不依赖饮食；原已运行exit0 |
| 03_inspect_interfaces.py、03b_compact_interfaces.py | 当前远程输入、模块与schema读取；exit0 |
| 04_build_candidates.py | 分模块基础E、两候选饮食表、三套用药状态、Daily、W15并集、盲流转；exit0 |
| 05_independent_verify.py | batch127独立重建所有成员资格、汇总、用药状态、Daily源值与权重；exit0 |
| 06_publish_candidates.py | 正式发布、加入Daily的统一registry、版本化字典和规则、日志及MEMO；exit0 |
| 07_final_readback.py | 正式36产物hash、来源未变、W13锁、流转、registry、并集与null状态独立读回；exit0 |
| 08_seal_delivery.py | 归档07日志及代码hash；MEMO追加不受FUSE支持，exit1保留 |
| 08b_seal_without_append.py | 独立文件完成凭证，重核36产物hash，保留08失败；exit0 |
| prepare_transfer.py | 本地语法与hash；压缩后通过浏览器Clipboard及终端传入、远程hash核验并保存真实退出码 |
| record_local_delivery.py | 聚合转写hash验收、本地报告与进度更新，无服务器访问 |

下一步W15，不在W14生成角色或冻结设计。最低天数、饮食／用药模型接口、协变量及评分支持仍待W18/W22。本地报告人数不能替代TRE个体数据用于建模。
''';(run/'MEMO.md').write_text(memo)
summary='W14已在TRE完成构建、独立重建、正式发布及最终读回，01/02/03/03b/04/05/06/07/08b实际exit均0，08挂载盘追加失败exit1已由08b恢复。10个人级模块＋Daily、两饮食候选，36正式产物；registry224477行，W15候选并集6305人。基础E：血糖5876、总脂4338、VAT4284，肝前五项各4591、声速4597，BMI6290。CGM独立访视汇总与日级共同日分开；六肝各自准入；UPF保留界限；用药按用户确认保留三窗口三状态，确认未用药证据缺失不改0。窗口／条件／支持／q/rank／角色和最终N尚未冻结，无训练或关联。下一步W15。'
paths=['reports/HPP_CURRENT_PLAN.md','reports/HPP_Team_Execution_Checklist_2026-10-26.md','reports/HPP_Project_TODO_v3_2026-10-23.md','reports/HPP_Project_Roadmap_v3_2026-10-23.md','HPP_LOCAL_VERIFIED_FACTS.md','HPP_PROJECT_BACKGROUND.md','HPP_DATA_BACKGROUND.md','HPP_SERVER_BACKGROUND.md','outputs/researchwrite/hpp_multisystem_20261021/team_checklist_20261026/workpackage_progress.json']
for rel in paths:
 p=root/rel;backup=run/'before_w14_update'/rel;backup.parent.mkdir(parents=True,exist_ok=True);assert not backup.exists();shutil.copyfile(p,backup)
 if rel.endswith('.json'):continue
 s=p.read_text();link=Path(report).name if rel.startswith('reports/') else report;banner=f'\n> **当前实际进度（W14，2026-10-29）：** {summary} [W14报告]({link})。\n';pos=s.index('\n')+1;s=s[:pos]+banner+s[pos:]
 if rel=='reports/HPP_CURRENT_PLAN.md':
  s=s.replace('W04–W13已按各包范围交付，下一步W14','W04–W14已按各包范围交付，下一步W15');s+='\n\n## 2026-10-29 W14正式交付\n\n'+summary+'\n\n本次读回以W14报告与TRE completion_status为准；前文“W14待执行”“W13下一步”是历史状态。\n'
 if rel=='reports/HPP_Team_Execution_Checklist_2026-10-26.md':
  s=re.sub(r'^- \[ \] \*\*W14([^\n]*)$',lambda m:'- [x] **W14'+m.group(1)+'（Codex代执行；独立代码验收通过，人工复核待进行）',s,count=1,flags=re.M)
  i=s.index('### W14 ');j=s.index('### W15 ',i);part=s[i:j];assert '**状态：** 待执行。' in part;part=part.replace('**状态：** 待执行。','**状态：** '+summary+' [完整报告]('+link+')。',1);s=s[:i]+part+s[j:]
 p.write_text(s)
p=root/paths[-1];d=json.loads(p.read_text());d.update(date='2026-10-29',latest_run_id=rid,latest_local_workpackage='W14',latest_local_run_id=rid,evidence='audits/w14_eligibility_2026-10-29/MEMO.md',next_recommended_workpackage='W15',parallel_preparation=['W16合成变换／估计接口']);d['tasks']['W14'].update(checked=True,short_status='基础E、两饮食候选表、Daily、三用药窗口与流转完成；最终N待后续冻结',status_text=summary,what_ran='01/02沿用既有成功执行；03/03b接口读取、04构建、05独立重建、06发布、07最终读回、08b归档恢复均实际exit0；08追加失败exit1留存',code_location=P+'/scripts/'+rid,result_location=P+'/data_qc/'+rid,local_report=report,human_review_complete=False,next_execution_note='W14已完成；进入W15统一角色与历史接触账本，不提前训练');p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
(root/'config/w14_eligibility_20261029').mkdir(exist_ok=True);(root/'config/w14_eligibility_20261029/execution_status.json').write_text(json.dumps({'run_id':rid,'status':'completed_technical_candidate_delivery','summary':summary,'aggregate_sha256':h,'medication_window_user_decision':'retain_food_outcome_envelope_choose_at_W18','next':'W15','human_review_complete':False,'models_run':False},ensure_ascii=False,indent=2)+'\n')
(run/'local_update_paths.json').write_text(json.dumps(paths+[report],ensure_ascii=False,indent=2)+'\n')
# Plain offline HTML, displaying the verified tables and preserving complete Markdown as escaped source.
rows=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [names[r['analysis_id']],r['exposure_id'],r['base_E'],r['window_ge1'],r['ge7_diagnostic'],r['age_sex_smoking_candidate']])+'</tr>' for r in agg['counts'])
page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>HPP W14 交付</title><style>body{font:16px/1.7 system-ui,sans-serif;max-width:1180px;margin:40px auto;padding:0 24px;color:#172d38;background:#f7fafb}h1{font-size:30px}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #cddce2;padding:8px;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:24px;border:1px solid #cddce2}.note{padding:18px;background:#e6f2ec}</style><h1>W14 基础资格与候选表：已验收</h1><p class="note">'+html.escape(summary)+'</p><p>下表均为候选人数。≥7日和年龄性别吸烟完整性尚未冻结为正式门槛。</p><table><thead><tr><th>模块</th><th>饮食候选</th><th>基础E</th><th>可用≥1日</th><th>≥7日诊断</th><th>≥1日＋年龄性别吸烟</th></tr></thead><tbody>'+rows+'</tbody></table><h2>完整报告与路径</h2><pre>'+html.escape(text)+'</pre></html>'
(root/report).with_suffix('.html').write_text(page);(run/'html_build.json').write_text(json.dumps({'source_sha256':hashlib.sha256(text.encode()).hexdigest(),'output':str(Path(report).with_suffix('.html')),'external_resources':False,'browser_visual_QA':False},indent=2)+'\n')
# Existing HTMLs remain dated history, with an authoritative new banner; avoid rebuilding from stale templates.
for rel in ['reports/HPP_Team_Execution_Checklist_2026-10-26.html','reports/HPP_最新版方案_双路线与工作路径_v3_2026-10-23.html']:
 p=root/rel
 if p.exists():
  old=p.read_text();b=run/'before_w14_update'/rel;b.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,b);notice='<aside style="padding:20px;background:#e6f2ec;color:#172d38;border-bottom:2px solid #3b8560"><strong>2026-10-29最新：W14技术交付已完成，下一步W15。</strong> '+html.escape(summary)+' <a href="HPP_W14_Eligibility_Cohort_Flow_2026-10-29.html">W14完整报告与候选人数</a>。下方既有正文保留其历史状态，冲突处以本提示及当前Markdown/进度JSON为准。</aside>';old=re.sub(r'(<body\b[^>]*>)',lambda m:m.group(1)+notice,old,count=1,flags=re.I);p.write_text(old)
print('LOCAL_DELIVERY_RECORDED',h)
