# -*- coding: utf-8 -*-
"""Record local-only W04 completion; preserve previous status and navigation snapshots."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'outputs/w04_diet_definition_20261026';D=ROOT/'outputs/researchwrite/hpp_multisystem_20261021/team_checklist_20261026'
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def main():
 snap=OUT/'before_w04_update';snap.mkdir(exist_ok=False);recs=[]
 for name in ['HPP_LOCAL_VERIFIED_FACTS.md','HPP_PROJECT_BACKGROUND.md','reports/HPP_CURRENT_PLAN.md','reports/HPP_Team_Execution_Checklist_2026-10-26.md',str((D/'workpackage_progress.json').relative_to(ROOT)),str((D/'execution_update.md').relative_to(ROOT))]:
  p=ROOT/name;b=p.read_bytes();target=snap/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b);recs.append({'path':name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'snapshot':str(target.relative_to(ROOT))})
 save(snap/'manifest.json',recs)
 progress=json.loads((D/'workpackage_progress.json').read_text());progress['latest_local_workpackage']='W04';progress['latest_local_run_id']='w04_definition_20261026_v1';progress['evidence']='outputs/w04_diet_definition_20261026/MEMO.md'
 progress['tasks']['W03']['status_text']=progress['tasks']['W03']['status_text'].replace('留W11','留W12／W13')
 progress['tasks']['W03']['short_status']=progress['tasks']['W03']['short_status'].replace('待W11','待W12／W13')
 progress['tasks']['W04']={'checked':True,'short_status':'本地定义／重叠草案完成；真实特征和主方案尚未冻结；Codex执行','status_text':'2026-10-26已完成本地定义与重叠初版：UPF精确Other!D9、由待验收摄入量派生的纤维密度、4项来源和1项NC-Diet适配I-MEDAS背景分；等权日均／累计比分开；旧NOVA4／纤维×CGM结果接触已登记。原始补表hash与来源单元格核验通过；未登录TRE、未算个体特征、未改V3或冻结A/B。团队成员人工复核未冒称完成。','what_ran':'04_build_diet_spec.py（来源单元格／原hash／定义配置／数学演示）；validate_w04.py（定义及重叠交付核验）；build_report.mjs（离线报告）。另人工审读并写重叠表与决策日志。','code_location':'本地 `outputs/w04_diet_definition_20261026/scripts/`；[代码备忘录](../outputs/w04_diet_definition_20261026/MEMO.md)。','result_location':'本地 `config/w04_diet_features_20261026/`：diet_feature_spec_draft.yaml、prior_work_overlap.md、decision_log.md；[W04阅读页](HPP_W04_Diet_Feature_Definitions_2026-10-26.html)与[Markdown](HPP_W04_Diet_Feature_Definitions_2026-10-26.md)。本轮未写入服务器。'}
 save(D/'workpackage_progress.json',progress)
 p=D/'execution_update.md';s=p.read_text().replace('（W03已运行）','（W04定义草案已完成）')
 anchor='W02／W03由Codex在TRE代执行'
 s=s.replace('\n\n'+anchor,'\n| W04 | **本地定义与历史重叠草案完成** | 2优先候选、4来源、1适配背景分；来源单元格／原hash核验通过，未算真实特征／未正式冻结 |\n\n'+anchor,1)
 s=s.replace('分别留W11／W06处理','分别留W12／W13和W06处理').replace('当前可继续W04特征定义／W05营养素语义；W11须','W04已完成，下一步W05营养素语义；W12／W13须')
 s+='\n\nW04交付：[饮食定义与重叠阅读页](HPP_W04_Diet_Feature_Definitions_2026-10-26.html)。UPF／纤维均为待真实数据验收候选；原项目NOVA4／纤维×CGM探索接触已登记，不能把新划分称为从未接触。旧UPF分类能量分母与当前全部能量候选不混用。背景为NC-Diet适配版I-MEDAS 0–17，不强行复现未明配方。W04没有新增N或确认结果。任务号以团队总表为准：W11协变量、W12样本映射、W13菌群QC。\n'
 p.write_text(s)
 p=ROOT/'HPP_LOCAL_VERIFIED_FACTS.md';s=p.read_text().replace('W11','W12／W13')
 row='\n| W04 | 本地候选定义及重叠初版完成，未算个体特征／未冻结正式主方案 | [W04报告](reports/HPP_W04_Diet_Feature_Definitions_2026-10-26.md)、[配置](config/w04_diet_features_20261026/diet_feature_spec_draft.yaml) |'
 s=s.replace('\n\n## 服务器位置',row+'\n\n## 服务器位置',1)
 s+='\n\n## W04饮食定义：后续直接查本地\n\n两个候选为UPF全部记录能量占比、纤维g/1000kcal；来源Other!D9与Nutrients!A5/A2、Dash!C6:D6。等权日均和累计能量比使用不同ID，W18正式决定。未知NOVA不当非UPF，保留上下界；营养字段实际摄入／每100g语义待W05。4食物来源和1个NC-Diet适配I-MEDAS背景分默认不增加主要确认假设。\n\n上一口腔投稿稿与本项目早期NOVA4／纤维×CGM探索分开登记；历史结果接触不能因新划分消失。精确样本交集未知，详见config/w04_diet_features_20261026/prior_work_overlap.md。既有W03备注把样本QC写作W11的编号已更正：当前工作包W11是条件变量，W12映射样本、W13矩阵QC。历史运行数据及源码不改。\n'
 p.write_text(s)
 notice='\n\n## 2026-10-26 W04本地定义工作包已完成\n\n[W04饮食定义报告]({report})已完成：UPF、纤维密度两个候选，4项食品来源及1个NC-Diet适配I-MEDAS背景分。配置／历史重叠／决策日志位于本地config/w04_diet_features_20261026；备忘录在outputs/w04_diet_definition_20261026。未登录TRE、未算个体特征、未正式改变V3或冻结A/B。日均与累计比、旧分类能量分母差别及历史NOVA4／纤维×CGM探索接触已登记。下一步W05验营养语义。菌群样本映射／QC正确编号为W12／W13，旧W03部分W11简称不作为当前执行编号。\n'
 for name,report in [('HPP_PROJECT_BACKGROUND.md','reports/HPP_W04_Diet_Feature_Definitions_2026-10-26.html'),('reports/HPP_CURRENT_PLAN.md','HPP_W04_Diet_Feature_Definitions_2026-10-26.html')]:
  p=ROOT/name;s=p.read_text();first=s.index('\n');s=s[:first+1]+'\nW04已完成本地定义草案，见[饮食候选与重叠报告]('+report+')；真实营养计算与正式选择仍待后续工作包。\n'+s[first+1:];p.write_text(s+notice.format(report=report))
 for name in ['MEMO.md','review.md','00_scope.md','01_research_canon.md','02_evidence_table.md','03_argument_map.md','04_section_contracts.md']:
  with (D/name).open('a') as f:f.write('\n\n## W04本地定义接续\n\nW04本地定义与历史重叠草案完成，未连接TRE、未算个体D/Y、未冻结主暴露或预算。2候选／4来源／1适配背景分，精确单元格及原hash核验，旧NOVA4／纤维×CGM探索接触明确登记。源配置和MEMO在outputs/w04_diet_definition_20261026及config/w04_diet_features_20261026；总表进度源已加入W04。W11条件变量、W12样本映射、W13菌群QC编号已在当前说明更正，旧运行记录保留。\n')
 p=D/'state.json';v=json.loads(p.read_text());v.update(stage='W04_definition_draft_complete_empirical_validation_pending',W04='local_definition_and_overlap_draft_complete',latest_local_run_id='w04_definition_20261026_v1',TRE_accessed_for_W04=False,individual_features_computed_W04=False,formal_diet_package_frozen=False);save(p,v)
 print('W04 persistent status, local-first cache and current navigation updated. No remote mutations.')
if __name__=='__main__':main()
