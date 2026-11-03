from pathlib import Path
import json,hashlib,re
R=Path.cwd();D=R/'audits/w14_eligibility_2026-10-29';o=json.loads((D/'observed_aggregate.json').read_text());a=o['aggregate'];h=hashlib.sha256(json.dumps(a,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest();assert h==o['canonical_sha256'];assert o['remote_final_readback_passed'] and o['actual_exits']['08']==1 and o['actual_exits']['08b']==0
assert a['production_artifacts']==36 and a['eligibility_registry_rows']==a['source_union_person_visits']*11==224477;assert a['independent_checks']['person_candidate_rows_rebuilt']==sum(x['base_E'] for x in a['counts'] if x['analysis_id']!='Daily_CGM');assert a['independent_checks']['medication_states_rebuilt']==a['independent_checks']['person_candidate_rows_rebuilt']*21;assert a['independent_checks']['daily_rows_source_matched']==123266
for row in a['counts']:assert 0<=row['ge7_diagnostic']<=row['window_ge1']<=row['base_E'];assert 0<=row['age_sex_smoking_candidate']<=row['window_ge1']
for row in a['medication_window_counts']:assert row['unknown']==676760 and set(row)=={'window','unknown'}
for p in D.glob('*.py'):compile(p.read_bytes(),str(p),'exec')
for p in D.glob('source_manifest_*.json'):
 m=json.loads(p.read_text());assert hashlib.sha256((D/m['script']).read_bytes()).hexdigest()==m['sha256']
for name in ['04_build_exit0.png','05_independent_exit0.png','06_publish_exit0.png','07_final_readback_exit0.png','08_append_guard_exit1.png','08b_complete_exit0.png']:assert (D/name).stat().st_size>10000
p=R/'outputs/researchwrite/hpp_multisystem_20261021/team_checklist_20261026/workpackage_progress.json';d=json.loads(p.read_text());assert d['tasks']['W14']['checked'] and d['latest_local_workpackage']=='W14' and d['next_recommended_workpackage']=='W15' and not d['tasks']['W14']['human_review_complete']
for p in ['reports/HPP_CURRENT_PLAN.md','reports/HPP_Team_Execution_Checklist_2026-10-26.md','reports/HPP_Project_TODO_v3_2026-10-23.md','reports/HPP_Project_Roadmap_v3_2026-10-23.md','HPP_LOCAL_VERIFIED_FACTS.md','HPP_PROJECT_BACKGROUND.md','HPP_DATA_BACKGROUND.md','HPP_SERVER_BACKGROUND.md']:
 s=(R/p).read_text();assert '当前实际进度（W14' in s[:1500];assert (D/'before_w14_update'/p).exists()
s=(R/'reports/HPP_Team_Execution_Checklist_2026-10-26.md').read_text();assert '- [x] **W14' in s and '- [ ] **W15' in s
report=R/'reports/HPP_W14_Eligibility_Cohort_Flow_2026-10-29.md';s=report.read_text()
for n in ['5,876','4,338','4,284','4,591','4,597','6,305','103,925','全部为unknown','08b','最终模型N','未训练模型']:assert n in s,n
b=json.loads((D/'html_build.json').read_text());assert b['source_sha256']==hashlib.sha256(report.read_bytes()).hexdigest();ht=report.with_suffix('.html').read_text();assert '<html lang="zh-CN">' in ht and ht.count('<tr>')==len(a['counts'])+1 and 'W14' in ht;assert not b['external_resources'] and not b['browser_visual_QA']
for rel in ['reports/HPP_Team_Execution_Checklist_2026-10-26.html','reports/HPP_最新版方案_双路线与工作路径_v3_2026-10-23.html']:assert 'W14技术交付已完成，下一步W15' in (R/rel).read_text()
# Existing scientific register is not silently rewritten into a finalized protocol.
reg=json.loads((R/'reports/HPP_Cohort_Analysis_Register_v3_2026-10-23.json').read_text());assert all(m['actual_q'] is None and m['actual_rank'] is None and m['final_n'] is None for m in reg['modules'])
files=[]
for p in sorted(D.iterdir()):
 if p.is_file():files.append({'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(D/'local_artifact_manifest.json').write_text(json.dumps(files,ensure_ascii=False,indent=2)+'\n');out={'passed':True,'aggregate_sha256_matches_remote_screen':h,'remote_actual_exits':o['actual_exits'],'all_transferred_python_source_hashes_verified':True,'count_conservation':True,'screenshots':6,'W14_checked_W15_next':True,'formal_N_and_q_rank_not_frozen':True,'unknown_medication_not_filled':True,'originals_snapshotted':True,'report_HTML_source_hash_verified':True,'HTML_browser_visual_QA':False,'individual_data_exported':False};(D/'local_delivery_validation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False))
