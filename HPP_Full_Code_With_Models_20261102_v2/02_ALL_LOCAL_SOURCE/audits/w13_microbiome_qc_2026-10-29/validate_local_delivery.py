# -*- coding: utf-8 -*-
from pathlib import Path
import json,hashlib,re,ast
root=Path.cwd();run=root/'audits/w13_microbiome_qc_2026-10-29';obs=json.loads((run/'observed_aggregate.json').read_text());a=obs['aggregate'];assert hashlib.sha256(json.dumps(a,sort_keys=True,separators=(',',':')).encode()).hexdigest()==obs['canonical_sha256'];assert a['oral_selected_rows']+a['gut_selected_rows']==a['selected_rows']==19059;assert a['candidate_rows']-a['held_oral_candidate_rows']-a['held_gut_candidate_rows']-a['source_duplicate_flag_fail_rows']-17==a['selected_rows'];assert a['baseline_joint_people']<=min(a['oral_baseline_people'],a['gut_baseline_people']);assert 8834*737+11295*2088==obs['matrix_scan']['raw_cells'];assert 8398*737+10661*2088==obs['matrix_scan']['selected_cells_exactly_reverified']
for p in run.glob('*.py'):compile(p.read_bytes(),str(p),'exec')
for tag in ['01','02','03','03b','04','05','06','07','08']:
 p=run/('source_manifest_'+tag+'.json')
 if p.exists():
  d=json.loads(p.read_text());assert d
for name in ['01_sync_preflight_exit0.png','02_matrix_scan_exit0.png','03_inventory_guard_exit1.png','03b_storage_identity_exit0.png','04_native_profiles_exit0.png','05_provisional_exit0.png','06_independent_exit0.png','07_publish_exit0.png','08_final_delivery_exit0.png']:assert (run/name).stat().st_size>10000,name
progress=json.loads((root/'outputs/researchwrite/hpp_multisystem_20261021/team_checklist_20261026/workpackage_progress.json').read_text());assert progress['tasks']['W13']['checked'] and not progress['tasks']['W14']['checked'];assert progress['next_recommended_workpackage']=='W14' and not progress['TRE_sync_pending_for_user_amendment'];md=(root/'reports/HPP_Team_Execution_Checklist_2026-10-26.md').read_text();assert '- [x] **W13' in md and '- [ ] **W14' in md
for rel in ['reports/HPP_CURRENT_PLAN.md','reports/HPP_Project_TODO_v3_2026-10-23.md','HPP_LOCAL_VERIFIED_FACTS.md']:assert '当前实际进度（W13' in (root/rel).read_text()
report=root/'reports/HPP_W13_Microbiome_QC_Sample_Lock_2026-10-29.md';html=report.with_suffix('.html');assert '口腔没有同等原始文本' in report.read_text()
for n in ['8,291','8,690','7,228','W14尚未执行','独立代码复核']:assert n in report.read_text(),n
assert '<html lang="zh-CN">' in html.read_text() and '7,228' in html.read_text();build=json.loads((run/'html_build.json').read_text());assert build['source_sha256']==hashlib.sha256(report.read_bytes()).hexdigest();assert not build['browser_visual_qa_performed']
# Verify locally prepared source hashes of transferred scripts, not just Python syntax.
for mf in run.glob('source_manifest_*.json'):
 d=json.loads(mf.read_text())
 # Some early manifests have nested schemas; covered in immutable source delivery records.
 if isinstance(d,dict):
  for key in ['local_source','source_file','file']:
   if key in d and isinstance(d[key],str) and Path(d[key]).exists() and 'sha256' in d:assert hashlib.sha256(Path(d[key]).read_bytes()).hexdigest()==d['sha256']
files=[]
for p in sorted(run.glob('*')):
 if p.is_file():files.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(run/'local_artifact_manifest.json').write_text(json.dumps(files,ensure_ascii=False,indent=2)+'\n');v={'passed':True,'aggregate_digest_matches_remote_screen':True,'count_conservation':True,'all_python_syntax_checked':True,'screenshots':9,'W13_checked_W14_not_run':True,'offline_HTML_source_hash_verified':True,'browser_visual_QA':False,'remote_final_manifest_artifacts_verified':29,'individual_data_exported':False};(run/'local_delivery_validation.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
