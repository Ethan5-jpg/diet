"""Check local aggregate conservation and artifacts; no access to private TRE rows."""
from pathlib import Path
from urllib.parse import unquote
import hashlib,json,re
R=Path(__file__).resolve().parents[2];A=Path(__file__).parent
D=R/'outputs/researchwrite/hpp_multisystem_20261021/team_checklist_20261026'
x=json.loads((A/'observed_aggregate.json').read_text());v=json.loads((A/'delivery_validation.json').read_text())
w=x['W11'];s=x['W12'];broken=[]
assert sum(w['baseline_smoking'].values())==sum(w['baseline_oral_health_frequency'].values())==w['baseline_union_people']==11124
assert all(r['available']+r['missing_unavailable']==11124 for r in w['baseline_coverage'])
assert all(r['available_coded_state']+r['unknown']==11124 for r in w['baseline_medication_code_availability'])
assert sum(w['medication_start_precision'].values())==sum(w['medication_start_vs_food'].values())==w['valid_key_medication_rows']==42618
assert w['family_edges']+w['household_edges']==w['directed_family_household_edges']==1780
assert s['oral_resolved_abundance_all_stages']+s['oral_unresolved_abundance_unknown_stage']==s['oral_abundance_rows']==8834
assert s['oral_invalid_index_rows']+s['oral_valid_index_only_rows']==s['oral_index_only_rows']==418
assert s['oral_abundance_rows']+s['oral_index_only_rows']==s['oral_candidate_rows']==9252
assert s['gut_index_other_stage_only']+s['gut_person_absent_primary_index']+s['gut_repeated_profile_rows_needing_specimen_link']==s['gut_held_rows']==649
assert s['gut_identity_ready_all_stages']+s['gut_held_rows']==s['gut_abundance_rows']==11295
assert s['oral_abundance_rows']+s['gut_abundance_rows']==s['abundance_candidate_rows']==20129
assert s['oral_candidate_rows']+s['gut_abundance_rows']==s['candidate_rows']==20547
assert 0<s['baseline_joint_identity_ready_people_not_final_QC']<=min(s['oral_baseline_identity_ready_people'],s['gut_baseline_identity_ready_people'])
assert len(x['actual_exits'])==12 and all(i==0 for i in x['actual_exits'].values())
assert x['verification']['script10_abundance_lag_rows']+x['verification']['script12_index_only_lag_rows']==20547
assert not s['W13_QC_done'] and s['final_samples_selected']==0 and not x['model_fitted']
stem='HPP_W11_W12_Covariates_Sample_Candidates_2026-10-29';p=R/'reports'/f'{stem}.md';md=p.read_text();html=p.with_suffix('.html').read_text()
for doc in [p,A/'MEMO.md']:
 for a,b in re.findall(r'\]\((?:<([^>]+)>|([^\s)]+))\)',doc.read_text()):
  t=unquote(a or b)
  if t.startswith(('http:','https:','#','mailto:')):continue
  if not (doc.parent/t.split('#')[0]).exists():broken.append(t)
assert not broken,broken
assert hashlib.sha256(p.read_bytes()).hexdigest()==json.loads((A/'html_build.json').read_text())['source_sha256']
assert '<script src=' not in html
for source in A.glob('*.py'):compile(source.read_bytes(),str(source),'exec')
for i in range(1,13):
 m=json.loads((A/f'source_manifest_{i:02}.json').read_text())
 source=A/m['script']
 assert source.stat().st_size==m['bytes']
 assert hashlib.sha256(source.read_bytes()).hexdigest()==m['sha256']
assert all((R/e).is_file() and (R/e).stat().st_size>1000 for e in v['evidence'])
progress=json.loads((D/'workpackage_progress.json').read_text())
assert all(progress['tasks'][w]['checked'] for w in ['W11','W12'])
assert not progress['tasks'].get('W13',{}).get('checked',False)
assert progress['next_recommended_workpackage']=='W13'
assert json.loads((D/'validation.json').read_text())['passed']
manifest=[{'path':str(f.relative_to(R)),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(A.rglob('*')) if f.is_file() and f.name not in ['local_artifact_manifest.json','local_delivery_validation.json']]
(A/'local_artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
result={'passed':True,'aggregate_count_conservation':True,'local_source_syntax_passed':True,'broken_local_links':broken,'HTML_source_hash_verified':True,'checklist_and_protected_append_chain_passed':True,'actual_remote_exits_observed':12,'individual_data_downloaded':False,'W13_QC_claimed':False,'final_samples_selected':0,'browser_visual_QA_claimed':False,'scope':'local artifact verification only; individual source-value verification ran in TRE via10/12'}
(A/'local_delivery_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
