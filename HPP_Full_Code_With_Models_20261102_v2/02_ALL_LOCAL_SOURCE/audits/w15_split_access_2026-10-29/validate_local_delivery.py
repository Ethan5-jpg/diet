from pathlib import Path
import json,hashlib,re
from html.parser import HTMLParser
D=Path(__file__).resolve().parent;W=D.parent.parent;checks=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ok(n,x):
 assert x,n
 checks.append(n)
a=json.loads((D/'observed_aggregate.json').read_text());digest=hashlib.sha256(json.dumps(a,sort_keys=True,separators=(',',':')).encode()).hexdigest();ok('screen_aggregate_digest',digest=='ac54b8aad00e6228c107e6e781306b9243b864ebba5d071faaa1b64c8e0d5284')
e=json.loads((W/'config/w15_split_access_20261029/execution_status.json').read_text());ok('five_observed_exit_zero',e['actual_exit_codes']=={str(i).zfill(2):0 for i in range(1,6)});ok('no_individual_data_export',e['individual_data_exported'] is False)
ok('family_count_conservation',a['candidate_units']==a['candidate_multi_person_units']+a['candidate_singleton_units'] and a['candidate_people']==a['people_in_multi_person_candidate_units']+a['candidate_singleton_units']);ok('module_count_conservation',len(e['module_membership_counts'])==11 and sum(e['module_membership_counts'].values())==54151)
ok('new_scientific_access_closed',a['new_scientific_Y_access_opened_people']==a['design_core_selected_people']==a['actual_final_roles_assigned_people']==0 and a['final_split_fraction'] is None);ok('history_limit_preserved',a['history_exact_ID_overlap']=='unknown' and a['historically_unexposed_validation_established_people']==0 and a['actor_declarations']==4 and a['history_event_rows']==8)
ok('remote_verified_checks',e['independent_BFS_checks']==61 and e['published_readback_checks']==82 and e['production_artifacts_verified']==10)
for i in range(1,6):
 m=json.loads((D/('source_manifest_'+str(i).zfill(2)+'.json')).read_text());p=D/m['script'];ok('remote_source_'+m['script'],sha(p)==m['sha256'] and p.stat().st_size==m['bytes']);compile(p.read_bytes(),str(p),'exec')
for p in D.glob('*.py'):compile(p.read_bytes(),str(p),'exec')
for name in ['01_preflight_exit0.jpg','02_build_exit0.jpg','03_independent_exit0.jpg','04_publish_exit0.jpg','05_final_readback_exit0.jpg']:ok('proof_'+name,(D/name).read_bytes().startswith(b'\xff\xd8\xff') and (D/name).stat().st_size>10000)
manifest=json.loads((D/'local_delivery_manifest.json').read_text());rp=W/'reports/HPP_W15_Split_Access_Ledger_2026-10-29.md';hp=rp.with_suffix('.html');ok('report_md_hash',sha(rp)==manifest['report_markdown_sha256']);ok('report_html_hash',sha(hp)==manifest['report_html_sha256']);ok('status_json_hash',sha(W/'config/w15_split_access_20261029/execution_status.json')==manifest['status_json_sha256'])
for m in json.loads((D/'before_w15_manifest.json').read_text()):ok('preserved_before_'+m['path'],sha(D/'before_w15_update'/m['path'])==m['sha256'])
for rel in manifest['updated_files']:
 p=W/rel;ok('updated_hash_'+rel,sha(p)==manifest['updated_file_sha256'][rel])
 if p.suffix=='.md':ok('latest_banner_'+rel,'当前实际进度（W15，2026-10-29）' in p.read_text().splitlines()[2])
progress=json.loads((W/'outputs/researchwrite/hpp_multisystem_20261021/team_checklist_20261026/workpackage_progress.json').read_text());ok('current_progress_W15',progress['latest_local_workpackage']=='W15' and progress['next_recommended_workpackage']=='W16' and progress['tasks']['W15']['checked'] is True);ok('later_work_not_claimed',not progress['tasks'].get('W16',{}).get('checked',False) and progress['tasks']['W15']['final_role_freeze_complete'] is False and progress['tasks']['W15']['human_review_complete'] is False)
team=(W/'reports/HPP_Team_Execution_Checklist_2026-10-26.md').read_text();ok('MD_checked_and_card_current','- [x] **W15 ' in team and '**状态：** 待执行。' not in team[team.index('### W15 '):team.index('### W16 ')])
th=(W/'reports/HPP_Team_Execution_Checklist_2026-10-26.html').read_text();ok('HTML_card_current','<p><strong>状态：</strong> 待执行。</p>' not in th[th.index('<details class="task" id="W15"'):th.index('<details class="task" id="W16"')]);ok('HTML_checkbox_checked',bool(re.search(r'type="checkbox" checked=""> <strong>W15 ',th)));ok('HTML_latest_notice_not_sticky','<div data-w15-current="true"' in th)
md=rp.read_text()
for label,target in re.findall(r'\[([^\]]+)\]\(([^)]+)\)',md):
 if not target.startswith(('http:','https:')):ok('local_link_'+label,(rp.parent/target).exists())
class CheckHTML(HTMLParser):
 def __init__(self):super().__init__();self.stack=[]
 def handle_starttag(self,tag,attrs):
  if tag not in ['meta','link','br','hr','img','input']:self.stack.append(tag)
 def handle_endtag(self,tag):
  assert self.stack and self.stack[-1]==tag,('malformed_HTML',tag,self.stack[-3:]);self.stack.pop()
parser=CheckHTML();parser.feed(hp.read_text());ok('HTML_structure_balanced',not parser.stack)
result={'status':'passed','checks':len(checks),'check_names':checks,'aggregate_sha256':digest,'remote_execution_status':'01-05 actual exit 0 visually observed','independent_server_checks':61,'published_readback_checks':82,'no_individual_data_exported':True,'local_first_attempt_note':'Aggregate digest transcription omitted one a; assertion stopped before report/status mutation; corrected from saved remote screenshots, recomputed digest now identical.'};(D/'local_delivery_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('LOCAL_W15_VALIDATION_PASS',len(checks),digest)
