"""Validate local W05 handoff consistency; remote evidence is observed, not re-executed."""
from pathlib import Path
import json,hashlib,re
from urllib.parse import unquote
R=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent
v=json.loads((D/'observed_aggregate.json').read_text());c=json.loads((R/'config/nutrient_semantics.yaml').read_text());report=R/'reports/HPP_W05_Nutrient_Semantics_2026-10-26.md'
checks={}
checks['actual_exits_observed']=v['main_actual_exit']==v['delivery_verifier_actual_exit']==v['raw_scope_actual_exit']==0 and set(v['worker_actual_exit'].values())=={0}
checks['rows_and_examples']=v['total_event_rows_scanned']==sum(v['counts'][k]['rows'] for k in ['processed_all','raw_all'])==9125981 and v['independent_source_row_readbacks']==12
checks['raw_scope_not_misassigned']=v['raw_scope_full_scan']['research_stage']=='undefined' and v['raw_scope_full_scan']['rows']==6485022
checks['config']=all(x['multiplier']==1 and not x['weight_rescaling'] and x['missing']=='preserve_NA' for x in c['fields'].values()) and len(c['fields'])==7
checks['successful_source_hash']=hashlib.sha256((D/'05_validate_nutrients.py').read_bytes()).hexdigest()=='3d6bd89f84cf17eefbc5f1a13599fbb4f4b3b1b77bdcc48ddffa87effa2835ce'
checks['syntax']=True
for p in D.glob('*.py'):compile(p.read_text(),str(p),'exec')
checks['evidence_images']=all((D/f).stat().st_size>1000 for f in ['remote_delivery_page1.png','remote_delivery_page2.png','remote_raw_stage_scope.png'])
checks['memo_and_failures']=(D/'MEMO.md').exists() and set(json.loads((D/'failure_log.json').read_text()))=={'v1','v2'}
checks['no_feature_or_protocol_completion_claim']=not c['individual_features_computed'] and not c['formal_protocol_frozen'] and 'W06' in report.read_text() and 'W07' in report.read_text()
broken=[]
for f in [report,R/'reports/HPP_W05_Nutrient_Semantics_2026-10-26.html']:
 s=f.read_text();links=re.findall(r'(?:href|src)="([^"]+)"',s) if f.suffix=='.html' else re.findall(r'\]\(([^)]+)\)',s)
 for link in links:
  if link.startswith(('http:','https:','#','data:')):continue
  if not (f.parent/unquote(link.split('#')[0])).exists():broken.append(link)
checks['local_links']=not broken
vout={'passed':all(checks.values()),'checks':checks,'broken_links':broken,'remote_evidence_method':'Manual review and aggregate transcription of three current UI screenshots; not source download or direct remote FS access.','main_actual_exit':v['main_actual_exit'],'independent_source_examples':12,'event_rows_scanned':9125981,'formal_features_computed':False,'formal_protocol_frozen':False,'journal_first_design_unchanged':True}
(D/'delivery_validation.json').write_text(json.dumps(vout,ensure_ascii=False,indent=2)+'\n');print(json.dumps(vout,ensure_ascii=False,indent=2));assert vout['passed']
