# -*- coding: utf-8 -*-
"""Validate transcription consistency and delivery; cannot revalidate TRE values locally."""
from pathlib import Path
import json,hashlib,re
from urllib.parse import unquote
R=Path(__file__).resolve().parents[2];A=Path(__file__).resolve().parent
j=json.loads((A/'observed_aggregate_summary.json').read_text());b=j['baseline']
assert b['event_days_before_index']-b['event_days_after_index']==b['index_excluded_days']==8468
assert b['people_before_index']-b['people_after_index']==b['people_without_any_indexed_day']==650
assert b['strict_NOVA_people_ge7']==1707 and b['zero_unknown_energy_NOVA_people_ge7']==1714
assert b['fiber_complete_people_ge7']==4662
assert j['external_sources_allowed'] is False and j['not_final_cohort_N'] is True
assert j['actual_exit_codes']=={'01':0,'02':0,'03':0,'04':1,'05':1,'06':0,'07':0,'08':0,'09':0,'10':0}
for p in A.glob('*.payload.json'):
 x=json.loads(p.read_text());src=A/x['filename'];assert hashlib.sha256(src.read_bytes()).hexdigest()==x['sha256'];compile(src.read_text(),str(src),'exec')
spec=json.loads((R/'config/w04_followup_20261027/implementation_reconciliation.json').read_text())
assert spec['formal_primary_exposures'] is None and not spec['test_budget_changed'] and not spec['additional_external_sources_allowed']
md=R/'reports/HPP_W04_Mapping_Six_Issues_2026-10-27.md';html=md.with_suffix('.html');text=md.read_text()
links=[]
for t in re.findall(r'\]\(([^)]+)\)',text):
 if t.startswith(('https:','http:','#')):continue
 p=md.parent/unquote(t);assert p.exists(),t;links.append(t)
hb=json.loads((A/'html_build.json').read_text());assert hb['source_sha256']==hashlib.sha256(md.read_bytes()).hexdigest()
assert '<script src=' not in html.read_text()
assert (A/'09_final_verification.png').is_file() and (A/'10_internal_source_scope_verified.png').is_file()
status={'passed':True,'role':'local artifacts and screen-transcription consistency only; TRE verification is separate','code_payload_hashes_match':True,'local_source_syntax_passed':True,'baseline_flow_consistent':True,'reported_diagnostic_counts_not_final_N':True,'formal_protocol_not_frozen':True,'external_sources_disallowed':True,'report_links_checked':len(links),'HTML_source_hash_match':True,'browser_visual_QA_performed':False}
(A/'local_delivery_validation.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
manifest=[{'path':str(p.relative_to(R)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in A.iterdir() if p.is_file() and p.name!='local_artifact_manifest.json']
(A/'local_artifact_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(status,ensure_ascii=False))
