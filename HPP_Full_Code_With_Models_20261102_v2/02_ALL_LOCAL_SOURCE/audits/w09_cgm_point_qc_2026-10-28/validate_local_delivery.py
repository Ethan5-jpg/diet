# -*- coding: utf-8 -*-
from pathlib import Path
import json,re,hashlib
from html.parser import HTMLParser
from urllib.parse import unquote
A=Path(__file__).resolve().parent;ROOT=A.parents[1];M=ROOT/'reports/HPP_W09_CGM_QC_W08_Common_Days_2026-10-28.md';H=M.with_suffix('.html');r=json.loads((A/'observed_aggregate.json').read_text());b=r['W08_counts']['baseline_10k'];text=M.read_text();html=H.read_text()
assert r['W09_verified'] and r['W08_verified'] and set(r['actual_exit_codes'].values())=={0}
assert r['prior_failed_runs']['03_initial_actual_exit']==1
assert r['all_fragments_verified']==9497 and r['point_accounting']['input_points']==10089326
for name in ['indexed_common','energy_common','fiber_common','UPF_zeroU_common','UPF_strict_common']:
 z=b[name];assert str(format(z['people'],',')) in text and str(format(z['days'],',')) in text
assert sum(x['removed_days_at_this_step'] for x in r['CGM_day_flow'])==599
assert r['legacy_difference_audit']['baseline_W08_days_unaffected_by_extra_CGM_days'] and r['legacy_reconciliation']['both_means_available']==104845
assert '<script src=' not in html and 'W09' in html and 'W08' in html
hb=json.loads((A/'html_build.json').read_text());assert hb['source_sha256']==hashlib.sha256(M.read_bytes()).hexdigest()
broken=[]
for a,b in re.findall(r'\]\((?:<([^>]+)>|([^\s)]+))\)',text):
 t=unquote(a or b)
 if t.startswith(('http:','https:','#','mailto:')):continue
 p=Path(t.split('#')[0]);p=p if p.is_absolute() else M.parent/p
 if not p.exists():broken.append(t)
assert not broken,broken
x={'passed':True,'report_source_hash_matches':True,'aggregate_counts_present':True,'no_individual_data_exported':not r['individual_data_exported'],'local_links_valid':True,'actual_success_and_preserved_failure_distinguished':True,'browser_visual_QA_performed':False,'individual_curve_visual_review_performed':False,'protocol_freeze_claimed':False};(A/'local_delivery_validation.json').write_text(json.dumps(x,indent=2));print(json.dumps(x,indent=2))
