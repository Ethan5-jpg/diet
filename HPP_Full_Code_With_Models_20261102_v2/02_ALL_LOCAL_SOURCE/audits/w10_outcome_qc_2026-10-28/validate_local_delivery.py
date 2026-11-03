"""Validate local handoff structure/count conservation; not a substitute for TRE QC."""
from pathlib import Path
import json,hashlib,re
from urllib.parse import unquote
R=Path(__file__).resolve().parents[2];A=Path(__file__).parent;D=R/'outputs/researchwrite/hpp_multisystem_20261021/team_checklist_20261026';x=json.loads((A/'observed_aggregate.json').read_text());v=json.loads((A/'delivery_validation.json').read_text());stem='HPP_W10_Outcome_Fields_QC_2026-10-28';p=R/'reports'/f'{stem}.md';md=p.read_text();html=p.with_suffix('.html').read_text();b=json.loads((A/'html_build.json').read_text());broken=[]
for one,two in re.findall(r'\]\((?:<([^>]+)>|([^\s)]+))\)',md):
 t=unquote(one or two)
 if t.startswith(('http:','https:','#','mailto:')):continue
 if not (p.parent/t.split('#')[0]).exists():broken.append(t)
assert not broken,broken
for row in x['counts']:assert row['baseline_source_people']>=row['baseline_eligible_people']>=row['baseline_with_indexed_food_same_visit']>=row['baseline_with_energy_food_same_visit']
assert len(x['counts'])==9 and len({r['outcome_id'] for r in x['counts']})==9
assert x['actual_exits']['02_initial']==1 and all(value==0 for tag,value in x['actual_exits'].items() if tag!='02_initial')
assert v['passed'] and not v['CGM_person_values_materialized'] and not v['manufacturer_QC_claimed']
assert hashlib.sha256(p.read_bytes()).hexdigest()==b['source_sha256']
assert '<script src=' not in html and all(str(r['baseline_eligible_people']) in md.replace(',','') for r in x['counts'])
assert json.loads((D/'validation.json').read_text())['passed']
assert json.loads((D/'workpackage_progress.json').read_text())['tasks']['W10']['checked']
for f in A.glob('*.py'):compile(f.read_bytes(),str(f),'exec')
manifest=[]
for f in sorted(A.iterdir()):
 if f.is_file() and f.name not in ['local_artifact_manifest.json','local_delivery_validation.json']:manifest.append({'path':str(f.relative_to(R)),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
(A/'local_artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
result={'passed':True,'broken_local_links':broken,'count_monotonicity':True,'nine_independent_static_outcomes':True,'failed_initial_probe_retained':True,'final_actual_exit_observed':0,'local_source_syntax_passed':True,'HTML_source_hash_verified':True,'checklist_and_protected_append_chain_passed':True,'individual_data_downloaded':False,'browser_visual_QA_claimed':False,'scope':'local artifact integrity only; actual source-value verification occurred inside TRE via05'};(A/'local_delivery_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
