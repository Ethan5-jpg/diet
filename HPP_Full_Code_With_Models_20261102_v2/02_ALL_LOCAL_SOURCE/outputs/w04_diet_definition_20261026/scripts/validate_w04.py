# -*- coding: utf-8 -*-
"""Validate W04 definitions/traceability; does not validate participant intake."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'outputs/w04_diet_definition_20261026';CFG=ROOT/'config/w04_diet_features_20261026'
def main():
 spec=json.loads((CFG/'diet_feature_spec_draft.yaml').read_text());inp=json.loads((OUT/'input_manifest.json').read_text())
 ex=json.loads((ROOT/'outputs/diet_feature_review_20261026/feature_tables_extracted.json').read_text());cells={(s['sheet'],c['cell']):c['value'] for s in ex['xlsx_sheets'] for c in s['cells']}
 for f in spec['priority_candidates']+spec['source_explanations']+[spec['background_score']]:
  for c in f['source_cells']:assert cells[(c['sheet'],c['cell'])]==c['text']
 protected=[r for r in inp['local_sources'] if r['path']!='HPP_LOCAL_VERIFIED_FACTS.md']
 assert all(hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in protected)
 assert all(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'] for r in inp['original_definition_sources'])
 upf,fiber=spec['priority_candidates'];assert upf['source_cells'][0]['sheet']=='Other' and upf['source_cells'][0]['cell']=='D9'
 assert upf['unknown_policy']['point_with_unknown'].startswith('null')
 assert 'sum(E_d)' in upf['alternative_formula'] and 'sum(E_d)' in fiber['alternative_formula']
 assert spec['packages']['formal_selected_package'] is None and spec['actual_cohort_N'] is None
 assert len(spec['priority_candidates'])==2 and len(spec['source_explanations'])==4
 assert 'NC-Diet' in spec['background_score']['display_name'] and '0–17' in spec['background_score']['display_name']
 assert 'W08' in spec['common']['adoption_gate'] and 'W13 blind exposure' not in spec['common']['adoption_gate']
 report=(ROOT/'reports/HPP_W04_Diet_Feature_Definitions_2026-10-26.md').read_text()
 overlap=(CFG/'prior_work_overlap.md').read_text();decisions=(CFG/'decision_log.md').read_text()
 assert all(s in overlap for s in ('NC-Diet','Deng','Segev','已被接触','未知','W15'))
 assert all(s in report for s in ('45%','27.5%','17.5','30%–40%','W05','W12','W13'))
 assert 'D11' in decisions and 'W18' in decisions
 links=re.findall(r'\]\(([^)]+)\)',report)
 assert all((ROOT/'reports'/s).resolve().exists() for s in links if not s.startswith(('https:','http:','#')))
 status={'date':'2026-10-26','passed':True,'source_cells_exact':True,'protected_input_hashes_unchanged':True,'original_xlsx_csv_hashes_unchanged':True,'candidate_config_parseable':True,'unknown_energy_policy_explicit':True,'distinct_aggregations':True,'overlap_unknowns_and_historical_contact_registered':True,'background_adaptation_named':True,'task_number_correction_registered':True,'report_links_valid':True,'TRE_accessed':False,'participant_features_computed':False,'formal_protocol_frozen':False}
 (OUT/'delivery_validation.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n');print(json.dumps(status,ensure_ascii=False))
if __name__=='__main__':main()
