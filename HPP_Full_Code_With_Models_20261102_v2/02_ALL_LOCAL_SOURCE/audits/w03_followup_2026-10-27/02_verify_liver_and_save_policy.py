"""Read back liver metadata, print six families/units, persist user-approved rules.
No phenotype values or individual identifiers are loaded.
"""
from pathlib import Path
import json,hashlib,sys
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
O=P/'data_qc/w03_liver_schema_20261027';S=P/'scripts/w03_liver_schema_20261027';L=Path('/home/ec2-user/multisystem_runtime_20261021/w03_liver_schema_20261027')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 b=(json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode();p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);assert p.read_bytes()==b
USER_POLICY=json.loads('{"version": "user_clarification_20261027", "status": "user_authorized_rules_recorded_not_yet_applied_to_analysis_tables", "supersedes_conflicting_earlier_date_admission_assumptions": true, "person_visit_key": ["participant_id", "cohort", "research_stage"], "visit_summary_phenotypes": {"domains": ["body_composition", "liver_ultrasound", "CGM_visit_summary"], "require_matching_valid_person_cohort_stage": true, "require_phenotype_identity_units_and_value_QC": true, "require_exact_collection_timestamp": false, "require_timestamp_parsing": false, "require_fixed_date_gap_or_unique_collection_date": false, "diet_view": "qualified_food_window_in_same_visit", "require_food_CGM_shared_calendar_days_for_visit_summary": false, "retain_dates_as_descriptive_metadata_if_available": true, "observation_window_14_days": "user_recalls_approximately_14_days_not_yet_verified_for_all_records", "no_automatic_replacement_of_existing_common_day_CGM_estimates": true}, "daily_CGM_or_meal_models": {"daily_requires_real_food_CGM_calendar_date_alignment": true, "daily_requires_point_coverage_QC": true, "meal_analysis_requires_validated_ingestion_time": true, "same_visit_does_not_substitute_for_daily_alignment": true}, "microbiome_multiple_samples": {"primary_rule": "randomly_select_one_eligible_biological_sample_per_person_cohort_stage_and_site", "seed": 20261027, "stable_candidate_order": "source_version_and_sample_ID_or_other_verified_unique_sample_identity", "match_index_and_abundance_before_selection": true, "QC_before_random_selection": true, "freeze_once_reuse_same_selection_across_domains": true, "do_not_select_using_phenotype_or_association_results": true, "599_unmatched_gut_keys": "not_resolved_by_random_choice; provenance_still_to_be_classified", "sample_registry_executed": false}, "food_dates": {"primary_calendar_date": "collection_date_agrees_with_Asia_Jerusalem_aware_collection_timestamp", "local_timestamp_not_an_admission_requirement": true, "regional_context": "Israel/Asia_Jerusalem_per_user_and_source_aware_timestamp", "1066_local_calendar_conflicts": "retained_as_diagnostic; not_claimed_resolved_or_automatically_excluded", "event_day_without_index": {"all_source_unmatched_day_keys": 14713, "all_source_matched_day_keys": 149912, "all_source_event_day_keys": 164625, "unmatched_pct_of_event_day_keys": 8.937281700835232, "has_dated_food_events": true, "not_equivalent_to_no_food_or_no_date": true, "user_proposal": "consider_excluding_only_affected_days_if_small", "admission_policy_frozen": false, "next_check": "baseline_day_and_person_impact_and_distribution_before_primary_exclusion; index_matched_only_may_be_a_sensitivity_view"}}, "liver": {"user_identified_summary_file": "/home/ec2-user/studies/hpp_datasets/liver_ultrasound/liver_ultrasound_aggregated.parquet", "six_intended_phenotype_fields": null, "attenuation_priority_per_user": true, "do_not_require_all_six_nonmissing_for_each_single_outcome": true, "six_field_source_and_aggregation_confirmation_pending": true, "no_new_confirmatory_hypothesis_budget_frozen": true}, "execution_scope": "policy_and_metadata_only_no_new_sample_selection_no_individual_feature_or_model_computation"}')
EXPECTED_PROBE_SHA='5e521a40dd620f1082ef6f755d9e9bd677989628d213518ddb27f87560ba6f05'
sources=json.loads((O/'source_metadata_TRE_ONLY.json').read_text());dictionary=json.loads((O/'dictionary_evidence_TRE_ONLY.json').read_text())
assert (L/'actual_exit_code').read_text().strip()=='0'
assert (L/'03_actual_exit_code').read_text().strip()=='0'
inventory=json.loads((O/'all_liver_parquet_inventory_TRE_ONLY.json').read_text())
assert inventory['traversal_complete_within_scope'] and inventory['no_data_rows_read'] and inventory['parquet_or_pq_files']==2
assert hashlib.sha256((S/'03_inventory_liver_parquets.py').read_bytes()).hexdigest()=='33438d200908333f8006d33b780b9cb49468ba0ed865ee2a0808c53ac802dd10'
for s in sources:
 p=Path(s['path']);st=p.stat();assert st.st_size==s['bytes'] and st.st_mtime_ns==s['mtime_ns']
assert sha(S/'01_check_liver_schema.py')==EXPECTED_PROBE_SHA
raw,agg=sources
assert raw['rows']==26881 and agg['rows']==8937
families=['elasticity','velocity','viscosity','dispersion','attenuation','speed_of_sound']
a_names=[x['name'] for x in agg['columns']]
assert not any('attenuation' in c or 'speed_of_sound' in c for c in a_names)
raw_fields=['elasticity_qbox_median','velocity_qbox_median','viscosity_qbox_median','dispersion_qbox_median','attenuation_coefficient_qbox','speed_of_sound_qbox']
unit_evidence={}
for f in raw_fields:
 hits=[e for e in dictionary if e['row'].get('tabular_field_name')==f]
 assert hits,f
 unit_evidence[f]=[{'units':h['row'].get('units'),'label':h['row'].get('field_string'),'description':h['row'].get('description_string')} for h in hits]
C=P/'config/w03_user_clarification_20261027';save(C/'visit_and_sampling_policy.json',USER_POLICY)
(S/'02_verify_liver_and_save_policy.py').write_bytes(Path(__file__).read_bytes())
result={'metadata_readback_passed':True,'probe_actual_exit':0,'probe_code_sha256':EXPECTED_PROBE_SHA,'raw_rows_metadata':raw['rows'],'aggregate_rows_metadata':agg['rows'],'raw_columns':len(raw['columns']),'aggregate_columns':len(agg['columns']),'aggregated_family_column_counts':{f:sum(f in c for c in a_names) for f in families},'aggregate_other_columns':[x for x in agg['columns'] if not any(f in x['name'] for f in families)],'raw_six_family_representative_fields':unit_evidence,'phenotype_values_read':False,'six_final_fields_chosen':False,'sample_selection_executed':False,'recursive_inventory_passed':True,'recursive_inventory_files':2,'inventory_actual_exit':0,'policy_path':str(C/'visit_and_sampling_policy.json')}
save(O/'verification_and_family_inventory.json',result)
memo='''# W03 follow-up: liver metadata and user rules (2026-10-27)\n\n01_check_liver_schema.py: schema/footer + local dictionaries only, no phenotype values.\n02_verify_liver_and_save_policy.py: source metadata fingerprints/code hash/first actual exit readback, six representative raw fields and units, aggregate family inventory, saves user clarification config.\n\nResult: data_qc/w03_liver_schema_20261027. Code: scripts/w03_liver_schema_20261027. Config: config/w03_user_clarification_20261027/visit_and_sampling_policy.json.\n\nCurrent aggregate source has 8937 rows, 36 columns: 10 elasticity, 10 velocity, 6 viscosity, 6 dispersion columns; no attenuation/speed-of-sound columns. Those two are in raw detail (26881 rows, 28 columns). No final six-field selection or new per-visit aggregation was invented.\n\nUser rules recorded: Body/liver/CGM visit-summary use valid person+cohort+stage, without exact date or lag admission; daily/meal CGM remains calendar/ingestion dependent. Multiple eligible biological samples: random one after correct mapping and QC, seed20261027, frozen across domains; not executed. 599 unmatched gut keys remain unresolved. Food missing-index days: 14713/164625=8.937% all-source event day keys, contain dated food events; exclusion is not frozen, not equivalent to no food.\n\n03_inventory_liver_parquets.py recursively verified all Parquet files within the liver_ultrasound directory and its subdirectories: exactly 2, actual exit0, no rows read. No unrelated project-wide search.\n\nNo new participant cohort, biological sample registry, clinical QC, phenotype distributions, tests or models.\n'''
(O/'MEMO.md').write_text(memo);M=P/'memos/w03_liver_schema_20261027';M.mkdir(parents=True,exist_ok=True);(M/'MEMO.md').write_text(memo)
artifacts=[]
for p in [S/'01_check_liver_schema.py',S/'02_verify_liver_and_save_policy.py',O/'source_metadata_TRE_ONLY.json',O/'dictionary_evidence_TRE_ONLY.json',O/'verification_and_family_inventory.json',O/'MEMO.md',C/'visit_and_sampling_policy.json',S/'03_inventory_liver_parquets.py',O/'all_liver_parquet_inventory_TRE_ONLY.json']:
 artifacts.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
save(O/'artifact_manifest.json',{'artifacts':artifacts,'scope':'Metadata and policy, not phenotype validation'})
print('\n'*30+'W03_LIVER_METADATA_READBACK_PASS')
print('RECURSIVE_PARQUET_INVENTORY',inventory['parquet_or_pq_files'],'FILES; ENTRIES',inventory['directory_entries_seen'],'COMPLETE',True,'ACTUAL_EXIT',0)
for r in inventory['sources']:print('FILE_SUMMARY',r['relative'],'ROWS',r['rows_metadata'],'COLUMNS',r['physical_columns'],'LIVER_MEASURE_COLUMNS',r['known_liver_measurement_columns'])
print('RAW',raw['rows'],len(raw['columns']),'AGGREGATE',agg['rows'],len(agg['columns']))
print('AGGREGATE_FAMILY_COUNTS',json.dumps(result['aggregated_family_column_counts']))
print('AGGREGATE_OTHER_COLUMNS',json.dumps(result['aggregate_other_columns']))
for f,ev in unit_evidence.items():print('RAW_FIELD_UNIT',f,json.dumps(ev))
print('CODE_SHA',sha(S/'01_check_liver_schema.py'));print('POLICY_SHA',sha(C/'visit_and_sampling_policy.json'));print('ARTIFACT_HASHES_VERIFIED',len(artifacts));print('FINAL_SIX_FIELDS_NOT_SELECTED');print('NO_PHENOTYPE_VALUES_OR_MODELS');print('RESULT',O);print('W03_LIVER_READBACK_DONE')
