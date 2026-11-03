# -*- coding: utf-8 -*-
"""Cache manually observed TRE aggregate outputs; never copy participant data."""
from pathlib import Path
import json
D=Path(__file__).parent
records=[
('total_fat_mass','总脂肪重量','body_comp_total_fat_mass','g',6671,6595,6209,6178,1),
('vat_volume','内脏脂肪体积','total_scan_vat_volume','cm³',6671,6513,6132,6102,31),
('bmi','BMI','bmi','kg/m²',10402,10382,9065,9034,0),
('liver_elasticity','肝弹性','elasticity_qbox_mean','kPa',6782,6758,6380,6351,0),
('liver_velocity','剪切波速度','velocity_qbox_mean','m/s',6782,6758,6380,6351,0),
('liver_viscosity','黏度','viscosity_qbox_mean','Pa·s',6782,6757,6379,6350,0),
('liver_dispersion','频散','dispersion_qbox_mean','(m/s)/kHz',6782,6757,6379,6350,0),
('liver_attenuation','衰减系数','attenuation_coefficient_qbox','dB/cm/MHz',6782,6758,6378,6349,0),
('liver_speed_of_sound','超声声速','speed_of_sound_qbox','m/s',6782,6764,6384,6355,0)]
rows=[dict(zip(['outcome_id','label','source_field','unit','baseline_source_people','baseline_eligible_people','baseline_with_indexed_food_same_visit','baseline_with_energy_food_same_visit','baseline_zero_measurement_rows'],r)) for r in records]
x={'date':'2026-10-28','provenance':'manual transcription of TRE UI aggregate outputs from scripts04/05/06; no source or individual dataset download','status':'construction_and_independent05_verified_finalizer07_pending_observation','baseline_scope':{'cohort':'10k','research_stage':'00_00_visit'},'counts':rows,'not_final_microbiome_analysis_N':True,'static_outcomes':9,'liver_outcomes':6,'registry_rows_all_stages':84962,'measurement_QC_rows_all_stages':192434,'independent_full_registry_recomputation_passed':True,'build_artifact_hashes_verified':8,'binary_mirrors_verified':3,'actual_exits':{'01':0,'02_initial':1,'02b_index_fix':0,'03':0,'04':0,'05':0,'06':0},'CGM':{'qualified_baseline_people_from_W09':9402,'person_outcome_values_materialized_in_W10':False,'formula_registered':'module-specific qualified common-day daily means, equal day weights','deferred_to':'W14'},'QC_scope':'source finite and mathematical domain, no manufacturer/device flag fields in inspected tables, no invented clinical cutoffs','VAT_zero_policy':'31 baseline zero values retained and flagged; not proof of absence, no logarithm','fat_zero_policy':'1 baseline zero excluded for fat-mass outcome only','BMI_formula':{'all_stage_comparable_rows':12925,'max_absolute_error':0.0,'recomputed_or_imputed':False},'fat_mass_exceeding_body_or_tissue_mass':0,'nonselected_fat_percentage_columns':{'stored_scale':'fraction 0–1 despite dictionary %','ratio_reconciliation_rows':8989,'max_abs_ratio_error_under':5e-10,'not_selected_as_outcome':True},'liver_old_summary_comparison_all_visits':[
{'outcome_id':'liver_elasticity','both_values':8936,'difference_gt_1e_minus_6':5889,'max_abs_difference':1.385,'baseline_comparable':6757,'baseline_difference_gt_1e_minus_6':4491},
{'outcome_id':'liver_velocity','both_values':8936,'difference_gt_1e_minus_6':5963,'max_abs_difference':.392,'baseline_comparable':6757,'baseline_difference_gt_1e_minus_6':4490},
{'outcome_id':'liver_viscosity','both_values':8935,'difference_gt_1e_minus_6':6011,'max_abs_difference':.925,'baseline_comparable':6756,'baseline_difference_gt_1e_minus_6':4547},
{'outcome_id':'liver_dispersion','both_values':8935,'difference_gt_1e_minus_6':6126,'max_abs_difference':1.64,'baseline_comparable':6756,'baseline_difference_gt_1e_minus_6':4604}],
'liver_aggregation':'equal-weight arithmetic mean of valid detail measurements, independently for all six fields within explicit same visit','liver_aggregate_source_limit':'cannot substitute the old published means as identical; most absolute differences small, unknown larger differences kept diagnostic','protocol_frozen':False,'team_human_review_claimed':False,'association_models_run':False,'raw_data_exported':False}
(D/'observed_aggregate.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
print('cached aggregate observations; finalizer still pending')
