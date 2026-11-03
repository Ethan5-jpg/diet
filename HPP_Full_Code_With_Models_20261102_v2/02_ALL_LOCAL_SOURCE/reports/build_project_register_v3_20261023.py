#!/usr/bin/env python3
"""Generate design-only V3 registry. Reads no participant or server data."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VERSION = "multisystem_v3_20261023"
SHARED = ["diet", "oral_microbiome", "gut_microbiome"]
DOMAIN_SPECS = [
    ("Glu", "cgm", "food_cgm_common", True),
    ("TotalFat", "dxa_total_fat", "food_window", False),
    ("VAT", "dxa_vat", "food_window", False),
    ("Liver", "liver_fat_related", "food_window", False),
    ("BMI", "anthropometric_bmi", "food_window", False),
]
COUNTS = ["base_n", "window_n", "covariate_n", "support_n", "final_n",
          "discovery_n", "confirmation_n", "family_n", "person_days",
          "within_person_information_n"]


def module(analysis_id, sources, view, m, role, overlap):
    return {
        "analysis_id": analysis_id, "role": role,
        "required_sources": sources,
        "requires_all_four_outcomes": analysis_id == "Common4",
        "requires_food_cgm_overlap": overlap,
        "diet_view": view, "output_count": m,
        "minimum_valid_days_candidate": 7,
        "day_definition": "valid_common_day" if overlap else "valid_food_day",
        "score_count_max_candidate": 2, "actual_q": None, "actual_rank": None,
        "counts": {name: None for name in COUNTS}, "final_n": None,
        "microbiome_time_policy_ref": "same_visit_primary",
        "eligibility_flags": ["base_eligible", "window_eligible", "covariate_eligible", "support_eligible"],
        "role_assignment_changes_after_support_filter": False,
        "freeze_status": "design_specified_data_acceptance_and_inferential_freeze_pending",
    }


modules, base_sets = [], []
for domain, outcome, view, overlap in DOMAIN_SPECS:
    base = f"E_{domain}"
    base_sets.append({
        "base_eligibility_id": base, "required_sources": SHARED + [outcome],
        "sample_identity_visit_and_necessary_field_qc": True,
        "inherits_other_module_score_support_exclusions": False,
        "window_covariate_and_score_support_are_separate_flags": True,
        "n": None,
    })
    row = module(f"D_{domain}", SHARED + [outcome], view, 1,
                 "optional_secondary" if domain == "BMI" else "domain_primary", overlap)
    row.update({"base_eligibility_id": base, "score_scope": f"D_{domain}_discovery_only"})
    modules.append(row)

for pair, parents, outcomes, view, overlap in [
    ("Glu_Liver", ["E_Glu", "E_Liver"], ["cgm", "liver_fat_related"], "food_cgm_common", True),
    ("VAT_TotalFat", ["E_VAT", "E_TotalFat"], ["dxa_vat", "dxa_total_fat"], "food_window", False),
]:
    row = module(f"P_{pair}", SHARED + outcomes, view, 2, "prespecified_pair_comparison", overlap)
    row.update({
        "candidate_parent_intersection": parents,
        "candidate_parent_layer": "base_E_not_final_D_support_sets",
        "revalidate_common_window_and_support": True,
        "inherits_independent_domain_score_support": False,
        "score_scope": f"P_{pair}_shared_pair_discovery",
        "same_person_window_sample_score_dictionary_scale_reference_required": True,
        "independent_domain_score_coefficients_may_be_subtracted": False,
        "domain_difference_restriction_count": "actual_identifiable_q_not_2q",
    })
    modules.append(row)

row = module("Common4", SHARED + ["cgm", "dxa_total_fat", "dxa_vat", "liver_fat_related"],
             "food_cgm_common", 4, "joint_extension_not_domain_gate", True)
row.update({"candidate_parent_intersection": ["E_Glu", "E_TotalFat", "E_VAT", "E_Liver"],
            "candidate_parent_layer": "base_E_not_final_D_support_sets",
            "revalidate_common_window_and_support": True,
            "gates_other_modules": False, "score_scope": "Common4_shared_discovery"})
modules.append(row)
row = module("Daily_CGM", SHARED + ["cgm"], "food_cgm_common_daily", 1,
             "within_person_support", True)
row.update({"base_eligibility_id": "E_Glu", "score_scope": "frozen_D_Glu_scores_only_primary",
            "primary_score_source": "D_Glu", "primary_score_count": "q_D_Glu",
            "requires_body_or_liver_labels": False,
            "additional_requirement": "within_person_exposure_variation_and_residual_design_rank",
            "default_day_weight": "1/n_valid_days_per_person",
            "fixed_effects_absorb_time_invariant_main_effects": True,
            "validates_pair_score_automatically": False,
            "pair_score_daily_application": "registered_exploratory_extension_unless_alpha_reallocated_before_first_confirmation"})
modules.append(row)

hypotheses = []


def hypothesis(hypothesis_id, analysis_id, null, restrictions, source, kind):
    hypotheses.append({
        "hypothesis_id": hypothesis_id, "analysis_id": analysis_id, "kind": kind,
        "null_hypothesis": null, "contrast_matrix_spec": source,
        "restriction_count_rule": restrictions,
        "actual_contrast_matrix": None, "actual_rank": None,
        "status": "not_run", "raw_p": None, "adjusted_p": None,
        "inactive_calculation_value": 1.0,
        "inactive_requires_reason_and_no_estimate_or_biological_zero_claim": True,
        "exact_duplicates_retained_in_fixed_15_positions": True,
    })


for d in ["Glu", "TotalFat", "VAT", "Liver"]:
    hypothesis(f"H_{d}", f"D_{d}", "all_available_gamma_zero", "rank_of_full_identifiable_interaction_block",
               "identity_selector_on_all_D_domain_gamma_columns", "domain_joint_zero")
for pair in ["Glu_Liver", "VAT_TotalFat"]:
    hypothesis(f"P_{pair}_domain_difference", f"P_{pair}", "(1,-1)L_Theta_zero_for_shared_scores",
               "rank_of_q_difference_contrasts_not_2q",
               "I_q tensor ([1,-1] diag(1/sigma_1,1/sigma_2)) applied to vectorized_Theta_with_declared_order",
               "pair_domain_difference")
for d in ["Glu", "TotalFat", "VAT", "Liver"]:
    for site, other in [("oral", "gut"), ("gut", "oral")]:
        hypothesis(f"{d}_{site}_given_{other}", f"D_{d}", f"gamma_{site}_block_zero_conditioning_on_other_site_and_same_main_effects",
                   f"identifiable_retained_{site}_block_rank_zero_if_site_absent",
                   f"selector_for_{site}_gamma_in_joint_D_domain_model", "site_conditional_block")
hypothesis("Daily_CGM_eta_zero", "Daily_CGM", "all_eta_for_frozen_D_Glu_scores_zero",
           "rank_of_identifiable_within_person_score_interaction_block",
           "identity_selector_on_eta_with_D_Glu_mapping_only", "within_person_joint_zero")

registry = {
    "version": VERSION, "date": "2026-10-23", "author": "Codex_implementation_of_Astra_review_per_user_authorization",
    "basis_review_version": "cohort_specific_v2_same_visit_2026-10-21",
    "basis_review": "Astra_Latest_Plan_Full_Review_2026-10-22.md",
    "current_version_full_astra_rereview_run": False,
    "status": "design_implemented_data_qc_calibration_and_confirmation_freeze_pending",
    "data_evidence_cutoff": "2026-10-18", "counts_recomputed": False,
    "server_connected_this_run": False, "new_models_or_simulations_run": False,
    "base_eligibility_sets": base_sets, "modules": modules,
    "global_person_family_split": True,
    "global_split_population": "union_of_base_E_sets_and_daily_candidates_before_score_support_filtering",
    "split_policy": {
        "history_access_ledger_required": True,
        "preassigned_person_family_random_order": True,
        "optional_design_core": "predeclared_discovery_only_no_return_to_confirmation",
        "final_role_and_ratio_freeze_before_real_supervised_score_learning": True,
        "after_score_learning_role_reassignment": False,
        "all_confirmatory_modules_freeze_before_first_confirmation": True,
        "confirmation_access_includes_association_loss_daily_and_selective_plots": True,
        "late_modules": "prespecified_inactive_with_fixed_hypothesis_positions_or_exploratory_after_first_unblinding",
        "current_split_fraction": None,
    },
    "cohort_specific_source_failures_do_not_block_unrelated_modules": True,
    "score_count_max_candidate": 2,
    "microbiome_time_policy": {
        "policy_id": "same_visit_primary", "mode": "same_person_same_cohort_same_visit",
        "visit_assignment_must_be_verified": True,
        "source_stage_semantics_and_available_visit_provenance_required": True,
        "independent_visit_id_assumed_to_exist": False,
        "sample_identity_and_qc_must_pass": True,
        "fixed_max_lag_days": None, "require_sampling_before_diet_or_outcome": False,
        "require_oral_gut_same_calendar_date": False,
        "require_unique_sampling_date_per_site_visit": False,
        "require_exact_sampling_date_for_main_eligibility": False,
        "multiple_samples": "fixed_verified_processing_version_then_validated_technical_qc_priority_then_stable_sample_id",
        "actual_processing_version": None, "actual_qc_priority_columns": None,
        "sample_ids_fixed_across_analysis_modules": True,
        "default_choose_nearest_to_outcome": False,
        "missing_dates": "unknown_retained_if_identity_visit_qc_verified_never_imputed_zero",
        "lag_for_wrong_visit_review_not_automatic_exclusion": True,
        "date_description_required_by_module_with_denominators": True,
        "time_sensitivity": {"required": False, "gates_main_analysis": False, "prespecify_if_performed": True},
        "relative_stability": "working_assumption_not_HPP_validated",
        "primary_interpretation": "retrospective_observed_same_visit_marker",
        "food_cgm_true_date_alignment_unchanged": True,
        "historical_counts_not_relabelled": True,
    },
    "estimand_contract": {
        "scientific_estimand": "conditional_mean_diet_contrast_heterogeneity_in_supported_reference_population",
        "operational_projection_scope": "person_level_state_pair_and_Common4_modules_only",
        "operational_projection": "population_OLS_projection_in_analysis_specific_eligible_population_given_frozen_discovery_dictionary_and_scores",
        "daily_operational_projection": "within_person_fixed_effects_weighted_projection_with_day_weight_1_over_n_i_given_frozen_D_Glu_scores_and_daily_dictionary",
        "stronger_interpretation_requires": ["model_adequacy", "joint_support", "explicit_information_time_and_selection_assumptions"],
        "theta_native": "gamma_times_frozen_score_high_low_difference_given_u_high_minus_low_equals_one",
        "score_identifier": "analysis_id:site:score_version",
        "native_unit_point_estimates_and_descriptive_intervals_required": True,
        "causal_intervention_or_prospective_deployment_claims": False,
    },
    "multiplicity": {
        "default_mode": "15_block_Holm_with_descriptive_pointwise_CI_only",
        "method": "Holm", "alpha": 0.05, "hypothesis_count": 15, "hypotheses": hypotheses,
        "Common4_root_gate": False, "independent_domain_p_gates_pair": False,
        "rank_failure_policy": "after_q_and_hypothesis_freeze_residual_interaction_or_covariance_rank_failure_marks_affected_hypothesis_inactive_with_p_one_no_generalized_inverse_or_posthoc_hypothesis_change",
        "drop_inactive_hypotheses_after_results": False,
        "deduplicate_exact_hypotheses_in_default_family": False,
        "interval_policy": {
            "default": "pointwise_95_percent_descriptive_not_formal_component_confirmation",
            "report_complete_predeclared_component_list": True,
            "ordinary_CI_excluding_zero_or_threshold_is_additional_confirmation": False,
            "block_rejection_authorizes_component_claim": False,
            "formal_component_direction_threshold_equivalence_claims_enabled": False,
            "component_inventory_rule": "domain_Theta_all_retained_q; pair_standardized_difference_all_q; Daily_eta_all_q; extra_pair_native_Theta_is_separate_parameter_if_formal",
            "K_for_listed_inventory_only": "sum_four_domain_q + sum_two_pair_q + q_Daily",
            "K_if_each_q_equals_2": 14,
            "optional_pre_unblinding_upgrade": {
                "enabled": False, "mode": "Holm_blocks_plus_Bonferroni_component_intervals",
                "alpha_total": 0.05, "alpha_blocks_candidate": 0.04,
                "alpha_component_family_candidate": 0.01,
                "per_component_interval_coverage": "1-alpha_component_family/K_frozen",
                "extra_components_require_recount_K": True,
                "selection_after_confirmation_results": False,
            },
        },
        "additional_confirmatory_claims": "preallocate_alpha_before_first_confirmation_otherwise_exploratory",
    },
    "routes": {
        "journal": {"status": "priority_scientific_discovery_design_ready_for_data_acceptance",
                    "four_domain_primary_status_unchanged": True,
                    "narrative_focus": "P_Glu_Liver_then_P_VAT_TotalFat_without_outcome_selection_or_alpha_change"},
        "conference_methods": {"status": "candidate_G0_G1_then_G2_before_full_development",
                               "candidate_question": "weak_reference_association_and_common_direction_structure_under_honest_learned_scores",
                               "hypothesis_must_distinguish": ["fixed_reference_direction", "unknown_common_direction_rank_b_Theta", "proportional_to_average_b"],
                               "counterexamples_required": ["b_zero_positive_negative_strength_cancellation", "q_one_rank_Theta_tautology"],
                               "blocks_journal_preparation": False},
        "conference_agent": {"status": "separate_translation_candidate_information_prediction_action_evaluation_contract_pending",
                             "selected_as_final_conference_type": False,
                             "health_benefit_from_retrospective_replay_claim": False,
                             "blocks_journal_preparation": False},
    },
    "acceptance_gates": {"data_qc_passed": False, "statistical_calibration_passed": False,
                         "confirmation_precision_passed": False, "protocol_inferentially_frozen": False,
                         "confirmation_unblinded_this_run": False},
}

assert len(modules) == 9 and len(hypotheses) == 15
assert len({h["hypothesis_id"] for h in hypotheses}) == 15
assert all(p.startswith("E_") for m in modules for p in m.get("candidate_parent_intersection", []))
target = ROOT / "HPP_Cohort_Analysis_Register_v3_2026-10-23.json"
target.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"registry": str(target), "modules": len(modules), "hypotheses": len(hypotheses),
                  "data_or_models_read": False}, ensure_ascii=False))
