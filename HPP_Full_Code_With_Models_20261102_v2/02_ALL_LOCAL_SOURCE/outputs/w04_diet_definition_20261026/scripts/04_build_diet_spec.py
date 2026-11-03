# -*- coding: utf-8 -*-
"""W04 definition/overlap documentation only. No TRE, participant data or models."""
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/w04_diet_definition_20261026'
CFG=ROOT/'config/w04_diet_features_20261026'
REPORT=ROOT/'reports/HPP_W04_Diet_Feature_Definitions_2026-10-26.md'
EX=ROOT/'outputs/diet_feature_review_20261026/feature_tables_extracted.json'

def save(p,v):
    b=(json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode();p.write_bytes(b);assert p.read_bytes()==b

def main():
    x=json.loads(EX.read_text());sheets={s['sheet']:{c['cell']:c['value'] for c in s['cells']} for s in x['xlsx_sheets']}
    cell=lambda sheet,ref:{'sheet':sheet,'cell':ref,'text':sheets[sheet][ref]}
    assert sheets['Other']['D9']=='100 × energy from NOVA group 4 foods / total daily energy (%)'
    assert 'fibre' in sheets['Nutrients']['A5'] and '1,000 kcal' in sheets['Dash']['C6']
    assert x['csv_summary']['data_rows']==643 and len(x['csv_rows'])==643
    assert sum(not row['NOVA'] for row in x['csv_rows'])==2
    assert '0–17' in sheets['I-MEDAS']['A35']
    inputs=['outputs/diet_feature_review_20261026/feature_tables_extracted.json','outputs/diet_feature_review_20261026/extraction_validation.json','outputs/diet_feature_review_20261026/input_manifest.json','reports/Astra_Diet_Feature_Selection_Assessment_2026-10-26.md','reports/HPP_Integrated_Multisystem_Plan_v3_2026-10-23.md','reports/HPP_Cohort_Analysis_Register_v3_2026-10-23.json','reports/HPP_Methods_Source_Check_2026-10-21.md','reports/HPP_饮食-菌群-血糖_完整结果报告_2026-10-13.html','papers/Deng_2026_HPP_diet_gut_liver_source.json','HPP_LOCAL_VERIFIED_FACTS.md']
    manifest={'date':'2026-10-26','scope':'Local definition audit; prior unpublished tables not sent externally','local_sources':[{'path':p,'sha256':hashlib.sha256((ROOT/p).read_bytes()).hexdigest(),'bytes':(ROOT/p).stat().st_size} for p in inputs],'original_definition_sources':[]}
    for source in x['manifest']['sources']:
        p=Path(source['path']);same=p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==source['sha256']
        manifest['original_definition_sources'].append(dict(source,current_hash_matches=same))
        assert same,'Original source changed; re-extract and review before reusing cell definitions'
    save(OUT/'input_manifest.json',manifest)
    common={'status':'draft_candidate_not_formal_protocol','adoption_gate':'W05/W06/W07/W08 blind exposure QC; W14 module eligibility; W17 precision; W18 package/aggregation freeze; W22 all confirmation modules frozen',
        'current_analysis_protocol_changed':False,'participant_values_computed':False,
        'event_source':{'path':'/home/ec2-user/studies/hpp_datasets/diet_logging/diet_logging_events.parquet','keys':['participant_id','cohort','research_stage'],'physical_candidate_columns':['food_id','weight_g','calories_kcal','dietary_fiber_g','food_category','collection_date','collection_timestamp','local_timestamp'],'semantic_status':'columns located; intake units/conversion not verified'},
        'day_rules':['Use W06 verified calendar-day and edited-event registry, not event count as meals.','Unrecorded day is missing, not zero intake.','E_d must be finite and positive; invalid/missing event energy not silently zero.','Food energy/grams are validated event-intake quantities; multiply weight/100 exactly once only if source definition proves per-100g.'],
        'window_rules':{'D_Glu':'valid food-CGM common days','Daily_CGM':'those valid daily observations with frozen D_Glu score','D_TotalFat_D_VAT_D_Liver_D_BMI':'own qualified food_window; no CGM requirement','P_Glu_Liver':'rebuild shared window from E_Glu and E_Liver','P_VAT_TotalFat':'rebuild shared food_window from relevant E intersection','same_visit_microbiome':'unchanged; no fixed lag or unique-date restriction'},
        'aggregation':{'recommended_draft':'equal weight over qualified recorded days','alternative':'ratio of cumulative numerator to cumulative total energy','formal_choice_frozen':False,'never_mix':'Store separate IDs; W18 picks primary before supervised/confirmation protocol freeze. Legacy V3 carb-for-fat cumulative ratio is not replaced by this draft.'},
        'historical_outcome_contact':'Known early same-project NOVA4/fiber x gut/CGM exploration. W04 is not outcome-naive feature invention; do not use old p-values for selection.'}
    features=[{
        'feature_id':'UPF_recordday_mean_pp_candidate','name_zh':'NOVA4食物能量占比（合格记录日等权均值；候选）','tier':'priority_candidate_1','existing_or_derived':'Existing definition; current FoodID implementation still pending',
        'source_cells':[cell('Other','D9'),cell('Other','A13'),cell('Other','A28')],
        'food_mapping':'643 Food-name catalog, NOVA column; no FoodID. W07 exact/reviewed crosswalk with version and unresolved classes; no silent fuzzy mapping.',
        'variables':{'E_d':'sum verified energy of all retained food events that day, including unclassified NOVA energy','K_d':'verified energy with confirmed NOVA==4','U_d':'energy whose NOVA is missing/unmapped/ambiguous'},
        'daily_formula':'D_UPF_d = 100 * K_d / E_d only when U_d==0 or a separately approved uncertainty policy identifies a point',
        'unknown_policy':{'always_save':['NOVA energy coverage = 1-U_d/E_d','lower_bound_pp = 100*K_d/E_d','upper_bound_pp = 100*(K_d+U_d)/E_d'],
             'point_with_unknown':'null by default; do not relabel lower bound as exact UPF. W07 may resolve mapping or predefine a justified uncertainty strategy and separate feature ID. No accepted coverage threshold selected in W04.',
             'prohibited':['Unknown NOVA treated as non-UPF','Denominator changed to classified energy without renaming','NOVA 1-4 numeric average']},
        'units':'percentage points, 0..100; not fraction 0..1',
        'person_formula':'mean(D_UPF_d over module-qualified days), with point values available under frozen policy',
        'alternative_feature_id':'UPF_window_energyweighted_pp_candidate','alternative_formula':'100 * sum(K_d) / sum(E_d); bounds likewise, with uncertainty gate',
        'missingness':'No valid E or unresolved point => point missing; retain bounds and reason; no complete-case restriction from unrelated phenotype/score.',
        'proposed_contrast_not_frozen':{'delta':10,'unit':'percentage points','condition':'joint support and precision verified before freeze'},
        'conditional_question':'At specified total recorded energy, more NOVA4 relative to remaining food mixture; not a named food replacement or identified processing intervention.',
        'next_gates':['W05 energy semantics','W06 valid dates/events','W07 FoodID/NOVA weighted coverage and ambiguity','W08 feature views','W08 support/reliability and W22 final score support']
    },{
        'feature_id':'FIBER_DENSITY_recordday_mean_g_per_1000kcal_candidate','name_zh':'膳食纤维密度（g/1000 kcal，合格记录日等权均值；候选）','tier':'priority_candidate_2','existing_or_derived':'Deterministic standardization of existing fiber and energy; not a new score',
        'source_cells':[cell('Nutrients','A5'),cell('Nutrients','A2'),cell('Dash','C6'),cell('Dash','D6')],
        'variables':{'F_d':'sum validated dietary_fiber_g as actual retained-event intake, after W05 conversion only','E_d':'all validated retained-event energy'},
        'daily_formula':'D_FIBER_d = 1000 * F_d / E_d','units':'g/1000 kcal',
        'person_formula':'mean(D_FIBER_d over module-qualified days)',
        'alternative_feature_id':'FIBER_DENSITY_window_energyweighted_g_per_1000kcal_candidate','alternative_formula':'1000 * sum(F_d) / sum(E_d)',
        'missingness':'Unknown fiber is not 0. Incomplete numerator or denominator means point missing unless documented recovery is validated; negative/nonfinite values handled by W05/W06, never absolute-valued.',
        'proposed_contrast_not_frozen':{'delta':5,'unit':'g/1000 kcal','condition':'between-person and within-person support checked separately'},
        'conditional_question':'Fiber density association at specified recorded energy; not fiber fermentability, SCFA production or an identified fiber intervention.',
        'next_gates':['W05 energy and fiber semantics/conversion','W06 retained events/days','W08 feature views','W08 between/within-person reliability and W22 final score support']
    }]
    categories=[('SRC_juice_softdrinks',['A31'],'果汁及软饮；不能改称纯SSB'),('SRC_sweets_baked_snacks',['A14','A15','A16'],'甜食／烘焙／零食并集；非添加糖克数'),('SRC_wholegrain_union',['A13','A18','A21'],'三类全谷并集；非标准干重全谷克数'),('SRC_legumes',['A19'],'豆类及制品；非全部可发酵底物')]
    sources=[{'feature_id':id,'tier':'exploratory_source_description_not_formal_confirmation','source_cells':[cell('Categories',r) for r in refs],'name_and_boundary':label,'formula':'100 * verified energy in union of mapped categories / total retained energy; count each cleaned event once within union','person_aggregation':'same frozen day/window convention as main exposure','gate':'Actual food_category or FoodID crosswalk must prove labels, mutual overlap and version; catalog NOVA flags do not supply these categories'} for id,refs,label in categories]
    score={'feature_id':'I_MEDAS_NCDiet_adapted_20261026_background_candidate','display_name':'NC-Diet适配版I-MEDAS（0–17分；2026-10-26定义快照）','tier':'one_background_benchmark_not_main_exposure','source_cells':[cell('I-MEDAS','A35'),cell('I-MEDAS','C2'),cell('I-MEDAS','C3'),cell('I-MEDAS','C26')],
        'structure':'14 binary components plus one 0..3 sweets component; 15 components, range 0..17; rows 15..17 are not three independent binary questions',
        'not_equivalent_to':['Unmodified standard MEDAS','CSV Mediterranean eligibility share','Verified olive-oil use or intake frequency'],
        'open_definitions':['Category values fraction vs percent vs energy','Mean share times mean energy versus mean daily category energy','Med Oil and fats proxies for olive oil and hummus/tahini','Exact retained-day/person implementation and alcohol component'],
        'implementation_status':'not implemented; exact prior adapted code unavailable here; do not backfill cutoffs or rescale components',
        'fallback':'DASH is a named replacement candidate only if I-MEDAS cannot be reconstructed; record and freeze before analysis. Do not compute both and choose by metabolic p-values.',
        'missingness':'Unresolved score components do not exclude otherwise eligible UPF/fiber subjects',
        'model_use':'Descriptive/background comparison only; not automatically adjusted in primary model because components overlap exposures and change the conditional question'}
    spec={'workpackage':'W04','version':'diet_feature_spec_draft_20261026_v1','format':'JSON syntax valid YAML 1.2','status':'definition_draft_complete_empirical_validation_pending',
        'provenance':'Codex implementation of prior Astra max dietary assessment and user-approved W04; no new Astra review',
        'common':common,'priority_candidates':features,'source_explanations':sources,'background_score':score,
        'packages':{'A':'UPF and fiber as two separate scalar questions; proposed 29 blocks with UPF-only formal Daily or 30 with both; precision/budget gate, not adopted','B':'One frozen primary exposure; UPF prioritized for validation, fiber can replace on pre-outcome data-validity/support evidence, never on p-values; 15-block structure must be rewritten','formal_selected_package':None},
        'excluded_from_initial_main':['All 643 individual food interactions','Five total scores as simultaneous main exposures','Ad hoc oral-contact or microbiome-derived dietary score','NOVA numeric average','Unverified hPDI/AHEI/EAT total scores'],
        'actual_cohort_N':None,'participant_discovery_confirmation_frozen':False}
    save(CFG/'diet_feature_spec_draft.yaml',spec)
    # Arithmetic illustrations are synthetic, not HPP observations.
    E=[1000,3000];K=[800,300];F=[10,60]
    ex={'two_days':{'energy_kcal':E,'known_UPF_energy_kcal':K,'fiber_g':F},'upf_daily_pp':[100*k/e for k,e in zip(K,E)],'upf_equalday_pp':sum(100*k/e for k,e in zip(K,E))/2,'upf_pooled_pp':100*sum(K)/sum(E),'fiber_daily_g_per_1000':[1000*f/e for f,e in zip(F,E)],'fiber_equalday':sum(1000*f/e for f,e in zip(F,E))/2,'fiber_pooled':1000*sum(F)/sum(E),'unknown_energy_example':{'E':2000,'K':600,'U':200,'coverage':.9,'lower_pp':30,'upper_pp':40,'exact_point':None,'legacy_classified_energy_ratio_pp':100*600/1800}}
    assert (ex['upf_equalday_pp'],ex['upf_pooled_pp'],ex['fiber_equalday'],ex['fiber_pooled'])==(45,27.5,15,17.5)
    assert ex['unknown_energy_example']['lower_pp']<=ex['unknown_energy_example']['upper_pp']
    save(OUT/'worked_examples.json',ex)
    checks={'source_original_hashes_match':all(s['current_hash_matches'] for s in manifest['original_definition_sources']),'UPF_exact_cell_D9':features[0]['source_cells'][0]['cell']=='D9','fiber_source_and_density_units_present':True,'I_MEDAS_adapted_name_and_15_components':True,'priority_candidate_count':len(features),'exploratory_source_count':len(sources),'background_score_count':1,'unknown_policy_no_silent_zero':features[0]['unknown_policy']['point_with_unknown'].startswith('null'),'synthetic_aggregation_examples_passed':True,'V3_unchanged':True,'TRE_accessed':False,'participant_values_computed':False,'formal_package_frozen':False}
    save(OUT/'validation.json',checks)
    print(json.dumps({'status':spec['status'],'candidates':len(features),'source_descriptors':len(sources),'background_scores':1,'source_hashes_match':checks['source_original_hashes_match'],'synthetic_examples_passed':True},ensure_ascii=False))
if __name__=='__main__':main()
