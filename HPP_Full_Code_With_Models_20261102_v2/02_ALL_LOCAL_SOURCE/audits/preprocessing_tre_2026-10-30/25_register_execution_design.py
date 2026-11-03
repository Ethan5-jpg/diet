"""W18 execution registration and globally consistent roles; no Y values."""
from pathlib import Path
import hashlib,json,datetime,importlib.util
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
Q=P/'data_qc/preprocessing_20261030_v1';L=Path('/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1')
assert (L/'21_actual_exit_code').read_text().strip()=='0'
root=P/'config/preprocessing_20261030_v1';root.mkdir(exist_ok=False)
F=P/'freeze/w18_preprocessing_20261030_v1';F.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=P/'freeze/split_order_TRE_ONLY.parquet'
receipt={'purpose':'W18_whole_unit_role_assignment_no_Y_values',
 'columns':['participant_id','cohort','research_stage','split_unit_id','unit_order','person_order','initial_role','final_role'],
 'source':str(old),'source_sha256':sha(old),'user_authorization':'continue W16-W18 and agreed preprocessing; stop expanding method comparisons',
 'timestamp_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(root/'role_preaccess_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
d=pq.ParquetFile(old,pre_buffer=False).read(columns=receipt['columns'],use_threads=False).to_pandas(ignore_metadata=True)
assert d.final_role.isna().all() and not d.duplicated(['cohort','participant_id']).any()
assert d.split_unit_id.nunique()==5699 and len(d)==6305
cutoff=int(np.floor(d.unit_order.max()*.5));assert cutoff==2849
d['final_role']=np.where(d.unit_order.le(cutoff),'discovery','confirmation_reserved')
assert d.groupby('split_unit_id').final_role.nunique().max()==1
manifest=json.loads((P/'data_qc/w15_split_access_20261029/final_artifact_manifest.json').read_text())
guardfile=next(Path(v['path']) for v in manifest if Path(v['path']).name=='w15_access_guard.py')
assert sha(guardfile)=='129ae0ac28f01208ec5c88c7a6f8b947b0dcb55ff2ba314f256c7ac3119602b2'
ns={};exec(compile(guardfile.read_text(),str(guardfile),'exec'),ns)
for oldrole,newrole in d[['initial_role','final_role']].drop_duplicates().itertuples(index=False,name=None):ns['validate_transition'](oldrole,newrole)
roles=F/'participant_roles_TRE_ONLY.parquet'
pq.write_table(pa.Table.from_pandas(d,preserve_index=False),roles,compression='zstd')
back=pq.ParquetFile(roles,pre_buffer=False).read().to_pandas(ignore_metadata=True)
pd.testing.assert_frame_equal(d,back,check_dtype=False)
design={'version':'W18_execution_preprocessing_20261030_v1',
 'status':'technical_execution_registration','user_authorized_execution':True,
 'human_review':{'user':'authorized continued execution in chat','colleague':'not yet recorded; no signature fabricated'},
 'supersedes_for_this_execution_only':'candidate minimum days and UNASSIGNED roles in earlier W14/W15; earlier published files remain immutable',
 'diet_scheme':'A_two_separate_scalar_questions','person_exposures':['UPF','fiber_density'],
 'primary_aggregation':'equal_valid_day_mean','aggregation_sensitivity':'cumulative_ratio',
 'primary_minimum_valid_diet_days':3,'minimum_days_sensitivity':7,
 'energy_days':'new food_qc_v1 accepted [500,4000] days without quarantined events',
 'low_person_energy_2p5SD':'flag only in primary; exclusion sensitivity',
 'food_window':'same participant/cohort/visit; food-only window for person outcomes',
 'microbiome_window':'inherit W13 same-visit selection lock; no new lag cutoff',
 'CGM_person':'same-visit QC-qualified CGM aggregate; minimum 3 valid CGM days; not required to overlap food day-by-day',
 'Daily':'fiber only, actual matching food-CGM calendar dates, minimum 3 common valid days, equal total person weight and person/family-cluster inference',
 'roles':{'path':str(roles),'sha256':sha(roles),'unit_count':5699,'training_unit_prefix':2849,'training_fraction_units':.5,
          'reason':'blind food-QC comparison retains useful training counts while maximizing confirmation precision among evaluated fractions',
          'same_person_and_known_family_role_across_all_modules':True,'confirmation_never_returns_to_discovery':True},
 'training_transform_reference':'one row per W13-selected sample/site among global discovery people; all later CV refits transformations excluding its validation families',
 'phenotype_threshold_reference':'per endpoint, global discovery people with eligible baseline endpoint; shared across the two diet questions',
 'microbiome':{'prevalence':.05,'zero_replacement':'half training positive abundance first percentile','closure':'after retained taxa','transform':'natural_log_CLR','scale':'training population SD','whole_sample_missing':'not imputed'},
 'phenotype':{'reference':'shortest ceil(95% n) training interval; leftmost tie','SD':'sample SD','exclude_abs_SD_gt':8,'winsorize_abs_SD_gt':5,'retain_original_and_flags':True,'Daily':'retain point/day QC; no person-level outlier rule'},
 'UPF_method':'shared_FoodID_probability_regression_calibration; integrate entire nonlinear and interaction dictionary',
 'UPF_probability_model':'source-group weighted category-set shrinkage10 to global Beta(1,1) mean; group-CV chosen on discovery reference foods',
 'UPF_uncertainty':'source-group probability draws plus one common latent NOVA4 label per FoodID; family sandwich plus catalogue perturbation covariance',
 'UPF_probability_sensitivity_odds':[.25,.5,1,2,4],
 'primary_condition_set':['age_calendar_years','sex_code','smoking_current_status_label_with_explicit_unknown','same_window_mean_daily_energy'],
 'diet_host_nuisance_interactions':['diet_by_age','diet_by_sex'],
 'medication':'retain three window states; envelope is primary reporting window; uncertain is not non-use; no unverifiable medication interaction in primary',
 'optional_conditions_sensitivity':['alcohol','activity','sleep'],'parallel_outcomes_mutually_adjusted':False,
 'person_primary_modules':9,'six_liver_endpoints':'separate, inherited W14 exact fields; no composite','BMI':'secondary descriptive, outside primary 55 family',
 'deferred':['pair_cross_domain_comparisons','common4','meal_level_PPGR','additional_generic_method_development'],
 'primary_family':{'positions':55,'formula':'2 exposures x 9 endpoints x 3 blocks + 1 fiber Daily block','blocks':['oral+gut_joint','oral_given_gut','gut_given_oral'],'adjustment':'one Holm family alpha .05','inactive_p':1},
 'learner':'inherited block ridge interaction; first SVD score per site; total q<=2; family outer5/inner3 CV; all transforms refit within training folds',
 'learner_selection':'prefer simplest within one standard error; include null direction; stability decided in discovery before W22',
 'q0':'null model selected, rank deficient, absent stable direction, or training score SD<=1e-8; keep registered p=1 positions',
 'main_effects':'restricted cubic diet spline; core conditions and diet-host terms; transparent oral/gut PCA background plus fixed-score main effects; final discovery-selected nuisance dictionary sealed before confirmation',
 'contrast_rule':'training 25th-to-75th percentile diet contrast and one training SD of frozen score; UPF percentiles from model-implied latent draws; actual common-support contrasts sealed W22',
 'unestimable':'record inactive and p=1 rather than change role or choose confirmation-favorable score',
 'method_assessment':str(Q/'method_assessment_v1/method_decision.json'),
 'new_real_Y_values_read_by_this_registration':0,
 'next_allowed':'technical preprocessing of X and endpoint-specific Y QC with preaccess receipts; supervised discovery remains W19/W20 and confirmation remains W22/W23'}
config=root/'execution_design.json';config.write_text(json.dumps(design,indent=2,ensure_ascii=False)+'\n')
pointer={'version':design['version'],'base_protocol':str(P/'config/ACTIVE_PROTOCOL.json'),
 'base_protocol_sha256':sha(P/'config/ACTIVE_PROTOCOL.json'),'execution_design':str(config),'execution_design_sha256':sha(config),
 'roles':str(roles),'roles_sha256':sha(roles),'status':'technical_registered_roles_frozen_for_preprocessing'}
(root/'active_execution.json').write_text(json.dumps(pointer,indent=2)+'\n')
assert sha(old)==receipt['source_sha256']
print('W18_EXECUTION_DESIGN_REGISTERED',json.dumps({'path':str(config),'design_sha256':sha(config),
 'role_counts':d.final_role.value_counts().to_dict(),'training_units':2849,'confirmation_units':2850,
 'minimum_diet_days':3,'family_positions':55,'new_Y_value_reads':0,'old_W15_role_file_unchanged':True,'joint_human_review_not_fabricated':True}),flush=True)
meta=json.loads((Q/'preflight_TRE_ONLY.json').read_text())
for name,v in meta['inputs'].items():print('MATRIX_INPUT',name,json.dumps(v))
for line in (L/'24.log').read_text().splitlines():
    if line.startswith('PUBLISHED_PARQUET_SCHEMA_GROUP '):
        v=json.loads(line.split(' ',1)[1]);cols=v['columns']
        if 'Y_value' in cols:print('Y_SOURCE_FILES',json.dumps(v['files']),'Y_FIELDS',','.join(c for c in cols if c.startswith('Y_')))
print('REGISTER_EXECUTION_COMPLETE')
