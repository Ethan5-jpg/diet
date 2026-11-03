from pathlib import Path
import json
D=Path(__file__).parent
s=Path('/private/tmp/hpp_w19_active_20261101/32_register_W19_inputs.py').read_text()
s=s.replace('"""Register discovery-only W19 recipe before scientific reads, prepare raw folds."""','"""Register user-authorized 70/30 v2 before projecting discovery Y."""')
s=s.replace("B=P/'data_qc/w19_discovery_20261101_v1'", "B=P/'data_qc/w20_simplified_20261101_v2'")
s=s.replace("(L/'31_actual_exit_code')", "(L/'40_actual_exit_code')")
start=s.index("spec={'version'");end=s.index("sequence=0",start)
spec={'version':'W19_W20_simplified_20261101_v2','seed':20261101042,
 'authorization':'user explicitly requested anti-defense-research simplified rerun with 70/30 and single5; preserve v1',
 'role_change':'original W15 whole-unit prefix<=floor5699*.7; remaining old reserved retained; no Y-based reassignment',
 'prior_exposure':'historical exploration and W19-W20 v1 discovery results seen; no pristine external-validation claim',
 'read_role':'new discovery IDs filtered before scientific Y projection','confirmation_Y_reads':0,
 'endpoints':['D_Glu','D_TotalFat','D_VAT','D_Liver_elasticity','D_Liver_velocity','D_Liver_viscosity','D_Liver_attenuation','D_Liver_dispersion','D_Liver_speed_of_sound'],
 'exposures':['UPF','fiber_density'],'folds':5,'nested_CV':False,
 'background_PC_per_site':[5,10],'PC_squares':False,'host_terms':'unchanged W19 host terms',
 'ridge_objective':'mean squared loss + lambda_main*||b_Z||^2 + lambda_interaction*||b_uZ||^2; host and PC background unpenalized',
 'main_lambda':[.1,1.],'interaction_lambda':[.01,.1,1.,10.],
 'selection':'minimum mean 5-fold MSE for main-only background and main lambda, then minimum among interaction lambdas at selected background/main lambda',
 'null':'same full-CLR main effects and matching background; report prediction winner separately, no null-MSE eligibility gate for frozen-score association',
 'stability':'continuous descriptive correlations and fold availability; no 4/5 or .5 exclusion threshold',
 'prediction':'retain full block-ridge coefficients; no same-Y unpenalized compressed refit',
 'score':'one interaction direction/site, training SD normalization is output re-expression only; confirmation low-dimensional testing not started',
 'data_QC':'inherit agreed food, phenotype, microbiome and endpoint-specific membership; all fitted parameters refit on new discovery/CV training',
 'UPF':'same category shrinkage10 and1024 shared FoodID draws; rebuild known reference-food membership for expanded discovery and refit per fold',
 'metrics':'selected-CV descriptive development metrics, not unbiased post-selection performance',
 'confirmation_inference_started':False,'primary_family':'existing55 Holm unchanged; no new P values',
 'control':'one fiber/Glu module, real discovery X and synthetic Y at0/1/5percent residual signal, one fixed seed, oracle alongside; diagnostic, not precision power study'}
replacement="spec="+repr(spec)+"\nspec.update({'roles_sha256':pointer['roles_sha256'],'execution_design_sha256':pointer['execution_design_sha256']})\n"
replacement+="(B/'execution_spec.json').write_text(json.dumps(spec,indent=2)+'\\n')\n"
replacement+="roles=pq.ParquetFile(pointer['roles'],pre_buffer=False).read(columns=K+['split_unit_id','unit_order','final_role']).to_pandas(ignore_metadata=True)\n"
replacement+="assert len(roles)==6305 and roles.split_unit_id.nunique()==5699\nroles['previous_final_role']=roles.final_role\ncutoff=int(np.floor(roles.unit_order.max()*.7));assert cutoff==3989\nroles['final_role']=np.where(roles.unit_order.le(cutoff),'discovery','confirmation_reserved')\nassert roles.groupby('split_unit_id').final_role.nunique().max()==1\nassert not ((roles.previous_final_role=='discovery')&(roles.final_role!='discovery')).any()\nallroles=roles.copy();pq.write_table(pa.Table.from_pandas(allroles,preserve_index=False),B/'roles_v2_full_TRE_ONLY.parquet',compression='zstd')\n"
replacement+="oldpaths=[Path(pointer['roles']),Path(pointer['execution_design']),P/'config/preprocessing_20261030_v1/active_execution.json']\noldmanifest=P/'data_qc/w20_discovery_20261101_v1/score_results_v1/delivery/artifact_manifest_TRE_ONLY.json'\noldpaths.extend(Path(v['path']) for v in json.loads(oldmanifest.read_text()))\noldpaths.append(oldmanifest)\noldhash={str(p):sha(p) for p in set(oldpaths)}\n(B/'protected_v1_hashes.json').write_text(json.dumps(oldhash,indent=2)+'\\n')\n"
replacement+="role_summary={'role_counts':roles.final_role.value_counts().to_dict(),'new_discovery_people':int(((roles.previous_final_role=='confirmation_reserved')&(roles.final_role=='discovery')).sum()),'discovery_units':int(roles.loc[roles.final_role.eq('discovery'),'split_unit_id'].nunique()),'confirmation_units':int(roles.loc[roles.final_role.eq('confirmation_reserved'),'split_unit_id'].nunique()),'unit_cutoff':cutoff,'confirmation_Y_reads':0}\n(B/'role_summary.json').write_text(json.dumps(role_summary,indent=2)+'\\n');print('V2_ROLES',json.dumps(role_summary),flush=True)\n"
replacement+="reserved=set(roles.loc[roles.final_role.eq('confirmation_reserved'),'participant_id']);roles=roles.loc[roles.final_role.eq('discovery')].reset_index(drop=True)\nassert not set(roles.participant_id)&reserved\n"
s=s[:start]+replacement+s[end:]
s=s.replace('W19_discovery_raw_fold_inputs','v2_discovery_raw_fold_inputs').replace('W19_discovery','v2_discovery')
start=s.index('for o in range(5):');end=s.index("save(roles,B/'discovery_roles_folds_TRE_ONLY.parquet')",start)
s=s[:start]+"roles['cv_fold']=roles.outer_fold\nassert roles.groupby('split_unit_id').cv_fold.nunique().max()==1\n"+s[end:]
start=s.index("reference=pq.ParquetFile(Q/");end=s.index('# Fold-training reference labels',start)
s=s[:start]+s[end:]
at=s.index("cgm=read(")
ref=Path('/private/tmp/hpp_preprocessing_active_20261030/15_blind_probability_model.py').read_text()
a=ref.index("C=read(P/'config/food_id_crosswalk_versioned.parquet')")
b=ref.index('def fold(v):',a)
ref=ref[a:b].replace('food_ids','set(presence.food_id)')
ref=ref.replace("C=read(P/'config/food_id_crosswalk_versioned.parquet')", "C=pq.ParquetFile(P/'config/food_id_crosswalk_versioned.parquet',pre_buffer=False).read().to_pandas(ignore_metadata=True)")
ref+="\nsave(reference[['food_id','label_group','category','y','weight']],B/'reference_food_labels_TRE_ONLY.parquet')\n"
ref+="unknown=pq.ParquetFile(Q/'UPF_probability_inputs_v1b/FoodID_probabilities_TRE_ONLY.parquet',pre_buffer=False).read(columns=['food_id','category']).to_pandas(ignore_metadata=True)\nsave(unknown,B/'unknown_food_categories_TRE_ONLY.parquet')\n"
s=s[:at]+ref+s[at:]
s=s.replace("'inner_folds':3", "'nested_CV':False")
s=s.replace("'new_model_fits':0", "'new_model_fits':0,'reference_FoodIDs':len(reference),'role_summary':role_summary")
s=s.replace('W19_DISCOVERY_ENDPOINT_RAW','V2_DISCOVERY_ENDPOINT_RAW').replace('W19_INPUTS_REGISTERED','V2_INPUTS_REGISTERED')
compile(s,'41_register_v2_inputs.py','exec');(D/'41_register_v2_inputs.py').write_text(s)
(D/'execution_spec_draft.json').write_text(json.dumps(spec,indent=2)+'\n')
print('BUILT_INPUT',len(s))
