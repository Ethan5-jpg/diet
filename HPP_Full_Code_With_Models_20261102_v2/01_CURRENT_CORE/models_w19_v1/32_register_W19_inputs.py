"""Register discovery-only W19 recipe before scientific reads, prepare raw folds."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,hashlib,datetime,io,resource
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq,pyarrow.dataset as ds
from scipy import sparse
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');Q=P/'data_qc/preprocessing_20261030_v1'
L=Path('/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1');B=P/'data_qc/w19_discovery_20261101_v1'
assert (L/'31_actual_exit_code').read_text().strip()=='0';B.mkdir(exist_ok=False);R=B/'access_receipts';R.mkdir()
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
K=['participant_id','cohort','research_stage'];KD=K+['date']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pointer=json.loads((P/'config/preprocessing_20261030_v1/active_execution.json').read_text())
assert sha(pointer['roles'])==pointer['roles_sha256'] and sha(pointer['execution_design'])==pointer['execution_design_sha256']
endpoints=['D_Glu','D_TotalFat','D_VAT','D_Liver_elasticity','D_Liver_velocity','D_Liver_viscosity','D_Liver_attenuation','D_Liver_dispersion','D_Liver_speed_of_sound']
spec={'version':'W19_discovery_20261101_v1','seed':20261101019,'authorized_by':'user requested next W19-W20 step after preprocessing delivery',
 'roles_sha256':pointer['roles_sha256'],'execution_design_sha256':pointer['execution_design_sha256'],
 'read_role':'discovery_only_before_Y_projection','confirmation_Y_reads':0,'endpoints':endpoints,'exposures':['UPF','fiber_density'],
 'outer_folds':5,'inner_folds':3,'family_partition':'balanced seeded shuffle of sorted whole split_unit_ids; one common partition across modules',
 'candidates':['host_only','oral_main','gut_main','oral_gut_main'],'same_rows_Y_weights_per_candidate':True,
 'core':['age','sex','smoking_explicit_unknown','same_diet_window_energy'],'host_dictionary':['intercept','core','linear_diet','RCS3_diet','diet_age','diet_sex'],
 'PCA_dimensions_per_site':[5,10,20],'microbiome_main_dictionary':['linear_PCs','linear_PCs_plus_first5_squares'],
 'microbiome_main_ridge_penalties':[10.,100.],'host_terms':'unpenalized; training pivoted QR drops dependent columns',
 'selection':'one standard error of best mean family-held-fold MSE; prefer linear, fewer PCs, stronger ridge',
 'UPF':'category shrinkage10; source-group weight renormalized after fold-training food selection; full nonlinear basis integrated over1024 shared-label draws',
 'UPF_knots':'fold-training latent predictive 10/50/90 percentiles; center median, scale .1 fraction',
 'fiber_knots':'fold-training observed 10/50/90 percentiles; center median, scale 5 g/1000kcal',
 'microbiome_reference':'all global discovery people whose families belong to current training fold; selected raw W13 profiles',
 'phenotype_reference':'all endpoint-eligible discovery people in current training families; refit shortest95/8SD/5SD',
 'CV_validation_QC':'fixed fold-training rules only; no global-Y-cleaned cohort prefilter; same resulting rows in four models',
 'metrics':['OOF_MSE','OOF_RMSE','OOF_MAE','OOF_R2','paired_delta_MSE_vs_host'],
 'W20_predeclared_grid':{'main_penalty':[10.,100.],'interaction_penalty':[10.,100.,'null'],
 'selection':'inner3 oneSE prefer null, then stronger penalties','stability':'at least4of5 outer fits active and median absolute heldfold score correlation with full-discovery score>=.5; no confirmation selection',
 'sign':'largest absolute original CLR coefficient positive; scale one training score population SD','q_max_per_site':1},
 'no_new_method_comparison':True,'W20_started':False,'confirmation_inference_started':False}
(B/'execution_spec.json').write_text(json.dumps(spec,indent=2)+'\n')
roles=pq.ParquetFile(pointer['roles'],pre_buffer=False).read(columns=K+['split_unit_id','final_role']).to_pandas(ignore_metadata=True)
reserved=set(roles.loc[roles.final_role.eq('confirmation_reserved'),'participant_id']);roles=roles.loc[roles.final_role.eq('discovery')].reset_index(drop=True)
assert not set(roles.participant_id)&reserved and len(roles)==3177
sequence=0
filt=(ds.field('cohort')=='10k')&(ds.field('research_stage')=='00_00_visit')&ds.field('participant_id').isin(roles.participant_id.tolist())
def read(path,cols=None,keys=K,Y=False):
 global sequence
 path=Path(path);pf=pq.ParquetFile(path,pre_buffer=False);cols=list(cols or pf.schema_arrow.names)
 assert set(keys)<=set(cols) and pf.metadata.num_rows*len(cols)*16<1024**3
 sequence+=1
 receipt={'path':str(path),'columns':cols,'purpose':'W19_discovery_raw_fold_inputs','contains_Y':Y,'filter_before_materialization':'sealed discovery IDs,10k,00_00_visit','roles_sha256':pointer['roles_sha256'],'source_sha256':sha(path)}
 (R/(str(sequence).zfill(3)+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
 d=ds.dataset(path,format='parquet').scanner(columns=cols,filter=filt,use_threads=False,batch_size=1024).to_table().to_pandas(ignore_metadata=True)
 assert not d.duplicated(keys).any() and not set(d.participant_id)&reserved
 return d
def save(d,p):pq.write_table(pa.Table.from_pandas(d.reset_index(drop=True),preserve_index=False),p,compression='zstd')
def npz(path,**arrays):
 buf=io.BytesIO();np.savez_compressed(buf,**arrays)
 with Path(path).open('xb') as f:f.write(buf.getvalue())
 with np.load(io.BytesIO(Path(path).read_bytes())) as check:
  for k,a in arrays.items():np.testing.assert_array_equal(check[k],a)
units=np.array(sorted(roles.split_unit_id.unique()));rng=np.random.default_rng(spec['seed']);rng.shuffle(units)
outer={u:i%5 for i,u in enumerate(units)};roles['outer_fold']=roles.split_unit_id.map(outer)
for o in range(5):
 uu=np.array(sorted(roles.loc[roles.outer_fold.ne(o),'split_unit_id'].unique()));np.random.default_rng(spec['seed']+o+1).shuffle(uu);mapping={u:i%3 for i,u in enumerate(uu)}
 roles['inner_for_outer_'+str(o)]=roles.split_unit_id.map(mapping).fillna(-1).astype(int)
save(roles,B/'discovery_roles_folds_TRE_ONLY.parquet')
D=read(Q/'food_qc_v1/food_daily_primary_TRE_ONLY.parquet',KD+['energy_kcal','fiber_g','fiber_density_g_per_1000kcal','fiber_eligible'],keys=KD)
energy=D.groupby(K).agg(n_energy_days=('date','size'),mean_daily_energy_UPF=('energy_kcal','mean')).reset_index()
fd=D.loc[D.fiber_eligible]
fiber=fd.groupby(K).agg(n_fiber_days=('date','size'),fiber_density=('fiber_density_g_per_1000kcal','mean'),mean_daily_energy_fiber=('energy_kcal','mean')).reset_index()
C=read(P/'analysis/shared/covariate_registry_TRE_ONLY.parquet',K+['age_calendar_years','sex_code','smoking_current_status_label'])
C['smoking_status']=C.smoking_current_status_label.astype('string').fillna('__UNKNOWN__').replace('', '__UNKNOWN__')
X=roles.merge(C,on=K,how='left',validate='one_to_one').merge(energy,on=K,how='left',validate='one_to_one').merge(fiber,on=K,how='left',validate='one_to_one')
for c in ['age_calendar_years','sex_code']:X[c]=pd.to_numeric(X[c],errors='coerce')
G=Q/'foodid_interval_geometry_v1b'
geom=read(G/'matrix_person_rows_TRE_ONLY.parquet',K+['matrix_row_ordinal','UPF_lower_equalday_fraction','UPF_upper_equalday_fraction'])
X=X.merge(geom,on=K,how='left',validate='one_to_one');X['input_row']=np.arange(len(X))
save(X,B/'discovery_covariates_food_TRE_ONLY.parquet')
for site in ['oral','gut']:
 path=P/'analysis/shared'/(site+'_selected_abundance_TRE_ONLY.parquet');names=[n for n in pq.ParquetFile(path).schema_arrow.names if n.startswith('k__') or n.startswith('s__') or '|s__' in n]
 d=read(path,K+names);d=X[K].merge(d,on=K,how='left',validate='one_to_one');a=d[names].to_numpy(float)
 assert np.isfinite(a).all() and np.allclose(a.sum(axis=1),1,atol=1e-3,rtol=0)
 npz(B/(site+'_raw_TRE_ONLY.npz'),abundance=a)
 (B/(site+'_taxa.json')).write_text(json.dumps(names)+'\n')
# Extract sparse geometry rows by discovery identities, not all-source matrix ordinal assumptions.
A=sparse.load_npz(io.BytesIO((G/'unknown_foodid_equalday_weights_TRE_ONLY.npz').read_bytes()));valid=X.matrix_row_ordinal.notna().to_numpy()
AA=sparse.lil_matrix((len(X),A.shape[1]));AA[np.flatnonzero(valid),:]=A[X.loc[valid,'matrix_row_ordinal'].astype(int).to_numpy()];AA=AA.tocsr()
npz(B/'unknown_weights_TRE_ONLY.npz',data=AA.data,indices=AA.indices,indptr=AA.indptr,shape=np.array(AA.shape))
reference=pq.ParquetFile(Q/'UPF_probability_inputs_v1b/grouped_CV_predictions_TRE_ONLY.parquet').read().to_pandas(ignore_metadata=True)
unknown=pq.ParquetFile(Q/'UPF_probability_inputs_v1b/FoodID_probabilities_TRE_ONLY.parquet').read(columns=['food_id','category']).to_pandas(ignore_metadata=True)
save(reference,B/'reference_food_labels_TRE_ONLY.parquet');save(unknown,B/'unknown_food_categories_TRE_ONLY.parquet')
# Fold-training reference labels can only come from foods actually used in those training families.
source=P/'analysis/shared/food_events_clean_TRE_ONLY.parquet'
(R/'food_event_preaccess.json').write_text(json.dumps({'path':str(source),'columns':KD+['food_id','calories_kcal_clean'],'purpose':'fold_training_reference_food_presence','filter_before_materialization':'discovery IDs','source_sha256':sha(source)},indent=2)+'\n')
events=ds.dataset(source,format='parquet').scanner(columns=KD+['food_id','calories_kcal_clean'],filter=filt,use_threads=False,batch_size=4096).to_table().to_pandas(ignore_metadata=True)
events=events.loc[events.calories_kcal_clean.gt(0)&events.food_id.notna()].merge(D[KD],on=KD,how='inner',validate='many_to_one')
presence=events[K+['food_id']].drop_duplicates().merge(X[K+['input_row']],on=K,validate='many_to_one')
save(presence[['input_row','food_id']],B/'reference_food_presence_TRE_ONLY.parquet')
cgm=read(P/'analysis/shared/cgm_visit_phenotypes_TRE_ONLY.parquet',K+['n_qualified_CGM_days'])
for endpoint in endpoints:
 y=read(P/'analysis'/endpoint/'UPF_bounds/base_table_TRE_ONLY.parquet',K+['Y_value','Y_unit'],Y=True)
 # W15 membership is the source endpoint eligibility contract; no globally fitted outlier exclusion.
 m=read(P/'freeze/module_membership_TRE_ONLY.parquet',K+['analysis_id'],keys=K+['analysis_id'])
 m=m.loc[m.analysis_id.eq(endpoint),K]
 y=y.merge(m,on=K,how='inner',validate='one_to_one')
 if endpoint=='D_Glu':
  y=y.merge(cgm,on=K,how='left',validate='one_to_one');y=y.loc[y.n_qualified_CGM_days.ge(3)].copy()
 y=y.merge(X[K+['input_row']],on=K,how='inner',validate='one_to_one')
 save(y,B/(endpoint+'_raw_Y_TRE_ONLY.parquet'))
 print('W19_DISCOVERY_ENDPOINT_RAW',endpoint,len(y),flush=True)
report={'status':'W19_recipe_and_raw_discovery_inputs_registered','people':len(X),'families':int(X.split_unit_id.nunique()),'confirmation_Y_reads':0,'recipe_sha256':sha(B/'execution_spec.json'),'folds':5,'inner_folds':3,'input_receipts':sequence,'raw_profile_features':{s:len(json.loads((B/(s+'_taxa.json')).read_text())) for s in ['oral','gut']},'smoking_levels':sorted(X.smoking_status.unique().tolist()),'new_model_fits':0}
(B/'input_summary.json').write_text(json.dumps(report,indent=2)+'\n');print('W19_INPUTS_REGISTERED',json.dumps(report),flush=True)
