"""Register user-authorized 70/30 v2 before projecting discovery Y."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,hashlib,datetime,io,resource
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq,pyarrow.dataset as ds
from scipy import sparse
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');Q=P/'data_qc/preprocessing_20261030_v1'
L=Path('/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1');B=P/'data_qc/w20_simplified_20261101_v2'
assert (L/'40_actual_exit_code').read_text().strip()=='0';B.mkdir(exist_ok=False);R=B/'access_receipts';R.mkdir()
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
K=['participant_id','cohort','research_stage'];KD=K+['date']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pointer=json.loads((P/'config/preprocessing_20261030_v1/active_execution.json').read_text())
assert sha(pointer['roles'])==pointer['roles_sha256'] and sha(pointer['execution_design'])==pointer['execution_design_sha256']
endpoints=['D_Glu','D_TotalFat','D_VAT','D_Liver_elasticity','D_Liver_velocity','D_Liver_viscosity','D_Liver_attenuation','D_Liver_dispersion','D_Liver_speed_of_sound']
spec={'version': 'W19_W20_simplified_20261101_v2', 'seed': 20261101042, 'authorization': 'user explicitly requested anti-defense-research simplified rerun with 70/30 and single5; preserve v1', 'role_change': 'original W15 whole-unit prefix<=floor5699*.7; remaining old reserved retained; no Y-based reassignment', 'prior_exposure': 'historical exploration and W19-W20 v1 discovery results seen; no pristine external-validation claim', 'read_role': 'new discovery IDs filtered before scientific Y projection', 'confirmation_Y_reads': 0, 'endpoints': ['D_Glu', 'D_TotalFat', 'D_VAT', 'D_Liver_elasticity', 'D_Liver_velocity', 'D_Liver_viscosity', 'D_Liver_attenuation', 'D_Liver_dispersion', 'D_Liver_speed_of_sound'], 'exposures': ['UPF', 'fiber_density'], 'folds': 5, 'nested_CV': False, 'background_PC_per_site': [5, 10], 'PC_squares': False, 'host_terms': 'unchanged W19 host terms', 'ridge_objective': 'mean squared loss + lambda_main*||b_Z||^2 + lambda_interaction*||b_uZ||^2; host and PC background unpenalized', 'main_lambda': [0.1, 1.0], 'interaction_lambda': [0.01, 0.1, 1.0, 10.0], 'selection': 'minimum mean 5-fold MSE for main-only background and main lambda, then minimum among interaction lambdas at selected background/main lambda', 'null': 'same full-CLR main effects and matching background; report prediction winner separately, no null-MSE eligibility gate for frozen-score association', 'stability': 'continuous descriptive correlations and fold availability; no 4/5 or .5 exclusion threshold', 'prediction': 'retain full block-ridge coefficients; no same-Y unpenalized compressed refit', 'score': 'one interaction direction/site, training SD normalization is output re-expression only; confirmation low-dimensional testing not started', 'data_QC': 'inherit agreed food, phenotype, microbiome and endpoint-specific membership; all fitted parameters refit on new discovery/CV training', 'UPF': 'same category shrinkage10 and1024 shared FoodID draws; rebuild known reference-food membership for expanded discovery and refit per fold', 'metrics': 'selected-CV descriptive development metrics, not unbiased post-selection performance', 'confirmation_inference_started': False, 'primary_family': 'existing55 Holm unchanged; no new P values', 'control': 'one fiber/Glu module, real discovery X and synthetic Y at0/1/5percent residual signal, one fixed seed, oracle alongside; diagnostic, not precision power study'}
spec.update({'roles_sha256':pointer['roles_sha256'],'execution_design_sha256':pointer['execution_design_sha256']})
(B/'execution_spec.json').write_text(json.dumps(spec,indent=2)+'\n')
roles=pq.ParquetFile(pointer['roles'],pre_buffer=False).read(columns=K+['split_unit_id','unit_order','final_role']).to_pandas(ignore_metadata=True)
assert len(roles)==6305 and roles.split_unit_id.nunique()==5699
roles['previous_final_role']=roles.final_role
cutoff=int(np.floor(roles.unit_order.max()*.7));assert cutoff==3989
roles['final_role']=np.where(roles.unit_order.le(cutoff),'discovery','confirmation_reserved')
assert roles.groupby('split_unit_id').final_role.nunique().max()==1
assert not ((roles.previous_final_role=='discovery')&(roles.final_role!='discovery')).any()
allroles=roles.copy();pq.write_table(pa.Table.from_pandas(allroles,preserve_index=False),B/'roles_v2_full_TRE_ONLY.parquet',compression='zstd')
oldpaths=[Path(pointer['roles']),Path(pointer['execution_design']),P/'config/preprocessing_20261030_v1/active_execution.json']
oldmanifest=P/'data_qc/w20_discovery_20261101_v1/score_results_v1/delivery/artifact_manifest_TRE_ONLY.json'
oldpaths.extend(Path(v['path']) for v in json.loads(oldmanifest.read_text()))
oldpaths.append(oldmanifest)
oldhash={str(p):sha(p) for p in set(oldpaths)}
(B/'protected_v1_hashes.json').write_text(json.dumps(oldhash,indent=2)+'\n')
role_summary={'role_counts':roles.final_role.value_counts().to_dict(),'new_discovery_people':int(((roles.previous_final_role=='confirmation_reserved')&(roles.final_role=='discovery')).sum()),'discovery_units':int(roles.loc[roles.final_role.eq('discovery'),'split_unit_id'].nunique()),'confirmation_units':int(roles.loc[roles.final_role.eq('confirmation_reserved'),'split_unit_id'].nunique()),'unit_cutoff':cutoff,'confirmation_Y_reads':0}
(B/'role_summary.json').write_text(json.dumps(role_summary,indent=2)+'\n');print('V2_ROLES',json.dumps(role_summary),flush=True)
reserved=set(roles.loc[roles.final_role.eq('confirmation_reserved'),'participant_id']);roles=roles.loc[roles.final_role.eq('discovery')].reset_index(drop=True)
assert not set(roles.participant_id)&reserved
sequence=0
filt=(ds.field('cohort')=='10k')&(ds.field('research_stage')=='00_00_visit')&ds.field('participant_id').isin(roles.participant_id.tolist())
def read(path,cols=None,keys=K,Y=False):
 global sequence
 path=Path(path);pf=pq.ParquetFile(path,pre_buffer=False);cols=list(cols or pf.schema_arrow.names)
 assert set(keys)<=set(cols) and pf.metadata.num_rows*len(cols)*16<1024**3
 sequence+=1
 receipt={'path':str(path),'columns':cols,'purpose':'v2_discovery_raw_fold_inputs','contains_Y':Y,'filter_before_materialization':'sealed discovery IDs,10k,00_00_visit','roles_sha256':pointer['roles_sha256'],'source_sha256':sha(path)}
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
roles['cv_fold']=roles.outer_fold
assert roles.groupby('split_unit_id').cv_fold.nunique().max()==1
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
# Fold-training reference labels can only come from foods actually used in those training families.
source=P/'analysis/shared/food_events_clean_TRE_ONLY.parquet'
(R/'food_event_preaccess.json').write_text(json.dumps({'path':str(source),'columns':KD+['food_id','calories_kcal_clean'],'purpose':'fold_training_reference_food_presence','filter_before_materialization':'discovery IDs','source_sha256':sha(source)},indent=2)+'\n')
events=ds.dataset(source,format='parquet').scanner(columns=KD+['food_id','calories_kcal_clean'],filter=filt,use_threads=False,batch_size=4096).to_table().to_pandas(ignore_metadata=True)
events=events.loc[events.calories_kcal_clean.gt(0)&events.food_id.notna()].merge(D[KD],on=KD,how='inner',validate='many_to_one')
presence=events[K+['food_id']].drop_duplicates().merge(X[K+['input_row']],on=K,validate='many_to_one')
save(presence[['input_row','food_id']],B/'reference_food_presence_TRE_ONLY.parquet')
C=pq.ParquetFile(P/'config/food_id_crosswalk_versioned.parquet',pre_buffer=False).read().to_pandas(ignore_metadata=True)
reference=C.loc[C.food_id.isin(set(presence.food_id))&C.NOVA.isin([1,2,3,4])].copy().reset_index(drop=True)
assert reference.food_id.is_unique
parent={}
def root(v):
    parent.setdefault(v,v)
    while parent[v]!=v:parent[v]=parent[parent[v]];v=parent[v]
    return v
def union(a,b):
    a,b=root(a),root(b)
    if a!=b:parent[max(a,b)]=min(a,b)
names=[]
for v in reference.short_names_json:
    nn=sorted(set(str(x).strip().casefold() for x in json.loads(v) if str(x).strip()))
    assert nn,'known catalogue label must have short-name provenance'
    for t in nn:root(t)
    for t in nn[1:]:union(nn[0],t)
    names.append(nn)
reference['label_group']=[root(v[0]) for v in names]
reference['y']=reference.NOVA.eq(4).astype(int)
assert reference.groupby('label_group').y.nunique().max()==1
reference['category']=reference.source_categories_json.map(lambda v:json.dumps(sorted(str(x) for x in json.loads(v)),separators=(',',':')))
counts=reference.groupby('label_group').size()
reference['weight']=1/reference.label_group.map(counts)

save(reference[['food_id','label_group','category','y','weight']],B/'reference_food_labels_TRE_ONLY.parquet')
unknown=pq.ParquetFile(Q/'UPF_probability_inputs_v1b/FoodID_probabilities_TRE_ONLY.parquet',pre_buffer=False).read(columns=['food_id','category']).to_pandas(ignore_metadata=True)
save(unknown,B/'unknown_food_categories_TRE_ONLY.parquet')
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
 print('V2_DISCOVERY_ENDPOINT_RAW',endpoint,len(y),flush=True)
report={'status':'W19_recipe_and_raw_discovery_inputs_registered','people':len(X),'families':int(X.split_unit_id.nunique()),'confirmation_Y_reads':0,'recipe_sha256':sha(B/'execution_spec.json'),'folds':5,'nested_CV':False,'input_receipts':sequence,'raw_profile_features':{s:len(json.loads((B/(s+'_taxa.json')).read_text())) for s in ['oral','gut']},'smoking_levels':sorted(X.smoking_status.unique().tolist()),'new_model_fits':0,'reference_FoodIDs':len(reference),'role_summary':role_summary}
(B/'input_summary.json').write_text(json.dumps(report,indent=2)+'\n');print('V2_INPUTS_REGISTERED',json.dumps(report),flush=True)
