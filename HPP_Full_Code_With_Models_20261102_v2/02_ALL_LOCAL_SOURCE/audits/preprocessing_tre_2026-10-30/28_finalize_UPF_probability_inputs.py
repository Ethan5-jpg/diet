"""Promote the already evaluated discovery-only food model after role freeze."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import hashlib,json,io,resource
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq
from scipy import sparse
from upf_catalogue_calibration_v2 import grouped_probability_draws
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
Q=P/'data_qc/preprocessing_20261030_v1';L=Path('/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1')
assert (L/'25_actual_exit_code').read_text().strip()=='0' and (L/'17_actual_exit_code').read_text().strip()=='0'
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(600,600))
B=Q/'UPF_probability_inputs_v1';B.mkdir(exist_ok=False)
C=Q/'probability_model_candidate_v1';G=Q/'foodid_interval_geometry_v1b'
pointer=json.loads((P/'config/preprocessing_20261030_v1/active_execution.json').read_text())
assert hashlib.sha256(Path(pointer['roles']).read_bytes()).hexdigest()==pointer['roles_sha256']
spec=json.loads((C/'preaccess_spec.json').read_text());design=json.loads(Path(pointer['execution_design']).read_text())
assert spec['unit_order_cutoff']==design['roles']['training_unit_prefix']==2849
def read(p):return pq.ParquetFile(p,pre_buffer=False).read().to_pandas(ignore_metadata=True)
roles=read(pointer['roles']);assert roles.final_role.eq('discovery').equals(roles.unit_order.le(spec['unit_order_cutoff']))
evaluation=json.loads((C/'evaluation_summary.json').read_text())
assert int(roles.final_role.eq('discovery').sum())==evaluation['hypothetical_training_people']
receipt={'purpose':'finalize food probability model with same discovery identities as evaluated candidate',
 'role_sha256':pointer['roles_sha256'],'source_candidate':str(C),'matching_unit_prefix':2849,
 'new_parameter_selection':False,'new_Y_value_columns':[],'draws':1024,'seed':20261030028,
 'provenance':'candidate code uses W15 unit_order<=2849; frozen discovery set is exactly that set',
 'candidate_code_sha256':hashlib.sha256((P/'scripts/preprocessing_20261030_v1/15_blind_probability_model.py').read_bytes()).hexdigest()}
assert receipt['candidate_code_sha256']=='42dfd3597673f89b8447aab74b5e5c601e15fdbcaa703b3f7957fe53638fb5c9'
assert evaluation['selected_candidate']=='category'
(B/'preaccess_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
ref=read(C/'grouped_CV_predictions_TRE_ONLY.parquet');params=read(C/'hypothetical_training_category_parameters_TRE_ONLY.parquet')
foods=read(G/'matrix_foodid_columns_TRE_ONLY.parquet');rows=read(G/'matrix_person_rows_TRE_ONLY.parquet')
foods['category']=foods.source_categories_json.map(lambda v:json.dumps(sorted(str(x) for x in json.loads(v)),separators=(',',':')))
global_p=evaluation['global_training_posterior_mean'];lookup=dict(zip(params.category,params.q))
foods['q_fitted_reference_mean']=foods.category.map(lookup).fillna(global_p) if evaluation['selected_candidate']=='category' else global_p
qd=grouped_probability_draws(ref.label_group.to_numpy(),ref.category.to_numpy(),ref.y.to_numpy(float),ref.weight.to_numpy(float),foods.category.to_numpy(),
    draws=receipt['draws'],seed=receipt['seed'],shrinkage=10.)
assert qd.shape==(receipt['draws'],len(foods)) and np.isfinite(qd).all() and ((qd>=0)&(qd<=1)).all()
z=(np.random.default_rng(receipt['seed']+1).random(qd.shape)<qd)
foods['q_predictive_mean']=qd.mean(axis=0)
A=sparse.load_npz(io.BytesIO((G/'unknown_foodid_equalday_weights_TRE_ONLY.npz').read_bytes()))
lo=rows.UPF_lower_equalday_fraction.to_numpy(float);hi=rows.UPF_upper_equalday_fraction.to_numpy(float)
mu=lo+np.asarray(A@qd.mean(axis=0)).ravel()
assert np.all(mu>=lo-1e-12) and np.all(mu<=hi+1e-12)
rows['UPF_model_expected_fraction']=mu
rows=rows.merge(roles[['participant_id','cohort','research_stage','final_role']],on=['participant_id','cohort','research_stage'],how='left',validate='one_to_one')
np.testing.assert_array_equal(rows.matrix_row_ordinal.to_numpy(),np.arange(len(rows)))
train=rows.final_role.eq('discovery')&rows.n_energy_days.ge(3)
draw=lo[train][None,:]+np.asarray(A[train.to_numpy()]@z.T.astype(float)).T
quantiles=np.quantile(draw.ravel(),[.1,.25,.5,.75,.9],method='linear').tolist()
buf=io.BytesIO();np.savez_compressed(buf,q_probability_draws=qd,shared_food_labels=z)
file=B/'catalogue_probability_and_labels_TRE_ONLY.npz'
with file.open('xb') as f:f.write(buf.getvalue())
with np.load(io.BytesIO(file.read_bytes())) as back:
    np.testing.assert_array_equal(back['shared_food_labels'],z);np.testing.assert_array_equal(back['q_probability_draws'],qd)
pq.write_table(pa.Table.from_pandas(foods,preserve_index=False),B/'FoodID_probabilities_TRE_ONLY.parquet',compression='zstd')
pq.write_table(pa.Table.from_pandas(rows,preserve_index=False),B/'window_calibration_inputs_TRE_ONLY.parquet',compression='zstd')
for name in ['grouped_CV_predictions_TRE_ONLY.parquet','hypothetical_training_category_parameters_TRE_ONLY.parquet']:
    b=(C/name).read_bytes()
    with (B/name.replace('hypothetical_training','discovery')).open('xb') as f:f.write(b)
summary={'status':'discovery_probability_inputs_prepared','training_people':int(roles.final_role.eq('discovery').sum()),
 'training_primary_food_windows':int(train.sum()),'reference_FoodIDs':len(ref),'reference_source_groups':int(ref.label_group.nunique()),
 'selected_model':evaluation['selected_candidate'],'draws':receipt['draws'],'unknown_FoodID_columns':len(foods),
 'latent_training_quantile_probabilities':[.1,.25,.5,.75,.9],'latent_training_quantile_values':quantiles,
 'spline_knots_rule':'use predictive latent diet 10/50/90 percentiles of discovery windows',
 'contrast_rule':'predictive latent diet discovery 25th to 75th percentile; final support sealed W22',
 'source_uncertainty_included':True,'same_FoodID_label_shared_across_every_person':True,'labels_written_as_truth':False,
 'UPF_model_expected_fraction_is':'conditional model expectation, not an observed or midpoint intake',
 'CV':'refit source probability recipe within each discovery training fold; full-discovery inputs are for final fit',
 'scientific_Y_value_reads':0,'associations_fitted':False}
(B/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('UPF_PROBABILITY_INPUTS_COMPLETE',json.dumps({k:v for k,v in summary.items() if k!='latent_training_quantile_values'}),flush=True)
