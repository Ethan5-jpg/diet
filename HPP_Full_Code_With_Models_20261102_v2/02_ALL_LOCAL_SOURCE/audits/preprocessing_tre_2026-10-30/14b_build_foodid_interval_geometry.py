"""Food-only interval geometry and catalogue support; no Y or model fitting."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import hashlib,json,resource,io
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
resource.setrlimit(resource.RLIMIT_CPU,(600,600))
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
Q=P/'data_qc/preprocessing_20261030_v1';F=Q/'food_qc_v1';B=Q/'foodid_interval_geometry_v1b'
L=Path('/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1')
assert (L/'10_actual_exit_code').read_text().strip()=='0'
B.mkdir(exist_ok=False)
K=['participant_id','cohort','research_stage'];KD=K+['date']
receipt={'purpose':'construct exact FoodID energy-weight matrices for future model-dependent UPF evaluation',
         'scope':'baseline 10k/00_00_visit, new-QC eligible food days, all available window lengths',
         'event_columns':KD+['source_row_ordinal','food_id','calories_kcal_clean'],
         'matrices':'equal-day and cumulative-energy candidates kept separate',
         'parameter_fitting':False,'probabilities_assigned':False,'scientific_Y_columns':[]}
(B/'preaccess_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
def read(p,cols=None):return pq.ParquetFile(p,pre_buffer=False).read(columns=cols).to_pandas(ignore_metadata=True)
W=read(F/'food_windows_TRE_ONLY.parquet');W=W.loc[(W.cohort=='10k')&(W.research_stage=='00_00_visit')].sort_values(K).reset_index(drop=True)
assert not W.duplicated(K).any();W['matrix_row_ordinal']=np.arange(len(W))
D=read(F/'food_daily_primary_TRE_ONLY.parquet',KD+['energy_kcal'])
D=D.merge(W[K+['matrix_row_ordinal','n_energy_days','energy_kcal_sum_energy_days']],on=K,how='inner',validate='many_to_one')
assert not D.duplicated(KD).any()
C=read(P/'config/food_id_crosswalk_versioned.parquet')
assert C.food_id.is_unique
known=C.loc[C.NOVA.isin([1,2,3,4])];unknown=C.loc[~C.NOVA.isin([1,2,3,4])].sort_values('food_id').reset_index(drop=True)
unknown['matrix_column_ordinal']=np.arange(len(unknown))
lookup=dict(zip(known.food_id.astype(int),known.NOVA.astype(int)))
columns=dict(zip(unknown.food_id.astype(int),unknown.matrix_column_ordinal.astype(int)))
rr=[];cc=[];equal=[];cumulative=[];lower=np.zeros(len(W));lower_cum=np.zeros(len(W));total=np.zeros(len(W));total_cum=np.zeros(len(W))
missing_id_energy=0.;events_scanned=admitted_events=0
source=P/'analysis/shared/food_events_clean_TRE_ONLY.parquet'
for rb in pq.ParquetFile(source,pre_buffer=False).iter_batches(columns=receipt['event_columns'],batch_size=8192,use_threads=False):
    d=rb.to_pandas(ignore_metadata=True);events_scanned+=len(d)
    d=d.loc[(d.cohort=='10k')&(d.research_stage=='00_00_visit')]
    d=d.merge(D,on=KD,how='inner',validate='many_to_one');admitted_events+=len(d)
    if not len(d):continue
    e=pd.to_numeric(d.calories_kcal_clean,errors='coerce').to_numpy(float)
    assert np.isfinite(e).all() and np.all(e>=0)
    row=d.matrix_row_ordinal.to_numpy(int)
    we=e/d.energy_kcal.to_numpy(float)/d.n_energy_days.to_numpy(float)
    wc=e/d.energy_kcal_sum_energy_days.to_numpy(float)
    np.add.at(total,row,we);np.add.at(total_cum,row,wc)
    nova=d.food_id.map(lookup)
    upf=nova.eq(4).to_numpy();np.add.at(lower,row[upf],we[upf]);np.add.at(lower_cum,row[upf],wc[upf])
    un=~nova.isin([1,2,3,4]).to_numpy();col=d.food_id.map(columns)
    no_id=un&col.isna().to_numpy();missing_id_energy+=float(e[no_id].sum())
    valid=un&col.notna().to_numpy()&(e>0)
    rr.append(row[valid]);cc.append(col.loc[valid].to_numpy(int));equal.append(we[valid]);cumulative.append(wc[valid])
assert missing_id_energy==0,'positive unknown energy without resolvable FoodID; cannot pool unrelated unknown records'
assert events_scanned==pq.ParquetFile(source).metadata.num_rows
A=sparse.coo_matrix((np.concatenate(equal),(np.concatenate(rr),np.concatenate(cc))),shape=(len(W),len(unknown))).tocsr();A.sum_duplicates()
Ac=sparse.coo_matrix((np.concatenate(cumulative),(np.concatenate(rr),np.concatenate(cc))),shape=A.shape).tocsr();Ac.sum_duplicates()
np.testing.assert_allclose(total,1.,atol=1e-10);np.testing.assert_allclose(total_cum,1.,atol=1e-10)
np.testing.assert_allclose(lower,W.UPF_lower_equal_day_mean_all_energy_days/100,atol=1e-10)
np.testing.assert_allclose(lower+np.asarray(A.sum(axis=1)).ravel(),W.UPF_upper_equal_day_mean_all_energy_days/100,atol=1e-10)
np.testing.assert_allclose(lower_cum,W.UPF_lower_cumulative_all_energy_days/100,atol=1e-10)
np.testing.assert_allclose(lower_cum+np.asarray(Ac.sum(axis=1)).ravel(),W.UPF_upper_cumulative_all_energy_days/100,atol=1e-10)
rows=W[K+['matrix_row_ordinal','n_energy_days','n_fiber_days','first_recorded_date','last_recorded_date']].copy()
rows['UPF_lower_equalday_fraction']=lower;rows['UPF_lower_cumulative_fraction']=lower_cum
rows['UPF_upper_equalday_fraction']=lower+np.asarray(A.sum(axis=1)).ravel()
rows['UPF_upper_cumulative_fraction']=lower_cum+np.asarray(Ac.sum(axis=1)).ravel()
def save_sparse(path,matrix):
    buf=io.BytesIO();sparse.save_npz(buf,matrix);payload=buf.getvalue()
    with path.open('xb') as f:f.write(payload)
    reread=sparse.load_npz(io.BytesIO(path.read_bytes()))
    assert (reread-matrix).nnz==0
save_sparse(B/'unknown_foodid_equalday_weights_TRE_ONLY.npz',A)
save_sparse(B/'unknown_foodid_cumulative_weights_TRE_ONLY.npz',Ac)
pq.write_table(pa.Table.from_pandas(rows,preserve_index=False),B/'matrix_person_rows_TRE_ONLY.parquet',compression='zstd')
pq.write_table(pa.Table.from_pandas(unknown,preserve_index=False),B/'matrix_foodid_columns_TRE_ONLY.parquet',compression='zstd')
assert (sparse.load_npz(io.BytesIO((B/'unknown_foodid_equalday_weights_TRE_ONLY.npz').read_bytes()))-A).nnz==0
assert (sparse.load_npz(io.BytesIO((B/'unknown_foodid_cumulative_weights_TRE_ONLY.npz').read_bytes()))-Ac).nnz==0
def category(v):
    a=json.loads(v)
    return tuple(sorted(str(x) for x in a))
known_category_counts=known.source_categories_json.map(category).value_counts().to_dict()
unknown['known_FoodIDs_in_same_exact_category_set']=[int(known_category_counts.get(category(v),0)) for v in unknown.source_categories_json]
contribution=np.asarray(A.sum(axis=0)).ravel();total_unknown=contribution.sum()
category_ge20=unknown.known_FoodIDs_in_same_exact_category_set.ge(20).to_numpy()
nonzero=contribution>0
summary={'status':'exact_FoodID_interval_geometry_verified','event_rows_scanned':events_scanned,
         'baseline_primary_events':admitted_events,'baseline_people':len(W),'unknown_FoodID_columns':len(unknown),
         'nonzero_person_FoodID_cells':A.nnz,'unknown_FoodIDs_with_positive_weight':int(nonzero.sum()),
         'same_FoodID_coupling':'one column per unresolved FoodID shared by all people',
         'equalday_and_cumulative_bounds_verified_against_W07_compatible_new_QC_windows':True,
         'missing_resolvable_FoodID_positive_energy':missing_id_energy,
         'effective_unknown_FoodIDs_equalday_weight_concentration':float(total_unknown**2/np.sum(contribution**2)),
         'top10_unknown_FoodID_weight_share':float(np.sort(contribution)[-10:].sum()/total_unknown),
         'unknown_weight_exact_category_set_with_ge20_known_FoodIDs_fraction':float(contribution[category_ge20].sum()/total_unknown),
         'unknown_positive_weight_IDs_exact_category_set_without_any_known_FoodIDs':int((nonzero&unknown.known_FoodIDs_in_same_exact_category_set.eq(0).to_numpy()).sum()),
         'unknown_positive_weight_IDs_multiple_product_names':int((nonzero&unknown.n_products.gt(1).to_numpy()).sum()),
         'unknown_weight_multiple_product_names_fraction':float(contribution[unknown.n_products.gt(1)].sum()/total_unknown),
         'NOVA_probability_model_fitted':False,'formal_W18_frozen':False,'new_real_Y_reads':0,
         'interpretation':'weight concentration is a technical uncertainty diagnostic, not a number of independent participants or validated classification probabilities'}
pq.write_table(pa.Table.from_pandas(unknown,preserve_index=False),B/'foodid_category_support_TRE_ONLY.parquet',compression='zstd')
(B/'verification_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('FOODID_GEOMETRY_VERIFIED',json.dumps(summary),flush=True)
