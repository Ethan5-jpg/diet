"""Outcome-blind, grouped reference-label probability evaluation.

Hypothetical 50% W15 whole-family prefix only, not a production W18 role freeze.
No individual outcome values. Legacy catalogue names define label-provenance
groups so repeated FoodIDs do not manufacture independent label information.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from collections import defaultdict
import hashlib,json,resource,io
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
resource.setrlimit(resource.RLIMIT_CPU,(600,600))
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
Q=P/'data_qc/preprocessing_20261030_v1';F=Q/'food_qc_v1';G=Q/'foodid_interval_geometry_v1b'
L=Path('/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1')
assert (L/'14b_actual_exit_code').read_text().strip()=='0'
B=Q/'probability_model_candidate_v1';B.mkdir(exist_ok=False)
K=['participant_id','cohort','research_stage'];KD=K+['date']
spec={'version':'blind_grouped_probability_candidate_v1','seed':20261030015,'hypothetical_fraction':.5,
      'unit_order_cutoff':2849,'folds':5,'reference_label':'inherited exact legacy NOVA catalogue, not newly adjudicated truth',
      'label_group':'connected components of normalized legacy short names sharing FoodIDs',
      'group_weight':'one total weight per inherited label-provenance group',
      'models':['global Laplace mean (Beta(1,1))','exact source-category-set with fixed shrinkage10 to global'],
      'category_shrinkage':10,'selection':'category if grouped CV logloss improves >=.01 AND abs calibration-in-large<=.05 AND ECE<=.10; otherwise global',
      'probability_sensitivity_logodds_offsets':[-1.38629436112,-.69314718056,0,.69314718056,1.38629436112],
      'source_assumptions':['known legacy classifications treated as reference','unresolved foods exchangeable with reference after category conditioning',
                            'FoodID is a coherent processing-class unit unless source audit disproves it'],
      'production_fit':False,'new_real_Y_reads':0,'NOVA_unknown_labels_filled':False}
(B/'preaccess_spec.json').write_text(json.dumps(spec,indent=2)+'\n')
def read(p,cols=None):return pq.ParquetFile(p,pre_buffer=False).read(columns=cols).to_pandas(ignore_metadata=True)
orderfile=P/'freeze/split_order_TRE_ONLY.parquet'
pf=pq.ParquetFile(orderfile,pre_buffer=False)
assert {'participant_id','unit_order'}<=set(pf.schema_arrow.names)
order=read(orderfile,['participant_id','unit_order']);assert order.participant_id.is_unique
train_people=set(order.loc[order.unit_order.le(spec['unit_order_cutoff']),'participant_id'])
days=read(F/'food_daily_primary_TRE_ONLY.parquet',KD)
days=days.loc[(days.cohort=='10k')&(days.research_stage=='00_00_visit')&days.participant_id.isin(train_people)]
assert not days.duplicated(KD).any()
food_ids=set();source=P/'analysis/shared/food_events_clean_TRE_ONLY.parquet'
for rb in pq.ParquetFile(source,pre_buffer=False).iter_batches(columns=KD+['food_id','calories_kcal_clean'],batch_size=8192,use_threads=False):
    d=rb.to_pandas(ignore_metadata=True)
    d=d.loc[(d.cohort=='10k')&(d.research_stage=='00_00_visit')&d.participant_id.isin(train_people)]
    d=d.merge(days,on=KD,how='inner',validate='many_to_one')
    food_ids.update(int(v) for v in d.loc[d.calories_kcal_clean.gt(0),'food_id'].dropna())
C=read(P/'config/food_id_crosswalk_versioned.parquet')
reference=C.loc[C.food_id.isin(food_ids)&C.NOVA.isin([1,2,3,4])].copy().reset_index(drop=True)
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
def fold(v):return int(hashlib.sha256((str(spec['seed'])+'|'+v).encode()).hexdigest()[:16],16)%spec['folds']
reference['fold']=reference.label_group.map(fold)
def fit(z):
    p=(float((z.weight*z.y).sum())+1)/(float(z.weight.sum())+2)
    rows=[]
    for cat,a in z.groupby('category',sort=False):
        n=float(a.weight.sum());s=float((a.weight*a.y).sum());k=spec['category_shrinkage']
        rows.append({'category':cat,'effective_label_groups':n,'weighted_UPF_groups':s,'q':(s+k*p)/(n+k)})
    return p,pd.DataFrame(rows)
reference['pred_global']=np.nan;reference['pred_category']=np.nan
for f in range(spec['folds']):
    training=reference.loc[reference.fold.ne(f)];held=reference.fold.eq(f)
    p,table=fit(training);lookup=dict(zip(table.category,table.q))
    reference.loc[held,'pred_global']=p
    reference.loc[held,'pred_category']=reference.loc[held,'category'].map(lookup).fillna(p)
def metrics(col):
    y=reference.y.to_numpy(float);p=np.clip(reference[col].to_numpy(float),1e-8,1-1e-8);w=reference.weight.to_numpy(float)
    bins=np.minimum((p*10).astype(int),9);ece=0.;binrows=[]
    for b in range(10):
        z=bins==b
        if not z.any():continue
        n=float(w[z].sum());obs=float(np.average(y[z],weights=w[z]));pred=float(np.average(p[z],weights=w[z]))
        ece+=n/w.sum()*abs(obs-pred);binrows.append({'bin':b,'effective_label_groups':n,'observed':obs,'predicted':pred})
    return {'weighted_logloss':float(np.average(-y*np.log(p)-(1-y)*np.log(1-p),weights=w)),
            'weighted_Brier':float(np.average((y-p)**2,weights=w)),
            'calibration_in_large':float(np.average(y-p,weights=w)),'ECE':ece,'bins':binrows}
global_m=metrics('pred_global');category_m=metrics('pred_category')
selected='category' if (category_m['weighted_logloss']<=global_m['weighted_logloss']-.01 and abs(category_m['calibration_in_large'])<=.05 and category_m['ECE']<=.10) else 'global'
p,params=fit(reference)
params['selected_model']=selected
pq.write_table(pa.Table.from_pandas(reference,preserve_index=False),B/'grouped_CV_predictions_TRE_ONLY.parquet',compression='zstd')
pq.write_table(pa.Table.from_pandas(params,preserve_index=False),B/'hypothetical_training_category_parameters_TRE_ONLY.parquet',compression='zstd')
U=read(G/'matrix_foodid_columns_TRE_ONLY.parquet')
U['category']=U.source_categories_json.map(lambda v:json.dumps(sorted(str(x) for x in json.loads(v)),separators=(',',':')))
lookup=dict(zip(params.category,params.effective_label_groups))
U['effective_reference_groups_in_exact_category']=U.category.map(lookup).fillna(0)
A=sparse.load_npz(io.BytesIO((G/'unknown_foodid_equalday_weights_TRE_ONLY.npz').read_bytes()))
contribution=np.asarray(A.sum(axis=0)).ravel();assert len(contribution)==len(U)
summary={'status':'blind_grouped_probability_candidate_evaluated','hypothetical_training_people':len(train_people),
         'training_people_with_eligible_food_days':int(days.participant_id.nunique()),'training_positive_energy_FoodIDs':len(food_ids),
         'reference_FoodIDs':len(reference),'independent_label_provenance_groups':int(reference.label_group.nunique()),
         'reference_UPF_group_prevalence':float((reference.weight*reference.y).sum()/reference.weight.sum()),
         'fold_label_groups':{str(k):int(v) for k,v in reference.groupby('fold').label_group.nunique().items()},
         'global_CV':global_m,'category_CV':category_m,'selected_candidate':selected,'global_training_posterior_mean':p,
         'unknown_energyweight_exact_category_ge20_reference_groups_fraction':float(contribution[U.effective_reference_groups_in_exact_category.ge(20)].sum()/contribution.sum()),
         'unknown_energyweight_exact_category_zero_reference_groups_fraction':float(contribution[U.effective_reference_groups_in_exact_category.eq(0)].sum()/contribution.sum()),
         'unknown_positive_weight_FoodIDs_multiple_product_names':int(((contribution>0)&U.n_products.gt(1)).sum()),
         'unknown_weight_multiple_product_names_fraction':float(contribution[U.n_products.gt(1)].sum()/contribution.sum()),
         'unknown_positive_weight_FoodIDs_multiple_short_names':int(((contribution>0)&U.n_short_names.gt(1)).sum()),
         'production_q_estimation_uncertainty_validated':False,'production_ready':False,'formal_roles_frozen':False,
         'real_scientific_Y_reads':0,'labels_filled':False,
         'interpretation':'reference-catalogue cross-validation is not external validation of unresolved foods or proof of conditional exchangeability'}
(B/'evaluation_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print('BLIND_PROBABILITY_MODEL_EVALUATED',json.dumps(summary),flush=True)
