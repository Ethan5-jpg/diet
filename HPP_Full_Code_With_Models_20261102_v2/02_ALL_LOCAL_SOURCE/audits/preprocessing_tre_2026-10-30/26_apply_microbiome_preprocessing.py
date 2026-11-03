"""Fit agreed microbiome transformations on discovery roles and apply to both."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import hashlib,json,resource
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq
from preprocessing_v1_20261030 import MicrobiomeCLR
from w18_preprocessing_reader import TechnicalPreprocessingReader,K
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
Q=P/'data_qc/preprocessing_20261030_v1';L=Path('/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1')
assert (L/'25_actual_exit_code').read_text().strip()=='0'
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(900,900))
B=Q/'microbiome_transformed_v1';B.mkdir(exist_ok=False)
sources={site:P/'analysis/shared'/(site+'_selected_abundance_TRE_ONLY.parquet') for site in ['oral','gut']}
taxa={}
for site,p in sources.items():
    schema=pq.ParquetFile(p,pre_buffer=False).schema_arrow
    taxa[site]=[n for n in schema.names if n.startswith('k__') or n.startswith('s__') or '|s__' in n]
    assert len(taxa[site])>=2
    assert all(pa.types.is_floating(schema.field(n).type) for n in taxa[site])
reader=TechnicalPreprocessingReader(B/'access_receipts',[{'path':str(p),'purpose':'microbiome_transform','columns':K+taxa[site]} for site,p in sources.items()])
summary=[]
for site,p in sources.items():
    d=reader.read(p,K+taxa[site],'microbiome_transform')
    x=d[taxa[site]].to_numpy(float);assert np.isfinite(x).all() and (x>=0).all()
    assert np.allclose(x.sum(axis=1),1,atol=1e-3,rtol=0)
    train=d.final_role.eq('discovery').to_numpy();ids=d.loc[train,'participant_id'].tolist()
    assert len(ids)==len(reader.training(d)) and len(set(ids))==len(ids)
    model=MicrobiomeCLR().fit(x[train],taxa[site],ids,site=site,role='training')
    dest=B/site;dest.mkdir();model.save(dest/'transform.json')
    result=model.transform(x,taxa[site],site=site)
    loaded=MicrobiomeCLR.load(dest/'transform.json').transform(x,taxa[site],site=site)
    np.testing.assert_allclose(result['values'],loaded['values'],rtol=0,atol=0,equal_nan=True)
    supported=result['supported'];z=result['values'];fitrows=train&supported
    np.testing.assert_allclose(z[fitrows].mean(axis=0),0,atol=1e-10)
    np.testing.assert_allclose(z[fitrows].std(axis=0,ddof=0),1,atol=1e-10)
    rows=d[K+['split_unit_id','final_role']].copy();rows['matrix_row']=np.arange(len(rows));rows['supported']=supported
    pq.write_table(pa.Table.from_pandas(rows,preserve_index=False),dest/'rows_TRE_ONLY.parquet',compression='zstd')
    table=pa.Table.from_arrays([pa.array(z[:,i],type=pa.float64()) for i in range(z.shape[1])],names=result['feature_names'])
    pq.write_table(table,dest/'CLR_standardized_TRE_ONLY.parquet',compression='zstd',row_group_size=1024)
    check=pq.ParquetFile(dest/'CLR_standardized_TRE_ONLY.parquet',pre_buffer=False).read().to_pandas(ignore_metadata=True).to_numpy(float)
    np.testing.assert_allclose(check,z,atol=0,rtol=0,equal_nan=True)
    report={'site':site,'people':len(d),'training_people':int(train.sum()),'confirmation_people':int((~train).sum()),
            'source_species':len(taxa[site]),'CLR_reference_species':len(model.state['reference_indices']),
            'output_species':z.shape[1],'unsupported_rows':int((~supported).sum()),
            'fit_only_discovery':True,'save_load_exact':True,'parquet_readback_exact':True,
            'parameter_sha256':hashlib.sha256((dest/'transform.json').read_bytes()).hexdigest()}
    summary.append(report);print('REAL_MICROBIOME_TRANSFORMED',json.dumps(report),flush=True)
    del d,x,z,check,table,loaded,result
(B/'summary.json').write_text(json.dumps({'sites':summary,'new_real_Y_reads':0,
 'CV_use':'reference transforms for full discovery fit only; refit in each training fold before CV','supervised_score_fitted':False},indent=2)+'\n')
print('MICROBIOME_PREPROCESSING_COMPLETE_TRAIN_FIT_CONFIRM_APPLY',flush=True)
