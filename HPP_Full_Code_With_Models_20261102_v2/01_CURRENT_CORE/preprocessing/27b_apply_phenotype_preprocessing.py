"""Endpoint-specific technical Y QC: discovery fit, fixed apply to reserved Y."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,hashlib,resource,importlib.util
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq
from preprocessing_v1_20261030 import PhenotypeOutliers
from w18_preprocessing_reader import TechnicalPreprocessingReader,K
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
Q=P/'data_qc/preprocessing_20261030_v1';L=Path('/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1')
assert (L/'25_actual_exit_code').read_text().strip()=='0'
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(600,600))
ids=['D_Glu','D_TotalFat','D_VAT','D_Liver_elasticity','D_Liver_velocity','D_Liver_viscosity','D_Liver_attenuation','D_Liver_dispersion','D_Liver_speed_of_sound','D_BMI']
sources={v:P/'analysis'/v/'UPF_bounds/base_table_TRE_ONLY.parquet' for v in ids}
B=Q/'phenotypes_transformed_v1b';B.mkdir(exist_ok=False)
requests=[{'path':str(p),'purpose':'phenotype_QC','columns':K+['Y_value']} for p in sources.values()]
cgm=P/'analysis/shared/cgm_visit_phenotypes_TRE_ONLY.parquet'
requests.append({'path':str(cgm),'purpose':'covariate_encoding','columns':K+['n_qualified_CGM_days']})
reader=TechnicalPreprocessingReader(B/'access_receipts',requests)
cgm_days=reader.read(cgm,K+['n_qualified_CGM_days'],'covariate_encoding')[K+['n_qualified_CGM_days']]
manifest=json.loads((P/'data_qc/w15_split_access_20261029/final_artifact_manifest.json').read_text())
guardfile=next(Path(v['path']) for v in manifest if Path(v['path']).name=='w15_access_guard.py')
assert hashlib.sha256(guardfile.read_bytes()).hexdigest()=='129ae0ac28f01208ec5c88c7a6f8b947b0dcb55ff2ba314f256c7ac3119602b2'
ns={};exec(compile(guardfile.read_text(),str(guardfile),'exec'),ns)
summaries=[]
for endpoint,source in sources.items():
    # A scientific Y projection begins only after the technical preaccess receipt.
    d=reader.read(source,K+['Y_value'],'phenotype_QC')
    membership=ns['load_blind_membership'](endpoint,K)
    d=d.merge(membership,on=K,how='inner',validate='one_to_one')
    assert len(d)==len(membership),'W14 base / W15 membership mismatch'
    if endpoint=='D_Glu':
        d=d.merge(cgm_days,on=K,how='left',validate='one_to_one')
        d=d.loc[d.n_qualified_CGM_days.ge(3)].copy()
    training=reader.training(d)
    model=PhenotypeOutliers().fit(training.Y_value.to_numpy(float),training.participant_id.tolist(),endpoint=endpoint,role='training')
    dest=B/endpoint;dest.mkdir();model.save(dest/'transform.json')
    transformed=model.transform(d.Y_value.to_numpy(float),endpoint=endpoint)
    restored=PhenotypeOutliers.load(dest/'transform.json').transform(d.Y_value.to_numpy(float),endpoint=endpoint)
    for name in transformed:np.testing.assert_equal(transformed[name],restored[name])
    output=d[K+['split_unit_id','final_role']].copy().reset_index(drop=True)
    output['analysis_id']=endpoint
    output['Y_original']=transformed['original'];output['Y_processed']=transformed['values']
    for name in ['nonfinite','excluded_8sd','winsorized_low','winsorized_high']:output[name]=transformed[name]
    output['Y_primary_usable']=np.isfinite(output.Y_processed)
    file=dest/'values_TRE_ONLY.parquet'
    pq.write_table(pa.Table.from_pandas(output,preserve_index=False),file,compression='zstd')
    back=pq.ParquetFile(file,pre_buffer=False).read().to_pandas(ignore_metadata=True)
    pd.testing.assert_frame_equal(output,back,check_dtype=False)
    report={'endpoint':endpoint,'people_before_endpoint_rule':len(d),'training_fit_people':len(training),
            'training_usable':int((output.final_role.eq('discovery')&output.Y_primary_usable).sum()),
            'confirmation_usable':int((output.final_role.eq('confirmation_reserved')&output.Y_primary_usable).sum()),
            'excluded_8sd':int(output.excluded_8sd.sum()),'winsorized_low':int(output.winsorized_low.sum()),
            'winsorized_high':int(output.winsorized_high.sum()),'nonfinite':int(output.nonfinite.sum()),
            'fit_only_discovery':True,'confirmation_fixed_transform_only':True,'save_load_and_full_parquet_readback':True,
            'parameter_sha256':hashlib.sha256((dest/'transform.json').read_bytes()).hexdigest()}
    summaries.append(report);print('REAL_PHENOTYPE_QC_COMPLETE',json.dumps(report),flush=True)
summary={'endpoints':summaries,'scientific_Y_values_accessed_for_technical_QC':True,
         'confirmation_Y_access':'fixed outlier transformation only; no association, model fit, loss or selection',
         'supervised_scores_fitted':False,'associations_fitted':False,'Daily_extra_outlier_rule_applied':False}
(B/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('PHENOTYPE_PREPROCESSING_COMPLETE_NO_ASSOCIATION',flush=True)
