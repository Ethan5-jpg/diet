"""Assemble processed inputs and participant flow; no association fitting."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import hashlib,json,resource
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq
import pyarrow.dataset as ds
from w16_estimators import CovariateEncoder
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
Q=P/'data_qc/preprocessing_20261030_v1';L=Path('/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1')
K=['participant_id','cohort','research_stage'];KD=K+['date']
for tag in ['26','27b','28']:assert (L/(tag+'_actual_exit_code')).read_text().strip()=='0',tag
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(600,600))
B=Q/'analysis_ready_v1';B.mkdir(exist_ok=False);R=B/'access_receipts';R.mkdir()
pointer=json.loads((P/'config/preprocessing_20261030_v1/active_execution.json').read_text())
assert hashlib.sha256(Path(pointer['roles']).read_bytes()).hexdigest()==pointer['roles_sha256']
roles=pq.ParquetFile(pointer['roles'],pre_buffer=False).read(columns=K+['split_unit_id','final_role']).to_pandas(ignore_metadata=True)
sequence=0
def read(path,columns=None,*,keys=K,has_Y=False):
    global sequence
    path=Path(path);pf=pq.ParquetFile(path,pre_buffer=False)
    columns=list(columns or pf.schema_arrow.names);assert set(keys)<=set(columns)
    assert pf.metadata.num_rows*len(columns)*16<1024**3
    sequence+=1;receipt={'path':str(path),'columns':columns,'purpose':'assemble_model_ready_inputs_no_association',
      'contains_Y_values':has_Y,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'roles_sha256':pointer['roles_sha256'],
      'supervised_score_or_association_fitting':False}
    (R/(str(sequence).zfill(3)+'_preaccess.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    filt=(ds.field('cohort')=='10k')&(ds.field('research_stage')=='00_00_visit')&ds.field('participant_id').isin(roles.participant_id.tolist())
    d=ds.dataset(path,format='parquet').scanner(columns=columns,filter=filt,use_threads=False).to_table().to_pandas(ignore_metadata=True)
    assert not d.duplicated(keys).any(),str(path)
    return d
def save(d,path):
    d=d.reset_index(drop=True);pq.write_table(pa.Table.from_pandas(d,preserve_index=False),path,compression='zstd')
    check=pq.ParquetFile(path,pre_buffer=False).read().to_pandas(ignore_metadata=True)
    pd.testing.assert_frame_equal(d,check,check_dtype=False)
foodcols=KD+['energy_kcal','fiber_g','fiber_density_g_per_1000kcal','fiber_eligible','new_QC_personal_low25SD_flag']
days=read(Q/'food_qc_v1/food_daily_primary_TRE_ONLY.parquet',foodcols,keys=KD)
days['date']=pd.to_datetime(days.date).dt.strftime('%Y-%m-%d')
energy=days.groupby(K).agg(n_energy_days=('date','size'),mean_daily_energy_UPF=('energy_kcal','mean'),
    low25SD_days=('new_QC_personal_low25SD_flag','sum'),food_first=('date','min'),food_last=('date','max')).reset_index()
fd=days.loc[days.fiber_eligible].copy()
fiber=fd.groupby(K).agg(n_fiber_days=('date','size'),fiber_density=('fiber_density_g_per_1000kcal','mean'),
    mean_daily_energy_fiber=('energy_kcal','mean'),fiber_sum=('fiber_g','sum'),fiber_energy_sum=('energy_kcal','sum')).reset_index()
fiber['fiber_density_cumulative']=1000*fiber.fiber_sum/fiber.fiber_energy_sum
food=energy.merge(fiber,on=K,how='left',validate='one_to_one')
upf=read(Q/'UPF_probability_inputs_v1/window_calibration_inputs_TRE_ONLY.parquet',K+['matrix_row_ordinal','UPF_lower_equalday_fraction','UPF_upper_equalday_fraction','UPF_model_expected_fraction'])
food=food.merge(upf,on=K,how='left',validate='one_to_one')
covnames=['age_calendar_years','sex_code','smoking_current_status_label']
cov=read(P/'analysis/shared/covariate_registry_TRE_ONLY.parquet',K+covnames)
for n in ['age_calendar_years','sex_code']:cov[n]=pd.to_numeric(cov[n],errors='coerce')
cov['smoking_status']=cov.smoking_current_status_label.astype('string').fillna('__UNKNOWN__').replace('', '__UNKNOWN__')
cov=cov[K+['age_calendar_years','sex_code','smoking_status']]
X=roles.copy()
for site,short in [('oral','O'),('gut','G')]:
    rows=read(Q/'microbiome_transformed_v1'/site/'rows_TRE_ONLY.parquet',K+['matrix_row','supported'])
    rows=rows.rename(columns={'matrix_row':short+'_matrix_row','supported':short+'_supported'})
    X=X.merge(rows,on=K,how='left',validate='one_to_one')
X=X.merge(cov,on=K,how='left',validate='one_to_one').merge(food,on=K,how='left',validate='one_to_one')
X['core_available']=np.isfinite(X.age_calendar_years)&X.sex_code.isin([0,1])
X['microbiome_available']=X.O_supported.eq(True)&X.G_supported.eq(True)
summaries=[];flows=[]
primary=['D_Glu','D_TotalFat','D_VAT','D_Liver_elasticity','D_Liver_velocity','D_Liver_viscosity','D_Liver_attenuation','D_Liver_dispersion','D_Liver_speed_of_sound']
for endpoint in primary+['D_BMI']:
    y=read(Q/'phenotypes_transformed_v1b'/endpoint/'values_TRE_ONLY.parquet',K+['Y_original','Y_processed','Y_primary_usable','excluded_8sd','winsorized_low','winsorized_high'],has_Y=True)
    unitdata=read(P/'analysis'/endpoint/'UPF_bounds/base_table_TRE_ONLY.parquet',K+['Y_unit'])
    units=unitdata.Y_unit.dropna().unique().tolist();assert len(units)==1
    full=X.merge(y,on=K,how='inner',validate='one_to_one');full['Y_unit']=units[0]
    for exposure in ['UPF','fiber_density']:
        d=full.copy();d['analysis_id']=endpoint;d['exposure_id']=exposure
        if exposure=='UPF':
            diet=d.n_energy_days.ge(3)&np.isfinite(d.UPF_model_expected_fraction)
            d['mean_daily_energy']=d.mean_daily_energy_UPF
            d['exposure_model_mean']=d.UPF_model_expected_fraction
            d['exposure_is_model_expectation']=True
            d['diet_days']=d.n_energy_days
        else:
            diet=d.n_fiber_days.ge(3)&np.isfinite(d.fiber_density)
            d['mean_daily_energy']=d.mean_daily_energy_fiber
            d['exposure_model_mean']=d.fiber_density
            d['exposure_is_model_expectation']=False
            d['diet_days']=d.n_fiber_days
        masks=[('endpoint_members',np.ones(len(d),bool)),('new_diet_QC_and_3days',diet.to_numpy()),
               ('required_conditions',d.core_available.to_numpy()),('supported_O_and_G',d.microbiome_available.to_numpy()),
               ('endpoint_8SD_QC',d.Y_primary_usable.to_numpy())]
        keep=np.ones(len(d),bool)
        for stage,mask in masks:
            keep&=mask
            for role in ['discovery','confirmation_reserved']:
                flows.append({'analysis_id':endpoint,'exposure_id':exposure,'stage':stage,'role':role,'people':int((keep&d.final_role.eq(role).to_numpy()).sum())})
        d=d.loc[keep].reset_index(drop=True);assert len(d)>0
        dest=B/endpoint/exposure;dest.mkdir(parents=True)
        numeric=['age_calendar_years','sex_code','mean_daily_energy'];categorical=['smoking_status']
        training=d.final_role.eq('discovery').to_numpy()
        encoder=CovariateEncoder().fit({n:d.loc[training,n].tolist() for n in numeric+categorical},d.loc[training,'participant_id'].tolist(),numeric=numeric,categorical=categorical,role='training')
        encoder.save(dest/'conditions_transform.json')
        enc=encoder.transform({n:d[n].tolist() for n in numeric+categorical})
        for j in range(enc.shape[1]):d['C_encoded_'+str(j)]=enc[:,j]
        save(d,dest/'table_TRE_ONLY.parquet')
        r={'analysis_id':endpoint,'exposure_id':exposure,'primary':endpoint in primary,'discovery':int(training.sum()),
           'confirmation':int((~training).sum()),'units':int(d.split_unit_id.nunique()),'people_ge7days':int(d.diet_days.ge(7).sum()),
           'Y_unit':units[0],'path':str(dest/'table_TRE_ONLY.parquet')}
        summaries.append(r);print('ANALYSIS_READY_PERSON_TABLE',json.dumps(r),flush=True)
# Daily retains daily QC and does not inherit person-level CGM outlier exclusions.
dailyfile=P/'analysis/Daily_CGM/fiber_density/base_table_TRE_ONLY.parquet'
schema=pq.ParquetFile(dailyfile,pre_buffer=False).schema_arrow.names
cols=KD+['Y_value']+(['base_eligible'] if 'base_eligible' in schema else [])
daily=read(dailyfile,cols,keys=KD,has_Y=True)
if 'base_eligible' in daily:daily=daily.loc[daily.base_eligible].copy()
daily['date']=pd.to_datetime(daily.date).dt.strftime('%Y-%m-%d')
daily=daily.merge(fd[KD+['energy_kcal','fiber_density_g_per_1000kcal','new_QC_personal_low25SD_flag']],on=KD,how='inner',validate='one_to_one')
daily=daily.merge(X[K+['split_unit_id','final_role','O_matrix_row','G_matrix_row','core_available','microbiome_available','age_calendar_years','sex_code','smoking_status']],on=K,how='inner',validate='many_to_one')
daily=daily.loc[daily.core_available&daily.microbiome_available&np.isfinite(daily.Y_value)&np.isfinite(daily.fiber_density_g_per_1000kcal)].copy()
daily['n_common_days']=daily.groupby(K).date.transform('size');daily=daily.loc[daily.n_common_days.ge(3)].copy()
daily['person_weight']=1/daily.n_common_days
np.testing.assert_allclose(daily.groupby(K).person_weight.sum().to_numpy(),1.,atol=1e-12)
daily['within_person_fiber_variance']=daily.groupby(K).fiber_density_g_per_1000kcal.transform('var')
daily['recording_day_index']=pd.to_datetime(daily.date).groupby([daily[n] for n in K]).transform(lambda s:(s-s.min()).dt.days)
daily['weekday']=pd.to_datetime(daily.date).dt.dayofweek
dest=B/'Daily_CGM/fiber_density';dest.mkdir(parents=True)
save(daily,dest/'table_TRE_ONLY.parquet')
daily_summary={'days':len(daily),'people':int(daily.participant_id.nunique()),
 'discovery_people':int(daily.loc[daily.final_role.eq('discovery'),'participant_id'].nunique()),
 'confirmation_people':int(daily.loc[daily.final_role.eq('confirmation_reserved'),'participant_id'].nunique()),
 'people_positive_within_fiber_variance':int(daily.loc[daily.within_person_fiber_variance.gt(1e-12),'participant_id'].nunique()),
 'no_person_outlier_rule':True,'path':str(dest/'table_TRE_ONLY.parquet')}
pd.DataFrame(flows).to_csv(B/'participant_flow.csv',index=False)
summary={'person_tables':summaries,'daily':daily_summary,'roles_sha256':pointer['roles_sha256'],
 'associations_fitted':False,'supervised_scores_fitted':False,
 'CV_instruction':'Read raw inputs and refit preprocessing within each training fold; reference transformed matrices are for full-discovery final fits',
 'UPF_instruction':'exposure_model_mean is not sufficient for nonlinear regression; integrate basis using shared FoodID draw inputs and frozen source model',
 'next':'W19/W20 by assigned line; preserve confirmation association reserve until W22'}
(B/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('ANALYSIS_READY_TABLES_COMPLETE',json.dumps(daily_summary),flush=True)
