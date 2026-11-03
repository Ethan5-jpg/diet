"""Exploratory within-person daily association; raw individual data stay in TRE."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,time,hashlib,resource,io
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq,pyarrow.dataset as ds
from scipy import stats
from targeted_core import cluster_fit
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');B=P/'data_qc/w20_simplified_20261101_v2';N=P/'data_qc/w20_targeted_20261101_v3';O=N/'results'/'attempt49b';K=['participant_id','cohort','research_stage'];KD=K+['date'];start=time.monotonic()
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(3600,3600))
assert json.loads((O/'model_summary.json').read_text())['status']=='fiber_Glu_targeted_complete'
def rd(p,cols=None):return pq.ParquetFile(p,pre_buffer=False).read(columns=cols).to_pandas(ignore_metadata=True)
X=rd(B/'discovery_covariates_food_TRE_ONLY.parquet');assert X.final_role.eq('discovery').all() and X.participant_id.is_unique
reserved=set(rd(B/'roles_v2_full_TRE_ONLY.parquet',['participant_id','final_role']).query('final_role=="confirmation_reserved"').participant_id)
filt=(ds.field('cohort')=='10k')&(ds.field('research_stage')=='00_00_visit')&ds.field('participant_id').isin(X.participant_id.tolist())
receipts=[]
def read(p,cols,purpose):
 pf=pq.ParquetFile(p,pre_buffer=False);assert pf.metadata.num_rows*len(cols)*16<512*1024**2
 receipts.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'fields':cols,'purpose':purpose,'filter':'discovery v2 IDs,10k,00_00_visit before materialization'})
 (O/'daily_input_receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
 d=ds.dataset(p,format='parquet').scanner(columns=cols,filter=filt,use_threads=False,batch_size=1024).to_table().to_pandas(ignore_metadata=True)
 assert not set(d.participant_id)&reserved
 d['date']=pd.to_datetime(d.date,errors='raise').dt.normalize();assert d.date.notna().all();assert not d.duplicated(KD).any()
 return d
raw=P/'analysis/Daily_CGM/fiber_density/base_table_TRE_ONLY.parquet';schema=pq.ParquetFile(raw).schema_arrow.names
D=read(raw,KD+['Y_value']+([ 'base_eligible'] if 'base_eligible' in schema else [])+(['Y_unit'] if 'Y_unit' in schema else []),'daily CGM baseline');flow={'discovery_raw_cgm_days':len(D),'discovery_raw_cgm_people':int(D.participant_id.nunique())}
if 'base_eligible' in D:D=D.loc[D.base_eligible].copy()
D=D.loc[np.isfinite(D.Y_value)].copy();flow['cgm_valid_days']=len(D)
food=P/'data_qc/preprocessing_20261030_v1/food_qc_v1/food_daily_primary_TRE_ONLY.parquet'
F=read(food,KD+['energy_kcal','fiber_g','fiber_density_g_per_1000kcal','fiber_eligible','energy_eligible'],'daily primary food');flow['discovery_primary_food_days']=len(F)
F=F.loc[F.fiber_eligible&F.energy_eligible&np.isfinite(F.energy_kcal)&F.energy_kcal.gt(0)&np.isfinite(F.fiber_density_g_per_1000kcal)].copy();flow['valid_fiber_food_days']=len(F)
# Source energy primary eligibility is built into this primary food table; retain its exact daily fiber flag.
np.testing.assert_allclose(F.fiber_density_g_per_1000kcal,1000*F.fiber_g/F.energy_kcal,atol=1e-8)
J=D.merge(F,on=KD,how='inner',validate='one_to_one');flow.update(exact_common_days=len(J),exact_common_people=int(J.participant_id.nunique()),unmatched_CGM_days=len(D)-len(J),unmatched_fiber_days=len(F)-len(J))
J=J.merge(X[K+['split_unit_id','cv_fold','age_calendar_years','sex_code']],on=K,how='inner',validate='many_to_one')
J=J.loc[np.isfinite(J.age_calendar_years)&J.sex_code.isin([0,1])].copy();J['n_i']=J.groupby('participant_id').date.transform('size');J=J.loc[J.n_i.ge(3)].copy();flow.update(ge3_people=int(J.participant_id.nunique()),ge3_days=len(J),ge3_families=int(J.split_unit_id.nunique()))
Q=rd(O/'development_predictions_and_scores_TRE_ONLY.parquet',K+['daily_oral_score','daily_gut_score']);J=J.merge(Q,on=K,how='left',validate='many_to_one');valid=np.isfinite(J[['daily_oral_score','daily_gut_score']]).all(axis=1);flow['people_lacking_held_family_score']=int(J.loc[~valid,'participant_id'].nunique());J=J.loc[valid].copy()
J['n_i']=J.groupby('participant_id').date.transform('size');assert J.n_i.ge(3).all();J['person_weight']=1/J.n_i;np.testing.assert_allclose(J.groupby('participant_id').person_weight.sum(),1.,atol=1e-12)
J['recording_day_index']=(J.date-J.groupby('participant_id').date.transform('min')).dt.days;J['weekday']=J.date.dt.dayofweek
J['fiber5']=J.fiber_density_g_per_1000kcal/5.;J['energy1000']=J.energy_kcal/1000.;J['fiber_oral']=J.fiber5*J.daily_oral_score;J['fiber_gut']=J.fiber5*J.daily_gut_score
# Host interactions preserve the cross-sectional score-learning background while fixed effects absorb host main terms.
J['fiber_age10']=J.fiber5*(J.age_calendar_years-50)/10.;J['fiber_sex']=J.fiber5*J.sex_code
for w in range(1,7):J['weekday'+str(w)]=(J.weekday==w).astype(float)
columns=['fiber5','fiber_oral','fiber_gut','energy1000','recording_day_index','fiber_age10','fiber_sex']+['weekday'+str(w) for w in range(1,7)]
A=J[columns].to_numpy(float);Y=J.Y_value.to_numpy(float);group=J.groupby('participant_id',sort=False);M=group[columns].transform('mean').to_numpy(float);Ay=A-M;yy=Y-group.Y_value.transform('mean').to_numpy(float)
# Remove only algebraically zero/redundant nuisance columns; never remove a primary interaction based on P or stability.
from scipy import linalg
_,R,piv=linalg.qr(Ay*np.sqrt(J.person_weight.to_numpy()[:,None]),mode='economic',pivoting=True);rank=np.sum(np.abs(np.diag(R))>max(np.max(np.abs(np.diag(R))),1)*1e-10);keep=sorted(piv[:rank]);assert all(j in keep for j in [0,1,2]);names=[columns[i] for i in keep]
xx=Ay[:,keep];people=int(J.participant_id.nunique());beta,cov,ci,G,res=cluster_fit(yy,xx,J.person_weight,J.split_unit_id,absorbed=people)
zero=group.fiber5.var().fillna(0).le(1e-12);flow.update(final_people=people,final_families=G,final_days=len(J),people_zero_within_fiber_variance=int(zero.sum()),min_common_days=int(J.n_i.min()),median_common_days=float(group.size().median()),max_common_days=int(J.n_i.max()),total_weight=float(J.person_weight.sum()),fold_people={str(k):int(v) for k,v in J.drop_duplicates('participant_id').cv_fold.value_counts().sort_index().items()},date_min=str(J.date.min().date()),date_max=str(J.date.max().date()))
unit=J.Y_unit.dropna().unique().tolist() if 'Y_unit' in J else [json.loads((O/'model_summary.json').read_text())['Y_unit']];assert len(unit)==1
rows=[]
for j,name in enumerate(names):
 row={'term':name,'beta':float(beta[j]),'SE':float(np.sqrt(cov[j,j])),'CI95':ci[j].tolist(),'Y_unit':unit[0],'df_household':G-1};rows.append(row);print('DAILY_EFFECT',json.dumps(row),flush=True)
ix=[names.index('fiber_oral'),names.index('fiber_gut')];W=float(beta[ix]@np.linalg.solve(cov[np.ix_(ix,ix)],beta[ix]));joint={'Wald_chi2':W,'F':W/2,'df1':2,'df2':G-1,'exploratory_p':float(stats.f.sf(W/2,2,G-1))}
# Within R2 is for exactly the weighted demeaned target; compare matched no-score-interaction regression.
nullix=[j for j in range(len(names)) if j not in ix];nb,nc,nci,ng,nr=cluster_fit(yy,xx[:,nullix],J.person_weight,J.split_unit_id,absorbed=people)
ss=float(np.sum(J.person_weight.to_numpy()*yy**2));r2=1-float(np.sum(J.person_weight.to_numpy()*res**2))/ss;nr2=1-float(np.sum(J.person_weight.to_numpy()*nr**2))/ss
result={'status':'daily_development_within_person_complete','python':os.sys.executable,'pid':os.getpid(),'seconds':time.monotonic()-start,'flow':flow,'effects':rows,'joint':joint,'within_R2':r2,'matched_null_within_R2':nr2,'delta_within_R2_descriptive':r2-nr2,'score_source':'fixed k5, main1, interaction10 v2 D_Glu/fiber; own household excluded','same_person_total_weight':1,'fixed_effect':'participant_id','cluster':'split_unit_id','calendar_join':'exact normalized date; no shift','no_person_mean_8SD_filter':True,'CI_scope':'conditional on learned scores; household CR1, N-personFE-rank df correction; does not include direction-learning uncertainty or adaptation','confirmation_Y_materialized':0,'formal55_tests_started':False,'Y_unit':unit[0],'retained_columns':names,'removed_algebraic_nuisance':[c for c in columns if c not in names]}
(O/'daily_summary.json').write_text(json.dumps(result,indent=2)+'\n');pd.DataFrame(rows).to_csv(O/'daily_effects.csv',index=False)
pq.write_table(pa.Table.from_pandas(J.reset_index(drop=True),preserve_index=False),O/'daily_analysis_TRE_ONLY.parquet',compression='zstd');pd.testing.assert_frame_equal(J.reset_index(drop=True),rd(O/'daily_analysis_TRE_ONLY.parquet'),check_dtype=False)
buf=io.BytesIO();np.savez_compressed(buf,y=yy,x=xx,weights=J.person_weight.to_numpy(),groups=J.split_unit_id.astype(str).to_numpy(),beta=beta,cov=cov,CI=ci,names=np.array(names))
with (O/'daily_regression_TRE_ONLY.npz').open('xb') as fh:fh.write(buf.getvalue())
print('DAILY_FLOW',json.dumps(flow),flush=True);print('DAILY_COMPLETE',json.dumps({k:v for k,v in result.items() if k not in ['effects','flow']}),flush=True)
