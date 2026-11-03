"""Materialize outcome and CLR/PC preprocessing views; no predictive model."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib,io,subprocess
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq
O=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1')
L=Path('/home/ec2-user/multisystem_runtime_20261021/collaborator_handoff_20261102_v1')
assert (L/'actual_exit_code.txt').read_text().strip()=='0'
sys.path.insert(0,str(O/'preprocessing/code'))
from preprocessing_v1_20261030 import PhenotypeOutliers
def rd(p):return pq.ParquetFile(p,pre_buffer=False).read().to_pandas(ignore_metadata=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def js(p,x):
 with p.open('x') as f:f.write(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
schema=[]
def save(d,p):
 p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists()
 pq.write_table(pa.Table.from_pandas(d,preserve_index=False),p,compression='zstd')
 back=rd(p);pd.testing.assert_frame_equal(d.reset_index(drop=True),back,check_dtype=False)
 schema.append(dict(path=str(p.relative_to(O)),rows=len(d),columns=list(d.columns)))
B=O/'inputs/discovery_v2';X=rd(B/'discovery_covariates_food_TRE_ONLY.parquet');n=len(X)
assert n==4427 and X.final_role.eq('discovery').all()
np.testing.assert_array_equal(X.input_row,np.arange(n))
keys=['participant_id','cohort','research_stage'];endpoints=sorted(B.glob('D_*_raw_Y_TRE_ONLY.parquet'));assert len(endpoints)==9
counts=[];refs=[];processed=O/'preprocessing/processed';processed.mkdir(exist_ok=False)
ready=O/'inputs/analysis_ready/full';ready.mkdir(parents=True,exist_ok=False)
for key in ['fold0','fold1','fold2','fold3','fold4','full']:
 F=O/'preprocessing/folds'/key;D=processed/key;D.mkdir()
 with np.load(io.BytesIO((F/'transformed_discovery_TRE_ONLY.npz').read_bytes()),allow_pickle=False) as f:a={k:f[k] for k in f.files}
 tr=a['training'].astype(bool)
 np.testing.assert_array_equal(tr,np.ones(n,bool) if key=='full' else X.cv_fold.ne(int(key[-1])).to_numpy())
 for site in ['oral','gut']:
  z=a[site+'_z'];pc=a[site+'_PC'];state=json.loads((F/(site+'_transform.json')).read_text())['CLR']
  features=[state['source_features'][state['reference_indices'][int(j)]] for j in state['output_indices']]
  assert len(features)==z.shape[1]
  cols=['CLR_'+str(i+1).zfill(4) for i in range(z.shape[1])]
  pcols=['PC'+str(i+1).zfill(2) for i in range(pc.shape[1])]
  frame=pd.DataFrame(np.column_stack([z,pc]),columns=cols+pcols)
  frame.loc[~a['supported'],cols+pcols]=np.nan
  frame.insert(0,'supported',a['supported']);frame.insert(0,'input_row',X.input_row.to_numpy())
  save(frame,D/(site+'_CLR_PC_TRE_ONLY.parquet'))
  js(D/(site+'_column_dictionary.json'),dict(CLR=dict(zip(cols,features)),PC=pcols,source=str((F/(site+'_transform.json')).relative_to(O)),unsupported='NaN using joint-site supported mask'))
 transforms={}
 for yp in endpoints:
  endpoint=yp.name.removesuffix('_raw_Y_TRE_ONLY.parquet');yd=rd(yp);rows=yd.input_row.to_numpy(int)
  assert not yd.duplicated(keys).any() and len(set(rows))==len(rows)
  np.testing.assert_array_equal(X.iloc[rows][keys].astype(str).to_numpy(),yd[keys].astype(str).to_numpy())
  raw=yd.Y_value.to_numpy(float);fit=tr[rows]&np.isfinite(raw)
  conv=PhenotypeOutliers().fit(raw[fit],rows[fit].tolist(),endpoint=endpoint)
  allraw=np.full(n,np.nan);allraw[rows]=raw;r=conv.transform(allraw,endpoint=endpoint)
  transforms[endpoint]=conv.state
  units=yd.Y_unit.dropna().unique().tolist();assert len(units)==1
  frame=X[keys+['input_row','split_unit_id','cv_fold']].copy()
  frame['Y_original']=r['original'];frame['Y_processed']=r['values'];frame['Y_unit']=units[0]
  for flag in ['nonfinite','excluded_8sd','winsorized_low','winsorized_high']:frame[flag]=r[flag]
  frame['is_training_role']=tr
  for exposure,days,energy,prefix in [('fiber_density','n_fiber_days','mean_daily_energy_fiber','fiber'),('UPF','n_energy_days','mean_daily_energy_UPF','UPF')]:
   eligible=np.asarray(X[days].ge(3)&a['supported']&np.isfinite(X.age_calendar_years)&X.sex_code.isin([0,1])&np.isfinite(X[energy])&np.isfinite(a[prefix+'_u'])&np.isfinite(a[prefix+'_h'])&np.isfinite(r['values']))
   frame['eligible_'+exposure]=eligible
   if key=='full':
    t=X.loc[eligible].copy()
    for col in ['Y_original','Y_processed','Y_unit','nonfinite','excluded_8sd','winsorized_low','winsorized_high']:t[col]=frame.loc[eligible,col].to_numpy()
    t['diet_u']=a[prefix+'_u'][eligible];t['diet_h']=a[prefix+'_h'][eligible]
    t['analysis_id']=endpoint;t['exposure_id']=exposure
    assert not any('score' in c.lower() or 'predict' in c.lower() for c in t.columns)
    save(t,ready/endpoint/(exposure+'_table_TRE_ONLY.parquet'))
    counts.append(dict(analysis_id=endpoint,exposure_id=exposure,people=len(t),families=int(t.split_unit_id.nunique()),Y_unit=units[0]))
   refs.append(dict(preprocessing=key,analysis_id=endpoint,exposure_id=exposure,eligible=int(eligible.sum()),training=int((tr&eligible).sum()),validation=int((~tr&eligible).sum())))
  save(frame,D/(endpoint+'_processed_Y_TRE_ONLY.parquet'))
 js(D/'phenotype_transforms.json',transforms)
 print('PROCESSED',key,'2 microbiome views / 9 outcome views',flush=True)
assert len(counts)==18
glu=[r for r in counts if r['analysis_id']=='D_Glu' and r['exposure_id']=='fiber_density'];assert glu[0]['people']==3343
pd.DataFrame(counts).to_csv(O/'manifests/analysis_ready_counts.csv',index=False)
js(O/'manifests/processed_schema.json',schema);js(O/'manifests/preprocessed_flow.json',refs)
docs=json.loads((L/'processed_authored_files.json').read_text())
for rel,body in docs.items():
 p=O/rel
 with p.open('x') as f:f.write(body)
js(O/'manifests/processed_views_status.json',dict(status='PASS',predictive_model_fitted=False,full_ready_tables=18,
 fold_outcome_tables=54,microbiome_parquet_views=12,pid=os.getpid(),python=sys.executable))
payload=[]
for p in sorted(O.rglob('*')):
 if p.is_file():payload.append(dict(relative_path=str(p.relative_to(O)),bytes=p.stat().st_size,sha256=sha(p)))
js(O/'manifests/payload_sha256_final.json',payload)
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
rc=subprocess.run([sys.executable,str(O/'examples/check_complete_package.py')],capture_output=True,text=True,env=env)
assert rc.returncode==0,rc.stderr
print('FINAL_HASH_CHECK',rc.stdout,flush=True)
print('READY_COUNTS',json.dumps(counts),flush=True)
print('PROCESSED_DATA_COMPLETE 18 full analysis tables / 54 fold outcome tables / 12 microbiome parquet views; NO PREDICTIVE MODELS',flush=True)
