"""Create and verify a discovery-only collaborator snapshot within the TRE."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys, json, hashlib, shutil, subprocess, datetime, io
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.dataset as ds

P = Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
B = P / 'data_qc/w20_simplified_20261101_v2'
V = P / 'data_qc/w20_targeted_20261101_v3/results/attempt49b'
O = P / 'collaborator_handoff_20261102_v1'
L = Path('/home/ec2-user/multisystem_runtime_20261021/collaborator_handoff_20261102_v1')
assert sys.executable == '/home/ec2-user/miniconda/envs/pheno/bin/python'
assert not O.exists()
docs = json.loads((L / 'authored_files.json').read_text())
sha = lambda p: file_sha(Path(p))
def file_sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

plan = []
def include(src, dst):
    assert src.is_file(), str(src)
    plan.append((src, Path(dst)))
names = ['discovery_covariates_food_TRE_ONLY.parquet','discovery_roles_folds_TRE_ONLY.parquet',
    'roles_v2_full_TRE_ONLY.parquet','oral_raw_TRE_ONLY.npz','gut_raw_TRE_ONLY.npz',
    'oral_taxa.json','gut_taxa.json','reference_food_labels_TRE_ONLY.parquet',
    'reference_food_presence_TRE_ONLY.parquet','unknown_food_categories_TRE_ONLY.parquet',
    'unknown_weights_TRE_ONLY.npz','input_summary.json','role_summary.json']
names += sorted(p.name for p in B.glob('D_*_raw_Y_TRE_ONLY.parquet'))
assert len([n for n in names if '_raw_Y_' in n]) == 9
for name in names:
    include(B/name, Path('inputs/discovery_v2')/name)
for p in sorted((B / 'results/fold_preprocessing').rglob('*')):
    if p.is_file() and p.name in ['diet_basis.json','oral_transform.json','gut_transform.json','transformed_discovery_TRE_ONLY.npz']:
        include(p, Path('preprocessing/folds') / p.relative_to(B / 'results/fold_preprocessing'))
include(P/'scripts/preprocessing_20261030_v1/preprocessing_v1_20261030.py',Path('preprocessing/code/preprocessing_v1_20261030.py'))
include(P/'data_qc/preprocessing_20261030_v1/food_qc_v1/rules.json',Path('preprocessing/food_rules.json'))
assert len(set(str(dst) for _,dst in plan)) == len(plan)
total = sum(src.stat().st_size for src,_ in plan)
assert total < 2*1024**3, 'Unexpectedly large snapshot; inspect before copying'
assert shutil.disk_usage(P).free > total*2 + 128*1024**2
print('PLAN', json.dumps(dict(files=len(plan), bytes=total, output=str(O))), flush=True)

O.mkdir(exist_ok=False)
for d in ['manifests','experiments','examples']:
    (O / d).mkdir()
copies = []
for src,dst in plan:
    dest = O / dst
    dest.parent.mkdir(parents=True, exist_ok=True)
    source_sha = sha(src)
    with src.open('rb') as fi, dest.open('xb') as fo:
        shutil.copyfileobj(fi,fo,1024*1024)
    assert source_sha == sha(dest) == sha(src), str(src)
    copies.append(dict(source=str(src), relative_path=str(dst), bytes=dest.stat().st_size, sha256=source_sha))
for rel, record in docs.items():
    content = record['body'].encode('utf-8')
    assert hashlib.sha256(content).hexdigest() == record['sha256']
    dest = O / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('xb') as f: f.write(content)
    assert sha(dest) == record['sha256']

def write_json(rel,obj):
    with (O / rel).open('x') as f:
        f.write(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
write_json('manifests/source_copies.json', copies)
X=pq.ParquetFile(O/'inputs/discovery_v2/discovery_covariates_food_TRE_ONLY.parquet',pre_buffer=False).read(
    columns=['participant_id','cohort','research_stage','final_role']).to_pandas(ignore_metadata=True)
assert len(X)==4427 and X.final_role.eq('discovery').all()
filt=(ds.field('cohort')=='10k')&(ds.field('research_stage')=='00_00_visit')&ds.field('participant_id').isin(X.participant_id.tolist())
K=['participant_id','cohort','research_stage'];KD=K+['date']
derived=[]
def project_daily(src,cols,rel,filtered=True):
    source_sha=sha(src)
    names=pq.ParquetFile(src,pre_buffer=False).schema_arrow.names
    required=KD+['Y_value'] if 'cgm_days' in rel else KD+['energy_kcal','fiber_g'] if 'food_days' in rel else KD+['Y_value','person_weight']
    assert set(required)<=set(names)
    cols=[c for c in cols if c in names]
    t=ds.dataset(src,format='parquet').scanner(columns=cols,filter=filt if filtered else None,
        use_threads=False,batch_size=1024).to_table()
    d=t.to_pandas(ignore_metadata=True)
    assert set(d.participant_id)<=set(X.participant_id) and not d.duplicated(KD).any()
    assert not any('score' in c.lower() or 'predict' in c.lower() for c in d.columns)
    dest=O/rel;dest.parent.mkdir(parents=True,exist_ok=True)
    pq.write_table(pa.Table.from_pandas(d,preserve_index=False),dest,compression='zstd')
    assert pq.ParquetFile(dest,pre_buffer=False).metadata.num_rows==len(d)
    assert source_sha==sha(src)
    derived.append(dict(source=str(src),source_sha256=source_sha,relative_path=rel,
        sha256=sha(dest),bytes=dest.stat().st_size,columns=cols,rows=len(d),
        filter='v2 discovery IDs,10k,00_00_visit before materialization' if filtered else 'already discovery-only; explicit no-score columns'))
    return d
project_daily(P/'analysis/Daily_CGM/fiber_density/base_table_TRE_ONLY.parquet',KD+['Y_value','Y_unit','base_eligible'],
    'inputs/daily/cgm_days_discovery_TRE_ONLY.parquet')
project_daily(P/'data_qc/preprocessing_20261030_v1/food_qc_v1/food_daily_primary_TRE_ONLY.parquet',
    KD+['energy_kcal','fiber_g','fiber_density_g_per_1000kcal','energy_eligible','fiber_eligible'],
    'inputs/daily/food_days_discovery_TRE_ONLY.parquet')
daily=project_daily(V/'daily_analysis_TRE_ONLY.parquet',KD+['Y_value','Y_unit','energy_kcal','fiber_g',
    'fiber_density_g_per_1000kcal','energy_eligible','fiber_eligible','split_unit_id','cv_fold',
    'age_calendar_years','sex_code','n_i','person_weight','recording_day_index','weekday'],
    'inputs/daily/common_days_no_scores_TRE_ONLY.parquet',False)
write_json('manifests/derived_daily_sources.json',derived)
schema = []
for entry in copies+derived:
    p=O / entry['relative_path']
    if p.suffix == '.parquet':
        pf=pq.ParquetFile(p,pre_buffer=False)
        schema.append(dict(path=entry['relative_path'], rows=pf.metadata.num_rows,
            schema=[dict(name=f.name,type=str(f.type)) for f in pf.schema_arrow]))
write_json('manifests/parquet_schemas.json',schema)
import pandas, scipy, pyarrow
write_json('manifests/environment.json',dict(python=sys.executable, version=sys.version,
    numpy=np.__version__, pandas=pandas.__version__, scipy=scipy.__version__, pyarrow=pyarrow.__version__,
    pid=os.getpid(), created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    scope='discovery-only data snapshot; confirmation role metadata only; no confirmation Y',
    source_project=str(P), source_bytes=total, copy_files=len(copies)))

env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
env['PYTHONPATH']=str(O/'preprocessing/code')
check=subprocess.run([sys.executable,str(O/'examples/load_fiber_fold.py')],capture_output=True,text=True,env=env)
with (O/'manifests/fold_check_stdout.txt').open('x') as f:f.write(check.stdout)
with (O/'manifests/fold_check_stderr.txt').open('x') as f:f.write(check.stderr)
assert check.returncode == 0, check.stderr
fc=json.loads(check.stdout);assert fc['status']=='PASS' and fc['total_CV_n']==3343
assert len(daily)==21279 and daily.participant_id.nunique()==3028 and daily.split_unit_id.nunique()==2782
assert not daily.duplicated(['participant_id','cohort','research_stage','date']).any()
assert daily.n_i.ge(3).all()
np.testing.assert_allclose(daily.groupby('participant_id').person_weight.sum(),1,rtol=0,atol=1e-12)
write_json('manifests/data_checks.json',dict(status='PASS',fiber_Glu=fc,daily=dict(days=len(daily),
    people=int(daily.participant_id.nunique()),families=int(daily.split_unit_id.nunique()),person_weight_sum=1)))
assert all(sha(Path(i['source']))==i['sha256'] for i in copies)
assert all(sha(Path(i['source']))==i['source_sha256'] for i in derived)
payload=[]
for p in sorted(O.rglob('*')):
    if p.is_file():payload.append(dict(relative_path=str(p.relative_to(O)),bytes=p.stat().st_size,sha256=sha(p)))
write_json('manifests/payload_sha256.json',payload)
pc=subprocess.run([sys.executable,str(O/'examples/check_package.py')],capture_output=True,text=True,env=env)
assert pc.returncode==0,pc.stderr
print('FOLD_CHECK',json.dumps(fc),flush=True)
print('PACKAGE_CHECK',pc.stdout.strip(),flush=True)
print('DAILY_CHECK 3028 people / 2782 households / 21279 days; weights PASS',flush=True)
print('SOURCE_HASHES_UNCHANGED',len(copies),flush=True)
print('HANDOFF_COMPLETE',str(O),flush=True)
