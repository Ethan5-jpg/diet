# -*- coding: utf-8 -*-
"""Read-only independent verification of W06 artifacts, no outcomes or subject export."""
import os
for x in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:os.environ[x]='1'
import sys,json,time,hashlib,math,resource
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');L=Path('/home/ec2-user/multisystem_runtime_20261021/w06_food_cleaning_20261028');Q=P/'data_qc/w06_food_cleaning_20261028'
assert sys.executable=='/home/ec2-user/miniconda/envs/pheno/bin/python'
limit=time.monotonic()+1800
while not (L/'02_actual_exit_code').exists():
 assert time.monotonic()<limit,'W06 build not finished within verification wait'
 time.sleep(5)
assert (L/'02_actual_exit_code').read_text().strip()=='0','W06 build failed; inspect 02.log'
r=json.loads((Q/'w06_summary_TRE_ONLY.json').read_text());K=['participant_id','cohort','research_stage','date'];N=['calories_kcal','carbohydrate_g','lipid_g','protein_g','sodium_mg','alcohol_g','dietary_fiber_g']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
code=P/'scripts/w06_food_cleaning_20261028';assert sha(code/'02_build_food_views.py')=='5f759c399b03986111061675a9f70187a9eb7cde764555e40459982cc67ea098'
D=pq.ParquetFile(P/'analysis/shared/food_day_qc_TRE_ONLY.parquet').read().to_pandas();assert not D.duplicated(K).any()
counts=Counter();dispositions=Counter();n=0;flag_exact=0;flag_candidate=0
for rb in pq.ParquetFile(P/'analysis/shared/food_event_lineage_TRE_ONLY.parquet').iter_batches(batch_size=8192):
 d=rb.to_pandas();assert np.array_equal(d.source_row_ordinal.to_numpy(),np.arange(n,n+len(d)));n+=len(d);counts.update(d[K].itertuples(index=False,name=None));dispositions.update(d.qc_row_disposition);flag_exact+=int((d.qc_exact_signature_n>1).sum());flag_candidate+=int((d.qc_same_food_time_n>1).sum())
assert n==r['processed_rows']==2640959
assert dict(dispositions)==r['event_disposition']
# All dates are valid in the expected source. Explicitly stop if hidden NA-key canonicalization would be needed.
assert D[K].notna().all().all(),'Missing day key requires explicit independent reconciliation'
for row in D[K+['n_events']].itertuples(index=False,name=None):assert counts.get(row[:4],0)==row[4]
assert sum(counts.values())==int(D.n_events.sum())
assert flag_exact==r['signature_summary']['exact']['rows_in_repeated_groups']
assert flag_candidate==r['signature_summary']['candidate']['rows_in_repeated_groups']
cn=0;last=-1;sums=np.zeros(len(N));bad=np.zeros(len(N),np.int64);samples=[]
p=P/'analysis/shared/food_events_clean_TRE_ONLY.parquet';total=pq.ParquetFile(p).metadata.num_rows;targets=set(np.linspace(0,total-1,12).astype(int).tolist())
for rb in pq.ParquetFile(p).iter_batches(batch_size=8192):
 d=rb.to_pandas();v=d.source_row_ordinal.to_numpy();assert (np.diff(v)>0).all() and (len(v)==0 or v[0]>last);last=int(v[-1])
 assert d.qc_primary_eligible.all() and d.qc_index_matched.all() and (d.qc_row_disposition=='retained').all()
 for j,x in enumerate(N):
  c=d[x+'_clean'];orig=pd.to_numeric(d[x],errors='coerce');valid=np.isfinite(orig)&(orig>=0)
  assert c.notna().equals(valid);assert np.allclose(c[valid],orig[valid],rtol=0,atol=0);sums[j]+=float(c.sum());bad[j]+=int(c.isna().sum())
 for i in sorted(t for t in targets if cn<=t<cn+len(d)):
  row=d.iloc[i-cn];samples.append({'ordinal':int(row.source_row_ordinal),'values':{x:row[x] for x in ['participant_id','cohort','research_stage','collection_date','food_id']+N}})
 cn+=len(d)
assert cn==r['kept_event_rows']==int(D.loc[D.primary_day,'n_events'].sum())
for j,x in enumerate(N):
 z=D.loc[D.primary_day];assert math.isclose(sums[j],float(z[x+'_valid_partial_sum'].sum()),rel_tol=1e-10,abs_tol=1e-6);assert bad[j]==int(z[x+'_invalid_rows'].sum())
 expected=D.primary_day&(D.calories_kcal_invalid_rows==0)&(D.calories_kcal_complete_sum>0)
 assert D.energy_eligible.equals(expected)
assert D.fiber_eligible.equals(D.energy_eligible&(D.dietary_fiber_g_invalid_rows==0))
assert not D.meal_ingestion_time_validated.any() and not D.complete_daily_intake_established.any()
# Twelve retained records are independently re-read from exact original row-group positions.
src=pq.ParquetFile('/home/ec2-user/studies/hpp_datasets/diet_logging/diet_logging_events.parquet',pre_buffer=False)
off=0;checked=0
for g in range(src.num_row_groups):
 end=off+src.metadata.row_group(g).num_rows;wanted=[x for x in samples if off<=x['ordinal']<end]
 if wanted:
  d=src.read_row_group(g,columns=list(wanted[0]['values'])).to_pandas(ignore_metadata=True)
  for sample in wanted:
   row=d.iloc[sample['ordinal']-off]
   for key,v in sample['values'].items():
    actual=row[key];assert (pd.isna(v) and pd.isna(actual)) or actual==v,(key,sample['ordinal'])
   checked+=1
 off=end
assert checked==12
for m in r['mirrors']:
 a=Path(m['runtime']);b=Path(m['cloud']);assert a.stat().st_size==b.stat().st_size==m['bytes'];assert sha(a)==sha(b)==m['sha256']
 if a.suffix=='.parquet':assert pq.ParquetFile(a).metadata.num_rows==pq.ParquetFile(b).metadata.num_rows
for name,m in r['source_inputs'].items():
 p=Path('/home/ec2-user/studies/hpp_datasets/diet_logging')/name;st=p.stat();assert st.st_size==m['bytes'] and st.st_mtime_ns==m['mtime_ns'];assert pq.ParquetFile(p).metadata.num_rows==m['rows']
rules=json.loads((P/'config/food_day_rules.yaml').read_text());assert rules['energy_multiplier']==1 and rules['raw_stage_imputed'] is False and rules['automatic_event_deduplication'] is False and rules['global_minimum_days'] is None
out={'verified':True,'processed_rows':n,'clean_rows':cn,'day_rows':len(D),'source_samples_verified':checked,'lineage_day_counts_verified':True,'all_seven_nutrients_verified':True,'mirrors_verified':len(r['mirrors']),'source_metadata_unchanged':True,'code_sha256':{x.name:sha(x) for x in sorted(code.glob('*.py'))},'build_actual_exit':0,'outcomes_read':False,'human_team_review_done':False}
(Q/'verification.json').write_text(json.dumps(out,indent=2));assert json.loads((Q/'verification.json').read_text())==out
M=P/'memos/w06_food_cleaning_20261028/MEMO.md';M.write_text(M.read_text()+'\n## Independent verification\n\n'+json.dumps(out,indent=2)+'\n\nVerifier actual exit must be checked in 03_actual_exit_code; team human review is not claimed.\n')
print('W06_INDEPENDENT_VERIFICATION_PASS',json.dumps(out),flush=True)
