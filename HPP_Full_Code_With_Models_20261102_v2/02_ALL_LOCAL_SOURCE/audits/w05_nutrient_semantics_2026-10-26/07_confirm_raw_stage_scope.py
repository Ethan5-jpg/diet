"""Bounded complete raw cohort/stage projection; no identity values read."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,resource,time,hashlib,sys
from pathlib import Path
from collections import Counter
resource.setrlimit(resource.RLIMIT_AS,(1536*1024**2,1536*1024**2))
import pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
x=json.loads((P/'W05_LATEST_RUN.json').read_text());assert x['status']=='completed';q=Path(x['result'])
p=Path('/home/ec2-user/studies/hpp_datasets/diet_logging/raw_diet_logging_events.parquet');st=p.stat();pf=pq.ParquetFile(p,pre_buffer=False);cols=['cohort','research_stage'];counts=Counter();n=0;t=time.monotonic()
for b in pf.iter_batches(batch_size=8192,columns=cols,use_threads=False):
    d=b.to_pydict();counts.update(zip(d['cohort'],d['research_stage']));n+=b.num_rows
    assert time.monotonic()-t<180,'Scope scan timeout'
assert n==pf.metadata.num_rows and p.stat().st_size==st.st_size and p.stat().st_mtime_ns==st.st_mtime_ns
r={'run':x['run'],'rows_scanned':n,'full_scan':True,'columns':cols,'cohort_stage_row_counts':[{'cohort':c,'research_stage':s,'rows':v} for (c,s),v in sorted(counts.items(),key=lambda kv:str(kv[0]))],'caveat':'Raw literal baseline filter is not a cohort-admission result. Do not assign undefined to baseline. W06 must establish raw edit/identity lineage before baseline linkage.','peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
(q/'raw_stage_scope.json').write_text(json.dumps(r,indent=2)+'\n')
code=P/'scripts'/x['run']/'07_confirm_raw_stage_scope.py';code.write_bytes(Path(__file__).read_bytes())
memo='''# W05 addendum: raw event stage scope\n\n07_confirm_raw_stage_scope.py fully scans only cohort and research_stage, no participant IDs. It explains why the raw literal baseline subset in 05 is empty; this is not zero baseline people. Full counts: data_qc/RUN/raw_stage_scope.json. Do not manufacture baseline visits. W06 must establish lineage before using raw event rows for baseline QC.\n'''
(P/'memos'/x['run']/'RAW_SCOPE_MEMO.md').write_text(memo)
print('\n'*30+'W05_RAW_SCOPE_PASS',json.dumps(r,separators=(',',':')))
print('RAW_SCOPE_CODE',code);print('RAW_SCOPE_END')
