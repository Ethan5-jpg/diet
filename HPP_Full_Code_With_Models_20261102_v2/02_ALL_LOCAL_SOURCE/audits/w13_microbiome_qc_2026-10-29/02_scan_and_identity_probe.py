# -*- coding: utf-8 -*-
"""Complete bounded species scans + source technical metadata and specimen-pointer diagnostics."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
from collections import Counter
import json,hashlib,resource,shutil,time
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');R=Path('/home/ec2-user/studies/hpp_datasets');L=Path('/home/ec2-user/multisystem_runtime_20261021/w13_microbiome_qc_20261029');Q=P/'data_qc/w13_microbiome_qc_20261029';K=['participant_id','cohort','research_stage']
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(300,300));assert (L/'01_actual_exit_code').read_text().strip()=='0'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):
 assert not p.exists();p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n')
def guard():
 mem=int(dict(x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines())['MemAvailable'].split()[0]);assert mem>2*1024**2;assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1200*1024
M=json.loads((Q/'source_metadata_TRE_ONLY.json').read_text());active=json.loads((P/'config/ACTIVE_PROTOCOL.json').read_text());sync=json.loads((Q/'protocol_sync_receipt.json').read_text());assert sync['passed'] and active['version']=='multisystem_v3_user_amendment_20261029'
for x in sync['source_receipts']:assert sha(x['path'])==x['sha256']
print('ACTIVE_PROTOCOL_VERIFIED',active['version'],'source_files',len(sync['source_receipts']),flush=True)
candidates=pq.read_table(P/'analysis/shared/sample_candidates_TRE_ONLY.parquet').to_pandas(ignore_metadata=True);assert len(candidates)==20547 and not candidates.selected_primary_sample.any();allqc=[];summaries={};probes={};rowfiles=[]
for site in ['oral','gut']:
 role=site+'_species';meta=M[role];src=Path(meta['path']);tax=meta['taxon_columns'];assert len(tax)==len(set(tax)) and src.stat().st_size==meta['bytes'] and src.stat().st_mtime_ns==meta['mtime_ns'];pf=pq.ParquetFile(src,pre_buffer=False);assert meta['decoded_bytes_all_columns']<512*1024**2
 sourcehash=sha(src);old=json.loads((P/'config/sample_selection_rules.yaml').read_text())['source_versions_fixed'][role];assert old['sha256']==sourcehash
 frequency=np.zeros(len(tax),dtype='int64');rows=[];n=0
 for b in pf.iter_batches(columns=tax,batch_size=128,use_threads=False):
  guard();a=np.column_stack([b.column(i).to_numpy(zero_copy_only=False) for i in range(b.num_columns)]).astype('<f8');finite=np.isfinite(a);negative=finite&(a<0);positive=finite&(a>0);frequency+=positive.sum(axis=0);sums=np.where(finite,a,0).sum(axis=1)
  for i in range(len(a)):
   x=a[i].copy();x[x==0]=0.;rows.append({'site':site,'abundance_row_ordinal':n+i+1,'taxon_count':len(tax),'nonfinite_values':int((~finite[i]).sum()),'negative_values':int(negative[i].sum()),'detected_taxa':int(positive[i].sum()),'finite_abundance_sum':float(sums[i]),'maximum_finite_value':float(np.max(x[finite[i]])) if finite[i].any() else None,'minimum_finite_value':float(np.min(x[finite[i]])) if finite[i].any() else None,'profile_sha256_float64':hashlib.sha256(x.tobytes()).hexdigest(),'source_sha256':sourcehash})
  n+=len(a)
 assert n==meta['rows'];d=pd.DataFrame(rows);allqc.extend(rows);summary={'rows_scanned':n,'taxon_columns':len(tax),'nonfinite_rows':int(d.nonfinite_values.gt(0).sum()),'negative_rows':int(d.negative_values.gt(0).sum()),'zero_sum_rows':int(d.finite_abundance_sum.eq(0).sum()),'sum_quantiles':{str(q):float(d.finite_abundance_sum.quantile(q)) for q in [0,.01,.5,.99,1]},'sum_near_one_rows':int(np.isclose(d.finite_abundance_sum,1,atol=1e-6,rtol=0).sum()),'sum_near_100_rows':int(np.isclose(d.finite_abundance_sum,100,atol=1e-5,rtol=0).sum()),'global_max':float(d.maximum_finite_value.max()),'zero_prevalence_taxa':int((frequency==0).sum()),'identical_profile_groups':int(d.profile_sha256_float64.value_counts().gt(1).sum()),'no_prevalence_filter_or_transform_fitted':True};summaries[site]=summary;print('MATRIX_SCAN',site,json.dumps(summary),flush=True)
 dump(Q/(site+'_taxon_inventory_TRE_ONLY.json'),[{'source_column':t,'positive_rows':int(c)} for t,c in zip(tax,frequency)])
 # Exact available metadata, never fuzzy matching taxon strings.
 index=Path(M[site+'_index']['path']);schema=pq.ParquetFile(index).schema_arrow.names;fields=[x for x in K+['array_index','sample_name','collection_date','raw_read_count','trimmed_read_count','duplicate_reads_count','excessive_duplicate_reads','metaphlan4_results_txt','metaphlan_abundance_species_parquet','run_name'] if x in schema];ir=pq.ParquetFile(index).read(columns=fields,use_threads=False).to_pandas(ignore_metadata=True);pr={'rows':len(ir),'numeric_metadata':{},'pointers':{}}
 for col in ['raw_read_count','trimmed_read_count','duplicate_reads_count']:
  if col in ir:
   v=pd.to_numeric(ir[col],errors='coerce');pr['numeric_metadata'][col]={'nonmissing':int(v.notna().sum()),'negative':int(v.lt(0).sum()),'zero':int(v.eq(0).sum()),'min':float(v.min()) if v.notna().any() else None,'max':float(v.max()) if v.notna().any() else None}
 pr['excessive_duplicate_reads_counts']=ir['excessive_duplicate_reads'].astype(str).value_counts().astype(int).to_dict() if 'excessive_duplicate_reads' in ir else None
 if all(c in ir for c in ['raw_read_count','trimmed_read_count']):pr['trimmed_gt_raw_rows']=int((ir.trimmed_read_count>ir.raw_read_count).sum())
 def resolve(v):
  if not isinstance(v,str) or v.strip().lower() in ['','undefined','none','nan']:return None
  q=Path(v)
  options=[q] if q.is_absolute() else [R/q,R.parent/q]
  return next((x for x in options if x.is_file()),None)
 for col in ['metaphlan4_results_txt','metaphlan_abundance_species_parquet']:
  if col not in ir:continue
  values=[v for v in ir[col] if isinstance(v,str) and v.strip().lower() not in ['','undefined','none','nan']];uniq=list(dict.fromkeys(values));sample=[]
  for v in uniq[:3]:
   p=resolve(v);z={'raw_pointer':v,'resolved_path':str(p) if p else None,'exists':p is not None}
   if p:
    z['bytes']=p.stat().st_size
    if col.endswith('parquet'):
     pf2=pq.ParquetFile(p,pre_buffer=False);z['rows']=pf2.metadata.num_rows;z['columns']=len(pf2.schema_arrow.names);z['schema']=[{'name':f.name,'type':str(f.type)} for f in pf2.schema_arrow]
    elif p.stat().st_size<4*1024**2:
     lines=p.read_text(errors='replace').splitlines();z['first_header_lines']=[x for x in lines[:15] if x.startswith('#')];z['first_noncomment_column_count']=next((len(x.split('\t')) for x in lines if x and not x.startswith('#')),None)
   sample.append(z)
  pr['pointers'][col]={'nonmissing_rows':len(values),'unique_pointer_values':len(uniq),'probe':sample};print('POINTER_PROBE',site,col,'nonmissing',len(values),'unique',len(uniq),'first3_exist',[x['exists'] for x in sample],'first3_shape',[(x.get('rows'),x.get('columns'),x.get('first_noncomment_column_count')) for x in sample],flush=True)
 probes[site]=pr;print('SOURCE_TECH_METADATA',site,json.dumps({k:v for k,v in pr.items() if k!='pointers'}),flush=True)
 # Retain exact pointers to investigate held groups without exporting IDs.
 pp=ir.copy();pp['index_row_ordinal']=np.arange(1,len(pp)+1);runtime=L/(site+'_index_projection_TRE_ONLY.parquet');target=Q/runtime.name;assert not runtime.exists() and not target.exists();pq.write_table(pa.Table.from_pandas(pp,preserve_index=False),runtime,compression='zstd');shutil.copyfile(runtime,target);assert sha(runtime)==sha(target)
qdf=pd.DataFrame(allqc);runtime=L/'profile_numeric_qc_TRE_ONLY.parquet';target=Q/runtime.name;assert not runtime.exists() and not target.exists();pq.write_table(pa.Table.from_pandas(qdf,preserve_index=False),runtime,compression='zstd');shutil.copyfile(runtime,target);assert sha(runtime)==sha(target)
dump(Q/'matrix_scan_summary.json',summaries);dump(Q/'technical_pointer_probes_TRE_ONLY.json',probes);print('W13_COMPLETE_MATRIX_SCAN',len(qdf),'peak_RSS_KiB',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
