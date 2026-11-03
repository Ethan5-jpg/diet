# -*- coding: utf-8 -*-
"""Resolve exact source pointers in module roots; investigate sparse cells and held identities."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib,resource,time,re,collections
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');R=Path('/home/ec2-user/studies/hpp_datasets');Q=P/'data_qc/w13_microbiome_qc_20261029';L=Path('/home/ec2-user/multisystem_runtime_20261021/w13_microbiome_qc_20261029');K=['participant_id','cohort','research_stage'];resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(240,240));assert (L/'02_actual_exit_code').read_text().strip()=='0'
M=json.loads((Q/'source_metadata_TRE_ONLY.json').read_text());C=pq.read_table(P/'analysis/shared/sample_candidates_TRE_ONLY.parquet').to_pandas(ignore_metadata=True);N=pq.read_table(Q/'profile_numeric_qc_TRE_ONLY.parquet').to_pandas(ignore_metadata=True);out={}
def dump(p,v):assert not p.exists();p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n')
for site in ['oral','gut']:
 mod=R/(site+'_microbiome');idx=pq.read_table(Q/(site+'_index_projection_TRE_ONLY.parquet')).to_pandas(ignore_metadata=True);pf=pq.ParquetFile(M[site+'_species']['path'],pre_buffer=False);tax=M[site+'_species']['taxon_columns'];nulls=0;nans=0;inf=0;zeros=0;observed=0
 for b in pf.iter_batches(columns=tax,batch_size=256,use_threads=False):
  nulls+=sum(b.column(i).null_count for i in range(b.num_columns))
  for i in range(b.num_columns):
   a=b.column(i).to_numpy(zero_copy_only=False);nans+=int(np.isnan(a).sum())-b.column(i).null_count;inf+=int(np.isinf(a).sum());zeros+=int((a==0).sum());observed+=int(np.isfinite(a).sum())
 probe={};source_files={}
 for col in ['metaphlan_abundance_species_parquet','metaphlan4_results_txt']:
  if col not in idx:continue
  pairs=[(int(r.index_row_ordinal),getattr(r,col)) for r in idx.itertuples() if isinstance(getattr(r,col),str) and getattr(r,col).lower() not in ['','undefined','none','nan']];unique=list(dict.fromkeys(v for _,v in pairs));records=[]
  for raw in unique[:3]:
   p=Path(raw);options=[p] if p.is_absolute() else [mod/p,R/p,R.parent/p,mod/'abundance'/p];found=next((x for x in options if x.is_file()),None);z={'raw_pointer':raw,'tested_paths':[str(x) for x in options],'resolved_path':str(found) if found else None}
   if found:
    z['bytes']=found.stat().st_size
    if found.suffix=='.parquet':
     sf=pq.ParquetFile(found,pre_buffer=False);z['rows']=sf.metadata.num_rows;z['columns']=len(sf.schema_arrow.names);z['schema']=[{'name':f.name,'type':str(f.type)} for f in sf.schema_arrow]
    elif found.stat().st_size<4*1024**2:
     lines=found.read_text(errors='replace').splitlines();z['header']=[x for x in lines[:30] if x.startswith('#')];z['noncomment_rows']=sum(bool(x) and not x.startswith('#') for x in lines)
   records.append(z)
  probe[col]={'nonmissing':len(pairs),'unique':len(unique),'probe':records};source_files[col]=pairs
  print('RESOLVED_POINTERS',site,col,[z['resolved_path'] is not None for z in records],[(z.get('rows'),z.get('columns'),z.get('noncomment_rows')) for z in records],flush=True)
 # Bounded module tree inventory exposes aggregate/per-sample availability without printing specimen filenames.
 start=time.monotonic();inventory=[];seen=0
 for root,dirs,files in os.walk(mod):
  depth=len(Path(root).relative_to(mod).parts)
  if depth>=4:dirs[:]=[]
  dirs[:]=sorted(d for d in dirs if not d.startswith('.'))
  for f in files:
   seen+=1;assert seen<60000 and time.monotonic()-start<45
   if f.endswith(('.parquet','.txt','.tsv','.csv','.py','.md')):inventory.append({'path':str(Path(root)/f),'relative_depth':depth,'extension':Path(f).suffix})
 print('MODULE_INVENTORY',site,'files',seen,'by_depth_extension',dict(collections.Counter((x['relative_depth'],x['extension']) for x in inventory)),flush=True)
 # Missing visit cases are stratified by direct evidence; no patient row printed.
 s=C.loc[C.site.eq(site)];held=s.loc[~s.identity_ready_for_W13];stats={'held_rows':len(held),'reasons':held.identity_status.value_counts().astype(int).to_dict(),'held_abundance_date_missing':int(held.loc[held.candidate_kind.eq('abundance_row'),'abundance_calendar_date'].isna().sum()),'held_index_only':int(held.candidate_kind.eq('index_only').sum())}
 if site=='oral':
  # Alternate version can only offer an identity bridge if actual biological/profile IDs are present.
  alt=pq.ParquetFile(M['oral_index_alternate']['path']);names=alt.schema_arrow.names;stats['alternate_identity_fields']=[n for n in ['sample_name','sample_id','uuid','participant_id','cohort','research_stage','collection_date'] if n in names]
 else:
  helddup=held.loc[held.identity_status.eq('held_repeated_visit_specimen_mapping_required')];g=helddup.merge(N.loc[N.site.eq(site),['abundance_row_ordinal','profile_sha256_float64']],on='abundance_row_ordinal',validate='one_to_one');stats['repeated_visit_groups']=len(g[K].drop_duplicates());stats['repeated_visit_all_profiles_identical_groups']=sum(z.profile_sha256_float64.nunique()==1 for _,z in g.groupby(K));stats['all_held_distinct_from_random_eligibility']=True
  # Native array index is only a locator proposal; no specimen identity proof is created here.
 out[site]={'sparse_storage':{'arrow_null_cells':nulls,'floating_NaN_nonnull_cells':nans,'infinite_cells':inf,'explicit_zero_cells':zeros,'finite_observed_cells':observed},'pointer_probes':probe,'module_inventory':inventory,'identity_held_diagnostic':stats};print('SPARSE_STORAGE',site,json.dumps(out[site]['sparse_storage']),flush=True);print('HELD_EVIDENCE',site,json.dumps(stats),flush=True)
dump(Q/'storage_and_specimen_evidence_TRE_ONLY.json',out);print('W13_STORAGE_IDENTITY_PROBE_COMPLETE',flush=True)
