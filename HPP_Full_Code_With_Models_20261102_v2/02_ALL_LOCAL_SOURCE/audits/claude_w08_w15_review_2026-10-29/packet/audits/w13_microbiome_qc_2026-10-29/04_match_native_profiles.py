# -*- coding: utf-8 -*-
"""Validate source implicit-zero encoding and recover only unique full-vector native-profile links."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
from collections import defaultdict,Counter
import json,hashlib,resource,shutil
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');R=Path('/home/ec2-user/studies/hpp_datasets');Q=P/'data_qc/w13_microbiome_qc_20261029';L=Path('/home/ec2-user/multisystem_runtime_20261021/w13_microbiome_qc_20261029');K=['participant_id','cohort','research_stage'];resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(300,300));assert (L/'03b_actual_exit_code').read_text().strip()=='0'
M=json.loads((Q/'source_metadata_TRE_ONLY.json').read_text());C=pq.read_table(P/'analysis/shared/sample_candidates_TRE_ONLY.parquet').to_pandas(ignore_metadata=True);idx=pq.read_table(Q/'gut_index_projection_TRE_ONLY.parquet').to_pandas(ignore_metadata=True);probe=json.loads((Q/'storage_and_specimen_evidence_v2_TRE_ONLY.json').read_text());tax=M['gut_species']['taxon_columns'];labels=[n.split('|')[-1] for n in tax];assert len(labels)==len(set(labels));im={int(r.index_row_ordinal):r._asdict() for r in idx.itertuples(index=False)};bykey=defaultdict(list)
for i,r in im.items():bykey[tuple(r[k] for k in K)].append(i)
held=C.loc[C.site.eq('gut')&C.identity_status.eq('held_repeated_visit_specimen_mapping_required')];keys=set(tuple(r[k] for k in K) for r in held.to_dict('records'));first=C.loc[C.site.eq('gut')&C.identity_ready_for_W13].sort_values('abundance_row_ordinal').head(3);targets=set(held.abundance_row_ordinal.astype(int))|set(first.abundance_row_ordinal.astype(int));vectors={};ordinal=0
for b in pq.ParquetFile(M['gut_species']['path'],pre_buffer=False).iter_batches(columns=tax,batch_size=128,use_threads=False):
 chosen=[i for i in range(len(b)) if ordinal+i+1 in targets]
 if chosen:
  a=np.column_stack([b.column(i).to_numpy(zero_copy_only=False) for i in range(b.num_columns)]).astype(float)
  for i in chosen:vectors[ordinal+i+1]=a[i]
 ordinal+=len(b)
assert set(vectors)==targets
parsed={};filechecks=[]
def native(i):
 if i in parsed:return parsed[i]
 r=im[i];raw=r.get('metaphlan4_results_txt');assert isinstance(raw,str);q=Path(raw);options=[q] if q.is_absolute() else [R/'gut_microbiome'/q,R/q,R.parent/q];p=next((x for x in options if x.is_file()),None)
 if p is None:parsed[i]=None;return None
 assert p.stat().st_size<4*1024**2
 text=p.read_text();lines=text.splitlines();header=next((x.lstrip('#').strip().split('\t') for x in lines if x.startswith('#clade_name')),None);assert header and 'relative_abundance' in header and 'clade_name' in header;vi=header.index('relative_abundance');ni=header.index('clade_name');values={}
 for line in lines:
  if not line or line.startswith('#'):continue
  a=line.split('\t');assert len(a)>max(ni,vi);name=a[ni].split('|')[-1]
  if not name.startswith('s__'):continue
  assert name not in values;v=float(a[vi]);assert np.isfinite(v) and v>=0;values[name]=v
 vec=np.array([values.get(n,0.) for n in labels]);extra={n:v for n,v in values.items() if n not in labels and v>0};e={'index_row_ordinal':i,'source_path':str(p),'source_sha256':hashlib.sha256(text.encode()).hexdigest(),'source_species':len(values),'positive_species_outside_aggregate':len(extra),'source_sum':sum(values.values()),'text_header':header,'input_reads_processed':next((x for x in lines[:12] if 'reads processed' in x),None)};filechecks.append(e);parsed[i]=(vec,values,e);return parsed[i]
# Fixed first three 1:1 profiles selected before any values; test every observed AND implicit cell.
checks=[]
for r in first.itertuples():
 i=int(r.linked_index_row_ordinal);n=native(i);assert n is not None;v,values,e=n;a=vectors[int(r.abundance_row_ordinal)];observed=np.isfinite(a);expected=np.nan_to_num(a,nan=0.);error=float(np.max(np.abs(expected-v)));ok=not e['positive_species_outside_aggregate'] and np.allclose(expected,v,rtol=0,atol=1e-8) and all(labels[j] not in values or values[labels[j]]==0 for j in np.flatnonzero(~observed));checks.append({'abundance_row_ordinal':int(r.abundance_row_ordinal),'index_row_ordinal':i,'all_species_and_implicit_cells_match':bool(ok),'implicit_cells':int((~observed).sum()),'max_abs_difference':error});assert ok
print('NATIVE_GUT_SPARSE_ENCODING_VERIFIED',len(checks),'all_species_per_profile',len(tax),'implicit_cells',sum(x['implicit_cells'] for x in checks),flush=True)
# A unique full composition within an already matching person/cohort/visit supports provenance;
# array_index equality alone is never used. Ambiguous matches remain held.
links=[]
for k in sorted(keys):
 profiles=held.loc[(held[K]==pd.Series(k,index=K)).all(axis=1)];options=bykey[k];matches={}
 for r in profiles.itertuples():
  a=np.nan_to_num(vectors[int(r.abundance_row_ordinal)],nan=0.);hit=[]
  for i in options:
   n=native(i)
   if n is not None and n[2]['positive_species_outside_aggregate']==0 and np.allclose(a,n[0],rtol=0,atol=1e-8):hit.append(i)
  matches[int(r.abundance_row_ordinal)]=hit
 unique=[h[0] for h in matches.values() if len(h)==1];bijective=len(unique)==len(matches) and len(set(unique))==len(unique)
 chosen=[im[i] for i in unique] if bijective else [];dates=[pd.Timestamp(x['collection_date']).date().isoformat() if pd.notna(x.get('collection_date')) else None for x in chosen];names=[x.get('sample_name') for x in chosen];distinct_dated_biological=bijective and len(chosen)>1 and None not in dates and len(set(dates))==len(dates) and len(set(names))==len(names)
 for a,h in matches.items():links.append({'site':'gut','abundance_row_ordinal':a,'matched_native_index_ordinals_json':json.dumps(h),'unique_native_profile_link':len(h)==1,'group_bijective_native_links':bijective,'group_distinct_source_sample_names_and_sampling_dates':distinct_dated_biological,'resolved_index_row_ordinal':h[0] if bijective else None,'biological_repeat_status':'distinct_source_sampling_dates_and_names' if distinct_dated_biological else 'technical_vs_biological_repeat_unresolved','approved_for_selection_recovery':distinct_dated_biological,'match_tolerance_abs':1e-8,'identity_evidence':'same person/cohort/stage plus unique full-species native source profile match; not association-based'})
D=pd.DataFrame(links);assert len(D)==len(held)==50;out=L/'native_specimen_links_TRE_ONLY.parquet';dest=Q/out.name;assert not out.exists() and not dest.exists();pq.write_table(pa.Table.from_pandas(D,preserve_index=False),out,compression='zstd');shutil.copyfile(out,dest);assert hashlib.sha256(out.read_bytes()).hexdigest()==hashlib.sha256(dest.read_bytes()).hexdigest()
result={'fixed_native_profiles_verified':len(checks),'fixed_profile_cell_checks':len(checks)*len(tax),'all_fixed_native_profiles_match':True,'gut_implicit_arrow_null_means_unreported_taxon_in_native_profile':True,'oral_direct_native_profile_validation_available':False,'oral_storage_evidence':probe['oral']['sparse_storage'],'oral_full_scan_sum_near_100_all_profiles':True,'oral_encoding_inference':'same HPP MetaPhlAn aggregation family, only positive observed values, exclusively Arrow nulls, all profiles close to100; explicit assumption if using implicit-zero view, original null mask retained','repeated_abundance_rows_investigated':len(D),'unique_native_profile_link_rows':int(D.unique_native_profile_link.sum()),'bijective_link_rows':int(D.group_bijective_native_links.sum()),'recoverable_distinct_dated_biological_rows':int(D.approved_for_selection_recovery.sum()),'remaining_repeated_rows_held':int((~D.approved_for_selection_recovery).sum()),'native_source_files_read':len(filechecks),'no_outcome_or_diet_used_for_matching':True}
for name,obj in [('native_sparse_encoding_checks_TRE_ONLY.json',checks),('native_profile_source_receipts_TRE_ONLY.json',filechecks),('native_evidence_summary.json',result)]:
 p=Q/name;assert not p.exists();p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n')
print('NATIVE_SPECIMEN_EVIDENCE',json.dumps(result),flush=True);print('W13_NATIVE_PROFILE_MATCH_COMPLETE',flush=True)
