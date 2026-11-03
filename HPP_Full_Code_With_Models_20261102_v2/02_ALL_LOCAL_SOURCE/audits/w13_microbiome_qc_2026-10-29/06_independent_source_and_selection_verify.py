# -*- coding: utf-8 -*-
"""Independent full source-to-matrix, metadata, identity and random-winner verification."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,date
import json,hashlib,resource,csv,io
import numpy as np,pandas as pd,pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');R=Path('/home/ec2-user/studies/hpp_datasets');Q=P/'data_qc/w13_microbiome_qc_20261029';L=Path('/home/ec2-user/multisystem_runtime_20261021/w13_microbiome_qc_20261029');K=['participant_id','cohort','research_stage'];BAD={'','undefined','none','null','nan','na','n/a','unknown','<na>'};resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(360,360))
actual={n:int((L/(n+'_actual_exit_code')).read_text()) for n in ['01','02','03','03b','04','05']};assert all(actual[n]==0 for n in ['01','02','03b','04','05']) and actual['03']==1
M=json.loads((Q/'source_metadata_TRE_ONLY.json').read_text());F=pq.read_table(Q/'microbiome_sample_flags_provisional_TRE_ONLY.parquet').to_pandas(ignore_metadata=True);S=pq.read_table(Q/'sample_registry_provisional_TRE_ONLY.parquet').to_pandas(ignore_metadata=True);original=pq.read_table(P/'analysis/shared/sample_candidates_TRE_ONLY.parquet').to_pandas(ignore_metadata=True);links=pq.read_table(Q/'native_specimen_links_TRE_ONLY.parquet').to_pandas(ignore_metadata=True);rules=json.loads((Q/'w13_QC_selection_rules_provisional.json').read_text());assert rules['seed']==20261027 and len(F)==len(original)==20547 and not S.duplicated(K+['site']).any()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def valid(r):
 if any(r.get(k) is None or str(r[k]).lower() in BAD for k in K):return False
 try:return int(r['participant_id'])>0 and int(r['participant_id'])==float(r['participant_id'])
 except (ValueError,TypeError):return False
def ds(v):return pd.Timestamp(v).date().isoformat() if v is not None and not pd.isna(v) else None
def kv(r):return tuple(r[k] for k in K)
for x in json.loads((Q/'provisional_artifact_manifest.json').read_text()):
 assert sha(x['path'])==x['sha256']
 if 'runtime_source' in x:assert sha(x['runtime_source'])==x['sha256']
assert set(F.candidate_id)==set(original.candidate_id) and set(S.candidate_id)==set(F.loc[F.selected_primary_sample,'candidate_id'])
matched_native_cells=0;raw_matrix_cells=0;selected_cells=0;mask_cells=0;identity_checks=0;metadata_checks=0;recovered_checks=0
for site in ['oral','gut']:
 meta=M[site+'_species'];src=Path(meta['path']);sourcehash=sha(src);sr=F.loc[F.site.eq(site)&F.candidate_kind.eq('abundance_row')];fm={int(r.abundance_row_ordinal):r._asdict() for r in sr.itertuples(index=False)};assert set(fm)==set(range(1,meta['rows']+1));tax=meta['taxon_columns'];labels=[x.split('|')[-1] for x in tax];source_key_counts=Counter(tuple(r[k] for k in K) for r in sr.to_dict('records'))
 names=pq.ParquetFile(M[site+'_index']['path']).schema_arrow.names;ic=[k for k in K+['collection_date','sample_name','raw_read_count','trimmed_read_count','duplicate_reads_count','excessive_duplicate_reads','metaphlan4_results_txt'] if k in names];ix=pq.ParquetFile(M[site+'_index']['path']).read(columns=ic,use_threads=False).to_pylist();bykey=defaultdict(list);bypd=defaultdict(list)
 for i,r in enumerate(ix,1):
  if valid(r):bykey[kv(r)].append(i)
  if r['participant_id'] is not None and ds(r.get('collection_date')):bypd[(str(r['participant_id']),ds(r['collection_date']))].append(i)
 out=pq.read_table(Q/(site+'_selected_abundance_TRE_ONLY.parquet')).to_pandas(ignore_metadata=True);masks=pq.read_table(Q/(site+'_source_null_masks_TRE_ONLY.parquet')).to_pandas(ignore_metadata=True);assert set(out.sample_id)==set(S.loc[S.site.eq(site),'sample_id'])==set(masks.sample_id);assert list(out.columns[:len(tax)])==tax and not out.sample_id.duplicated().any();outmap={int(r.abundance_row_ordinal):i for i,r in out.iterrows()};maskmap=dict(zip(masks.sample_id,masks.source_null_mask_packbits_little));n=0
 pf=pq.ParquetFile(src,pre_buffer=False);idcols=[x for x in K+['collection_date','array_index'] if x in pf.schema_arrow.names]
 native_cache={}
 def native(i):
  if i in native_cache:return native_cache[i]
  raw=ix[i-1]['metaphlan4_results_txt'];p=Path(raw);p=p if p.is_absolute() else R/'gut_microbiome'/p;assert p.is_file() and p.stat().st_size<4*1024**2;lines=p.read_text().splitlines();head=next(x.lstrip('#').strip().split('\t') for x in lines if x.startswith('#clade_name'));records=csv.DictReader(io.StringIO('\n'.join(x for x in lines if x and not x.startswith('#'))),fieldnames=head,delimiter='\t');d={}
  for r in records:
   name=r['clade_name'].split('|')[-1]
   if name.startswith('s__'):assert name not in d;d[name]=float(r['relative_abundance'])
  native_cache[i]=d;return d
 for b in pf.iter_batches(columns=idcols+tax,batch_size=127,use_threads=False):
  a=np.column_stack([b.column(b.schema.get_field_index(t)).to_numpy(zero_copy_only=False) for t in tax]).astype(float);ident=b.select(idcols).to_pylist();raw_matrix_cells+=a.size
  for i,raw in enumerate(ident):
   ordinal=n+i+1;r=fm[ordinal];cid=hashlib.sha256((sourcehash+'|'+str(ordinal)+'|'+site).encode()).hexdigest();assert r['candidate_id']==cid and r['random_priority_after_required_identity_and_technical_QC']==hashlib.sha256(('20261027|'+site+'|'+cid).encode()).hexdigest();finite=np.isfinite(a[i]);assert r['nonfinite_values']==int((~finite).sum()) and r['negative_values']==int((a[i][finite]<0).sum());total=float(a[i][finite].sum());assert abs(total-r['finite_abundance_sum'])<1e-9
   resolved=None
   if site=='oral':
    hits=[j for j in bypd.get((str(raw['participant_id']),ds(raw.get('collection_date'))),[]) if valid(ix[j-1]) and all(raw.get(k) is None or str(raw[k]).lower() in BAD or str(raw[k])==str(ix[j-1][k]) for k in ['cohort','research_stage'])];resolved=hits[0] if len(hits)==1 else None
   elif valid(raw):
    options=bykey[kv(raw)]
    if len(options)==1 and source_key_counts[kv(raw)]==1:resolved=options[0]
    elif r['identity_status']=='resolved_unique_native_profile_same_visit_distinct_dated_biological_repeats':
     matches=[];expected=np.nan_to_num(a[i],nan=0.)
     for j in options:
      d=native(j);vector=np.array([d.get(t,0.) for t in labels]);outside=any(v>0 and t not in set(labels) for t,v in d.items())
      if not outside and np.allclose(expected,vector,rtol=0,atol=1e-8):matches.append(j)
     assert len(matches)==1;resolved=matches[0];recovered_checks+=1;matched_native_cells+=len(tax)
   if r['identity_ready_for_W13']:
    assert resolved is not None and resolved==int(r['linked_index_row_ordinal']) and kv(r)==kv(ix[resolved-1]);identity_checks+=1
   else:assert ordinal not in outmap
   if resolved is not None:
    meta_row=ix[resolved-1];stored=json.loads(r['technical_metadata_json'])
    for c in ['raw_read_count','trimmed_read_count','duplicate_reads_count','excessive_duplicate_reads']:
     if c in meta_row:assert stored.get(c)==meta_row[c];metadata_checks+=1
    sourcefail=any(meta_row.get(c) is not None and float(meta_row[c])<=0 for c in ['raw_read_count','trimmed_read_count']);sourcefail|=meta_row.get('excessive_duplicate_reads') is True
    numericpass=(not (a[i][finite]<0).any() and total>0 and np.max(a[i][finite])<=100+1e-8 and abs(total-100)<=max(.001,len(tax)*.5e-5+1e-8));assert bool(r['available_technical_QC_pass'])==(numericpass and not sourcefail)
   if ordinal in outmap:
    z=out.iloc[outmap[ordinal]];expected=np.where(np.isnan(a[i]),0.,a[i])/100.;got=z[tax].to_numpy(dtype=float);assert np.array_equal(got,expected);assert all(z[k]==r[k] for k in K);bits=np.unpackbits(np.frombuffer(maskmap[z.sample_id],dtype=np.uint8),bitorder='little')[:len(tax)];assert np.array_equal(bits.astype(bool),np.isnan(a[i]));selected_cells+=len(tax);mask_cells+=len(tax)
  n+=len(b)
 assert n==meta['rows'];print('INDEPENDENT_FULL_SOURCE_PASS',site,n,'selected_rows',len(out),flush=True)
# Winner reconstruction does not trust the producer's selected flag or stored ordering.
expected=set()
for key,g in F.loc[F.eligible_for_primary_selection].groupby(K+['site'],dropna=False,observed=True):
 def priority(cid):return hashlib.sha256(('20261027|'+key[-1]+'|'+cid).encode()).hexdigest()
 expected.add(min(g.candidate_id,key=lambda cid:(priority(cid),cid)))
assert expected==set(S.candidate_id)
# Recovered groups must still be bijective and have distinct actual source sampling dates, not array locators.
for key,g in F.loc[F.identity_status.eq('resolved_unique_native_profile_same_visit_distinct_dated_biological_repeats')].groupby(K,observed=True):
 assert len(g)>1 and g.linked_index_row_ordinal.is_unique and g.descriptive_sample_calendar_date.notna().all() and g.descriptive_sample_calendar_date.is_unique and g.index_sample_name.is_unique
summary=json.loads((Q/'provisional_summary.json').read_text())
for site in ['oral','gut']:
 z=S.loc[S.site.eq(site)];b=z.loc[z.cohort.eq('10k')&z.research_stage.eq('00_00_visit')];assert len(z)==summary['site'][site]['selected_all_source_rows'] and b.participant_id.nunique()==summary['site'][site]['selected_baseline_people']
assert recovered_checks==34
v={'passed':True,'actual_exits_before_verifier':actual,'all_source_abundance_rows':sum(M[s+'_species']['rows'] for s in ['oral','gut']),'candidate_rows':len(F),'selected_rows':len(S),'raw_matrix_cells_recomputed':raw_matrix_cells,'selected_matrix_cells_exactly_recomputed':selected_cells,'original_null_mask_cells_reverified':mask_cells,'identity_links_rebuilt':identity_checks,'source_technical_metadata_cells_rechecked':metadata_checks,'recovered_rows_native_profiles_rechecked':recovered_checks,'recovered_native_cells_compared':matched_native_cells,'random_winners_rehashed_all_groups':True,'all_original_taxa_retained':True,'oral_implicit_zero_contract_is_inferred_not_direct_native_validation':True,'no_CGM_diet_or_other_Y_values_read':True,'no_model_or_training_transform_fit':True,'shared_sample_registry_not_yet_published':True};p=Q/'independent_W13_verification.json';assert not p.exists();p.write_text(json.dumps(v,indent=2)+'\n');print('INDEPENDENT_W13_PASS',json.dumps(v),flush=True)
