# -*- coding: utf-8 -*-
"""Outcome-blind numerical/source QC, biological-repeat random choice, provisional zero-view matrices."""
import os
for n in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
from pathlib import Path
from collections import Counter
import json,hashlib,resource,shutil
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');Q=P/'data_qc/w13_microbiome_qc_20261029';L=Path('/home/ec2-user/multisystem_runtime_20261021/w13_microbiome_qc_20261029');K=['participant_id','cohort','research_stage'];G=K+['site'];resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(300,300));assert (L/'04_actual_exit_code').read_text().strip()=='0'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
art=[]
def dump(name,obj):
 p=Q/name;assert not p.exists();p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n');art.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
def parquet(df,name):
 r=L/name;p=Q/name;assert not p.exists() and not r.exists();pq.write_table(pa.Table.from_pandas(df,preserve_index=False),r,compression='zstd');shutil.copyfile(r,p);assert sha(r)==sha(p) and pq.ParquetFile(p).metadata.num_rows==len(df);art.append({'path':str(p),'runtime_source':str(r),'sha256':sha(p),'rows':len(df),'bytes':p.stat().st_size});return p
C=pq.read_table(P/'analysis/shared/sample_candidates_TRE_ONLY.parquet').to_pandas(ignore_metadata=True);N=pq.read_table(Q/'profile_numeric_qc_TRE_ONLY.parquet').to_pandas(ignore_metadata=True);E=pq.read_table(Q/'native_specimen_links_TRE_ONLY.parquet').to_pandas(ignore_metadata=True);M=json.loads((Q/'source_metadata_TRE_ONLY.json').read_text());sparse=json.loads((Q/'storage_and_specimen_evidence_v2_TRE_ONLY.json').read_text());native=json.loads((Q/'native_evidence_summary.json').read_text());assert len(C)==20547 and native['all_fixed_native_profiles_match']
# Numerical and source flags are defined without D/Y, never tuned by association or source depth quantiles.
for site in ['oral','gut']:
 a=sparse[site]['sparse_storage'];assert a['floating_NaN_nonnull_cells']==a['infinite_cells']==a['explicit_zero_cells']==0;assert a['arrow_null_cells']+a['finite_observed_cells']==M[site+'_species']['rows']*len(M[site+'_species']['taxon_columns'])
 idx=pq.read_table(Q/(site+'_index_projection_TRE_ONLY.parquet')).to_pandas(ignore_metadata=True);D={int(r.index_row_ordinal):r._asdict() for r in idx.itertuples(index=False)}
 if site=='gut':
  recovery={int(r.abundance_row_ordinal):r for r in E.loc[E.approved_for_selection_recovery].itertuples()}
  for n in C.index[C.site.eq('gut')&C.abundance_row_ordinal.notna()]:
   row=C.loc[n];rec=recovery.get(int(row.abundance_row_ordinal))
   if rec is None:continue
   ir=D[int(rec.resolved_index_row_ordinal)];assert tuple(row[k] for k in K)==tuple(ir[k] for k in K);C.loc[n,'identity_ready_for_W13']=True;C.loc[n,'identity_status']='resolved_unique_native_profile_same_visit_distinct_dated_biological_repeats';C.loc[n,'identity_evidence']=rec.identity_evidence;C.loc[n,'linked_index_row_ordinal']=rec.resolved_index_row_ordinal;C.loc[n,'index_sample_name']=ir['sample_name'];C.loc[n,'sample_id_kind']='source_sample_name_with_native_profile_and_distinct_sampling_date_evidence';C.loc[n,'descriptive_sample_calendar_date']=pd.Timestamp(ir['collection_date']).date().isoformat();C.loc[n,'linked_index_calendar_date']=pd.Timestamp(ir['collection_date']).date().isoformat();C.loc[n,'sample_date_missing']=False;C.loc[n,'technical_metadata_json']=json.dumps({k:ir[k] for k in ['raw_read_count','trimmed_read_count','duplicate_reads_count','excessive_duplicate_reads','run_name'] if k in ir},default=str)
 # Verify metadata for every linked candidate against current exact index row.
 for row in C.loc[C.site.eq(site)&C.linked_index_row_ordinal.notna()].itertuples():
  if row.candidate_kind!='abundance_row':continue
  ir=D[int(row.linked_index_row_ordinal)];assert tuple(getattr(row,k) for k in K)==tuple(ir[k] for k in K)
F=C.merge(N,on=['site','abundance_row_ordinal'],how='left',validate='many_to_one');assert len(F)==len(C) and F.candidate_id.is_unique
# Null cells inside an observed normalized sparse profile mean unreported/not detected taxa under
# the explicit source encoding contract; missing profiles are never manufactured as zero rows.
# Gut verified from native text; oral inferred from common pipeline + all-source sparse/closure evidence.
def qc_reason(row,meta):
 if row['candidate_kind']!='abundance_row':return 'no_abundance_profile'
 if row['negative_values']>0:return 'negative_abundance'
 if row['finite_abundance_sum']<=0:return 'zero_or_invalid_profile_sum'
 if row['maximum_finite_value']>100+1e-8:return 'abundance_exceeds_percent_scale'
 tol=max(.001,float(row['taxon_count'])*.5e-5+1e-8)
 if abs(row['finite_abundance_sum']-100)>tol:return 'species_composition_not_closed_at_rounding_tolerance'
 for key in ['raw_read_count','trimmed_read_count']:
  v=meta.get(key)
  if v is not None:
   try:
    if np.isfinite(float(v)) and float(v)<=0:return key+'_nonpositive'
   except (ValueError,TypeError):return key+'_unusable'
 v=meta.get('excessive_duplicate_reads')
 if v is True or (isinstance(v,str) and v.lower()=='true'):return 'source_excessive_duplicate_reads_flag'
 return 'pass_available_numerical_and_source_QC'
# Meaningful checks: do not convert missing profiles to all-zero; source failure precedes random choice.
syn={'candidate_kind':'abundance_row','negative_values':0,'finite_abundance_sum':100.,'maximum_finite_value':80.,'taxon_count':10}
assert qc_reason(syn,{})=='pass_available_numerical_and_source_QC';assert qc_reason(dict(syn,candidate_kind='index_only'),{})=='no_abundance_profile';assert qc_reason(dict(syn,negative_values=1),{})=='negative_abundance';assert qc_reason(syn,{'raw_read_count':0})=='raw_read_count_nonpositive';assert qc_reason(syn,{'excessive_duplicate_reads':True})=='source_excessive_duplicate_reads_flag'
F['available_QC_reason']=[qc_reason(r,json.loads(r['technical_metadata_json'])) for r in F.to_dict('records')];F['available_technical_QC_pass']=F.available_QC_reason.eq('pass_available_numerical_and_source_QC');F['eligible_for_primary_selection']=F.identity_ready_for_W13&F.available_technical_QC_pass;F['final_selection_reason']=np.where(~F.identity_ready_for_W13,F.identity_status,np.where(~F.available_technical_QC_pass,F.available_QC_reason,'eligible_not_yet_selected'));F['selected_primary_sample']=False
eligible=F.loc[F.eligible_for_primary_selection].sort_values(G+['random_priority_after_required_identity_and_technical_QC','candidate_id'],kind='stable');chosen=eligible.drop_duplicates(G).index;F.loc[chosen,'selected_primary_sample']=True;F.loc[F.eligible_for_primary_selection,'final_selection_reason']='eligible_biological_alternate_not_selected_fixed_hash';F.loc[chosen,'final_selection_reason']='selected_minimum_fixed_hash_after_identity_and_QC';S=F.loc[F.selected_primary_sample].copy();assert not S.duplicated(G).any();assert S.candidate_kind.eq('abundance_row').all()
# Random-order invariance and failure-before-selection are tested on actual eligible groups.
rev=F.loc[F.eligible_for_primary_selection].iloc[::-1].sort_values(G+['random_priority_after_required_identity_and_technical_QC','candidate_id']).drop_duplicates(G);assert set(rev.candidate_id)==set(S.candidate_id)
S['sample_id']=S.candidate_id;S['profile_locator_kind']='versioned_source_aggregate_row_not_claimed_universal_biological_UUID';S['source_implicit_nondetection_encoding']='Arrow null -> nondetected in source sparse composition; not biological absence or missing-sample imputation';S['abundance_view_unit']='fraction_0_1_from_source_percent_divided_by_100';S['sample_lock_status']='provisional_pending_independent_verification';S['protocol_version']='multisystem_v3_user_amendment_20261029';S['oral_direct_native_text_verified']=False
parquet(F,'microbiome_sample_flags_provisional_TRE_ONLY.parquet');parquet(S,'sample_registry_provisional_TRE_ONLY.parquet')
# Materialize selected matrices with exact row lineage and the original cell-null mask preserved separately.
for site in ['oral','gut']:
 selected=S.loc[S.site.eq(site)].copy();lookup={int(r.abundance_row_ordinal):r._asdict() for r in selected.itertuples(index=False)};tax=M[site+'_species']['taxon_columns'];source=Path(M[site+'_species']['path']);assert source.stat().st_size==M[site+'_species']['bytes'] and source.stat().st_mtime_ns==M[site+'_species']['mtime_ns'];writer=None;mw=None;n=0;kept=0;runtime=L/(site+'_selected_abundance_TRE_ONLY.parquet');maskrun=L/(site+'_source_null_masks_TRE_ONLY.parquet');assert not runtime.exists() and not maskrun.exists()
 for b in pq.ParquetFile(source,pre_buffer=False).iter_batches(columns=tax,batch_size=128,use_threads=False):
  pos=[i for i in range(len(b)) if n+i+1 in lookup]
  if pos:
   a=np.column_stack([b.column(i).to_numpy(zero_copy_only=False) for i in range(b.num_columns)]).astype('float64');raw=a[pos];mask=np.isnan(raw);values=np.where(mask,0.,raw)/100.;assert np.isfinite(values).all() and (values>=0).all();d=pd.DataFrame(values,columns=tax);ords=[n+i+1 for i in pos]
   for key in ['sample_id']+K+['site','abundance_row_ordinal']:d[key]=[lookup[z][key] for z in ords]
   table=pa.Table.from_pandas(d,preserve_index=False);md=pd.DataFrame({'sample_id':[lookup[z]['sample_id'] for z in ords],'site':site,'source_null_mask_packbits_little':[np.packbits(x,bitorder='little').tobytes() for x in mask],'source_taxon_columns':len(tax)});mt=pa.Table.from_pandas(md,preserve_index=False)
   if writer is None:writer=pq.ParquetWriter(runtime,table.schema,compression='zstd');mw=pq.ParquetWriter(maskrun,mt.schema,compression='zstd')
   writer.write_table(table);mw.write_table(mt);kept+=len(d)
  n+=len(b)
 assert n==M[site+'_species']['rows'] and kept==len(selected) and writer is not None;writer.close();mw.close()
 for r in [runtime,maskrun]:
  target=Q/r.name;assert not target.exists();shutil.copyfile(r,target);assert sha(r)==sha(target) and pq.ParquetFile(target).metadata.num_rows==kept;art.append({'path':str(target),'runtime_source':str(r),'sha256':sha(target),'rows':kept,'bytes':target.stat().st_size})
 print('PROVISIONAL_MATRIX',site,'selected_rows',kept,'taxa_retained',len(tax),flush=True)
rules={'version':'W13_provisional_20261029','protocol':'multisystem_v3_user_amendment_20261029','source_percent_to_fraction':100,'negative_abundance_allowed':False,'source_null_contract':'implicit non-detection only within observed complete source composition; all missing profiles remain missing','gut_null_validation':'3 fixed native profiles, all2088 cells each, exact positive/implicit-nondetection match','oral_null_validation':'inferred from same HPP MetaPhlAn aggregate family, no nonnull NaN/inf/explicit-zero cells, all finite-positive profiles sum100 within rounding; direct per-sample native text unavailable','preserve_original_null_mask':True,'sample_numerical_closure_tolerance':'max(0.001, n_taxa*0.5e-5+1e-8) percentage points; rounding tolerance only','source_failure_flags':['raw_read_count<=0 if observed','trimmed_read_count<=0 if observed','excessive_duplicate_reads==True per HPP source flag (>1e5 absolute duplicates)'],'source_flag_not_new_association_tuned_cutoff':True,'no_arbitrary_minimum_depth_threshold':True,'unobserved_depth_or_contamination_not_claimed_passed':True,'specimen_recovery':'only bijective native-profile links within same person/cohort/stage, then distinct source sample names and sampling dates to establish biological repetition; all remaining ambiguity held','seed':20261027,'ordering':'minimum stored SHA256(seed|site|candidate_id) after eligibility; row locator/source version fixed','no_lag_date_difference_admission_rule':True,'no_outcome_diet_association_used':True,'prevalence_filter_CLR_PCA_or_supervised_fit':False,'all_source_taxa_retained':True,'lock_reused_across_all_outcomes':True,'final_shared_release':'only after independent W13 verifier'}
summary={'candidate_rows':len(F),'selected_all_source_person_visit_site_rows':len(S),'site':{},'baseline_joint_selected_people':None,'source_duplicate_flag_fail_rows':int(F.available_QC_reason.eq('source_excessive_duplicate_reads_flag').sum()),'native_recovered_rows':int(E.approved_for_selection_recovery.sum()),'recovered_rows_selected':int(S.identity_status.eq('resolved_unique_native_profile_same_visit_distinct_dated_biological_repeats').sum()),'synthetic_and_order_invariance_checks_passed':True,'no_model_fit':True,'no_raw_data_export':True}
for site in ['oral','gut']:
 z=S.loc[S.site.eq(site)];b=z.loc[z.cohort.eq('10k')&z.research_stage.eq('00_00_visit')];f=F.loc[F.site.eq(site)];summary['site'][site]={'selected_all_source_rows':len(z),'selected_all_source_people':int(z.participant_id.nunique()),'selected_baseline_people':int(b.participant_id.nunique()),'candidate_rows':len(f),'identity_ready_rows':int(f.identity_ready_for_W13.sum()),'identity_held_rows':int((~f.identity_ready_for_W13).sum()),'source_QC_fail_with_ready_identity':int((f.identity_ready_for_W13&~f.available_technical_QC_pass).sum()),'selection_reasons':f.final_selection_reason.value_counts().astype(int).to_dict()}
o=set(S.loc[S.site.eq('oral')&S.cohort.eq('10k')&S.research_stage.eq('00_00_visit'),'participant_id'].astype(int));g=set(S.loc[S.site.eq('gut')&S.cohort.eq('10k')&S.research_stage.eq('00_00_visit'),'participant_id'].astype(int));summary['baseline_joint_selected_people']=len(o&g)
dump('w13_QC_selection_rules_provisional.json',rules);dump('provisional_summary.json',summary);dump('provisional_artifact_manifest.json',art);print('W13_PROVISIONAL_SUMMARY',json.dumps(summary),flush=True);print('W13_PROVISIONAL_BUILD_COMPLETE',flush=True)
