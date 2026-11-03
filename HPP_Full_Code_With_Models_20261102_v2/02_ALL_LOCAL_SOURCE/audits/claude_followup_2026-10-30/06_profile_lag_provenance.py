"""Drill down identical profiles, create separate lag correction overlay, reconcile execution provenance."""
import runpy
from pathlib import Path
B=runpy.run_path(str(Path(__file__).with_name('03_source_QC.py')),run_name='auditlib');globals().update({k:v for k,v in B.items() if not k.startswith('__')})
import pyarrow as pa
F=read('sample_flags');CV=read('covariates',K+['observed_related_group_id']);W=read('W15_order',K+['split_unit_id']);ident=[];source_meta=json.loads((P/'data_qc/w13_microbiome_qc_20261029/source_metadata_TRE_ONLY.json').read_text())
for site in ['oral','gut']:
 a=F.loc[F.site.eq(site)&F.candidate_kind.eq('abundance_row')];dup=a.loc[a.profile_sha256_float64.duplicated(keep=False)];idx=read(site+'_index_projection');specmeta=source_meta[site+'_species'];p=Path(specmeta['path']);sp=pq.ParquetFile(p,pre_buffer=False);tax=specmeta['taxon_columns'];lookup={};ordinals=set(dup.abundance_row_ordinal.astype(int));assert len(ordinals)<=100
 if ordinals:
  expected_hash=set(dup.abundance_source_sha256);assert len(expected_hash)==1 and sha(p) in expected_hash;save('access_receipts/'+uuid.uuid4().hex+'.json',{'time_UTC':datetime.now(timezone.utc).isoformat(),'script':str(Path(__file__)),'script_sha256':sha(__file__),'path':str(p),'source_sha256':sha(p),'columns':tax,'source_rows':sp.metadata.num_rows,'purpose':'X03 bounded streaming exact raw duplicate-profile confirmation; keep only preidentified ordinals in TRE','technical_Y_value_read':False,'scientific_association_or_training':False,'individual_data_exported':False,'selected_ordinal_count':len(ordinals),'batch_rows':128});n=0
  for b in sp.iter_batches(columns=tax,batch_size=128,use_threads=False):
   pos=[i for i in range(len(b)) if n+i+1 in ordinals]
   if pos:
    arr=np.column_stack([b.column(i).to_numpy(zero_copy_only=False) for i in range(b.num_columns)]).astype('float64')
    for i in pos:lookup[n+i+1]=arr[i].copy()
   n+=len(b)
   if n>=max(ordinals):break
 for h,g in dup.groupby('profile_sha256_float64'):
  loc=idx.loc[idx.index_row_ordinal.isin(g.linked_index_row_ordinal.dropna().astype(int))];vectors=[lookup[int(x)] for x in g.abundance_row_ordinal];numeric_identical=all(np.array_equal(vectors[0],v,equal_nan=True) for v in vectors[1:]);hash_matches=True
  for v in vectors:
   v[v==0]=0.;hash_matches &= hashlib.sha256(v.tobytes()).hexdigest()==h
  row={'site':site,'member_rows':len(g),'person_visits':len(g[K].drop_duplicates()),'selected_rows':int(g.selected_primary_sample.sum()),'baseline_selected_rows':int((g.selected_primary_sample&g.cohort.eq('10k')&g.research_stage.eq('00_00_visit')).sum()),'W15_candidate_members':len(g[K].drop_duplicates().merge(W[K],on=K)),'raw_numeric_profiles_identical_including_nulls':numeric_identical,'raw_hash_matches_prior_QC':bool(hash_matches),'detected_taxa_min':int(min(np.count_nonzero(v>0) for v in vectors)),'detected_taxa_max':int(max(np.count_nonzero(v>0) for v in vectors)),'maximum_relative_abundance_percent':float(max(np.nanmax(v) for v in vectors)),'same_metadata_fields':{}}
  for c in ['cohort','research_stage','index_sample_name','descriptive_sample_calendar_date','abundance_row_ordinal','linked_index_row_ordinal']:row['same_metadata_fields'][c]={'all_observed':bool(g[c].notna().all()),'distinct_nonmissing':int(g[c].nunique())}
  for c in ['sample_name','collection_date','raw_read_count','trimmed_read_count','duplicate_reads_count','excessive_duplicate_reads','run_name','metaphlan4_results_txt','metaphlan_abundance_species_parquet']:
   if c in loc:
    vals=loc[c].map(str);row['same_metadata_fields']['index_'+c]={'all_observed':bool(loc[c].notna().all()),'distinct_nonmissing':int(loc[c].nunique()),'undefined_values':int(vals.str.lower().isin(['undefined','nan','none','']).sum())}
  z=g[K].merge(CV,on=K,how='left',validate='many_to_one');row['known_observed_related_groups_distinct']=int(z.observed_related_group_id.nunique());zz=g[K].merge(W,on=K,how='left',validate='many_to_one');row['W15_known_split_units_distinct']=int(zz.split_unit_id.nunique());row['sample_reassignment_or_deduplication_done']=False;ident.append(row)
profile={'groups':ident,'gut_duplicate_groups':sum(r['site']=='gut' for r in ident),'oral_duplicate_groups':sum(r['site']=='oral' for r in ident),'equality_is_not_automatic_evidence_of_shared_specimen':True};save('X03_identical_profile_drilldown.json',profile);print('X03_PROFILE_DRILLDOWN',json.dumps(profile),flush=True)
# Corrected lag values in a new optional overlay, never overwrite the locked W13 registry/flags.
rec=F.loc[F.identity_status.eq('resolved_unique_native_profile_same_visit_distinct_dated_biological_repeats')].copy();out=rec[['candidate_id']+K+['site','selected_primary_sample','descriptive_sample_calendar_date']].copy()
for label,role,flag,col in [('food','food_daily',None,'sample_minus_nearest_observed_food_day'),('CGM','cgm_daily_selected_legacy','cgm_day_eligible','sample_minus_nearest_qualified_CGM_day')]:
 d=read(role,K+['date']+([flag] if flag else []));d=d if flag is None else d.loc[d[flag]];d['ordinal']=pd.to_datetime(d.date).map(lambda x:x.date().toordinal());maps={k:sorted(set(g.ordinal)) for k,g in d.groupby(K)};vals=[]
 for r in rec.itertuples(index=False):
  ds=maps.get(tuple(getattr(r,k) for k in K));sd=r.descriptive_sample_calendar_date;v=None
  if ds and not pd.isna(sd):
   n=pd.Timestamp(sd).date().toordinal();i=bisect.bisect_left(ds,n);near=min(ds[max(0,i-1):min(len(ds),i+1)],key=lambda t:(abs(n-t),t));v=n-near
  vals.append(v)
 out[col+'_original']=rec[col].to_numpy();out[col+'_corrected']=pd.array(vals,dtype='Int64')
out['overlay_status']='audit_correction_proposal_original_sample_lock_unchanged';op=Q/'X12_recovered_lag_correction_overlay_TRE_ONLY.parquet';assert not op.exists();pq.write_table(pa.Table.from_pandas(out,preserve_index=False),op,compression='zstd');r=pq.read_table(op).to_pandas(ignore_metadata=True);assert len(r)==34 and not r.candidate_id.duplicated().any();lag={'overlay_rows':len(r),'selected_rows':int(r.selected_primary_sample.sum()),'food_computable_rows':int(r.sample_minus_nearest_observed_food_day_corrected.notna().sum()),'CGM_computable_rows':int(r.sample_minus_nearest_qualified_CGM_day_corrected.notna().sum()),'overlay_sha256':sha(op),'changes_eligibility_or_sample_choice':False};save('X12_overlay_verification.json',lag);print('X12_SEPARATE_OVERLAY',json.dumps(lag),flush=True)
# Historical log presence + exact success markers, without printing participant-level content.
L=Path('/home/ec2-user/multisystem_runtime_20261021');run=L/'w09_cgm_point_qc_20261028';rows=[]
for tag,marker in [('03b','W09_BUILD_DONE'),('09','CALENDAR_CONTRACT_FIX_DONE'),('04b','W09_INDEPENDENT_PASS'),('05','W08_COMMON'),('06','W08_COMMON_INDEPENDENT_PASS'),('07','FINAL_MEMO_WRITTEN')]:
 p=run/(tag+'.log');row={'tag':tag,'log_exists':p.exists(),'exit':(run/(tag+'_actual_exit_code')).read_text().strip()}
 if p.exists():
  assert p.stat().st_size<10*1024**2;s=p.read_text();row.update(bytes=p.stat().st_size,sha256=sha(p),expected_marker=marker,marker_found=marker in s,traceback_present='Traceback (most recent call last)' in s)
 rows.append(row)
ver=json.loads((P/'data_qc/w09_cgm_point_qc_20261028/verification.json').read_text());prov={'logs':rows,'canonical_03_exit_is_dependency_marker':True,'actual_corrected_build_exit_tag':'03b','original_03_failure_preserved':(run/'03_initial_actual_exit_code').read_text().strip()=='1','actual_final_calendar_verifier_tag':'04b','stored_W09_verification_summary':{k:ver.get(k) for k in ['verified','source_fragments_hash_verified','independent_source_fragments','independent_source_days','mirrors_verified']},'fresh_raw_point_replay_done_in_this_followup':False,'remote_script_hash_comparison':{'expected_total':60,'found_and_identical':59,'missing_file':'03_record_delivery.py','missing_reason_verified_from_local_frozen_source':'local report/aggregate transcription script writes to local workspace; not a TRE scientific execution script'}};save('X10_execution_provenance_reconciled.json',prov);print('X10_PROVENANCE',json.dumps(prov),flush=True)
addendum={'scope':'additive audit note; frozen W15 history ledger not rewritten','colleague_history':'unknown pending本人确认; generic CGM-related domain in actor row is a placeholder and must not imply verified CGM result contact','historical_exact_ID_overlap':'unknown; new split cannot erase prior exposure','technical_contacts_previously_not_separate_rows':['W08 diet QC','W09 CGM numerical QC','W10 static outcome QC','W11 covariate source processing','W12 sample identity/date QC','W13 microbiome numerical QC'],'current_followup_contacts':'technical source/derived Y values read for declared numerical QC with immutable per-projection receipts; no scientific association or training','future_real_Y_access_control_gap':'not implemented by this read-only audit; technical receipts here do not establish a future workflow wrapper or filesystem ACL'};save('X11_history_and_access_addendum.json',addendum);print('PROFILE_LAG_PROVENANCE_FINISHED',flush=True)
