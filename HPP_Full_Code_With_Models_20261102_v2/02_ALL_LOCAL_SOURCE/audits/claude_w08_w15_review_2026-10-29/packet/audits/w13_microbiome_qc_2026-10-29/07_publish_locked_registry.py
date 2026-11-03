# -*- coding: utf-8 -*-
"""Publish independently verified W13 products; preserve all prior source and protocol versions."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib,shutil,sys,platform,resource
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');Q=P/'data_qc/w13_microbiome_qc_20261029';L=Path('/home/ec2-user/multisystem_runtime_20261021/w13_microbiome_qc_20261029');T=P/'analysis/shared';K=['participant_id','cohort','research_stage','site'];resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(120,120))
assert (L/'06_actual_exit_code').read_text().strip()=='0';V=json.loads((Q/'independent_W13_verification.json').read_text());assert V['passed']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
art=[]
def publish(src,dst,rows=None):
 assert not dst.exists();dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);assert sha(src)==sha(dst)
 if rows is not None:assert pq.ParquetFile(dst).metadata.num_rows==rows
 art.append({'path':str(dst),'runtime_or_verified_source':str(src),'sha256':sha(dst),'bytes':dst.stat().st_size,'rows':rows});return dst

def dump(name,x):
 src=L/name;assert not src.exists();src.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str,allow_nan=False)+'\n');return publish(src,Q/name)
def parq(df,name):
 src=L/name;assert not src.exists();pq.write_table(pa.Table.from_pandas(df,preserve_index=False),src,compression='zstd');return publish(src,T/name,len(df))
F=pq.read_table(Q/'microbiome_sample_flags_provisional_TRE_ONLY.parquet').to_pandas(ignore_metadata=True);S=pq.read_table(Q/'sample_registry_provisional_TRE_ONLY.parquet').to_pandas(ignore_metadata=True);summary=json.loads((Q/'provisional_summary.json').read_text());M=json.loads((Q/'source_metadata_TRE_ONLY.json').read_text());assert not S.duplicated(K).any() and len(S)==V['selected_rows'];S['sample_lock_status']='locked_W13_independent_full_source_verification_passed';F['sample_lock_status']=np.where(F.selected_primary_sample,'locked_primary_W13','not_primary_reason_recorded_W13');reg=parq(S,'sample_registry_TRE_ONLY.parquet');flags=parq(F,'microbiome_sample_flags_TRE_ONLY.parquet')
for site in ['oral','gut']:
 for suffix in ['selected_abundance_TRE_ONLY.parquet','source_null_masks_TRE_ONLY.parquet']:publish(Q/(site+'_'+suffix),T/(site+'_'+suffix),summary['site'][site]['selected_all_source_rows'])
# Aggregate QC/reason tables only; individual selection evidence remains in TRE Parquet.
qc=[]
for site in ['oral','gut']:
 a=M[site+'_species'];z=summary['site'][site];qc.append({'site':site,'source_abundance_rows':a['rows'],'source_taxa_retained':len(a['taxon_columns']),**{k:v for k,v in z.items() if k!='selection_reasons'},'oral_null_encoding_inferred_not_direct':site=='oral','arbitrary_depth_cutoff_used':False,'unmeasured_contamination_QC_claimed':False})
for name,df in [('microbiome_abundance_QC.csv',pd.DataFrame(qc)),('sample_selection_reasons.csv',F.groupby(['site','final_selection_reason'],dropna=False).size().reset_index(name='candidate_rows'))]:
 src=L/name;assert not src.exists();df.to_csv(src,index=False);publish(src,Q/name);publish(src,P/'data_qc'/name)
rules=json.loads((Q/'w13_QC_selection_rules_provisional.json').read_text());rules.update({'version':'W13_locked_20261029','final_shared_release':'published_after_independent_full_source_verification','sample_registry':str(reg),'sample_registry_sha256':sha(reg),'sample_flags':str(flags),'sample_flags_sha256':sha(flags),'independent_verification':str(Q/'independent_W13_verification.json'),'independent_verification_sha256':sha(Q/'independent_W13_verification.json'),'selected_summary':summary,'apply_same_lock_across_outcome_modules':True,'no_reselection_on_outcome_window_or_effect':True});rulepath=P/'config/w13_sample_selection_lock.json';src=L/'w13_sample_selection_lock.json';assert not src.exists();src.write_text(json.dumps(rules,indent=2,ensure_ascii=False)+'\n');publish(src,rulepath)
# Preserve previous pointer bytes before advancing the reversible current pointer.
active=P/'config/ACTIVE_PROTOCOL.json';backup=Q/'ACTIVE_PROTOCOL_before_W13.json';assert active.exists() and not backup.exists();shutil.copyfile(active,backup);old=json.loads(active.read_text());assert old['version']=='multisystem_v3_user_amendment_20261029';new=dict(old);new['W13_sample_selection_lock']={'path':str(rulepath),'sha256':sha(rulepath),'status':'locked_after_independent_verification'};new['W13_sample_registry']={'path':str(reg),'sha256':sha(reg)};new['W13_matrix_manifest']=str(Q/'final_artifact_manifest.json');tmp=L/'ACTIVE_PROTOCOL_after_W13.json';tmp.write_text(json.dumps(new,indent=2,ensure_ascii=False)+'\n');shutil.copyfile(tmp,active);assert sha(active)==sha(tmp);publish(active,Q/'ACTIVE_PROTOCOL_after_W13.json')
# Source fingerprints, packages, scripts, completed logs and actual exits.
sources=[]
for role,m in M.items():
 p=Path(m['path']);assert p.stat().st_size==m['bytes'] and p.stat().st_mtime_ns==m['mtime_ns'];sources.append({'role':role,'path':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':sha(p),'rows':m['rows']})
exits={}
logdir=P/'logs/w13_microbiome_qc_20261029';logdir.mkdir(parents=True,exist_ok=True)
for tag in ['01','02','03','03b','04','05','06']:
 exits[tag]=int((L/(tag+'_actual_exit_code')).read_text());publish(L/(tag+'.log'),logdir/(tag+'.log'));publish(L/(tag+'_actual_exit_code'),logdir/(tag+'_actual_exit_code'))
scripts=[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted((P/'scripts/w13_microbiome_qc_20261029').glob('*.py'))]
dump('run_provenance.json',{'protocol':'multisystem_v3_user_amendment_20261029','python':sys.version,'executable':sys.executable,'platform':platform.platform(),'versions':{'numpy':np.__version__,'pandas':pd.__version__,'pyarrow':pa.__version__},'seed':20261027,'source_inputs':sources,'script_inputs':scripts,'actual_exits':exits,'guard_failure_preserved':'03 timed out bounded directory inventory; 03b exact pointers passed','maximum_address_space_GiB':4})
# Compact approved aggregate package is the only dataset-derived content transcribed locally.
agg={'candidate_rows':len(F),'selected_rows':len(S),'oral_selected_rows':summary['site']['oral']['selected_all_source_rows'],'oral_baseline_people':summary['site']['oral']['selected_baseline_people'],'oral_taxa':len(M['oral_species']['taxon_columns']),'gut_selected_rows':summary['site']['gut']['selected_all_source_rows'],'gut_baseline_people':summary['site']['gut']['selected_baseline_people'],'gut_taxa':len(M['gut_species']['taxon_columns']),'baseline_joint_people':summary['baseline_joint_selected_people'],'recovered_biological_candidate_rows':summary['native_recovered_rows'],'selected_recovered_candidate_rows':summary['recovered_rows_selected'],'held_oral_candidate_rows':summary['site']['oral']['identity_held_rows'],'held_gut_candidate_rows':summary['site']['gut']['identity_held_rows'],'source_duplicate_flag_fail_rows':summary['source_duplicate_flag_fail_rows'],'independent_full_source_pass':True,'oral_null_contract_inferred':True,'no_model_fit':True};digest=hashlib.sha256(json.dumps(agg,sort_keys=True,separators=(',',':')).encode()).hexdigest();dump('aggregate_delivery.json',{'aggregate':agg,'canonical_sha256':digest})
memo='''# W13 microbiome QC and sample lock — 2026-10-29

Code: P/scripts/w13_microbiome_qc_20261029 (P is this project root).
Results: P/data_qc/w13_microbiome_qc_20261029. Runtime and live logs: /home/ec2-user/multisystem_runtime_20261021/w13_microbiome_qc_20261029.

Latest user-amended protocol synchronized into versioned P/config/protocol_amendment_20261029 before execution; original hashed W10/W11/W12 configurations preserved. ACTIVE_PROTOCOL.json now additionally points to the W13 sample lock.

Both complete abundance sources scanned. Numerical/source failures are outcome blind. No arbitrary read-depth cutoff; unmeasured contamination or other unavailable assay QC is not certified. Gut HPP excessive_duplicate_reads True is excluded by the existing source flag. Original percentage values are divided by 100. All species retained; no prevalence/CLR/PCA/supervised fitting.

Arrow NULL is an implicit unreported taxon within an observed sparse normalized profile. Three pre-fixed gut native MetaPhlAn profiles matched all 2088 cells each. Oral has no equivalent native text: its encoding is inferred from the shared aggregation family, exclusively positive observed values, no nonnull NaN/Inf/explicit zeros and all-source closure. This is an explicit limitation, not direct validation. Original packed cell-null masks and unchanged raw sources retained. Never create zero rows for missing samples; non-detection does not prove biological absence.

34 gut rows recovered only after unique native-profile matches and distinct names/collection dates established biological repeat evidence. Remaining ambiguity held. SHA256(seed|site|candidate_id), seed 20261027, is minimized after identity and available QC. Same locked sample reused across outcomes, never selected by Y/effect/nearest date. sample_id is a versioned aggregate-row locator, not a claimed universal biological UUID.

Independent script 06 reread all raw matrix values, reconstructed identities and source metadata, exactly compared every selected cell plus source-null mask, and independently rehashed winners. Script 03 directory inventory hit its timeout guard; preserved, corrected by exact-source script 03b. See run_provenance.json for true exits and versions.

Final shared registry/flags, matrices and masks are in P/analysis/shared. Reasons and QC CSV in P/data_qc. Final hashes in final_artifact_manifest.json. Numbers in aggregate_delivery.json are sample-level QC availability, not final outcome-specific analysis N.

Next W14: build each outcome separately, six formal liver phenotypes; rebuild CGM person phenotypes from qualified same-visit CGM period without requiring food-day overlap; sequence/daily analyses retain actual dates. Explicit study-window medication exposed/confirmed-unexposed/unknown; missing record is unknown. No deferred glucose–liver comparison. UPF bounds/coverage alternative retained, no NOVA-completion gate. Hypothesis/exposure budget remains W18/W22.
''';memodir=P/'memos/w13_microbiome_qc_20261029';src=L/'MEMO.md';src.write_text(memo);publish(src,memodir/'MEMO.md');dump('final_artifact_manifest.json',art.copy());assert not S.duplicated(K).any();print('W13_LOCKED_AGGREGATE',json.dumps(agg,sort_keys=True,separators=(',',':')),flush=True);print('AGGREGATE_CANONICAL_SHA256',digest,flush=True);print('W13_PUBLISH_COMPLETE_PENDING_FINAL_READBACK',flush=True)
