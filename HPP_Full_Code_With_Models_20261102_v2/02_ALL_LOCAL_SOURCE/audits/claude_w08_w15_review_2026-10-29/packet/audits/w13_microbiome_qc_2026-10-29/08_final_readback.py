# -*- coding: utf-8 -*-
"""Independent final shared-product hash and invariant readback; no source modifications."""
from pathlib import Path
import json,hashlib,shutil,resource
import pandas as pd,pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');Q=P/'data_qc/w13_microbiome_qc_20261029';L=Path('/home/ec2-user/multisystem_runtime_20261021/w13_microbiome_qc_20261029');T=P/'analysis/shared';resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(120,120));assert (L/'07_actual_exit_code').read_text().strip()=='0'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
a=json.loads((Q/'final_artifact_manifest.json').read_text());assert len({x['path'] for x in a})==len(a)
for x in a:
 p=Path(x['path']);assert sha(p)==x['sha256'] and p.stat().st_size==x['bytes']
 if x.get('rows') is not None:assert pq.ParquetFile(p).metadata.num_rows==x['rows']
S=pq.read_table(T/'sample_registry_TRE_ONLY.parquet').to_pandas(ignore_metadata=True);F=pq.read_table(T/'microbiome_sample_flags_TRE_ONLY.parquet').to_pandas(ignore_metadata=True);old=pq.read_table(Q/'sample_registry_provisional_TRE_ONLY.parquet').to_pandas(ignore_metadata=True);assert set(S.candidate_id)==set(old.candidate_id)==set(F.loc[F.selected_primary_sample,'candidate_id']);assert not S.duplicated(['participant_id','cohort','research_stage','site']).any() and S.sample_id.is_unique and S.sample_lock_status.eq('locked_W13_independent_full_source_verification_passed').all()
for site in ['oral','gut']:
 z=S.loc[S.site.eq(site)];m=pq.read_table(T/(site+'_selected_abundance_TRE_ONLY.parquet'),columns=['sample_id']).column(0).to_pylist();mask=pq.read_table(T/(site+'_source_null_masks_TRE_ONLY.parquet'),columns=['sample_id']).column(0).to_pylist();assert len(m)==len(set(m))==len(z) and set(m)==set(mask)==set(z.sample_id)
active=json.loads((P/'config/ACTIVE_PROTOCOL.json').read_text());lock=json.loads((P/'config/w13_sample_selection_lock.json').read_text());assert active['version']=='multisystem_v3_user_amendment_20261029' and sha(active['W13_sample_selection_lock']['path'])==active['W13_sample_selection_lock']['sha256'] and sha(T/'sample_registry_TRE_ONLY.parquet')==active['W13_sample_registry']['sha256']==lock['sample_registry_sha256'];assert lock['apply_same_lock_across_outcome_modules'] and lock['no_reselection_on_outcome_window_or_effect']
# Verify synchronized original authored documents still match their preflight receipts.
receipt=json.loads((Q/'protocol_sync_receipt.json').read_text())
for x in receipt['source_receipts']+receipt['old_configs_preserved']:assert sha(x['path'])==x['sha256']
agg=json.loads((Q/'aggregate_delivery.json').read_text());assert hashlib.sha256(json.dumps(agg['aggregate'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==agg['canonical_sha256']
for site in ['oral','gut']:
 z=S.loc[S.site.eq(site)];b=z.loc[z.cohort.eq('10k')&z.research_stage.eq('00_00_visit')];assert len(z)==agg['aggregate'][site+'_selected_rows'] and b.participant_id.nunique()==agg['aggregate'][site+'_baseline_people']
for name in ['07.log','07_actual_exit_code']:
 dst=P/'logs/w13_microbiome_qc_20261029'/name;assert not dst.exists();shutil.copyfile(L/name,dst);assert sha(L/name)==sha(dst)
v={'passed':True,'workpackage':'W13','protocol_sync_and_preserved_old_config_hashes_pass':True,'final_manifest_artifacts_verified':len(a),'candidate_rows':len(F),'locked_registry_rows':len(S),'shared_matrix_registry_masks_one_to_one':True,'lock_reused_across_modules':True,'no_model_fit':True,'aggregate_canonical_sha256':agg['canonical_sha256'],'actual_prior_exits':{n:int((L/(n+'_actual_exit_code')).read_text()) for n in ['01','02','03','03b','04','05','06','07']},'07_log_saved_and_verified':True};out=Q/'final_delivery_verification.json';assert not out.exists();out.write_text(json.dumps(v,indent=2)+'\n');assert json.loads(out.read_text())==v;print('FINAL_W13_READBACK_PASS',json.dumps(v),flush=True);print('FINAL_W13_AGGREGATE',json.dumps(agg['aggregate'],sort_keys=True,separators=(',',':')),flush=True);print('FINAL_W13_CANONICAL_SHA256',agg['canonical_sha256'],flush=True)
