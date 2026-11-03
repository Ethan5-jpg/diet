"""Independent readback of new food outputs; summary only, no outcome values."""
from pathlib import Path
import hashlib,json,resource
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
Q=P/'data_qc/preprocessing_20261030_v1';B=Q/'food_qc_v1'
L=Path('/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1')
assert (L/'07_actual_exit_code').read_text().strip()=='0'
K=['participant_id','cohort','research_stage'];KD=K+['date']
receipt={'purpose':'independent source-to-flag and day-to-window readback',
         'sources':['food_events_clean_TRE_ONLY.parquet','food_daily_TRE_ONLY.parquet'],
         'scientific_Y_columns':[],'old_artifact_hash_check':True}
(B/'independent_preaccess_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
def read(p,cols=None):return pq.ParquetFile(p,pre_buffer=False).read(columns=cols).to_pandas(ignore_metadata=True)
flags=read(B/'event_flags_TRE_ONLY.parquet')
assert flags.source_row_ordinal.is_unique
source=P/'analysis/shared/food_events_clean_TRE_ONLY.parquet';offset=0
for rb in pq.ParquetFile(source,pre_buffer=False).iter_batches(columns=['source_row_ordinal','calories_kcal','weight_g'],batch_size=8192,use_threads=False):
    d=rb.to_pandas(ignore_metadata=True);f=flags.iloc[offset:offset+len(d)];offset+=len(d)
    assert np.array_equal(d.source_row_ordinal.to_numpy(),f.source_row_ordinal.to_numpy())
    e=pd.to_numeric(d.calories_kcal,errors='coerce').to_numpy(float);w=pd.to_numeric(d.weight_g,errors='coerce').to_numpy(float)
    be=np.isfinite(e)&((e<0)|(e>3000));bw=np.isfinite(w)&((w<0)|(w>2400))
    assert np.array_equal(be,f.event_energy_out_of_range.to_numpy())
    assert np.array_equal(bw,f.event_weight_out_of_range.to_numpy())
    assert np.array_equal(be|bw,f.event_excluded_new_QC.to_numpy())
assert offset==len(flags)
bad=flags.loc[flags.event_excluded_new_QC].groupby(KD).size().rename('independent_bad_n').reset_index()
old=read(P/'analysis/shared/food_daily_TRE_ONLY.parquet')
old=old.merge(bad,on=KD,how='left',validate='one_to_one');old.independent_bad_n=old.independent_bad_n.fillna(0)
expected=old.energy_eligible&old.independent_bad_n.eq(0)&old.energy_kcal.between(500,4000,inclusive='both')
new=read(B/'food_daily_flags_TRE_ONLY.parquet')
assert not new.duplicated(KD).any()
check=old[KD].assign(expected=expected).merge(new[KD+['energy_eligible']],on=KD,validate='one_to_one')
assert check.expected.equals(check.energy_eligible)
primary=read(B/'food_daily_primary_TRE_ONLY.parquet')
assert len(primary)==expected.sum() and not primary.duplicated(KD).any()
assert set(map(tuple,primary[KD].to_numpy()))==set(map(tuple,old.loc[expected,KD].to_numpy()))
windows=read(B/'food_windows_TRE_ONLY.parquet').set_index(K).sort_index()
assert windows.index.is_unique
g=primary.groupby(K)
independent=g.agg(n=('date','size'),lo=('UPF_lower_pp','mean'),hi=('UPF_upper_pp','mean'),
                  E=('energy_kcal','sum'),known=('known_UPF_kcal','sum'),unknown=('unknown_NOVA_kcal','sum'))
independent=independent.reindex(windows.index)
assert (independent.n==windows.n_energy_days).all()
for name,values in [('UPF_lower_equal_day_mean_all_energy_days',independent.lo),
                    ('UPF_upper_equal_day_mean_all_energy_days',independent.hi),
                    ('UPF_lower_cumulative_all_energy_days',100*independent.known/independent.E),
                    ('UPF_upper_cumulative_all_energy_days',100*(independent.known+independent.unknown)/independent.E)]:
    np.testing.assert_allclose(windows[name],values,rtol=1e-10,atol=1e-10,equal_nan=True)
fiber=primary.loc[primary.fiber_eligible].groupby(K).agg(n=('date','size'),mean=('fiber_density_g_per_1000kcal','mean'),F=('fiber_g','sum'),E=('energy_kcal','sum')).reindex(windows.index)
assert np.array_equal(fiber.n.fillna(0).to_numpy(),windows.n_fiber_days.to_numpy())
np.testing.assert_allclose(windows.fiber_equal_day_mean,fiber['mean'],equal_nan=True,rtol=1e-10)
np.testing.assert_allclose(windows.fiber_cumulative_ratio,1000*fiber.F/fiber.E,equal_nan=True,rtol=1e-10)
st=primary.groupby(K).energy_kcal.agg(['count','mean','std'])
tmp=new.merge(st,on=K,how='left',validate='many_to_one')
low=tmp.energy_eligible&tmp['count'].ge(3)&tmp['std'].gt(0)&tmp.energy_kcal.lt(tmp['mean']-2.5*tmp['std'])
assert low.equals(tmp.new_QC_personal_low25SD_flag)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
verified=0
for run in ['w13_microbiome_qc_20261029','w14_eligibility_20261029','w15_split_access_20261029']:
    for e in json.loads((P/'data_qc'/run/'final_artifact_manifest.json').read_text()):
        assert sha(e['path'])==e['sha256'];verified+=1
summary=json.loads((B/'summary_TRE_ONLY.json').read_text())
report={'status':'food_QC_independent_readback_passed','event_rows_checked':offset,
        'day_keys_checked':len(old),'primary_days':len(primary),'window_keys':len(windows),
        'original_W13_W14_W15_artifact_hashes_verified':verified,'real_scientific_Y_reads':0,
        'formal_W18_freeze':False,'real_parameter_fitting_started':False}
(B/'independent_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print('FOOD_INDEPENDENT_VERIFICATION',json.dumps(report),flush=True)
print('FOOD_QC_COMPACT_SUMMARY',json.dumps({k:v for k,v in summary.items() if k not in ['rules','role_scenarios']}),flush=True)
print('STAGE_EXIT_STATUS',json.dumps({tag:(L/(tag+'_actual_exit_code')).read_text().strip() if (L/(tag+'_actual_exit_code')).exists() else 'not_finished' for tag in ['01','02','03','04','05','06','06b','07','08','09']}),flush=True)
