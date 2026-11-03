"""Outcome-blind additional food QC. Preserve source and old W06-W15 artifacts."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,importlib.util,json,resource,time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
resource.setrlimit(resource.RLIMIT_CPU,(600,600))
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
Q=P/'data_qc/preprocessing_20261030_v1'
L=Path('/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1')
B=Q/'food_qc_v1';B.mkdir(exist_ok=False)
K=['participant_id','cohort','research_stage'];KD=K+['date']
rules={'version':'food_QC_amendment_20261030_v1','outcome_blind':True,
       'event_energy_exclusion':'raw calories_kcal <0 or >3000; endpoints retained',
       'event_weight_exclusion':'raw weight_g <0 or >2400; endpoints retained',
       'weight_missing':'flag only; no zero imputation or inferred serving multiplier',
       'affected_event_day':'quarantine entire day for primary diet; dropping bad event cannot prove remaining intake complete',
       'day_energy_exclusion':'complete original day energy <500 or >4000; endpoints retained',
       'personal_low_day':'flag only if energy < mean-2.5*sample_SD among otherwise admitted days; n>=3, n<7 marked unstable',
       'primary_low_day_exclusion':False,'minimum_days':'candidates 1,3,7; W18 not yet frozen',
       'UPF':'preserve inherited NOVA and lower/upper bounds; never midpoint or unknown=non-UPF',
       'nutrient_source':'already event intake amounts, multiplier 1; weight only plausibility check',
       'source_tables_unchanged':True,'scientific_Y_columns':[]}
(B/'rules.json').write_text(json.dumps(rules,indent=2)+'\n')
assert 0<=3000<=3000 and 0<=2400<=2400
assert not (500<500 or 4000>4000)
paths=[P/'analysis/shared/food_events_clean_TRE_ONLY.parquet',P/'analysis/shared/food_daily_TRE_ONLY.parquet',
       P/'analysis/shared/food_windows_TRE_ONLY.parquet']
fingerprints={str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in paths}
columns=KD+['source_row_ordinal','calories_kcal','weight_g','calories_kcal_clean']
(B/'preaccess_receipt.json').write_text(json.dumps({'purpose':'authorized additional food QC and blind design counts',
    'source_columns':columns,'real_scientific_Y_reads':0,'rules_sha256':hashlib.sha256((B/'rules.json').read_bytes()).hexdigest()},indent=2)+'\n')
bad_days=defaultdict(Counter);counts=Counter();writer=None
for rb in pq.ParquetFile(paths[0],pre_buffer=False).iter_batches(columns=columns,batch_size=8192,use_threads=False):
    d=rb.to_pandas(ignore_metadata=True)
    e=pd.to_numeric(d.calories_kcal,errors='coerce').to_numpy(float)
    w=pd.to_numeric(d.weight_g,errors='coerce').to_numpy(float)
    badE=np.isfinite(e)&((e<0)|(e>3000));badW=np.isfinite(w)&((w<0)|(w>2400))
    d['event_energy_out_of_range']=badE;d['event_weight_out_of_range']=badW
    d['event_weight_nonfinite']=~np.isfinite(w);d['event_excluded_new_QC']=badE|badW
    counts['events_scanned']+=len(d)
    for name in ['event_energy_out_of_range','event_weight_out_of_range','event_weight_nonfinite','event_excluded_new_QC']:
        counts[name]+=int(d[name].sum())
    for key,mask in zip(d[KD].itertuples(index=False,name=None),badE|badW):
        if mask:bad_days[key]['bad_event_count']+=1
    flags=d[KD+['source_row_ordinal','event_energy_out_of_range','event_weight_out_of_range','event_weight_nonfinite','event_excluded_new_QC']]
    table=pa.Table.from_pandas(flags,preserve_index=False)
    if writer is None:writer=pq.ParquetWriter(B/'event_flags_TRE_ONLY.parquet',table.schema,compression='zstd')
    writer.write_table(table)
writer.close()
D=pq.ParquetFile(paths[1],pre_buffer=False).read().to_pandas(ignore_metadata=True)
assert not D.duplicated(KD).any()
D['new_QC_bad_event_count']=[bad_days.get(k,{}).get('bad_event_count',0) for k in D[KD].itertuples(index=False,name=None)]
D['new_QC_bad_event_day']=D.new_QC_bad_event_count.gt(0)
D['new_QC_energy_below500']=D.energy_kcal.lt(500)
D['new_QC_energy_above4000']=D.energy_kcal.gt(4000)
admit=D.energy_eligible&~D.new_QC_bad_event_day&~D.new_QC_energy_below500&~D.new_QC_energy_above4000
D['energy_eligible_before_amendment']=D.energy_eligible
D['fiber_eligible_before_amendment']=D.fiber_eligible
D['energy_eligible']=admit;D['fiber_eligible']=admit&D.fiber_eligible_before_amendment
stats=D.loc[admit].groupby(K).energy_kcal.agg(['count','mean','std']).rename(columns={'count':'personal_energy_day_n','mean':'personal_energy_mean','std':'personal_energy_sample_SD'})
D=D.merge(stats,on=K,how='left',validate='many_to_one')
D['new_QC_personal_low25SD_flag']=admit.to_numpy()&D.personal_energy_day_n.ge(3)&D.personal_energy_sample_SD.gt(0)&D.energy_kcal.lt(D.personal_energy_mean-2.5*D.personal_energy_sample_SD)
D['personal_low_day_rule_unstable']=D.personal_energy_day_n.lt(7)|D.personal_energy_day_n.isna()
# Source amounts and original derived quantities remain intact in audit table.
pq.write_table(pa.Table.from_pandas(D,preserve_index=False),B/'food_daily_flags_TRE_ONLY.parquet',compression='zstd')
selected=D.loc[admit].copy()
# The inherited helper needs candidate exact-point/fiber columns masked separately.
for name in ['UPF_point_zero_unknown_energy_pp','UPF_point_all_rows_classified_pp']:
    selected[name]=selected[name].where(selected.energy_eligible)
selected['fiber_density_g_per_1000kcal']=selected.fiber_density_g_per_1000kcal.where(selected.fiber_eligible)
selected['fiber_g']=selected.fiber_g.where(selected.fiber_eligible)
adapter=P/'scripts/w07_w08_diet_views_20261028/common_day_adapter.py'
assert adapter.exists()
sp=importlib.util.spec_from_file_location('inherited_food_window_adapter',adapter)
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
W=m.window(selected)
assert not W.duplicated(K).any() and int(W.n_recorded_days.sum())==len(selected)
assert selected.energy_kcal.between(500,4000).all() and not selected.new_QC_bad_event_day.any()
assert np.all(selected.UPF_lower_pp<=selected.UPF_upper_pp+1e-8)
pq.write_table(pa.Table.from_pandas(selected,preserve_index=False),B/'food_daily_primary_TRE_ONLY.parquet',compression='zstd')
pq.write_table(pa.Table.from_pandas(W,preserve_index=False),B/'food_windows_TRE_ONLY.parquet',compression='zstd')
def desc(x):
    v=pd.to_numeric(x,errors='coerce').dropna()
    return {'n':len(v),'quantiles':{str(q):float(v.quantile(q)) for q in [0,.1,.25,.5,.75,.9,1]}} if len(v) else {'n':0}
baseline=(D.cohort=='10k')&(D.research_stage=='00_00_visit')
z=D.loc[baseline];w=W.loc[(W.cohort=='10k')&(W.research_stage=='00_00_visit')]
summary={'rules':rules,'event_counts':dict(counts),'baseline_days_before':len(z),
         'baseline_energy_days_before':int(z.energy_eligible_before_amendment.sum()),
         'baseline_energy_days_after':int(z.energy_eligible.sum()),'baseline_fiber_days_after':int(z.fiber_eligible.sum()),
         'baseline_bad_event_days':int(z.new_QC_bad_event_day.sum()),
         'baseline_energy_gt4000_days':int(z.new_QC_energy_above4000.sum()),
         'baseline_energy_lt500_days':int(z.new_QC_energy_below500.sum()),
         'baseline_personal_low_flag_days':int(z.new_QC_personal_low25SD_flag.sum()),
         'baseline_UPF_bound_width_pp_by_min_days':{},'role_scenarios':[],'real_scientific_Y_reads':0}
for nd in [1,3,7]:
    a=w.loc[w.n_energy_days.ge(nd)]
    summary['baseline_UPF_bound_width_pp_by_min_days'][str(nd)]=desc(a.UPF_upper_equal_day_mean_all_energy_days-a.UPF_lower_equal_day_mean_all_energy_days)
# Membership-only reads through the published W15 guard. No Y projection.
manifest=json.loads((P/'data_qc/w15_split_access_20261029/final_artifact_manifest.json').read_text())
gp=[Path(x['path']) for x in manifest if x['path'].endswith('/w15_access_guard.py')];assert len(gp)==1
sp=importlib.util.spec_from_file_location('W15_guard_food_QC',gp[0]);g=importlib.util.module_from_spec(sp);sp.loader.exec_module(g)
active=json.loads((P/'config/ACTIVE_PROTOCOL.json').read_text());register=json.loads(Path(active['analysis_register']).read_text())
for mod in register['modules']:
    name=mod['analysis_id']
    if name.startswith('P_') or name=='Common4':continue
    members=g.load_blind_membership(name,K+['analysis_id','split_unit_id','unit_order','person_order'])
    joined=members.merge(w,on=K,how='left',validate='many_to_one')
    for nd in [1,3,7]:
        for fraction in [.5,.6,.7,.8]:
            discovery=joined.unit_order.le(int(np.floor(5699*fraction)))
            for exposure,eligible in [('UPF',joined.n_energy_days.ge(nd)),('fiber',joined.n_fiber_days.ge(nd))]:
                for role,mask in [('D',discovery),('C',~discovery)]:
                    a=joined.loc[mask&eligible]
                    summary['role_scenarios'].append({'analysis_id':name,'fraction':fraction,'minimum_days':nd,
                        'exposure':exposure,'role':role,'people':len(a),'units':int(a.split_unit_id.nunique())})
for p,old in fingerprints.items():
    assert [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]==old
assert pq.ParquetFile(B/'event_flags_TRE_ONLY.parquet').metadata.num_rows==counts['events_scanned']
assert pq.ParquetFile(B/'food_windows_TRE_ONLY.parquet').metadata.num_rows==len(W)
(B/'summary_TRE_ONLY.json').write_text(json.dumps(summary,indent=2)+'\n')
print('FOOD_QC_PASS',json.dumps({k:v for k,v in summary.items() if k not in ['rules','role_scenarios']}),flush=True)
print('FOOD_BLIND_DESIGN_COUNTS',json.dumps([x for x in summary['role_scenarios'] if x['fraction'] in [.5,.6] and x['minimum_days'] in [3,7] and x['analysis_id'] in ['D_Glu','D_TotalFat','D_VAT','D_Liver_elasticity']]),flush=True)
