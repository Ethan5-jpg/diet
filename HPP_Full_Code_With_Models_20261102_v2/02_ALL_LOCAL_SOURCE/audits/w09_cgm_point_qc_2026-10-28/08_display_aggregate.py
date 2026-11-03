import json
from pathlib import Path
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');Q=P/'data_qc/w09_cgm_point_qc_20261028';L=Path('/home/ec2-user/multisystem_runtime_20261021/w09_cgm_point_qc_20261028')
s=json.loads((Q/'summary_TRE_ONLY.json').read_text());w=json.loads((Q/'w08_common_summary_TRE_ONLY.json').read_text());v=json.loads((Q/'verification.json').read_text());wv=json.loads((Q/'w08_common_verification.json').read_text());assert v['verified'] and wv['verified']
print('ACTUAL_EXITS',[(x,(L/(x+'_actual_exit_code')).read_text()) for x in ['01','03b','09','04b','05','06','07']]);print('COUNT_COLUMNS people days people_visits ge7_DIAGNOSTIC median_days')
for label,x in s['counts']['baseline_10k_selected'].items():print('CGM_BASE',label,x['people'],x['days'],x['people_visits'],x['visits_ge7_DIAGNOSTIC'],x['median_days'])
for label,x in w['counts']['baseline_10k'].items():
 if isinstance(x,dict):print('COMMON_BASE',label,x['people'],x['days'],x['people_visits'],x['visits_ge7_DIAGNOSTIC'],x['median_days'])
 else:print('COMMON_BASE',label,x)
print('POINTS',json.dumps(s['point_accounting']));print('DAY_LENGTHS',json.dumps(s['all_day_lengths']));print('OFFGRID',s['offgrid_connections']);print('RECON',json.dumps(s['legacy_reconciliation']));print('PRECISION',json.dumps(s['precision_random20pct_mask_dense_days']));print('WINDOW_RECALC',json.dumps(w['window_recalculation_check']))
for x in s['baseline_sequential_CGM_day_flow']:print('FLOW',x['step'],x['removed_days_at_this_step'],x['remaining_days'],x['remaining_people'])
for x in s['multi_connection_aggregate']:print('MULTI',json.dumps(x))
print('CONNECTION_QC',json.dumps(s['baseline_connection_qc']));print('VERIFY',v['source_fragments_hash_verified'],v['independent_source_fragments'],v['independent_source_days'],v['mirrors_verified'],wv['verified']);print('SHARED_ROWS',[(Path(x['cloud']).name,x['rows']) for x in s['artifacts']+w['artifacts']])
print('READING_AGGREGATES_DONE')
