# -*- coding: utf-8 -*-
"""Repair date representation, not dates/values; reuse verified full-scan checkpoint."""
import pickle,json,hashlib,shutil
from pathlib import Path
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');Q=P/'data_qc/w09_cgm_point_qc_20261028';L=Path('/home/ec2-user/multisystem_runtime_20261021/w09_cgm_point_qc_20261028')
assert (L/'03b_actual_exit_code').read_text().strip()=='0';assert (L/'04_initial_actual_exit_code').read_text().strip()=='0'
old_summary=json.loads((Q/'summary_TRE_ONLY.json').read_text());source=P/'scripts/w09_cgm_point_qc_20261028/03b_build_point_qc_legacy_fix.py';code=source.read_text();cut=code.index('for ordinal,p in enumerate(parts):');prefix=code[:cut].replace("B=L/'artifacts_v2'","B=L/'artifacts_v3'")
# Prefix performs source/dictionary and legacy-column preflight; no point scan loop is executed here.
exec(compile(prefix,str(source)+'::contract_prefix','exec'),globals())
with (L/'completed_point_scan_v2_TRE_ONLY.pkl').open('rb') as f:cache=pickle.load(f)
C=cache['C'].drop(columns=['qualified_days','qualified_food_common_days','selected_connection']);D=cache['D'];manifest=cache['manifest'];pointcount=cache['counts'];intervals=cache['intervals'];masked=cache['masked'];curve_samples=cache['curve_samples']
assert len(C)==len(manifest)==9497 and C.connection_id.is_unique and int(D.n_points.sum())==pointcount['valid_unique_points']==old_summary['point_accounting']['valid_unique_points']
# W06 canonical date is ISO string, whereas our initial CGM output was Arrow date32.
assert str(pq.ParquetFile(P/'analysis/shared/food_daily_TRE_ONLY.parquet').schema_arrow.field('date').type)=='string'
food_date=pd.to_datetime(food.date,format='%Y-%m-%d',errors='raise').dt.strftime('%Y-%m-%d');assert food.date.eq(food_date).all()
D['date']=pd.to_datetime(D.date,errors='raise').dt.strftime('%Y-%m-%d');assert not D.duplicated(KC+['date']).any()
# Explicit independent key intersection must be non-empty before replacing own provisional outputs.
probe=D.merge(food.loc[food.energy_eligible,KD],on=KD,how='inner',validate='many_to_one');assert len(probe)>0
print('DATE_CONTRACT_CONFIRMED','food=ISO_string','CGM_normalized_to_same_ISO_string','potential_energy_common',len(probe),flush=True)
backup=Q/'before_calendar_contract_fix';backup.mkdir(exist_ok=False);arch=[]
paths=[P/'data_qc/cgm_point_connection_QC.parquet',Q/'cgm_daily_all_connections_TRE_ONLY.parquet',P/'analysis/shared/cgm_daily_TRE_ONLY.parquet',Q/'cgm_daily_reconciliation_TRE_ONLY.parquet',Q/'summary_TRE_ONLY.json',Q/'fragment_processing_manifest_TRE_ONLY.json',Q/'multi_connection_decisions_TRE_ONLY.json',Q/'fixed_curve_samples_TRE_ONLY.json',Q/'verification.json',Q/'fixed_curve_QC_TRE_ONLY.html',P/'data_qc/cgm_daily_reconciliation.csv',P/'config/cgm_point_daily_rules_W09_20261028.json']
for p in paths:
 if not p.exists():continue
 h=sha(p);dst=backup/p.name;assert not dst.exists();p.rename(dst);assert sha(dst)==h;arch.append({'original':str(p),'archive':str(dst),'sha256':h})
(backup/'archive_manifest.json').write_text(json.dumps(arch,indent=2))
# Replay only connection selection, reconciliation, QC summaries and output saving from our own source.
suffix=code[code.index("D['food_energy_day_present']"):]
suffix=suffix.replace("old['date']=pd.to_datetime(old.collection_date,utc=True).dt.tz_convert(TZ).dt.date","old['date']=pd.to_datetime(old.collection_date,utc=True).dt.tz_convert(TZ).dt.strftime('%Y-%m-%d')")
suffix=suffix.replace("completed_point_scan_v2_TRE_ONLY.pkl","completed_point_scan_v3_TRE_ONLY.pkl")
exec(compile(suffix,str(source)+'::rebuild_after_calendar_contract_fix','exec'),globals())
s=json.loads((Q/'summary_TRE_ONLY.json').read_text());s['date_contract_repair']={'food_date_type':'ISO YYYY-MM-DD string','initial_CGM_date_type':'date32[day]','final_CGM_date_type':'ISO YYYY-MM-DD string','calendar_values_changed':False,'full_point_scan_reused_from_own_checkpoint':True,'prior_artifacts_preserved':str(backup),'original_scan_elapsed_seconds':old_summary['elapsed_seconds']};(Q/'summary_TRE_ONLY.json').write_text(json.dumps(s,indent=2));assert s['counts']['baseline_10k_selected']['qualified_CGM_food_energy']['people']>0
print('CALENDAR_CONTRACT_FIX_DONE',json.dumps(s['counts']),flush=True)
