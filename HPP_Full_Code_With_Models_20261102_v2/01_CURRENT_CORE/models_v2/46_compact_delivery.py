import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,hashlib,datetime
import pandas as pd
from matched_oracle_control import evaluate
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');B=P/'data_qc/w20_simplified_20261101_v2';O=B/'results';L=Path('/home/ec2-user/multisystem_runtime_20261021/w20_simplified_20261101_v2')
assert (L/'45_actual_exit_code').read_text().strip()=='0'
status=json.loads((O/'delivery/verified_status.json').read_text());df=pd.read_csv(O/'discovery_comparison.csv',keep_default_na=False);ctrl=pd.read_csv(O/'synthetic_controls.csv');st=pd.read_csv(O/'score_stability.csv')
oracle=evaluate(B,O)
C=P/'config/w20_simplified_20261101_v2';C.mkdir(exist_ok=False)
pointer={'version':'W19_W20_simplified_20261101_v2','status':'discovery_complete_verified_not_confirmation_opened','roles':str(B/'roles_v2_full_TRE_ONLY.parquet'),'roles_sha256':hashlib.sha256((B/'roles_v2_full_TRE_ONLY.parquet').read_bytes()).hexdigest(),'execution_spec':str(B/'execution_spec.json'),'execution_spec_sha256':hashlib.sha256((B/'execution_spec.json').read_bytes()).hexdigest(),'verified_delivery':str(O/'delivery/verified_status.json'),'prior_W18_active_pointer_preserved':True,'next':'W22 global support/rank/dictionary freeze before W23; no confirmation tests in this run'}
(C/'active_execution.json').write_text(json.dumps(pointer,indent=2)+'\n')
print('V2_DELIVERY: actual exits41/42/43/44/45='+','.join((L/(t+'_actual_exit_code')).read_text().strip() for t in ['41','42','43','44','45']))
print('ROLES discovery=4427 confirmation=1878; families3989/1710; added1250; confirmation Y reads0')
print('VERIFIED scores',status['score_formula_readbacks'],'CV metrics',status['CV_metric_readbacks'],'predictors',status['prediction_formula_readbacks'],'protected v1 files',status['protected_v1_files_unchanged'])
print('Development CV only; gain vs matched full-CLR-main null; formal P values NOT computed')
print('endpoint                     diet        n     train_min/max  null_R2   int_R2    delta_R2     O_corr G_corr  lambda')
for r in df.itertuples():
 print(f'{r.analysis_id:28s} {r.exposure_id:13s} {r.n:4d} {r.training_n_min:4d}/{r.training_n_max:<4d} {r.null_R2:9.6f} {r.interaction_R2:9.6f} {r.delta_R2:+10.6f}  {r.oral_stability:.3f}  {r.gut_stability:.3f}  {r.interaction_lambda:g}')
print('CONTROL fraction / learned deltaR2 / oracle MATCHED deltaR2 (one seed, NOT power)')
for r in oracle:print(r['signal_residual_fraction'],f"{r['learned_delta_R2_vs_its_matched_null']:+.6f}",f"{r['oracle_delta_R2_vs_its_matched_null']:+.6f}")
print('GAIN wins',int((df.delta_R2>0).sum()),'/18; maximum',df.delta_R2.max(),'median',df.delta_R2.median())
print('STABILITY full-reference descriptive median correlations min/max',st.median_absolute_full_reference_correlation.min(),st.median_absolute_full_reference_correlation.max())
print('DONE; v1 preserved; v2 config pointer separate; W22/W23 not started')
(O/'delivery/final_supplement_manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [O/'oracle_matched_control_clarification.csv',C/'active_execution.json',Path(__file__),Path(__file__).parent/'matched_oracle_control.py']},indent=2)+'\n')
