from pathlib import Path
import json,hashlib,sys,os,datetime,subprocess
import numpy as np,pandas as pd,pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');S=Path(__file__).parent;O=P/'data_qc/w20_targeted_20261101_v3/results/attempt49b';L=Path('/home/ec2-user/multisystem_runtime_20261021/w20_targeted_20261101_v3')
assert sys.executable=='/home/ec2-user/miniconda/envs/pheno/bin/python'
exits={t:int((L/(t+'_actual_exit_code')).read_text()) for t in ['47','48','49','49b','50b','51','51b','52']};assert all(exits[t]==0 for t in ['47','48','49b','50b','51b'])
v=json.loads((O/'verified_summary.json').read_text());assert v['verification']['status']=='verified'
st=pd.read_csv(O/'score_stability.csv');diag=st.groupby(['model','site']).agg(median_reference_correlation=('signed_full_reference_correlation','median'),min_reference_correlation=('signed_full_reference_correlation','min'),max_reference_correlation=('signed_full_reference_correlation','max'),min_fold_slope_SD=('SD_slope_per5','min'),max_fold_slope_SD=('SD_slope_per5','max')).reset_index().to_dict('records')
q=pq.ParquetFile(O/'development_predictions_and_scores_TRE_ONLY.parquet').read(columns=['participant_id','Y']).to_pandas(ignore_metadata=True);j=pq.ParquetFile(O/'daily_analysis_TRE_ONLY.parquet').read(columns=['participant_id']).to_pandas(ignore_metadata=True);withoutY=int(j.drop_duplicates().merge(q,on='participant_id',validate='one_to_one').Y.isna().sum())
pids={p.name:p.read_text().strip() for p in L.glob('*pid')};cmd=['ps','-p',','.join(sorted(set(pids.values()))),'-o','pid,ppid,etime,stat,args'];process=subprocess.run(cmd,capture_output=True,text=True)
result={'status':'sealed','actual_exits':exits,'pids':pids,'ps_stdout':process.stdout,'ps_status':process.returncode,'stability_descriptive_full_reference_overlap':diag,'full_slope_SD':v['models']['full_slope_SD'],'daily_people_without_personlevel_CV_Y':withoutY,'confirmation_Y_used':False,'formal55_tests_started':False,'seal_pid':os.getpid(),'time_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(O/'seal_summary_final.json').write_text(json.dumps(result,indent=2)+'\n')
with (O/'FINAL_REPORT.md').open('x') as f:f.write('# HPP targeted v3 verified final report\n\nExploratory observational association modification. Model and daily runs verified. Confirmation Y unused; formal 55 Holm tests not started.\n\n## Verified scientific aggregate results\n\n```json\n'+json.dumps(v,indent=2)+'\n```\n\n## Final seal\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n')
paths=[p for p in O.iterdir() if p.is_file() and p.name!='final_artifact_manifest_TRE_ONLY.json'];manifest=[{'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(paths)]
(O/'final_artifact_manifest_TRE_ONLY.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('SEALED',json.dumps({'exits':exits,'artifacts':len(manifest),'seal_pid':os.getpid(),'daily_people_without_personlevel_CV_Y':withoutY,'verification_pid':v['verification']['verification_pid']}))
for d in diag:print('STABILITY',d['model'],d['site'],'median/range',round(d['median_reference_correlation'],6),round(d['min_reference_correlation'],6),round(d['max_reference_correlation'],6),'slopeSD',round(d['min_fold_slope_SD'],6),round(d['max_fold_slope_SD'],6))
for d in v['models']['full_slope_SD']:print('FULL_SLOPE_SD',d['model'],d['site'],d['SD_per_fiber5'],d['Y_unit'])
print('PS recorded previous workers:',process.stdout.strip());print('REPORT',str(O/'FINAL_REPORT.md'));print('MANIFEST',str(O/'final_artifact_manifest_TRE_ONLY.json'))
