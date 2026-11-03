"""Check full-scan outputs, make TRE-only real-date examples, save final memo."""
from pathlib import Path
import json, hashlib, sys, html, shutil
import pandas as pd

OUT=Path('/home/ec2-user/studies/diet_gut_cgm_v2/phenotype_food_dates_20261018')
V2=OUT.parent
KEY=['participant_id','cohort','research_stage']
assert '/envs/pheno/' in sys.executable
assert (OUT/'11_exit_code').read_text().strip()=='0'
s=json.loads((OUT/'summary.json').read_text());m=json.loads((OUT/'source_manifest.json').read_text())
assert all(x['rows_metadata']==x['rows_scanned'] for x in m)
assert hashlib.sha256((OUT/'11_audit.py').read_bytes()).hexdigest()==s['source_sha256']
d=pd.read_csv(OUT/'daily_food_CGM_TRE_ONLY.csv',usecols=KEY+['date','cgm_observed','food_event_rows'])
p=pd.read_csv(OUT/'person_visit_all_dates_TRE_ONLY.csv')
assert not d.duplicated(KEY+['date']).any()
assert not p.duplicated(KEY).any()
assert len(d)==s['food_daily']['person_days']
assert int(d.food_event_rows.sum())==next(x['baseline_event_rows'] for x in m if x['role']=='diet')
c=d.groupby(KEY,observed=True).cgm_observed.sum().rename('from_daily')
check=p.set_index(KEY).join(c,validate='one_to_one')
assert check.from_daily.eq(check.common_days).all()
source=Path(__file__).read_bytes();(OUT/'12_verify_report.py').write_bytes(source)
for name,src in [('10_console.log','/tmp/hpp_10_discover.log'),('11_console.log','/tmp/hpp_11_audit.log')]:
    shutil.copyfile(src,OUT/name)
labels={'anthropometrics':'人体测量（体重/BMI/腰围）','body_composition':'DXA体成分','liver_ultrasound':'肝脏超声','oral':'口腔菌群','gut':'肠道菌群'}
rows=[]
for role,label in labels.items():
    x=s['timing'][role]['diet'];y=s['timing'][role]['cgm']
    rows.append({'数据':label,'有日期人数':s['people'][role],'与饮食同人同访视':x['same_visit'],'其中测量日期唯一':x['unique_point_date'],'距最近饮食日中位天数':x['nearest_observed_day_abs'].get('median'),'饮食期前':x['before_span'],'饮食期内':x['inside_span'],'饮食期后':x['after_span'],'距饮食日≤30天':x['within_days']['30'],'距最近CGM日中位天数':y['nearest_observed_day_abs'].get('median')})
t=pd.DataFrame(rows);t.to_csv(OUT/'aggregate_timing_table.csv',index=False)
parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>HPP日期与食物逐日表</title><style>body{font:16px system-ui;max-width:1300px;margin:36px auto;padding:20px;color:#182638}table{border-collapse:collapse;font-size:14px;width:100%;margin:18px 0}th,td{border:1px solid #cdd7e2;padding:8px;text-align:left}th{background:#edf3f8}p{line-height:1.7}.note{background:#fff2d7;padding:14px}</style><h1>HPP实际日期与逐日食物记录核查</h1><p class="note">本文件含真实个体日期示例，仅在TRE内使用。汇总限定10k基线访视；日期可配不代表饮食完整、餐次正确或CGM质量合格。</p>',t.to_html(index=False,border=0),'<h2>实际个体示例：按固定编号排序选取，不代表典型受试者</h2>']
eligible=p.loc[p.common_days.ge(7)].copy()
for role in labels:eligible=eligible.loc[eligible[role+'_distinct_dates'].eq(1)]
selected=eligible.sort_values(KEY).head(5)
for i,(_,r) in enumerate(selected.iterrows(),1):
    data=[{'数据':'饮食','实际日期/范围':str(r.diet_first)+' 至 '+str(r.diet_last),'有记录天数':int(r.diet_days)},{'数据':'CGM','实际日期/范围':str(r.cgm_first)+' 至 '+str(r.cgm_last),'有记录天数':int(r.cgm_days)},{'数据':'共同日期','实际日期/范围':'按实际日期求交集，范围内可能缺日','有记录天数':int(r.common_days)}]
    data.extend({'数据':label,'实际日期/范围':str(r[role+'_dates']),'有记录天数':1} for role,label in labels.items())
    parts+=['<h3>示例 '+str(i)+'</h3>',pd.DataFrame(data).to_html(index=False,border=0)]
    if i==1:
        # Display a genuine day-by-day table, including gaps and CGM-only days.
        q=d.loc[(d[KEY]==r[KEY]).all(axis=1)].copy()
        span=pd.date_range(min(pd.Timestamp(r.diet_first),pd.Timestamp(r.cgm_first)),max(pd.Timestamp(r.diet_last),pd.Timestamp(r.cgm_last)),freq='D')
        if len(span)<=90:
            daytable=pd.DataFrame({'date':span.strftime('%Y-%m-%d')}).merge(q[['date','food_event_rows','cgm_observed']],on='date',how='left',validate='one_to_one')
            daytable['有饮食记录']=daytable.food_event_rows.notna()
            # CGM dates outside food days are unavailable in this derived CSV;
            # leave them unverified, never infer presence from the span.
            daytable['CGM当日有记录']=daytable.cgm_observed.map({1.0:'是',0.0:'否'}).fillna('本派生表未核实')
            daytable=daytable.rename(columns={'date':'真实日期','food_event_rows':'食物事件条数'})[['真实日期','有饮食记录','食物事件条数','CGM当日有记录']]
            parts+=['<p>示例1逐日表：无饮食记录日不能从本食物日表推断CGM是否存在，已明确标为未核实。</p>',daytable.to_html(index=False,border=0,na_rep='—')]
parts+=['<h2>逐日食物文件</h2><p>daily_food_CGM_TRE_ONLY.csv：一人、一次访视、一个有食物记录的日期一行；包含食物事件条数、已记录能量及营养素合计、非缺失计数和CGM当日观测标记。它不是每餐表，也未验证一天是否记录完整。原始事件表另有食物名、食物ID、重量和时间。</p></html>']
b=''.join(parts).encode();(OUT/'TABLES_TRE_ONLY.html').write_bytes(b);assert (OUT/'TABLES_TRE_ONLY.html').read_bytes()==b
note='\n## 12_verify_report.py与最终验收\n\n核对8张输入完整扫描、11版源码SHA256、一人一访视一日唯一性、逐日事件条数与扫描总数，以及由逐日表重新统计的共同日期数逐人一致。生成aggregate_timing_table.csv与TABLES_TRE_ONLY.html（真实个体示例仅留TRE）。同一时间字段不存在时不推测；例表中无食物日的CGM存在性标为未核实。运行日志已完整保存。12_exit_code=0。\n'
for path in [OUT/'MEMO.md',V2/'SERVER_BACKGROUND.md']:
    old=path.read_text();b=(old+note+('本次核查目录：'+str(OUT)+'\n')).encode();path.write_bytes(b);assert path.read_bytes()==b
context=V2/'PROJECT_CONTEXT.json';obj=json.loads(context.read_text());obj['latest_phenotype_food_audit']=str(OUT);obj['latest_phenotype_food_audit_status']='COMPLETE_VERIFIED';obj['old_results_migrated']=False
b=json.dumps(obj,indent=2).encode();context.write_bytes(b);assert context.read_bytes()==b
status={'status':'PHENOTYPE_FOOD_DATES_VERIFIED','input_full_scans':len(m),'daily_rows':len(d),'person_visit_rows':len(p),'per_person_overlap_recheck':'PASS','source11_sha256':s['source_sha256'],'source12_sha256':hashlib.sha256(source).hexdigest(),'individual_tables':'TRE_ONLY','old_audit_migrated':False,'12_exit_code':0}
b=json.dumps(status,indent=2).encode();(OUT/'verification.json').write_bytes(b);assert (OUT/'verification.json').read_bytes()==b
(OUT/'12_exit_code').write_text('0\n')
print('PHENOTYPE_FOOD_DATES_VERIFIED')
print('INPUT_FULL_SCANS',len(m),'DAILY_ROWS',len(d),'PERSON_VISIT_ROWS',len(p))
print('UNIQUE_KEYS / EVENT_TOTALS / PER_PERSON_COMMON_DAYS_RECHECK: PASS')
print('TRE_TABLES: daily_food_CGM_TRE_ONLY.csv / person_visit_all_dates_TRE_ONLY.csv / TABLES_TRE_ONLY.html')
print('AGGREGATE_REPORT: REPORT.md / aggregate_timing_table.csv / summary.json')
print('MEMO_AND_BACKGROUND_FULL_WRITE_READBACK: PASS')
print('OUTPUT_DIR',OUT)
print('VERIFICATION_EXIT_CODE_0')
seven=p.loc[p.common_days.ge(7)].copy()
for role in labels:seven=seven.loc[seven[role+'_distinct_dates'].ge(1)]
print('ALL_SEVEN_SOURCES_WITH_COMMON_DAYS_7',len(seven))
for rec in m:
    print('FIELD_QC',rec['role'],'DATE_CONFLICTS',rec['primary_vs_collection_date_conflicts'],'LOCAL_DATE_CONFLICTS',rec['local_date_conflicts'],'FEATURE_PEOPLE',json.dumps(rec['feature_nonmissing_baseline_person_visits']))
disc=json.loads((OUT/'discovery.json').read_text())
for dd in disc['selected']['diet_logging']['dictionaries']:
    for r in dd['matching_rows']:
        if any(v in ['weight_g','calories_kcal','carbohydrate_g','protein_g','lipid_g','dietary_fiber_g','local_timestamp','collection_timestamp'] for v in r.values()):
            print('FOOD_DEFINITION',json.dumps({k:v for k,v in r.items() if 'name' in k or 'description' in k or k=='units'},ensure_ascii=False))
