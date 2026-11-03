"""Actual-day food/CGM and point-measurement timing audit; identifiers stay in TRE.

Body timestamp is numeric: use its explicit collection_date, never infer epoch units.
This is availability/timing auditing, not completeness, meal or causal validation.
"""
from pathlib import Path
from collections import defaultdict, Counter
import sys, json, hashlib, resource, time, traceback, datetime, csv
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

ROOT=Path('/home/ec2-user/studies/hpp_datasets')
OUT=Path('/home/ec2-user/studies/diet_gut_cgm_v2/phenotype_food_dates_20261018')
KEY=['participant_id','cohort','research_stage']
TZ='Asia/Jerusalem'
NUT=['calories_kcal','carbohydrate_g','protein_g','lipid_g','dietary_fiber_g']
manifest=[]
daily={}

def save(name,obj):
    b=json.dumps(obj,indent=2,default=str,ensure_ascii=False).encode()
    p=OUT/name; p.write_bytes(b); assert p.read_bytes()==b

def stats(v):
    a=np.asarray(list(v),dtype=float)
    return dict(n=len(a),min=float(a.min()),p10=float(np.quantile(a,.1)),median=float(np.median(a)),p90=float(np.quantile(a,.9)),max=float(a.max())) if len(a) else {'n':0}

def norm(s):
    a=pd.to_datetime(s,errors='coerce')
    if isinstance(a.dtype,pd.DatetimeTZDtype): a=a.dt.tz_convert(TZ).dt.tz_localize(None)
    return a.dt.normalize()

def scan(module,name,role,features=()):
    p=ROOT/module/name; f=pq.ParquetFile(p,pre_buffer=False)
    sch={x.name:str(x.type) for x in f.schema_arrow}
    assert set(KEY)<=set(sch)
    primary='collection_timestamp' if role in ['diet','gut'] and 'tz=' in sch.get('collection_timestamp','') else 'collection_date'
    assert primary in sch
    cols=list(dict.fromkeys(KEY+[primary]+[c for c in ['collection_date','local_timestamp','logging_day'] if c in sch]+(['connection_id'] if role=='cgm' else [])+[c for c in features if c in sch]))
    projected=sum(f.metadata.row_group(i).column(j).total_uncompressed_size for i in range(f.metadata.num_row_groups) for j,n in enumerate(f.schema.names) if n in cols)
    assert projected<4*1024**3,'Projected columns exceed budget'
    print('SCAN_START',role,f.metadata.num_rows,flush=True)
    groups=defaultdict(set); conns=defaultdict(set); present={c:set() for c in features if c in sch}
    seen=bad=dates_missing=date_conflict=compared=local_conflict=local_compared=0
    valid_times=nonmidnight=baseline_events=0
    start=time.monotonic()
    for batch in f.iter_batches(batch_size=16384,columns=cols,use_threads=False):
        avail=int(next(s.split()[1] for s in open('/proc/meminfo') if s.startswith('MemAvailable:')))*1024
        assert avail>2*1024**3 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1536*1024,'Memory budget stop'
        assert time.monotonic()-start<1200,'Time budget stop; incomplete scans not accepted'
        d=batch.to_pandas(ignore_metadata=True); seen+=len(d)
        good=d[KEY].notna().all(axis=1)
        for c in KEY[1:]: good &= ~d[c].astype(str).str.strip().str.lower().isin(['','undefined','none','null','nan'])
        bad+=int((~good).sum())
        day=norm(d[primary]); dates_missing+=int(day.isna().sum())
        if primary!='collection_date' and 'collection_date' in d:
            other=norm(d.collection_date); ok=day.notna()&other.notna(); compared+=int(ok.sum()); date_conflict+=int((day[ok]!=other[ok]).sum())
        if 'local_timestamp' in d:
            other=norm(d.local_timestamp); ok=day.notna()&other.notna(); local_compared+=int(ok.sum()); local_conflict+=int((day[ok]!=other[ok]).sum())
        keep=good&day.notna()&d.cohort.eq('10k')&d.research_stage.eq('00_00_visit')
        q=d.loc[keep].copy(); q['day']=day[keep]
        for row in q[KEY+['day']].drop_duplicates().itertuples(index=False,name=None):
            k=(int(row[0]),str(row[1]),str(row[2])); groups[k].add(pd.Timestamp(row[3]).date())
        for c in present:
            for row in q.loc[q[c].notna(),KEY].drop_duplicates().itertuples(index=False,name=None): present[c].add((int(row[0]),str(row[1]),str(row[2])))
        if role=='cgm':
            for row in q.dropna(subset=['connection_id'])[KEY+['day','connection_id']].drop_duplicates().itertuples(index=False,name=None):
                conns[(int(row[0]),str(row[1]),str(row[2]),str(row[4]))].add(pd.Timestamp(row[3]).date())
        if role=='diet':
            baseline_events+=len(q)
            ts=pd.to_datetime(q[primary],errors='coerce')
            if isinstance(ts.dtype,pd.DatetimeTZDtype): ts=ts.dt.tz_convert(TZ).dt.tz_localize(None)
            valid_times+=int(ts.notna().sum()); nonmidnight+=int((ts.notna()&ts.ne(ts.dt.normalize())).sum())
            g=q.groupby(KEY+['day'],observed=True,sort=False)
            counts=g.size(); sums=g[NUT].sum(min_count=1); ns=g[NUT].count()
            for ix,n in counts.items():
                k=(int(ix[0]),str(ix[1]),str(ix[2]),pd.Timestamp(ix[3]).date())
                if k not in daily: daily[k]={'food_event_rows':0,**{c+'_sum':0.0 for c in NUT},**{c+'_nonnull':0 for c in NUT}}
                z=daily[k]; z['food_event_rows']+=int(n)
                for c in NUT:
                    nn=int(ns.loc[ix,c]); z[c+'_nonnull']+=nn
                    if nn: z[c+'_sum']+=float(sums.loc[ix,c])
    assert seen==f.metadata.num_rows
    rec=dict(role=role,path=str(p),rows_metadata=f.metadata.num_rows,rows_scanned=seen,bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,primary_date=primary,projected_columns=cols,projected_uncompressed_bytes=projected,invalid_key_rows=bad,missing_date_rows=dates_missing,primary_vs_collection_date_compared=compared,primary_vs_collection_date_conflicts=date_conflict,local_date_compared=local_compared,local_date_conflicts=local_conflict,baseline_person_visits=len(groups),multiple_distinct_dates=sum(len(v)>1 for v in groups.values()),feature_nonmissing_baseline_person_visits={c:len(v) for c,v in present.items()})
    if role=='diet': rec.update(baseline_event_rows=baseline_events,timestamp_valid_rows=valid_times,nonmidnight_timestamp_rows=nonmidnight)
    manifest.append(rec)
    print('SCAN_DONE',role,'BASELINE_PEOPLE',len(groups),'MULTI_DATE_KEYS',rec['multiple_distinct_dates'],flush=True)
    return dict(groups),dict(conns)

def gap(a,b): return min(abs((x-y).days) for x in a for y in b)

def pairing(a,b):
    keys=set(a)&set(b)
    unique={k for k in keys if len(a[k])==1}
    vals=[gap(a[k],b[k]) for k in unique]
    signed=[]; before=inside=after=0
    for k in unique:
        d=next(iter(a[k])); near=min(b[k],key=lambda x:(abs((d-x).days),x)); signed.append((d-near).days)
        before+=d<min(b[k]); after+=d>max(b[k]); inside+=min(b[k])<=d<=max(b[k])
    return dict(same_visit=len(keys),unique_point_date=len(unique),ambiguous_point_dates=len(keys)-len(unique),nearest_observed_day_abs=stats(vals),signed_nearest_day=stats(signed),before_span=before,inside_span=inside,after_span=after,within_days={str(t):sum(v<=t for v in vals) for t in [0,7,14,30,60,90]})

def main():
    assert '/envs/pheno/' in sys.executable
    assert gap({datetime.date(2025,1,1)},{datetime.date(2025,1,3)})==2
    assert norm(pd.Series(pd.to_datetime(['2025-01-01T22:30:00Z'],utc=True))).iloc[0]==pd.Timestamp('2025-01-02')
    source=Path(__file__).read_bytes(); (OUT/'11_audit.py').write_bytes(source); assert (OUT/'11_audit.py').read_bytes()==source
    maps={}
    maps['diet'],_=scan('diet_logging','diet_logging_events.parquet','diet',NUT+['food_id'])
    maps['diet_daily_index'],_=scan('diet_logging','diet_logging.parquet','diet_daily_index')
    maps['cgm_union'],conns=scan('cgm','iglu_daily.parquet','cgm')
    maps['anthropometrics'],_=scan('anthropometrics','anthropometrics.parquet','anthropometrics',['weight','bmi','waist_circumference'])
    maps['body_composition'],_=scan('body_composition','body_composition.parquet','body_composition',['body_comp_total_tissue_percent_fat','total_scan_vat_mass','total_scan_vat_volume'])
    maps['liver_ultrasound'],_=scan('liver_ultrasound','liver_ultrasound.parquet','liver_ultrasound',['elasticity_qbox_median','attenuation_coefficient_qbox','speed_of_sound_qbox'])
    maps['oral'],_=scan('oral_microbiome','oral_microbiome.parquet','oral')
    maps['gut'],_=scan('gut_microbiome','gut_microbiome.parquet','gut')
    choices=defaultdict(list)
    for k,ds in conns.items(): choices[k[:3]].append((k[3],ds))
    chosen={}; chosen_id={}
    for k in set(maps['diet'])&set(maps['cgm_union']):
        opts=sorted(choices[k],key=lambda x:(-len(x[1]&maps['diet'][k]),min(x[1]),x[0]))
        if opts: chosen_id[k],chosen[k]=opts[0]
    maps['cgm']=chosen
    overlaps={k:len(maps['diet'][k]&v) for k,v in chosen.items()}
    same=set(maps['diet'])&set(maps['diet_daily_index'])
    dicts=json.loads((OUT/'discovery.json').read_text())
    excerpts=[]
    for m,v in dicts['selected'].items():
        for dd in v['dictionaries']:
            for r in dd['matching_rows']:
                if any(t in r.values() for t in ['collection_date','collection_timestamp','local_timestamp','logging_day','total_logging_days','weight_g','calories_kcal']): excerpts.append({'module':m,'dictionary_path':dd['path'],'row':r})
    save('date_and_food_dictionary_excerpts.json',excerpts)
    # Aggregated ultrasound table has no dates; recover only from unique-date raw keys.
    ap=ROOT/'liver_ultrasound/liver_ultrasound_aggregated.parquet'; af=pq.ParquetFile(ap,pre_buffer=False)
    agg=set(); aseen=0
    for b in af.iter_batches(columns=KEY,batch_size=8192,use_threads=False):
        d=b.to_pandas(ignore_metadata=True); aseen+=len(d); d=d.loc[d.cohort.eq('10k')&d.research_stage.eq('00_00_visit')].dropna(subset=KEY)
        agg.update((int(x[0]),str(x[1]),str(x[2])) for x in d.itertuples(index=False,name=None))
    assert aseen==af.metadata.num_rows
    body_roles=['anthropometrics','body_composition','liver_ultrasound','oral','gut']
    summary=dict(scope='10k / 00_00_visit; same-visit actual-date availability, no causal or quality eligibility',python=sys.executable,source_sha256=hashlib.sha256(source).hexdigest(),people={r:len(v) for r,v in maps.items()},diet_cgm=dict(same_visit=len(chosen),common_days=stats(overlaps.values()),at_least={str(t):sum(x>=t for x in overlaps.values()) for t in [1,5,7,10,14]}),diet_daily_index=dict(paired_people=len(same),identical_date_sets=sum(maps['diet'][k]==maps['diet_daily_index'][k] for k in same),event_dates_not_in_index=sum(len(maps['diet'][k]-maps['diet_daily_index'][k]) for k in same),index_dates_without_events=sum(len(maps['diet_daily_index'][k]-maps['diet'][k]) for k in same)),food_daily=dict(person_days=len(daily),recorded_days_per_person=stats(len(v) for v in maps['diet'].values()),event_rows_per_person_day=stats(v['food_event_rows'] for v in daily.values()),definition='food-event rows are not meals; nutrient sums are recorded intake, not verified complete intake'),timing={},liver_aggregate=dict(path=str(ap),rows=aseen,baseline_people=len(agg),keys_with_raw_dates=len(agg&set(maps['liver_ultrasound'])),keys_with_unique_raw_date=sum(len(maps['liver_ultrasound'][k])==1 for k in agg&set(maps['liver_ultrasound'])),date_recovery_rule='use raw date only if one distinct collection_date per person/cohort/stage; do not arbitrarily collapse multiple dates'))
    for role in body_roles:
        summary['timing'][role]={target:pairing(maps[role],maps[target]) for target in ['diet','cgm']}
        summary['timing'][role]['versus_microbiome']={site:pairing(maps[role],{k:v for k,v in maps[site].items() if len(v)==1}) for site in ['oral','gut'] if site!=role}
    summary['complete_source_intersection']=len(set.intersection(*(set(maps[r]) for r in ['diet','cgm','anthropometrics','body_composition','liver_ultrasound','oral','gut'])))
    # Real person/day tables stay within TRE, including food records without CGM.
    with (OUT/'daily_food_CGM_TRE_ONLY.csv').open('w',newline='') as h:
        fields=KEY+['date','cgm_observed','selected_connection_id','food_event_rows']+[c+'_sum' for c in NUT]+[c+'_nonnull' for c in NUT]
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader()
        for k,z in sorted(daily.items()):
            row=dict(zip(KEY,k[:3]));row.update(date=k[3],cgm_observed=int(k[3] in chosen.get(k[:3],set())),selected_connection_id=chosen_id.get(k[:3],''),**z)
            for c in NUT:
                if not z[c+'_nonnull']:row[c+'_sum']=''
            w.writerow(row)
    rows=[]
    for k in sorted(maps['diet']):
        row=dict(zip(KEY,k)); row.update(diet_first=min(maps['diet'][k]),diet_last=max(maps['diet'][k]),diet_days=len(maps['diet'][k]),cgm_first=min(chosen[k]) if k in chosen else '',cgm_last=max(chosen[k]) if k in chosen else '',cgm_days=len(chosen.get(k,set())),common_days=overlaps.get(k,0))
        for role in body_roles:
            ds=maps[role].get(k,set());row[role+'_dates']=';'.join(map(str,sorted(ds)));row[role+'_distinct_dates']=len(ds)
            row[role+'_nearest_diet_day_abs']=gap(ds,maps['diet'][k]) if len(ds)==1 else ''
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'person_visit_all_dates_TRE_ONLY.csv',index=False)
    save('summary.json',summary);save('source_manifest.json',manifest)
    table=['| 数据 | 基线有日期人数 | 同人同访视有饮食 | 唯一测量日可配饮食人数 | 距最近饮食日中位天数 | 饮食前/期内/后 | ≤30天人数 |','|---|---:|---:|---:|---:|---|---:|']
    for role in body_roles:
        r=summary['timing'][role]['diet'];table.append(f"| {role} | {len(maps[role])} | {r['same_visit']} | {r['unique_point_date']} | {r['nearest_observed_day_abs'].get('median','NA')} | {r['before_span']}/{r['inside_span']}/{r['after_span']} | {r['within_days']['30']} |")
    report='# 实际日期与逐日食物记录核查\n\n本报告使用真实文件和完整投影扫描。表中人数均限定10k基线访视。距离是到最近一个实际有饮食记录的日期的绝对天数，前/期内/后是相对首末记录日；期内不保证当天有记录。\n\n'+'\n'.join(table)+'\n\n## 饮食与CGM\n\n'+json.dumps(summary['diet_cgm'],ensure_ascii=False,indent=2)+'\n\n## 饮食逐日表\n\n'+json.dumps(summary['food_daily'],ensure_ascii=False,indent=2)+'\n\n逐日食物与CGM标记表：daily_food_CGM_TRE_ONLY.csv；跨模块实际日期表：person_visit_all_dates_TRE_ONLY.csv。含参与者信息，只留TRE。food_event_rows不是餐次数；营养素合计不是完整摄入量验收。\n\n完整字段、缺失、日期冲突、扫描预算与版本见source_manifest.json；字典摘录见date_and_food_dictionary_excerpts.json；完整时间分布见summary.json。\n'
    (OUT/'REPORT.md').write_text(report)
    memo=(OUT/'MEMO.md').read_text()+'\n## 11_audit.py\n\n完整分批投影扫描饮食事件、饮食日索引、CGM日表、人体测量、DXA体成分、肝超声明细与双部位菌群日期。生成逐日食物汇总、跨模块日期表、聚合报告与来源清单。修改入口ROOT、OUT、KEY、TZ、NUT、scan日期选择和pairing窗口。所有个体表只留TRE；不读原始CGM曲线，不验证进食时刻、漏报或传感器覆盖。body_composition数字型时间戳未猜测单位，采用明确collection_date；肝超声汇总表日期仅通过明细唯一日期恢复。\n\n11运行成功，11_exit_code=0。每次更改分析规则另存新版，禁止直接重跑覆盖本目录。\n'
    (OUT/'MEMO.md').write_text(memo)
    for p in [OUT/'REPORT.md',OUT/'MEMO.md',OUT/'summary.json']:
        b=p.read_bytes();assert b
    (OUT/'11_exit_code').write_text('0\n')
    print('DIET_CGM',json.dumps(summary['diet_cgm']),flush=True)
    print('FOOD_DAILY',json.dumps(summary['food_daily']),flush=True)
    print('DAILY_INDEX',json.dumps(summary['diet_daily_index']),flush=True)
    for role in body_roles:
        r=summary['timing'][role]['diet'];c=summary['timing'][role]['cgm']
        print('TIMING',role,'PEOPLE',len(maps[role]),'DIET_PAIRED',r['same_visit'],'UNIQUE',r['unique_point_date'],'DIET_GAP_MEDIAN',r['nearest_observed_day_abs'].get('median'),'BEFORE_INSIDE_AFTER',r['before_span'],r['inside_span'],r['after_span'],'WITHIN_30',r['within_days']['30'],'CGM_GAP_MEDIAN',c['nearest_observed_day_abs'].get('median'),flush=True)
    print('LIVER_AGGREGATE',json.dumps(summary['liver_aggregate']),flush=True)
    print('ALL_SEVEN_SOURCES',summary['complete_source_intersection'],flush=True)
    print('PHENOTYPE_FOOD_AUDIT_COMPLETE',OUT,'SOURCE_SHA256',summary['source_sha256'],flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        (OUT/'11_exit_code').write_text('1\n');traceback.print_exc();raise
