"""Audit HPP dates using projected streaming reads; keep individual data in TRE.

This checks recorded dates and observation-day overlap, NOT dietary completeness,
sensor quality, consumption-time truth, or causal/clinical eligibility.
"""
from pathlib import Path
from collections import defaultdict, Counter
import os, sys, json, datetime, hashlib, traceback, shutil, resource, time
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

ROOT = Path('/home/ec2-user/studies/hpp_datasets')
TEMP = Path('/home/ec2-user/hpp_time_alignment_audit_20261018T073819Z')
CLOUD = Path('/home/ec2-user/studies/diet_gut_cgm/time_alignment_audit_20261018')
KEY = ['participant_id', 'cohort', 'research_stage']
TZ = 'Asia/Jerusalem'
MANIFEST, DATES, LOG = [], {}, []

def emit(*a):
    s = ' '.join(map(str, a)); LOG.append(s); print(s, flush=True)

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str))

def stats(values):
    x = np.asarray(list(values), dtype=float)
    return dict(n=len(x), min=float(x.min()), p10=float(np.quantile(x,.1)),
                median=float(np.median(x)), p90=float(np.quantile(x,.9)), max=float(x.max())) if len(x) else dict(n=0)

def dates(s):
    d = pd.to_datetime(s, errors='coerce')
    if isinstance(d.dtype, pd.DatetimeTZDtype):
        d = d.dt.tz_convert(TZ).dt.tz_localize(None)
    return d.dt.normalize()

def clean_keys(d):
    valid = d[KEY].notna().all(axis=1)
    for c in ['cohort', 'research_stage']:
        valid &= ~d[c].astype(str).str.strip().str.lower().isin(['', 'undefined', 'none', 'null', 'nan'])
    return valid

def scan(module, filename, role):
    p = ROOT / module / filename
    f = pq.ParquetFile(p, pre_buffer=False)
    schema = {x.name: str(x.type) for x in f.schema_arrow}
    if not set(KEY).issubset(schema):
        raise ValueError(f'Missing physical keys: {p}: {schema}')
    cols = KEY + [c for c in schema if c not in KEY and
                 (c in ['collection_timestamp','collection_date','local_timestamp','timezone','connection_id','logging_day','connection_day']
                  or (role=='cgm_summary' and any(t in c.lower() for t in ['date','time','first','last','day','connection'])))]
    if role in ['diet','gut'] and 'collection_timestamp' in schema and 'tz=' in schema['collection_timestamp']:
        primary = 'collection_timestamp'
    elif 'collection_date' in schema:
        primary = 'collection_date'
    elif 'collection_timestamp' in schema:
        raise ValueError('Timestamp has no explicit timezone and no collection_date; manual review required')
    else:
        raise ValueError(f'No accepted date field for {p}')
    rec = dict(role=role, path=str(p), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns,
               rows_metadata=f.metadata.num_rows, schema=schema, selected_columns=cols, primary_date=primary,
               row_groups=f.metadata.num_row_groups, projected_uncompressed_bytes=sum(
                   f.metadata.row_group(i).column(j).total_uncompressed_size
                   for i in range(f.metadata.num_row_groups) for j,n in enumerate(f.schema.names) if n in cols))
    emit('SCAN_START', role, 'rows', rec['rows_metadata'], 'date', primary,
         'projected_MB', round(rec['projected_uncompressed_bytes']/1e6,2))
    if rec['projected_uncompressed_bytes'] > 20*1024**3:
        raise RuntimeError('Projected scan exceeds 20 GiB budget; requires a revised plan')
    started=time.monotonic()
    seen, badkey, baddate, disagree, compared, local_disagree, local_compared = 0,0,0,0,0,0,0
    groups, counts, cgconn = defaultdict(set), Counter(), defaultdict(set)
    strata = Counter()
    for b in f.iter_batches(batch_size=16384, columns=cols, use_threads=False):
        available=int(next(x.split()[1] for x in open('/proc/meminfo') if x.startswith('MemAvailable:')))*1024
        if available < 2*1024**3 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 1024*1024:
            raise RuntimeError('Memory safety stop: available<2 GiB or peak RSS>1 GiB')
        if time.monotonic()-started > 1200:
            raise RuntimeError('Per-source scan time budget exceeded; partial scan not reported as complete')
        d = b.to_pandas(ignore_metadata=True)
        seen += len(d)
        good = clean_keys(d); badkey += int((~good).sum())
        for row in d.loc[good,['cohort','research_stage']].itertuples(index=False,name=None):
            strata[tuple(map(str,row))] += 1
        day = dates(d[primary]); baddate += int(day.isna().sum())
        if 'collection_timestamp' in d and 'collection_date' in d:
            a,z = dates(d.collection_timestamp),dates(d.collection_date)
            ok = a.notna() & z.notna(); compared += int(ok.sum()); disagree += int((a[ok]!=z[ok]).sum())
        if 'local_timestamp' in d:
            z = dates(d.local_timestamp); ok=day.notna() & z.notna()
            local_compared += int(ok.sum()); local_disagree += int((day[ok]!=z[ok]).sum())
        q = d.loc[good & day.notna(),KEY].copy()
        q['day'] = day.loc[q.index]
        # Only observation dates are retained, never abundance/nutrition vectors.
        v = q.groupby(KEY, observed=True, sort=False)['day'].agg(['min','max','count'])
        for k,row in v.iterrows():
            counts[(int(k[0]),str(k[1]),str(k[2]))] += int(row['count'])
        u = q.drop_duplicates(KEY+['day'])
        for row in u.itertuples(index=False,name=None):
            k = (int(row[0]),str(row[1]),str(row[2])); groups[k].add(pd.Timestamp(row[3]).date())
        if role=='cgm_daily':
            if 'connection_id' not in d:
                raise ValueError('CGM daily lacks connection_id; do not silently mix connections')
            qq=q.copy(); qq['connection_id']=d.loc[qq.index,'connection_id']
            rec['missing_connection_rows']=rec.get('missing_connection_rows',0)+int(qq.connection_id.isna().sum())
            for row in qq.dropna(subset=['connection_id']).drop_duplicates(KEY+['connection_id','day']).itertuples(index=False,name=None):
                k=(int(row[0]),str(row[1]),str(row[2]),str(row[4])); cgconn[k].add(pd.Timestamp(row[3]).date())
        if seen % (16384*50)==0:
            emit('PROGRESS',role,seen,'/',rec['rows_metadata'])
    assert seen==rec['rows_metadata'], (role,seen,rec['rows_metadata'])
    rec.update(rows_scanned=seen, invalid_key_rows=badkey, missing_date_rows=baddate,
               timestamp_date_compared_rows=compared, timestamp_date_disagreement_rows=disagree,
               local_date_compared_rows=local_compared, local_date_disagreement_rows=local_disagree,
               person_visit_keys=len(groups), people=len({k[0] for k in groups}),
               multi_date_keys=sum(len(v)>1 for v in groups.values()),
               strata=[dict(cohort=k[0],stage=k[1],rows=v) for k,v in sorted(strata.items())])
    MANIFEST.append(rec); DATES[role]=groups
    emit('SCAN_DONE',role,'people',rec['people'],'person_visits',len(groups),'bad_key',badkey,
         'missing_dates',baddate,'timestamp_date_conflicts',disagree)
    return groups,counts,cgconn

def baseline(groups):
    return {k:v for k,v in groups.items() if k[1:] == ('10k','00_00_visit')}

def orientation(sample, days):
    a,b=min(days),max(days)
    return 'before' if sample<a else ('after' if sample>b else 'inside_span')

def gap_to_set(sample, ds):
    return min(abs((sample-d).days) for d in ds)

def main():
    if '/envs/pheno/' not in sys.executable:
        raise RuntimeError('Expected pheno interpreter')
    assert dates(pd.Series(pd.to_datetime(['2025-01-01T22:30:00Z'],utc=True))).iloc[0]==pd.Timestamp('2025-01-02')
    assert dates(pd.Series(pd.to_datetime(['2025-01-01']))).iloc[0]==pd.Timestamp('2025-01-01')
    testkeys=pd.DataFrame(dict(participant_id=[1,2,3],cohort=['10k','10k','10k'],research_stage=['00_00_visit','undefined',None]))
    assert clean_keys(testkeys).tolist()==[True,False,False]
    emit('SYNTHETIC_DATE_AND_KEY_CHECKS_PASS')
    TEMP.mkdir(exist_ok=True)
    out=TEMP; cloud_error=None
    try:
        CLOUD.mkdir(exist_ok=True)
        # A real audit document, not merely os.access(), verifies file writing.
        with (CLOUD/'AUDIT_SETUP.md').open('x') as h:
            h.write('# Time alignment audit\nCreated by 02_time_alignment.py; data stay in TRE.\n')
        out=CLOUD
    except FileExistsError:
        with (CLOUD/'AUDIT_SETUP.md').open('a') as h: h.write('Additional audit run.\n')
        out=CLOUD
    except OSError as e:
        cloud_error=repr(e); emit('CLOUD_WRITE_BLOCKED',cloud_error)
    source=Path(__file__).read_bytes()
    (out/'02_time_alignment.py').write_bytes(source)
    if (TEMP/'01_discover.py').exists() and out!=TEMP:
        shutil.copy2(TEMP/'01_discover.py',out/'01_discover.py')
        shutil.copy2(TEMP/'discovery.json',out/'discovery.json')
    emit('OUTPUT_DIR',out,'PYTHON',sys.executable)
    emit('SOURCE_SHA256',hashlib.sha256(source).hexdigest())
    diet,_,_=scan('diet_logging','diet_logging_events.parquet','diet')
    cgm,_,conn=scan('cgm','iglu_daily.parquet','cgm_daily')
    cgsummary,_,_=scan('cgm','cgm.parquet','cgm_summary')
    oral,_,_=scan('oral_microbiome','oral_microbiome.parquet','oral')
    gut,_,_=scan('gut_microbiome','gut_microbiome.parquet','gut')
    alt,_,_=scan('oral_microbiome','oral_microbiome_2025_04_UUID.parquet','oral_alternative')
    oral_equal=oral==alt
    emit('ORAL_VERSION_DATE_KEYS_EQUAL',oral_equal)
    diet,cgm,oral,gut=map(baseline,[diet,cgm,oral,gut])
    sets=dict(diet=set(diet),cgm=set(cgm),oral=set(oral),gut=set(gut))
    same=set.intersection(*sets.values())
    dc=sets['diet'] & sets['cgm']
    connection_lists=defaultdict(list)
    for k,ds in conn.items():
        if k[:3] in dc: connection_lists[k[:3]].append((k[3],ds))
    paired=[]
    for k in sorted(dc):
        choices=connection_lists[k]
        # Deterministic diet/date-only choice; no glucose values used.
        choices=sorted(choices,key=lambda x:(-len(x[1]&diet[k]),min(x[1]),x[0]))
        if not choices: continue
        chosen,ds=choices[0]
        overlap=diet[k]&ds
        assert len(overlap)<=min(len(diet[k]),len(ds))
        paired.append(dict(participant_id=k[0],cohort=k[1],research_stage=k[2],
                           diet_days=len(diet[k]),cgm_union_days=len(cgm[k]),
                           cgm_selected_connection=chosen,cgm_connection_count=len(choices),
                           selected_cgm_days=len(ds),overlap_days=len(overlap),
                           union_overlap_days=len(diet[k]&cgm[k]),
                           diet_start=min(diet[k]),diet_end=max(diet[k]),
                           cgm_start=min(ds),cgm_end=max(ds),
                           cgm_start_minus_diet_start=(min(ds)-min(diet[k])).days,
                           span_overlap=max(min(diet[k]),min(ds))<=min(max(diet[k]),max(ds))))
    panel=pd.DataFrame(paired)
    assert not panel.duplicated(KEY).any()
    panel.to_csv(out/'person_visit_alignment_TRE_ONLY.csv',index=False)
    pindex={tuple(x[c] for c in KEY):x for x in paired}
    n=lambda n:int((panel.overlap_days>=n).sum())
    summary=dict(scope="10k / 00_00_visit; dates with observations, not quality-filtered analysis days",
                 counts={k:len(v) for k,v in sets.items()}, same_visit_four_sources=len(same),
                 diet_cgm_same_visit=len(dc), pairs_with_connection=len(panel),
                 actual_common_days_at_least={str(t):n(t) for t in [1,5,7,10,14]},
                 diet_days=stats(panel.diet_days),selected_cgm_days=stats(panel.selected_cgm_days),
                 common_days=stats(panel.overlap_days),
                 cgm_start_minus_diet_start=stats(panel.cgm_start_minus_diet_start),
                 multiple_connections=int((panel.cgm_connection_count>1).sum()),
                 span_overlap_but_no_common_days=int((panel.span_overlap & panel.overlap_days.eq(0)).sum()),
                 oral_versions_date_keys_equal=oral_equal)
    emit('BASELINE_COUNTS',summary['counts'])
    emit('DIET_CGM_SAME_VISIT',len(dc),'FOUR_SOURCE_SAME_VISIT',len(same))
    emit('ACTUAL_COMMON_DAYS_AT_LEAST',summary['actual_common_days_at_least'])
    emit('DIET_DAYS',summary['diet_days'],'CGM_DAYS',summary['selected_cgm_days'])
    emit('COMMON_DAYS',summary['common_days'])
    emit('CGM_START_MINUS_DIET_START',summary['cgm_start_minus_diet_start'])
    emit('MULTIPLE_CONNECTIONS',summary['multiple_connections'])
    sample_rows=[]; gaps=defaultdict(list); positions=defaultdict(Counter)
    four=Counter()
    for k in sorted(same):
        if k not in pindex: continue
        p=pindex[k]
        four['with_connection']+=1
        if len(oral[k])!=1 or len(gut[k])!=1:
            four['ambiguous_microbiome_dates']+=1; continue
        o,g=next(iter(oral[k])),next(iter(gut[k])); four['unique_microbiome_dates']+=1
        for t in [1,5,7,10,14]:
            four['common_days_at_least_'+str(t)]+=int(p['overlap_days']>=t)
        ds=diet[k]
        cs=dict(connection_lists[k])[p['cgm_selected_connection']]
        row=dict(participant_id=k[0],cohort=k[1],research_stage=k[2],oral_date=o,gut_date=g,
                 common_days=p['overlap_days'],gut_minus_oral_days=(g-o).days)
        gaps['gut_minus_oral_days'].append((g-o).days)
        for name,s in [('oral',o),('gut',g)]:
            for window,days in [('diet',ds),('cgm',cs)]:
                label=name+'_vs_'+window
                positions[label][orientation(s,days)]+=1
                exact=s in days; positions[label]['on_observed_day']+=int(exact)
                gap=gap_to_set(s,days); gaps[label+'_nearest_observed_day_abs'].append(gap)
                for threshold in [0,7,14,30,90]:
                    positions[label]['nearest_day_within_'+str(threshold)]+=int(gap<=threshold)
                row[label+'_nearest_day_abs']=gap
                row[label+'_minus_start']=(s-min(days)).days
                gaps[label+'_minus_start'].append((s-min(days)).days)
                row[label+'_position']=orientation(s,days)
        sample_rows.append(row)
    pd.DataFrame(sample_rows).to_csv(out/'four_source_dates_TRE_ONLY.csv',index=False)
    assert four['unique_microbiome_dates']+four['ambiguous_microbiome_dates']==four['with_connection']
    assert four['common_days_at_least_1']<=n(1)
    for value in positions.values():
        assert sum(value[x] for x in ['before','inside_span','after'])==four['unique_microbiome_dates']
    summary.update(four_source=dict(four),sample_positions={k:dict(v) for k,v in positions.items()},
                   sample_gaps={k:stats(v) for k,v in gaps.items()})
    emit('FOUR_SOURCE',dict(four))
    for label,v in positions.items():
        emit('POSITION',label,dict(v))
        emit('GAP',label,summary['sample_gaps'][label+'_nearest_observed_day_abs'])
    emit('ORAL_GUT_SIGNED_GAP',summary['sample_gaps'].get('gut_minus_oral_days'))
    dictionaries=[]
    for module in ['diet_logging','cgm','oral_microbiome','gut_microbiome']:
        for p in (ROOT/module/'metadata').glob('*data_dictionary.csv'):
            text=p.read_text()
            relevant=[x for x in text.splitlines() if any(t in x.lower() for t in ['collection_date','collection_timestamp','local_timestamp','connection_id'])]
            dictionaries.append(dict(path=str(p),lines=relevant))
    dump(out/'date_field_dictionary_excerpts.json',dictionaries)
    dump(out/'source_manifest.json',dict(sources=MANIFEST,python=sys.executable,
         source_sha256=hashlib.sha256(source).hexdigest(),timezone=TZ,cloud_error=cloud_error,
         intended_output=str(CLOUD),actual_output=str(out),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    dump(out/'aggregate_summary.json',summary)
    memo='''# 时间对齐核查备忘录\n\n本程序只读HPP原始数据。\n\n## 代码\n\n- 01_discover.py：四个模块的有界文件盘点和Parquet元数据核查；不读取丰度正文。\n- 02_time_alignment.py：完整分批投影扫描饮食事件、CGM日表、CGM连接汇总、口腔两版本和肠道样本表的关联键与日期。按10k基线访视核查真实观测日期交集、连接数、双部位采样先后及间隔。\n\n## 修改入口\n\n脚本顶部ROOT/TEMP/CLOUD为输入与输出目录；TZ为本地时区；baseline()定义队列和访视；scan()定义投影列和日期选择；main()定义按观测重叠选择CGM连接及汇总阈值。改变这些规则应另建运行目录并保留原版本。\n\n## 输出\n\n- discovery.json：有界目录和schema记录。\n- source_manifest.json：真实输入、完整扫描行数、字段、缺失、日期冲突、源码SHA256、环境与存储状态。\n- aggregate_summary.json：聚合人数、重叠天数和采样时间间隔。\n- person_visit_alignment_TRE_ONLY.csv：参与者访视层时间审计，仅留在TRE。\n- four_source_dates_TRE_ONLY.csv：四类数据配对日期，仅留在TRE。\n- date_field_dictionary_excerpts.json：日期字段原始字典摘录。\n- audit.log、02_exit_code：实际运行日志和完成状态。\n\n## 解读边界\n\n这是日期可对齐性核查，不是进食时间真实性、漏餐程度或CGM有效覆盖率验收。CGM采用iglu_daily中有观测的实际日期，未全量扫描15分钟原始读数。多个CGM连接按与饮食实际共同日期最多、再最早起始日期、再连接ID的预定顺序选择；不使用血糖结果。菌群同一访视有多个不同采样日期时不任意去重，主采样间隔汇总排除这些歧义键并报告数量。口腔两版本的键日期相等也不代表丰度或样本身份完全相同。采样日期相近不能证明口腔到肠道传播，或菌群是暴露前基线。\n\n'''
    memo += '实际输出目录：'+str(out)+'\n\n云盘写入错误：'+str(cloud_error)+'\n\n'
    memo += '源码SHA256：'+hashlib.sha256(source).hexdigest()+'\n\n聚合结果：\n\n```json\n'+json.dumps(summary,indent=2,ensure_ascii=False,default=str)+'\n```\n'
    (out/'MEMO.md').write_text(memo)
    (out/'02_exit_code').write_text('0\n')
    emit('AUDIT_COMPLETE_EXIT_CODE_0','MAX_RSS_KIB',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (out/'audit.log').write_text('\n'.join(LOG)+'\n')

if __name__=='__main__':
    try:
        main()
    except Exception:
        TEMP.mkdir(exist_ok=True)
        (TEMP/'02_failure.log').write_text(traceback.format_exc())
        (TEMP/'02_exit_code').write_text('1\n')
        raise
