"""Compute joint timing feasibility from saved TRE-only alignment audit tables."""
from pathlib import Path
import json, hashlib, shutil
import pandas as pd
import pyarrow.parquet as pq

T=Path('/home/ec2-user/hpp_time_alignment_audit_20261018T073819Z')
C=Path('/home/ec2-user/studies/diet_gut_cgm/time_alignment_audit_20261018')
assert (T/'03_exit_code').read_text().strip()=='0'
d=pd.read_csv(T/'four_source_dates_TRE_ONLY.csv')
assert not d.duplicated(['participant_id','cohort','research_stage']).any()
fields=[f'{site}_vs_{target}_nearest_day_abs' for site in ['oral','gut'] for target in ['diet','cgm']]
rows=[]
for t in [7,14,30,60,90]:
    close=d[fields].le(t).all(axis=1)
    row=dict(max_nearest_day_gap=t,both_sites_near_both_streams=int(close.sum()),
             plus_common_days_5=int((close & d.common_days.ge(5)).sum()),
             plus_common_days_7=int((close & d.common_days.ge(7)).sum()),
             plus_common_days_10=int((close & d.common_days.ge(10)).sum()),
             plus_common_days_7_and_inter_site_gap_within_threshold=int((close & d.common_days.ge(7) & d.gut_minus_oral_days.abs().le(t)).sum()))
    rows.append(row)
    print('JOINT_WINDOW',row,flush=True)
prior=(d.oral_vs_cgm_minus_start<=0)&(d.gut_vs_cgm_minus_start<=0)
result=dict(n=len(d),definition='Both sites within threshold of at least one observed diet day AND at least one observed CGM day; absolute calendar-day distance. These feasibility cutoffs are not validated biological equivalence windows.',
            rows=rows,both_sites_on_or_before_first_cgm_day=int(prior.sum()),
            both_sites_on_or_before_first_cgm_day_and_common_days_7=int((prior&d.common_days.ge(7)).sum()))
print('BOTH_SITES_ON_OR_BEFORE_CGM_FIRST_DAY',result['both_sites_on_or_before_first_cgm_day'],
      'PLUS_COMMON_DAYS_7',result['both_sites_on_or_before_first_cgm_day_and_common_days_7'],flush=True)
(T/'joint_window_feasibility.json').write_text(json.dumps(result,indent=2))
root=Path('/home/ec2-user/studies/hpp_datasets')
locations=[]
for module in ['cgm','oral_microbiome','gut_microbiome']:
    p=root/module/(module+'.parquet')
    f=pq.ParquetFile(p,pre_buffer=False)
    # Some file pointers are dictionary-encoded strings, not plain strings.
    fields=[x.name for x in f.schema_arrow if 'string' in str(x.type) and any(t in x.name.lower() for t in ['parquet','arrow','time_series','timeseries'])]
    target=[x for x in fields if 'species' in x or 'time_series' in x or 'timeseries' in x]
    if not target: target=fields[:1]
    if target:
        b=next(f.iter_batches(batch_size=32,columns=target,use_threads=False))
        for field in target:
            values=[x for x in b.column(b.schema.get_field_index(field)).to_pylist() if x]
            if not values: continue
            raw=values[0]
            resolved=(p.parent/raw).resolve() if not str(raw).startswith(('s3:','http:','https:')) else None
            rec=dict(module=module,field=field,reference=raw,resolved=str(resolved) if resolved else None,
                     exists=resolved.exists() if resolved else None)
            locations.append(rec)
            print('RESOLVED_SOURCE',rec,flush=True)
rawexamples=[]
for p in (root/'cgm'/'timeseries').glob('*.parquet'):
    if len(rawexamples)>=3: break
    f=pq.ParquetFile(p,pre_buffer=False)
    rec=dict(path=str(p),rows=f.metadata.num_rows,schema=[(x.name,str(x.type)) for x in f.schema_arrow])
    rawexamples.append(rec)
    print('RAW_CGM_REPRESENTATIVE_METADATA',dict(rows=rec['rows'],schema=rec['schema']),flush=True)
(T/'resolved_source_locations.json').write_text(json.dumps(dict(locations=locations,raw_cgm_representative_metadata=rawexamples),indent=2))
with (T/'MEMO.md').open('a') as h:
    h.write('\n## 05_joint_windows.py：双部位共同时间窗\n\n读取已保存的TRE内日期审计表，不重新拟合模型。统计口腔和肠道同时邻近饮食与CGM观测日的交集；阈值为可行性描述，不是已验证的生物学可交换时间窗。on_or_before_first_cgm_day只表示日历日期，不证明日内先后。修改阈值见脚本for t；共同天数条件见common_days。\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n')
with (T/'MEMO.md').open('a') as h:
    h.write('\n05版另支持字典编码的文件指针，定位CGM、口腔与肠道物种矩阵路径，保存resolved_source_locations.json；只检查3份CGM原始分片的schema和行数，不读取其血糖正文，不代表全量原始时序验收。\n')
cloud_error=None
try:
    for name in ['05_joint_windows.py','joint_window_feasibility.json','resolved_source_locations.json','MEMO.md']:
        shutil.copy2(T/name,C/name)
except OSError as e: cloud_error=repr(e)
(T/'05_exit_code').write_text('0\n')
print('JOINT_FEASIBILITY_EXIT_CODE_0','SOURCE_SHA256',hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'CLOUD_COPY_ERROR',cloud_error,flush=True)
